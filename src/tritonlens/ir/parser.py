"""A line-oriented parser for Triton GPU IR (TTGIR) as printed by ``CompiledKernel.asm``.

This intentionally does not depend on the MLIR Python bindings: it only needs to understand
the textual subset that Triton prints, and it must work on machines without a GPU.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .attrs import Encoding, parse_attr_dict, parse_encoding
from .types import MemDescType, PointerType, ScalarType, TensorType, Type, parse_type, split_top_level

_ALIAS_RE = re.compile(r"^#(\w+)\s*=\s*(.*)$")
_LOC_SUFFIX_RE = re.compile(r"\s+loc\((#\w+|\".*?\"(?::\d+:\d+)?|unknown)\)\s*$")
_RESULT_RE = re.compile(r"^\s*((?:%[\w$.#-]+(?::\d+)?)(?:\s*,\s*%[\w$.#-]+(?::\d+)?)*)\s*=\s*")
_OPNAME_RE = re.compile(r'^\s*("?)([a-z_][\w]*(?:\.[\w]+)+)\1')
_VALUE_RE = re.compile(r"%[\w$.-]+(?:#\d+)?")
_FILELOC_RE = re.compile(r'"((?:[^"\\]|\\.)*)":(\d+):(\d+)')


@dataclass
class Loc:
    file: str
    line: int
    col: int

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.col}"

    def to_json(self) -> dict[str, Any]:
        return {"file": self.file, "line": self.line, "col": self.col}


@dataclass(eq=False)
class Value:
    name: str
    type: Type | None
    kind: str  # "arg", "result", "block_arg"
    owner: Op | None = None
    index: int = 0
    attrs: dict[str, Any] = field(default_factory=dict)
    init: Value | None = None  # loop-carried block args: the value entering the loop
    yielded: Value | None = None  # loop-carried block args: the value from the back edge

    def __repr__(self) -> str:
        return f"Value({self.name})"

    @property
    def uid(self) -> str:
        if self.owner is not None:
            return f"{self.name}@{self.owner.id}"
        return self.name


@dataclass(eq=False)
class Op:
    id: int
    name: str
    line_no: int
    text: str
    results: list[Value] = field(default_factory=list)
    operands: list[Value] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    operand_types: list[Type] = field(default_factory=list)
    loc: Loc | None = None
    parent: Op | None = None
    depth: int = 0
    block_args: list[Value] = field(default_factory=list)
    words: list[str] = field(default_factory=list)  # bare keywords, e.g. "slt" or "x"

    def __repr__(self) -> str:
        return f"Op({self.id}: {self.name} @ line {self.line_no})"

    @property
    def short_name(self) -> str:
        return self.name.split(".", 1)[-1]

    @property
    def dialect(self) -> str:
        return self.name.split(".", 1)[0]

    def is_(self, *names: str) -> bool:
        """Match by op name ignoring the dialect prefix, e.g. ``op.is_("local_load")``."""
        return self.short_name in names or self.name in names

    @property
    def result(self) -> Value | None:
        return self.results[0] if self.results else None

    @property
    def result_type(self) -> Type | None:
        return self.results[0].type if self.results else None


@dataclass
class Func:
    name: str
    args: list[Value]
    ops: list[Op]
    line_no: int = 0
    loc: Loc | None = None

    def find(self, *names: str) -> list[Op]:
        return [op for op in self.ops if op.is_(*names)]


@dataclass
class Module:
    attrs: dict[str, Any]
    aliases: dict[str, Encoding]
    locs: dict[str, Loc | None]
    funcs: list[Func]
    text: str

    @property
    def num_warps(self) -> int:
        return int(self._attr("num-warps", 4))

    @property
    def threads_per_warp(self) -> int:
        return int(self._attr("threads-per-warp", 32))

    @property
    def num_ctas(self) -> int:
        return int(self._attr("num-ctas", 1))

    @property
    def target(self) -> str:
        return str(self._attr("target", ""))

    def _attr(self, suffix: str, default: Any) -> Any:
        for key, value in self.attrs.items():
            if key.split(".")[-1] == suffix:
                return value
        return default

    @property
    def func(self) -> Func:
        return self.funcs[0]

    @property
    def layouts(self) -> dict[str, Encoding]:
        return {k: v for k, v in self.aliases.items() if isinstance(v, Encoding)}


def _strip_loc(line: str) -> tuple[str, str | None]:
    m = _LOC_SUFFIX_RE.search(line)
    if not m:
        return line.rstrip(), None
    return line[: m.start()].rstrip(), m.group(1)


def _find_top_level(text: str, token: str) -> int:
    depth = 0
    in_str = False
    for i, ch in enumerate(text):
        if in_str:
            if ch == '"':
                in_str = False
            continue
        if depth == 0 and text.startswith(token, i):
            return i
        if ch == '"':
            in_str = True
        elif ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            if not (ch == ">" and i > 0 and text[i - 1] == "-"):
                depth -= 1
    return -1


def _resolve_locs(raw: dict[str, str]) -> dict[str, Loc | None]:
    """Resolve ``#locN`` aliases, following callsite/named/fused locations.

    For ``callsite(callee at caller)`` we prefer the frame in user code: Triton's own
    ``triton/language`` helpers (``tl.cdiv`` etc.) are skipped in favour of the caller.
    """
    resolved: dict[str, Loc | None] = {}

    def resolve_text(text: str, seen: frozenset) -> Loc | None:
        text = text.strip()
        if text.startswith("loc(") and text.endswith(")"):
            text = text[4:-1].strip()
        if text.startswith("callsite("):
            inner = text[len("callsite("):-1]
            callee, _, caller = inner.partition(" at ")
            callee_loc = resolve_text(callee, seen)
            caller_loc = resolve_text(caller, seen)
            if callee_loc and not _is_library_file(callee_loc.file):
                return callee_loc
            return caller_loc or callee_loc
        if text.startswith("fused"):
            for ref in re.findall(r"#\w+", text):
                loc = resolve_alias(ref[1:], seen)
                if loc:
                    return loc
        m = _FILELOC_RE.match(text)
        if m:
            return Loc(m.group(1).replace("\\\\", "\\"), int(m.group(2)), int(m.group(3)))
        named = re.match(r'"[^"]*"\((.*)\)$', text)
        if named:
            return resolve_text(named.group(1), seen)
        if text.startswith("#"):
            return resolve_alias(text[1:], seen)
        m = _FILELOC_RE.search(text)
        if m:
            return Loc(m.group(1).replace("\\\\", "\\"), int(m.group(2)), int(m.group(3)))
        return None

    def resolve_alias(name: str, seen: frozenset) -> Loc | None:
        if name in resolved:
            return resolved[name]
        if name in seen or name not in raw:
            return None
        loc = resolve_text(raw[name], seen | {name})
        resolved[name] = loc
        return loc

    for name in raw:
        resolve_alias(name, frozenset())
    return resolved


def _is_library_file(path: str) -> bool:
    p = path.replace("\\", "/")
    return "triton/language" in p or "triton/runtime" in p


class _Scope:
    def __init__(self, parent: _Scope | None = None):
        self.parent = parent
        self.values: dict[str, Value] = {}

    def lookup(self, name: str) -> Value | None:
        scope: _Scope | None = self
        base, _, idx = name.partition("#")
        while scope is not None:
            if name in scope.values:
                return scope.values[name]
            if idx and base in scope.values:
                return scope.values[base]
            scope = scope.parent
        return None


def parse_module(text: str) -> Module:
    lines = text.splitlines()
    raw_aliases: dict[str, str] = {}
    raw_locs: dict[str, str] = {}
    for line in lines:
        m = _ALIAS_RE.match(line.strip())
        if not m:
            continue
        name, body = m.group(1), m.group(2).strip()
        if body.startswith("loc("):
            raw_locs[name] = body
        else:
            raw_aliases[name] = body

    aliases: dict[str, Encoding] = {}
    for name, body in raw_aliases.items():
        try:
            enc = parse_encoding(body, aliases)
        except ValueError:
            continue
        enc.alias = name
        aliases[name] = enc
    # Second pass so that aliases referring to later aliases (e.g. parent = #mma) resolve.
    for name, body in raw_aliases.items():
        try:
            enc = parse_encoding(body, aliases)
        except ValueError:
            continue
        enc.alias = name
        aliases[name].params = enc.params
    locs = _resolve_locs(raw_locs)

    module_attrs: dict[str, Any] = {}
    funcs: list[Func] = []
    builder: _FuncBuilder | None = None
    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("//"):
            continue
        if stripped.startswith("module"):
            m = re.search(r"attributes\s*(\{.*\})", stripped)
            if m:
                module_attrs = parse_attr_dict(m.group(1), aliases)
            continue
        if re.match(r"tt\.func\b", stripped):
            builder = _FuncBuilder(stripped, line_no, aliases, locs)
            continue
        if builder is None:
            continue
        if stripped.startswith("tt.return"):
            builder.add_op(line, line_no)
            continue
        if builder.depth == 0 and stripped.startswith("}"):
            funcs.append(builder.build())
            builder = None
            continue
        builder.feed(line, line_no)
    if builder is not None:
        funcs.append(builder.build())
    return Module(attrs=module_attrs, aliases=aliases, locs=locs, funcs=funcs, text=text)


class _FuncBuilder:
    def __init__(self, header: str, line_no: int, aliases: dict[str, Encoding],
                 locs: dict[str, Loc | None]):
        self.aliases = aliases
        self.locs = locs
        self.ops: list[Op] = []
        self.scope = _Scope()
        self.region_stack: list[Op] = []
        self.pending_generic: list[Op] = []
        self.line_no = line_no
        m = re.search(r"@([\w$.]+)\s*\(", header)
        self.name = m.group(1) if m else "kernel"
        self.args = self._parse_args(header)
        for arg in self.args:
            self.scope.values[arg.name] = arg

    @property
    def depth(self) -> int:
        return len(self.region_stack)

    def _loc(self, ref: str | None) -> Loc | None:
        if ref is None:
            return None
        if ref.startswith("#"):
            return self.locs.get(ref[1:])
        m = _FILELOC_RE.match(ref)
        if m:
            return Loc(m.group(1).replace("\\\\", "\\"), int(m.group(2)), int(m.group(3)))
        return None

    def _parse_args(self, header: str) -> list[Value]:
        start = header.find("(", header.find("@"))
        depth, end = 0, start
        for i in range(start, len(header)):
            if header[i] == "(":
                depth += 1
            elif header[i] == ")":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        args = []
        for idx, part in enumerate(split_top_level(header[start + 1:end])):
            part = re.sub(r"\s+loc\(.*\)$", "", part).strip()
            m = re.match(r"(%[\w$.-]+)\s*:\s*(.*)$", part)
            if not m:
                continue
            name, rest = m.group(1), m.group(2)
            attrs: dict[str, Any] = {}
            brace = _find_top_level(rest, "{")
            if brace >= 0:
                attrs = parse_attr_dict(rest[brace:], self.aliases)
                rest = rest[:brace]
            args.append(Value(name=name, type=parse_type(rest, self.aliases), kind="arg",
                              index=idx, attrs=attrs))
        return args

    def feed(self, line: str, line_no: int) -> None:
        stripped = line.strip()
        if stripped.startswith("^bb"):
            # Block header inside a region, e.g. ^bb0(%arg5: f32 loc(...), %arg6: f32 ...):
            inner = stripped[stripped.find("(") + 1: stripped.rfind(")")]
            owner = self.region_stack[-1] if self.region_stack else None
            for idx, part in enumerate(split_top_level(inner)):
                part = re.sub(r"\s+loc\(.*\)$", "", part)
                m = re.match(r"(%[\w$.-]+)\s*:\s*(.*)$", part.strip())
                if m:
                    val = Value(m.group(1), parse_type(m.group(2), self.aliases), "block_arg",
                                owner=owner, index=idx)
                    self.scope.values[val.name] = val
                    if owner is not None:
                        owner.block_args.append(val)
            return
        if stripped.startswith("}"):
            body, loc_ref = _strip_loc(stripped)
            closing_owner = self.region_stack.pop() if self.region_stack else None
            self.scope = self.scope.parent or self.scope
            if closing_owner is not None and closing_owner.loc is None:
                closing_owner.loc = self._loc(loc_ref)
            if closing_owner is not None and body.startswith("})") and ":" in body:
                self._finish_generic(closing_owner, body[2:].strip())
            if body.rstrip().endswith("{"):
                # "} else {" -- reopen a region on the same op.
                if closing_owner is not None:
                    self.region_stack.append(closing_owner)
                self.scope = _Scope(self.scope)
            return
        self.add_op(line, line_no)

    def add_op(self, line: str, line_no: int) -> Op | None:
        body, loc_ref = _strip_loc(line.strip())
        opens_region = body.endswith("{")
        if opens_region:
            body = body[:-1].rstrip()
            if body.endswith("("):
                body = body[:-1].rstrip()
        results: list[str] = []
        m = _RESULT_RE.match(body)
        if m:
            for res in split_top_level(m.group(1)):
                results.append(res)
            body = body[m.end():]
        m = _OPNAME_RE.match(body)
        if not m:
            return None
        name = m.group(2)
        rest = " " + body[m.end():].strip()
        op = Op(id=len(self.ops), name=name, line_no=line_no, text=line.strip(),
                loc=self._loc(loc_ref), parent=self.region_stack[-1] if self.region_stack else None,
                depth=self.depth)
        generic = bool(m.group(1))
        self._parse_body(op, rest, results, generic)
        self.ops.append(op)
        if opens_region:
            self.region_stack.append(op)
            self.scope = _Scope(self.scope)
            self._bind_region_args(op, rest)
        return op

    def _parse_body(self, op: Op, rest: str, results: list[str], generic: bool) -> None:
        # Split "operands {attrs} : types" at the first top-level colon.
        colon = _find_top_level(rest, " : ")
        operand_part = rest if colon < 0 else rest[:colon]
        type_part = "" if colon < 0 else rest[colon + 3:].strip()

        if op.is_("for"):
            operand_part = operand_part.split("iter_args", 1)[0]
        attr_text = ""
        brace = _find_top_level(operand_part, "{")
        if brace >= 0:
            attr_text = operand_part[brace:]
            operand_part = operand_part[:brace]
        props = re.search(r"<\{(.*)\}>", operand_part)
        if props:
            op.attrs.update(parse_attr_dict(props.group(1), self.aliases))
            operand_part = operand_part[: props.start()] + operand_part[props.end():]
        if attr_text:
            try:
                op.attrs.update(parse_attr_dict(attr_text, self.aliases))
            except ValueError:
                pass
        # Inline "key = value" attributes (e.g. tt.dot's inputPrecision = tf32).
        for key, value in re.findall(r"(?<![%\w])(\w+)\s*=\s*([\w.]+)", operand_part):
            op.attrs.setdefault(key, _parse_number(value))
        operand_names = _VALUE_RE.findall(operand_part)
        for name in operand_names:
            val = self.scope.lookup(name)
            if val is not None:
                op.operands.append(val)
        op.words = re.findall(r"(?<![%#\w.])([a-z][a-z_]*)(?![\w.(])", re.sub(r"\w+\s*=\s*[\w.]+", "", operand_part))
        if op.is_("constant"):
            m = re.match(r"\s*(dense<(.*?)>|[-\w.+]+)", operand_part)
            if m:
                raw = m.group(2) if m.group(2) is not None else m.group(1)
                op.attrs["value"] = _parse_number(raw)

        if op.is_("for", "if", "while"):
            # Region ops print their result types as "-> (types)" before the optional " : iv".
            arrow = _find_top_level(rest, "->")
            result_types = []
            if arrow >= 0:
                tail = rest[arrow + 2:]
                end = _find_top_level(tail, " : ")
                result_types = self._types(tail if end < 0 else tail[:end])
        else:
            result_types = self._result_types(op, type_part)
        flat_results = []
        for res in results:
            base, _, count = res.partition(":")
            if count:
                flat_results.extend(f"{base}#{i}" for i in range(int(count)))
            else:
                flat_results.append(base)
        for idx, res in enumerate(flat_results):
            rtype = result_types[idx] if idx < len(result_types) else None
            val = Value(name=res, type=rtype, kind="result", owner=op, index=idx)
            op.results.append(val)
            self.scope.values[res] = val
            if "#" in res and idx == 0:
                self.scope.values[res.split("#")[0]] = val

    def _result_types(self, op: Op, type_part: str) -> list[Type]:
        if not type_part:
            return []
        arrow = _find_top_level(type_part, "->")
        to = _find_top_level(type_part, " to ")
        if arrow >= 0:
            ins = type_part[:arrow].strip()
            outs = type_part[arrow + 2:].strip()
            op.operand_types = self._types(ins)
            return self._types(outs)
        if to >= 0:
            op.operand_types = self._types(type_part[:to])
            return self._types(type_part[to + 4:])
        types = self._types(type_part)
        op.operand_types = types
        if op.is_("load") and types and isinstance(types[0], TensorType):
            ptr = types[0]
            elem = ptr.element.pointee if isinstance(ptr.element, PointerType) else ptr.element
            return [TensorType(text=ptr.text, shape=ptr.shape, element=elem, encoding=ptr.encoding)]
        if op.is_("load") and types and isinstance(types[0], PointerType):
            return [types[0].pointee] if types[0].pointee else []
        if op.is_("cmpi", "cmpf") and types:
            t = types[-1]
            if isinstance(t, TensorType):
                return [TensorType(text=t.text, shape=t.shape, element=ScalarType("i1", "i1"),
                                   encoding=t.encoding)]
            return [ScalarType("i1", "i1")]
        return types[-1:] if not op.is_("store", "local_store", "yield") else []

    def _types(self, text: str) -> list[Type]:
        text = text.strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]
        out = []
        for part in split_top_level(text):
            for piece in split_top_level(part, " * "):
                try:
                    out.append(parse_type(piece, self.aliases))
                except (ValueError, IndexError):
                    out.append(ScalarType(piece, piece))
        return out

    def _bind_region_args(self, op: Op, rest: str) -> None:
        if not op.is_("for"):
            return
        m = re.match(r"\s*(%[\w$.-]+)\s*=", rest)
        if m:
            iv = Value(m.group(1), ScalarType("i32", "i32"), "block_arg", owner=op, index=0)
            self.scope.values[iv.name] = iv
            op.block_args.append(iv)
        m = re.search(r"iter_args\((.*?)\)\s*->", rest)
        if not m:
            return
        result_types = [r.type for r in op.results]
        for idx, pair in enumerate(split_top_level(m.group(1))):
            name, _, init = pair.partition("=")
            init_val = self.parent_lookup(init.strip())
            val = Value(name.strip(), result_types[idx] if idx < len(result_types) else None,
                        "block_arg", owner=op, index=idx + 1, init=init_val)
            self.scope.values[val.name] = val
            op.block_args.append(val)

    def parent_lookup(self, name: str) -> Value | None:
        scope = self.scope.parent or self.scope
        return scope.lookup(name)

    def _finish_generic(self, op: Op, type_part: str) -> None:
        if type_part.startswith(":"):
            type_part = type_part[1:].strip()
        types = self._result_types(op, type_part)
        for val, t in zip(op.results, types):
            val.type = t

    def build(self) -> Func:
        # Link scf.yield values back to loop-carried block args.
        for op in self.ops:
            if op.is_("yield") and op.parent is not None and op.parent.is_("for"):
                carried = [a for a in op.parent.block_args if a.init is not None]
                for arg, val in zip(carried, op.operands):
                    arg.yielded = val
        return Func(name=self.name, args=self.args, ops=self.ops, line_no=self.line_no)


def _parse_number(raw: str) -> Any:
    raw = raw.strip()
    if raw in ("true", "false"):
        return raw == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def parse_file(path) -> Module:
    from pathlib import Path

    return parse_module(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "Encoding", "Func", "Loc", "MemDescType", "Module", "Op", "PointerType", "ScalarType",
    "TensorType", "Value", "parse_file", "parse_module",
]
