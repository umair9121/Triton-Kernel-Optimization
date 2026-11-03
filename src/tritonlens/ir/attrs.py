"""Parsing of MLIR attribute syntax as printed by Triton.

Handles the subset Triton emits for layouts and op attributes::

    #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], order = [1, 0]}>
    #ttg.slice<{dim = 1, parent = #blocked}>
    {end = 1024 : i32, start = 0 : i32}
    array<i32: 1, 0>
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

DIALECT_PREFIXES = ("triton_gpu.", "ttg.", "triton_nvidia_gpu.", "ttng.", "triton_amdgpu.", "amdgpu.")

# Newer Triton releases renamed some encodings; map them to the 3.2 names used internally.
KIND_ALIASES = {
    "swizzled_shared": "shared",
    "nvmma_shared": "nvmma_shared",
    "shared_memory": "shared_memory",
}


@dataclass
class Encoding:
    """A layout encoding attribute, e.g. ``#blocked`` or an inline ``#ttg.slice<...>``."""

    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    alias: str | None = None
    text: str = ""

    def __repr__(self) -> str:
        name = self.alias or self.kind
        return f"Encoding({name})"

    @property
    def name(self) -> str:
        return self.alias or self.text or self.kind

    def key(self) -> str:
        """Stable identity used to deduplicate layouts in reports."""
        if self.alias:
            return self.alias
        return self.text


def strip_dialect(name: str) -> str:
    for prefix in DIALECT_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return KIND_ALIASES.get(name, name)


class _Parser:
    def __init__(self, text: str, aliases: dict[str, Encoding] | None = None):
        self.text = text
        self.pos = 0
        self.aliases = aliases or {}

    def peek(self) -> str:
        self.skip_ws()
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def skip_ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def expect(self, token: str) -> None:
        self.skip_ws()
        if not self.text.startswith(token, self.pos):
            raise ValueError(f"expected {token!r} at {self.pos} in {self.text!r}")
        self.pos += len(token)

    def accept(self, token: str) -> bool:
        self.skip_ws()
        if self.text.startswith(token, self.pos):
            self.pos += len(token)
            return True
        return False

    def ident(self) -> str:
        self.skip_ws()
        m = re.compile(r"[A-Za-z_][\w.$-]*").match(self.text, self.pos)
        if not m:
            raise ValueError(f"expected identifier at {self.pos} in {self.text!r}")
        self.pos = m.end()
        return m.group(0)

    def balanced(self, open_ch: str, close_ch: str) -> str:
        """Consume a balanced ``open_ch ... close_ch`` group and return its inner text."""
        self.expect(open_ch)
        start = self.pos
        depth = 1
        while self.pos < len(self.text):
            ch = self.text[self.pos]
            if ch == '"':
                self.pos = self.text.index('"', self.pos + 1)
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    inner = self.text[start:self.pos]
                    self.pos += 1
                    return inner
            self.pos += 1
        raise ValueError(f"unbalanced {open_ch} in {self.text!r}")

    def dict_body(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        while self.peek() and self.peek() != "}":
            if self.peek() == '"':
                key = self.string()
            else:
                key = self.ident()
            if self.accept("="):
                out[key] = self.value()
            else:
                out[key] = True
            if not self.accept(","):
                break
        return out

    def string(self) -> str:
        self.expect('"')
        end = self.text.index('"', self.pos)
        value = self.text[self.pos:end]
        self.pos = end + 1
        return value

    def value(self) -> Any:
        ch = self.peek()
        if ch == "[":
            self.pos += 1
            items = []
            while self.peek() != "]":
                items.append(self.value())
                if not self.accept(","):
                    break
            self.expect("]")
            return items
        if ch == "{":
            self.pos += 1
            body = self.dict_body()
            self.expect("}")
            return body
        if ch == '"':
            return self.string()
        if ch == "#":
            return self.encoding()
        m = re.compile(r"-?\d+(\.\d+)?([eE][-+]?\d+)?").match(self.text, self.pos)
        if m:
            self.pos = m.end()
            number: Any = float(m.group(0)) if (m.group(1) or m.group(2)) else int(m.group(0))
            self._skip_type_suffix()
            return number
        word = self.ident()
        if word in ("true", "false"):
            return word == "true"
        if word == "array" and self.peek() == "<":
            inner = self.balanced("<", ">")
            _, _, nums = inner.partition(":")
            return [int(x) for x in nums.split(",") if x.strip()]
        if word == "dense" and self.peek() == "<":
            return {"dense": self.balanced("<", ">").strip()}
        return word

    def _skip_type_suffix(self) -> None:
        save = self.pos
        self.skip_ws()
        if self.text.startswith(":", self.pos) and not self.text.startswith("::", self.pos):
            self.pos += 1
            self.skip_ws()
            m = re.compile(r"[A-Za-z_][\w.]*").match(self.text, self.pos)
            if m:
                self.pos = m.end()
                return
        self.pos = save

    def encoding(self) -> Encoding:
        start = self.pos
        self.expect("#")
        name = self.ident()
        if self.peek() == "<":
            inner = self.balanced("<", ">").strip()
            params: dict[str, Any] = {}
            if inner.startswith("{") and inner.endswith("}"):
                params = _Parser(inner[1:-1], self.aliases).dict_body()
            text = self.text[start:self.pos]
            return Encoding(kind=strip_dialect(name), params=params, text=text)
        if name in self.aliases:
            return self.aliases[name]
        return Encoding(kind=strip_dialect(name), alias=None if "." in name else name,
                        text="#" + name)


def parse_attr_dict(text: str, aliases: dict[str, Encoding] | None = None) -> dict[str, Any]:
    """Parse ``{a = 1 : i32, b = [1, 2]}`` (braces optional)."""
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    return _Parser(text, aliases).dict_body()


def parse_encoding(text: str, aliases: dict[str, Encoding] | None = None) -> Encoding:
    """Parse ``#blocked`` (alias lookup) or an inline ``#dialect.kind<{...}>``."""
    return _Parser(text.strip(), aliases).encoding()
