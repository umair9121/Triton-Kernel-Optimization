"""PTX statistics: memory instructions attributed to source lines through ``.loc``.

This closes the loop between the static prediction and what the compiler actually emitted:
for every ``tt.load`` we can show whether PTX contains ``ld.global.v4.b32`` (16 B per lane)
or scalar ``ld.global.b32``.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

_LOC_RE = re.compile(r"^\s*\.loc\s+(\d+)\s+(\d+)\s+(\d+)")
_FILE_RE = re.compile(r'^\s*\.file\s+(\d+)\s+"([^"]*)"')
_MEM_RE = re.compile(
    r"(?:^|[\s;{])(?:@!?%p\d+\s+)?"
    r"(ldmatrix|stmatrix|cp\.async|ld|st|atom|red)"
    r"((?:\.[\w]+)*)(?=[\s{\[])"
)
_TYPE_BITS = {"b8": 8, "u8": 8, "s8": 8, "b16": 16, "u16": 16, "s16": 16, "f16": 16,
              "b32": 32, "u32": 32, "s32": 32, "f32": 32, "b64": 64, "u64": 64, "s64": 64,
              "f64": 64, "b128": 128}


@dataclass
class MemInstr:
    opcode: str
    space: str  # "global", "shared", "local", ...
    bytes_per_lane: int
    text: str
    file: str | None
    line: int | None
    col: int | None

    def to_json(self) -> dict:
        return {"opcode": self.opcode, "space": self.space, "bytes": self.bytes_per_lane,
                "line": self.line, "col": self.col}


@dataclass
class PtxStats:
    instrs: list[MemInstr] = field(default_factory=list)
    mma: int = 0
    registers: int | None = None

    def histogram(self) -> dict[str, int]:
        return dict(Counter(i.opcode for i in self.instrs))

    def at(self, line: int, col: int | None = None, space: str | None = None,
           kinds: tuple[str, ...] = (), file: str | None = None) -> list[MemInstr]:
        out = [i for i in self.instrs if i.line == line and (space is None or i.space == space)]
        if file:
            base = _basename(file)
            out = [i for i in out if i.file is None or _basename(i.file) == base]
        if kinds:
            out = [i for i in out if any(i.opcode == k or i.opcode.startswith(k + ".")
                                         for k in kinds)]
        if col is not None:
            exact = [i for i in out if i.col == col]
            if exact:
                return exact
        return out

    def to_json(self) -> dict:
        return {"histogram": self.histogram(), "mma": self.mma,
                "instructions": [i.to_json() for i in self.instrs]}


def _basename(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def _bytes(suffixes: list[str], opcode: str) -> int:
    vec = 1
    bits = 32
    for s in suffixes:
        m = re.fullmatch(r"v(\d)", s)
        if m:
            vec = int(m.group(1))
        elif s in _TYPE_BITS:
            bits = _TYPE_BITS[s]
        elif s in ("x1", "x2", "x4") and opcode == "ldmatrix":
            vec = int(s[1])
    if opcode == "ldmatrix":
        return 16  # every lane supplies one 16-byte row address per matrix
    return vec * bits // 8


def parse_ptx(text: str) -> PtxStats:
    stats = PtxStats()
    files: dict[str, str] = {}
    cur: tuple[str | None, int | None, int | None] = (None, None, None)
    for raw in text.splitlines():
        line = raw.split("//", 1)[0]
        m = _FILE_RE.match(line)
        if m:
            files[m.group(1)] = m.group(2)
            continue
        m = _LOC_RE.match(line)
        if m:
            cur = (files.get(m.group(1)), int(m.group(2)), int(m.group(3)))
            continue
        if "mma.sync" in line:
            stats.mma += 1
            continue
        if ".reg" in line or not line.strip():
            continue
        for m in _MEM_RE.finditer(line):
            opcode = m.group(1)
            suffixes = [s for s in m.group(2).split(".") if s]
            space = next((s for s in suffixes if s in ("global", "shared", "local", "const",
                                                       "param")), "")
            if opcode in ("ld", "st") and not space:
                continue
            if opcode == "ld" and space == "param":
                continue
            if opcode == "cp.async":
                if not suffixes or suffixes[0] not in ("ca", "cg"):
                    continue
                size = re.search(r",\s*(0x[0-9a-fA-F]+|\d+)\s*(?:,|;)", line)
                nbytes = int(size.group(1), 0) if size else 16
                space = "global->shared"
            else:
                nbytes = _bytes(suffixes, opcode)
            name = opcode + ("." + ".".join(suffixes) if suffixes else "")
            stats.instrs.append(MemInstr(name, space, nbytes, raw.strip(), *cur))
    return stats
