"""MLIR type parsing for the types that appear in Triton GPU IR."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .attrs import Encoding, parse_encoding

ELEMENT_BYTES = {
    "i1": 1, "i8": 1, "i16": 2, "i32": 4, "i64": 8,
    "f8E4M3FN": 1, "f8E5M2": 1, "f8E4M3FNUZ": 1, "f8E5M2FNUZ": 1,
    "f16": 2, "bf16": 2, "f32": 4, "f64": 8, "tf32": 4,
}


@dataclass
class Type:
    text: str

    @property
    def is_tensor(self) -> bool:
        return False

    @property
    def is_memdesc(self) -> bool:
        return False


@dataclass
class ScalarType(Type):
    name: str = ""

    @property
    def nbytes(self) -> int:
        return ELEMENT_BYTES.get(self.name, 8 if self.name.startswith("!tt.ptr") else 4)


@dataclass
class PointerType(Type):
    pointee: Type | None = None
    address_space: int = 1


@dataclass
class TensorType(Type):
    shape: list[int] = field(default_factory=list)
    element: Type | None = None
    encoding: Encoding | None = None

    @property
    def is_tensor(self) -> bool:
        return True

    @property
    def is_pointer_tensor(self) -> bool:
        return isinstance(self.element, PointerType)

    @property
    def element_name(self) -> str:
        elem = self.element
        if isinstance(elem, PointerType):
            elem = elem.pointee
        return getattr(elem, "name", "") or (elem.text if elem else "")

    @property
    def element_bytes(self) -> int:
        """Size of the (pointee) element in bytes."""
        return ELEMENT_BYTES.get(self.element_name, 4)

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n


@dataclass
class MemDescType(Type):
    shape: list[int] = field(default_factory=list)
    element: Type | None = None
    encoding: Encoding | None = None
    memory_space: str = ""
    mutable: bool = False

    @property
    def is_memdesc(self) -> bool:
        return True

    @property
    def element_name(self) -> str:
        return getattr(self.element, "name", "")

    @property
    def element_bytes(self) -> int:
        return ELEMENT_BYTES.get(self.element_name, 4)


def split_top_level(text: str, sep: str = ",") -> list[str]:
    """Split on ``sep`` outside of any bracket pair or string."""
    parts, depth, start, i = [], 0, 0, 0
    in_str = False
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            # "->" is an arrow, not a closing bracket.
            if not (ch == ">" and i > 0 and text[i - 1] == "-"):
                depth -= 1
        elif depth == 0 and text.startswith(sep, i):
            parts.append(text[start:i])
            i += len(sep)
            start = i
            continue
        i += 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


def _shape_and_element(body: str) -> tuple[list[int], str]:
    shape: list[int] = []
    rest = body.strip()
    while True:
        m = re.match(r"(\d+|\?)x", rest)
        if not m:
            break
        shape.append(int(m.group(1)) if m.group(1) != "?" else -1)
        rest = rest[m.end():]
    return shape, rest


def parse_type(text: str, aliases: dict[str, Encoding] | None = None) -> Type:
    text = text.strip()
    if text.startswith("tensor<") and text.endswith(">"):
        parts = split_top_level(text[len("tensor<"):-1])
        shape, elem_text = _shape_and_element(parts[0])
        enc = parse_encoding(parts[1], aliases) if len(parts) > 1 else None
        return TensorType(text=text, shape=shape, element=parse_type(elem_text, aliases),
                          encoding=enc)
    m = re.match(r"!(?:tt|triton_gpu|ttg)\.memdesc<(.*)>$", text)
    if m:
        parts = split_top_level(m.group(1))
        shape, elem_text = _shape_and_element(parts[0])
        has_enc = len(parts) > 1 and parts[1].startswith("#")
        enc = parse_encoding(parts[1], aliases) if has_enc else None
        space = parts[2] if len(parts) > 2 else ""
        return MemDescType(text=text, shape=shape, element=parse_type(elem_text, aliases),
                           encoding=enc, memory_space=space, mutable="mutable" in parts[2:])
    m = re.match(r"!tt\.ptr<(.*)>$", text)
    if m:
        parts = split_top_level(m.group(1))
        space = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
        return PointerType(text=text, pointee=parse_type(parts[0], aliases), address_space=space)
    return ScalarType(text=text, name=text)
