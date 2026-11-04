"""Which part of the output each program writes: a concrete per-``program_id`` evaluator.

The pointer operand of the kernel's main ``tt.store`` is evaluated element-wise at the tile's
corner coordinates for every program in the launch grid, using the captured scalar argument
values. Element offsets are mapped back to (row, col) of the destination tensor through its
real strides. This reproduces any indexing scheme (grouped/swizzled matmul ordering, one row
per program, transposed stores) without pattern matching.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..ir import Func, Op, PointerType, TensorType, Value

MAX_PROGRAMS = 16384


class _Unknown(Exception):
    pass


def _trunc_div(a: int, b: int) -> int:
    if b == 0:
        raise _Unknown
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def _trunc_rem(a: int, b: int) -> int:
    return a - b * _trunc_div(a, b)


_CMP = {
    "eq": lambda a, b: a == b, "ne": lambda a, b: a != b,
    "slt": lambda a, b: a < b, "sle": lambda a, b: a <= b,
    "sgt": lambda a, b: a > b, "sge": lambda a, b: a >= b,
    "ult": lambda a, b: a < b, "ule": lambda a, b: a <= b,
    "ugt": lambda a, b: a > b, "uge": lambda a, b: a >= b,
}


class Evaluator:
    """Evaluate integer/pointer SSA values at one tensor coordinate for one program."""

    def __init__(self, arg_values: dict[str, int], pid: tuple[int, int, int],
                 grid: tuple[int, int, int]):
        self.arg_values = arg_values
        self.pid = pid
        self.grid = grid
        self.memo: dict[tuple[int, tuple], int] = {}

    def eval(self, val: Value, coord: tuple[int, ...]) -> int:
        key = (id(val), coord)
        if key in self.memo:
            return self.memo[key]
        result = self._eval(val, coord)
        self.memo[key] = result
        return result

    def _eval(self, val: Value, coord: tuple[int, ...]) -> int:
        if val.kind == "arg":
            if isinstance(val.type, PointerType):
                return 0  # offsets are measured relative to the pointer argument
            if val.name in self.arg_values:
                return self.arg_values[val.name]
            raise _Unknown
        if val.kind == "block_arg":
            if val.init is not None:
                return self.eval(val.init, coord)
            owner = val.owner
            if owner is not None and owner.is_("for") and owner.operands:
                return self.eval(owner.operands[0], ())  # first iteration of the loop
            raise _Unknown
        op = val.owner
        if op is None:
            raise _Unknown
        return self._op(op, val, coord)

    def _op(self, op: Op, val: Value, coord: tuple[int, ...]) -> int:
        name = op.short_name
        ops = op.operands
        if name == "constant":
            v = op.attrs.get("value")
            if isinstance(v, bool) or not isinstance(v, int):
                raise _Unknown
            return v
        if name == "get_program_id":
            axis = "xyz".index(op.words[0]) if op.words else 0
            return self.pid[axis]
        if name == "get_num_programs":
            axis = "xyz".index(op.words[0]) if op.words else 0
            return self.grid[axis]
        if name == "make_range":
            return int(op.attrs.get("start", 0)) + coord[0]
        if name == "splat":
            return self.eval(ops[0], ())
        if name == "expand_dims":
            axis = int(op.attrs.get("axis", 0))
            return self.eval(ops[0], coord[:axis] + coord[axis + 1:])
        if name == "broadcast":
            src_shape = ops[0].type.shape if isinstance(ops[0].type, TensorType) else []
            src_coord = tuple(0 if s == 1 else c for s, c in zip(src_shape, coord))
            return self.eval(ops[0], src_coord)
        if name in ("convert_layout", "extsi", "extui", "trunci", "index_cast", "bitcast"):
            return self.eval(ops[0], coord)
        if name == "trans":
            order = op.attrs.get("order")
            if not isinstance(order, list):
                order = list(range(len(coord)))[::-1]
            src = [0] * len(coord)
            for out_dim, in_dim in enumerate(order):
                src[in_dim] = coord[out_dim]
            return self.eval(ops[0], tuple(src))
        if name == "addptr":
            elem = _pointee_bytes(op.results[0])
            return self.eval(ops[0], coord) + self.eval(ops[1], coord) * elem
        if name == "select":
            return self.eval(ops[1] if self.eval(ops[0], coord) else ops[2], coord)
        if name == "cmpi":
            pred = op.words[0] if op.words else "eq"
            return int(_CMP[pred](self.eval(ops[0], coord), self.eval(ops[1], coord)))
        if len(ops) >= 2:
            a, b = self.eval(ops[0], coord), self.eval(ops[1], coord)
            if name == "addi":
                return a + b
            if name == "subi":
                return a - b
            if name == "muli":
                return a * b
            if name in ("divsi", "divui"):
                return _trunc_div(a, b)
            if name in ("remsi", "remui"):
                return _trunc_rem(a, b)
            if name == "ceildivsi":
                return -_trunc_div(-a, b)
            if name in ("minsi", "minui"):
                return min(a, b)
            if name in ("maxsi", "maxui"):
                return max(a, b)
            if name == "andi":
                return a & b
            if name == "ori":
                return a | b
            if name == "shli":
                return a << b
            if name == "shrsi":
                return a >> b
        raise _Unknown


def _pointee_bytes(val: Value) -> int:
    t = val.type
    if isinstance(t, TensorType):
        return t.element_bytes
    if isinstance(t, PointerType):
        from ..ir.types import ELEMENT_BYTES

        return ELEMENT_BYTES.get(getattr(t.pointee, "name", ""), 4)
    return 4


@dataclass
class Tiling:
    op: Op
    matrix_shape: list[int]
    tile_shape: list[int]
    grid: list[int]
    programs: list[dict]
    arg: str | None

    def to_json(self) -> dict:
        return {
            "op": self.op.id,
            "line": self.op.line_no,
            "loc": self.op.loc.to_json() if self.op.loc else None,
            "matrixShape": self.matrix_shape,
            "tileShape": self.tile_shape,
            "grid": self.grid,
            "programs": self.programs,
            "arg": self.arg,
        }


def _main_store(func: Func) -> Op | None:
    stores = [op for op in func.ops if op.is_("store") and isinstance(op.operands[0].type,
                                                                        TensorType)]
    if not stores:
        return None
    return max(stores, key=lambda op: op.operands[0].type.numel)


def program_tiling(func: Func, arg_values: dict[str, int], grid: list[int],
                   base_tensor: dict | None, base_name: str | None = None) -> Tiling | None:
    """Compute the output tile written by every program (``None`` if not evaluable)."""
    store = _main_store(func)
    if store is None or not grid:
        return None
    ptr = store.operands[0]
    shape = list(ptr.type.shape)
    elem = ptr.type.element_bytes
    grid3 = tuple((list(grid) + [1, 1, 1])[:3])
    total = grid3[0] * grid3[1] * grid3[2]
    if total > MAX_PROGRAMS:
        return None
    strides = (base_tensor or {}).get("strides") or [1]
    tshape = (base_tensor or {}).get("shape") or None

    def to_rc(byte_off: int) -> list[int]:
        off = byte_off // elem
        if len(strides) >= 2:
            order = sorted(range(len(strides)), key=lambda d: -strides[d])
            out = [0] * len(strides)
            for d in order:
                out[d], off = divmod(off, strides[d]) if strides[d] else (0, off)
            return out[-2:]
        return [off]

    origin = tuple(0 for _ in shape)
    unit = [tuple(1 if i == d else 0 for i in range(len(shape))) for d in range(len(shape))]
    programs = []
    tile_dims: list[int] | None = None
    for z in range(grid3[2]):
        for y in range(grid3[1]):
            for x in range(grid3[0]):
                pid = (x, y, z)
                ev = Evaluator(arg_values, pid, grid3)
                try:
                    o = ev.eval(ptr, origin)
                    if tile_dims is None:
                        tile_dims = _tile_extent(ev, ptr, shape, origin, unit, to_rc, o)
                except (_Unknown, KeyError, IndexError, TypeError):
                    return None
                programs.append({"pid": [x, y, z], "origin": to_rc(o)})
    mshape = list(tshape[-2:]) if tshape and len(tshape) >= 2 else (
        [tshape[0]] if tshape else None)
    if mshape is None:
        ends = [max(p["origin"][i] for p in programs) for i in range(len(programs[0]["origin"]))]
        mshape = [e + t for e, t in zip(ends, tile_dims or [1] * len(ends))]
    return Tiling(op=store, matrix_shape=mshape, tile_shape=tile_dims or [1],
                  grid=list(grid3), programs=programs, arg=base_name)


def _tile_extent(ev: Evaluator, ptr: Value, shape, origin, unit, to_rc, o) -> list[int]:
    """Size of one program's tile in destination (row, col) coordinates."""
    base = to_rc(o)
    extent = [1] * len(base)
    for d, step in enumerate(unit):
        if shape[d] <= 1:
            continue
        moved = to_rc(ev.eval(ptr, step))
        for axis, (a, b) in enumerate(zip(base, moved)):
            if a != b:
                extent[axis] = max(extent[axis], shape[d] * abs(b - a))
    return extent
