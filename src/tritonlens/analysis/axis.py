"""Axis analysis: per-dimension contiguity, divisibility and constancy of integer tensors.

This follows the semantics of Triton's own ``AxisInfo`` pass, which is what decides how wide
each global load/store can be vectorized:

* ``contiguity[d]``: length of runs of consecutive integers along ``d``.
* ``divisibility[d]``: largest power of two dividing the first element of every run.
  For pointers it is measured in bytes.
* ``constancy[d]``: length of runs of equal values along ``d``.

On top of that we track a symbolic ``stride`` per dimension (element offset per index step:
an int, ``(coef, arg_name)`` for a runtime scalar, or ``None``) and the scalar ``origin`` that
positions each dimension's tile, which is what the coalescing and tiling views need.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..ir import Func, Op, PointerType, TensorType, Value

CAP = 1 << 20  # stand-in for "divisible by anything" (e.g. the constant 0)


def pow2_divisor(n: int) -> int:
    if n == 0:
        return CAP
    n = abs(n)
    return min(n & -n, CAP)


def _gcd(a: int, b: int) -> int:
    return math.gcd(a, b) if a and b else max(a, b)


Stride = object  # int | tuple[int, str] | None


def _stride_add(a, b, sign: int = 1):
    if a is None or b is None:
        return None
    if isinstance(a, int) and isinstance(b, int):
        return a + sign * b
    if isinstance(b, int) and b == 0:
        return a
    if isinstance(a, int) and a == 0:
        return b if sign == 1 else (_stride_scale(b, -1))
    if isinstance(a, tuple) and isinstance(b, tuple) and a[1] == b[1]:
        return (a[0] + sign * b[0], a[1])
    return None


def _stride_scale(s, k: int):
    if s is None:
        return None
    if isinstance(s, int):
        return s * k
    return (s[0] * k, s[1])


@dataclass
class AxisInfo:
    contiguity: list[int]
    divisibility: list[int]
    constancy: list[int]
    stride: list = field(default_factory=list)
    value: int | None = None  # known constant (scalar or splat)
    sym: tuple[int, str] | None = None  # scalar = coef * <arg>
    origin: list = field(default_factory=list)  # per dim: scalar Value positioning the tile
    approx: bool = False

    @property
    def rank(self) -> int:
        return len(self.contiguity)

    def to_json(self) -> dict:
        def fmt_stride(s):
            if isinstance(s, tuple):
                return f"{s[0]}*{s[1]}" if s[0] != 1 else s[1]
            return s

        return {
            "contiguity": self.contiguity,
            "divisibility": self.divisibility,
            "constancy": self.constancy,
            "stride": [fmt_stride(s) for s in self.stride],
            "value": self.value,
            "approx": self.approx,
        }


def _shape(val: Value) -> list[int]:
    t = val.type
    if isinstance(t, TensorType):
        return list(t.shape)
    return [1]


def _elem_bytes(val: Value) -> int:
    t = val.type
    if isinstance(t, TensorType):
        return t.element_bytes
    if isinstance(t, PointerType):
        from ..ir.types import ELEMENT_BYTES

        return ELEMENT_BYTES.get(getattr(t.pointee, "name", ""), 4)
    return 4


def _default(shape: list[int]) -> AxisInfo:
    n = len(shape)
    return AxisInfo([1] * n, [1] * n, [1] * n, [None] * n, origin=[None] * n)


def _splat_info(src: AxisInfo, shape: list[int]) -> AxisInfo:
    n = len(shape)
    div = src.divisibility[0] if src.divisibility else 1
    return AxisInfo([1] * n, [div] * n, list(shape), [0] * n, value=src.value, sym=src.sym,
                    origin=[None] * n)


class AxisAnalysis:
    """Run the analysis over a function. ``arg_values`` optionally maps arg names to ints."""

    def __init__(self, func: Func, arg_values: dict[str, int] | None = None):
        self.func = func
        self.arg_values = arg_values or {}
        self.info: dict[int, AxisInfo] = {}  # keyed by id(Value)
        self._run()

    def __getitem__(self, val: Value) -> AxisInfo:
        return self.info.get(id(val)) or _default(_shape(val))

    def _run(self) -> None:
        for arg in self.func.args:
            self.info[id(arg)] = self._arg_info(arg)
        # Two passes: loop-carried values start from their init, then widen with the yield.
        for pass_no in range(2):
            for op in self.func.ops:
                if op.is_("for"):
                    self._enter_for(op, widen=pass_no == 1)
                try:
                    self._visit(op)
                except (IndexError, KeyError, TypeError, ZeroDivisionError):
                    for res in op.results:
                        self.info[id(res)] = _default(_shape(res))

    def _arg_info(self, arg: Value) -> AxisInfo:
        div = int(arg.attrs.get("tt.divisibility", 1))
        info = AxisInfo([1], [div], [1], [0], origin=[None])
        if isinstance(arg.type, PointerType):
            return info
        value = self.arg_values.get(arg.name)
        if isinstance(value, int):
            info.value = value
        info.sym = (1, arg.name)
        return info

    def _enter_for(self, op: Op, widen: bool) -> None:
        for arg in op.block_args:
            if arg.init is None:
                # Induction variable: lb + i * step.
                lb = self[op.operands[0]] if op.operands else _default([1])
                step = self[op.operands[2]] if len(op.operands) > 2 else _default([1])
                div = _gcd(lb.divisibility[0], step.divisibility[0])
                self.info[id(arg)] = AxisInfo([1], [div], [1], [0], origin=[None])
                continue
            init = self[arg.init]
            if widen and arg.yielded is not None:
                self.info[id(arg)] = _meet(init, self[arg.yielded])
            else:
                self.info[id(arg)] = init
        for idx, res in enumerate(op.results):
            carried = [a for a in op.block_args if a.init is not None]
            if idx < len(carried):
                self.info[id(res)] = self.info.get(id(carried[idx]), _default(_shape(res)))

    # -- transfer functions -------------------------------------------------------------

    def _visit(self, op: Op) -> None:
        if not op.results:
            return
        name = op.short_name
        res = op.results[0]
        shape = _shape(res)
        handler = getattr(self, f"_op_{name}", None)
        if op.is_("for"):
            carried = [a for a in op.block_args if a.init is not None]
            for idx, r in enumerate(op.results):
                if idx < len(carried):
                    self.info[id(r)] = self.info.get(id(carried[idx]), _default(_shape(r)))
            return
        info = handler(op, shape) if handler else None
        if info is None:
            info = _default(shape)
        self.info[id(res)] = info
        for extra in op.results[1:]:
            self.info[id(extra)] = _default(_shape(extra))

    def _ops(self, op: Op) -> list[AxisInfo]:
        return [self[v] for v in op.operands]

    def _op_constant(self, op: Op, shape):
        value = op.attrs.get("value")
        if not isinstance(value, int) or isinstance(value, bool):
            return AxisInfo([1] * len(shape), [1] * len(shape), list(shape), [0] * len(shape),
                            origin=[None] * len(shape))
        n = len(shape)
        return AxisInfo([1] * n, [pow2_divisor(value)] * n, list(shape), [0] * n, value=value,
                        origin=[None] * n)

    def _op_make_range(self, op: Op, shape):
        start = int(op.attrs.get("start", 0))
        return AxisInfo([shape[0]], [pow2_divisor(start)], [1], [1], origin=[None])

    def _op_get_program_id(self, op: Op, shape):
        return AxisInfo([1], [1], [1], [0], origin=[None])

    _op_get_num_programs = _op_get_program_id

    def _op_splat(self, op: Op, shape):
        info = _splat_info(self[op.operands[0]], shape)
        return info

    def _op_expand_dims(self, op: Op, shape):
        src = self[op.operands[0]]
        axis = int(op.attrs.get("axis", 0))
        div_new = min(src.divisibility) if src.divisibility else 1
        info = AxisInfo(list(src.contiguity), list(src.divisibility), list(src.constancy),
                        list(src.stride), src.value, src.sym, list(src.origin), src.approx)
        info.contiguity.insert(axis, 1)
        info.divisibility.insert(axis, div_new)
        info.constancy.insert(axis, 1)
        info.stride.insert(axis, 0)
        info.origin.insert(axis, None)
        return info

    def _op_broadcast(self, op: Op, shape):
        src = self[op.operands[0]]
        src_shape = _shape(op.operands[0])
        info = AxisInfo(list(src.contiguity), list(src.divisibility), list(src.constancy),
                        list(src.stride), src.value, src.sym, list(src.origin), src.approx)
        for d, (s_old, s_new) in enumerate(zip(src_shape, shape)):
            if s_old == 1 and s_new != 1:
                info.constancy[d] = s_new
                info.contiguity[d] = 1
                info.stride[d] = 0
        return info

    def _passthrough(self, op: Op, shape):
        src = self[op.operands[0]]
        return AxisInfo(list(src.contiguity), list(src.divisibility), list(src.constancy),
                        list(src.stride), src.value, src.sym, list(src.origin), src.approx)

    _op_convert_layout = _passthrough
    _op_extsi = _passthrough
    _op_extui = _passthrough
    _op_trunci = _passthrough
    _op_index_cast = _passthrough
    _op_bitcast = _passthrough

    def _op_trans(self, op: Op, shape):
        src = self[op.operands[0]]
        order = op.attrs.get("order") or list(range(src.rank))[::-1]
        if not isinstance(order, list):
            order = list(range(src.rank))[::-1]
        perm = lambda xs: [xs[i] for i in order]  # noqa: E731
        return AxisInfo(perm(src.contiguity), perm(src.divisibility), perm(src.constancy),
                        perm(src.stride), src.value, src.sym, perm(src.origin), src.approx)

    def _binary_add(self, op: Op, shape, sign: int):
        a, b = self._ops(op)[:2]
        n = len(shape)
        info = _default(shape)
        for d in range(n):
            info.contiguity[d] = max(_gcd(a.contiguity[d], b.constancy[d]),
                                     _gcd(a.constancy[d], b.contiguity[d]))
            info.divisibility[d] = _gcd(a.divisibility[d], b.divisibility[d])
            info.constancy[d] = _gcd(a.constancy[d], b.constancy[d])
            info.stride[d] = _stride_add(a.stride[d], b.stride[d], sign)
            info.origin[d] = a.origin[d] or b.origin[d]
            # tile origin: splat(scalar) + range along d
            if info.origin[d] is None:
                if b.contiguity[d] > 1 and a.constancy[d] == shape[d] and op.operands[0].owner \
                        is not None and op.operands[0].owner.is_("splat"):
                    info.origin[d] = op.operands[0].owner.operands[0]
                elif a.contiguity[d] > 1 and b.constancy[d] == shape[d] and \
                        op.operands[1].owner is not None and op.operands[1].owner.is_("splat"):
                    info.origin[d] = op.operands[1].owner.operands[0]
        if a.value is not None and b.value is not None:
            info.value = a.value + sign * b.value
        info.sym = _sym_add(a, b, sign)
        info.approx = a.approx or b.approx
        if sign == -1:
            # Subtraction breaks contiguity unless the rhs is constant.
            for d in range(n):
                if b.constancy[d] <= 1:
                    info.contiguity[d] = 1
        return info

    def _op_addi(self, op: Op, shape):
        return self._binary_add(op, shape, 1)

    def _op_subi(self, op: Op, shape):
        return self._binary_add(op, shape, -1)

    def _op_addptr(self, op: Op, shape):
        ptr, off = self._ops(op)[:2]
        elem = _elem_bytes(op.results[0])
        n = len(shape)
        info = _default(shape)
        for d in range(n):
            info.contiguity[d] = max(_gcd(ptr.contiguity[d], off.constancy[d]),
                                     _gcd(ptr.constancy[d], off.contiguity[d]))
            info.divisibility[d] = _gcd(ptr.divisibility[d], off.divisibility[d] * elem)
            info.constancy[d] = _gcd(ptr.constancy[d], off.constancy[d])
            info.stride[d] = _stride_add(ptr.stride[d], off.stride[d])
            info.origin[d] = ptr.origin[d] or off.origin[d]
        info.approx = ptr.approx or off.approx
        return info

    def _op_muli(self, op: Op, shape):
        a, b = self._ops(op)[:2]
        n = len(shape)
        info = _default(shape)
        for d in range(n):
            a_div = a.divisibility[d] if (a.contiguity[d] <= 1 or b.value == 1) else 1
            b_div = b.divisibility[d] if (b.contiguity[d] <= 1 or a.value == 1) else 1
            info.divisibility[d] = min(a_div * b_div, CAP)
            info.constancy[d] = _gcd(a.constancy[d], b.constancy[d])
            if b.value == 1:
                info.contiguity[d] = a.contiguity[d]
            elif a.value == 1:
                info.contiguity[d] = b.contiguity[d]
            info.stride[d] = _stride_mul(a, b, d)
            info.origin[d] = a.origin[d] or b.origin[d]
        if a.value is not None and b.value is not None:
            info.value = a.value * b.value
        elif a.value is not None and b.sym is not None:
            info.sym = (b.sym[0] * a.value, b.sym[1])
        elif b.value is not None and a.sym is not None:
            info.sym = (a.sym[0] * b.value, a.sym[1])
        info.approx = a.approx or b.approx
        return info

    def _op_remsi(self, op: Op, shape):
        a, b = self._ops(op)[:2]
        n = len(shape)
        info = _default(shape)
        for d in range(n):
            if b.constancy[d] == shape[d] or b.constancy[d] > 1:
                info.contiguity[d] = _gcd(a.contiguity[d], b.divisibility[d])
            info.divisibility[d] = _gcd(a.divisibility[d], b.divisibility[d])
            info.constancy[d] = _gcd(a.constancy[d], b.constancy[d])
            # x % M keeps the stride inside a tile unless the tile straddles M.
            info.stride[d] = a.stride[d] if b.stride[d] == 0 else None
            info.origin[d] = a.origin[d]
        if a.value is not None and b.value:
            info.value = int(math.fmod(a.value, b.value))
        info.approx = True
        return info

    _op_remui = _op_remsi

    def _op_divsi(self, op: Op, shape):
        a, b = self._ops(op)[:2]
        n = len(shape)
        info = _default(shape)
        for d in range(n):
            info.constancy[d] = a.constancy[d]
            if b.value and a.contiguity[d] > 1:
                info.constancy[d] = max(a.constancy[d], _gcd(a.contiguity[d], b.value))
            info.stride[d] = 0 if a.stride[d] == 0 else None
        if a.value is not None and b.value:
            info.value = int(a.value / b.value)
            info.divisibility = [pow2_divisor(info.value)] * n
        return info

    _op_divui = _op_divsi

    def _op_cmpi(self, op: Op, shape):
        a, b = self._ops(op)[:2]
        n = len(shape)
        info = _default(shape)
        pred = op.words[0] if op.words else ""
        for d in range(n):
            if a.constancy[d] > 1 and b.constancy[d] > 1:
                info.constancy[d] = _gcd(a.constancy[d], b.constancy[d])
            elif pred in ("slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"):
                if a.contiguity[d] > 1 and b.constancy[d] > 1:
                    info.constancy[d] = _gcd(a.contiguity[d],
                                             _gcd(a.divisibility[d], b.divisibility[d]))
                elif b.contiguity[d] > 1 and a.constancy[d] > 1:
                    info.constancy[d] = _gcd(b.contiguity[d],
                                             _gcd(a.divisibility[d], b.divisibility[d]))
            info.stride[d] = None
        return info

    def _op_andi(self, op: Op, shape):
        a, b = self._ops(op)[:2]
        n = len(shape)
        info = _default(shape)
        for d in range(n):
            info.constancy[d] = _gcd(a.constancy[d], b.constancy[d])
        return info

    _op_ori = _op_andi
    _op_xori = _op_andi

    def _op_select(self, op: Op, shape):
        c, a, b = self._ops(op)[:3]
        info = _meet(a, b)
        for d in range(len(shape)):
            info.constancy[d] = _gcd(info.constancy[d], c.constancy[d])
        return info

    def _op_minsi(self, op: Op, shape):
        a, b = self._ops(op)[:2]
        info = _meet(a, b)
        if a.value is not None and b.value is not None:
            info.value = min(a.value, b.value)
        return info

    def _op_maxsi(self, op: Op, shape):
        a, b = self._ops(op)[:2]
        info = _meet(a, b)
        if a.value is not None and b.value is not None:
            info.value = max(a.value, b.value)
        return info


def _sym_add(a: AxisInfo, b: AxisInfo, sign: int):
    if a.sym is not None and b.value is not None:
        return a.sym if b.value == 0 else None
    if b.sym is not None and a.value == 0:
        return (b.sym[0] * sign, b.sym[1])
    return None


def _stride_mul(a: AxisInfo, b: AxisInfo, d: int):
    sa, sb = a.stride[d], b.stride[d]
    if sa is None or sb is None:
        return None
    if sb == 0:
        if b.value is not None:
            return _stride_scale(sa, b.value)
        if sa == 0:
            return 0
        if isinstance(sa, int) and b.sym is not None:
            return (sa * b.sym[0], b.sym[1])
        return None
    if sa == 0:
        if a.value is not None:
            return _stride_scale(sb, a.value)
        if isinstance(sb, int) and a.sym is not None:
            return (sb * a.sym[0], a.sym[1])
        return None
    return None


def _meet(a: AxisInfo, b: AxisInfo) -> AxisInfo:
    n = min(a.rank, b.rank)
    return AxisInfo(
        [_gcd(a.contiguity[d], b.contiguity[d]) for d in range(n)],
        [_gcd(a.divisibility[d], b.divisibility[d]) for d in range(n)],
        [_gcd(a.constancy[d], b.constancy[d]) for d in range(n)],
        [a.stride[d] if a.stride[d] == b.stride[d] else None for d in range(n)],
        a.value if a.value == b.value else None,
        a.sym if a.sym == b.sym else None,
        [a.origin[d] or b.origin[d] for d in range(n)],
        a.approx or b.approx,
    )
