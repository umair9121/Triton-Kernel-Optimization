"""Global-memory access model: vector width and sector coalescing per load/store.

For every ``tt.load``, ``tt.store`` and ``async_copy_global_to_local``:

* the vector width Triton can use is ``min(layout contiguity, pointer contiguity,
  pointer alignment, mask constancy, 16 bytes)`` along the layout's fastest dimension, which
  is the rule Triton's own lowering applies; each limiting factor is reported;
* the addresses warp 0 touches are generated from the register layout and the symbolic
  strides, giving 32-byte sectors per request and a coalescing efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..ir import Func, Op, TensorType, Value
from ..layouts import BlockedLayout, DistributedLayout, UnsupportedLayout, from_encoding
from .axis import AxisAnalysis, AxisInfo
from .ptx import PtxStats

SECTOR = 32
UNKNOWN_STRIDE = 1 << 14  # elements; any value >> one sector reproduces the access pattern


@dataclass
class Limit:
    factor: str  # "layout", "contiguity", "alignment", "mask", "max"
    elements: int
    detail: str = ""


@dataclass
class GlobalAccess:
    op: Op
    kind: str
    shape: list[int]
    elem_bytes: int
    layout: DistributedLayout
    fast_dim: int
    vec: int
    potential_vec: int
    limits: list[Limit]
    ptr_info: AxisInfo
    mask_info: AxisInfo | None
    base_arg: Value | None
    sectors_per_request: float
    ideal_sectors: float
    requests_per_warp: int
    sample: list[dict] = field(default_factory=list)
    ptx: list[dict] = field(default_factory=list)

    @property
    def vec_bytes(self) -> int:
        return self.vec * self.elem_bytes

    @property
    def efficiency(self) -> float:
        return self.ideal_sectors / self.sectors_per_request if self.sectors_per_request else 1.0

    @property
    def ptx_bytes(self) -> int | None:
        widths = [p["bytes"] for p in self.ptx]
        return max(widths) if widths else None

    def binding_limits(self) -> list[Limit]:
        return [lim for lim in self.limits if lim.elements == self.vec and lim.factor != "max"]

    def to_json(self) -> dict:
        return {
            "op": self.op.id,
            "opName": self.op.name,
            "kind": self.kind,
            "line": self.op.line_no,
            "loc": self.op.loc.to_json() if self.op.loc else None,
            "shape": self.shape,
            "elemBytes": self.elem_bytes,
            "layout": self.layout.describe(),
            "fastDim": self.fast_dim,
            "vec": self.vec,
            "vecBytes": self.vec_bytes,
            "potentialVec": self.potential_vec,
            "limits": [{"factor": lim.factor, "elements": lim.elements, "detail": lim.detail}
                       for lim in self.limits],
            "ptr": self.ptr_info.to_json(),
            "mask": self.mask_info.to_json() if self.mask_info else None,
            "baseArg": self.base_arg.name if self.base_arg is not None else None,
            "sectorsPerRequest": round(self.sectors_per_request, 2),
            "idealSectors": round(self.ideal_sectors, 2),
            "efficiency": round(self.efficiency, 3),
            "requestsPerWarp": self.requests_per_warp,
            "sample": self.sample,
            "ptx": self.ptx,
            "ptxBytes": self.ptx_bytes,
        }


def _pow2_floor(n: int) -> int:
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def base_pointer_arg(val: Value) -> Value | None:
    """Walk addptr/splat/broadcast/loop-carried chains back to the pointer function argument."""
    seen = 0
    cur: Value | None = val
    while cur is not None and seen < 256:
        seen += 1
        if cur.kind == "arg":
            return cur
        if cur.kind == "block_arg":
            cur = cur.init
            continue
        owner = cur.owner
        if owner is None or not owner.operands:
            return None
        cur = owner.operands[0]
    return None


def _concrete_stride(stride, arg_values: dict[str, int]) -> int:
    if isinstance(stride, int):
        return stride
    if isinstance(stride, tuple):
        coef, name = stride
        value = arg_values.get(name)
        return coef * (value if isinstance(value, int) else UNKNOWN_STRIDE)
    return UNKNOWN_STRIDE


def _operands(op: Op) -> tuple[Value, Value | None, TensorType | None]:
    ptr = op.operands[0]
    mask = None
    if op.is_("load"):
        mask = op.operands[1] if len(op.operands) > 1 else None
    elif op.is_("store"):
        mask = op.operands[2] if len(op.operands) > 2 else None
    elif op.is_("async_copy_global_to_local"):
        mask = op.operands[2] if len(op.operands) > 2 else None
    ptype = ptr.type if isinstance(ptr.type, TensorType) else None
    if mask is not None and not (isinstance(mask.type, TensorType) and
                                 mask.type.element_name == "i1"):
        mask = None
    return ptr, mask, ptype


def analyze_access(op: Op, axis: AxisAnalysis, num_warps: int, tpw: int,
                   arg_values: dict[str, int], ptx: PtxStats | None = None
                   ) -> GlobalAccess | None:
    ptr, mask, ptype = _operands(op)
    if ptype is None or ptype.encoding is None:
        return None
    try:
        layout = from_encoding(ptype.encoding, num_warps, tpw)
    except (UnsupportedLayout, KeyError):
        return None
    if not isinstance(layout, DistributedLayout):
        return None
    shape = list(ptype.shape)
    elem = ptype.element_bytes
    fast = layout.order[0] if isinstance(layout, BlockedLayout) else len(shape) - 1
    info = axis[ptr]
    mask_info = axis[mask] if mask is not None else None

    max_elems = max(1, 16 // elem)
    layout_contig = layout.contiguous_elements(shape, fast)
    contig = info.contiguity[fast] if fast < info.rank else 1
    align_elems = max(1, info.divisibility[fast] // elem) if fast < info.rank else 1
    base = base_pointer_arg(ptr)
    limits = [
        Limit("layout", layout_contig, f"sizePerThread gives {layout_contig} contiguous "
                                       f"elements per thread along dim {fast}"),
        Limit("contiguity", contig, f"pointer offsets are contiguous in runs of {contig} "
                                    f"along dim {fast}"),
        Limit("alignment", align_elems, f"addresses are only known to be "
                                        f"{info.divisibility[fast]}-byte aligned"),
        Limit("max", max_elems, "128-bit maximum per lane"),
    ]
    if mask_info is not None and fast < mask_info.rank:
        limits.append(Limit("mask", mask_info.constancy[fast],
                            f"mask is constant over runs of {mask_info.constancy[fast]}"))
    vec = _pow2_floor(max(1, min(lim.elements for lim in limits)))
    potential = _pow2_floor(max(1, min(contig, max_elems, shape[fast])))
    for lim in limits:
        lim.elements = _pow2_floor(max(1, lim.elements))

    strides = [_concrete_stride(s, arg_values) for s in info.stride]
    sample, sectors, ideal, n_req = _simulate(layout, shape, strides, elem, vec)

    access = GlobalAccess(
        op=op, kind=op.short_name, shape=shape, elem_bytes=elem, layout=layout, fast_dim=fast,
        vec=vec, potential_vec=potential, limits=limits, ptr_info=info, mask_info=mask_info,
        base_arg=base, sectors_per_request=sectors, ideal_sectors=ideal,
        requests_per_warp=n_req, sample=sample,
    )
    if ptx is not None and op.loc is not None:
        kinds = ("cp.async",) if op.is_("async_copy_global_to_local") else (
            ("ld",) if op.is_("load") else ("st",))
        space = "global->shared" if kinds == ("cp.async",) else "global"
        matched = ptx.at(op.loc.line, op.loc.col, space=space, kinds=kinds, file=op.loc.file)
        access.ptx = [m.to_json() for m in matched]
    return access


def _simulate(layout: DistributedLayout, shape: list[int], strides: list[int], elem: int,
              vec: int) -> tuple[list[dict], float, float, int]:
    """Sectors per warp request for warp 0, averaged over its requests."""
    lanes = [layout.thread_elements(shape, 0, l) for l in range(layout.threads_per_warp)]
    n_req = max(1, max(len(e) for e in lanes) // vec)
    total_sectors = total_ideal = 0.0
    sample: list[dict] = []
    for j in range(n_req):
        sectors: set[int] = set()
        unique_bytes: set[int] = set()
        for l, elems in enumerate(lanes):
            if j * vec >= len(elems):
                continue
            coord = elems[j * vec]
            addr = sum(c * s for c, s in zip(coord, strides)) * elem
            for b in range(0, vec * elem, 4 if elem >= 4 else elem):
                unique_bytes.add(addr + b)
            for off in range(0, vec * elem, SECTOR):
                sectors.add((addr + off) // SECTOR)
            sectors.add((addr + vec * elem - 1) // SECTOR)
            if j == 0:
                sample.append({"lane": l, "coord": list(coord), "addr": addr,
                               "sector": addr // SECTOR})
        nbytes = len(unique_bytes) * (4 if elem >= 4 else elem)
        total_sectors += len(sectors)
        total_ideal += max(1, -(-nbytes // SECTOR))
    return sample, total_sectors / n_req, total_ideal / n_req, n_req


def analyze_function(func: Func, num_warps: int, tpw: int = 32,
                     arg_values: dict[str, int] | None = None, ptx: PtxStats | None = None,
                     axis: AxisAnalysis | None = None) -> list[GlobalAccess]:
    arg_values = arg_values or {}
    axis = axis or AxisAnalysis(func, arg_values)
    out = []
    for op in func.ops:
        if op.is_("load", "store", "async_copy_global_to_local"):
            access = analyze_access(op, axis, num_warps, tpw, arg_values, ptx)
            if access is not None:
                out.append(access)
    return out
