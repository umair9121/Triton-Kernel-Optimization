"""Warp-level shared-memory bank-conflict simulation.

Hardware model (NVIDIA Volta and later): shared memory has 32 banks of 4 bytes and serves
128 bytes per wavefront. A warp instruction where each lane accesses ``W`` bytes is split into
``128 / W`` lane groups for ``W >= 8`` (half-warps for 8 B, quarter-warps for 16 B); within a
group, the number of wavefronts is the largest number of *distinct* 4-byte words any single
bank must deliver. Lanes reading the same word are served by one broadcast.

``ldmatrix.x4`` is four phases of 8 lanes, each lane providing one 16-byte row address, which
is exactly the 16-byte quarter-warp case.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..ir import MemDescType, Op, TensorType, Value
from ..layouts import (
    BlockedLayout,
    DistributedLayout,
    DotOperandLayout,
    MmaV2Layout,
    PaddedScratchLayout,
    SharedLayout,
    SliceLayout,
    UnsupportedLayout,
    from_encoding,
)
from ..layouts.shared import NUM_BANKS, bank_of

AddrFn = Callable[[tuple], int]


@dataclass
class LaneAccess:
    lane: int
    addr: int
    bank: int
    coord: list[int]

    def to_json(self) -> dict:
        return {"lane": self.lane, "addr": self.addr, "bank": self.bank, "coord": self.coord}


@dataclass
class WarpInstr:
    width: int
    lanes: list[LaneAccess]
    wavefronts: int
    ideal: int

    @property
    def degree(self) -> float:
        return self.wavefronts / self.ideal if self.ideal else 1.0

    def to_json(self) -> dict:
        return {
            "width": self.width,
            "wavefronts": self.wavefronts,
            "ideal": self.ideal,
            "lanes": [la.to_json() for la in self.lanes],
        }


def wavefronts(addrs: list[int | None], width: int) -> tuple[int, int]:
    """Return ``(wavefronts, ideal)`` for one warp instruction.

    ``addrs[i]`` is lane ``i``'s byte address (``None`` for an inactive lane) and every lane
    accesses ``width`` bytes.
    """
    group = 32 if width <= 4 else max(1, 128 // width)
    total, ideal = 0, 0
    for start in range(0, len(addrs), group):
        words_per_bank: dict[int, set[int]] = {}
        active = False
        for addr in addrs[start:start + group]:
            if addr is None:
                continue
            active = True
            first, last = addr // 4, (addr + max(width, 1) - 1) // 4
            for word in range(first, last + 1):
                words_per_bank.setdefault(word % NUM_BANKS, set()).add(word)
        if active:
            total += max(len(w) for w in words_per_bank.values())
            ideal += 1
    return total, ideal


@dataclass
class BufferRef:
    """Where a shared-memory view lives inside its backing allocation."""

    alloc: Op | None
    alloc_shape: list[int]
    elem_bytes: int
    layout: SharedLayout | PaddedScratchLayout
    offsets: list[int]  # per alloc dim, offset of the view's origin
    view_shape: list[int]
    unknown_offsets: bool = False

    def addr_fn(self) -> AddrFn:
        lead = len(self.alloc_shape) - len(self.view_shape)

        def addr(coord) -> int:
            full = list(self.offsets)
            for i, c in enumerate(coord):
                full[lead + i] += c
            return self.layout.byte_offset(full, self.alloc_shape, self.elem_bytes)

        return addr


def _const_int(val: Value | None) -> int | None:
    if val is None or val.owner is None:
        return None
    if val.owner.is_("constant"):
        v = val.owner.attrs.get("value")
        return v if isinstance(v, int) else None
    return None


def resolve_buffer(val: Value) -> BufferRef | None:
    """Follow ``memdesc_subview`` and loop-carried values back to the ``local_alloc``."""
    chain: list[tuple[Op, MemDescType]] = []
    view_type = val.type if isinstance(val.type, MemDescType) else None
    cur: Value | None = val
    seen = 0
    while cur is not None and seen < 64:
        seen += 1
        if cur.kind == "block_arg" and cur.init is not None:
            cur = cur.init
            continue
        owner = cur.owner
        if owner is None:
            return None
        if owner.is_("memdesc_subview", "memdesc_index"):
            chain.append((owner, owner.operands[0].type))
            cur = owner.operands[0]
            continue
        if owner.is_("local_alloc"):
            atype = owner.result_type
            if not isinstance(atype, MemDescType) or atype.encoding is None:
                return None
            try:
                layout = from_encoding(atype.encoding)
            except UnsupportedLayout:
                return None
            if not isinstance(layout, SharedLayout):
                return None
            offsets = [0] * len(atype.shape)
            unknown = False
            for sub, src_type in chain:
                src_rank = len(src_type.shape) if isinstance(src_type, MemDescType) else 0
                lead = len(atype.shape) - src_rank
                for i, operand in enumerate(sub.operands[1:]):
                    off = _const_int(operand)
                    if off is None:
                        unknown = True
                        off = 0
                    if lead + i < len(offsets):
                        offsets[lead + i] += off
            vshape = view_type.shape if view_type is not None else atype.shape
            return BufferRef(alloc=owner, alloc_shape=list(atype.shape),
                             elem_bytes=atype.element_bytes, layout=layout, offsets=offsets,
                             view_shape=list(vshape), unknown_offsets=unknown)
        return None
    return None


def _pow2_floor(n: int) -> int:
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def simulate_register_access(layout: DistributedLayout, shape: list[int], addr: AddrFn,
                             elem_bytes: int, vec_elems: int) -> list[list[WarpInstr]]:
    """Simulate a register<->shared copy where each lane moves ``vec_elems`` per instruction.

    Returns one list of instructions per warp.
    """
    out: list[list[WarpInstr]] = []
    width = vec_elems * elem_bytes
    for w in range(layout.num_warps):
        per_lane = [layout.thread_elements(shape, w, l) for l in range(layout.threads_per_warp)]
        n_instr = max(len(e) for e in per_lane) // vec_elems
        instrs = []
        for j in range(n_instr):
            lanes, addrs = [], []
            for l, elems in enumerate(per_lane):
                if j * vec_elems >= len(elems):
                    addrs.append(None)
                    continue
                coord = elems[j * vec_elems]
                a = addr(coord)
                addrs.append(a)
                lanes.append(LaneAccess(l, a, bank_of(a), list(coord)))
            wf, ideal = wavefronts(addrs, width)
            instrs.append(WarpInstr(width, lanes, wf, ideal))
        out.append(instrs)
    return out


def simulate_ldmatrix(layout: DotOperandLayout, shape: list[int], addr: AddrFn,
                      shared_order: tuple[int, ...]) -> list[list[WarpInstr]]:
    """Simulate ``ldmatrix.x4`` loads of a 16-bit MMA v2 operand tile.

    Every 8x8 matrix is one phase: 8 lanes each supply the address of a 16-byte row. A row
    runs along the shared layout's contiguous dimension (``ldmatrix.trans`` handles operands
    whose K dimension is not the contiguous one), so a phase touches 8 rows x 16 bytes.
    """
    contig = shared_order[0]
    other = 1 - contig
    out: list[list[WarpInstr]] = []
    for w in range(layout.num_warps):
        coords = set()
        for l in range(layout.threads_per_warp):
            coords.update(layout.thread_elements(shape, w, l))
        matrices = sorted({(c[0] // 8, c[1] // 8) for c in coords})
        instrs: list[WarpInstr] = []
        for group in range(0, len(matrices), 4):
            lanes: list[LaneAccess] = []
            addrs: list[int | None] = []
            for mi, (b0, b1) in enumerate(matrices[group:group + 4]):
                base = [b0 * 8, b1 * 8]
                for row in range(8):
                    coord = list(base)
                    coord[other] += row
                    a = addr(tuple(coord))
                    addrs.append(a)
                    lanes.append(LaneAccess(mi * 8 + row, a, bank_of(a), coord))
            while len(addrs) < 32:
                addrs.append(None)
            wf, ideal = wavefronts(addrs, 16)
            instrs.append(WarpInstr(16, lanes, wf, ideal))
        out.append(instrs)
    return out


@dataclass
class SharedAccess:
    op: Op
    direction: str  # "load" or "store"
    mechanism: str
    width: int
    buffer: BufferRef
    warps: list[list[WarpInstr]]
    register_layout: dict
    estimated: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def instructions_per_warp(self) -> float:
        return sum(len(w) for w in self.warps) / max(1, len(self.warps))

    @property
    def wavefronts(self) -> int:
        return sum(i.wavefronts for w in self.warps for i in w)

    @property
    def ideal(self) -> int:
        return sum(i.ideal for w in self.warps for i in w)

    @property
    def degree(self) -> float:
        return self.wavefronts / self.ideal if self.ideal else 1.0

    def worst(self) -> WarpInstr | None:
        best = None
        for w in self.warps:
            for instr in w:
                if best is None or instr.degree > best.degree:
                    best = instr
        return best

    def to_json(self) -> dict:
        worst = self.worst()
        return {
            "op": self.op.id,
            "opName": self.op.name,
            "line": self.op.line_no,
            "loc": self.op.loc.to_json() if self.op.loc else None,
            "direction": self.direction,
            "mechanism": self.mechanism,
            "width": self.width,
            "buffer": self.buffer.alloc.id if self.buffer.alloc else None,
            "viewShape": self.buffer.view_shape,
            "offsets": self.buffer.offsets,
            "registerLayout": self.register_layout,
            "instructionsPerWarp": self.instructions_per_warp,
            "wavefronts": self.wavefronts,
            "ideal": self.ideal,
            "degree": round(self.degree, 3),
            "estimated": self.estimated,
            "notes": self.notes,
            "worst": worst.to_json() if worst else None,
            "warp0": [i.to_json() for i in self.warps[0][:16]] if self.warps else [],
        }


def _contig_dim_of(layout: SharedLayout | PaddedScratchLayout) -> int:
    return layout.order[0]


def _register_vec(reg_layout: DistributedLayout, shape: list[int], buf: BufferRef,
                  elem_bytes: int, max_bytes: int = 16) -> int:
    contig = _contig_dim_of(buf.layout)
    lead = len(buf.view_shape) - len(shape)
    dim = contig - lead if contig >= lead else contig
    vec = reg_layout.contiguous_elements(shape, dim)
    if isinstance(buf.layout, SharedLayout) and buf.layout.swizzled:
        vec = min(vec, buf.layout.vec)
    vec = min(vec, max(1, max_bytes // elem_bytes))
    return _pow2_floor(max(1, vec))


def _layout_of(val: Value, num_warps: int, tpw: int) -> DistributedLayout | None:
    t = val.type
    if not isinstance(t, TensorType) or t.encoding is None:
        return None
    try:
        layout = from_encoding(t.encoding, num_warps, tpw)
    except (UnsupportedLayout, KeyError):
        return None
    return layout if isinstance(layout, DistributedLayout) else None


def analyze_op(op: Op, num_warps: int, tpw: int = 32) -> SharedAccess | None:
    """Simulate the shared-memory traffic of a single TTGIR op (``None`` if it has none)."""
    if op.is_("local_load") and op.operands:
        buf = resolve_buffer(op.operands[0])
        rtype = op.result_type
        if buf is None or not isinstance(rtype, TensorType):
            return None
        reg = _layout_of(op.results[0], num_warps, tpw)
        if reg is None:
            return None
        addr = buf.addr_fn()
        if isinstance(reg, DotOperandLayout) and buf.elem_bytes == 2 and \
                isinstance(buf.layout, SharedLayout):
            k_dim = 1 if reg.op_idx == 0 else 0
            trans = buf.layout.order[0] != k_dim
            warps = simulate_ldmatrix(reg, rtype.shape, addr, buf.layout.order)
            mech = "ldmatrix.x4" + (".trans" if trans else "")
            return SharedAccess(op, "load", mech, 16, buf, warps, reg.describe())
        vec = _register_vec(reg, rtype.shape, buf, buf.elem_bytes)
        warps = simulate_register_access(reg, rtype.shape, addr, buf.elem_bytes, vec)
        return SharedAccess(op, "load", f"ld.shared ({vec * buf.elem_bytes}B)",
                            vec * buf.elem_bytes, buf, warps, reg.describe())

    if op.is_("local_alloc") and op.operands:
        buf = resolve_buffer(op.results[0])
        return _store_from_registers(op, op.operands[0], buf, num_warps, tpw, "st.shared")

    if op.is_("local_store") and len(op.operands) >= 2:
        buf = resolve_buffer(op.operands[1])
        return _store_from_registers(op, op.operands[0], buf, num_warps, tpw, "st.shared")

    if op.is_("async_copy_global_to_local") and len(op.operands) >= 2:
        buf = resolve_buffer(op.operands[1])
        return _store_from_registers(op, op.operands[0], buf, num_warps, tpw, "cp.async")

    if op.is_("convert_layout") and op.operands:
        return _convert_layout(op, num_warps, tpw)
    return None


def _store_from_registers(op: Op, src: Value, buf: BufferRef | None, num_warps: int, tpw: int,
                          mech: str) -> SharedAccess | None:
    if buf is None or not isinstance(src.type, TensorType):
        return None
    reg = _layout_of(src, num_warps, tpw)
    if reg is None:
        return None
    shape = src.type.shape
    vec = _register_vec(reg, shape, buf, buf.elem_bytes)
    warps = simulate_register_access(reg, shape, buf.addr_fn(), buf.elem_bytes, vec)
    width = vec * buf.elem_bytes
    return SharedAccess(op, "store", f"{mech} ({width}B)", width, buf, warps, reg.describe())


def _cta_tile(layout: DistributedLayout, shape: list[int]) -> list[int]:
    if isinstance(layout, (BlockedLayout, MmaV2Layout)):
        return [min(t, s) for t, s in zip(layout.cta_tile(), shape)]
    return list(shape)


def _order_of(layout: DistributedLayout) -> list[int] | None:
    if isinstance(layout, BlockedLayout):
        return layout.order
    if isinstance(layout, MmaV2Layout):
        return [1, 0]
    return None


def _convert_layout(op: Op, num_warps: int, tpw: int) -> SharedAccess | None:
    """Estimate the scratch-buffer traffic of a ``convert_layout`` that goes through shared.

    Mirrors Triton 3.x ``getScratchConfigForCvt``: the scratch tile is the larger of the two
    CTA tiles, stored in the destination order with the fastest dimension padded by
    ``max(inVec, outVec)`` elements to break bank conflicts.
    """
    src_val, dst_val = op.operands[0], op.results[0]
    src = _layout_of(src_val, num_warps, tpw)
    dst = _layout_of(dst_val, num_warps, tpw)
    if src is None or dst is None or isinstance(src, SliceLayout) or isinstance(dst, SliceLayout):
        return None
    shape = dst_val.type.shape
    if src.ownership(shape).threads == dst.ownership(shape).threads:
        return None
    out_ord = _order_of(dst)
    if out_ord is None or len(shape) != 2:
        return None
    rank = len(shape)
    rep = [max(a, b) for a, b in zip(_cta_tile(src, shape), _cta_tile(dst, shape))]
    rep = [min(r, s) for r, s in zip(rep, shape)]
    inner = rank - 1
    in_vec = 1 if out_ord[0] != inner else src.contiguous_elements(shape, out_ord[0])
    out_vec = 1 if out_ord[0] != inner else dst.contiguous_elements(shape, out_ord[0])
    padded = list(rep)
    pad_dim = out_ord[0] if isinstance(dst, BlockedLayout) else inner
    padded[pad_dim] += max(in_vec, out_vec)
    scratch = PaddedScratchLayout(rep_shape=rep, padded_shape=padded, order=out_ord)
    elem = dst_val.type.element_bytes
    buf = BufferRef(alloc=None, alloc_shape=list(shape), elem_bytes=elem, layout=scratch,
                    offsets=[0] * rank, view_shape=list(shape))
    addr = buf.addr_fn()
    store_vec = _pow2_floor(max(1, min(in_vec, 16 // elem)))
    load_vec = _pow2_floor(max(1, min(out_vec, 16 // elem)))
    st = simulate_register_access(src, shape, addr, elem, store_vec)
    ld = simulate_register_access(dst, shape, addr, elem, load_vec)
    combined = [a + b for a, b in zip(st, ld)]
    access = SharedAccess(op, "store+load", f"scratch st {store_vec * elem}B / ld "
                          f"{load_vec * elem}B", load_vec * elem, buf, combined,
                          {"src": src.describe(), "dst": dst.describe()}, estimated=True)
    access.notes.append(f"scratch tile {rep} padded to {padded} (order {out_ord})")
    st_wf = sum(i.wavefronts for w in st for i in w)
    st_ideal = sum(i.ideal for w in st for i in w)
    ld_wf = sum(i.wavefronts for w in ld for i in w)
    ld_ideal = sum(i.ideal for w in ld for i in w)
    access.notes.append(f"store {st_wf}/{st_ideal} wavefronts, load {ld_wf}/{ld_ideal}")
    access.notes.append(f"scratch bytes {_prod(padded) * elem}")
    return access


def _prod(values) -> int:
    out = 1
    for v in values:
        out *= v
    return out


def analyze_function(func, num_warps: int, tpw: int = 32) -> list[SharedAccess]:
    results = []
    for op in func.ops:
        try:
            access = analyze_op(op, num_warps, tpw)
        except (UnsupportedLayout, KeyError, IndexError, ValueError):
            access = None
        if access is not None:
            results.append(access)
    return results
