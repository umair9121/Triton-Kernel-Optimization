"""Whole-kernel performance model: occupancy, waves and a roofline estimate.

Work and traffic are counted from the IR for one program and multiplied by the grid:

* FLOPs: ``2*M*N*K`` per ``tt.dot``, times the trip count of every enclosing ``scf.for``
  (loop bounds are evaluated with the captured scalar arguments);
* requested bytes: tile size of every global load/store/async copy, times trip counts. The
  software pipeliner clones loads into a prologue at the same source location; those clones
  are skipped when the same location also appears inside the loop;
* DRAM bytes: requested bytes per tensor divided by sector efficiency, capped at the size of
  the tensor (perfect L2 reuse) - an optimistic lower bound, so the roofline is an upper bound
  on attainable performance.

Elementwise FLOPs are ignored; kernels without ``tt.dot`` are treated as memory-bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..device import DeviceSpec
from ..ir import Op, TensorType
from . import occupancy as occ_mod
from .tiling import Evaluator, _Unknown

if TYPE_CHECKING:
    from ..analyze import Analysis


@dataclass
class Roofline:
    flops: float
    requested_bytes: float
    dram_bytes: float
    dot_dtype: str | None
    peak_tflops: float | None
    bandwidth_gbps: float | None
    trips_known: bool
    per_tensor: dict[str, float] = field(default_factory=dict)
    measured_ms: float | None = None
    wave_efficiency: float = 1.0
    l2_bytes: int = 0

    @property
    def l2_resident(self) -> bool:
        """Working set fits in L2: back-to-back replays never reach DRAM."""
        return bool(self.l2_bytes) and self.dram_bytes <= self.l2_bytes

    @property
    def intensity(self) -> float:
        return self.flops / self.dram_bytes if self.dram_bytes else 0.0

    @property
    def ridge(self) -> float | None:
        if not (self.peak_tflops and self.bandwidth_gbps):
            return None
        return self.peak_tflops * 1e12 / (self.bandwidth_gbps * 1e9)

    @property
    def memory_ms(self) -> float | None:
        return self.dram_bytes / (self.bandwidth_gbps * 1e9) * 1e3 if self.bandwidth_gbps else None

    @property
    def compute_ms(self) -> float | None:
        if not self.flops:
            return 0.0
        if not self.peak_tflops:
            return None
        # A partially filled last wave leaves SMs idle while the kernel is still running.
        return self.flops / (self.peak_tflops * 1e12) * 1e3 / self.wave_efficiency

    @property
    def bound(self) -> str:
        mem, comp = self.memory_ms or 0.0, self.compute_ms or 0.0
        return "compute" if comp > mem else "memory"

    @property
    def bound_ms(self) -> float | None:
        parts = [t for t in (self.memory_ms, self.compute_ms) if t is not None]
        return max(parts) if parts else None

    @property
    def attained(self) -> float | None:
        """Fraction of the roofline bound reached by the measured time."""
        if not self.measured_ms or not self.bound_ms:
            return None
        return self.bound_ms / self.measured_ms

    def to_json(self) -> dict:
        ms = self.measured_ms
        return {
            "flops": self.flops,
            "requestedBytes": self.requested_bytes,
            "dramBytes": self.dram_bytes,
            "perTensor": self.per_tensor,
            "dotDtype": self.dot_dtype,
            "peakTflops": self.peak_tflops,
            "bandwidthGbps": self.bandwidth_gbps,
            "intensity": self.intensity,
            "ridge": self.ridge,
            "memoryMs": self.memory_ms,
            "computeMs": self.compute_ms,
            "boundMs": self.bound_ms,
            "bound": self.bound,
            "tripsKnown": self.trips_known,
            "l2Resident": self.l2_resident,
            "l2Bytes": self.l2_bytes,
            "waveEfficiency": self.wave_efficiency,
            "measuredMs": ms,
            "achievedTflops": self.flops / (ms * 1e-3) / 1e12 if ms else None,
            "achievedGbps": self.dram_bytes / (ms * 1e-3) / 1e9 if ms else None,
            "attained": self.attained,
        }


@dataclass
class Performance:
    device: DeviceSpec
    occupancy: occ_mod.Occupancy | None
    waves: occ_mod.Waves | None
    roofline: Roofline | None
    regs_needed_for_next_block: int | None = None
    has_dot: bool = False
    budget: object | None = None
    advice: list = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "device": self.device.to_json(),
            "occupancy": self.occupancy.to_json() if self.occupancy else None,
            "waves": self.waves.to_json() if self.waves else None,
            "roofline": self.roofline.to_json() if self.roofline else None,
            "regsForNextBlock": self.regs_needed_for_next_block,
            "hasDot": self.has_dot,
            "budget": self.budget.to_json() if self.budget else None,
            "advice": [s.to_json() for s in self.advice],
        }


def _enclosing_loops(op: Op) -> list[Op]:
    loops, cur = [], op.parent
    while cur is not None:
        if cur.is_("for"):
            loops.append(cur)
        cur = cur.parent
    return loops


def trip_count(loop: Op, ev: Evaluator) -> int:
    lb, ub, step = (ev.eval(v, ()) for v in loop.operands[:3])
    if step <= 0:
        raise _Unknown
    return max(0, -(-(ub - lb) // step))


def _numel(t) -> int:
    return math.prod(t.shape) if isinstance(t, TensorType) else 1


def _dot_dtype(op: Op) -> str:
    t = op.operands[0].type if op.operands else None
    name = getattr(t, "element_name", "") or ""
    if name in ("f16", "bf16"):
        return "fp16"
    if "f8" in name:
        return "fp8"
    if name == "f32" and str(op.attrs.get("inputPrecision", "")).startswith("tf32"):
        return "tf32"
    return "fp32"


def _peak(device: DeviceSpec, dtype: str | None) -> float | None:
    if dtype is None:
        return None
    if dtype in ("fp16", "fp8"):
        return device.peak_tflops("fp16")
    if dtype == "tf32":
        # Ampere/Ada consumer parts run TF32 tensor cores at half the FP16-with-FP32-acc rate.
        fp16 = device.peak_tflops("fp16")
        return fp16 / 2 if fp16 else None
    return device.peak_tflops("fp32")


def roofline(an: Analysis, device: DeviceSpec, wave_efficiency: float = 1.0) -> Roofline | None:
    grid = an.launch.grid if an.launch else None
    if not grid:
        return None
    programs = math.prod(grid)
    ev = Evaluator(an.arg_values, (0, 0, 0), tuple((list(grid) + [1, 1, 1])[:3]))
    trips_known = True

    def trips(op: Op) -> int:
        nonlocal trips_known
        n = 1
        for loop in _enclosing_loops(op):
            try:
                n *= trip_count(loop, ev)
            except (_Unknown, KeyError, IndexError, TypeError, ValueError):
                trips_known = False
        return n

    flops, dtypes = 0.0, []
    for op in an.func.find("dot"):
        a, out = op.operands[0].type, op.result_type
        if not (isinstance(a, TensorType) and isinstance(out, TensorType)):
            continue
        m, n = out.shape[-2:]
        k = a.shape[-1]
        flops += 2 * m * n * k * trips(op) * math.prod(out.shape[:-2] or [1]) * programs
        dtypes.append(_dot_dtype(op))

    in_loop_locs = {str(acc.op.loc) for acc in an.global_accesses
                    if acc.op.loc is not None and _enclosing_loops(acc.op)}
    requested: dict[str, float] = {}
    sectors: dict[str, float] = {}
    for acc in an.global_accesses:
        if acc.op.loc is not None and not _enclosing_loops(acc.op) and \
                str(acc.op.loc) in in_loop_locs:
            continue  # pipeliner prologue clone of an in-loop access
        nbytes = _numel(acc.op.operands[0].type) * acc.elem_bytes * trips(acc.op) * programs
        key = an.arg_names.get(acc.base_arg.name, acc.base_arg.name) if acc.base_arg else "?"
        requested[key] = requested.get(key, 0.0) + nbytes
        sectors[key] = sectors.get(key, 0.0) + nbytes / max(acc.efficiency, 1e-3)
    sizes = {}
    if an.launch is not None:
        for arg in an.launch.args:
            if arg.kind == "tensor" and arg.shape is not None and arg.element_size:
                sizes[arg.name] = math.prod(arg.shape) * arg.element_size
    dram = {k: min(v, sizes[k]) if k in sizes else v for k, v in sectors.items()}
    dtype = max(set(dtypes), key=dtypes.count) if dtypes else None
    return Roofline(
        flops=flops, requested_bytes=sum(requested.values()), dram_bytes=sum(dram.values()),
        dot_dtype=dtype, peak_tflops=_peak(device, dtype), bandwidth_gbps=device.bandwidth_gbps,
        trips_known=trips_known, per_tensor=dram,
        measured_ms=an.metadata.get("time_ms"), wave_efficiency=wave_efficiency,
        l2_bytes=device.l2_bytes,
    )


def pipeline_buffer_bytes(func, num_stages: int) -> int:
    """Bytes of shared memory per pipeline stage (0 if the kernel is not software-pipelined).

    The pipeliner allocates ``num_stages - 1`` copies of every operand tile as one
    ``local_alloc`` with a leading buffer dimension.
    """
    from ..ir.types import MemDescType

    per_stage = 0
    for op in func.find("local_alloc"):
        t = op.result_type
        if op.operands or not isinstance(t, MemDescType) or not t.mutable or len(t.shape) < 3:
            continue
        if t.shape[0] == num_stages - 1:
            per_stage += math.prod(t.shape[1:]) * t.element_bytes
    return per_stage


def regs_for_next_block(device: DeviceSpec, occ: occ_mod.Occupancy) -> int | None:
    """Largest registers/thread that would fit one more block per SM (if registers limit)."""
    if occ.limiter != "registers":
        return None
    for regs in range(occ.regs_per_thread - 1, 15, -1):
        nxt = occ_mod.occupancy(device, occ.num_warps, regs, occ.shared_bytes)
        if nxt.blocks_by_regs > occ.blocks_by_regs:
            return regs
    return None


def analyze_performance(an: Analysis, device: DeviceSpec | None = None) -> Performance:
    from ..device import for_target
    from .advise import advise
    from .budget import analyze_budget

    device = device or an.device or for_target(an.module.target)
    occ = occ_mod.from_metadata(device, an.metadata)
    grid = an.launch.grid if an.launch else None
    waves = occ_mod.waves(device, occ, grid) if occ is not None else None
    has_dot = bool(an.func.find("dot"))
    eff = waves.sm_balance if waves is not None and has_dot else 1.0
    try:
        roof = roofline(an, device, eff)
    except (_Unknown, KeyError, IndexError, TypeError, AttributeError):
        roof = None
    try:
        budget = analyze_budget(an, device)
    except (KeyError, TypeError, AttributeError):
        budget = None
    try:
        advice = advise(an, device, occ)
    except (KeyError, TypeError, AttributeError, ZeroDivisionError):
        advice = []
    return Performance(device, occ, waves, roof,
                       regs_for_next_block(device, occ) if occ else None, has_dot,
                       budget, advice)
