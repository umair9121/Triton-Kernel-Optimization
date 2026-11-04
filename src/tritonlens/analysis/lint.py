"""Lint rules over an :class:`~tritonlens.analyze.Analysis`.

=====  ========================================================================
TL001  global access narrower than 16 B per lane, with the limiting factor
TL002  uncoalesced global access (32 B sectors per request vs. ideal)
TL003  shared-memory bank conflicts
TL004  ``convert_layout`` that round-trips through shared memory
TL005  wasted threads: layout replication or mostly-masked tiles
TL006  launch arguments that defeat Triton's alignment specialization
TL007  register spills or occupancy too low to hide memory latency
TL008  SM load imbalance: the grid does not divide evenly over the SMs
TL009  shared memory (pipeline stages) limits blocks per SM
=====  ========================================================================

TL007-TL009 depend on the target GPU (``Analysis.device``): the same kernel can be fine on a
108-SM A100 and leave a 20-SM laptop GPU half idle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..ir import Op, TensorType
from ..layouts import UnsupportedLayout

if TYPE_CHECKING:
    from ..analyze import Analysis

SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}

RULES = {
    "TL001": "Under-vectorized global access",
    "TL002": "Uncoalesced global access",
    "TL003": "Shared-memory bank conflict",
    "TL004": "Layout conversion through shared memory",
    "TL005": "Wasted threads",
    "TL006": "Launch argument defeats alignment specialization",
    "TL007": "Register pressure / low occupancy",
    "TL008": "SM load imbalance",
    "TL009": "Shared memory limits occupancy",
}


@dataclass
class Finding:
    rule: str
    severity: str
    message: str
    op: Op | None = None
    fix: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def title(self) -> str:
        return RULES.get(self.rule, self.rule)

    @property
    def location(self) -> str:
        if self.op is not None and self.op.loc is not None:
            return str(self.op.loc)
        if self.op is not None:
            return f"ttgir:{self.op.line_no}"
        return ""

    def to_json(self) -> dict:
        return {
            "rule": self.rule,
            "title": self.title,
            "severity": self.severity,
            "message": self.message,
            "fix": self.fix,
            "op": self.op.id if self.op is not None else None,
            "line": self.op.line_no if self.op is not None else None,
            "loc": self.op.loc.to_json() if self.op is not None and self.op.loc else None,
            "data": self.data,
        }


def _what(access) -> str:
    kind = {"load": "load", "store": "store", "async_copy_global_to_local": "async copy"}
    return kind.get(access.kind, access.kind)


def _ptx_note(access) -> str:
    if not access.ptx:
        return ""
    opcodes = sorted({p["opcode"] for p in access.ptx})
    return f" PTX confirms: {', '.join(opcodes)}."


def _mask_bound(access, an: Analysis) -> str | None:
    """Name the scalar a mask compares against, if it is a function argument."""
    mask = None
    op = access.op
    if op.is_("load") and len(op.operands) > 1:
        mask = op.operands[1]
    elif op.is_("store", "async_copy_global_to_local") and len(op.operands) > 2:
        mask = op.operands[2]
    seen, stack = set(), [mask] if mask is not None else []
    while stack:
        val = stack.pop()
        if val is None or id(val) in seen:
            continue
        seen.add(id(val))
        if val.kind == "arg":
            if "tt.divisibility" not in val.attrs:
                return val.name
            continue
        if val.owner is not None:
            stack.extend(val.owner.operands)
    return None


def _undivisible_stride(access, an: Analysis) -> str | None:
    for s in access.ptr_info.stride:
        if isinstance(s, tuple):
            arg = next((a for a in an.func.args if a.name == s[1]), None)
            if arg is not None and "tt.divisibility" not in arg.attrs:
                return arg.name
    return None


def feeding_args(val, include_pointers: bool = False) -> list:
    """Function arguments that ``val`` is computed from (through any chain of ops)."""
    seen, stack, out = set(), [val], []
    while stack:
        cur = stack.pop()
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))
        if cur.kind == "arg":
            if include_pointers or not getattr(cur.type, "pointee", None):
                out.append(cur)
            continue
        if cur.kind == "block_arg":
            stack.extend([cur.init, cur.yielded])
            continue
        if cur.owner is not None:
            stack.extend(cur.owner.operands)
    return out


def _unaligned_offset_arg(access) -> str | None:
    for arg in feeding_args(access.op.operands[0]):
        if "tt.divisibility" not in arg.attrs:
            return arg.name
    return None


def rule_tl001(an: Analysis) -> list[Finding]:
    out = []
    for acc in an.global_accesses:
        if acc.vec >= acc.potential_vec or acc.vec_bytes >= 16:
            continue
        binding = {lim.factor for lim in acc.binding_limits()}
        reasons, fixes = [], []
        base = acc.base_arg
        if "alignment" in binding:
            stride_arg = _undivisible_stride(acc, an)
            offset_arg = _unaligned_offset_arg(acc)
            if base is not None and "tt.divisibility" not in base.attrs:
                reasons.append(f"pointer {an.pretty(base.name)} is not known to be 16-byte "
                               "aligned (no tt.divisibility)")
                fixes.append("pass tensors whose data_ptr() is a multiple of 16 (avoid "
                             "offset views such as x[1:]) or call tl.multiple_of on the pointer")
            elif stride_arg is not None:
                reasons.append(f"stride {an.pretty(stride_arg)} is not divisible by 16, so "
                               "row starts are misaligned")
                fixes.append(f"pad the row pitch passed as {an.pretty(stride_arg)} to a "
                             "multiple of 16 elements")
            elif offset_arg is not None:
                reasons.append(f"the offset is computed from {an.pretty(offset_arg)}, which is "
                               "not divisible by 16, so each program's base address is only "
                               f"{acc.ptr_info.divisibility[acc.fast_dim]}-byte aligned")
                fixes.append(f"pad {an.pretty(offset_arg)} to a multiple of 16 elements (e.g. "
                             "allocate rows with a padded pitch)")
            else:
                reasons.append(f"offsets are only {acc.ptr_info.divisibility[acc.fast_dim]}-"
                               "byte aligned")
                fixes.append("make the block offset a multiple of 16 bytes")
        if "mask" in binding:
            bound = _mask_bound(acc, an)
            if bound is not None:
                reasons.append(f"the mask bound {an.pretty(bound)} is not divisible by 16, so "
                               "the mask may change inside a vector")
                fixes.append(f"pad {an.pretty(bound)} to a multiple of 16, or split the tail "
                             "into a separate masked pass")
            else:
                reasons.append("the mask is not constant over 16-byte runs")
        if "contiguity" in binding:
            reasons.append(f"offsets are contiguous only in runs of "
                           f"{acc.ptr_info.contiguity[acc.fast_dim]}")
        if "layout" in binding and not reasons:
            per_thread = acc.layout.contiguous_elements(acc.shape, acc.fast_dim)
            reasons.append(f"the layout gives each thread only {per_thread} contiguous "
                           "elements (block too small for the number of warps)")
            fixes.append("increase the block size along the contiguous dimension or lower "
                         "num_warps")
        if not reasons:
            continue
        # Narrow but fully coalesced accesses cost load/store issue slots, not DRAM bandwidth:
        # on a DRAM-bound kernel the difference measured below 2% (benchmarks/lint_fixes.py).
        coalesced = acc.efficiency >= 0.9
        severity = "info" if coalesced else "warning"
        msg = (f"{_what(acc).capitalize()} uses {acc.vec_bytes}-byte accesses per lane; "
               f"{acc.potential_vec * acc.elem_bytes}-byte vectors are possible. Limited by: "
               + "; ".join(reasons) + "." + _ptx_note(acc))
        if coalesced:
            msg += (f" Sectors are still fully used, so this costs "
                    f"{acc.potential_vec // max(acc.vec, 1)}x more memory instructions rather "
                    "than bandwidth; it matters when the kernel is issue-bound or L2-resident.")
        out.append(Finding("TL001", severity, msg, acc.op, "; ".join(dict.fromkeys(fixes)),
                           {"vecBytes": acc.vec_bytes,
                            "potentialBytes": acc.potential_vec * acc.elem_bytes,
                            "limits": sorted(binding)}))
    return out


def rule_tl002(an: Analysis) -> list[Finding]:
    out = []
    for acc in an.global_accesses:
        if acc.efficiency >= 0.5:
            continue
        stride = acc.ptr_info.stride[acc.fast_dim] if acc.fast_dim < acc.ptr_info.rank else None
        if isinstance(stride, tuple):
            why = (f"the layout's fastest dimension (dim {acc.fast_dim}) advances by the "
                   f"runtime stride {an.pretty(stride[1])} instead of 1")
        elif isinstance(stride, int) and stride != 1:
            why = f"neighbouring lanes are {stride} elements apart"
        else:
            why = "neighbouring lanes do not touch neighbouring addresses"
        msg = (f"{_what(acc).capitalize()} touches {acc.sectors_per_request:.1f} sectors "
               f"per warp request where {acc.ideal_sectors:.1f} would suffice "
               f"({acc.efficiency:.0%} efficiency): {why}.")
        severity = "error" if acc.efficiency < 0.2 else "warning"
        out.append(Finding("TL002", severity, msg, acc.op,
                           "make the contiguous memory dimension the fastest-varying index "
                           "(order[0]) of the tile, or gather through shared memory",
                           {"efficiency": acc.efficiency,
                            "sectorsPerRequest": acc.sectors_per_request}))
    return out


def rule_tl003(an: Analysis, threshold: float = 1.05) -> list[Finding]:
    out = []
    for acc in an.shared_accesses:
        if acc.op.is_("convert_layout") or acc.degree < threshold:
            continue
        worst = acc.worst()
        msg = (f"{acc.mechanism} has {acc.degree:.2f}x wavefronts vs conflict-free "
               f"({acc.wavefronts} vs {acc.ideal} per CTA); worst instruction is "
               f"{worst.degree:.0f}-way." if worst else "")
        out.append(Finding("TL003", "warning", msg, acc.op,
                           "increase swizzling (maxPhase/perPhase) or pad the shared tile",
                           {"degree": acc.degree, "mechanism": acc.mechanism}))
    return out


def rule_tl004(an: Analysis) -> list[Finding]:
    out = []
    by_op = {a.op.id: a for a in an.shared_accesses if a.op.is_("convert_layout")}
    for op in an.func.find("convert_layout"):
        acc = by_op.get(op.id)
        if acc is None:
            continue
        src = op.operands[0].type.encoding.name if op.operands else "?"
        dst = op.result_type.encoding.name if isinstance(op.result_type, TensorType) else "?"
        scratch = next((n for n in acc.notes if n.startswith("scratch bytes")), "")
        in_loop = op.depth > 0
        msg = (f"convert_layout {src} -> {dst} stages the tile through shared memory "
               f"({scratch.replace('scratch bytes ', '')} B scratch, estimated "
               f"{acc.degree:.2f}x bank-conflict factor){' inside a loop' if in_loop else ''}.")
        severity = "warning" if in_loop or acc.degree >= 2 else "info"
        out.append(Finding("TL004", severity, msg, op,
                           "align the producer and consumer layouts (e.g. store in the layout "
                           "the data is computed in) to avoid the round trip",
                           {"degree": acc.degree, "notes": acc.notes}))
    return out


def rule_tl005(an: Analysis) -> list[Finding]:
    out = []
    mod = an.module
    total_threads = mod.num_warps * mod.threads_per_warp
    for acc in an.global_accesses:
        numel = 1
        for s in acc.shape:
            numel *= s
        if numel < total_threads:
            msg = (f"{_what(acc).capitalize()} of {numel} elements runs on {total_threads} "
                   f"threads: {total_threads - numel} threads hold redundant copies.")
            out.append(Finding("TL005", "warning", msg, acc.op,
                               "lower num_warps or enlarge the block", {"numel": numel}))
    for op in an.func.find("cmpi"):
        if not isinstance(op.result_type, TensorType) or len(op.operands) < 2:
            continue
        lhs, rhs = an.axis[op.operands[0]], op.operands[1]
        rhs_owner = rhs.owner
        if rhs_owner is None or not rhs_owner.is_("splat"):
            continue
        bound = rhs_owner.operands[0] if rhs_owner.operands else None
        if bound is None or bound.kind != "arg" or bound.name not in an.arg_values:
            continue
        shape = op.result_type.shape
        dim = next((d for d, c in enumerate(lhs.contiguity) if c > 1), None)
        if dim is None or lhs.origin[dim] is not None:
            continue
        value, block = an.arg_values[bound.name], shape[dim]
        if value <= 0 or value >= block:
            continue
        util = value / block
        if util < 0.8:
            msg = (f"Mask against {an.pretty(bound.name)}={value} keeps only {util:.0%} of the "
                   f"{block}-wide block active in every program.")
            out.append(Finding("TL005", "info", msg, op,
                               "use a block closer to the problem size or process several "
                               "rows per program", {"utilization": util}))
    return out


def rule_tl006(an: Analysis) -> list[Finding]:
    from .runtime_check import check_args

    if an.launch is None:
        return []
    # Only integers that reach an address or a mask matter for vectorization.
    used: set[str] = set()
    for acc in an.global_accesses:
        for val in acc.op.operands:
            used.update(an.arg_names.get(a.name, a.name) for a in feeding_args(val))
    findings = []
    for f in check_args(an.launch.args, an.launch.name):
        arg = f.data.get("arg")
        if "value" in f.data and an.arg_names and arg not in used:
            continue
        findings.append(f)
    return findings


def _anchor(an: Analysis) -> Op | None:
    """Op to attach whole-kernel findings to: the main dot, else the largest store."""
    from .tiling import _main_store

    dots = an.func.find("dot")
    if dots:
        return dots[0]
    return _main_store(an.func) or next(
        (op for op in an.func.ops if op.is_("store", "load") and op.loc is not None), None)


def _gpu(an: Analysis) -> str:
    dev = an.perf.device
    return f"{dev.name} ({dev.sm_count} SMs, sm_{dev.arch})"


def rule_tl007(an: Analysis) -> list[Finding]:
    perf = an.perf
    occ = perf.occupancy
    if occ is None:
        return []
    out = []
    spills = int(an.metadata.get("n_spills") or 0)
    if spills:
        out.append(Finding(
            "TL007", "warning",
            f"Registers spill: {spills} bytes of local memory per thread at "
            f"{occ.regs_per_thread} registers/thread. Spills go through L1/L2 and, on a 2 MB "
            "L2, often to DRAM.", _anchor(an),
            "reduce the tile size or num_stages, or raise num_warps so each thread holds "
            "fewer accumulator elements", {"spills": spills}))
    pct = occ.occupancy
    dev = perf.device
    limiter = (f"limited by {occ.limiter}: {occ.blocks_per_sm} block(s) of {occ.num_warps} "
               f"warps = {occ.active_warps}/{occ.max_warps_per_sm} warps per SM")
    nxt = perf.regs_needed_for_next_block
    hint = (f"; at <= {nxt} registers/thread (now {occ.regs_per_thread}) one more block fits"
            if nxt else "")
    if not perf.has_dot and pct < 0.25:
        out.append(Finding(
            "TL007", "warning",
            f"Occupancy {pct:.0%} on {_gpu(an)} ({limiter}{hint}). Memory-bound kernels need "
            "many warps in flight to hide DRAM latency.", _anchor(an),
            "lower the per-thread work (smaller BLOCK or more warps) or reduce shared memory",
            {"occupancy": pct, "limiter": occ.limiter, "regsForNextBlock": nxt}))
    elif perf.has_dot and occ.blocks_per_sm == 1 and occ.active_warps <= 8:
        out.append(Finding(
            "TL007", "info",
            f"Only {occ.active_warps} warps per SM on {_gpu(an)} ({limiter}{hint}). Large "
            "matmul tiles rely on instruction-level parallelism; with one block per SM there "
            f"is no other block to overlap its prologue/epilogue, and on {dev.sm_count} SMs "
            "the grid also becomes coarse (see TL008).", _anchor(an),
            "try a smaller tile (e.g. 64x64 or 64x128) that fits 2+ blocks per SM",
            {"occupancy": pct, "limiter": occ.limiter, "regsForNextBlock": nxt}))
    return out


def rule_tl008(an: Analysis) -> list[Finding]:
    perf = an.perf
    w = perf.waves
    if w is None:
        return []
    sms = w.sm_count
    if w.total_blocks < sms:
        idle = sms - w.total_blocks
        sev = "warning" if w.total_blocks <= sms // 2 else "info"
        return [Finding(
            "TL008", sev,
            f"The grid has {w.total_blocks} programs for {sms} SMs on {_gpu(an)}: {idle} SMs "
            f"stay idle ({idle / sms:.0%} of the GPU).", _anchor(an),
            "use smaller blocks so the grid has at least one program per SM (ideally several), "
            "or split the reduction dimension (split-K)",
            {"programs": w.total_blocks, "smCount": sms, "balance": w.sm_balance})]
    balance = w.sm_balance
    # Only work-heavy programs make the imbalance visible; memory-bound kernels with many
    # small programs are balanced by the hardware scheduler.
    if not perf.has_dot or balance >= 0.85:
        return []
    per_sm = -(-w.total_blocks // sms)
    msg = (f"{w.total_blocks} equal-work programs on {sms} SMs: some SMs run {per_sm} programs "
           f"while others run {per_sm - 1}, so the kernel takes as long as "
           f"{per_sm * sms} programs would ({balance:.0%} SM utilization). Measured on an "
           "RTX 3050 Ti: going from 20 to 21 matmul tiles doubles run time "
           "(benchmarks/waves.py).")
    better = []
    for factor, how in ((2, "halve BLOCK_M or BLOCK_N"), (4, "halve both")):
        t = w.total_blocks * factor
        b = t / (-(-t // sms) * sms)
        if b > balance + 0.05:
            better.append(f"{how} -> {t} programs, {b:.0%} utilization")
    return [Finding("TL008", "warning" if balance < 0.7 else "info", msg, _anchor(an),
                    "; ".join(better) or f"make the number of programs a multiple of {sms}",
                    {"programs": w.total_blocks, "smCount": sms, "balance": balance})]


def rule_tl009(an: Analysis) -> list[Finding]:
    from .occupancy import occupancy
    from .perf import pipeline_buffer_bytes

    perf = an.perf
    occ = perf.occupancy
    stages = int(an.metadata.get("num_stages") or 0)
    # Past 2 blocks/SM, trading stages for occupancy measured within noise
    # (benchmarks/device_rules.py); 1 -> 2 blocks gave 9%.
    if occ is None or occ.limiter != "shared memory" or stages < 3 or not occ.shared_bytes \
            or occ.blocks_per_sm > 2:
        return []
    per_stage = pipeline_buffer_bytes(an.func, stages)
    if not per_stage:
        return []
    # The allocator overlaps epilogue scratch with the (dead) pipeline buffers, so the total is
    # max(pipeline, scratch): anything above the pipeline buffers is scratch that stays.
    other = occ.shared_bytes if occ.shared_bytes > per_stage * (stages - 1) else 0
    options = []
    best = occ.blocks_per_sm
    for s in range(stages - 1, 1, -1):
        smem = max(other, per_stage * (s - 1))
        o = occupancy(perf.device, occ.num_warps, occ.regs_per_thread, smem)
        if o.blocks_per_sm > best:
            options.append((s, smem, o))
            best = o.blocks_per_sm
    if not options:
        return []
    lim = perf.device.limits
    alts = ", ".join(f"num_stages={s} (~{smem // 1024} KiB): {o.blocks_per_sm} blocks "
                     f"({o.occupancy:.0%})" for s, smem, o in options)
    return [Finding(
        "TL009", "info",
        f"num_stages={stages} buffers {per_stage * (stages - 1) // 1024} KiB of operand tiles "
        f"({occ.shared_bytes // 1024} KiB shared memory per block); {_gpu(an)} has "
        f"{lim.shared_per_sm // 1024} KiB per SM, so only {occ.blocks_per_sm} block(s) fit "
        f"({occ.occupancy:.0%} occupancy). Fewer stages: {alts}. Configs tuned on "
        "164-228 KiB parts often overshoot here.",
        _anchor(an), "benchmark the listed num_stages values; deeper pipelines only pay off if "
        "they do not cost occupancy",
        {"stages": stages, "blocksPerSm": occ.blocks_per_sm,
         "options": [{"stages": s, "shared": smem, "blocksPerSm": o.blocks_per_sm}
                     for s, smem, o in options]})]


ALL_RULES = [rule_tl001, rule_tl002, rule_tl003, rule_tl004, rule_tl005, rule_tl006,
             rule_tl007, rule_tl008, rule_tl009]


def _dedupe(findings: list[Finding]) -> list[Finding]:
    """Merge identical findings at the same source location.

    Software pipelining clones loads (prologue plus loop body), which would otherwise report
    the same problem several times for one line of Python.
    """
    merged: dict[tuple, Finding] = {}
    for f in findings:
        key = (f.rule, f.severity, f.message, f.location)
        if key in merged and f.location:
            merged[key].data["occurrences"] = merged[key].data.get("occurrences", 1) + 1
        else:
            merged.setdefault(key if f.location else (id(f),), f)
    out = list(merged.values())
    for f in out:
        if f.data.get("occurrences", 1) > 1:
            f.message += f" ({f.data['occurrences']} IR ops: the pipeliner duplicated this access.)"
    return out


def run_lint(an: Analysis) -> list[Finding]:
    findings: list[Finding] = []
    for rule in ALL_RULES:
        try:
            findings.extend(rule(an))
        except (UnsupportedLayout, KeyError, IndexError, AttributeError):
            continue
    findings = _dedupe(findings)
    findings.sort(key=lambda f: (SEVERITY_ORDER.get(f.severity, 9),
                                 f.op.line_no if f.op else 0, f.rule))
    return findings
