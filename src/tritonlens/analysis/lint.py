"""Lint rules over an :class:`~tritonlens.analyze.Analysis`.

=====  ========================================================================
TL001  global access narrower than 16 B per lane, with the limiting factor
TL002  uncoalesced global access (32 B sectors per request vs. ideal)
TL003  shared-memory bank conflicts
TL004  ``convert_layout`` that round-trips through shared memory
TL005  wasted threads: layout replication or mostly-masked tiles
TL006  launch arguments that defeat Triton's alignment specialization
=====  ========================================================================
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
        severity = "warning" if "layout" not in binding or len(binding) > 1 else "info"
        msg = (f"{_what(acc).capitalize()} uses {acc.vec_bytes}-byte accesses per lane; "
               f"{acc.potential_vec * acc.elem_bytes}-byte vectors are possible. Limited by: "
               + "; ".join(reasons) + "." + _ptx_note(acc))
        out.append(Finding("TL001", severity, msg, acc.op, " ".join(dict.fromkeys(fixes)),
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


ALL_RULES = [rule_tl001, rule_tl002, rule_tl003, rule_tl004, rule_tl005]


def run_lint(an: Analysis) -> list[Finding]:
    findings: list[Finding] = []
    for rule in ALL_RULES:
        try:
            findings.extend(rule(an))
        except (UnsupportedLayout, KeyError, IndexError, AttributeError):
            continue
    findings.sort(key=lambda f: (SEVERITY_ORDER.get(f.severity, 9),
                                 f.op.line_no if f.op else 0, f.rule))
    return findings
