"""Propose matmul configs that fit a small GPU: shared memory, SM balance, L2, occupancy.

Tutorials copy A100/H100 numbers (128-256 tiles, 4-5 pipeline stages, GROUP_M=8). On a 20-SM
100 KiB part those configs either fail to launch or leave SMs idle. This search stays on the
host: it uses the occupancy calculator and the shared-memory lower bound, never compiles.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from ..autotune import matmul_smem, sm_balance
from ..device import DeviceSpec
from . import occupancy as occ_mod
from .budget import block_size, fitting_group_m, group_a_stripe, launch_ints

if TYPE_CHECKING:
    from ..analyze import Analysis

_TILES = (64, 128)
_K = (32, 64)
_WARPS = (4, 8)
_STAGES = (2, 3, 4)


@dataclass
class Suggestion:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    group_m: int
    programs: int
    sm_balance: float
    smem: int
    l2_group: int | None
    l2_fits: bool | None
    est_blocks_per_sm: int
    est_occupancy: float
    current: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return (f"{self.block_m}x{self.block_n}x{self.block_k} w{self.num_warps} "
                f"s{self.num_stages} G{self.group_m}")

    def to_json(self) -> dict:
        data = asdict(self)
        data["label"] = self.label
        data["smBalance"] = data.pop("sm_balance")
        data["estBlocksPerSm"] = data.pop("est_blocks_per_sm")
        data["estOccupancy"] = data.pop("est_occupancy")
        data["l2Group"] = data.pop("l2_group")
        data["l2Fits"] = data.pop("l2_fits")
        data["numWarps"] = data.pop("num_warps")
        data["numStages"] = data.pop("num_stages")
        data["groupM"] = data.pop("group_m")
        data["blockM"] = data.pop("block_m")
        data["blockN"] = data.pop("block_n")
        data["blockK"] = data.pop("block_k")
        return data


class _Cfg:
    """Duck-typed ``triton.Config`` for :func:`matmul_smem`."""

    def __init__(self, bm, bn, bk, stages, warps, group):
        self.kwargs = {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": group}
        self.num_stages = stages
        self.num_warps = warps


def _est_regs(bm: int, bn: int, warps: int, cur: occ_mod.Occupancy | None,
              cur_bm: int | None, cur_bn: int | None, cur_warps: int | None) -> int:
    if not (cur and cur.regs_per_thread and cur_bm and cur_bn and cur_warps):
        return 0
    old = cur_bm * cur_bn / cur_warps
    new = bm * bn / warps
    if old <= 0:
        return 0
    return max(32, int(cur.regs_per_thread * new / old))


def _score(s: Suggestion) -> float:
    occ = min(s.est_blocks_per_sm, 4) / 4
    l2 = 1.0 if s.l2_fits else (0.4 if s.l2_fits is None else 0.0)
    tile = math.log2(max(s.block_m * s.block_n, 1)) / 14  # 128x128 ~ 1
    return s.sm_balance * 0.45 + occ * 0.30 + l2 * 0.15 + tile * 0.10


def advise(an: Analysis, device: DeviceSpec | None = None,
           occ: occ_mod.Occupancy | None = None) -> list[Suggestion]:
    """Ranked configs for a captured matmul-shaped kernel, current first if known."""
    launch = an.launch
    if launch is None:
        return []
    device = device or an.device
    if device is None:
        return []
    consts = launch_ints(launch.args)
    m, n, k = consts.get("M"), consts.get("N"), consts.get("K")
    bm = block_size(consts, "M")
    bn = block_size(consts, "N")
    bk = block_size(consts, "K")
    if None in (m, n, bm, bn):
        return []
    named = {a.name: a for a in launch.args}
    stages = int(an.metadata.get("num_stages") or 3)
    warps = int(an.metadata.get("num_warps") or an.module.num_warps)
    gm = int(consts.get("GROUP_M") or 1)
    occ = occ or occ_mod.from_metadata(device, an.metadata)

    def build(b_m, b_n, b_k, w, s, g) -> Suggestion | None:
        programs = math.ceil(m / b_m) * math.ceil(n / b_n)
        smem = matmul_smem(_Cfg(b_m, b_n, b_k, s, w, g), named)
        if smem is None:
            return None
        if smem > device.limits.max_shared_per_block:
            return None
        regs = _est_regs(b_m, b_n, w, occ, bm, bn, warps)
        o = occ_mod.occupancy(device, w, regs, smem)
        if not o.fits or o.blocks_per_sm == 0:
            return None
        stripe = group_a_stripe({"GROUP_M": g, "BLOCK_M": b_m, "K": k or 0},
                                next((a.element_size for a in launch.args
                                      if a.kind == "tensor" and a.element_size), 2) or 2)
        l2_ok = None if stripe is None or not device.l2_bytes else stripe <= device.l2_bytes
        return Suggestion(
            b_m, b_n, b_k, w, s, g, programs, sm_balance(programs, device.sm_count),
            smem, stripe, l2_ok, o.blocks_per_sm, o.occupancy)

    current = None
    if bm and bn:
        current = build(bm, bn or 64, bk or 32, warps, stages, gm)
        if current is not None:
            current.current = True
            current.reasons = ["current"]

    nbytes = next((a.element_size for a in launch.args
                   if a.kind == "tensor" and a.element_size), 2) or 2
    fit_gm = fitting_group_m(consts, nbytes, device.l2_bytes) or gm
    groups = sorted({g for g in (1, 2, 4, 8, gm, fit_gm) if g > 0})
    seen: dict[tuple, Suggestion] = {}
    for t_m in _TILES:
        if t_m > m and t_m > 64:
            continue
        for t_n in _TILES:
            if t_n > n and t_n > 64:
                continue
            for t_k in _K:
                for w in _WARPS:
                    for s in _STAGES:
                        best: Suggestion | None = None
                        for g in groups:
                            cand = build(t_m, t_n, t_k, w, s, g)
                            if cand is None or cand.sm_balance < 0.7:
                                continue
                            # One GROUP_M per tile: prefer a stripe that fits L2, then larger G.
                            key = (t_m, t_n, t_k, w, s)
                            rank = (cand.l2_fits is not False, cand.group_m)
                            if best is None or rank > (best.l2_fits is not False, best.group_m):
                                best = cand
                                seen[key] = cand
    ranked = sorted(seen.values(), key=_score, reverse=True)
    reasons_for = _reasons(current, device)
    top: list[Suggestion] = []
    seen_mn: set[tuple[int, int]] = set()
    if current is not None:
        top.append(current)
        seen_mn.add((current.block_m, current.block_n))
    for cand in ranked:
        if current is not None and cand.label == current.label:
            continue
        mn = (cand.block_m, cand.block_n)
        if mn in seen_mn:
            continue
        cand.reasons = reasons_for(cand)
        if not cand.reasons and current is not None:
            continue
        top.append(cand)
        seen_mn.add(mn)
        if len(top) >= 5:
            break
    return top


def _reasons(current: Suggestion | None, device: DeviceSpec):
    def reasons(cand: Suggestion) -> list[str]:
        out = []
        if current is None:
            return ["fits this GPU"]
        if cand.sm_balance > current.sm_balance + 0.04:
            out.append(f"SM utilization {cand.sm_balance:.0%} vs {current.sm_balance:.0%}")
        if cand.est_blocks_per_sm > current.est_blocks_per_sm:
            out.append(f"{cand.est_blocks_per_sm} blocks/SM vs {current.est_blocks_per_sm}")
        if cand.l2_fits and current.l2_fits is False:
            out.append(f"A-stripe fits the {device.l2_bytes // 1024} KiB L2")
        if cand.smem < current.smem * 0.6 and cand.est_blocks_per_sm >= current.est_blocks_per_sm:
            out.append(f"shared memory {cand.smem // 1024} KiB vs {current.smem // 1024} KiB")
        if not out:
            trade = []
            if cand.sm_balance + 0.02 < current.sm_balance:
                trade.append(f"SM {cand.sm_balance:.0%} vs {current.sm_balance:.0%}")
            if cand.est_blocks_per_sm != current.est_blocks_per_sm:
                trade.append(f"{cand.est_blocks_per_sm} vs {current.est_blocks_per_sm} blocks/SM")
            if trade:
                out.append("tradeoff: " + ", ".join(trade))
            elif cand.sm_balance >= 0.95:
                out.append("also fills the GPU")
        return out
    return reasons


def format_advice(name: str, suggestions: list[Suggestion], device: DeviceSpec | None = None
                  ) -> str:
    if not suggestions:
        return f"{name}: no BLOCK_M/BLOCK_N constexprs captured; nothing to retile"
    lines = [name + (f"  [{device.name}, {device.sm_count} SMs]" if device else "")]
    width = max(len(s.label) for s in suggestions)
    for s in suggestions:
        tag = "current" if s.current else "try    "
        extra = "; ".join(s.reasons) if s.reasons else ""
        lines.append(
            f"  {tag}  {s.label:<{width}}  {s.programs:>4} programs  "
            f"{s.sm_balance:>3.0%} SM  {s.est_blocks_per_sm} blk/SM  "
            f"{s.smem // 1024:>3} KiB smem" + (f"  {extra}" if extra else ""))
    if len(suggestions) == 1 and suggestions[0].current:
        lines.append("  (already a good fit for this GPU at this problem size)")
    return "\n".join(lines)
