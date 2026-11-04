"""VRAM headroom and L2 working-set estimates.

A 4 GB laptop GPU fails in ways an A100 does not: the CUDA context plus PyTorch plus a few
fp16 activations leave little room for Triton's compiler, and a 1-2 MB L2 cannot hold the
grouped-matmul A-stripe that tutorials size for a 40 MB L2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..capture import ArgInfo
from ..device import DeviceSpec

if TYPE_CHECKING:
    from ..analyze import Analysis

_BLOCK = {
    "M": ("BLOCK_M", "BLOCK_SIZE_M", "BM"),
    "N": ("BLOCK_N", "BLOCK_SIZE_N", "BN"),
    "K": ("BLOCK_K", "BLOCK_SIZE_K", "BK"),
}


def fmt_bytes(n: int | float) -> str:
    n = float(n)
    if abs(n) >= 1 << 30:
        return f"{n / (1 << 30):.2f} GiB"
    if abs(n) >= 1 << 20:
        return f"{n / (1 << 20):.1f} MiB"
    if abs(n) >= 1 << 10:
        return f"{n / (1 << 10):.1f} KiB"
    return f"{int(n)} B"


def launch_ints(args: list[ArgInfo] | None) -> dict[str, int]:
    """Integer runtime and constexpr arguments from a captured launch."""
    out: dict[str, int] = {}
    for arg in args or []:
        if arg.kind in ("int", "constexpr") and isinstance(arg.value, int):
            out[arg.name] = arg.value
    return out


def block_size(consts: dict[str, int], dim: str) -> int | None:
    for key in _BLOCK[dim]:
        if key in consts:
            return consts[key]
    return None


def tensor_footprint(args: list[ArgInfo] | None) -> tuple[int, list[tuple[str, int]]]:
    """Unique-storage tensor bytes (same ``data_ptr`` counted once, taking the larger view)."""
    seen: dict[object, tuple[str, int]] = {}
    anon = 0
    for arg in args or []:
        if arg.kind != "tensor" or not arg.shape or not arg.element_size:
            continue
        nbytes = int(math.prod(arg.shape) * arg.element_size)
        key: object = arg.data_ptr if arg.data_ptr else ("anon", anon)
        anon += 1
        prev = seen.get(key)
        if prev is None or nbytes > prev[1]:
            seen[key] = (arg.name, nbytes)
    items = list(seen.values())
    return sum(b for _, b in items), items


def elem_bytes(args: list[ArgInfo] | None, default: int = 2) -> int:
    for arg in args or []:
        if arg.kind == "tensor" and arg.element_size:
            return int(arg.element_size)
    return default


def group_a_stripe(consts: dict[str, int], nbytes_elem: int) -> int | None:
    """Bytes of A kept in L2 by grouped program ordering: ``GROUP_M * BLOCK_M * K * elem``.

    Neighbouring programs in a group share those rows of A while they sweep N. If the stripe
    does not fit in L2, the grouping (copied from A100 tutorials at GROUP_M=8) does nothing.
    """
    gm, bm, k = consts.get("GROUP_M"), block_size(consts, "M"), consts.get("K")
    if None in (gm, bm, k) or gm <= 1 or not nbytes_elem:
        return None
    return int(gm) * int(bm) * int(k) * nbytes_elem


def fitting_group_m(consts: dict[str, int], nbytes_elem: int, l2_bytes: int) -> int | None:
    bm, k = block_size(consts, "M"), consts.get("K")
    if not (bm and k and nbytes_elem and l2_bytes):
        return None
    per = bm * k * nbytes_elem
    if per <= 0:
        return None
    return max(1, l2_bytes // per)


@dataclass
class Budget:
    tensor_bytes: int
    per_tensor: list[tuple[str, int]]
    total_memory: int
    free_memory: int | None
    allocated: int | None
    l2_bytes: int
    l2_group_bytes: int | None
    group_m: int | None
    suggested_group_m: int | None
    elem_bytes: int = 2
    consts: dict[str, int] = field(default_factory=dict)

    @property
    def fraction(self) -> float:
        return self.tensor_bytes / self.total_memory if self.total_memory else 0.0

    @property
    def headroom(self) -> int | None:
        if self.free_memory is not None:
            return self.free_memory
        if self.total_memory:
            return max(0, self.total_memory - self.tensor_bytes)
        return None

    @property
    def l2_fits(self) -> bool | None:
        if self.l2_group_bytes is None or not self.l2_bytes:
            return None
        return self.l2_group_bytes <= self.l2_bytes

    def to_json(self) -> dict:
        return {
            "tensorBytes": self.tensor_bytes,
            "perTensor": [{"name": n, "bytes": b} for n, b in self.per_tensor],
            "totalMemory": self.total_memory,
            "freeMemory": self.free_memory,
            "allocated": self.allocated,
            "fraction": self.fraction,
            "headroom": self.headroom,
            "l2Bytes": self.l2_bytes,
            "l2GroupBytes": self.l2_group_bytes,
            "l2Fits": self.l2_fits,
            "groupM": self.group_m,
            "suggestedGroupM": self.suggested_group_m,
        }


def analyze_budget(an: Analysis, device: DeviceSpec) -> Budget | None:
    launch = an.launch
    if launch is None:
        return None
    args = launch.args
    tens, per = tensor_footprint(args)
    consts = launch_ints(args)
    nbytes = elem_bytes(args)
    stripe = group_a_stripe(consts, nbytes)
    meta = an.metadata
    return Budget(
        tensor_bytes=tens, per_tensor=per, total_memory=device.total_memory,
        free_memory=meta.get("free_memory"), allocated=meta.get("allocated"),
        l2_bytes=device.l2_bytes, l2_group_bytes=stripe,
        group_m=consts.get("GROUP_M"),
        suggested_group_m=fitting_group_m(consts, nbytes, device.l2_bytes),
        elem_bytes=nbytes, consts=consts,
    )
