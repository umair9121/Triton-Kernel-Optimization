"""Device-aware pruning for ``triton.autotune`` config lists.

Autotuning on a small GPU wastes most of its time on configs written for big ones: every
config is compiled (seconds each) before Triton discovers that it needs more shared memory
than the SM has, and configs whose grid leaves SMs idle are benchmarked only to lose. This
pruner drops both *before* compilation::

    from tritonlens.autotune import device_pruner

    @triton.autotune(configs=CONFIGS, key=["M", "N", "K"],
                     prune_configs_by={"early_config_prune": device_pruner()})
    @triton.jit
    def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ..., BLOCK_M: tl.constexpr, ...):

Rules, applied only when the needed quantities can be inferred (unknown configs are kept):

1. shared memory: software-pipelined operand tiles need at least
   ``(num_stages - 1) * (BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N) * element_size`` bytes; configs
   above the per-block limit can never launch on this GPU.
2. SM balance: with ``P`` equal-work programs on ``S`` SMs the kernel lasts as long as
   ``ceil(P/S)*S`` programs (validated within 6% in ``benchmarks/waves.py``). Configs below
   ``min_balance`` are dropped when some other config reaches ``good_balance``.
3. L2 stripe: ``GROUP_M * BLOCK_M * K * elem`` must fit in this GPU's L2 for grouped
   ordering to help. Larger ``GROUP_M`` values of an otherwise-identical tile are dropped
   when a smaller one fits (2 MB laptop L2 vs 40 MB on an A100).

``smem`` and ``programs`` callables override the matmul defaults for other kernels.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .device import DeviceSpec

_BLOCK_KEYS = {
    "M": ("BLOCK_M", "BLOCK_SIZE_M", "BM"),
    "N": ("BLOCK_N", "BLOCK_SIZE_N", "BN"),
    "K": ("BLOCK_K", "BLOCK_SIZE_K", "BK"),
}


def _block(kwargs: dict, dim: str) -> int | None:
    for key in _BLOCK_KEYS[dim]:
        if key in kwargs:
            return int(kwargs[key])
    return None


def _elem_bytes(named_args: dict) -> int:
    for value in named_args.values():
        size = getattr(value, "element_size", None)
        if callable(size):
            return int(size())
        if isinstance(size, int) and size > 0:
            return size
    return 2


def matmul_smem(config, named_args: dict) -> int | None:
    """Lower bound on shared memory for a pipelined ``BLOCK_M x BLOCK_N x BLOCK_K`` matmul."""
    bm, bn, bk = (_block(config.kwargs, d) for d in "MNK")
    if None in (bm, bn, bk):
        return None
    stages = max(int(getattr(config, "num_stages", 2) or 2), 2)
    return (stages - 1) * (bm * bk + bk * bn) * _elem_bytes(named_args)


def matmul_programs(config, named_args: dict) -> int | None:
    """``cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N)`` (times any batch/split ``*_SPLIT`` factor)."""
    bm, bn = _block(config.kwargs, "M"), _block(config.kwargs, "N")
    m, n = named_args.get("M"), named_args.get("N")
    if None in (bm, bn) or not isinstance(m, int) or not isinstance(n, int):
        return None
    return math.ceil(m / bm) * math.ceil(n / bn)


def matmul_l2_stripe(config, named_args: dict) -> int | None:
    """Bytes of A that grouped ordering wants to keep in L2 (``GROUP_M * BLOCK_M * K * elem``)."""
    gm = config.kwargs.get("GROUP_M")
    bm = _block(config.kwargs, "M")
    k = named_args.get("K")
    if gm is None or bm is None or not isinstance(k, int):
        return None
    return int(gm) * bm * k * _elem_bytes(named_args)


def sm_balance(programs: int, sm_count: int) -> float:
    return programs / (math.ceil(programs / sm_count) * sm_count)


@dataclass
class Decision:
    config: Any
    keep: bool
    reason: str
    smem: int | None = None
    programs: int | None = None
    balance: float | None = None
    l2_stripe: int | None = None


def _tile_key(config) -> tuple:
    kw = getattr(config, "kwargs", {})
    return (_block(kw, "M"), _block(kw, "N"), _block(kw, "K"),
            getattr(config, "num_stages", None), getattr(config, "num_warps", None))


def evaluate(configs, named_args: dict, device: DeviceSpec,
             smem: Callable | None = matmul_smem, programs: Callable | None = matmul_programs,
             min_balance: float = 0.75, good_balance: float = 0.9) -> list[Decision]:
    lim = device.limits
    out = []
    for cfg in configs:
        need = smem(cfg, named_args) if smem else None
        progs = programs(cfg, named_args) if programs else None
        bal = sm_balance(progs, device.sm_count) if progs else None
        stripe = matmul_l2_stripe(cfg, named_args)
        if need is not None and need > lim.max_shared_per_block:
            out.append(Decision(cfg, False, f"needs >= {need // 1024} KiB shared memory, "
                                f"limit {lim.max_shared_per_block // 1024} KiB",
                                need, progs, bal, stripe))
        else:
            out.append(Decision(cfg, True, "", need, progs, bal, stripe))
    fitting = [d for d in out if d.keep and d.balance is not None]
    if any(d.balance >= good_balance for d in fitting):
        for d in fitting:
            if d.balance < min_balance:
                d.keep = False
                d.reason = (f"{d.programs} programs on {device.sm_count} SMs: "
                            f"{d.balance:.0%} SM utilization")
    if device.l2_bytes:
        groups: dict[tuple, list[Decision]] = {}
        for d in out:
            if d.keep and d.l2_stripe is not None:
                groups.setdefault(_tile_key(d.config), []).append(d)
        for bucket in groups.values():
            if any(d.l2_stripe <= device.l2_bytes for d in bucket):
                for d in bucket:
                    if d.l2_stripe > device.l2_bytes:
                        d.keep = False
                        d.reason = (f"GROUP_M A-stripe {d.l2_stripe // 1024} KiB exceeds "
                                    f"{device.l2_bytes // 1024} KiB L2")
    if not any(d.keep for d in out):  # never prune everything
        for d in out:
            d.keep, d.reason = True, ""
    return out


def device_pruner(device: DeviceSpec | None = None, smem: Callable | None = matmul_smem,
                  programs: Callable | None = matmul_programs, min_balance: float = 0.75,
                  good_balance: float = 0.9, verbose: bool = False) -> Callable:
    """An ``early_config_prune`` callback for ``triton.autotune``."""
    state: dict[str, DeviceSpec | None] = {"device": device}

    def prune(configs, named_args, **kwargs):
        if state["device"] is None:
            from .device import detect

            state["device"] = detect()
        dev = state["device"]
        if dev is None:
            return list(configs)
        decisions = evaluate(configs, {**named_args, **kwargs}, dev, smem, programs,
                             min_balance, good_balance)
        if verbose:
            dropped = [d for d in decisions if not d.keep]
            print(f"[tritonlens] pruned {len(dropped)}/{len(decisions)} configs for {dev.name}",
                  file=sys.stderr)
            for d in dropped:
                print(f"  - {d.config}: {d.reason}", file=sys.stderr)
        return [d.config for d in decisions if d.keep]

    return prune
