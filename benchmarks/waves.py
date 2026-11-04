"""Tail effects on a 20-SM GPU: does run time follow waves or per-SM load balance?

A matmul with a fixed tile and fixed K is launched on 1..MAX_TILES output tiles, so every
block does identical work and only the grid size changes. Two models predict run time
(up to a constant):

* waves:      ceil(tiles / (blocks_per_SM * SMs))            - blocks run in lockstep waves
* SM balance: ceil(tiles / SMs) / blocks_per_SM-saturation  - each SM's throughput is fixed

The constant is fitted by least squares and the residual reported for both.

Usage: python benchmarks/waves.py [--out benchmarks/results]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import torch  # noqa: E402
from triton.testing import do_bench_cudagraph  # noqa: E402

import matmul as mm  # noqa: E402
from tritonlens import device as devmod  # noqa: E402
from tritonlens.analysis.occupancy import occupancy  # noqa: E402

MAX_TILES = 64
K = 4096
CONFIGS = [  # (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages)
    (128, 128, 32, 8, 3),
    (64, 64, 32, 4, 3),
]


def fit(pred: list[float], meas: list[float]) -> tuple[float, float]:
    """Scale ``pred`` to ``meas`` (least squares through 0); return scale and mean |error|."""
    scale = sum(p * m for p, m in zip(pred, meas)) / sum(p * p for p in pred)
    err = sum(abs(scale * p - m) / m for p, m in zip(pred, meas)) / len(meas)
    return scale, err


def launcher(a, b, c, tiles, bm, bn, bk, warps, stages):
    """One row of ``tiles`` output tiles: every program does identical work."""
    def run():
        return mm.matmul_kernel[(tiles,)](
            a, b, c, bm, bn * tiles, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1), BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=1,
            num_warps=warps, num_stages=stages)

    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "benchmarks" / "results"))
    args = parser.parse_args()
    dev = devmod.detect()
    results = []
    for bm, bn, bk, warps, stages in CONFIGS:
        a = torch.randn(bm, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, bn * MAX_TILES, device="cuda", dtype=torch.float16)
        c = torch.empty(bm, bn * MAX_TILES, device="cuda", dtype=torch.float16)
        rows, compiled = [], None
        for tiles in range(1, MAX_TILES + 1):
            run = launcher(a, b, c, tiles, bm, bn, bk, warps, stages)
            compiled = run()
            ms = sorted(do_bench_cudagraph(run, rep=30) for _ in range(5))[2]
            rows.append({"tiles": tiles, "ms": ms})
        occ = occupancy(dev, warps, compiled.n_regs, compiled.metadata.shared)
        per_wave = occ.blocks_per_sm * dev.sm_count
        t = [r["tiles"] for r in rows]
        meas = [r["ms"] for r in rows]
        wave_pred = [math.ceil(x / per_wave) for x in t]
        sm_pred = [math.ceil(x / dev.sm_count) for x in t]
        _, wave_err = fit(wave_pred, meas)
        _, sm_err = fit(sm_pred, meas)
        label = f"{bm}x{bn}x{bk} w{warps} s{stages}"
        print(f"{label}: {occ.blocks_per_sm} blocks/SM ({occ.limiter}); mean error "
              f"wave model {wave_err:.1%}, SM-balance model {sm_err:.1%}")
        results.append({"config": label, "blocksPerSm": occ.blocks_per_sm,
                        "limiter": occ.limiter, "waveErr": wave_err, "smErr": sm_err,
                        "rows": rows})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "waves.json").write_text(json.dumps({"device": dev.name, "K": K,
                                                "results": results}, indent=2),
                                    encoding="utf-8")
    lines = [f"### Tail effects vs grid size ({dev.name}, {dev.sm_count} SMs, K={K})", "",
             "| config | blocks/SM | wave-model error | SM-balance-model error | "
             "ms @ 20 tiles | ms @ 21 tiles | ms @ 40 tiles |", "|---|---|---|---|---|---|---|"]
    for r in results:
        ms = {x["tiles"]: x["ms"] for x in r["rows"]}
        lines.append(f"| {r['config']} | {r['blocksPerSm']} | {r['waveErr']:.1%} | "
                     f"{r['smErr']:.1%} | {ms[20]:.3f} | {ms[21]:.3f} | {ms[40]:.3f} |")
    (out / "waves.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
