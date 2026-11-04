"""Validate the occupancy model against the CUDA driver.

Compiles a sweep of matmul / softmax configurations (spanning register-, shared-memory- and
warp-limited cases), predicts blocks per SM from compile metadata only, and compares with
``cuOccupancyMaxActiveBlocksPerMultiprocessor`` on the loaded kernel. Configs that Triton
rejects with ``OutOfResources`` are checked against the model's "does not fit" prediction.

Usage: python benchmarks/occupancy_check.py [--out benchmarks/results]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import torch  # noqa: E402
import triton  # noqa: E402

import matmul as mm  # noqa: E402
import softmax as sm  # noqa: E402
from tritonlens import device as devmod  # noqa: E402
from tritonlens.analysis.occupancy import driver_blocks_per_sm, occupancy  # noqa: E402


def matmul_configs():
    a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    b = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    c = torch.empty(512, 512, device="cuda", dtype=torch.float16)
    for bm, bn, bk, warps, stages in itertools.product((32, 64, 128), (32, 64, 128), (32, 64),
                                                       (2, 4, 8), (2, 3, 4, 5)):
        if warps == 2 and bm * bn > 64 * 64:
            continue
        label = f"matmul {bm}x{bn}x{bk} w{warps} s{stages}"

        def launch(bm=bm, bn=bn, bk=bk, warps=warps, stages=stages):
            grid = (triton.cdiv(512, bm) * triton.cdiv(512, bn),)
            return mm.matmul_kernel[grid](a, b, c, 512, 512, 512, 512, 1, 512, 1, 512, 1,
                                          BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
                                          num_warps=warps, num_stages=stages)

        yield label, launch, warps


def softmax_configs():
    x = torch.randn(64, 4096, device="cuda")
    o = torch.empty_like(x)
    for block, warps in itertools.product((1024, 2048, 4096), (1, 2, 4, 8, 16)):
        def launch(block=block, warps=warps):
            return sm.softmax_kernel[(64,)](o, x, x.stride(0), o.stride(0), 4096,
                                            BLOCK_SIZE=block, num_warps=warps)

        yield f"softmax {block} w{warps}", launch, warps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "benchmarks" / "results"))
    args = parser.parse_args()
    dev = devmod.detect()
    rows = []
    for label, launch, warps in itertools.chain(matmul_configs(), softmax_configs()):
        try:
            compiled = launch()
        except triton.runtime.errors.OutOfResources as exc:
            pred = occupancy(dev, warps, 0, exc.required)
            rows.append({"config": label, "regs": None, "shared": exc.required,
                         "predicted": pred.blocks_per_sm, "driver": 0, "limiter": "does not fit",
                         "match": not pred.fits})
            continue
        meta = compiled.metadata
        pred = occupancy(dev, meta.num_warps, compiled.n_regs, meta.shared)
        actual = driver_blocks_per_sm(compiled)
        rows.append({"config": label, "regs": compiled.n_regs, "shared": meta.shared,
                     "predicted": pred.blocks_per_sm, "driver": actual, "limiter": pred.limiter,
                     "occupancy": pred.occupancy, "match": pred.blocks_per_sm == actual})
    torch.cuda.synchronize()
    ok = sum(r["match"] for r in rows)
    limiters = {}
    for r in rows:
        limiters[r["limiter"]] = limiters.get(r["limiter"], 0) + 1
    for r in rows:
        if not r["match"]:
            print("MISMATCH", r)
    print(f"{ok}/{len(rows)} configs match the driver; limiters: {limiters}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "occupancy_check.json").write_text(
        json.dumps({"device": dev.name, "rows": rows}, indent=2), encoding="utf-8")
    lines = [f"### Occupancy model vs `cuOccupancyMaxActiveBlocksPerMultiprocessor` ({dev.name})",
             "", f"**{ok}/{len(rows)} configurations match.** Limiting resource: " +
             ", ".join(f"{k} ({v})" for k, v in sorted(limiters.items())), "",
             "| config | regs/thread | shared B | predicted blocks/SM | driver | limiter |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['config']} | {r['regs'] if r['regs'] is not None else '-'} | "
                     f"{r['shared']} | {r['predicted']} | {r['driver']} | {r['limiter']} |")
    (out / "occupancy_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
