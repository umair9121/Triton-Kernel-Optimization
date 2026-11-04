"""Do the device-aware findings (TL007-TL009) point at real speedups on this GPU?

For each configuration flagged in ``examples/big_gpu_configs.py`` the flagged launch and the
alternatives named in the finding's fix are timed (median of interleaved rounds, CUDA-graph
replay). The prediction columns come from tritonlens on the captured IR of each variant.

Usage: python benchmarks/device_rules.py [--out benchmarks/results]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import torch  # noqa: E402
from triton.testing import do_bench_cudagraph  # noqa: E402

import big_gpu_configs as cfg  # noqa: E402
from matmul import matmul  # noqa: E402
from tritonlens.analysis.lint import run_lint  # noqa: E402
from tritonlens.analyze import analyze_captured  # noqa: E402
from tritonlens.capture import capturing  # noqa: E402

ROUNDS = 7


def predict(fn) -> dict:
    with capturing() as rec:
        fn()
    torch.cuda.synchronize()
    an = analyze_captured(next(iter(rec)))
    perf = an.perf
    return {
        "blocksPerSm": perf.occupancy.blocks_per_sm,
        "occupancy": perf.occupancy.occupancy,
        "programs": perf.waves.total_blocks,
        "smBalance": perf.waves.sm_balance,
        "rules": sorted({f.rule for f in run_lint(an) if f.rule in ("TL007", "TL008", "TL009")}),
    }


def cases():
    torch.manual_seed(0)
    a640 = torch.randn(640, 640, device="cuda", dtype=torch.float16)
    b640 = torch.randn(640, 640, device="cuda", dtype=torch.float16)
    a2k = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    b2k = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    x = torch.randn(8, 1 << 22, device="cuda")

    def mm(a, b, bm, bn, bk, w, s):
        return lambda: matmul(a, b, block_m=bm, block_n=bn, block_k=bk, num_warps=w,
                              num_stages=s)

    yield "matmul 640^3 fp16", 2 * 640 ** 3, None, [
        ("128x128x64 w8 s3 (flagged)", mm(a640, b640, 128, 128, 64, 8, 3)),
        ("64x128x64 w4 s3", mm(a640, b640, 64, 128, 64, 4, 3)),
        ("64x64x64 w4 s3", mm(a640, b640, 64, 64, 64, 4, 3)),
    ]
    yield "matmul 2048^3 fp16", 2 * 2048 ** 3, None, [
        ("64x64x64 w4 s5 (flagged)", mm(a2k, b2k, 64, 64, 64, 4, 5)),
        ("64x64x64 w4 s4", mm(a2k, b2k, 64, 64, 64, 4, 4)),
        ("64x64x64 w4 s3", mm(a2k, b2k, 64, 64, 64, 4, 3)),
        ("64x64x64 w4 s2", mm(a2k, b2k, 64, 64, 64, 4, 2)),
    ]
    yield "row sum 8 x 4M fp32", None, x.numel() * 4, [
        ("1 program per row (flagged)", lambda: cfg.row_sum(x)),
        ("split 5 per row (40 programs)", lambda: cfg.row_sum_split(x, 5)),
        ("split 20 per row (160 programs)", lambda: cfg.row_sum_split(x, 20)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "benchmarks" / "results"))
    args = parser.parse_args()
    dev = torch.cuda.get_device_name()
    rows = []
    for title, flops, nbytes, variants in cases():
        preds = [predict(fn) for _, fn in variants]
        samples = [[] for _ in variants]
        for _ in range(ROUNDS):
            for i, (_, fn) in enumerate(variants):
                samples[i].append(do_bench_cudagraph(fn, rep=30))
        base = None
        for (label, _), pred, times in zip(variants, preds, samples):
            ms = sorted(times)[len(times) // 2]
            base = base or ms
            rate = (f"{flops / (ms * 1e-3) / 1e12:.2f} TFLOP/s" if flops else
                    f"{nbytes / (ms * 1e-3) / 1e9:.0f} GB/s")
            rows.append({"case": title, "variant": label, "ms": ms, "speedup": base / ms,
                         "rate": rate, **pred})
            print(f"{title:<20} {label:<34} {pred['blocksPerSm']:>2} blk/SM  {pred['programs']:>5} "
                  f"prog  bal {pred['smBalance']:.0%}  {ms:.3f} ms  {rate:<14} "
                  f"{base / ms:.2f}x  {','.join(pred['rules']) or '-'}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "device_rules.json").write_text(json.dumps({"device": dev, "rows": rows}, indent=2),
                                           encoding="utf-8")
    lines = [f"### Device-aware findings vs measured time ({dev})", "",
             f"Median of {ROUNDS} interleaved rounds, CUDA-graph replay. Speedup is relative to "
             "the flagged configuration.", "",
             "| case | variant | findings | blocks/SM | programs | SM balance | ms | rate | "
             "speedup |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['case']} | {r['variant']} | {', '.join(r['rules']) or '-'} | "
                     f"{r['blocksPerSm']} | {r['programs']} | {r['smBalance']:.0%} | "
                     f"{r['ms']:.3f} | {r['rate']} | {r['speedup']:.2f}x |")
    (out / "device_rules.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
