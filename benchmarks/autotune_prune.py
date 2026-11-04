"""How much autotuning time does device-aware pruning save, and does it lose the best config?

A datacenter-style matmul sweep (BLOCK_M/N in {64,128,256}, BLOCK_K in {32,64},
num_stages in {3,4,5}, num_warps in {4,8}: 108 configs) is autotuned with and without
``tritonlens.autotune.device_pruner``. Every run is a fresh subprocess with an empty Triton
cache, so compile time is included exactly as a user would see it on first run.

Usage: python benchmarks/autotune_prune.py [--sizes 640 1024 2048] [--out benchmarks/results]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configs():
    import triton

    return [triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8},
                          num_stages=s, num_warps=w)
            for bm, bn, bk, s, w in itertools.product((64, 128, 256), (64, 128, 256), (32, 64),
                                                      (3, 4, 5), (4, 8))]


def worker(size: int, prune: bool) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "examples"))
    import torch
    import triton
    from triton.testing import do_bench_cudagraph

    import matmul as mm
    from tritonlens.autotune import device_pruner

    cfgs = configs()
    extra = {"prune_configs_by": {"early_config_prune": device_pruner()}} if prune else {}
    tuned = triton.autotune(cfgs, key=["M", "N", "K"], **extra)(mm.matmul_kernel)
    a = torch.randn(size, size, device="cuda", dtype=torch.float16)
    b = torch.randn(size, size, device="cuda", dtype=torch.float16)
    c = torch.empty(size, size, device="cuda", dtype=torch.float16)

    def run():
        grid = lambda meta: (triton.cdiv(size, meta["BLOCK_M"]) *  # noqa: E731
                             triton.cdiv(size, meta["BLOCK_N"]),)
        tuned[grid](a, b, c, size, size, size, a.stride(0), a.stride(1), b.stride(0),
                    b.stride(1), c.stride(0), c.stride(1))

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    run()
    torch.cuda.synchronize()
    tune_s = time.perf_counter() - t0
    timings = getattr(tuned, "configs_timings", None) or {}
    failed = sum(1 for v in timings.values()
                 if any(x == float("inf") for x in (v if isinstance(v, list) else [v])))
    torch.testing.assert_close(c, a @ b, atol=1e-1, rtol=1e-2)
    best_ms = sorted(do_bench_cudagraph(run, rep=50) for _ in range(5))[2]
    bc = tuned.best_config
    label = (f"{bc.kwargs['BLOCK_M']}x{bc.kwargs['BLOCK_N']}x{bc.kwargs['BLOCK_K']} "
             f"w{bc.num_warps} s{bc.num_stages}")
    return {"size": size, "prune": prune, "configs": len(cfgs), "benchmarked": len(timings),
            "failed": failed, "tuneSeconds": tune_s, "bestConfig": label,
            "bestMs": best_ms, "tflops": 2 * size ** 3 / (best_ms * 1e-3) / 1e12}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[640, 1024, 2048])
    parser.add_argument("--out", default=str(ROOT / "benchmarks" / "results"))
    parser.add_argument("--dry", action="store_true",
                        help="only print how many configs the pruner keeps (no compile)")
    parser.add_argument("--worker", nargs=2, metavar=("SIZE", "PRUNE"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.dry:
        sys.path.insert(0, str(ROOT / "src"))
        from tritonlens.autotune import evaluate
        from tritonlens.device import detect, for_target

        spec = detect() or for_target("cuda:86")
        print(spec.describe())
        cfgs = configs()
        for size in args.sizes:
            named = {"M": size, "N": size, "K": size}

            class Half:
                def element_size(self):
                    return 2

            named["a"] = Half()
            decisions = evaluate(cfgs, named, spec)
            keep = [d for d in decisions if d.keep]
            print(f"{size:>5}^3  keep {len(keep):>3}/{len(decisions)}  "
                  f"drop smem {sum(1 for d in decisions if 'shared' in d.reason)}  "
                  f"SM {sum(1 for d in decisions if 'SM' in d.reason)}  "
                  f"L2 {sum(1 for d in decisions if 'L2' in d.reason)}")
        return
    if args.worker:
        print("RESULT " + json.dumps(worker(int(args.worker[0]), args.worker[1] == "1")))
        return
    rows = []
    for size in args.sizes:
        for prune in (False, True):
            with tempfile.TemporaryDirectory() as cache:
                env = {**os.environ, "TRITON_CACHE_DIR": cache}
                proc = subprocess.run([sys.executable, __file__, "--worker", str(size),
                                       "1" if prune else "0"], env=env, capture_output=True,
                                      text=True)
            line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
            if line is None:
                print(proc.stdout[-2000:], proc.stderr[-4000:])
                raise SystemExit(f"worker failed for size={size} prune={prune}")
            row = json.loads(line[7:])
            rows.append(row)
            mode = "pruned" if prune else "all   "
            print(f"{size:>5} {mode} benchmarked {row['benchmarked']:>3} "
                  f"(failed {row['failed']:>2})  tune {row['tuneSeconds']:6.1f} s  best "
                  f"{row['bestMs']:.3f} ms ({row['tflops']:.2f} TFLOP/s)  {row['bestConfig']}")
    import torch

    device = torch.cuda.get_device_name()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "autotune_prune.json").write_text(json.dumps({"device": device, "rows": rows},
                                                        indent=2), encoding="utf-8")
    lines = [f"### Autotuning with device-aware pruning ({device})", "",
             "108-config datacenter-style matmul sweep, fp16, fresh Triton cache per run.", "",
             "| size | configs benchmarked | failed (out of resources) | autotune time | "
             "best kernel | best config |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['size']}^3 {'pruned' if r['prune'] else 'all'} | {r['benchmarked']} "
                     f"| {r['failed']} | {r['tuneSeconds']:.1f} s | {r['bestMs']:.3f} ms "
                     f"({r['tflops']:.2f} TFLOP/s) | `{r['bestConfig']}` |")
    (out / "autotune_prune.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
