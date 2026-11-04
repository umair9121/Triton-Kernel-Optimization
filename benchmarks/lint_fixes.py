"""Do the linter's global-memory findings predict real speedups?

Each case runs a kernel the linter flags and a variant with the suggested fix applied. For
both, tritonlens reports the predicted bytes per lane (TL001) and sector efficiency (TL002)
from the captured IR; the kernels are then timed with interleaved ``do_bench`` rounds and
converted to effective bandwidth (useful bytes / time).

Usage: python benchmarks/lint_fixes.py [--out benchmarks/results]
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
import triton  # noqa: E402
from triton.testing import do_bench  # noqa: E402

import bad_access  # noqa: E402
import layernorm as ln  # noqa: E402
import softmax as sm  # noqa: E402
import vector_add as va  # noqa: E402
from tritonlens.analysis.lint import run_lint  # noqa: E402
from tritonlens.analyze import analyze_captured  # noqa: E402
from tritonlens.capture import capturing  # noqa: E402

ROUNDS = 7


def predict(fn) -> dict:
    with capturing() as rec:
        fn()
    torch.cuda.synchronize()
    kernel = next(iter(rec))
    an = analyze_captured(kernel)
    findings = run_lint(an)
    loads = [a for a in an.global_accesses if a.kind == "load"] or an.global_accesses
    return {
        "kernel": kernel.name,
        "vec_bytes": min(a.vec_bytes for a in loads),
        "ptx_bytes": min((a.ptx_bytes for a in loads if a.ptx_bytes), default=None),
        "efficiency": min(a.efficiency for a in an.global_accesses),
        "rules": sorted({f"{f.rule} ({f.severity})" for f in findings
                         if f.rule in ("TL001", "TL002", "TL006")}),
    }


def case_vector_add():
    n = 1 << 24
    base = torch.randn(3, n + 4, device="cuda")
    aligned = [base[i, :n] for i in range(3)]
    shifted = [base[i, 1:n + 1] for i in range(3)]
    n_odd = n - 3

    def run(tensors, size):
        x, y, out = tensors
        va.add_kernel[(triton.cdiv(size, 1024),)](x, y, out, size, BLOCK_SIZE=1024)

    return ("vector add, 16M fp32", 3 * n * 4, [
        ("offset views + odd n (TL001, TL006)", lambda: run(shifted, n_odd)),
        ("aligned tensors, n % 16 == 0", lambda: run(aligned, n)),
    ])


def case_softmax():
    rows = 8192
    x781 = torch.randn(rows, 781, device="cuda")
    x784 = torch.randn(rows, 784, device="cuda")
    o781, o784 = torch.empty_like(x781), torch.empty_like(x784)

    def run(x, o):
        sm.softmax_kernel[(rows,)](o, x, x.stride(0), o.stride(0), x.shape[1], BLOCK_SIZE=1024,
                                   num_warps=4)

    return ("softmax, 8192 rows fp32", None, [
        ("781 columns (TL001)", lambda: run(x781, o781), 2 * rows * 781 * 4),
        ("784 columns (padded to 16)", lambda: run(x784, o784), 2 * rows * 784 * 4),
    ])


def case_layernorm():
    M, N = 4096, 8192
    x = torch.randn(M, N, device="cuda", dtype=torch.float16)
    w = torch.rand(N, device="cuda", dtype=torch.float16)
    b = torch.rand(N, device="cuda", dtype=torch.float16)
    y = torch.empty_like(x)

    def run(block):
        ln.layernorm_kernel[(M,)](x, y, w, b, x.stride(0), N, 1e-5, BLOCK_SIZE=block, num_warps=8)

    return ("layernorm, 4096x8192 fp16", 3 * M * N * 2, [
        ("BLOCK=1024, 8 warps (TL001 info)", lambda: run(1024)),
        ("BLOCK=2048, 8 warps", lambda: run(2048)),
    ])


def case_strided():
    n = 1 << 22
    src = torch.randn(n * 8 + 8, device="cuda")
    dst = torch.empty(n, device="cuda")

    def run(stride):
        bad_access.strided_gather_kernel[(triton.cdiv(n, 1024),)](src, dst, n, stride, BLOCK=1024)

    return ("gather 4M fp32", 2 * n * 4, [
        ("stride 8 (TL002)", lambda: run(8)),
        ("stride 1", lambda: run(1)),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "benchmarks" / "results"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.cuda.get_device_name()
    rows = []
    for case in (case_vector_add, case_softmax, case_layernorm, case_strided):
        title, nbytes, variants = case()
        preds = [predict(v[1]) for v in variants]
        samples = [[] for _ in variants]
        for _ in range(ROUNDS):
            for i, v in enumerate(variants):
                samples[i].append(do_bench(v[1], warmup=25, rep=100))
        for v, pred, times in zip(variants, preds, samples):
            times.sort()
            ms = times[len(times) // 2]
            useful = v[2] if len(v) > 2 else nbytes
            rows.append({"case": title, "variant": v[0], "ms": ms,
                         "gbps": useful / (ms * 1e-3) / 1e9, **pred})
            print(f"{title:<26} {v[0]:<40} {pred['vec_bytes']:>2} B/lane  "
                  f"eff {pred['efficiency']:.0%}  {ms:.3f} ms  {rows[-1]['gbps']:.1f} GB/s  "
                  f"{','.join(pred['rules']) or '-'}")
    (out / "lint_fixes.json").write_text(json.dumps({"device": device, "rows": rows}, indent=2),
                                         encoding="utf-8")
    lines = [f"### Global-memory findings vs measured bandwidth ({device})", "",
             f"Median of {ROUNDS} interleaved rounds.", "",
             "| case | variant | findings | predicted B/lane | PTX B/lane | predicted sector "
             "efficiency | measured ms | effective GB/s | speedup |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows):
        base = rows[i - 1] if i % 2 else r
        speedup = base["ms"] / r["ms"] if i % 2 else 1.0
        lines.append(f"| {r['case']} | {r['variant']} | {', '.join(r['rules']) or '-'} | "
                     f"{r['vec_bytes']} | {r['ptx_bytes'] or '-'} | {r['efficiency']:.0%} | "
                     f"{r['ms']:.3f} | {r['gbps']:.1f} | {speedup:.2f}x |")
    md = "\n".join(lines) + "\n"
    (out / "lint_fixes.md").write_text(md, encoding="utf-8")
    print()
    print(md)


if __name__ == "__main__":
    main()
