"""Predicted vs measured shared-memory bank conflicts on a real GPU.

The fp16 matmul is compiled normally, then its TTGIR is rewritten with different
``#shared`` swizzle parameters and recompiled from the edited IR with ``triton.compile``.
Every variant computes the same result; only the shared-memory address pattern changes.
For each variant tritonlens predicts the shared-memory wavefronts per CTA per K-iteration,
and the kernel is timed with ``triton.testing.do_bench``.

Usage: python benchmarks/bank_conflicts.py [--size 2048] [--out benchmarks/results]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import torch  # noqa: E402
import triton  # noqa: E402
from triton.testing import do_bench  # noqa: E402

from tritonlens.analysis.banks import analyze_function  # noqa: E402
from tritonlens.capture import capturing  # noqa: E402
from tritonlens.ir import parse_module  # noqa: E402

CONFIGS = [
    {"block_m": 64, "block_n": 64, "block_k": 32, "num_warps": 4, "num_stages": 3},
    {"block_m": 128, "block_n": 128, "block_k": 32, "num_warps": 4, "num_stages": 3},
]

ROUNDS = 7

SHARED_RE = re.compile(r"^(#shared\w*) = #triton_gpu\.shared<\{(.*?)\}>", re.M)


def set_swizzle(text: str, which: str, per_phase: int | None, max_phase: int | None) -> str:
    """Rewrite perPhase/maxPhase of shared encodings (which: 'all', 'A' or 'B')."""
    aliases = [m.group(1) for m in SHARED_RE.finditer(text)]
    targets = {"all": aliases, "A": aliases[:1], "B": aliases[1:2]}[which]

    def repl(m):
        if m.group(1) not in targets:
            return m.group(0)
        body = m.group(2)
        if per_phase is not None:
            body = re.sub(r"perPhase = \d+", f"perPhase = {per_phase}", body)
        if max_phase is not None:
            body = re.sub(r"maxPhase = \d+", f"maxPhase = {max_phase}", body)
        return f"{m.group(1)} = #triton_gpu.shared<{{{body}}}>"

    return SHARED_RE.sub(repl, text)


VARIANTS = [
    ("triton default", lambda t: t),
    ("maxPhase=2", lambda t: set_swizzle(t, "all", None, 2)),
    ("A unswizzled", lambda t: set_swizzle(t, "A", 1, 1)),
    ("B unswizzled", lambda t: set_swizzle(t, "B", 1, 1)),
    ("no swizzle", lambda t: set_swizzle(t, "all", 1, 1)),
]


def predict(ttgir: str) -> dict:
    mod = parse_module(ttgir)
    accesses = analyze_function(mod.func, mod.num_warps)
    in_loop = [a for a in accesses if a.op.depth >= 1 and not a.estimated]
    per_iter = sum(a.wavefronts for a in in_loop)
    ideal = sum(a.ideal for a in in_loop)
    by_mech: dict[str, float] = {}
    for a in in_loop:
        by_mech[a.mechanism] = max(by_mech.get(a.mechanism, 0), a.degree)
    return {"wavefronts_per_iter": per_iter, "ideal_per_iter": ideal,
            "degree": per_iter / ideal if ideal else 1.0, "worst_by_mechanism": by_mech,
            "shared": [m.group(0).split(" = ", 1)[1] for m in SHARED_RE.finditer(ttgir)]}


def run_config(cfg: dict, size: int, workdir: Path) -> list[dict]:
    import matmul as mm

    torch.manual_seed(0)
    a = torch.randn((size, size), device="cuda", dtype=torch.float16)
    b = torch.randn((size, size), device="cuda", dtype=torch.float16)
    ref = a @ b
    with capturing() as rec:
        mm.matmul(a, b, **cfg)
    kernel = next(iter(rec))
    base_ttgir = kernel.ttgir
    c = torch.empty((size, size), device="cuda", dtype=torch.float16)
    grid = (triton.cdiv(size, cfg["block_m"]) * triton.cdiv(size, cfg["block_n"]), 1, 1)
    args = (a, b, c, size, size, size, a.stride(0), b.stride(0), c.stride(0))

    rows, launches = [], []
    for name, transform in VARIANTS:
        ttgir = transform(base_ttgir)
        vdir = workdir / re.sub(r"\W+", "_", f"{cfg['block_m']}_{name}")
        vdir.mkdir(parents=True, exist_ok=True)
        path = vdir / "matmul_kernel.ttgir"
        path.write_text(ttgir, encoding="utf-8")
        compiled = triton.compile(str(path))
        launch = compiled[grid]
        c.zero_()
        launch(*args)
        torch.cuda.synchronize()
        ok = torch.allclose(c, ref, atol=1e-1, rtol=1e-2)
        launches.append(launch)
        rows.append({
            "config": f"{cfg['block_m']}x{cfg['block_n']}x{cfg['block_k']} "
                      f"w{cfg['num_warps']} s{cfg['num_stages']}",
            "variant": name,
            "correct": bool(ok),
            "ldmatrix_ptx": compiled.asm["ptx"].count("ldmatrix"),
            **predict(ttgir),
        })

    # Laptop GPUs change clocks with temperature: interleave variants over several rounds
    # and keep the median so every variant sees the same conditions.
    samples: list[list[float]] = [[] for _ in launches]
    for _ in range(ROUNDS):
        for i, launch in enumerate(launches):
            samples[i].append(do_bench(lambda launch=launch: launch(*args), warmup=25, rep=150))
    for row, times in zip(rows, samples):
        times.sort()
        row["ms"] = times[len(times) // 2]
        row["ms_min"], row["ms_max"] = times[0], times[-1]
        row["tflops"] = 2 * size ** 3 / (row["ms"] * 1e-3) / 1e12
        print(f"{row['config']:<22} {row['variant']:<16} pred {row['degree']:.2f}x "
              f"({row['wavefronts_per_iter']}/{row['ideal_per_iter']} wf/iter)  "
              f"{row['ms']:.3f} ms [{row['ms_min']:.3f}-{row['ms_max']:.3f}]  "
              f"{'ok' if row['correct'] else 'WRONG'}")
    base = rows[0]["ms"]
    for row in rows:
        row["slowdown"] = row["ms"] / base
    return rows


def to_markdown(rows: list[dict], size: int, device: str) -> str:
    lines = [
        f"### Shared-memory swizzle sweep (fp16 matmul {size}x{size}x{size}, {device})",
        "",
        f"Median of {ROUNDS} interleaved rounds; range in brackets.",
        "",
        "| config | variant | predicted conflict degree | wavefronts / iter (ideal) "
        "| measured ms | slowdown | correct |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['config']} | {r['variant']} | {r['degree']:.2f}x | "
            f"{r['wavefronts_per_iter']} ({r['ideal_per_iter']}) | {r['ms']:.3f} "
            f"[{r['ms_min']:.3f}-{r['ms_max']:.3f}] | "
            f"{r['slowdown']:.2f}x | {'yes' if r['correct'] else 'NO'} |"
        )
    lines.append("")
    for config in dict.fromkeys(r["config"] for r in rows):
        sub = [r for r in rows if r["config"] == config]
        corr = pearson([r["wavefronts_per_iter"] for r in sub], [r["ms"] for r in sub])
        lines.append(f"- `{config}`: Pearson r(predicted wavefronts, measured time) = {corr:.2f}")
    return "\n".join(lines) + "\n"


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx and vy else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--out", default=str(ROOT / "benchmarks" / "results"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.cuda.get_device_name()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for cfg in CONFIGS:
            rows.extend(run_config(cfg, args.size, Path(tmp)))
    (out / "bank_conflicts.json").write_text(json.dumps(
        {"device": device, "size": args.size, "rows": rows}, indent=2), encoding="utf-8")
    md = to_markdown(rows, args.size, device)
    (out / "bank_conflicts.md").write_text(md, encoding="utf-8")
    print()
    print(md)


if __name__ == "__main__":
    main()
