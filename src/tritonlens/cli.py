"""Command-line interface.

    tritonlens run examples/matmul.py -o reports/ -- --size 2048   # args after -- go to the script
    tritonlens analyze kernel.ttgir -o report.html          # analyze IR captured elsewhere
    tritonlens lint examples/softmax.py --format sarif      # CI-friendly findings
    tritonlens validate                                     # check layout models on this GPU
    tritonlens device --calibrate                           # GPU limits + measured peaks
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

from . import __version__
from .analysis.lint import SEVERITY_ORDER, Finding, run_lint
from .analyze import Analysis, analyze_captured
from .capture import CapturedKernel, capture_script, load_dump_dir
from .output import format_text, to_json, to_sarif
from .report import build_report, write_html


def _load_kernels(target: Path, script_args: list[str]) -> list[CapturedKernel]:
    if target.suffix == ".py":
        return capture_script(target, script_args)
    if target.suffix == ".ttgir":
        return [CapturedKernel.from_files(target)]
    if target.is_dir():
        kernels = load_dump_dir(target)
        if kernels:
            return kernels
    raise SystemExit(f"tritonlens: don't know how to analyze {target} (expected a .py script, "
                     "a .ttgir file or a directory of captured kernels)")


def _analyze_all(kernels: list[CapturedKernel]) -> list[tuple[Analysis, list[Finding]]]:
    out = []
    for kernel in kernels:
        if not kernel.ttgir:
            print(f"tritonlens: {kernel.name}: no TTGIR captured, skipping", file=sys.stderr)
            continue
        an = analyze_captured(kernel)
        out.append((an, run_lint(an)))
    return out


def _summary_line(an: Analysis, findings: list[Finding]) -> str:
    counts = {s: sum(1 for f in findings if f.severity == s) for s in ("error", "warning", "info")}
    parts = [f"{n} {s}" for s, n in counts.items() if n]
    return f"{an.name}: {', '.join(parts) if parts else 'clean'}"


def _write_reports(results, out_dir: Path, save_ir: bool) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    seen: dict[str, int] = {}
    for an, findings in results:
        n = seen.get(an.name, 0)
        seen[an.name] = n + 1
        stem = an.name if n == 0 else f"{an.name}_{n}"
        page = write_html(build_report(an, findings), out_dir / f"{stem}.html")
        pages.append(page)
        if save_ir and an.launch is not None:
            an.launch.save(out_dir / "ir")
    if len(results) > 1:
        _write_index(results, pages, out_dir / "index.html")
    return pages


def _write_index(results, pages: list[Path], path: Path) -> None:
    rows = []
    for (an, findings), page in zip(results, pages):
        counts = {s: sum(1 for f in findings if f.severity == s)
                  for s in ("error", "warning", "info")}
        badges = " ".join(f'<span class="b {s}">{n} {s}</span>' for s, n in counts.items() if n) \
            or '<span class="b ok">clean</span>'
        grid = "x".join(map(str, an.launch.grid)) if an.launch and an.launch.grid else "-"
        rows.append(f'<tr><td><a href="{html.escape(page.name)}">{html.escape(an.name)}</a></td>'
                    f"<td>{an.module.num_warps}</td><td>{grid}</td><td>{badges}</td></tr>")
    path.write_text(f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>tritonlens</title>
<style>body{{background:#0f1218;color:#d8dee9;font:14px system-ui,sans-serif;padding:24px}}
a{{color:#5fb3f9}}table{{border-collapse:collapse;min-width:640px}}td,th{{padding:7px 12px;
border-bottom:1px solid #2a3142;text-align:left}}th{{color:#8b95a7;font-weight:500}}
.b{{border-radius:4px;padding:1px 7px;font-size:12px;margin-right:4px}}
.error{{background:#4a1f26;color:#ff5c6c}}.warning{{background:#4a3a17;color:#f5b947}}
.info{{background:#1d3550;color:#6cb6ff}}.ok{{background:#1a3d2b;color:#57c785}}</style></head>
<body><h2>tritonlens {__version__}: {len(results)} kernels</h2><table>
<tr><th>kernel</th><th>warps</th><th>grid</th><th>findings</th></tr>{''.join(rows)}</table>
</body></html>""", encoding="utf-8")


def cmd_run(args) -> int:
    kernels = capture_script(Path(args.script), args.script_args)
    if not kernels:
        print("tritonlens: the script did not launch any Triton kernel", file=sys.stderr)
        return 1
    results = _analyze_all(kernels)
    pages = _write_reports(results, Path(args.output), args.save_ir)
    for (an, findings), page in zip(results, pages):
        print(f"{_summary_line(an, findings):<60} -> {page}")
    return 0


def cmd_analyze(args) -> int:
    target = Path(args.target)
    if target.suffix == ".ttgir":
        kernels = [CapturedKernel.from_files(target, args.ptx, args.launch)]
    else:
        kernels = _load_kernels(target, [])
    results = _analyze_all(kernels)
    out = Path(args.output)
    if len(results) == 1 and out.suffix == ".html":
        an, findings = results[0]
        report = build_report(an, findings)
        page = write_html(report, out)
        if args.json:
            Path(args.json).write_text(json.dumps(report), encoding="utf-8")
        print(f"{_summary_line(an, findings):<60} -> {page}")
        return 0
    out_dir = out if out.suffix != ".html" else out.parent
    for (an, findings), page in zip(results, _write_reports(results, out_dir, False)):
        print(f"{_summary_line(an, findings):<60} -> {page}")
    return 0


def cmd_lint(args) -> int:
    results = _analyze_all(_load_kernels(Path(args.target), args.script_args))
    by_kernel: dict[str, list[Finding]] = {}
    for an, findings in results:
        limit = SEVERITY_ORDER[args.min_severity]
        keep = [f for f in findings if SEVERITY_ORDER[f.severity] <= limit]
        by_kernel.setdefault(an.name, []).extend(keep)
    if args.format == "json":
        text = to_json(by_kernel)
    elif args.format == "sarif":
        text = json.dumps(to_sarif(by_kernel), indent=2)
    else:
        blocks = [format_text(k, v) for k, v in by_kernel.items() if v]
        total = sum(len(v) for v in by_kernel.values())
        blocks.append(f"{total} finding(s) in {len(by_kernel)} kernel(s)")
        text = "\n".join(blocks)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)
    if args.fail_on:
        limit = SEVERITY_ORDER[args.fail_on]
        if any(SEVERITY_ORDER[f.severity] <= limit for v in by_kernel.values() for f in v):
            return 2
    return 0


def cmd_validate(args) -> int:
    from .validate import format_results, run_probes

    results = run_probes(quick=args.quick)
    print(format_results(results))
    failed = [r for r in results if not r.ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} probes match the GPU")
    return 1 if failed else 0


def cmd_device(args) -> int:
    from . import device

    spec = device.calibrate() if args.calibrate else device.detect()
    if spec is None:
        print("tritonlens: no CUDA device found", file=sys.stderr)
        return 1
    print(json.dumps(spec.to_json(), indent=2) if args.json else spec.describe())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tritonlens", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"tritonlens {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a script, capture its kernels and write HTML reports")
    run.add_argument("script")
    run.add_argument("-o", "--output", default="reports", help="output directory")
    run.add_argument("--save-ir", action="store_true", help="also save every IR stage")
    run.set_defaults(func=cmd_run)

    an = sub.add_parser("analyze", help="analyze a .ttgir file or a directory of captured IR")
    an.add_argument("target")
    an.add_argument("-o", "--output", default="report.html", help="HTML file or directory")
    an.add_argument("--ptx", help="PTX for vector-width cross-checks (default: <stem>.ptx)")
    an.add_argument("--launch",
                    help="launch JSON with grid/arguments (default: <stem>.launch.json)")
    an.add_argument("--json", help="also write the raw report JSON here")
    an.set_defaults(func=cmd_analyze)

    lint = sub.add_parser("lint", help="print findings for a script, .ttgir or IR directory")
    lint.add_argument("target")
    lint.add_argument("--format", choices=["text", "json", "sarif"], default="text")
    lint.add_argument("--min-severity", choices=["error", "warning", "info"], default="info")
    lint.add_argument("--fail-on", choices=["error", "warning", "info"],
                      help="exit with status 2 if a finding at this severity or worse exists")
    lint.add_argument("-o", "--output", help="write to a file instead of stdout")
    lint.set_defaults(func=cmd_lint)

    val = sub.add_parser("validate", help="check the layout models against this GPU")
    val.add_argument("--quick", action="store_true")
    val.set_defaults(func=cmd_validate)

    dev = sub.add_parser("device", help="show GPU limits and measured peaks")
    dev.add_argument("--calibrate", action="store_true",
                     help="measure bandwidth and matmul throughput and cache the result")
    dev.add_argument("--json", action="store_true")
    dev.set_defaults(func=cmd_device)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Everything after "--" is passed to the user's script.
    script_args: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, script_args = argv[:split], argv[split + 1:]
    args = build_parser().parse_args(argv)
    args.script_args = script_args
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
