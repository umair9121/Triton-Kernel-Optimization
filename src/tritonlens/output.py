"""Text, JSON and SARIF renderings of lint findings."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from . import __version__
from .analysis.lint import RULES, Finding

_COLORS = {"error": "\033[31m", "warning": "\033[33m", "info": "\033[36m"}
_RESET = "\033[0m"
_SARIF_LEVEL = {"error": "error", "warning": "warning", "info": "note"}


def _use_color(stream) -> bool:
    return hasattr(stream, "isatty") and stream.isatty() and os.environ.get("NO_COLOR") is None


def format_text(kernel: str, findings: list[Finding], stream=None) -> str:
    color = _use_color(stream or sys.stdout)
    lines = []
    for f in findings:
        where = f.location or kernel
        sev = f.severity
        tag = f"{_COLORS[sev]}{sev}{_RESET}" if color else sev
        lines.append(f"{where}: {tag} {f.rule} [{kernel}] {f.message}")
        if f.fix:
            lines.append(f"    fix: {f.fix}")
    return "\n".join(lines)


def to_json(results: dict[str, list[Finding]]) -> str:
    return json.dumps({k: [f.to_json() for f in v] for k, v in results.items()}, indent=2)


def to_sarif(results: dict[str, list[Finding]]) -> dict:
    """SARIF 2.1.0, suitable for GitHub code scanning uploads."""
    rules = [{"id": rid, "name": title.replace(" ", ""), "shortDescription": {"text": title},
              "helpUri": "https://github.com/umair9121/Triton-Kernel-Optimization#lint-rules"}
             for rid, title in RULES.items()]
    sarif_results = []
    for kernel, findings in results.items():
        for f in findings:
            res = {
                "ruleId": f.rule,
                "level": _SARIF_LEVEL.get(f.severity, "note"),
                "message": {"text": f"[{kernel}] {f.message}" + (f" Fix: {f.fix}" if f.fix else "")},
                "properties": {"kernel": kernel, **{k: v for k, v in f.data.items()
                                                    if isinstance(v, (int, float, str))}},
            }
            loc = f.op.loc if f.op is not None else None
            if loc is not None:
                res["locations"] = [{"physicalLocation": {
                    "artifactLocation": {"uri": Path(loc.file).as_posix()},
                    "region": {"startLine": loc.line, "startColumn": max(1, loc.col)},
                }}]
            sarif_results.append(res)
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {"name": "tritonlens", "version": __version__,
                                "informationUri": "https://github.com/umair9121/Triton-Kernel-Optimization",
                                "rules": rules}},
            "results": sarif_results,
        }],
    }
