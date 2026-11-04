"""Report generation: JSON serialization and the self-contained HTML viewer."""

from __future__ import annotations

import json
from pathlib import Path

from .builder import build_report

ASSETS = Path(__file__).parent / "assets"


def render_html(report: dict) -> str:
    """Embed the report JSON, CSS and JS into one offline HTML document."""
    template = (ASSETS / "template.html").read_text(encoding="utf-8")
    css = (ASSETS / "viewer.css").read_text(encoding="utf-8")
    js = (ASSETS / "viewer.js").read_text(encoding="utf-8")
    data = json.dumps(report, separators=(",", ":")).replace("</", "<\\/")
    title = report.get("kernel", {}).get("name", "kernel")
    return (template.replace("__TITLE__", title).replace("__CSS__", css)
            .replace("__DATA__", data).replace("__JS__", js))


def write_html(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(report), encoding="utf-8")
    return path


__all__ = ["build_report", "render_html", "write_html"]
