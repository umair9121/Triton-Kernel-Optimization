import json
from pathlib import Path

from tritonlens.cli import main

FIXTURES = Path(__file__).parent / "fixtures"


def test_lint_text_min_severity(capsys):
    target = str(FIXTURES / "misaligned_add_kernel.ttgir")
    assert main(["lint", target, "--min-severity", "warning"]) == 0
    out = capsys.readouterr().out
    assert "TL006" in out
    assert "TL001" not in out


def test_lint_fail_on(tmp_path):
    target = str(FIXTURES / "strided_gather_kernel.ttgir")
    assert main(["lint", target, "--fail-on", "error", "-o", str(tmp_path / "x.txt")]) == 2
    assert main(["lint", str(FIXTURES / "add_kernel.ttgir"), "--fail-on", "warning",
                 "-o", str(tmp_path / "y.txt")]) == 0


def test_lint_sarif(tmp_path):
    out = tmp_path / "out.sarif"
    main(["lint", str(FIXTURES / "misaligned_add_kernel.ttgir"), "--format", "sarif",
          "-o", str(out)])
    sarif = json.loads(out.read_text())
    assert sarif["version"] == "2.1.0"
    results = sarif["runs"][0]["results"]
    assert {r["ruleId"] for r in results} >= {"TL001", "TL006"}
    located = [r for r in results if "locations" in r]
    region = located[0]["locations"][0]["physicalLocation"]["region"]
    assert region["startLine"] == 20


def test_analyze_writes_html(tmp_path, capsys):
    out = tmp_path / "r.html"
    assert main(["analyze", str(FIXTURES / "matmul_kernel.ttgir"), "-o", str(out),
                 "--json", str(tmp_path / "r.json")]) == 0
    assert out.exists() and out.stat().st_size > 10_000
    assert json.loads((tmp_path / "r.json").read_text())["kernel"]["name"] == "matmul_kernel"
