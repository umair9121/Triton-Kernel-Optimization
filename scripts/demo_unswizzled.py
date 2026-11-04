"""Write a report for the matmul fixture with swizzling disabled, to show bank conflicts.

Usage: python scripts/demo_unswizzled.py [out.html]
"""

import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tritonlens.analyze import analyze_captured  # noqa: E402
from tritonlens.capture import CapturedKernel  # noqa: E402
from tritonlens.report import build_report, write_html  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "reports" / "matmul_unswizzled.html")
    text = (FIXTURES / "matmul_kernel.ttgir").read_text(encoding="utf-8")
    text = re.sub(r"perPhase = \d+, maxPhase = \d+", "perPhase = 1, maxPhase = 1", text)
    with tempfile.TemporaryDirectory() as tmp:
        ttgir = Path(tmp) / "matmul_kernel.ttgir"
        ttgir.write_text(text, encoding="utf-8")
        for ext in (".ptx", ".launch.json"):
            shutil.copy(FIXTURES / f"matmul_kernel{ext}", Path(tmp) / f"matmul_kernel{ext}")
        an = analyze_captured(CapturedKernel.from_files(ttgir))
        print(write_html(build_report(an, source_roots=[ROOT]), out))


if __name__ == "__main__":
    main()
