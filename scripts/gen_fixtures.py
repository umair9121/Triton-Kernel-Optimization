"""Regenerate the IR fixtures in tests/fixtures from the example kernels (needs a GPU).

Absolute source paths inside ``loc(...)`` records are rewritten relative to the repository
root so fixtures are machine independent.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tritonlens.capture import capture_script  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
EXAMPLES = ["vector_add", "matmul", "softmax", "layernorm", "transpose", "bad_access"]


def sanitize(text: str) -> str:
    for variant in (str(ROOT), str(ROOT).replace("\\", "\\\\"), ROOT.as_posix()):
        for sep in ("\\\\", "\\", "/"):
            text = text.replace(variant + sep, "")
    return text


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for example in EXAMPLES:
        for kernel in capture_script(ROOT / "examples" / f"{example}.py"):
            for stage in ("ttgir", "ptx"):
                if stage in kernel.ir:
                    out = FIXTURES / f"{kernel.name}.{stage}"
                    out.write_text(sanitize(kernel.ir[stage]), encoding="utf-8")
            launch = kernel.to_json()
            for arg in launch["args"]:
                arg.pop("data_ptr_raw", None)
            (FIXTURES / f"{kernel.name}.launch.json").write_text(
                json.dumps(launch, indent=2), encoding="utf-8"
            )
            print(f"wrote {kernel.name}")


if __name__ == "__main__":
    main()
