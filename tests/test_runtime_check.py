from pathlib import Path

import pytest

from tritonlens.analysis.lint import run_lint
from tritonlens.analysis.runtime_check import check_args
from tritonlens.analyze import analyze_captured
from tritonlens.capture import ArgInfo, CapturedKernel

FIXTURES = Path(__file__).parent / "fixtures"


def tensor(name, ptr, shape=(1024,), strides=(1,), elem=4):
    return ArgInfo(name=name, kind="tensor", dtype="float32", shape=list(shape),
                   strides=list(strides), data_ptr=ptr, element_size=elem,
                   contiguous=list(strides)[-1] == 1)


def test_misaligned_pointer():
    findings = check_args([tensor("x", 0x1000), tensor("y", 0x1004)])
    assert len(findings) == 1
    assert findings[0].data == {"arg": "y", "misalignment": 4}
    assert findings[0].severity == "warning"


def test_row_pitch_and_noncontiguous():
    findings = check_args([
        tensor("a", 0, shape=(100, 781), strides=(781, 1)),
        tensor("b", 0, shape=(64, 64), strides=(1, 64)),
    ])
    messages = " ".join(f.message for f in findings)
    assert "row pitch of 781" in messages
    assert "not contiguous in its last dimension" in messages


def test_int_args():
    findings = check_args([ArgInfo("n", "int", 1000), ArgInfo("m", "int", 1024),
                           ArgInfo("one", "int", 1), ArgInfo("B", "constexpr", 7)])
    assert [f.data["arg"] for f in findings] == ["n"]


def test_tl006_only_reports_ints_that_reach_memory():
    an = analyze_captured(CapturedKernel.from_files(FIXTURES / "softmax_kernel.ttgir"))
    tl006 = [f for f in run_lint(an) if f.rule == "TL006"]
    reported = {f.data["arg"] for f in tl006}
    assert {"n_cols", "in_row_stride", "out_row_stride"} <= reported


@pytest.mark.gpu
def test_live_check_and_guard():
    import sys

    import torch

    sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
    import bad_access

    from tritonlens.analysis.runtime_check import check, guard

    base = torch.randn(3, 4097, device="cuda")
    x, y, out = base[0, 1:], base[1, 1:], base[2, 1:]
    findings = check(bad_access.misaligned_add_kernel, x, y, out, 4093, BLOCK=1024)
    assert {f.data["arg"] for f in findings if f.severity == "warning"} == {
        "x_ptr", "y_ptr", "out_ptr"}
    with pytest.raises(ValueError):
        with guard(raise_on="warning"):
            bad_access.misaligned_add_kernel[(4,)](x, y, out, 4093, BLOCK=1024)
