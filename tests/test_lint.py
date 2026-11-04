from pathlib import Path

import pytest

from tritonlens.analysis.axis import AxisAnalysis
from tritonlens.analysis.lint import run_lint
from tritonlens.analyze import analyze_captured
from tritonlens.capture import CapturedKernel
from tritonlens.ir import parse_module

FIXTURES = Path(__file__).parent / "fixtures"


def analysis(name):
    return analyze_captured(CapturedKernel.from_files(FIXTURES / f"{name}.ttgir"))


def rules(findings, rule):
    return [f for f in findings if f.rule == rule]


def test_axis_info_vector_add(fixture_text):
    func = parse_module(fixture_text("add_kernel.ttgir")).func
    ax = AxisAnalysis(func)
    load = func.find("load")[0]
    ptr, mask = ax[load.operands[0]], ax[load.operands[1]]
    assert ptr.contiguity == [1024]
    assert ptr.divisibility == [16]
    assert ptr.stride == [1]
    assert mask.constancy == [16]


def test_axis_info_matmul_runtime_stride(fixture_text):
    func = parse_module(fixture_text("matmul_kernel.ttgir")).func
    ax = AxisAnalysis(func)
    copy = func.find("async_copy_global_to_local")[0]
    info = ax[copy.operands[0]]
    assert info.stride == [(1, "%arg6"), 1]
    assert info.contiguity[1] == 32
    assert info.origin[0] is not None  # tile row origin = pid_m * BLOCK_M


def test_vector_widths_match_ptx():
    for name in ["add_kernel", "misaligned_add_kernel", "softmax_kernel", "layernorm_kernel",
                 "odd_block_kernel", "strided_gather_kernel", "matmul_kernel"]:
        an = analysis(name)
        for acc in an.global_accesses:
            if acc.ptx_bytes is not None:
                assert acc.vec_bytes == acc.ptx_bytes, (name, acc.op)


def test_clean_kernels_have_no_warnings():
    for name in ["add_kernel", "matmul_kernel"]:
        findings = run_lint(analysis(name))
        assert not [f for f in findings if f.severity in ("warning", "error")], name


def test_tl001_misaligned_pointer_and_mask():
    findings = rules(run_lint(analysis("misaligned_add_kernel")), "TL001")
    assert len(findings) == 3
    msg = findings[0].message
    assert "`x_ptr`" in msg and "not known to be 16-byte aligned" in msg
    assert "`n`" in msg and "PTX confirms: ld.global.b32" in msg


def test_tl001_names_the_unaligned_row_stride():
    findings = rules(run_lint(analysis("softmax_kernel")), "TL001")
    assert findings
    assert "`in_row_stride`" in findings[0].message
    assert "`n_cols`" in findings[0].message


def test_tl001_layout_limited():
    findings = rules(run_lint(analysis("layernorm_kernel")), "TL001")
    assert findings and all(f.severity == "info" for f in findings)
    assert "contiguous elements" in findings[0].message


def test_tl002_strided_gather():
    findings = rules(run_lint(analysis("strided_gather_kernel")), "TL002")
    assert len(findings) == 1
    assert findings[0].data["efficiency"] == pytest.approx(0.125)


def test_tl004_transpose_round_trip():
    findings = rules(run_lint(analysis("transpose_tiled_kernel")), "TL004")
    assert len(findings) == 1
    assert "shared memory" in findings[0].message


def test_tl005_mask_utilization():
    findings = rules(run_lint(analysis("softmax_kernel")), "TL005")
    assert findings and "76%" in findings[0].message
