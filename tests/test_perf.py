from pathlib import Path

import pytest

from tritonlens.analysis.lint import run_lint
from tritonlens.analysis.perf import pipeline_buffer_bytes
from tritonlens.analyze import analyze_captured
from tritonlens.capture import CapturedKernel
from tritonlens.device import DeviceSpec

FIXTURES = Path(__file__).parent / "fixtures"
GPU = DeviceSpec(name="test 3050", arch=86, sm_count=20, l2_bytes=2 << 20,
                 mem_clock_khz=6001000, mem_bus_width=128,
                 measured={"gbps": 170.0, "fp16_tflops": 14.0, "fp32_tflops": 4.0})


def load(name: str, **meta) -> CapturedKernel:
    kernel = CapturedKernel.from_files(FIXTURES / f"{name}.ttgir")
    kernel.metadata.update(meta)
    return kernel


def rules(an) -> set[str]:
    return {f.rule for f in run_lint(an)}


def test_matmul_roofline_counts_loop_trips():
    an = analyze_captured(load("matmul_kernel"), GPU)
    roof = an.perf.roofline
    assert roof.flops == 2 * 1024 ** 3  # 256 programs x 32 K-steps x two 64x64x16 dots
    assert roof.dot_dtype == "fp16"
    assert roof.dram_bytes == 3 * 1024 * 1024 * 2  # each matrix once (perfect L2 reuse)
    # A and B tiles are re-read by every program: 64x32 + 32x64 halves x 32 steps x 256.
    assert roof.requested_bytes == 256 * 32 * 2 * (64 * 32 * 2) + 1024 * 1024 * 2
    assert roof.bound == "compute"
    assert roof.ridge == pytest.approx(14e12 / 170e9)


def test_matmul_occupancy_and_balance():
    perf = analyze_captured(load("matmul_kernel"), GPU).perf
    assert perf.occupancy.blocks_per_sm == 5 and perf.occupancy.limiter == "registers"
    assert perf.regs_needed_for_next_block == 80
    assert perf.waves.sm_balance == pytest.approx(256 / 260)


def test_pipeline_buffers():
    an = analyze_captured(load("matmul_kernel"), GPU)
    assert pipeline_buffer_bytes(an.func, 3) == 64 * 32 * 2 + 32 * 64 * 2
    an = analyze_captured(load("transpose_tiled_kernel"), GPU)
    assert pipeline_buffer_bytes(an.func, 3) == 0


def test_tl008_imbalanced_grid():
    an = analyze_captured(load("matmul_kernel", n_regs=172), GPU)
    an.launch.grid = [25, 1, 1]
    an.metadata["num_warps"] = 8
    findings = [f for f in run_lint(an) if f.rule == "TL008"]
    assert findings and findings[0].severity == "warning"
    assert findings[0].data["balance"] == pytest.approx(25 / 40)
    assert "TL007" in rules(an)


def test_tl008_idle_sms():
    an = analyze_captured(load("softmax_kernel"), GPU)
    an.launch.grid = [8, 1, 1]
    found = [f for f in run_lint(an) if f.rule == "TL008"]
    assert found and "12 SMs stay idle" in found[0].message


def test_tl009_needs_pipelining_and_low_occupancy():
    an = analyze_captured(load("matmul_kernel", shared=98304, n_regs=96), GPU)
    an.metadata["num_stages"] = 3
    # 16 KiB of pipeline buffers cannot explain 96 KiB: the rest is scratch that stays.
    assert "TL009" not in rules(an)
    an = analyze_captured(load("transpose_tiled_kernel"), GPU)
    assert "TL009" not in rules(an)


def test_tl007_spills():
    an = analyze_captured(load("matmul_kernel", n_spills=64), GPU)
    found = [f for f in run_lint(an) if f.rule == "TL007"]
    assert found and "64 bytes of local memory" in found[0].message
