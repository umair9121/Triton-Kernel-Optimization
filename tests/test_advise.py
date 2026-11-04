from pathlib import Path

from tritonlens.analysis.advise import advise, format_advice
from tritonlens.analysis.budget import group_a_stripe, tensor_footprint
from tritonlens.analyze import analyze_captured
from tritonlens.capture import ArgInfo, CapturedKernel
from tritonlens.cli import main
from tritonlens.device import DeviceSpec

FIXTURES = Path(__file__).parent / "fixtures"
GPU = DeviceSpec(name="test 3050", arch=86, sm_count=20, total_memory=4 << 30,
                 l2_bytes=2 << 20)


def load(name: str, **meta) -> CapturedKernel:
    kernel = CapturedKernel.from_files(FIXTURES / f"{name}.ttgir")
    kernel.metadata.update(meta)
    return kernel


def test_tensor_footprint_dedupes_views():
    a = ArgInfo("a", "tensor", shape=[1024, 1024], element_size=2, data_ptr=100)
    view = ArgInfo("a_view", "tensor", shape=[512, 1024], element_size=2, data_ptr=100)
    b = ArgInfo("b", "tensor", shape=[1024, 1024], element_size=2, data_ptr=200)
    total, items = tensor_footprint([a, view, b])
    assert total == 2 * 1024 * 1024 * 2
    assert len(items) == 2


def test_group_stripe():
    assert group_a_stripe({"GROUP_M": 8, "BLOCK_M": 64, "K": 1024}, 2) == 1 << 20
    assert group_a_stripe({"GROUP_M": 1, "BLOCK_M": 64, "K": 1024}, 2) is None


def test_advise_includes_current_and_finer_tiles_for_coarse_grid():
    an = analyze_captured(load("matmul_kernel"), GPU)
    for arg in an.launch.args:
        if arg.name in ("M", "N"):
            arg.value = 640
        if arg.name in ("BLOCK_M", "BLOCK_N"):
            arg.value = 128
        if arg.name == "BLOCK_K":
            arg.value = 64
    an.launch.grid = [25, 1, 1]
    an.metadata["num_warps"] = 8
    an.metadata["num_stages"] = 3
    an._perf = None
    suggestions = advise(an, GPU)
    assert suggestions and suggestions[0].current
    assert suggestions[0].programs == 25
    labels = [s.label for s in suggestions]
    assert any("64x64" in lab and not s.current for s, lab in zip(suggestions, labels))
    finer = next(s for s in suggestions if "64x64" in s.label and not s.current)
    assert finer.sm_balance > suggestions[0].sm_balance


def test_advise_cli(capsys):
    target = str(FIXTURES / "matmul_kernel.ttgir")
    assert main(["advise", target]) == 0
    out = capsys.readouterr().out
    assert "matmul_kernel" in out
    assert "current" in out


def test_format_advice_empty():
    assert "nothing to retile" in format_advice("k", [])
