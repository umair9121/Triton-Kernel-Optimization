import pytest

from tritonlens.analysis.banks import (
    analyze_function,
    simulate_register_access,
    wavefronts,
)
from tritonlens.ir import parse_module
from tritonlens.layouts import BlockedLayout, PaddedScratchLayout, SharedLayout


def test_wavefronts_basic_patterns():
    # Consecutive words: one wavefront.
    assert wavefronts([4 * i for i in range(32)], 4) == (1, 1)
    # Same word everywhere: broadcast, one wavefront.
    assert wavefronts([0] * 32, 4) == (1, 1)
    # Stride of 32 words: every lane hits bank 0 with a distinct word.
    assert wavefronts([128 * i for i in range(32)], 4) == (32, 1)
    # Stride of 2 words: 2-way conflict.
    assert wavefronts([8 * i for i in range(32)], 4) == (2, 1)


def test_wavefronts_vector_widths():
    # 16 B per lane, contiguous: 512 B = 4 quarter-warp wavefronts, conflict free.
    assert wavefronts([16 * i for i in range(32)], 16) == (4, 4)
    # 8 B per lane, contiguous: two half-warps.
    assert wavefronts([8 * i for i in range(32)], 8) == (2, 2)
    # 16 B rows with a 128 B pitch (ldmatrix on an unswizzled 64-wide fp16 tile).
    assert wavefronts([128 * i for i in range(32)], 16) == (32, 4)


def _column_reader():
    # One warp, lane l owns row l: reading column j touches (l, j) for all lanes.
    return BlockedLayout([1, 1], [32, 1], [1, 1], [0, 1])


def _degree(warps):
    wf = sum(i.wavefronts for w in warps for i in w)
    ideal = sum(i.ideal for w in warps for i in w)
    return wf / ideal


@pytest.mark.parametrize(
    "layout,expected",
    [
        (SharedLayout(vec=1, per_phase=1, max_phase=1, order=(1, 0)), 32.0),
        (SharedLayout(vec=1, per_phase=1, max_phase=32, order=(1, 0)), 1.0),
        (SharedLayout(vec=4, per_phase=1, max_phase=8, order=(1, 0)), 4.0),
        (PaddedScratchLayout([32, 32], [32, 33], [1, 0]), 1.0),
    ],
)
def test_transpose_32x32_fp32(layout, expected):
    shape = [32, 32]

    def addr(coord):
        return layout.byte_offset(coord, shape, 4)

    warps = simulate_register_access(_column_reader(), shape, addr, 4, 1)
    assert len(warps[0]) == 32
    assert _degree(warps) == expected


def test_matmul_fixture_is_conflict_free(fixture_text):
    mod = parse_module(fixture_text("matmul_kernel.ttgir"))
    accesses = analyze_function(mod.func, mod.num_warps)
    mechs = {a.mechanism for a in accesses}
    assert "ldmatrix.x4" in mechs and "ldmatrix.x4.trans" in mechs
    assert any(m.startswith("cp.async (16B)") for m in mechs)
    for access in accesses:
        if not access.estimated:
            assert access.degree == 1.0, access.op
    loads = [a for a in accesses if a.mechanism.startswith("ldmatrix")]
    # 64x16 A tile on a 2x2 warp grid: 32x16 per warp = 8 matrices = 2 ldmatrix.x4.
    assert all(a.instructions_per_warp == 2 for a in loads)


def test_matmul_subview_offsets_resolved(fixture_text):
    mod = parse_module(fixture_text("matmul_kernel.ttgir"))
    accesses = {a.op.id: a for a in analyze_function(mod.func, mod.num_warps)}
    inner = [a for a in accesses.values() if a.op.depth == 1 and a.op.is_("local_load")]
    assert sorted(a.buffer.offsets for a in inner)[-1] in ([0, 0, 16], [0, 16, 0])


def test_unswizzled_matmul_ldmatrix_conflicts(fixture_text):
    text = fixture_text("matmul_kernel.ttgir")
    text = text.replace("perPhase = 2, maxPhase = 4", "perPhase = 1, maxPhase = 1")
    text = text.replace("perPhase = 1, maxPhase = 8", "perPhase = 1, maxPhase = 1")
    mod = parse_module(text)
    loads = [a for a in analyze_function(mod.func, mod.num_warps)
             if a.mechanism.startswith("ldmatrix")]
    # A: 32 fp16 = 64 B rows -> row m starts in 16 B group (4m + c) % 8, so 8 rows share
    # only 2 groups (4-way).
    # B: 64 fp16 = 128 B rows -> all 8 rows hit the same bank group (8-way).
    degrees = {a.mechanism: a.degree for a in loads}
    assert degrees["ldmatrix.x4"] == 4.0
    assert degrees["ldmatrix.x4.trans"] == 8.0


def test_transpose_convert_layout_estimate(fixture_text):
    mod = parse_module(fixture_text("transpose_tiled_kernel.ttgir"))
    accesses = analyze_function(mod.func, mod.num_warps)
    assert len(accesses) == 1
    cvt = accesses[0]
    assert cvt.estimated
    assert cvt.op.is_("convert_layout")
    assert "padded to [64, 68]" in cvt.notes[0]
