from tritonlens.ir import parse_module
from tritonlens.layouts.shared import SharedLayout, bank_of


def test_unswizzled_row_major_offsets():
    layout = SharedLayout(vec=1, per_phase=1, max_phase=1, order=(1, 0))
    assert layout.offset((3, 5), [32, 32]) == 3 * 32 + 5
    # 32 fp32 columns = 128 bytes per row: every row starts in bank 0.
    assert {layout.bank((r, 0), [32, 32], 4) for r in range(32)} == {0}


def test_swizzle_rotates_chunks():
    # Triton's choice for fp16 64-wide B tiles in the matmul fixture.
    layout = SharedLayout(vec=8, per_phase=1, max_phase=8, order=(1, 0))
    for row in range(8):
        assert layout.physical_col(row, 0, 64) == 8 * row
    # Column 0 of rows 0..7 lands in 8 distinct 16-byte bank groups.
    groups = {layout.bank((r, 0), [32, 64], 2) // 4 for r in range(8)}
    assert groups == set(range(8))


def test_swizzle_is_a_permutation_of_each_row():
    layout = SharedLayout(vec=8, per_phase=2, max_phase=4, order=(1, 0))
    for row in range(16):
        cols = sorted(layout.physical_col(row, c, 32) for c in range(32))
        assert cols == list(range(32))


def test_column_major_order():
    layout = SharedLayout(vec=1, per_phase=1, max_phase=1, order=(0, 1))
    # order (0, 1): dim 0 is contiguous.
    assert layout.offset((5, 2), [16, 8]) == 2 * 16 + 5


def test_multibuffer_leading_dim():
    layout = SharedLayout(vec=8, per_phase=1, max_phase=8, order=(1, 0))
    single = layout.offset((3, 9), [32, 64])
    assert layout.offset((1, 3, 9), [2, 32, 64]) == 32 * 64 + single


def test_bank_of():
    assert bank_of(0) == 0
    assert bank_of(4) == 1
    assert bank_of(128) == 0
    assert bank_of(132) == 1


def test_fixture_shared_layouts(fixture_text):
    mod = parse_module(fixture_text("matmul_kernel.ttgir"))
    layout = SharedLayout.from_params(mod.layouts["shared"].params)
    assert (layout.vec, layout.per_phase, layout.max_phase) == (8, 2, 4)
    banks = layout.bank_map([64, 32], 2)
    assert len(banks) == 64 and len(banks[0]) == 32
