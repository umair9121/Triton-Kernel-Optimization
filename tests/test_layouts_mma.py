from tritonlens.ir import parse_module
from tritonlens.layouts import (
    BlockedLayout,
    DotOperandLayout,
    MmaV2Layout,
    SliceLayout,
    from_encoding,
)


def test_mma_accumulator_fragment_matches_ptx_spec():
    # PTX ISA m16n8k16 f32 accumulator: c0,c1 at (g, 2t..2t+1); c2,c3 at (g+8, 2t..2t+1).
    mma = MmaV2Layout([1, 1])
    for lane in range(32):
        g, t = lane // 4, lane % 4
        assert mma.thread_elements([16, 8], 0, lane) == [
            (g, 2 * t), (g, 2 * t + 1), (g + 8, 2 * t), (g + 8, 2 * t + 1)
        ]


def test_mma_warps_are_row_major():
    mma = MmaV2Layout([2, 2])
    assert mma.thread_elements([32, 16], 1, 0)[0] == (0, 8)   # warp 1 -> right of warp 0
    assert mma.thread_elements([32, 16], 2, 0)[0] == (16, 0)  # warp 2 -> below warp 0
    own = mma.ownership([64, 64])
    assert own.covers_all() and own.replication == 1
    assert own.regs_per_thread == 32


def test_dot_operand_a_fragment_matches_ptx_spec():
    # PTX m16n8k16 f16 A fragment: a0..a7 rows g/g+8, cols 2t..2t+1 and 2t+8..2t+9.
    a = DotOperandLayout(0, MmaV2Layout([1, 1]), k_width=2)
    lane = 6  # g = 1, t = 2
    assert a.thread_elements([16, 16], 0, lane) == [
        (1, 4), (1, 5), (9, 4), (9, 5), (1, 12), (1, 13), (9, 12), (9, 13)
    ]


def test_dot_operand_b_fragment_matches_ptx_spec():
    # PTX m16n8k16 f16 B fragment: b0..b3 rows 2t..2t+1 and 2t+8..2t+9, col g.
    b = DotOperandLayout(1, MmaV2Layout([1, 1]), k_width=2)
    lane = 6
    assert b.thread_elements([16, 8], 0, lane) == [(4, 1), (5, 1), (12, 1), (13, 1)]


def test_dot_operands_line_up_with_accumulator_tiles():
    """Each warp must hold the A rows and B columns of the C tile it computes."""
    mma = MmaV2Layout([2, 2])
    a = DotOperandLayout(0, mma, 2)
    b = DotOperandLayout(1, mma, 2)
    M, N, K = 64, 64, 32
    for warp in range(4):
        c_rows, c_cols = set(), set()
        for lane in range(32):
            for r, c in mma.thread_elements([M, N], warp, lane):
                c_rows.add(r)
                c_cols.add(c)
        a_rows = {r for lane in range(32) for r, _ in a.thread_elements([M, K], warp, lane)}
        b_cols = {c for lane in range(32) for _, c in b.thread_elements([K, N], warp, lane)}
        assert a_rows == c_rows
        assert b_cols == c_cols


def test_dot_operand_a_broadcast_across_warp_columns():
    a = DotOperandLayout(0, MmaV2Layout([2, 2]), 2)
    own = a.ownership([32, 16])
    # Warps 0 and 1 share C's first row of tiles, so they hold identical A fragments.
    assert own.replication == 2
    assert a.thread_elements([32, 16], 0, 3) == a.thread_elements([32, 16], 1, 3)


def test_slice_layout_removes_dimension():
    parent = BlockedLayout([1, 4], [8, 4], [4, 1], [1, 0])
    rows = SliceLayout(1, parent)
    # Keeping dim 0: lanes 0-3 share a row, so each row value is held by 4 threads.
    own = rows.ownership([32])
    assert own.covers_all()
    assert own.replication == 4
    assert rows.thread_elements([32], 0, 5) == [(1,)]


def test_fixture_dot_operand_layouts(fixture_text):
    mod = parse_module(fixture_text("matmul_kernel.ttgir"))
    load = mod.func.find("local_load")[0]
    layout = from_encoding(load.result_type.encoding, mod.num_warps)
    assert isinstance(layout, DotOperandLayout)
    own = layout.ownership(load.result_type.shape)
    assert own.covers_all()
