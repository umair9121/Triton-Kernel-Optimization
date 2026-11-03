from tritonlens.ir import parse_module
from tritonlens.layouts import BlockedLayout, from_encoding


def test_vector_add_layout_1d():
    layout = BlockedLayout([4], [32], [4], [0])
    assert layout.cta_tile() == [512]
    # 1024 elements = 2 repetitions of the 512-element CTA tile.
    assert layout.thread_elements([1024], 1, 3) == [
        (140,), (141,), (142,), (143,), (652,), (653,), (654,), (655,)
    ]
    own = layout.ownership([1024])
    assert own.covers_all()
    assert own.replication == 1
    assert own.regs_per_thread == 8


def test_matmul_a_operand_load_layout():
    # #blocked = sizePerThread [1, 8], threadsPerWarp [8, 4], warpsPerCTA [4, 1], order [1, 0]
    layout = BlockedLayout([1, 8], [8, 4], [4, 1], [1, 0])
    assert layout.cta_tile() == [32, 32]
    elems = layout.thread_elements([64, 32], warp=1, lane=5)
    # lane 5 -> (row lane 1, col lane 1); warp 1 -> rows 8..15 of the tile.
    assert elems[:8] == [(9, c) for c in range(8, 16)]
    assert elems[8:] == [(41, c) for c in range(8, 16)]
    assert layout.contiguous_elements([64, 32], dim=1) == 8


def test_column_major_order():
    layout = BlockedLayout([4, 1], [16, 2], [1, 4], [0, 1])
    elems = layout.thread_elements([64, 64], warp=0, lane=1)
    # order [0, 1]: lanes walk down dim 0 first.
    assert elems[:4] == [(4, 0), (5, 0), (6, 0), (7, 0)]
    assert layout.contiguous_elements([64, 64], dim=0) == 4
    assert layout.contiguous_elements([64, 64], dim=1) == 1


def test_small_tensor_is_broadcast():
    layout = BlockedLayout([1], [32], [4], [0])
    own = layout.ownership([32])
    # 128 threads hold a 32-element tensor: every element lives in 4 threads.
    assert own.replication == 4
    assert own.owners[5] == [(0, 5, 0), (1, 5, 0), (2, 5, 0), (3, 5, 0)]


def test_from_encoding_on_fixture(fixture_text):
    mod = parse_module(fixture_text("transpose_tiled_kernel.ttgir"))
    for enc in mod.layouts.values():
        layout = from_encoding(enc, mod.num_warps)
        own = layout.ownership([64, 64])
        assert own.covers_all()
        assert own.num_warps == 4 and own.threads_per_warp == 32
