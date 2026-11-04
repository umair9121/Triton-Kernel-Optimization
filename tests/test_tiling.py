from pathlib import Path

from tritonlens.analysis.coalescing import base_pointer_arg
from tritonlens.analysis.tiling import _main_store, program_tiling
from tritonlens.analyze import analyze_captured
from tritonlens.capture import CapturedKernel

FIXTURES = Path(__file__).parent / "fixtures"


def tiling_for(name):
    kernel = CapturedKernel.from_files(FIXTURES / f"{name}.ttgir")
    an = analyze_captured(kernel)
    store = _main_store(an.func)
    py = an.arg_names.get(base_pointer_arg(store.operands[0]).name)
    tensor = next(vars(a) for a in kernel.args if a.name == py)
    return program_tiling(an.func, an.arg_values, kernel.grid, tensor, py)


def test_vector_add_tiles():
    t = tiling_for("add_kernel")
    assert t.matrix_shape == [98432]
    assert t.tile_shape == [1024]
    assert [p["origin"] for p in t.programs[:3]] == [[0], [1024], [2048]]


def test_matmul_grouped_ordering():
    t = tiling_for("matmul_kernel")
    assert t.matrix_shape == [1024, 1024]
    assert t.tile_shape == [64, 64]
    origins = [tuple(p["origin"]) for p in t.programs]
    # GROUP_M = 8: the first 8 programs walk down a column of tiles before moving right.
    assert origins[:9] == [(64 * i, 0) for i in range(8)] + [(0, 64)]
    assert len(set(origins)) == 256


def test_transpose_writes_transposed_tiles():
    t = tiling_for("transpose_tiled_kernel")
    assert t.matrix_shape == [1536, 2048]
    assert t.tile_shape == [64, 64]
    # pid (x=1, y=0) handles input rows 64..127, i.e. output columns 64..127.
    assert t.programs[1]["origin"] == [0, 64]


def test_softmax_one_row_per_program():
    t = tiling_for("softmax_kernel")
    assert t.tile_shape == [1, 1024]
    assert [p["origin"] for p in t.programs[:3]] == [[0, 0], [1, 0], [2, 0]]
