from tritonlens.ir import MemDescType, TensorType, parse_encoding, parse_module, parse_type
from tritonlens.ir.attrs import parse_attr_dict


def test_parse_blocked_encoding_inline():
    enc = parse_encoding(
        "#triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], "
        "warpsPerCTA = [4, 1], order = [1, 0]}>"
    )
    assert enc.kind == "blocked"
    assert enc.params["sizePerThread"] == [1, 8]
    assert enc.params["order"] == [1, 0]


def test_new_dialect_prefix_is_accepted():
    enc = parse_encoding(
        "#ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>"
    )
    assert enc.kind == "shared"
    assert enc.params["maxPhase"] == 8


def test_slice_resolves_parent_alias():
    parent = parse_encoding("#triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], "
                            "warpsPerCTA = [4], order = [0]}>")
    parent.alias = "blocked"
    enc = parse_encoding("#triton_gpu.slice<{dim = 1, parent = #blocked}>", {"blocked": parent})
    assert enc.kind == "slice"
    assert enc.params["parent"] is parent


def test_attr_dict_typed_values():
    attrs = parse_attr_dict("{end = 1024 : i32, start = 0 : i32, isVolatile = false}")
    assert attrs == {"end": 1024, "start": 0, "isVolatile": False}


def test_parse_pointer_tensor_type():
    t = parse_type("tensor<64x32x!tt.ptr<f16>, #blocked>")
    assert isinstance(t, TensorType)
    assert t.shape == [64, 32]
    assert t.is_pointer_tensor
    assert t.element_bytes == 2


def test_parse_memdesc_type():
    t = parse_type("!tt.memdesc<2x64x32xf16, #shared, #triton_gpu.shared_memory, mutable>")
    assert isinstance(t, MemDescType)
    assert t.shape == [2, 64, 32]
    assert t.mutable
    assert t.element_bytes == 2


def test_vector_add_module(fixture_text):
    mod = parse_module(fixture_text("add_kernel.ttgir"))
    assert mod.num_warps == 4
    assert mod.threads_per_warp == 32
    assert mod.target == "cuda:86"
    func = mod.func
    assert func.name == "add_kernel"
    assert [a.attrs.get("tt.divisibility") for a in func.args] == [16, 16, 16, 16]
    loads = func.find("load")
    assert len(loads) == 2
    assert loads[0].result_type.shape == [1024]
    assert loads[0].result_type.element_name == "f32"
    assert loads[0].loc.line == 13
    store = func.find("store")[0]
    assert [o.name for o in store.operands] == ["%14", "%15", "%6"]


def test_make_range_and_program_id_attrs(fixture_text):
    func = parse_module(fixture_text("add_kernel.ttgir")).func
    rng = func.find("make_range")[0]
    assert rng.attrs["start"] == 0 and rng.attrs["end"] == 1024
    pid = func.find("get_program_id")[0]
    assert pid.words == ["x"]


def test_matmul_shared_and_mma_layouts(fixture_text):
    mod = parse_module(fixture_text("matmul_kernel.ttgir"))
    layouts = mod.layouts
    assert layouts["mma"].kind == "nvidia_mma"
    assert layouts["mma"].params["warpsPerCTA"] == [2, 2]
    assert layouts["shared"].params == {
        "vec": 8, "perPhase": 2, "maxPhase": 4, "order": [1, 0], "hasLeadingOffset": False
    }
    allocs = mod.func.find("local_alloc")
    assert all(isinstance(a.result_type, MemDescType) for a in allocs)
    assert allocs[0].result_type.shape == [2, 64, 32]
    loads = mod.func.find("local_load")
    enc = loads[0].result_type.encoding
    assert enc.kind == "dot_op"
    assert enc.params["opIdx"] == 0
    assert enc.params["parent"].kind == "nvidia_mma"


def test_matmul_loop_carried_values(fixture_text):
    func = parse_module(fixture_text("matmul_kernel.ttgir")).func
    loop = func.find("for")[0]
    assert len(loop.results) == 10
    assert loop.results[0].type.shape == [64, 64]
    carried = {a.name: a for a in loop.block_args}
    assert carried["%arg10"].init.name == "%cst_2"
    assert carried["%arg10"].yielded.owner.is_("dot")
    dots = func.find("dot")
    assert all(d.parent is loop for d in dots)
    # The truncf after the loop consumes the first loop result.
    trunc = func.find("truncf")[0]
    assert trunc.operands[0] is loop.results[0]


def test_reduce_generic_form(fixture_text):
    func = parse_module(fixture_text("softmax_kernel.ttgir")).func
    reduce_ops = func.find("reduce")
    assert len(reduce_ops) == 2
    assert reduce_ops[0].attrs["axis"] == 0
    assert reduce_ops[0].result_type.text == "f32"
    assert reduce_ops[0].loc is not None


def test_scoped_values_in_sibling_loops(fixture_text):
    func = parse_module(fixture_text("layernorm_kernel.ttgir")).func
    loads = func.find("load")
    assert len(loads) == 5
    # Every load's pointer operand must resolve inside its own loop body.
    for load in loads:
        assert load.operands[0].owner.parent is load.parent


def test_callsite_locations_point_at_user_code(fixture_text):
    mod = parse_module(fixture_text("matmul_kernel.ttgir"))
    cdiv = [op for op in mod.func.ops if op.is_("divsi")][0]
    assert cdiv.loc.file.endswith("matmul.py")
