from dataclasses import dataclass, field

from tritonlens.autotune import device_pruner, evaluate, matmul_programs, matmul_smem
from tritonlens.device import DeviceSpec

GPU = DeviceSpec(name="test 3050", arch=86, sm_count=20, l2_bytes=2 << 20,
                 total_memory=4 << 30)


@dataclass
class Cfg:  # the attributes of triton.Config the pruner reads
    kwargs: dict
    num_warps: int = 4
    num_stages: int = 3
    extra: dict = field(default_factory=dict)


class Half:
    def element_size(self):
        return 2


def cfg(bm, bn, bk, stages=3):
    return Cfg({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_stages=stages)


def test_smem_estimate_matches_triton():
    # Compiled on sm_86: 64x64x64 s5 -> 65536 B, 128x128x32 s4 -> 49152 B.
    assert matmul_smem(cfg(64, 64, 64, 5), {"a": Half()}) == 65536
    assert matmul_smem(cfg(128, 128, 32, 4), {"a": Half()}) == 49152


def test_drops_configs_that_cannot_fit():
    args = {"a": Half(), "M": 2048, "N": 2048}
    big = cfg(256, 256, 64, 4)  # 3 * 32 KiB * 2 = 192 KiB
    ok = cfg(64, 64, 32)
    keep = device_pruner(GPU)([big, ok], args)
    assert keep == [ok]


def test_drops_imbalanced_grids_only_when_better_exists():
    args = {"a": Half(), "M": 640, "N": 640}
    coarse = cfg(128, 128, 32)  # 25 programs -> 62%
    fine = cfg(64, 64, 32)  # 100 programs -> 100%
    decisions = evaluate([coarse, fine], args, GPU)
    assert [d.keep for d in decisions] == [False, True]
    assert "62%" in decisions[0].reason
    # Alone, the coarse config is kept: pruning never empties the list.
    assert device_pruner(GPU)([coarse], args) == [coarse]


def test_unknown_configs_are_kept():
    odd = Cfg({"TILE": 128})
    assert matmul_smem(odd, {}) is None and matmul_programs(odd, {"M": 1}) is None
    assert device_pruner(GPU)([odd], {}) == [odd]


def test_drops_group_m_that_misses_l2_when_a_fit_exists():
    args = {"a": Half(), "M": 4096, "N": 4096, "K": 4096}
    fat = Cfg({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8})   # 4 MiB stripe
    slim = Cfg({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 2})  # 1 MiB stripe
    decisions = evaluate([fat, slim], args, GPU)
    assert [d.keep for d in decisions] == [False, True]
    assert "L2" in decisions[0].reason


def test_prunes_datacenter_sweep():
    import itertools

    args = {"a": Half(), "M": 640, "N": 640, "K": 640}
    sweep = [Cfg({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8},
                 num_stages=s, num_warps=w)
             for bm, bn, bk, s, w in itertools.product(
                 (64, 128, 256), (64, 128, 256), (32, 64), (3, 4, 5), (4, 8))]
    keep = device_pruner(GPU)(sweep, args)
    assert len(sweep) == 108
    assert 10 <= len(keep) < 70  # drops 256x256, 25-program 128x128, deep 256-wide pipelines
    assert not any(c.kwargs["BLOCK_M"] == 256 and c.kwargs["BLOCK_N"] == 256 for c in keep)
