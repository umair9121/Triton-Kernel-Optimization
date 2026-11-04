import pytest

from tritonlens.analysis.occupancy import occupancy, waves
from tritonlens.device import DeviceSpec, for_target, limits_for

RTX3050 = DeviceSpec(name="test 3050", arch=86, sm_count=20, mem_clock_khz=6001000,
                     mem_bus_width=128)


# (warps, regs, shared) -> blocks/SM reported by cuOccupancyMaxActiveBlocksPerMultiprocessor
# on an RTX 3050 Ti Laptop GPU (benchmarks/occupancy_check.py).
@pytest.mark.parametrize("warps,regs,shared,blocks,limiter", [
    (2, 72, 4096, 14, "registers"),
    (2, 72, 8192, 11, "shared memory"),
    (8, 40, 4096, 6, "warps"),
    (8, 108, 49152, 2, "registers"),
    (4, 254, 49152, 2, "registers"),
    (8, 166, 65536, 1, "registers"),
    (8, 176, 98304, 1, "registers"),
    (1, 71, 0, 16, "block slots"),
])
def test_matches_driver(warps, regs, shared, blocks, limiter):
    occ = occupancy(RTX3050, warps, regs, shared)
    assert occ.blocks_per_sm == blocks
    assert occ.limiter == limiter


def test_does_not_fit():
    occ = occupancy(RTX3050, 4, 0, 131072)
    assert not occ.fits and occ.blocks_per_sm == 0


def test_waves_tail():
    occ = occupancy(RTX3050, 8, 166, 65536)  # 1 block per SM
    w = waves(RTX3050, occ, [30, 1, 1])
    assert w.full_waves == 1 and w.tail_blocks == 10
    assert w.efficiency == pytest.approx(30 / 40)
    w = waves(RTX3050, occ, [8, 1, 1])
    assert w.busy_sms == 8 and w.efficiency == pytest.approx(0.4)


def test_device_spec():
    assert RTX3050.theoretical_gbps == pytest.approx(192.03, abs=0.1)
    assert limits_for(86).max_warps_per_sm == 48
    assert limits_for(120).max_warps_per_sm == 64  # falls back to the newest known arch
    spec = for_target("cuda:86", "NVIDIA GeForce RTX 3050 Ti Laptop GPU")
    assert spec.arch == 86 and spec.sm_count == 20
