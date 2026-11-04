"""Occupancy and wave quantization from compile metadata (registers, shared memory, warps).

Mirrors the CUDA occupancy calculator. On a 20-SM GPU these effects dominate: a grid of 30
blocks at 1 block/SM runs as 2 waves with the second one 50% empty, and a kernel using
96 KB of shared memory leaves room for exactly one block (4 warps out of 48) per SM.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from ..device import DeviceSpec


def _round_up(value: int, unit: int) -> int:
    return -(-value // unit) * unit


@dataclass
class Occupancy:
    num_warps: int
    regs_per_thread: int
    shared_bytes: int
    blocks_by_warps: int
    blocks_by_regs: int
    blocks_by_shared: int
    blocks_by_slots: int
    blocks_per_sm: int
    limiter: str
    max_warps_per_sm: int
    fits: bool

    @property
    def active_warps(self) -> int:
        return self.blocks_per_sm * self.num_warps

    @property
    def occupancy(self) -> float:
        return self.active_warps / self.max_warps_per_sm

    def to_json(self) -> dict:
        return {**asdict(self), "active_warps": self.active_warps, "occupancy": self.occupancy}


def occupancy(device: DeviceSpec, num_warps: int, regs_per_thread: int,
              shared_bytes: int) -> Occupancy:
    lim = device.limits
    blocks_by_warps = lim.max_warps_per_sm // num_warps
    if regs_per_thread > 0:
        regs_per_warp = _round_up(regs_per_thread * 32, lim.reg_alloc_unit)
        # The register file is split across the 4 SM sub-partitions; warps are placed per
        # sub-partition, which floors warps-per-SM to a multiple of 4.
        warps_by_regs = 4 * ((lim.regs_per_sm // 4) // regs_per_warp)
        blocks_by_regs = warps_by_regs // num_warps
    else:
        blocks_by_regs = lim.max_blocks_per_sm
    per_block = _round_up(shared_bytes + lim.reserved_shared_per_block, lim.shared_alloc_unit)
    fits = shared_bytes <= lim.max_shared_per_block and regs_per_thread <= lim.max_regs_per_thread
    blocks_by_shared = lim.shared_per_sm // per_block if fits else 0
    limits = {
        "warps": blocks_by_warps,
        "registers": blocks_by_regs,
        "shared memory": blocks_by_shared,
        "block slots": lim.max_blocks_per_sm,
    }
    blocks = min(limits.values())
    limiter = min(limits, key=lambda k: (limits[k], list(limits).index(k)))
    return Occupancy(num_warps, regs_per_thread, shared_bytes, blocks_by_warps, blocks_by_regs,
                     blocks_by_shared, lim.max_blocks_per_sm, blocks, limiter,
                     lim.max_warps_per_sm, fits)


@dataclass
class Waves:
    total_blocks: int
    blocks_per_wave: int
    waves: float
    full_waves: int
    tail_blocks: int
    efficiency: float  # useful block-slots / allocated block-slots over all waves
    busy_sms: int  # SMs that receive at least one block
    sm_count: int

    @property
    def sm_balance(self) -> float:
        """Average / maximum blocks per SM: idle SM time if every SM runs at full throughput
        regardless of how many blocks it holds (the wave model is pessimistic when several
        blocks share an SM)."""
        per_sm = -(-self.total_blocks // self.sm_count)
        return self.total_blocks / (per_sm * self.sm_count)

    def to_json(self) -> dict:
        return {**asdict(self), "smBalance": self.sm_balance}


def waves(device: DeviceSpec, occ: Occupancy, grid: list[int] | None) -> Waves | None:
    if not grid or occ.blocks_per_sm == 0:
        return None
    total = math.prod(grid)
    per_wave = occ.blocks_per_sm * device.sm_count
    full, tail = divmod(total, per_wave)
    n = full + (1 if tail else 0)
    eff = total / (n * per_wave) if n else 0.0
    return Waves(total, per_wave, total / per_wave, full, tail, eff,
                 min(total, device.sm_count), device.sm_count)


def from_metadata(device: DeviceSpec, metadata: dict) -> Occupancy | None:
    if "num_warps" not in metadata:
        return None
    return occupancy(device, int(metadata["num_warps"]), int(metadata.get("n_regs") or 0),
                     int(metadata.get("shared") or 0))


def driver_blocks_per_sm(compiled) -> int:
    """Ask the CUDA driver (``cuOccupancyMaxActiveBlocksPerMultiprocessor``) for ground truth."""
    import ctypes
    import sys

    lib = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
    compiled._init_handles()
    shared = compiled.metadata.shared
    if shared > 49152:
        set_attr = lib.cuFuncSetAttribute
        set_attr.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        set_attr(ctypes.c_void_p(compiled.function), 8, shared)  # MAX_DYNAMIC_SHARED_SIZE_BYTES
    query = lib.cuOccupancyMaxActiveBlocksPerMultiprocessor
    query.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int,
                      ctypes.c_size_t]
    out = ctypes.c_int(0)
    err = query(ctypes.byref(out), ctypes.c_void_p(compiled.function),
                compiled.metadata.num_warps * 32, shared)
    if err != 0:
        raise RuntimeError(f"cuOccupancyMaxActiveBlocksPerMultiprocessor failed: {err}")
    return out.value
