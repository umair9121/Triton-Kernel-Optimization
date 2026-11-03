"""NVIDIA tensor-core layouts for Ampere (``nvidia_mma`` v2) and its ``dot_op`` operands.

Accumulator (``mma.sync.m16n8k16``) per warp, ``g = lane // 4``, ``t = lane % 4``::

    c0, c1 -> (g,     2t), (g,     2t + 1)
    c2, c3 -> (g + 8, 2t), (g + 8, 2t + 1)

Warps tile the CTA in row-major order (``warp = wm * warpsPerCTA[1] + wn``). Operand A of a
``dot_op`` with ``kWidth`` = k holds a 16 x 8k tile per warp and is broadcast across the warps
that share a row of C; operand B holds an 8k x 8 tile and is broadcast across warp rows.
"""

from __future__ import annotations

from .base import Coord, DistributedLayout, delinearize, prod


class MmaV2Layout(DistributedLayout):
    kind = "nvidia_mma"

    def __init__(self, warps_per_cta, instr_shape=(16, 8), version_major=2, version_minor=0,
                 threads_per_warp: int = 32):
        super().__init__(num_warps=prod(warps_per_cta), threads_per_warp=threads_per_warp)
        self.warps_per_cta = list(warps_per_cta)
        self.instr_shape = list(instr_shape)
        self.version_major = version_major
        self.version_minor = version_minor
        if version_major != 2:
            raise ValueError(f"only MMA v2 (Ampere) is modelled, got v{version_major}")

    @classmethod
    def from_params(cls, params: dict) -> MmaV2Layout:
        return cls(params["warpsPerCTA"], params.get("instrShape", [16, 8]),
                   params.get("versionMajor", 2), params.get("versionMinor", 0))

    @property
    def rank(self) -> int:
        return len(self.warps_per_cta)

    def warp_coords(self, warp: int) -> list[int]:
        # Row-major warp order: warps advance along N first.
        return delinearize(warp, self.warps_per_cta, [1, 0])

    def cta_tile(self) -> list[int]:
        return [16 * self.warps_per_cta[0], 8 * self.warps_per_cta[1]]

    def thread_elements(self, shape: list[int], warp: int, lane: int) -> list[Coord]:
        wm, wn = self.warp_coords(warp)
        tile = self.cta_tile()
        reps_m = max(1, shape[0] // tile[0])
        reps_n = max(1, shape[1] // tile[1])
        g, t = lane // 4, lane % 4
        coords: list[Coord] = []
        for rm in range(reps_m):
            for rn in range(reps_n):
                for c in range(4):
                    row = rm * tile[0] + wm * 16 + g + 8 * (c // 2)
                    col = rn * tile[1] + wn * 8 + 2 * t + (c % 2)
                    coords.append((row % shape[0], col % shape[1]))
        return list(dict.fromkeys(coords))

    def describe(self) -> dict:
        return {
            "kind": self.kind,
            "versionMajor": self.version_major,
            "warpsPerCTA": self.warps_per_cta,
            "instrShape": self.instr_shape,
            "ctaTile": self.cta_tile(),
        }


class DotOperandLayout(DistributedLayout):
    kind = "dot_op"

    def __init__(self, op_idx: int, parent: MmaV2Layout, k_width: int):
        super().__init__(num_warps=parent.num_warps, threads_per_warp=parent.threads_per_warp)
        self.op_idx = op_idx
        self.parent = parent
        self.k_width = k_width

    @property
    def rank(self) -> int:
        return 2

    @property
    def k_tile(self) -> int:
        return 8 * self.k_width

    def warp_tile(self) -> list[int]:
        return [16, self.k_tile] if self.op_idx == 0 else [self.k_tile, 8]

    def cta_tile(self) -> list[int]:
        wm, wn = self.parent.warps_per_cta
        if self.op_idx == 0:
            return [16 * wm, self.k_tile]
        return [self.k_tile, 8 * wn]

    def thread_elements(self, shape: list[int], warp: int, lane: int) -> list[Coord]:
        wm, wn = self.parent.warp_coords(warp)
        kw = self.k_width
        g, t = lane // 4, lane % 4
        tile = self.cta_tile()
        coords: list[Coord] = []
        if self.op_idx == 0:
            reps_m = max(1, shape[0] // tile[0])
            reps_k = max(1, shape[1] // tile[1])
            for rm in range(reps_m):
                for rk in range(reps_k):
                    for r in range(4 * kw):
                        k = (r % kw) + t * kw + ((r // (2 * kw)) % 2) * 4 * kw
                        m = g + ((r // kw) % 2) * 8
                        row = rm * tile[0] + wm * 16 + m
                        col = rk * tile[1] + k
                        coords.append((row % shape[0], col % shape[1]))
        else:
            reps_k = max(1, shape[0] // tile[0])
            reps_n = max(1, shape[1] // tile[1])
            for rn in range(reps_n):
                for rk in range(reps_k):
                    for r in range(2 * kw):
                        k = (r % kw) + t * kw + ((r // kw) % 2) * 4 * kw
                        row = rk * tile[0] + k
                        col = rn * tile[1] + wn * 8 + g
                        coords.append((row % shape[0], col % shape[1]))
        return list(dict.fromkeys(coords))

    def describe(self) -> dict:
        return {
            "kind": self.kind,
            "opIdx": self.op_idx,
            "kWidth": self.k_width,
            "parent": self.parent.describe(),
            "warpTile": self.warp_tile(),
            "ctaTile": self.cta_tile(),
        }
