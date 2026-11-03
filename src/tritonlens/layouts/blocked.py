"""``#triton_gpu.blocked`` layouts.

A blocked layout tiles the tensor with a CTA tile of ``sizePerThread * threadsPerWarp *
warpsPerCTA`` elements per dimension. Along each dimension a thread owns ``sizePerThread``
consecutive elements; lanes and warps are laid out with ``order[0]`` varying fastest. If the
tensor is larger than the CTA tile the tile repeats in registers; if it is smaller, the
coordinates wrap and several threads hold the same element (broadcast).
"""

from __future__ import annotations

from .base import Coord, DistributedLayout, delinearize, prod


class BlockedLayout(DistributedLayout):
    kind = "blocked"

    def __init__(self, size_per_thread, threads_per_warp, warps_per_cta, order):
        super().__init__(num_warps=prod(warps_per_cta), threads_per_warp=prod(threads_per_warp))
        self.size_per_thread = list(size_per_thread)
        self.threads_per_warp_dims = list(threads_per_warp)
        self.warps_per_cta = list(warps_per_cta)
        self.order = list(order)

    @classmethod
    def from_params(cls, params: dict) -> BlockedLayout:
        return cls(params["sizePerThread"], params["threadsPerWarp"], params["warpsPerCTA"],
                   params["order"])

    @property
    def rank(self) -> int:
        return len(self.size_per_thread)

    def cta_tile(self) -> list[int]:
        return [s * t * w for s, t, w in
                zip(self.size_per_thread, self.threads_per_warp_dims, self.warps_per_cta)]

    def reps(self, shape: list[int]) -> list[int]:
        return [max(1, s // t) for s, t in zip(shape, self.cta_tile())]

    def thread_elements(self, shape: list[int], warp: int, lane: int) -> list[Coord]:
        lane_idx = delinearize(lane, self.threads_per_warp_dims, self.order)
        warp_idx = delinearize(warp, self.warps_per_cta, self.order)
        tile = self.cta_tile()
        reps = self.reps(shape)
        spt = self.size_per_thread
        coords: list[Coord] = []
        n_inner = prod(spt)
        for rep in range(prod(reps)):
            rep_idx = delinearize(rep, reps, self.order)
            for inner in range(n_inner):
                inner_idx = delinearize(inner, spt, self.order)
                coord = []
                for d in range(self.rank):
                    c = (rep_idx[d] * tile[d]
                         + warp_idx[d] * spt[d] * self.threads_per_warp_dims[d]
                         + lane_idx[d] * spt[d]
                         + inner_idx[d])
                    coord.append(c % shape[d])
                coords.append(tuple(coord))
        # A tensor smaller than sizePerThread would repeat elements inside one thread; Triton
        # drops those redundant registers.
        return list(dict.fromkeys(coords))

    def describe(self) -> dict:
        return {
            "kind": self.kind,
            "sizePerThread": self.size_per_thread,
            "threadsPerWarp": self.threads_per_warp_dims,
            "warpsPerCTA": self.warps_per_cta,
            "order": self.order,
            "ctaTile": self.cta_tile(),
        }
