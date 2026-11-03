"""``#triton_gpu.slice`` layouts: a parent layout with one dimension removed.

Element ``i`` of the sliced tensor lives in every thread that holds a parent element whose
coordinates (with the sliced dimension re-inserted at size 1) project onto ``i``. Threads that
differ only along the removed dimension therefore hold the same values (broadcast).
"""

from __future__ import annotations

from .base import Coord, DistributedLayout


class SliceLayout(DistributedLayout):
    kind = "slice"

    def __init__(self, dim: int, parent: DistributedLayout):
        super().__init__(num_warps=parent.num_warps, threads_per_warp=parent.threads_per_warp)
        self.dim = dim
        self.parent = parent

    @property
    def rank(self) -> int:
        return self.parent.rank - 1

    def parent_shape(self, shape: list[int]) -> list[int]:
        out = list(shape)
        out.insert(self.dim, 1)
        return out

    def thread_elements(self, shape: list[int], warp: int, lane: int) -> list[Coord]:
        parent_elems = self.parent.thread_elements(self.parent_shape(shape), warp, lane)
        sliced = [tuple(c for d, c in enumerate(coord) if d != self.dim) for coord in parent_elems]
        return list(dict.fromkeys(sliced))

    def describe(self) -> dict:
        return {"kind": self.kind, "dim": self.dim, "parent": self.parent.describe()}
