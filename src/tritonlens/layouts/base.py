"""Common machinery for distributed (register) layouts."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

Coord = tuple[int, ...]


def prod(values) -> int:
    out = 1
    for v in values:
        out *= v
    return out


def delinearize(index: int, sizes: list[int], order: list[int]) -> list[int]:
    """Split ``index`` into per-dimension indices, ``order[0]`` varying fastest."""
    out = [0] * len(sizes)
    for dim in order:
        out[dim] = index % sizes[dim]
        index //= sizes[dim]
    return out


def linearize(coord, shape, order=None) -> int:
    order = order if order is not None else list(range(len(shape)))[::-1]
    index, stride = 0, 1
    for dim in order:
        index += coord[dim] * stride
        stride *= shape[dim]
    return index


@dataclass
class OwnershipMap:
    """Which (warp, lane, register) triples hold each element of a tensor.

    ``threads[w][l]`` is the register list of thread ``l`` in warp ``w``; each entry is a
    tensor coordinate. ``owners[flat_index]`` lists every (warp, lane, reg) holding the
    element; more than one entry means the layout replicates (broadcasts) that element.
    """

    shape: list[int]
    threads: list[list[list[Coord]]]

    @cached_property
    def owners(self) -> list[list[tuple[int, int, int]]]:
        owners: list[list[tuple[int, int, int]]] = [[] for _ in range(prod(self.shape))]
        for w, lanes in enumerate(self.threads):
            for l, regs in enumerate(lanes):
                for r, coord in enumerate(regs):
                    owners[linearize(coord, self.shape)].append((w, l, r))
        return owners

    @property
    def num_warps(self) -> int:
        return len(self.threads)

    @property
    def threads_per_warp(self) -> int:
        return len(self.threads[0]) if self.threads else 0

    @property
    def regs_per_thread(self) -> int:
        return len(self.threads[0][0]) if self.threads and self.threads[0] else 0

    @property
    def replication(self) -> int:
        """Max number of threads/registers holding the same element (1 = no broadcast)."""
        return max((len(o) for o in self.owners), default=0)

    def owner(self, coord: Coord) -> tuple[int, int, int]:
        return self.owners[linearize(coord, self.shape)][0]

    def covers_all(self) -> bool:
        return all(self.owners)


class DistributedLayout:
    """A register layout: maps (warp, lane, register) to tensor coordinates."""

    kind = "distributed"

    def __init__(self, num_warps: int, threads_per_warp: int = 32):
        self.num_warps = num_warps
        self.threads_per_warp = threads_per_warp

    @property
    def rank(self) -> int:
        raise NotImplementedError

    def thread_elements(self, shape: list[int], warp: int, lane: int) -> list[Coord]:
        """Tensor coordinates held by one thread, in register order."""
        raise NotImplementedError

    def ownership(self, shape: list[int]) -> OwnershipMap:
        shape = list(shape)
        threads = [
            [self.thread_elements(shape, w, l) for l in range(self.threads_per_warp)]
            for w in range(self.num_warps)
        ]
        return OwnershipMap(shape=shape, threads=threads)

    def regs_per_thread(self, shape: list[int]) -> int:
        return len(self.thread_elements(shape, 0, 0))

    def contiguous_elements(self, shape: list[int], dim: int) -> int:
        """Elements a thread holds back-to-back along ``dim`` (the vector width candidate)."""
        elems = self.thread_elements(shape, 0, 0)
        if not elems:
            return 1
        best, run = 1, 1
        for prev, cur in zip(elems, elems[1:]):
            same_other = all(prev[d] == cur[d] for d in range(len(cur)) if d != dim)
            if same_other and cur[dim] == prev[dim] + 1:
                run += 1
                best = max(best, run)
            else:
                run = 1
        return best

    def describe(self) -> dict:
        return {"kind": self.kind}

    def to_json(self) -> dict:
        return self.describe()
