"""Swizzled shared-memory layouts (``#triton_gpu.shared``) and bank mapping.

For the two fastest dimensions ``col = order[0]`` and ``row = order[1]``, Triton stores the
element at logical ``(row, col)`` at::

    phase    = (row // perPhase) % maxPhase
    phys_col = col XOR (phase * vec)          (mod numCols)
    offset   = row * numCols + phys_col

so consecutive rows rotate their ``vec``-element chunks across banks. Shared memory has 32
banks of 4 bytes; byte address ``a`` lives in bank ``(a // 4) % 32``.
"""

from __future__ import annotations

from dataclasses import dataclass

NUM_BANKS = 32
BANK_BYTES = 4


def bank_of(byte_addr: int) -> int:
    return (byte_addr // BANK_BYTES) % NUM_BANKS


@dataclass
class SharedLayout:
    vec: int = 1
    per_phase: int = 1
    max_phase: int = 1
    order: tuple[int, ...] = (1, 0)
    has_leading_offset: bool = False
    kind: str = "shared"

    @classmethod
    def from_params(cls, params: dict) -> SharedLayout:
        return cls(
            vec=int(params.get("vec", 1)),
            per_phase=int(params.get("perPhase", 1)),
            max_phase=int(params.get("maxPhase", 1)),
            order=tuple(params.get("order", [1, 0])),
            has_leading_offset=bool(params.get("hasLeadingOffset", False)),
        )

    @property
    def swizzled(self) -> bool:
        return self.max_phase > 1

    def phase(self, row: int) -> int:
        return (row // self.per_phase) % self.max_phase

    def physical_col(self, row: int, col: int, num_cols: int) -> int:
        if num_cols <= 0:
            return col
        return col ^ ((self.vec * self.phase(row)) % num_cols)

    def offset(self, coord, shape) -> int:
        """Element offset of logical ``coord`` inside a buffer of ``shape``.

        Leading dimensions beyond the encoding's rank (multi-buffering) are treated as whole
        buffers laid out back to back.
        """
        rank = len(self.order)
        lead = len(shape) - rank
        tile_shape = list(shape[lead:])
        tile_coord = list(coord[lead:])
        buffer_index = 0
        for d in range(lead):
            buffer_index = buffer_index * shape[d] + coord[d]
        tile_elems = 1
        for s in tile_shape:
            tile_elems *= s
        if rank == 1:
            return buffer_index * tile_elems + tile_coord[0]
        col_dim, row_dim = self.order[0], self.order[1]
        num_cols = tile_shape[col_dim]
        row = tile_coord[row_dim]
        phys = self.physical_col(row, tile_coord[col_dim], num_cols)
        # Remaining (slower) dims, in order, form the row index together with row_dim.
        outer = 0
        stride = 1
        for d in self.order[1:]:
            outer += tile_coord[d] * stride
            stride *= tile_shape[d]
        return buffer_index * tile_elems + outer * num_cols + phys

    def byte_offset(self, coord, shape, elem_bytes: int) -> int:
        return self.offset(coord, shape) * elem_bytes

    def bank(self, coord, shape, elem_bytes: int, base_bytes: int = 0) -> int:
        return bank_of(base_bytes + self.byte_offset(coord, shape, elem_bytes))

    def bank_map(self, shape, elem_bytes: int) -> list[list[int]]:
        """Bank of every element of a 2D tile, indexed ``[row][col]`` in logical order."""
        rows, cols = shape[-2], shape[-1]
        return [[self.bank((r, c), [rows, cols], elem_bytes) for c in range(cols)]
                for r in range(rows)]

    def describe(self) -> dict:
        return {
            "kind": self.kind,
            "vec": self.vec,
            "perPhase": self.per_phase,
            "maxPhase": self.max_phase,
            "order": list(self.order),
            "hasLeadingOffset": self.has_leading_offset,
        }


@dataclass
class PaddedScratchLayout:
    """Row-major scratch buffer with ``pad`` extra elements per row.

    Models the shared-memory staging buffer Triton allocates for ``convert_layout``.
    """

    rep_shape: list[int]
    padded_shape: list[int]
    order: list[int]
    kind: str = "padded_scratch"

    def offset(self, coord, shape=None) -> int:
        wrapped = [c % r for c, r in zip(coord, self.rep_shape)]
        index, stride = 0, 1
        for d in self.order:
            index += wrapped[d] * stride
            stride *= self.padded_shape[d]
        return index

    def byte_offset(self, coord, shape, elem_bytes: int) -> int:
        return self.offset(coord) * elem_bytes

    def describe(self) -> dict:
        return {"kind": self.kind, "repShape": self.rep_shape, "paddedShape": self.padded_shape,
                "order": self.order}
