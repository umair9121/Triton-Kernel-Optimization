"""Layout models: map Triton layout encodings to concrete thread/element ownership."""

from __future__ import annotations

from ..ir.attrs import Encoding
from .base import DistributedLayout, OwnershipMap
from .blocked import BlockedLayout


class UnsupportedLayout(ValueError):
    pass


def from_encoding(enc: Encoding, num_warps: int = 4, threads_per_warp: int = 32):
    """Build a layout model for a parsed encoding attribute."""
    kind = enc.kind
    params = enc.params
    if kind == "blocked":
        return BlockedLayout.from_params(params)
    raise UnsupportedLayout(f"layout kind {kind!r} is not modelled")


__all__ = ["BlockedLayout", "DistributedLayout", "OwnershipMap", "UnsupportedLayout",
           "from_encoding"]
