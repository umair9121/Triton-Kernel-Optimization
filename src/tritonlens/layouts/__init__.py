"""Layout models: map Triton layout encodings to concrete thread/element ownership."""

from __future__ import annotations

from ..ir.attrs import Encoding
from .base import DistributedLayout, OwnershipMap
from .blocked import BlockedLayout
from .mma import DotOperandLayout, MmaV2Layout
from .shared import PaddedScratchLayout, SharedLayout
from .slice import SliceLayout


class UnsupportedLayout(ValueError):
    pass


def from_encoding(enc: Encoding, num_warps: int = 4, threads_per_warp: int = 32):
    """Build a layout model for a parsed encoding attribute."""
    kind = enc.kind
    params = enc.params
    if kind == "blocked":
        return BlockedLayout.from_params(params)
    if kind == "slice":
        parent = params["parent"]
        return SliceLayout(int(params["dim"]), from_encoding(parent, num_warps, threads_per_warp))
    if kind == "nvidia_mma":
        try:
            return MmaV2Layout.from_params(params)
        except ValueError as exc:
            raise UnsupportedLayout(str(exc)) from exc
    if kind == "dot_op":
        parent = from_encoding(params["parent"], num_warps, threads_per_warp)
        if not isinstance(parent, MmaV2Layout):
            raise UnsupportedLayout("dot_op operands are modelled only for MMA v2 parents")
        return DotOperandLayout(int(params["opIdx"]), parent, int(params.get("kWidth", 2)))
    if kind == "shared":
        return SharedLayout.from_params(params)
    raise UnsupportedLayout(f"layout kind {kind!r} is not modelled")


__all__ = ["BlockedLayout", "DistributedLayout", "DotOperandLayout", "MmaV2Layout",
           "OwnershipMap", "PaddedScratchLayout", "SharedLayout", "SliceLayout",
           "UnsupportedLayout", "from_encoding"]
