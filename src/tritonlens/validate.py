"""Ground-truth validation of layout models on a real GPU.

Each probe kernel writes the hardware thread id (``%tid.x``) of the thread that holds every
tensor element. The layout Triton picked for that value is read back from the kernel's TTGIR
and the model's prediction is compared element by element against what the GPU wrote.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ir import TensorType, parse_module
from .layouts import UnsupportedLayout, from_encoding


@dataclass
class ProbeResult:
    name: str
    layout_kind: str
    layout: dict
    shape: list[int]
    elements: int
    mismatches: int
    replication: int

    @property
    def ok(self) -> bool:
        return self.mismatches == 0


def _kernels():
    from . import _probes

    return _probes.tid_2d, _probes.tid_1d, _probes.tid_slice, _probes.tid_mma


def _asm_layout(compiled):
    """Return (encoding, shape) of the first inline-asm result in the kernel's TTGIR."""
    mod = parse_module(compiled.asm["ttgir"])
    for op in mod.func.ops:
        if op.is_("elementwise_inline_asm"):
            rtype = op.result_type
            if isinstance(rtype, TensorType):
                return mod, rtype.encoding, rtype.shape
    raise RuntimeError("inline asm op not found in TTGIR")


def _compare(name, compiled, gpu_tids, shape) -> ProbeResult:
    mod, enc, asm_shape = _asm_layout(compiled)
    try:
        layout = from_encoding(enc, mod.num_warps, mod.threads_per_warp)
    except UnsupportedLayout:
        return ProbeResult(name=name, layout_kind=enc.kind, layout={}, shape=list(asm_shape),
                           elements=0, mismatches=-1, replication=0)
    own = layout.ownership(asm_shape)
    flat = gpu_tids.flatten().tolist()
    mismatches = 0
    tpw = mod.threads_per_warp
    for idx, tid in enumerate(flat):
        allowed = {w * tpw + l for (w, l, _r) in own.owners[idx]}
        if tid not in allowed:
            mismatches += 1
    return ProbeResult(name=name, layout_kind=enc.kind, layout=layout.describe(),
                       shape=list(asm_shape), elements=len(flat), mismatches=mismatches,
                       replication=own.replication)


def run_probes(quick: bool = False) -> list[ProbeResult]:
    import torch

    tid_2d, tid_1d, tid_slice, tid_mma = _kernels()
    results: list[ProbeResult] = []
    dev = "cuda"

    cases_2d = [(64, 64, 4), (32, 32, 4), (16, 128, 4), (128, 16, 8), (64, 4, 2), (8, 8, 1),
                (256, 32, 8), (32, 64, 1)]
    if quick:
        cases_2d = cases_2d[:3]
    for bm, bn, nw in cases_2d:
        for transposed in (False, True):
            out = torch.full((bm * bn,), -1, dtype=torch.int32, device=dev)
            k = tid_2d[(1,)](out, BM=bm, BN=bn, TRANSPOSED=transposed, num_warps=nw)
            grid = out.view(bn, bm).t() if transposed else out.view(bm, bn)
            results.append(_compare(f"2d {bm}x{bn} w{nw}{' T' if transposed else ''}", k,
                                    grid.contiguous().cpu(), [bm, bn]))

    for block, nw in [(1024, 4), (128, 4), (32, 4), (4096, 8)]:
        out = torch.full((block,), -1, dtype=torch.int32, device=dev)
        k = tid_1d[(1,)](out, BLOCK=block, num_warps=nw)
        results.append(_compare(f"1d {block} w{nw}", k, out.cpu(), [block]))

    for bm, bn, nw in [(64, 64, 4), (32, 128, 4)]:
        x = torch.ones((bm, bn), dtype=torch.float32, device=dev)
        out = torch.full((bm,), -1, dtype=torch.int32, device=dev)
        k = tid_slice[(1,)](x, out, BM=bm, BN=bn, num_warps=nw)
        results.append(_compare(f"slice {bm}x{bn} w{nw}", k, out.cpu(), [bm]))

    for bm, bn, bk, nw in [(64, 64, 32, 4), (32, 32, 16, 4), (128, 64, 32, 8), (16, 16, 16, 1)]:
        a = torch.zeros((bm, bk), dtype=torch.float16, device=dev)
        b = torch.zeros((bk, bn), dtype=torch.float16, device=dev)
        out = torch.full((bm, bn), -1, dtype=torch.int32, device=dev)
        k = tid_mma[(1,)](a, b, out, BM=bm, BN=bn, BK=bk, num_warps=nw)
        results.append(_compare(f"mma {bm}x{bn}x{bk} w{nw}", k, out.cpu(), [bm, bn]))
    return results


def format_results(results: list[ProbeResult]) -> str:
    lines = [f"{'probe':<22} {'layout':<11} {'elements':>8} {'repl':>5} {'mismatch':>8}  result"]
    for r in results:
        lines.append(f"{r.name:<22} {r.layout_kind:<11} {r.elements:>8} {r.replication:>5} "
                     f"{r.mismatches:>8}  {'PASS' if r.ok else 'FAIL'}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_results(run_probes()))
