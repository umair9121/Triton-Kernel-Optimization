"""Probe kernels for :mod:`tritonlens.validate` (imported lazily: requires Triton)."""

import triton
import triton.language as tl

TID_ASM = "mov.u32 $0, %tid.x;"


@triton.jit
def tid_2d(out_ptr, BM: tl.constexpr, BN: tl.constexpr, TRANSPOSED: tl.constexpr):
    rows = tl.arange(0, BM)[:, None]
    cols = tl.arange(0, BN)[None, :]
    if TRANSPOSED:
        offs = cols * BM + rows
    else:
        offs = rows * BN + cols
    tid = tl.inline_asm_elementwise("mov.u32 $0, %tid.x;", "=r,r", [offs], dtype=tl.int32,
                                    is_pure=False, pack=1)
    tl.store(out_ptr + offs, tid)


@triton.jit
def tid_1d(out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tid = tl.inline_asm_elementwise("mov.u32 $0, %tid.x;", "=r,r", [offs], dtype=tl.int32,
                                    is_pure=False, pack=1)
    tl.store(out_ptr + offs, tid)


@triton.jit
def tid_slice(x_ptr, out_ptr, BM: tl.constexpr, BN: tl.constexpr):
    rows = tl.arange(0, BM)[:, None]
    cols = tl.arange(0, BN)[None, :]
    x = tl.load(x_ptr + rows * BN + cols)
    r = tl.sum(x, axis=1)
    tid = tl.inline_asm_elementwise("mov.u32 $0, %tid.x;", "=r,r", [r], dtype=tl.int32,
                                    is_pure=False, pack=1)
    tl.store(out_ptr + tl.arange(0, BM), tid)


@triton.jit
def tid_mma(a_ptr, b_ptr, out_ptr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a = tl.load(a_ptr + rm[:, None] * BK + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * BN + rn[None, :])
    zero = tl.zeros((BM, BN), dtype=tl.int32)
    tid = tl.inline_asm_elementwise("mov.u32 $0, %tid.x;", "=r,r", [zero], dtype=tl.int32,
                                    is_pure=False, pack=1)
    acc = tl.dot(a, b, tid.to(tl.float32))
    tl.store(out_ptr + rm[:, None] * BN + rn[None, :], acc.to(tl.int32))
