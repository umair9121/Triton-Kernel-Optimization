"""Matrix transpose, three ways.

* ``transpose_naive``: each program copies one input row-tile and writes it with the *same*
  register layout, so every warp store is strided by the output row pitch (uncoalesced).
* ``transpose_tiled``: loads a 2D tile and writes ``tl.trans`` of it; Triton inserts a
  ``convert_layout`` that stages the tile through shared memory so both sides coalesce.
* ``transpose_column_read``: reads the input down columns (the fast register axis is the
  strided one), the mirror-image mistake.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def transpose_naive_kernel(x_ptr, y_ptr, M, N, stride_xm, stride_ym,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :], mask=mask)
    # y[n, m] = x[m, n] using x's layout: the store's fast axis walks rows of y.
    tl.store(y_ptr + offs_n[None, :] * stride_ym + offs_m[:, None], x, mask=mask)


@triton.jit
def transpose_tiled_kernel(x_ptr, y_ptr, M, N, stride_xm, stride_ym,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :], mask=mask)
    xt = tl.trans(x)
    mask_t = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(y_ptr + offs_n[:, None] * stride_ym + offs_m[None, :], xt, mask=mask_t)


@triton.jit
def transpose_column_read_kernel(x_ptr, y_ptr, M, N, stride_xm, stride_ym,
                                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Rows of the register tile index y's rows (x's columns): x reads are strided.
    mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    x = tl.load(x_ptr + offs_m[None, :] * stride_xm + offs_n[:, None], mask=mask)
    tl.store(y_ptr + offs_n[:, None] * stride_ym + offs_m[None, :], x, mask=mask)


KERNELS = {
    "naive": transpose_naive_kernel,
    "tiled": transpose_tiled_kernel,
    "column_read": transpose_column_read_kernel,
}


def transpose(x, variant="tiled", block_m=64, block_n=64, num_warps=4):
    M, N = x.shape
    y = torch.empty((N, M), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    KERNELS[variant][grid](x, y, M, N, x.stride(0), y.stride(0),
                           BLOCK_M=block_m, BLOCK_N=block_n, num_warps=num_warps)
    return y


def main():
    torch.manual_seed(0)
    x = torch.randn(2048, 1536, device="cuda", dtype=torch.float32)
    for variant in KERNELS:
        torch.testing.assert_close(transpose(x, variant), x.t().contiguous())
    print("transpose OK")


if __name__ == "__main__":
    main()
