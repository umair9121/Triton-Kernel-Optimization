"""Kernels with deliberate memory-access mistakes, used to exercise the linter.

* ``misaligned_add``: the tensors start 1 element past a 16-byte boundary and ``n`` is odd, so
  Triton cannot prove alignment and falls back to scalar 4-byte loads.
* ``strided_gather``: every lane reads with a runtime stride, so each warp request touches
  32 different sectors.
* ``odd_block``: a row of 1000 columns handled with a BLOCK of 1024 plus heavy masking on a
  column count that is not a multiple of 16.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def misaligned_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask),
             mask=mask)


@triton.jit
def strided_gather_kernel(x_ptr, out_ptr, n, stride, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs * stride, mask=mask)
    tl.store(out_ptr + offs, x * 2.0, mask=mask)


@triton.jit
def odd_block_kernel(x_ptr, out_ptr, n_cols, row_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row * row_stride + cols, x + 1.0, mask=mask)


def main():
    torch.manual_seed(0)
    n = 1 << 20
    base = torch.randn(3, n + 1, device="cuda")
    x, y, out = base[0, 1:], base[1, 1:], base[2, 1:]
    n_odd = n - 3
    misaligned_add_kernel[(triton.cdiv(n_odd, 1024),)](x, y, out, n_odd, BLOCK=1024)
    torch.testing.assert_close(out[:n_odd], x[:n_odd] + y[:n_odd])

    src = torch.randn(n * 8 + 5, device="cuda")
    dst = torch.empty(n, device="cuda")
    strided_gather_kernel[(triton.cdiv(n, 1024),)](src, dst, n, 8, BLOCK=1024)
    torch.testing.assert_close(dst, src[: n * 8 : 8] * 2.0)

    rows = torch.randn(512, 1000, device="cuda")
    res = torch.empty_like(rows)
    odd_block_kernel[(512,)](rows, res, 1000, rows.stride(0), BLOCK=1024)
    torch.testing.assert_close(res, rows + 1.0)
    print("bad_access OK")


if __name__ == "__main__":
    main()
