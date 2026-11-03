"""Row-wise fused softmax: one program per row, reductions along the contiguous axis."""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols,
                   BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * in_row_stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    tl.store(out_ptr + row * out_row_stride + cols, num / den, mask=mask)


def softmax(x: torch.Tensor):
    n_rows, n_cols = x.shape
    block = triton.next_power_of_2(n_cols)
    num_warps = 4 if block <= 2048 else 8
    out = torch.empty_like(x)
    softmax_kernel[(n_rows,)](out, x, x.stride(0), out.stride(0), n_cols,
                              BLOCK_SIZE=block, num_warps=num_warps)
    return out


def main():
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device="cuda")
    torch.testing.assert_close(softmax(x), torch.softmax(x, dim=1))
    print("softmax OK")


if __name__ == "__main__":
    main()
