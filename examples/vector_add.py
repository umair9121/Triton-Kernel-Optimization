"""Element-wise vector addition: the simplest Triton kernel, used as a smoke test."""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024):
    out = torch.empty_like(x)
    n = out.numel()
    grid = (triton.cdiv(n, block_size),)
    compiled = add_kernel[grid](x, y, out, n, BLOCK_SIZE=block_size)
    return out, compiled


def main():
    torch.manual_seed(0)
    x = torch.rand(98432, device="cuda")
    y = torch.rand(98432, device="cuda")
    out, _ = add(x, y)
    torch.testing.assert_close(out, x + y)
    print("vector_add OK")


if __name__ == "__main__":
    main()
