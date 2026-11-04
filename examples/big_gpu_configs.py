"""Kernel configurations that are sensible on a datacenter GPU but waste a small one.

Each launch below is correct, and typical of configs copied from tutorials tuned on
A100/H100 (108-132 SMs, 164-228 KB shared memory). On a 20-SM, 100 KB laptop GPU they trigger
tritonlens' device-aware rules:

* ``big_tiles``     128x128 tiles on a 640x640 output: 25 programs on 20 SMs (TL008, TL007)
* ``deep_pipeline`` 5 pipeline stages of 64x64x64 tiles: 64 KB shared per block (TL009)
* ``few_rows``      a row reduction over 8 rows: 12 of 20 SMs idle (TL008)

``python -m tritonlens lint examples/big_gpu_configs.py`` explains each one;
``benchmarks/device_rules.py`` times the suggested fixes.
"""

import torch
import triton
import triton.language as tl

from matmul import matmul


@triton.jit
def row_sum_kernel(x_ptr, out_ptr, n_cols, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, n_cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        acc += tl.load(x_ptr + row * stride + offs, mask=offs < n_cols, other=0.0)
    tl.store(out_ptr + row, tl.sum(acc, axis=0))


@triton.jit
def row_sum_split_kernel(x_ptr, out_ptr, n_cols, stride, cols_per_split,
                         BLOCK: tl.constexpr):
    """Split-K version: program (row, split) reduces one slice and adds it atomically."""
    row = tl.program_id(0)
    split = tl.program_id(1)
    begin = split * cols_per_split
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, cols_per_split, BLOCK):
        offs = begin + start + tl.arange(0, BLOCK)
        acc += tl.load(x_ptr + row * stride + offs, mask=offs < n_cols, other=0.0)
    tl.atomic_add(out_ptr + row, tl.sum(acc, axis=0))


def row_sum(x, block=1024, num_warps=4):
    out = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
    row_sum_kernel[(x.shape[0],)](x, out, x.shape[1], x.stride(0), BLOCK=block,
                                  num_warps=num_warps)
    return out


def row_sum_split(x, splits, block=1024, num_warps=4):
    out = torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
    per = triton.cdiv(triton.cdiv(x.shape[1], splits), block) * block
    row_sum_split_kernel[(x.shape[0], splits)](x, out, x.shape[1], x.stride(0), per,
                                               BLOCK=block, num_warps=num_warps)
    return out


def big_tiles(size=640):
    a = torch.randn(size, size, device="cuda", dtype=torch.float16)
    b = torch.randn(size, size, device="cuda", dtype=torch.float16)
    return matmul(a, b, block_m=128, block_n=128, block_k=64, num_warps=8, num_stages=3)


def deep_pipeline(size=2048):
    a = torch.randn(size, size, device="cuda", dtype=torch.float16)
    b = torch.randn(size, size, device="cuda", dtype=torch.float16)
    return matmul(a, b, block_m=64, block_n=64, block_k=64, num_warps=4, num_stages=5)


def few_rows(rows=8, cols=1 << 20):
    x = torch.randn(rows, cols, device="cuda")
    return row_sum(x)


def main():
    torch.manual_seed(0)
    big_tiles()
    deep_pipeline()
    few_rows()
    torch.cuda.synchronize()
    print("big_gpu_configs OK")


if __name__ == "__main__":
    main()
