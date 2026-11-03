"""LayerNorm forward: one program per row, looping over the row in BLOCK_SIZE chunks."""

import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(x_ptr, y_ptr, w_ptr, b_ptr, stride, N, eps,
                     BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * stride
    y_ptr += row * stride

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        acc += tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
    mean = tl.sum(acc, axis=0) / N

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)
        acc += x * x
    rstd = 1 / tl.sqrt(tl.sum(acc, axis=0) / N + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(w_ptr + cols, mask=mask)
        b = tl.load(b_ptr + cols, mask=mask)
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(y_ptr + cols, y.to(tl.float16), mask=mask)


def layernorm(x, weight, bias, eps=1e-5, block_size=1024):
    y = torch.empty_like(x)
    M, N = x.shape
    layernorm_kernel[(M,)](x, y, weight, bias, x.stride(0), N, eps,
                           BLOCK_SIZE=block_size, num_warps=8)
    return y


def main():
    torch.manual_seed(0)
    M, N = 1151, 8192
    x = torch.randn(M, N, device="cuda", dtype=torch.float16)
    w = torch.rand(N, device="cuda", dtype=torch.float16)
    b = torch.rand(N, device="cuda", dtype=torch.float16)
    ref = torch.nn.functional.layer_norm(x.float(), (N,), w.float(), b.float(), 1e-5).half()
    torch.testing.assert_close(layernorm(x, w, b), ref, atol=1e-2, rtol=0)
    print("layernorm OK")


if __name__ == "__main__":
    main()
