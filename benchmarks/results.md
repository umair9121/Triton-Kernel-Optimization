# Predicted vs measured

Everything here was measured on an **NVIDIA GeForce RTX 3050 Ti Laptop GPU** (Ampere, sm_86,
4 GB, 20 SMs, roughly 190 GB/s DRAM peak) with an i7-12700H, Windows 11, PyTorch 2.6 and
`triton-windows` 3.2. Raw data is in [`results/`](results/). Regenerate with:

```
python -m tritonlens validate                 # layout ground truth
python benchmarks/bank_conflicts.py           # shared-memory swizzle sweep
python benchmarks/lint_fixes.py               # global-memory findings vs bandwidth
python benchmarks/occupancy_check.py          # occupancy vs CUDA driver
python benchmarks/waves.py                    # SM-balance model vs time
python benchmarks/device_rules.py             # TL007-TL009 suggested fixes vs time
python benchmarks/autotune_prune.py --dry     # how many A100 configs survive on this GPU
```

Laptop GPUs change clocks with temperature, so every timing table interleaves the variants over
7 rounds and reports the median.

## 1. Layout models vs the hardware: 26/26 exact

Each probe kernel writes `%tid.x` (read with `tl.inline_asm_elementwise`) into every element of
a tensor. The layout Triton chose for that tensor is parsed from the kernel's own TTGIR, and the
model's owner prediction is compared element by element with what the GPU wrote.

| layout family | probes | elements checked | mismatches |
|---|---|---|---|
| `blocked` (1D, 2D, row and column major, 1 to 8 warps, including a broadcast case) | 20 | 44,832 | 0 |
| `slice` (results of `tl.sum` along an axis) | 2 | 96 | 0 |
| `nvidia_mma` v2 accumulators (`mma.sync.m16n8k16`, 1 to 8 warps) | 4 | 13,568 | 0 |

`dot_op` operand fragments cannot be probed this way, because Triton fills them from shared
memory. They are checked against the PTX ISA fragment tables and against the requirement that
each warp holds exactly the A rows and B columns of the C tile it computes
(`tests/test_layouts_mma.py`).

Full output: [`results/layout_ground_truth.txt`](results/layout_ground_truth.txt).

## 2. Shared-memory bank conflicts: swizzle sweep on a real matmul

The fp16 matmul from `examples/matmul.py` is compiled normally. Its TTGIR is then rewritten with
different `#shared` swizzle parameters and recompiled from the edited IR with `triton.compile`.
Every variant produces the correct result; only the shared-memory address pattern changes.
tritonlens predicts the wavefronts per CTA per K-iteration (`ldmatrix` + `cp.async`).

| config | variant | predicted conflict | wavefronts/iter (ideal) | measured ms | slowdown |
|---|---|---|---|---|---|
| 64x64x32, 4 warps | Triton default | 1.00x | 192 (192) | 1.174 | 1.00x |
| 64x64x32, 4 warps | maxPhase=2 | 2.33x | 448 (192) | 1.232 | 1.05x |
| 64x64x32, 4 warps | A unswizzled | 2.00x | 384 (192) | 1.197 | 1.02x |
| 64x64x32, 4 warps | B unswizzled | 3.33x | 640 (192) | 1.606 | 1.37x |
| 64x64x32, 4 warps | no swizzle | 4.33x | 832 (192) | 2.034 | 1.73x |
| 128x128x32, 4 warps | Triton default | 1.00x | 384 (384) | 1.215 | 1.00x |
| 128x128x32, 4 warps | no swizzle | 4.33x | 1664 (384) | 1.290 | 1.06x |

- With 64x64 tiles, shared-memory bandwidth is on the critical path, and the predicted
  wavefronts track the measured time: **Pearson r = 0.94**. The ordering is right except
  between the two mildest variants, which differ by less than the run-to-run noise.
- With 128x128 tiles on 4 warps, each warp computes a 64x64 sub-tile, so the kernel is bound by
  `mma.sync` throughput. Even a 4.33x conflict costs only 6%. The simulator predicts
  shared-memory *cost*, not runtime: whether that cost shows up depends on what else is
  saturated.
- The predicted instruction counts match the PTX: 2 `ldmatrix.x4` per warp per operand tile,
  and 12 16-byte `cp.async` per prologue/iteration pair.

Full table: [`results/bank_conflicts.md`](results/bank_conflicts.md).

## 3. Global-memory findings vs bandwidth

Each case pairs a kernel the linter flags with a variant that applies the suggested fix.

| case | variant | findings | predicted B/lane (PTX) | predicted sector efficiency | GB/s | speedup |
|---|---|---|---|---|---|---|
| vector add, 16M fp32 | offset views and odd `n` | TL001, TL006 | 4 (4) | 100% | 171.7 | 1.00x |
| vector add, 16M fp32 | aligned, `n % 16 == 0` | - | 16 (16) | 100% | 173.8 | 1.01x |
| softmax, 8192 rows | 781 columns | TL001, TL006 | 4 (4) | 100% | 163.1 | 1.00x |
| softmax, 8192 rows | 784 columns | - | 16 (16) | 100% | 174.4 | 1.06x |
| layernorm 4096x8192 fp16 | BLOCK=1024, 8 warps | TL001 | 8 (8) | 100% | 164.3 | 1.00x |
| layernorm 4096x8192 fp16 | BLOCK=2048, 8 warps | - | 16 (16) | 100% | 158.3 | 0.96x |
| gather 4M fp32 | stride 8 | **TL002** | 4 (4) | **12%** | 38.6 | 1.00x |
| gather 4M fp32 | stride 1 | - | 16 (16) | 100% | 166.5 | **4.31x** |

- The predicted vector width matched the width the compiler actually emitted in **every** case
  (the PTX column).
- **Sector efficiency (TL002) is what predicts bandwidth.** 12% predicted efficiency becomes a
  4.3x slowdown.
- **Vector width alone (TL001) barely matters when every sector is fully used.** These kernels
  are DRAM-bound at about 170 GB/s, and 32 lanes of scalar loads fill the same 32-byte sectors as
  8 lanes of `v4` loads. The difference stays within 6%. This measurement is why TL001 is an
  `info` finding for fully coalesced accesses and a `warning` only when coalescing also suffers.
  Narrow accesses still cost 2 to 4x more memory instructions, which matters for issue-bound
  or L2-resident kernels and on GPUs with far more bandwidth per SM than this one.

Full table: [`results/lint_fixes.md`](results/lint_fixes.md).

## 4. Occupancy, SM balance and device-rule fixes

These are the analyses that matter on a 20-SM, 4 GB, 2 MB-L2 laptop GPU. Full tables:

- [`results/occupancy_check.md`](results/occupancy_check.md) — **207/207** configs match
  `cuOccupancyMaxActiveBlocksPerMultiprocessor`.
- [`results/waves.md`](results/waves.md) — equal-work programs finish when the busiest SM
  does (`ceil(P/S)*S`). Wave-count is pessimistic when several blocks share an SM; the
  per-SM balance model stayed within **6%** of measured time.
- [`results/device_rules.md`](results/device_rules.md) — flagged configs vs the suggested
  fix, CUDA-graph median of 7 interleaved rounds:

| case | flagged → fix | speedup |
|---|---|---|
| matmul 640³ fp16, 128x128x64 w8 (25 programs / 62% SM) | 64x64x64 w4 (100 programs / 100%) | **1.63x** |
| matmul 2048³ fp16, 64x64x64 w4 s5 (1 block/SM) | s4 (2 blocks/SM) | **1.09x** |
| row sum 8 × 4M fp32 (8 programs, 12 SMs idle) | split-K to 40 programs | **1.80x** |

Going from 20 to 21 matmul tiles on this GPU **doubles** run time: the extra tile is a
second wave on one SM while 19 SMs sit idle.

## 5. Autotune pruning

A 108-config datacenter-style sweep (`BLOCK_M/N` ∈ {64,128,256}, `BLOCK_K` ∈ {32,64},
`num_stages` ∈ {3,4,5}, `num_warps` ∈ {4,8}, `GROUP_M=8`) is autotuned with and without
`tritonlens.autotune.device_pruner`. Each run uses a fresh Triton cache so compile time is
included. The pruner does not change the winning kernel; it only skips configs this GPU
cannot launch or that leave SMs idle when a balanced config exists.

| size | configs | failed compiles | autotune | best kernel | best config |
|---|---|---|---|---|---|
| 640³ all | 108 | 26 | 204 s | 0.076 ms (6.94 TFLOP/s) | `64x64x32 w4 s3` |
| 640³ pruned | 68 | 0 | **60 s (3.4×)** | 0.075 ms (7.00 TFLOP/s) | `64x64x32 w4 s3` |
| 1024³ all | 108 | 26 | 204 s | 0.291 ms (7.37 TFLOP/s) | `64x64x32 w4 s3` |
| 1024³ pruned | 82 | 0 | **105 s (1.9×)** | 0.292 ms (7.36 TFLOP/s) | `64x64x32 w4 s3` |
| 2048³ all | 108 | 26 | 209 s | 2.262 ms (7.59 TFLOP/s) | `64x64x32 w4 s4` |
| 2048³ pruned | 82 | 0 | **111 s (1.9×)** | 2.272 ms (7.56 TFLOP/s) | `64x64x32 w4 s4` |

The 26 failed compiles are 256-wide tiles that need more than 99 KiB of shared memory:
Triton still compiles them, then the launch returns OutOfResources. On a 4 GB laptop that
is minutes of heat and allocator pressure for kernels that can never run. At 640³ the
128×128 tiles (25 programs → 62% SM) are also dropped because a 64×64 config in the same
list reaches 100%. Full table: [`results/autotune_prune.md`](results/autotune_prune.md).

## What is estimated

- The scratch buffer of `convert_layout` is not visible in TTGIR. tritonlens models it after
  Triton 3.2's padded-scratch lowering (`getScratchConfigForCvt`). The model matches the PTX
  instruction counts (for the tiled transpose: 32 `st.shared.b32` and 8 `ld.shared.v4` per
  thread), but its conflict factor has not been checked against hardware counters. The viewer
  and the linter mark these numbers as estimated.
- Hardware bank-conflict counters (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_*` in Nsight
  Compute) were not available on this machine, so shared-memory predictions are validated
  through controlled timing (section 2) instead.
