### Autotuning with device-aware pruning (NVIDIA GeForce RTX 3050 Ti Laptop GPU)

108-config datacenter-style matmul sweep, fp16, fresh Triton cache per run. Pruning uses
`tritonlens.autotune.device_pruner` and does not change the winning kernel.

| size | configs benchmarked | failed (out of resources) | autotune time | best kernel | best config |
|---|---|---|---|---|---|
| 640³ all | 108 | 26 | 204.0 s | 0.076 ms (6.94 TFLOP/s) | `64x64x32 w4 s3` |
| 640³ pruned | 68 | 0 | 60.0 s | 0.075 ms (7.00 TFLOP/s) | `64x64x32 w4 s3` |
| 1024³ all | 108 | 26 | 203.9 s | 0.291 ms (7.37 TFLOP/s) | `64x64x32 w4 s3` |
| 1024³ pruned | 82 | 0 | 105.3 s | 0.292 ms (7.36 TFLOP/s) | `64x64x32 w4 s3` |
| 2048³ all | 108 | 26 | 209.3 s | 2.262 ms (7.59 TFLOP/s) | `64x64x32 w4 s4` |
| 2048³ pruned | 82 | 0 | 110.7 s | 2.272 ms (7.56 TFLOP/s) | `64x64x32 w4 s4` |

Host-side keep counts (no compile): `python benchmarks/autotune_prune.py --dry`.

| size | kept / 108 | dropped: shared memory | dropped: SM imbalance | dropped: L2 stripe |
|---|---|---|---|---|
| 640³ | 68 | 26 | 14 | 0 |
| 1024³ | 82 | 26 | 0 | 0 |
| 2048³ | 82 | 26 | 0 | 0 |
| 4096³ | 82 | 26 | 0 | 0 |
