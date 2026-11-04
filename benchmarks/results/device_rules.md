### Device-aware findings vs measured time (NVIDIA GeForce RTX 3050 Ti Laptop GPU)

Median of 7 interleaved rounds, CUDA-graph replay. Speedup is relative to the flagged configuration.

| case | variant | findings | blocks/SM | programs | SM balance | ms | rate | speedup |
|---|---|---|---|---|---|---|---|---|
| matmul 640^3 fp16 | 128x128x64 w8 s3 (flagged) | TL007, TL008 | 1 | 25 | 62% | 0.068 | 7.75 TFLOP/s | 1.00x |
| matmul 640^3 fp16 | 64x128x64 w4 s3 | TL008, TL009 | 2 | 50 | 83% | 0.049 | 10.62 TFLOP/s | 1.37x |
| matmul 640^3 fp16 | 64x64x64 w4 s3 | TL009 | 3 | 100 | 100% | 0.041 | 12.65 TFLOP/s | 1.63x |
| matmul 2048^3 fp16 | 64x64x64 w4 s5 (flagged) | TL007, TL009 | 1 | 1024 | 98% | 2.527 | 6.80 TFLOP/s | 1.00x |
| matmul 2048^3 fp16 | 64x64x64 w4 s4 | TL009 | 2 | 1024 | 98% | 2.314 | 7.42 TFLOP/s | 1.09x |
| matmul 2048^3 fp16 | 64x64x64 w4 s3 | TL009 | 3 | 1024 | 98% | 2.334 | 7.36 TFLOP/s | 1.08x |
| matmul 2048^3 fp16 | 64x64x64 w4 s2 | - | 4 | 1024 | 98% | 2.332 | 7.37 TFLOP/s | 1.08x |
| row sum 8 x 4M fp32 | 1 program per row (flagged) | TL008 | 12 | 8 | 40% | 1.372 | 98 GB/s | 1.00x |
| row sum 8 x 4M fp32 | split 5 per row (40 programs) | - | 12 | 40 | 100% | 0.761 | 176 GB/s | 1.80x |
| row sum 8 x 4M fp32 | split 20 per row (160 programs) | - | 12 | 160 | 100% | 0.768 | 175 GB/s | 1.79x |
