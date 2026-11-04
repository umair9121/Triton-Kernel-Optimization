### Occupancy model vs `cuOccupancyMaxActiveBlocksPerMultiprocessor` (NVIDIA GeForce RTX 3050 Ti Laptop GPU)

**207/207 configurations match.** Limiting resource: block slots (4), does not fit (2), registers (88), shared memory (104), warps (9)

| config | regs/thread | shared B | predicted blocks/SM | driver | limiter |
|---|---|---|---|---|---|
| matmul 32x32x32 w2 s2 | 72 | 4096 | 14 | 14 | registers |
| matmul 32x32x32 w2 s3 | 72 | 8192 | 11 | 11 | shared memory |
| matmul 32x32x32 w2 s4 | 72 | 12288 | 7 | 7 | shared memory |
| matmul 32x32x32 w2 s5 | 72 | 16384 | 5 | 5 | shared memory |
| matmul 32x32x32 w4 s2 | 48 | 4096 | 10 | 10 | registers |
| matmul 32x32x32 w4 s3 | 54 | 8192 | 9 | 9 | registers |
| matmul 32x32x32 w4 s4 | 54 | 12288 | 7 | 7 | shared memory |
| matmul 32x32x32 w4 s5 | 54 | 16384 | 5 | 5 | shared memory |
| matmul 32x32x32 w8 s2 | 40 | 4096 | 6 | 6 | warps |
| matmul 32x32x32 w8 s3 | 46 | 8192 | 5 | 5 | registers |
| matmul 32x32x32 w8 s4 | 46 | 12288 | 5 | 5 | registers |
| matmul 32x32x32 w8 s5 | 46 | 16384 | 5 | 5 | registers |
| matmul 32x32x64 w2 s2 | 80 | 8192 | 11 | 11 | shared memory |
| matmul 32x32x64 w2 s3 | 80 | 16384 | 5 | 5 | shared memory |
| matmul 32x32x64 w2 s4 | 80 | 24576 | 4 | 4 | shared memory |
| matmul 32x32x64 w2 s5 | 80 | 32768 | 3 | 3 | shared memory |
| matmul 32x32x64 w4 s2 | 56 | 8192 | 9 | 9 | registers |
| matmul 32x32x64 w4 s3 | 56 | 16384 | 5 | 5 | shared memory |
| matmul 32x32x64 w4 s4 | 56 | 24576 | 4 | 4 | shared memory |
| matmul 32x32x64 w4 s5 | 56 | 32768 | 3 | 3 | shared memory |
| matmul 32x32x64 w8 s2 | 44 | 8192 | 5 | 5 | registers |
| matmul 32x32x64 w8 s3 | 41 | 16384 | 5 | 5 | registers |
| matmul 32x32x64 w8 s4 | 42 | 24576 | 4 | 4 | shared memory |
| matmul 32x32x64 w8 s5 | 41 | 32768 | 3 | 3 | shared memory |
| matmul 32x64x32 w2 s2 | 108 | 6144 | 8 | 8 | registers |
| matmul 32x64x32 w2 s3 | 112 | 12288 | 7 | 7 | shared memory |
| matmul 32x64x32 w2 s4 | 112 | 18432 | 5 | 5 | shared memory |
| matmul 32x64x32 w2 s5 | 112 | 24576 | 4 | 4 | shared memory |
| matmul 32x64x32 w4 s2 | 72 | 6144 | 7 | 7 | registers |
| matmul 32x64x32 w4 s3 | 72 | 12288 | 7 | 7 | registers |
| matmul 32x64x32 w4 s4 | 72 | 18432 | 5 | 5 | shared memory |
| matmul 32x64x32 w4 s5 | 72 | 24576 | 4 | 4 | shared memory |
| matmul 32x64x32 w8 s2 | 46 | 6144 | 5 | 5 | registers |
| matmul 32x64x32 w8 s3 | 54 | 12288 | 4 | 4 | registers |
| matmul 32x64x32 w8 s4 | 54 | 18432 | 4 | 4 | registers |
| matmul 32x64x32 w8 s5 | 54 | 24576 | 4 | 4 | registers |
| matmul 32x64x64 w2 s2 | 128 | 12288 | 7 | 7 | shared memory |
| matmul 32x64x64 w2 s3 | 128 | 24576 | 4 | 4 | shared memory |
| matmul 32x64x64 w2 s4 | 128 | 36864 | 2 | 2 | shared memory |
| matmul 32x64x64 w2 s5 | 128 | 49152 | 2 | 2 | shared memory |
| matmul 32x64x64 w4 s2 | 72 | 12288 | 7 | 7 | registers |
| matmul 32x64x64 w4 s3 | 72 | 24576 | 4 | 4 | shared memory |
| matmul 32x64x64 w4 s4 | 72 | 36864 | 2 | 2 | shared memory |
| matmul 32x64x64 w4 s5 | 72 | 49152 | 2 | 2 | shared memory |
| matmul 32x64x64 w8 s2 | 48 | 12288 | 5 | 5 | registers |
| matmul 32x64x64 w8 s3 | 58 | 24576 | 4 | 4 | registers |
| matmul 32x64x64 w8 s4 | 58 | 36864 | 2 | 2 | shared memory |
| matmul 32x64x64 w8 s5 | 58 | 49152 | 2 | 2 | shared memory |
| matmul 32x128x32 w2 s2 | 168 | 10240 | 6 | 6 | registers |
| matmul 32x128x32 w2 s3 | 196 | 20480 | 4 | 4 | registers |
| matmul 32x128x32 w2 s4 | 206 | 30720 | 3 | 3 | shared memory |
| matmul 32x128x32 w2 s5 | 196 | 40960 | 2 | 2 | shared memory |
| matmul 32x128x32 w4 s2 | 96 | 10240 | 5 | 5 | registers |
| matmul 32x128x32 w4 s3 | 112 | 20480 | 4 | 4 | registers |
| matmul 32x128x32 w4 s4 | 112 | 30720 | 3 | 3 | shared memory |
| matmul 32x128x32 w4 s5 | 112 | 40960 | 2 | 2 | shared memory |
| matmul 32x128x32 w8 s2 | 76 | 10240 | 3 | 3 | registers |
| matmul 32x128x32 w8 s3 | 78 | 20480 | 3 | 3 | registers |
| matmul 32x128x32 w8 s4 | 78 | 30720 | 3 | 3 | registers |
| matmul 32x128x32 w8 s5 | 78 | 40960 | 2 | 2 | shared memory |
| matmul 32x128x64 w2 s2 | 224 | 20480 | 4 | 4 | registers |
| matmul 32x128x64 w2 s3 | 232 | 40960 | 2 | 2 | shared memory |
| matmul 32x128x64 w2 s4 | 252 | 61440 | 1 | 1 | shared memory |
| matmul 32x128x64 w2 s5 | 232 | 81920 | 1 | 1 | shared memory |
| matmul 32x128x64 w4 s2 | 128 | 20480 | 4 | 4 | registers |
| matmul 32x128x64 w4 s3 | 128 | 40960 | 2 | 2 | shared memory |
| matmul 32x128x64 w4 s4 | 128 | 61440 | 1 | 1 | shared memory |
| matmul 32x128x64 w4 s5 | 128 | 81920 | 1 | 1 | shared memory |
| matmul 32x128x64 w8 s2 | 76 | 20480 | 3 | 3 | registers |
| matmul 32x128x64 w8 s3 | 80 | 40960 | 2 | 2 | shared memory |
| matmul 32x128x64 w8 s4 | 80 | 61440 | 1 | 1 | shared memory |
| matmul 32x128x64 w8 s5 | 80 | 81920 | 1 | 1 | shared memory |
| matmul 64x32x32 w2 s2 | 96 | 6144 | 10 | 10 | registers |
| matmul 64x32x32 w2 s3 | 112 | 12288 | 7 | 7 | shared memory |
| matmul 64x32x32 w2 s4 | 112 | 18432 | 5 | 5 | shared memory |
| matmul 64x32x32 w2 s5 | 112 | 24576 | 4 | 4 | shared memory |
| matmul 64x32x32 w4 s2 | 64 | 6144 | 8 | 8 | registers |
| matmul 64x32x32 w4 s3 | 64 | 12288 | 7 | 7 | shared memory |
| matmul 64x32x32 w4 s4 | 66 | 18432 | 5 | 5 | shared memory |
| matmul 64x32x32 w4 s5 | 64 | 24576 | 4 | 4 | shared memory |
| matmul 64x32x32 w8 s2 | 47 | 6144 | 5 | 5 | registers |
| matmul 64x32x32 w8 s3 | 56 | 12288 | 4 | 4 | registers |
| matmul 64x32x32 w8 s4 | 56 | 18432 | 4 | 4 | registers |
| matmul 64x32x32 w8 s5 | 56 | 24576 | 4 | 4 | registers |
| matmul 64x32x64 w2 s2 | 128 | 12288 | 7 | 7 | shared memory |
| matmul 64x32x64 w2 s3 | 128 | 24576 | 4 | 4 | shared memory |
| matmul 64x32x64 w2 s4 | 128 | 36864 | 2 | 2 | shared memory |
| matmul 64x32x64 w2 s5 | 128 | 49152 | 2 | 2 | shared memory |
| matmul 64x32x64 w4 s2 | 72 | 12288 | 7 | 7 | registers |
| matmul 64x32x64 w4 s3 | 72 | 24576 | 4 | 4 | shared memory |
| matmul 64x32x64 w4 s4 | 72 | 36864 | 2 | 2 | shared memory |
| matmul 64x32x64 w4 s5 | 72 | 49152 | 2 | 2 | shared memory |
| matmul 64x32x64 w8 s2 | 51 | 12288 | 4 | 4 | registers |
| matmul 64x32x64 w8 s3 | 58 | 24576 | 4 | 4 | registers |
| matmul 64x32x64 w8 s4 | 58 | 36864 | 2 | 2 | shared memory |
| matmul 64x32x64 w8 s5 | 58 | 49152 | 2 | 2 | shared memory |
| matmul 64x64x32 w2 s2 | 164 | 8192 | 6 | 6 | registers |
| matmul 64x64x32 w2 s3 | 168 | 16384 | 5 | 5 | shared memory |
| matmul 64x64x32 w2 s4 | 168 | 24576 | 4 | 4 | shared memory |
| matmul 64x64x32 w2 s5 | 168 | 32768 | 3 | 3 | shared memory |
| matmul 64x64x32 w4 s2 | 96 | 8192 | 5 | 5 | registers |
| matmul 64x64x32 w4 s3 | 96 | 16384 | 5 | 5 | registers |
| matmul 64x64x32 w4 s4 | 96 | 24576 | 4 | 4 | shared memory |
| matmul 64x64x32 w4 s5 | 96 | 32768 | 3 | 3 | shared memory |
| matmul 64x64x32 w8 s2 | 62 | 9216 | 4 | 4 | registers |
| matmul 64x64x32 w8 s3 | 64 | 16384 | 4 | 4 | registers |
| matmul 64x64x32 w8 s4 | 64 | 24576 | 4 | 4 | registers |
| matmul 64x64x32 w8 s5 | 64 | 32768 | 3 | 3 | shared memory |
| matmul 64x64x64 w2 s2 | 200 | 16384 | 4 | 4 | registers |
| matmul 64x64x64 w2 s3 | 202 | 32768 | 3 | 3 | shared memory |
| matmul 64x64x64 w2 s4 | 202 | 49152 | 2 | 2 | shared memory |
| matmul 64x64x64 w2 s5 | 202 | 65536 | 1 | 1 | shared memory |
| matmul 64x64x64 w4 s2 | 116 | 16384 | 4 | 4 | registers |
| matmul 64x64x64 w4 s3 | 124 | 32768 | 3 | 3 | shared memory |
| matmul 64x64x64 w4 s4 | 128 | 49152 | 2 | 2 | shared memory |
| matmul 64x64x64 w4 s5 | 124 | 65536 | 1 | 1 | shared memory |
| matmul 64x64x64 w8 s2 | 78 | 16384 | 3 | 3 | registers |
| matmul 64x64x64 w8 s3 | 78 | 32768 | 3 | 3 | registers |
| matmul 64x64x64 w8 s4 | 78 | 49152 | 2 | 2 | shared memory |
| matmul 64x64x64 w8 s5 | 78 | 65536 | 1 | 1 | shared memory |
| matmul 64x128x32 w4 s2 | 162 | 12288 | 3 | 3 | registers |
| matmul 64x128x32 w4 s3 | 168 | 24576 | 3 | 3 | registers |
| matmul 64x128x32 w4 s4 | 168 | 36864 | 2 | 2 | shared memory |
| matmul 64x128x32 w4 s5 | 168 | 49152 | 2 | 2 | shared memory |
| matmul 64x128x32 w8 s2 | 100 | 12288 | 2 | 2 | registers |
| matmul 64x128x32 w8 s3 | 108 | 24576 | 2 | 2 | registers |
| matmul 64x128x32 w8 s4 | 108 | 36864 | 2 | 2 | registers |
| matmul 64x128x32 w8 s5 | 108 | 49152 | 2 | 2 | registers |
| matmul 64x128x64 w4 s2 | 168 | 24576 | 3 | 3 | registers |
| matmul 64x128x64 w4 s3 | 168 | 49152 | 2 | 2 | shared memory |
| matmul 64x128x64 w4 s4 | 168 | 73728 | 1 | 1 | shared memory |
| matmul 64x128x64 w4 s5 | 168 | 98304 | 1 | 1 | shared memory |
| matmul 64x128x64 w8 s2 | 112 | 24576 | 2 | 2 | registers |
| matmul 64x128x64 w8 s3 | 120 | 49152 | 2 | 2 | registers |
| matmul 64x128x64 w8 s4 | 124 | 73728 | 1 | 1 | shared memory |
| matmul 64x128x64 w8 s5 | 120 | 98304 | 1 | 1 | shared memory |
| matmul 128x32x32 w2 s2 | 168 | 10240 | 6 | 6 | registers |
| matmul 128x32x32 w2 s3 | 168 | 20480 | 4 | 4 | shared memory |
| matmul 128x32x32 w2 s4 | 168 | 30720 | 3 | 3 | shared memory |
| matmul 128x32x32 w2 s5 | 168 | 40960 | 2 | 2 | shared memory |
| matmul 128x32x32 w4 s2 | 94 | 10240 | 5 | 5 | registers |
| matmul 128x32x32 w4 s3 | 96 | 20480 | 4 | 4 | shared memory |
| matmul 128x32x32 w4 s4 | 96 | 30720 | 3 | 3 | shared memory |
| matmul 128x32x32 w4 s5 | 96 | 40960 | 2 | 2 | shared memory |
| matmul 128x32x32 w8 s2 | 64 | 10240 | 4 | 4 | registers |
| matmul 128x32x32 w8 s3 | 74 | 20480 | 3 | 3 | registers |
| matmul 128x32x32 w8 s4 | 75 | 30720 | 3 | 3 | registers |
| matmul 128x32x32 w8 s5 | 74 | 40960 | 2 | 2 | shared memory |
| matmul 128x32x64 w2 s2 | 196 | 20480 | 4 | 4 | registers |
| matmul 128x32x64 w2 s3 | 202 | 40960 | 2 | 2 | shared memory |
| matmul 128x32x64 w2 s4 | 206 | 61440 | 1 | 1 | shared memory |
| matmul 128x32x64 w2 s5 | 202 | 81920 | 1 | 1 | shared memory |
| matmul 128x32x64 w4 s2 | 128 | 20480 | 4 | 4 | registers |
| matmul 128x32x64 w4 s3 | 128 | 40960 | 2 | 2 | shared memory |
| matmul 128x32x64 w4 s4 | 128 | 61440 | 1 | 1 | shared memory |
| matmul 128x32x64 w4 s5 | 128 | 81920 | 1 | 1 | shared memory |
| matmul 128x32x64 w8 s2 | 70 | 20480 | 3 | 3 | registers |
| matmul 128x32x64 w8 s3 | 74 | 40960 | 2 | 2 | shared memory |
| matmul 128x32x64 w8 s4 | 74 | 61440 | 1 | 1 | shared memory |
| matmul 128x32x64 w8 s5 | 74 | 81920 | 1 | 1 | shared memory |
| matmul 128x64x32 w4 s2 | 158 | 12288 | 3 | 3 | registers |
| matmul 128x64x32 w4 s3 | 168 | 24576 | 3 | 3 | registers |
| matmul 128x64x32 w4 s4 | 168 | 36864 | 2 | 2 | shared memory |
| matmul 128x64x32 w4 s5 | 168 | 49152 | 2 | 2 | shared memory |
| matmul 128x64x32 w8 s2 | 94 | 12288 | 2 | 2 | registers |
| matmul 128x64x32 w8 s3 | 106 | 24576 | 2 | 2 | registers |
| matmul 128x64x32 w8 s4 | 104 | 36864 | 2 | 2 | registers |
| matmul 128x64x32 w8 s5 | 106 | 49152 | 2 | 2 | registers |
| matmul 128x64x64 w4 s2 | 168 | 24576 | 3 | 3 | registers |
| matmul 128x64x64 w4 s3 | 168 | 49152 | 2 | 2 | shared memory |
| matmul 128x64x64 w4 s4 | 168 | 73728 | 1 | 1 | shared memory |
| matmul 128x64x64 w4 s5 | 168 | 98304 | 1 | 1 | shared memory |
| matmul 128x64x64 w8 s2 | 116 | 24576 | 2 | 2 | registers |
| matmul 128x64x64 w8 s3 | 122 | 49152 | 2 | 2 | registers |
| matmul 128x64x64 w8 s4 | 122 | 73728 | 1 | 1 | shared memory |
| matmul 128x64x64 w8 s5 | 122 | 98304 | 1 | 1 | shared memory |
| matmul 128x128x32 w4 s2 | 246 | 16384 | 2 | 2 | registers |
| matmul 128x128x32 w4 s3 | 254 | 32768 | 2 | 2 | registers |
| matmul 128x128x32 w4 s4 | 254 | 49152 | 2 | 2 | registers |
| matmul 128x128x32 w4 s5 | 254 | 65536 | 1 | 1 | shared memory |
| matmul 128x128x32 w8 s2 | 150 | 17408 | 1 | 1 | registers |
| matmul 128x128x32 w8 s3 | 166 | 32768 | 1 | 1 | registers |
| matmul 128x128x32 w8 s4 | 168 | 49152 | 1 | 1 | registers |
| matmul 128x128x32 w8 s5 | 166 | 65536 | 1 | 1 | registers |
| matmul 128x128x64 w4 s2 | 254 | 32768 | 2 | 2 | registers |
| matmul 128x128x64 w4 s3 | 255 | 65536 | 1 | 1 | shared memory |
| matmul 128x128x64 w4 s4 | 255 | 98304 | 1 | 1 | shared memory |
| matmul 128x128x64 w4 s5 | - | 131072 | 0 | 0 | does not fit |
| matmul 128x128x64 w8 s2 | 166 | 32768 | 1 | 1 | registers |
| matmul 128x128x64 w8 s3 | 172 | 65536 | 1 | 1 | registers |
| matmul 128x128x64 w8 s4 | 176 | 98304 | 1 | 1 | registers |
| matmul 128x128x64 w8 s5 | - | 131072 | 0 | 0 | does not fit |
| softmax 1024 w1 | 71 | 0 | 16 | 16 | block slots |
| softmax 1024 w2 | 33 | 8 | 16 | 16 | block slots |
| softmax 1024 w4 | 25 | 16 | 12 | 12 | warps |
| softmax 1024 w8 | 20 | 32 | 6 | 6 | warps |
| softmax 1024 w16 | 20 | 64 | 3 | 3 | warps |
| softmax 2048 w1 | 92 | 0 | 16 | 16 | block slots |
| softmax 2048 w2 | 52 | 8 | 16 | 16 | block slots |
| softmax 2048 w4 | 33 | 16 | 12 | 12 | warps |
| softmax 2048 w8 | 25 | 32 | 6 | 6 | warps |
| softmax 2048 w16 | 20 | 64 | 3 | 3 | warps |
| softmax 4096 w1 | 168 | 0 | 12 | 12 | registers |
| softmax 4096 w2 | 92 | 8 | 10 | 10 | registers |
| softmax 4096 w4 | 48 | 16 | 10 | 10 | registers |
| softmax 4096 w8 | 33 | 32 | 6 | 6 | warps |
| softmax 4096 w16 | 25 | 64 | 3 | 3 | warps |
