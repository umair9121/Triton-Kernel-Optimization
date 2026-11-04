### Global-memory findings vs measured bandwidth (NVIDIA GeForce RTX 3050 Ti Laptop GPU)

Median of 7 interleaved rounds.

| case | variant | findings | predicted B/lane | PTX B/lane | predicted sector efficiency | measured ms | effective GB/s | speedup |
|---|---|---|---|---|---|---|---|---|
| vector add, 16M fp32 | offset views + odd n (TL001, TL006) | TL001 (info), TL006 (info), TL006 (warning) | 4 | 4 | 100% | 1.172 | 171.7 | 1.00x |
| vector add, 16M fp32 | aligned tensors, n % 16 == 0 | - | 16 | 16 | 100% | 1.159 | 173.8 | 1.01x |
| softmax, 8192 rows fp32 | 781 columns (TL001) | TL001 (info), TL006 (info) | 4 | 4 | 100% | 0.314 | 163.1 | 1.00x |
| softmax, 8192 rows fp32 | 784 columns (padded to 16) | - | 16 | 16 | 100% | 0.295 | 174.4 | 1.06x |
| layernorm, 4096x8192 fp16 | BLOCK=1024, 8 warps (TL001 info) | TL001 (info) | 8 | 8 | 100% | 1.225 | 164.3 | 1.00x |
| layernorm, 4096x8192 fp16 | BLOCK=2048, 8 warps | - | 16 | 16 | 100% | 1.272 | 158.3 | 0.96x |
| gather 4M fp32 | stride 8 (TL002) | TL002 (error) | 4 | 4 | 12% | 0.869 | 38.6 | 1.00x |
| gather 4M fp32 | stride 1 | - | 16 | 16 | 100% | 0.202 | 166.5 | 4.31x |
