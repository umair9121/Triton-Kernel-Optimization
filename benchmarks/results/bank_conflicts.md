### Shared-memory swizzle sweep (fp16 matmul 2048x2048x2048, NVIDIA GeForce RTX 3050 Ti Laptop GPU)

Median of 7 interleaved rounds; range in brackets.

| config | variant | predicted conflict degree | wavefronts / iter (ideal) | measured ms | slowdown | correct |
|---|---|---|---|---|---|---|
| 64x64x32 w4 s3 | triton default | 1.00x | 192 (192) | 1.174 [1.157-1.207] | 1.00x | yes |
| 64x64x32 w4 s3 | maxPhase=2 | 2.33x | 448 (192) | 1.232 [1.190-1.275] | 1.05x | yes |
| 64x64x32 w4 s3 | A unswizzled | 2.00x | 384 (192) | 1.197 [1.188-1.215] | 1.02x | yes |
| 64x64x32 w4 s3 | B unswizzled | 3.33x | 640 (192) | 1.606 [1.578-1.636] | 1.37x | yes |
| 64x64x32 w4 s3 | no swizzle | 4.33x | 832 (192) | 2.034 [2.002-2.048] | 1.73x | yes |
| 128x128x32 w4 s3 | triton default | 1.00x | 384 (384) | 1.215 [1.158-2.435] | 1.00x | yes |
| 128x128x32 w4 s3 | maxPhase=2 | 2.33x | 896 (384) | 1.200 [1.175-2.399] | 0.99x | yes |
| 128x128x32 w4 s3 | A unswizzled | 2.00x | 768 (384) | 1.188 [1.167-2.302] | 0.98x | yes |
| 128x128x32 w4 s3 | B unswizzled | 3.33x | 1280 (384) | 1.205 [1.151-2.227] | 0.99x | yes |
| 128x128x32 w4 s3 | no swizzle | 4.33x | 1664 (384) | 1.290 [1.177-2.210] | 1.06x | yes |

- `64x64x32 w4 s3`: Pearson r(predicted wavefronts, measured time) = 0.94
- `128x128x32 w4 s3`: Pearson r(predicted wavefronts, measured time) = 0.69
