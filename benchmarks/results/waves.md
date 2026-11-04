### Tail effects vs grid size (NVIDIA GeForce RTX 3050 Ti Laptop GPU, 20 SMs, K=4096)

| config | blocks/SM | wave-model error | SM-balance-model error | ms @ 20 tiles | ms @ 21 tiles | ms @ 40 tiles |
|---|---|---|---|---|---|---|
| 128x128x32 w8 s3 | 1 | 5.8% | 5.8% | 0.359 | 0.714 | 0.717 |
| 64x64x32 w4 s3 | 5 | 44.6% | 2.8% | 0.100 | 0.186 | 0.184 |
