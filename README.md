# Triton Kernel Optimization

`tritonlens` is a static analysis pass and interactive visualizer for Triton GPU kernels.
It parses Triton GPU IR (TTGIR), models how each layout maps threads to tensor elements and
shared-memory banks, and flags bank conflicts and misaligned or uncoalesced memory accesses.

Work in progress.
