# MKL fp64 Profile Plan

The first profiling target is the standard all-fp64 Intel MKL Sparse BLAS path:

```text
mkl_sparse_d_create_csr
mkl_sparse_optimize
mkl_sparse_d_mm
```

This establishes a baseline for the cost split between tensor-to-matrix layout preparation and the vendor SpMM call itself. The goal is measurement, not optimization.

This stage deliberately excludes:

- fp32
- mixed precision
- on-the-fly tensor traversal
- index compression
- value compression
- custom SpMM kernels

Each measured row in the CSV contains one mode, one rank, one thread count, and one repeat. The timed buckets make it possible to ask whether total runtime is dominated by sparse layout construction, MKL setup, dense output initialization, or the actual MKL SpMM call.
