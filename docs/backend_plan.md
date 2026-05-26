# Backend Plan

Stage 3 separates memory-accessor policy from sparse compute.

The earlier controlled prototype directly owns sparse traversal, factor conversion, and accumulation. That is useful for mechanism ablations, but it is not the final software architecture. The Stage 3 direction is:

```text
low-precision storage
    -> accessor converts/copies to fp64 workspace
    -> sparse/dense compute backend performs SpMM
```

## CSR SpMM Interpretation

Mode-wise sparse-dense TTM can be written as:

```text
Y = A F
```

where `A` is the sparse matrix formed by mode unfolding:

- mode 0: `rows = J * K`, `cols = I`
- mode 1: `rows = I * K`, `cols = J`
- mode 2: `rows = I * J`, `cols = K`

Each tensor nonzero maps to:

```text
csr_row = output_row
csr_col = factor_row
csr_value = tensor_value
```

The new CSR path builds this matrix explicitly and calls a backend SpMM routine.

## Controlled Baseline vs Mature Backend Baseline

The existing `factor_fp64_compute_fp64` variant is an output-stationary / CSR-style controlled baseline implemented inside this repository. It is good for precision and layout ablations.

The backend path is different: sparse traversal and accumulation are delegated to a backend interface. With `USE_MKL=OFF`, the backend is a simple in-repository custom CSR SpMM used to validate the interface and data movement. With `USE_MKL=ON`, the build system probes for MKL headers but does not require them; the current code still falls back to the custom backend if MKL is unavailable.

This means Stage 3 currently validates the architecture. It does not yet claim mature-library performance.

## Stage 3 Variants

`csr_fp64_factor_fp64_backend`

- CSR `A` stores fp64 values;
- factor `F` is fp64;
- output `Y` is fp64;
- backend computes `Y = A F`.

`csr_factor_fp32_global_upcast_fp64_backend`

- factor `F` is stored in fp32;
- accessor converts the whole factor matrix to a fp64 workspace;
- backend computes with fp64 CSR values and fp64 factor workspace.

`csr_factor_fp32_tiled_accessor_fp64_backend`

- factor `F` is stored in fp32;
- CSR is split into column/factor-row tiles;
- for each factor tile, accessor converts the matching fp32 factor tile to a fp64 workspace;
- backend accumulates `Y += A_tile * F_tile_workspace`.

This is the first matrix-level version of the memory accessor idea: the accessor owns storage-to-workspace movement, while SpMM owns sparse traversal and accumulation.

## MKL / Sparse BLAS Plan

The CMake option is:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_MKL=ON
```

If `mkl.h` is not found, configuration emits a warning and continues with the custom backend. This keeps the project buildable on clusters without MKL modules loaded.

The intended MKL path is:

- map `csr_fp64_factor_fp64_backend` to MKL sparse double CSR SpMM;
- map `csr_factor_fp32_global_upcast_fp64_backend` to factor fp32 storage plus whole-factor fp64 workspace plus MKL sparse double CSR SpMM;
- keep `csr_factor_fp32_tiled_accessor_fp64_backend` as the accessor experiment, calling MKL once per CSR column tile.

After matrix-level CSR SpMM is validated, tensor-specific external baselines should be added separately:

- SPLATT;
- HiCOO;
- HiParTI.

Those are future sparse tensor kernel baselines, not part of the current Stage 3 implementation.
