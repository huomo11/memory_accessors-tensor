# MKL Backend Plan

The current benchmark is MKL-focused and treats sparse-dense TTM as matrix SpMM:

```text
Y = A F
```

`A` is a CSR matrix from sparse tensor mode unfolding. `F` is a dense factor/sketch matrix. `Y` is dense output.

## CSR Unfolding

For a 3D tensor with shape `I x J x K`:

- mode 0: `rows = J * K`, `cols = I`
- mode 1: `rows = I * K`, `cols = J`
- mode 2: `rows = I * J`, `cols = K`

Each nonzero `x(i,j,k)` maps to:

```text
csr_row = output_row
csr_col = factor_row
csr_value = x
```

## Public Variants

`mkl_fp64`

- `A`: fp64 CSR
- `F`: fp64 dense
- `Y`: fp64 dense
- backend: `mkl_sparse_d_create_csr`, `mkl_sparse_d_mm`, `mkl_sparse_destroy`

`mkl_fp32`

- `A`: fp32 CSR
- `F`: fp32 dense
- `Y`: fp32 dense
- backend: `mkl_sparse_s_create_csr`, `mkl_sparse_s_mm`, `mkl_sparse_destroy`
- `rel_error` is measured against `mkl_fp64`

`mkl_mixed_factor_fp32_storage_fp64_compute`

- `A`: fp64 CSR
- `F`: fp32 storage
- accessor converts `F` to a fp64 workspace
- `Y`: fp64 dense
- backend: `mkl_sparse_d_create_csr`, `mkl_sparse_d_mm`, `mkl_sparse_destroy`

The mixed variant is the low-storage high-compute candidate.

## Build Policy

Use:

```bash
export MKLROOT=/opt/intel/oneapi/mkl/latest
export LD_LIBRARY_PATH=$MKLROOT/lib/intel64:$LD_LIBRARY_PATH
export MKL_THREADING_LAYER=SEQUENTIAL
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

cmake -S . -B build-mkl \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_MKL=ON \
  -DMKL_ROOT=$MKLROOT

cmake --build build-mkl -j 1
```

When MKL is found, CMake prints `MKL backend enabled`, defines `USE_MKL_BACKEND`, includes `mkl.h`, links `libmkl_rt`, and links `pthread`, `dl`, and `m` on Unix-like systems.

If MKL is not found, the code falls back to a custom CSR backend for buildability. Those fallback results are useful for smoke tests only. Formal results should report `backend=mkl`.

## Archived Exploration

Earlier versions explored hand-written output-stationary sparse loops, fp32 on-the-fly factor access, blocked factor tiles, and layout-aware sparse entry orderings. Those experiments established that factor-side storage is the relevant target, but they are no longer the public benchmark interface.

The current paper-facing path is the MKL three-line comparison above.
