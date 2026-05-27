# memory_accessors-tensor

`memory_accessors-tensor` is now an MKL-focused sparse-dense TTM kernel benchmark.

The benchmark studies mode-wise sparse-dense TTM as a CSR SpMM:

```text
Y = A F
```

where `A` is the sparse matrix from tensor mode unfolding, `F` is the dense factor/sketch matrix, and `Y` is the dense output.

This repository does not implement Tucker, HOSVD, STHOSVD, R-STHOSVD, SP-STHOSVD, GPU kernels, CUDA, or a general tensor library.

## MKL Three-Line Experiment

Formal performance runs focus on exactly three public variants:

- `mkl_fp64`: fp64 CSR `A`, fp64 dense `F`, fp64 output `Y`, using `mkl_sparse_d_mm`.
- `mkl_fp32`: fp32 CSR `A`, fp32 dense `F`, fp32 output `Y`, using `mkl_sparse_s_mm`; `rel_error` is measured against `mkl_fp64`.
- `mkl_mixed_factor_fp32_storage_fp64_compute`: fp64 CSR `A`, fp32 factor storage, fp64 factor workspace, fp64 output `Y`, using `mkl_sparse_d_mm`.

The mixed variant is the current low-storage high-compute candidate: the factor is stored in fp32, moved to a fp64 workspace by the accessor, and computed by MKL in fp64.

Older hand-written sparse-loop experiments are archived in the code history and docs for context, but they are no longer public benchmark variants.

## Build With MKL

Official results should be produced on compute nodes with MKL enabled:

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

If MKL is not found, the project falls back to a custom CSR backend so the code still builds, but formal results should use `backend=mkl` in the CSV.

## Smoke

```bash
bash scripts/run_smoke.sh
python3 scripts/check_kernel_results.py results/smoke.csv
```

Manual smoke:

```bash
./build-mkl/ttm_bench \
  --output results/smoke.csv \
  --dims 16,12,10 \
  --nnz 1000 \
  --ranks 4 \
  --modes 0 \
  --threads 1 \
  --repeats 1 \
  --seed 42 \
  --variants mkl_fp64,mkl_fp32,mkl_mixed_factor_fp32_storage_fp64_compute
```

## Formal Slurm Run

Do not run formal measurements on a login node. Submit:

```bash
sbatch jobs/mkl_three_variants_1c.slurm
```

The job runs:

- dims `512,512,512`
- nnz `1000000`
- ranks `64,128,256`
- mode `0`
- threads `1`
- repeats `10`
- seed `42`
- variants `mkl_fp64,mkl_fp32,mkl_mixed_factor_fp32_storage_fp64_compute`

Output:

```text
results/mkl_three_variants.csv
```

## Check Results

```bash
python3 scripts/check_kernel_results.py results/mkl_three_variants.csv
```

The checker reports median total/kernel/compute time, relative error, speedups against `mkl_fp64`, and backend counts. Confirm the three public variants report `backend=mkl` for formal results.

## CSV Fields

```text
total_ms,
format_prepare_ms,
upcast_prepare_ms,
compute_ms,
kernel_ms,
index_storage_bytes,
value_storage_bytes,
factor_storage_bytes,
output_storage_bytes,
index_logical_read_bytes,
value_logical_read_bytes,
factor_logical_read_bytes,
factor_compute_logical_read_bytes,
output_logical_write_bytes,
rel_error,
rank,
nnz,
mode,
thread_count,
seed,
variant,
dim0,
dim1,
dim2,
repeat,
backend,
csr_nrows,
csr_ncols,
csr_nnz
```

`format_prepare_ms` covers CSR construction from synthetic sparse COO tensor data. `upcast_prepare_ms` covers fp32 factor storage to fp64 workspace conversion for the mixed variant. `compute_ms` is the backend SpMM time. `kernel_ms = upcast_prepare_ms + compute_ms`.
