# memory_accessors_tensor_profile

This repository profiles one narrow question: after a 3D sparse tensor TTM is expressed as sparse matrix times dense matrix, how is time split across the standard fp64 Intel MKL SpMM pipeline?

It is not a complete Tucker decomposition system. It does not implement low precision, mixed precision, on-the-fly tensor kernels, index compression, value compression, or custom SpMM kernels. The current stage is profiling only.

## Build

Target environment:

- Linux
- CMake
- C++17
- Intel MKL
- OpenMP

Configure with `MKLROOT` from the environment:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
```

Or pass it explicitly:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DMKLROOT=$MKLROOT
cmake --build build -j
```

CMake intentionally fails if MKL cannot be found.

## Run

```bash
./build/ttm_profile \
  --output results/mkl_fp64_profile_t1.csv \
  --dims 256,256,256 \
  --nnz 1000000 \
  --ranks 64,128,256 \
  --modes 0,1,2 \
  --threads 1 \
  --repeats 5 \
  --seed 1
```

Convenience scripts:

```bash
bash scripts/run_smoke.sh
bash scripts/run_mkl_profile_t1.sh
bash scripts/run_mkl_profile_threads.sh
```

Analyze a CSV:

```bash
python3 scripts/analyze_profile_csv.py results/mkl_fp64_profile_t1.csv
```

## Timed Buckets

The CSV records `mapping_ms`, `csr_build_ms`, `mkl_create_ms`, `mkl_optimize_ms`, `factor_init_ms`, `output_init_ms`, `mkl_spmm_ms`, `checksum_ms`, and `total_ms`.

`prepare_pct` is the fraction of total time spent in mapping, CSR build, MKL CSR creation/optimization, factor initialization, and output initialization. `mkl_spmm_pct` is the fraction spent inside `mkl_sparse_d_mm` only.
