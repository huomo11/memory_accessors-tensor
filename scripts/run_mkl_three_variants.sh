#!/usr/bin/env bash
set -euo pipefail

export MKLROOT="${MKLROOT:-/opt/intel/oneapi/mkl/latest}"
export LD_LIBRARY_PATH="$MKLROOT/lib/intel64:${LD_LIBRARY_PATH:-}"
export MKL_THREADING_LAYER=SEQUENTIAL
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

mkdir -p build-mkl results logs
cmake -S . -B build-mkl \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_MKL=ON \
  -DMKL_ROOT="$MKLROOT"
cmake --build build-mkl -j 1

./build-mkl/ttm_bench \
  --output results/mkl_three_variants.csv \
  --dims 512,512,512 \
  --nnz 1000000 \
  --ranks 64,128,256 \
  --modes 0 \
  --threads 1 \
  --repeats 10 \
  --seed 42 \
  --variants mkl_fp64,mkl_fp32,mkl_mixed_factor_fp32_storage_fp64_compute

python3 scripts/check_kernel_results.py results/mkl_three_variants.csv
