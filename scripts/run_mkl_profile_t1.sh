#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j

mkdir -p results

./build/ttm_profile \
  --output results/mkl_fp64_profile_t1.csv \
  --dims 256,256,256 \
  --nnz 1000000 \
  --ranks 64,128,256 \
  --modes 0,1,2 \
  --threads 1 \
  --repeats 5 \
  --seed 1

python3 scripts/analyze_profile_csv.py results/mkl_fp64_profile_t1.csv
