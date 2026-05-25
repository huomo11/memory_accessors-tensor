#!/usr/bin/env bash
set -euo pipefail

mkdir -p build results logs
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/ttm_bench \
  --output results/smoke.csv \
  --dims 16,12,10 \
  --nnz 1000 \
  --ranks 4 \
  --modes 0 \
  --threads 2 \
  --repeats 1 \
  --seed 42

echo "wrote results/smoke.csv"
