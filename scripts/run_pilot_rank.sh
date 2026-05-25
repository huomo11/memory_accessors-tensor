#!/usr/bin/env bash
set -euo pipefail

mkdir -p build results logs
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/ttm_bench \
  --output results/pilot_rank.csv \
  --dims 512,512,512 \
  --nnz 1000000 \
  --ranks 8,16,32,64,128,256 \
  --modes 0 \
  --threads 20 \
  --repeats 5 \
  --seed 42

python3 scripts/check_kernel_results.py results/pilot_rank.csv
python3 scripts/plot_kernel_results.py results/pilot_rank.csv --out-dir results/pilot_rank_plots
