#!/usr/bin/env bash
set -euo pipefail

mkdir -p build results logs
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/ttm_bench \
  --output results/rank_sweep.csv \
  --dims 512,512,512 \
  --nnz 1000000 \
  --ranks 8,16,32,64,128,256 \
  --modes 0 \
  --threads 20 \
  --repeats 5 \
  --seed 42

./build/ttm_bench \
  --output results/nnz_sweep.csv \
  --dims 512,512,512 \
  --nnz 250000,500000,1000000,2000000 \
  --ranks 64 \
  --modes 0 \
  --threads 20 \
  --repeats 5 \
  --seed 42

./build/ttm_bench \
  --output results/thread_sweep.csv \
  --dims 512,512,512 \
  --nnz 1000000 \
  --ranks 64 \
  --modes 0 \
  --threads 1,2,4,8,12,16,20 \
  --repeats 5 \
  --seed 42

python3 scripts/check_kernel_results.py results/rank_sweep.csv
python3 scripts/check_kernel_results.py results/nnz_sweep.csv
python3 scripts/check_kernel_results.py results/thread_sweep.csv
python3 scripts/plot_kernel_results.py results/rank_sweep.csv --out-dir results/rank_sweep_plots
python3 scripts/plot_kernel_results.py results/thread_sweep.csv --out-dir results/thread_sweep_plots
