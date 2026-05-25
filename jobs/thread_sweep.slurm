#!/usr/bin/env bash
#SBATCH --job-name=ttm_thread_sweep
#SBATCH --cpus-per-task=20
#SBATCH --time=01:00:00
#SBATCH --output=logs/thread_sweep_%j.out
#SBATCH --error=logs/thread_sweep_%j.err

# Add site-specific #SBATCH --partition or --account lines if needed.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p build results logs
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/ttm_bench \
  --output results/thread_sweep.csv \
  --dims 512,512,512 \
  --nnz 1000000 \
  --ranks 64 \
  --modes 0 \
  --threads 1,2,4,8,12,16,20 \
  --repeats 5 \
  --seed 42
python3 scripts/check_kernel_results.py results/thread_sweep.csv
python3 scripts/plot_kernel_results.py results/thread_sweep.csv --out-dir results/thread_sweep_plots
