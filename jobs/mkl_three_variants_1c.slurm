#!/usr/bin/env bash
#SBATCH --job-name=mkl_three_variants
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/mkl_three_variants_%j.out
#SBATCH --error=logs/mkl_three_variants_%j.err

# Add site-specific #SBATCH --partition or --account lines if needed.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export MKLROOT=/opt/intel/oneapi/mkl/latest
export LD_LIBRARY_PATH="$MKLROOT/lib/intel64:${LD_LIBRARY_PATH:-}"
export MKL_THREADING_LAYER=SEQUENTIAL
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

bash scripts/run_mkl_three_variants.sh
