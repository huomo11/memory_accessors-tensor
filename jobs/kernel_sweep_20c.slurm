#!/usr/bin/env bash
#SBATCH --job-name=ttm_kernel_sweep
#SBATCH --cpus-per-task=20
#SBATCH --time=04:00:00
#SBATCH --output=logs/kernel_sweep_%j.out
#SBATCH --error=logs/kernel_sweep_%j.err

# Add site-specific #SBATCH --partition or --account lines if needed.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
bash scripts/run_kernel_sweep.sh
