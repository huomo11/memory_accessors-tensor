#!/usr/bin/env bash
#SBATCH --job-name=ttm_pilot_rank
#SBATCH --cpus-per-task=20
#SBATCH --time=01:00:00
#SBATCH --output=logs/pilot_rank_%j.out
#SBATCH --error=logs/pilot_rank_%j.err

# If your cluster requires a partition/account, add them here, for example:
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
bash scripts/run_pilot_rank.sh
