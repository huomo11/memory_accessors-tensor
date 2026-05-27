#!/usr/bin/env bash
#SBATCH --job-name=mkl_three_variants
#SBATCH --partition=c003t
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --comment=alama
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export MKLROOT=/opt/intel/oneapi/mkl/latest
export LD_LIBRARY_PATH="$MKLROOT/lib/intel64:${LD_LIBRARY_PATH:-}"
export MKL_THREADING_LAYER=SEQUENTIAL
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

bash scripts/run_mkl_three_variants.sh
