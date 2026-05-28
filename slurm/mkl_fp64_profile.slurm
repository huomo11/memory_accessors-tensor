#!/usr/bin/env bash
#SBATCH --job-name=mkl_fp64_ttm_profile
#SBATCH --output=mkl_fp64_ttm_profile.%j.out
#SBATCH --error=mkl_fp64_ttm_profile.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --partition=compute

set -euo pipefail

# Adjust this for your cluster environment.
module purge
module load intel-oneapi-mkl

export OMP_PROC_BIND=close
export OMP_PLACES=cores

mkdir -p results
bash scripts/run_mkl_profile_threads.sh
