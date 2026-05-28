#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j

mkdir -p results

./build/ttm_profile \
  --output results/smoke.csv \
  --dims 32,32,32 \
  --nnz 10000 \
  --ranks 8,16 \
  --modes 0,1,2 \
  --threads 1 \
  --repeats 2 \
  --seed 1

python3 scripts/analyze_profile_csv.py results/smoke.csv
