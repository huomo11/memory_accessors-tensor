#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j

mkdir -p results

OUT="results/mkl_fp64_profile_threads.csv"
rm -f "${OUT}"

for threads in 1 2 4 8 16 20 32; do
  TMP="results/.threads_${threads}.csv"
  ./build/ttm_profile \
    --output "${TMP}" \
    --dims 256,256,256 \
    --nnz 1000000 \
    --ranks 64,128,256 \
    --modes 0,1,2 \
    --threads "${threads}" \
    --repeats 5 \
    --seed 1

  if [[ ! -f "${OUT}" ]]; then
    cat "${TMP}" > "${OUT}"
  else
    tail -n +2 "${TMP}" >> "${OUT}"
  fi
  rm -f "${TMP}"
done

python3 scripts/analyze_profile_csv.py "${OUT}"
