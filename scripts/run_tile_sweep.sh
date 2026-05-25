#!/usr/bin/env bash
set -euo pipefail

mkdir -p build results logs
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

out="results/tile_sweep.csv"
tmp="results/tile_sweep.tmp.csv"
rm -f "$out" "$tmp"

first=1
for tile_rows in 8 16 32 64 128 256; do
  ./build/ttm_bench \
    --output "$tmp" \
    --dims 512,512,512 \
    --nnz 1000000 \
    --ranks 64,128 \
    --modes 0 \
    --threads 1 \
    --repeats 5 \
    --seed 42 \
    --tile-rows "$tile_rows"

  if [ "$first" -eq 1 ]; then
    cp "$tmp" "$out"
    first=0
  else
    tail -n +2 "$tmp" >> "$out"
  fi
done

rm -f "$tmp"
python3 scripts/check_kernel_results.py "$out"
python3 scripts/plot_kernel_results.py "$out" --out-dir results/tile_sweep_plots
