# memory_accessors-tensor

This repository is a Stage 1 kernel proof for a CCF HPC paper direction:

> Low-storage high-compute factor accessors for sparse-dense TTM / multi-TTM in sparse randomized Tucker decomposition.

The current code does **not** implement Tucker, HOSVD, STHOSVD, R-STHOSVD, SP-STHOSVD, GPU kernels, CUDA, or a general tensor library. It only benchmarks 3D COO sparse tensor times dense factor matrix for mode-wise sparse TTM.

## Current Scope

The experiment asks whether low-storage high-compute treatment of repeatedly accessed dense factor/sketch matrices is more useful than applying the same idea to sparse values.

Stage 1 uses a global-upcast prototype:

- sparse tensor/core stays sparse;
- `factor_fp32_compute_fp64` stores factors in fp32, upcasts the whole factor to fp64 workspace, and computes in fp64;
- `factor_fp32_onfly_compute_fp64` stores factors in fp32 and casts each factor element to fp64 directly in the compute loop;
- `factor_fp32_blocked_compute_fp64` stores factors in fp32, upcasts one factor-row tile at a time to fp64 workspace, and reuses that tile while processing its nonzeros;
- `factor_fp32_2dblocked_compute_fp64` uses an output-row-block by factor-row-tile layout before applying the tile workspace;
- `value_fp32_factor_fp64_compute_fp64` stores sparse values in fp32, upcasts values to fp64 workspace, and computes in fp64.

The blocked accessors are intentionally single-threaded for now and ignore `--threads` for those variants. The factor-row blocked variant can update the same output row from different factor tiles. The 2D blocked variant groups by output-row block first, which is layout-aware and preserves more output locality, but this first version still processes blocks serially to keep correctness simple and avoid atomics.

Later stages will replace global upcast with block-wise factor accessors.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The CMake file is written for C++11 and GCC 4.8.5 style compatibility. Non-MSVC builds explicitly pass `-std=c++11` and link `Threads::Threads`.

## Controlled Baseline vs External Library Baseline

`factor_fp64_compute_fp64` is an output-stationary / CSR-style controlled fp64 baseline for precision and layout ablations. It is not yet a mature sparse library baseline. See [docs/baseline_and_kernel_definition.md](docs/baseline_and_kernel_definition.md) for the kernel definition, third-layer on-the-fly factor variant, and planned external baselines.

## CSR Backend Prototype

Stage 3 adds a CSR SpMM backend prototype. Sparse-dense TTM is unfolded as `Y = A F`, where `A` is a CSR matrix and `F` is the dense factor/sketch matrix. With `USE_MKL=OFF`, the project uses a custom CSR SpMM backend to validate the accessor/backend interface. With `USE_MKL=ON` and a valid `MKL_ROOT`, CSR variants call oneMKL sparse double CSR SpMM; if MKL is not found, the build falls back to custom. See [docs/backend_plan.md](docs/backend_plan.md).

Example CSR smoke:

```bash
./build/ttm_bench \
  --output results/stage3_smoke.csv \
  --dims 16,12,10 \
  --nnz 1000 \
  --ranks 4 \
  --modes 0 \
  --threads 1 \
  --repeats 1 \
  --seed 42 \
  --tile-rows 8 \
  --variants factor_fp64_compute_fp64,csr_fp64_factor_fp64_backend,csr_factor_fp32_global_upcast_fp64_backend,csr_factor_fp32_tiled_accessor_fp64_backend
```

## Smoke Test

```bash
bash scripts/run_smoke.sh
```

or manually:

```bash
mkdir -p results
./build/ttm_bench \
  --output results/smoke.csv \
  --dims 16,12,10 \
  --nnz 1000 \
  --ranks 4 \
  --modes 0 \
  --threads 2 \
  --repeats 1 \
  --seed 42 \
  --tile-rows 64 \
  --output-block-rows 64
```

## Pilot Rank Sweep

```bash
bash scripts/run_pilot_rank.sh
```

This runs:

- dims `512,512,512`
- nnz `1000000`
- ranks `8,16,32,64,128,256`
- mode `0`
- threads `20`
- repeats `5`
- seed `42`

Output: `results/pilot_rank.csv`.

## Full Kernel Sweep

```bash
bash scripts/run_kernel_sweep.sh
```

Outputs:

- `results/rank_sweep.csv`
- `results/nnz_sweep.csv`
- `results/thread_sweep.csv`

## Tile Sweep

```bash
bash scripts/run_tile_sweep.sh
```

This runs the Stage 2 blocked accessor sweep with tile rows `8,16,32,64,128,256`, dims `512,512,512`, nnz `1000000`, ranks `64,128`, mode `0`, threads `1`, repeats `5`, and seed `42`.

## Layout Sweep

```bash
bash scripts/run_layout_sweep.sh
```

This runs the Stage 2.5 layout-aware sweep with tile rows `16,32,64,128,256` and output block rows `16,32,64,128,256`. The main comparison is the old `factor_tile` layout versus the new `output_factor_2d` layout.

## Check Results

```bash
python3 scripts/check_kernel_results.py results/pilot_rank.csv
```

The checker uses only the Python standard library and reports median timing, median relative error, fp32-factor speedup, fp32-value speedup, and factor logical traffic share.

## Plot SVGs

```bash
python3 scripts/plot_kernel_results.py results/pilot_rank.csv --out-dir results/pilot_rank_plots
```

Generated SVGs:

- `rank_vs_total_ms.svg`
- `rank_vs_kernel_ms.svg`
- `rank_vs_compute_ms.svg`
- `rank_vs_rel_error.svg`
- `rank_vs_compute_traffic_breakdown.svg`
- `variant_vs_phase_time.svg`

## CSV Fields

Fixed schema:

`total_ms, format_prepare_ms, upcast_prepare_ms, compute_ms, kernel_ms, index_storage_bytes, value_storage_bytes, factor_storage_bytes, output_storage_bytes, index_logical_read_bytes, value_logical_read_bytes, factor_logical_read_bytes, factor_compute_logical_read_bytes, tile_workspace_bytes, output_logical_write_bytes, rel_error, rank, nnz, mode, thread_count, seed, variant, dim0, dim1, dim2, tile_rows, repeat, layout, output_block_rows, backend, csr_nrows, csr_ncols, csr_nnz, num_factor_tiles`

`format_prepare_ms` measures mode-wise COO sorting and row-start construction. `upcast_prepare_ms` measures fp32 storage to fp64 workspace conversion. `compute_ms` measures the core sparse-dense TTM loop. `kernel_ms = upcast_prepare_ms + compute_ms`. `total_ms` remains `format_prepare_ms + upcast_prepare_ms + compute_ms`.

`layout` is `output_row`, `factor_tile`, or `output_factor_2d`. `factor_logical_read_bytes` is the storage-side factor traffic estimate. `factor_compute_logical_read_bytes` is the compute-side estimate: global-upcast fp32/fp64 compute reads fp64 workspace, while on-the-fly and blocked fp32/fp64 compute read fp32 factor storage. `tile_workspace_bytes` records the fp64 tile workspace footprint used by blocked variants.
