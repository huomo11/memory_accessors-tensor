# Sparse-Dense TTM Microbenchmark

This is a small C++17 CPU-only benchmark for validating whether dense factor
low-storage / high-compute is useful in sparse-dense tensor-times-matrix (TTM).
It intentionally does not implement a full Tucker decomposition library.

The benchmark generates a synthetic 3D sparse tensor in COO format and performs
mode-wise sparse TTM:

```text
SparseTensorCOO x DenseMatrix(mode)
```

Each nonzero contributes `rank` multiply-adds into a dense output whose target
mode is replaced by the factor rank.

## Variants

The CSV contains one row for each tested `(mode, rank, variant)` combination:

1. `factor_fp64_compute_fp64`: sparse value fp64, factor fp64 storage, fp64 compute
2. `factor_fp32_compute_fp64`: sparse value fp64, factor fp32 storage, fp64 compute
3. `factor_fp32_compute_fp32`: sparse value fp64, factor fp32 storage, fp32 compute
4. `value_fp32_factor_fp64_compute_fp64`: sparse value fp32 storage, factor fp64, fp64 compute

Variant 1 is used as the reference output for `rel_error`.

## Build

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

## Run

```powershell
.\build\Release\ttm_bench.exe --output results.csv --dims 128,128,128 --nnz 200000 --ranks 16,32,64 --modes 0,1,2 --threads 8 --repeats 3
```

On single-config generators the executable may be:

```powershell
.\build\ttm_bench.exe --output results.csv
```

## CSV Columns

The benchmark emits one row per `(mode, rank, variant, repeat)` case. The core
columns are emitted first:

```text
total_ms,format_prepare_ms,upcast_prepare_ms,compute_ms,index_storage_bytes,value_storage_bytes,factor_storage_bytes,output_storage_bytes,index_logical_read_bytes,value_logical_read_bytes,factor_logical_read_bytes,output_logical_write_bytes,rel_error,rank,nnz,mode,thread_count,seed
```

Additional context columns:

```text
variant,dim0,dim1,dim2,repeat
```

Column meanings:

- `total_ms`: sum of `format_prepare_ms + upcast_prepare_ms + compute_ms`.
- `format_prepare_ms`: COO format build overhead for the selected mode:
  computing output rows, sorting by output row, and building output row groups.
- `upcast_prepare_ms`: time to upcast low-storage fp32 factor or value arrays to
  fp64 workspaces for fp64 compute variants.
- `compute_ms`: core multiply-add phase after format preparation and any upcast.
- `index_storage_bytes`: COO index storage footprint, `nnz * 3 * sizeof(uint32)`.
- `value_storage_bytes`: sparse value storage footprint for the variant.
- `factor_storage_bytes`: dense factor storage footprint for the variant.
- `output_storage_bytes`: dense output footprint for the compute/output type.
- `index_logical_read_bytes`: estimated index reads, `nnz * 3 * sizeof(uint32)`.
- `value_logical_read_bytes`: estimated value reads using value storage type.
- `factor_logical_read_bytes`: estimated factor reads,
  `nnz * rank * sizeof(factor_storage_type)`.
- `output_logical_write_bytes`: estimated output writes,
  `max(output_storage_bytes, nnz * rank * sizeof(output_type))`.
- `rel_error`: relative L2 error against `factor_fp64_compute_fp64`.
- `seed`: RNG seed used to generate the sparse tensor and dense factors.

## Notes

- The sparse tensor is generated once from `seed`. For each `(mode, rank)`, the
  fp32 factor is rounded from the same fp64 factor, so all variants use matching
  initialization.
- The current executable builds the sorted/grouped COO format once per mode and
  reports that `format_prepare_ms` on each matching row. Treat it as format build
  overhead. A kernel-only analysis can focus on `compute_ms`, or reuse the
  prepared format and amortize this cost.
- Multi-threading uses `std::thread` and partitions the sorted output-row
  ranges across threads, avoiding atomics.
- Indices are stored as 32-bit integers.

## Sweep

Run the paper-oriented sweep:

```powershell
.\scripts\run_kernel_sweep.ps1 -Exe .\build\Release\ttm_bench.exe -Output kernel_sweep.csv
```

The script scans:

```text
ranks: 8 16 32 64 128 256
modes: 0 1 2
nnz: 100000 500000 1000000
threads: 1 4 8 16
repeats: 5
```

## Plot

```powershell
python .\scripts\plot_kernel_results.py kernel_sweep.csv --out-dir plots
```

The plotting script writes:

- `rank_vs_total_ms.svg`
- `rank_vs_rel_error.svg`
- `rank_vs_traffic_breakdown.svg`
- `variant_vs_phase_time.svg`
- `threads_vs_speedup.svg`

It uses only the Python standard library and writes SVG files.
