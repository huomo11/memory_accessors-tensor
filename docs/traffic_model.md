# Traffic Model

This Stage 1 model separates sparse COO traffic from dense factor traffic.

For each nonzero `x(i,j,k)` in a mode-wise TTM, the kernel reads:

- 3 COO indices: `3 * sizeof(uint32_t)`
- one sparse value: `sizeof(value_storage_type)`
- `rank` dense factor values: `rank * sizeof(factor_storage_type)`

It writes or updates `rank` output values for the selected output row.

## Why sparse value compression has limited upside

Sparse values contribute one scalar per nonzero. Changing sparse value storage from fp64 to fp32 saves 4 bytes per nonzero. That is useful, but the benefit does not scale with rank.

## Why dense factor traffic can dominate

Dense factor traffic scales as `nnz * rank * sizeof(factor_storage_type)`. At rank 8 it is already comparable to COO index traffic; at ranks 64, 128, and 256 it can dominate the logical traffic model.

Per-nnz read traffic:

```text
index traffic = 3 * 4 bytes = 12 bytes
value fp64 traffic = 8 bytes
value fp32 traffic = 4 bytes
factor fp64 traffic = rank * 8 bytes
factor fp32 traffic = rank * 4 bytes
```

Stage 1 reports:

```text
index_logical_read_bytes = nnz * 3 * sizeof(uint32_t)
value_logical_read_bytes = nnz * sizeof(value_storage_type)
factor_logical_read_bytes = nnz * rank * sizeof(factor_storage_type)
factor_compute_logical_read_bytes = nnz * rank * sizeof(factor_compute_read_type)
output_logical_write_bytes = max(output_storage_bytes, nnz * rank * sizeof(output_type))
```

The Stage 1.5 benchmark separates storage-side factor traffic from compute-side factor traffic. The global-upcast fp32/fp64 variant stores fp32 factors but computes from a whole fp64 workspace, so its compute-side factor reads are fp64. The on-the-fly fp32/fp64 variant reads fp32 factors in the compute loop and casts each element to double before multiply-add.

The Stage 2 blocked accessor keeps fp32 factor storage and upcasts one factor-row tile to a fp64 workspace. Its storage-side and compute-side factor read estimate remains `nnz * rank * sizeof(float)`, while `tile_workspace_bytes = min(tile_rows, factor_rows) * rank * sizeof(double)` records the temporary fp64 workspace footprint.

Stage 2.5 keeps the same logical factor traffic model but changes sparse entry layout. The 2D blocked layout sorts by output-row block and factor-row tile, aiming to balance factor workspace reuse with output locality.

Stage 2 will measure block-wise factor accessor traffic more directly.
