# Traffic Model

The traffic model is intentionally simple. It is not a precise hardware traffic model; it is a reference scale for interpreting profile results.

For a CSR matrix with `output_rows`, `factor_rows`, `nnz`, and dense rank `R`:

```text
index_bytes = nnz * sizeof(int) + (output_rows + 1) * sizeof(int)
value_bytes = nnz * sizeof(double)
factor_matrix_bytes = factor_rows * R * sizeof(double)
output_matrix_bytes = output_rows * R * sizeof(double)
factor_logical_bytes = nnz * R * sizeof(double)
y_write_min_bytes = output_rows * R * sizeof(double)
```

`factor_logical_bytes` estimates the logical amount of dense factor data touched by the sparse traversal if each nonzero consumes one rank-length row. Real cache behavior can make physical traffic smaller or larger depending on locality, threading, and MKL's internal implementation.
