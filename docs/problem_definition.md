# Problem Definition

We study sparse tensor TTM for a 3D sparse tensor:

```text
X[i,j,k] = value
```

For each mode-wise TTM, the operation can be represented as:

```text
Y = A * F
```

`A` is a sparse matrix produced by unfolding the sparse tensor, `F` is a dense fp64 factor matrix, and `Y` is a dense fp64 output matrix.

## Mode 0

```text
output_row = j * K + k
factor_row = i
output_rows = J * K
factor_rows = I
```

## Mode 1

```text
output_row = i * K + k
factor_row = j
output_rows = I * K
factor_rows = J
```

## Mode 2

```text
output_row = i * J + j
factor_row = k
output_rows = I * J
factor_rows = K
```

The sparse matrix entry is:

```text
A[output_row, factor_row] = value
```

Duplicate coordinates and duplicate CSR columns inside a row are allowed in this profiling stage. They are not merged.
