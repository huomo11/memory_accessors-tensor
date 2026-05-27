# Baseline and Kernel Definition

The benchmark studies sparse-dense TTM through a CSR SpMM formulation:

```text
Y = A F
```

For a tensor nonzero, mode unfolding produces:

```text
csr_row = output_row
csr_col = factor_row
csr_value = tensor_value
```

The equivalent scalar operation is:

```text
Y[output_row, r] += value * F[factor_row, r]
```

for `r = 0 ... rank - 1`.

## Current Baseline

The current baseline is:

```text
mkl_fp64
```

It uses fp64 CSR values, fp64 dense factor storage, fp64 output, and MKL sparse double SpMM.

## Precision Lines

`mkl_fp32` is the full single-precision baseline:

- fp32 CSR values;
- fp32 factor;
- fp32 output;
- MKL sparse single SpMM;
- relative error measured against `mkl_fp64`.

`mkl_mixed_factor_fp32_storage_fp64_compute` is the mixed storage/compute line:

- fp64 CSR values;
- fp32 factor storage;
- fp64 factor workspace;
- fp64 output;
- MKL sparse double SpMM;
- relative error measured against `mkl_fp64`.

## Controlled Prototype History

Earlier repository stages included hand-written output-stationary / CSR-style loops and blocked/layout variants. They were controlled prototypes for understanding factor traffic and precision policy. They are no longer public variants for the main benchmark.

Current formal experiments should use only:

```text
mkl_fp64
mkl_fp32
mkl_mixed_factor_fp32_storage_fp64_compute
```
