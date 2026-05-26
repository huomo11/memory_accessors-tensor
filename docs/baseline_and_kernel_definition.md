# Baseline and Kernel Definition

This repository studies a narrow sparse-dense TTM kernel, not a full tensor decomposition system.

## Kernel Definition

For a 3D sparse tensor entry `x(i,j,k)` and a dense factor matrix `F`, the mode-wise sparse-dense TTM kernel maps each nonzero to:

```text
output_row
factor_row
```

and performs:

```text
Y[output_row, r] += value * F[factor_row, r]
```

for `r = 0 ... rank - 1`.

## Controlled FP64 Baseline

The input sparse tensor is stored as COO entries, but the benchmark does not compute directly in arbitrary COO order.

The prepare stage:

- computes `output_row` and `factor_row` for each nonzero;
- sorts entries by `output_row`;
- builds `row_starts`, equivalent to CSR-style row pointers over output rows.

The compute stage:

- traverses output rows;
- visits the nonzeros contributing to each output row;
- accumulates into `Y[output_row, :]`;
- splits work by disjoint output-row ranges when multiple threads are used, avoiding atomics.

Therefore:

```text
factor_fp64_compute_fp64 = output-stationary / CSR-style controlled fp64 baseline
```

This is a controlled in-repository baseline for precision and layout ablations.

## Third-Layer On-The-Fly Factor Variant

The third-layer factor-access variant is:

```text
factor_fp32_onfly_compute_fp64
```

It uses the same output-stationary traversal as the fp64 baseline, but changes factor access:

- sparse values remain fp64;
- factor matrix storage is fp32;
- compute reads fp32 factor elements directly;
- each factor element is cast to double inside the multiply-add;
- output accumulation remains fp64.

Conceptually:

```text
Y[output_row, r] += double(value_fp64[nz]) * double(factor_fp32[factor_row, r])
```

Therefore:

```text
factor_fp32_onfly_compute_fp64 = output-stationary + fp32 factor storage + fp64 accumulation
```

## Global Upcast vs On-The-Fly

The two fp32-factor/fp64-compute variants have different meanings.

`factor_fp32_compute_fp64` is the global-upcast prototype:

- factor is stored in fp32;
- before compute, the whole factor matrix is converted to a fp64 workspace;
- compute reads the fp64 workspace.

This reduces factor storage footprint, but it does not reduce compute-side factor read bandwidth.

`factor_fp32_onfly_compute_fp64` is the true factor-side low-storage/high-compute variant:

- factor is stored in fp32;
- no whole-factor fp64 workspace is created;
- compute reads fp32 factor values directly;
- fp32 factor values are cast to double at use.

This is the variant that tests whether compute-side factor traffic can be reduced while preserving fp64 accumulation.

## Controlled Baseline vs External Library Baseline

The current fp64 baseline is a controlled baseline, not a mature sparse library baseline.

It is suitable for:

- isolating precision-policy effects;
- comparing factor storage strategies;
- testing sparse entry layout changes;
- measuring logical traffic models under one stable traversal.

It is not enough to claim performance superiority over mature sparse computation libraries.

Future external baselines should include:

- matrix SpMM level: MKL sparse mm / oneMKL sparse BLAS;
- sparse tensor kernel level: SPLATT / HiCOO / HiParTI.

The current stage should be interpreted as mechanism validation before external library comparison.

## Current Experimental Findings

So far, the experiments support these observations:

- factor traffic dominates sparse-dense TTM logical traffic;
- fp32 sparse value storage is not a robust direction;
- global factor upcast is not effective for compute-side bandwidth reduction;
- on-the-fly fp32 factor storage with fp64 accumulation is correct and numerically stable;
- on-the-fly factor speedup is modest and rank-sensitive;
- naive blocked and layout-aware variants are sensitive to sparse entry ordering and are not final optimized kernels.

In controlled confirmation runs, `factor_fp32_onfly_compute_fp64` showed about `1.13x` kernel speedup at rank 64, but about `0.98x` to `0.99x` at ranks 128 and 256. This suggests the third layer is implemented correctly, but its current simple form is not yet a strong performance result.
