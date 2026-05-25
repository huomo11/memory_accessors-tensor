# Project Plan

## Stage 1: sparse-dense TTM factor accessor kernel proof

Implement and benchmark only 3D COO sparse tensor times dense factor matrix:

`Y[output_row, r] += value * F[factor_row, r]`

The purpose is to test whether low-storage high-compute dense factor access is more promising than low-storage sparse value access. Stage 1 intentionally uses a global-upcast prototype instead of a block-wise accessor.

## Stage 2: block-wise factor accessor

Replace whole-factor upcast with block-wise factor accessors. The target is to keep factor storage compact while making compute kernels see fp64 or otherwise compute-friendly tiles. This stage should measure tile size, cache reuse, conversion overhead, and rank sensitivity.

The first Stage 2 implementation is `factor_fp32_blocked_compute_fp64`. It groups prepared entries by factor-row tile, upcasts one fp32 factor tile to a fp64 workspace, processes entries in that tile, and then moves to the next tile. The initial implementation is serial for this variant to avoid races on shared output rows.

## Stage 3: SP-STHOSVD / randomized sparse Tucker integration

After the kernel proof is convincing, integrate the accessor into sparse randomized Tucker workflows. This repository should still avoid becoming a full tensor library; integration should focus on the sparse-dense TTM / multi-TTM kernels that dominate factor access traffic.
