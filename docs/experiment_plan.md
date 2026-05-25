# Experiment Plan

## Pilot rank sweep

Primary first experiment:

- dims: `512,512,512`
- nnz: `1000000`
- ranks: `8,16,32,64,128,256`
- mode: `0`
- threads: `20`
- repeats: `5`

Expected use: show how factor traffic share and runtime change as rank grows.

## nnz sweep

Fix rank and vary nnz to test sparse traversal scaling and prepare overhead:

- dims: `512,512,512`
- nnz: `250000,500000,1000000,2000000`
- rank: `64`
- mode: `0`
- threads: `20`
- repeats: `5`

## thread sweep

Fix problem size and vary thread count:

- dims: `512,512,512`
- nnz: `1000000`
- rank: `64`
- mode: `0`
- threads: `1,2,4,8,12,16,20`
- repeats: `5`

Compute is partitioned by output row, so each thread writes disjoint output rows and no atomics are required.

## shape sweep

After pilot validation, test multiple tensor shapes and modes to capture output-row distribution effects:

- balanced: `512,512,512`
- tall mode-0: `4096,128,128`
- wide output mode-0: `128,1024,1024`
- all modes: `0,1,2`
