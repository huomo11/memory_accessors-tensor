#!/usr/bin/env python3
from __future__ import print_function

import csv
import sys


def median(values):
    vals = sorted(values)
    if not vals:
        return None
    n = len(vals)
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def as_float(row, key):
    if key == "kernel_ms" and key not in row:
        return float(row["upcast_prepare_ms"]) + float(row["compute_ms"])
    if key == "factor_compute_logical_read_bytes" and key not in row:
        return float(row["factor_logical_read_bytes"])
    return float(row[key])


def speedup_line(name, base_group, other_group):
    base_total = median([as_float(r, "total_ms") for r in base_group])
    base_kernel = median([as_float(r, "kernel_ms") for r in base_group])
    base_compute = median([as_float(r, "compute_ms") for r in base_group])
    other_total = median([as_float(r, "total_ms") for r in other_group])
    other_kernel = median([as_float(r, "kernel_ms") for r in other_group])
    other_compute = median([as_float(r, "compute_ms") for r in other_group])
    print("")
    print("%s vs fp64 baseline" % name)
    print("  speedup_total: %.6f" % (base_total / other_total if other_total else 0.0))
    print("  speedup_kernel: %.6f" % (base_kernel / other_kernel if other_kernel else 0.0))
    print("  speedup_compute: %.6f" % (base_compute / other_compute if other_compute else 0.0))


def main():
    if len(sys.argv) != 2:
        print("usage: check_kernel_results.py result.csv", file=sys.stderr)
        return 2

    rows = []
    with open(sys.argv[1], "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    by_variant = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)

    print("file:", sys.argv[1])
    print("rows:", len(rows))
    print("")

    for variant in sorted(by_variant):
        group = by_variant[variant]
        total = median([as_float(r, "total_ms") for r in group])
        kernel = median([as_float(r, "kernel_ms") for r in group])
        compute = median([as_float(r, "compute_ms") for r in group])
        rel = median([as_float(r, "rel_error") for r in group])
        storage_traffic = []
        compute_traffic = []
        for r in group:
            idx = as_float(r, "index_logical_read_bytes")
            val = as_float(r, "value_logical_read_bytes")
            fac_storage = as_float(r, "factor_logical_read_bytes")
            fac_compute = as_float(r, "factor_compute_logical_read_bytes")
            storage_denom = idx + val + fac_storage
            compute_denom = idx + val + fac_compute
            storage_traffic.append(fac_storage / storage_denom if storage_denom else 0.0)
            compute_traffic.append(fac_compute / compute_denom if compute_denom else 0.0)
        print("%s" % variant)
        print("  median total_ms: %.6f" % total)
        print("  median kernel_ms: %.6f" % kernel)
        print("  median compute_ms: %.6f" % compute)
        print("  median rel_error: %.12g" % rel)
        print("  median factor storage traffic share: %.6f" % median(storage_traffic))
        print("  median factor compute traffic share: %.6f" % median(compute_traffic))

    base_name = "factor_fp64_compute_fp64"
    if base_name in by_variant:
        for name in ("factor_fp32_global_upcast_compute_fp64",
                     "factor_fp32_compute_fp64",
                     "factor_fp32_onfly_compute_fp64",
                     "value_fp32_factor_fp64_compute_fp64"):
            if name in by_variant:
                speedup_line(name, by_variant[base_name], by_variant[name])

    return 0


if __name__ == "__main__":
    sys.exit(main())
