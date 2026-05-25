#!/usr/bin/env python3
from __future__ import print_function

import csv
import statistics
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
    return float(row[key])


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
        compute = median([as_float(r, "compute_ms") for r in group])
        rel = median([as_float(r, "rel_error") for r in group])
        traffic = []
        for r in group:
            idx = as_float(r, "index_logical_read_bytes")
            val = as_float(r, "value_logical_read_bytes")
            fac = as_float(r, "factor_logical_read_bytes")
            denom = idx + val + fac
            traffic.append(fac / denom if denom else 0.0)
        print("%s" % variant)
        print("  median total_ms: %.6f" % total)
        print("  median compute_ms: %.6f" % compute)
        print("  median rel_error: %.12g" % rel)
        print("  median factor traffic share: %.6f" % median(traffic))

    base_name = "factor_fp64_compute_fp64"
    if base_name in by_variant:
        base_total = median([as_float(r, "total_ms") for r in by_variant[base_name]])
        for name in ("factor_fp32_compute_fp64", "value_fp32_factor_fp64_compute_fp64"):
            if name in by_variant:
                other_total = median([as_float(r, "total_ms") for r in by_variant[name]])
                speedup = base_total / other_total if other_total else 0.0
                print("")
                print("%s speedup vs fp64 baseline: %.6f" % (name, speedup))

    return 0


if __name__ == "__main__":
    sys.exit(main())
