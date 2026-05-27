#!/usr/bin/env python3
from __future__ import print_function

import csv
import sys


VARIANTS = (
    "mkl_fp64",
    "mkl_fp32",
    "mkl_mixed_factor_fp32_storage_fp64_compute",
)


def median(values):
    vals = sorted(values)
    if not vals:
        return 0.0
    n = len(vals)
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def f(row, key):
    return float(row[key])


def speedup(base, other, field):
    b = median([f(r, field) for r in base])
    o = median([f(r, field) for r in other])
    return b / o if o else 0.0


def main():
    if len(sys.argv) != 2:
        print("usage: check_kernel_results.py result.csv", file=sys.stderr)
        return 2

    rows = []
    with open(sys.argv[1], "r", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append(row)

    by_variant = {}
    backend_counts = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)
        key = (row["variant"], row.get("backend", ""))
        backend_counts[key] = backend_counts.get(key, 0) + 1

    print("file:", sys.argv[1])
    print("rows:", len(rows))
    print("")

    for variant in VARIANTS:
        group = by_variant.get(variant, [])
        if not group:
            print("%s: missing" % variant)
            continue
        print("%s" % variant)
        print("  median total_ms: %.6f" % median([f(r, "total_ms") for r in group]))
        print("  median kernel_ms: %.6f" % median([f(r, "kernel_ms") for r in group]))
        print("  median compute_ms: %.6f" % median([f(r, "compute_ms") for r in group]))
        print("  median rel_error: %.12g" % median([f(r, "rel_error") for r in group]))

    base = by_variant.get("mkl_fp64", [])
    for variant in ("mkl_fp32", "mkl_mixed_factor_fp32_storage_fp64_compute"):
        group = by_variant.get(variant, [])
        if base and group:
            print("")
            print("%s vs mkl_fp64" % variant)
            print("  speedup_total: %.6f" % speedup(base, group, "total_ms"))
            print("  speedup_kernel: %.6f" % speedup(base, group, "kernel_ms"))
            print("  speedup_compute: %.6f" % speedup(base, group, "compute_ms"))

    print("")
    print("backend counts")
    for key in sorted(backend_counts):
        print("  %s,%s: %d" % (key[0], key[1], backend_counts[key]))

    return 0


if __name__ == "__main__":
    sys.exit(main())
