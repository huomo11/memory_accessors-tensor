#!/usr/bin/env python3
from __future__ import print_function

import csv
import sys


BASE = "factor_fp64_compute_fp64"
ONFLY = "factor_fp32_onfly_compute_fp64"
GLOBAL = "factor_fp32_compute_fp64"


GROUP_KEYS = (
    "rank",
    "nnz",
    "mode",
    "thread_count",
    "seed",
    "dim0",
    "dim1",
    "dim2",
    "repeat",
    "tile_rows",
    "output_block_rows",
)


def f(row, key):
    if key == "kernel_ms" and key not in row:
        return float(row["upcast_prepare_ms"]) + float(row["compute_ms"])
    return float(row[key])


def median(values):
    vals = sorted(values)
    if not vals:
        return 0.0
    n = len(vals)
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def group_key(row):
    values = []
    for key in GROUP_KEYS:
        values.append(row.get(key, ""))
    return tuple(values)


def factor_compute_share(row):
    idx = f(row, "index_logical_read_bytes")
    val = f(row, "value_logical_read_bytes")
    fac = f(row, "factor_compute_logical_read_bytes")
    denom = idx + val + fac
    return fac / denom if denom else 0.0


def add_speedups(label, base, test, out):
    if not test:
        return
    out.append({
        "label": label,
        "rank": test["rank"],
        "total": f(base, "total_ms") / f(test, "total_ms") if f(test, "total_ms") else 0.0,
        "kernel": f(base, "kernel_ms") / f(test, "kernel_ms") if f(test, "kernel_ms") else 0.0,
        "compute": f(base, "compute_ms") / f(test, "compute_ms") if f(test, "compute_ms") else 0.0,
        "rel_error": f(test, "rel_error"),
        "factor_compute_share": factor_compute_share(test),
    })


def print_summary(label, rows):
    group = [r for r in rows if r["label"] == label]
    if not group:
        return
    print("")
    print("%s vs %s" % (label, BASE))
    print("  matched groups: %d" % len(group))
    for rank in sorted(set(r["rank"] for r in group), key=lambda x: int(float(x))):
        rg = [r for r in group if r["rank"] == rank]
        print("  rank %s:" % rank)
        print("    median speedup_total: %.6f" % median([r["total"] for r in rg]))
        print("    median speedup_kernel: %.6f" % median([r["kernel"] for r in rg]))
        print("    median speedup_compute: %.6f" % median([r["compute"] for r in rg]))
        print("    median rel_error: %.12g" % median([r["rel_error"] for r in rg]))
        print("    median factor_compute_traffic_share: %.6f" %
              median([r["factor_compute_share"] for r in rg]))


def main():
    if len(sys.argv) != 2:
        print("usage: compare_onfly_vs_fp64.py result.csv", file=sys.stderr)
        return 2

    by_group = {}
    with open(sys.argv[1], "r", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            key = group_key(row)
            by_group.setdefault(key, {})[row["variant"]] = row

    results = []
    missing_onfly = 0
    missing_global = 0
    for variants in by_group.values():
        base = variants.get(BASE)
        if not base:
            continue
        onfly = variants.get(ONFLY)
        glob = variants.get(GLOBAL)
        if onfly:
            add_speedups(ONFLY, base, onfly, results)
        else:
            missing_onfly += 1
        if glob:
            add_speedups(GLOBAL, base, glob, results)
        else:
            missing_global += 1

    print("file: %s" % sys.argv[1])
    print("parameter-matched comparison keys: %s" % ",".join(GROUP_KEYS))
    print_summary(ONFLY, results)
    print_summary(GLOBAL, results)
    if missing_onfly or missing_global:
        print("")
        print("missing groups: onfly=%d global_upcast=%d" % (missing_onfly, missing_global))
    return 0


if __name__ == "__main__":
    sys.exit(main())
