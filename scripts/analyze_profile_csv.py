#!/usr/bin/env python3
import csv
import statistics
import sys
from collections import defaultdict


FIELDS = [
    "total_ms",
    "mkl_spmm_ms",
    "mkl_spmm_pct",
    "prepare_pct",
    "csr_build_ms",
    "output_init_ms",
]


def median(values):
    return statistics.median(values) if values else float("nan")


def main():
    if len(sys.argv) != 2:
        print("usage: analyze_profile_csv.py path/to/profile.csv", file=sys.stderr)
        return 2

    groups = defaultdict(list)
    with open(sys.argv[1], newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["mode"]), int(row["rank"]), int(row["threads"]))
            groups[key].append(row)

    print("mode,rank,threads,median_total_ms,median_mkl_spmm_ms,"
          "median_mkl_spmm_pct,median_prepare_pct,median_csr_build_ms,"
          "median_output_init_ms,judgement")

    for key in sorted(groups):
        rows = groups[key]
        med = {
            field: median([float(row[field]) for row in rows])
            for field in FIELDS
        }
        judgements = []
        if med["mkl_spmm_pct"] > 0.7:
            judgements.append("MKL SpMM compute dominates")
        if med["prepare_pct"] > 0.5:
            judgements.append("Prepare/layout dominates")
        if med["output_init_ms"] / med["total_ms"] > 0.2:
            judgements.append("Output initialization is significant")
        judgement = " | ".join(judgements) if judgements else "No single dominant bucket"

        mode, rank, threads = key
        print(
            f"{mode},{rank},{threads},"
            f"{med['total_ms']:.6f},{med['mkl_spmm_ms']:.6f},"
            f"{med['mkl_spmm_pct']:.6f},{med['prepare_pct']:.6f},"
            f"{med['csr_build_ms']:.6f},{med['output_init_ms']:.6f},"
            f"{judgement}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
