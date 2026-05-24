#!/usr/bin/env python3
import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


NUMERIC_COLUMNS = {
    "total_ms",
    "format_prepare_ms",
    "upcast_prepare_ms",
    "compute_ms",
    "index_storage_bytes",
    "value_storage_bytes",
    "factor_storage_bytes",
    "output_storage_bytes",
    "index_logical_read_bytes",
    "value_logical_read_bytes",
    "factor_logical_read_bytes",
    "output_logical_write_bytes",
    "rel_error",
    "rank",
    "nnz",
    "mode",
    "thread_count",
    "seed",
    "dim0",
    "dim1",
    "dim2",
    "repeat",
}

PALETTE = ["#2563eb", "#dc2626", "#059669", "#9333ea", "#ea580c", "#0891b2"]


def read_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in NUMERIC_COLUMNS:
            if key in row and row[key] != "":
                row[key] = float(row[key])
    return rows


def variants(rows):
    return sorted({row["variant"] for row in rows})


def grouped_median(rows, key_fields, value_field):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        groups[key].append(row[value_field])
    return {key: median(values) for key, values in groups.items()}


def write_svg(path, body, width=900, height=560):
    text = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        '<style>text{font-family:Arial,sans-serif;font-size:12px;fill:#111827}'
        '.title{font-size:18px;font-weight:700}.axis{stroke:#111827;stroke-width:1}'
        '.grid{stroke:#e5e7eb;stroke-width:1}.legend{font-size:11px}</style>\n'
        f"{body}\n</svg>\n"
    )
    path.write_text(text, encoding="utf-8")


def chart_frame(title, xlabel, ylabel, width=900, height=560):
    margin = {"left": 88, "right": 220, "top": 52, "bottom": 78}
    x0 = margin["left"]
    y0 = height - margin["bottom"]
    x1 = width - margin["right"]
    y1 = margin["top"]
    body = [
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" class="title">{html.escape(title)}</text>',
        f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" class="axis"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" class="axis"/>',
        f'<text x="{(x0 + x1) / 2:.1f}" y="{height - 24}" text-anchor="middle">{html.escape(xlabel)}</text>',
        f'<text transform="translate(22,{(y0 + y1) / 2:.1f}) rotate(-90)" text-anchor="middle">{html.escape(ylabel)}</text>',
    ]
    return body, x0, y0, x1, y1


def y_domain(values):
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return 0.0, 1.0
    lo = min(0.0, min(finite))
    hi = max(finite)
    if hi <= lo:
        hi = lo + 1.0
    pad = (hi - lo) * 0.08
    return lo, hi + pad


def line_chart(series, title, xlabel, ylabel, out_path, logy=False):
    width, height = 900, 560
    body, x0, y0, x1, y1 = chart_frame(title, xlabel, ylabel, width, height)
    all_x = sorted({x for points in series.values() for x, _ in points})
    if not all_x:
        write_svg(out_path, body, width, height)
        return

    x_positions = {x: x0 + (x1 - x0) * i / max(1, len(all_x) - 1) for i, x in enumerate(all_x)}
    raw_y = [y for points in series.values() for _, y in points]
    if logy:
        positive = [v for v in raw_y if v > 0.0 and math.isfinite(v)]
        floor = min(positive) / 10.0 if positive else 1e-16
        transform = lambda v: math.log10(max(v, floor))
        y_label = lambda v: f"{10 ** v:.1e}"
        y_values = [transform(v) for v in raw_y]
    else:
        transform = lambda v: v
        y_label = lambda v: f"{v:.3g}"
        y_values = raw_y

    ymin, ymax = y_domain(y_values)
    for t in range(6):
        frac = t / 5.0
        y = y0 - (y0 - y1) * frac
        value = ymin + (ymax - ymin) * frac
        body.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" class="grid"/>')
        body.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end">{y_label(value)}</text>')

    for i, x in enumerate(all_x):
        px = x_positions[x]
        body.append(f'<text x="{px:.1f}" y="{y0 + 20}" text-anchor="middle">{int(x)}</text>')

    for si, (name, points) in enumerate(series.items()):
        color = PALETTE[si % len(PALETTE)]
        coords = []
        for x, y in points:
            px = x_positions[x]
            ty = transform(y)
            py = y0 - (ty - ymin) / (ymax - ymin) * (y0 - y1)
            coords.append((px, py))
        if len(coords) >= 2:
            d = " ".join(f"{px:.1f},{py:.1f}" for px, py in coords)
            body.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
        for px, py in coords:
            body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{color}"/>')
        legend_y = y1 + si * 18
        body.append(f'<rect x="{x1 + 24}" y="{legend_y - 9}" width="10" height="10" fill="{color}"/>')
        body.append(f'<text x="{x1 + 40}" y="{legend_y}" class="legend">{html.escape(name)}</text>')

    write_svg(out_path, body, width, height)


def rank_vs_metric(rows, metric, title, ylabel, out_path, logy=False):
    med = grouped_median(rows, ["variant", "rank"], metric)
    series = {}
    for variant in variants(rows):
        points = sorted((rank, value) for (v, rank), value in med.items() if v == variant)
        series[variant] = points
    line_chart(series, title, "rank", ylabel, out_path, logy=logy)


def traffic_breakdown(rows, out_path):
    components = [
        ("index_logical_read_bytes", "index read"),
        ("value_logical_read_bytes", "value read"),
        ("factor_logical_read_bytes", "factor read"),
        ("output_logical_write_bytes", "output write"),
    ]
    ranks = sorted({int(row["rank"]) for row in rows})
    stacks = []
    totals = []
    for rank in ranks:
        rank_rows = [row for row in rows if int(row["rank"]) == rank]
        values = [median(row[field] for row in rank_rows) / (1024.0 * 1024.0) for field, _ in components]
        stacks.append(values)
        totals.append(sum(values))

    width, height = 900, 560
    body, x0, y0, x1, y1 = chart_frame(
        "rank vs estimated traffic breakdown",
        "rank",
        "median estimated logical traffic (MiB)",
        width,
        height,
    )
    _, ymax = y_domain(totals)
    bar_gap = 12
    bar_w = max(18, (x1 - x0) / max(1, len(ranks)) - bar_gap)

    for t in range(6):
        frac = t / 5.0
        y = y0 - (y0 - y1) * frac
        value = ymax * frac
        body.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" class="grid"/>')
        body.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end">{value:.3g}</text>')

    for i, rank in enumerate(ranks):
        cx = x0 + (i + 0.5) * (x1 - x0) / max(1, len(ranks))
        bottom = y0
        for ci, value in enumerate(stacks[i]):
            h = value / ymax * (y0 - y1) if ymax > 0 else 0
            color = PALETTE[ci % len(PALETTE)]
            body.append(f'<rect x="{cx - bar_w / 2:.1f}" y="{bottom - h:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
            bottom -= h
        body.append(f'<text x="{cx:.1f}" y="{y0 + 20}" text-anchor="middle">{rank}</text>')

    for ci, (_, label) in enumerate(components):
        legend_y = y1 + ci * 18
        color = PALETTE[ci % len(PALETTE)]
        body.append(f'<rect x="{x1 + 24}" y="{legend_y - 9}" width="10" height="10" fill="{color}"/>')
        body.append(f'<text x="{x1 + 40}" y="{legend_y}" class="legend">{html.escape(label)}</text>')
    write_svg(out_path, body, width, height)


def variant_phase_bars(rows, out_path):
    phase_fields = ["compute_ms", "format_prepare_ms", "upcast_prepare_ms"]
    phase_labels = ["compute", "format prepare", "upcast prepare"]
    names = variants(rows)
    values = [
        [median(row[field] for row in rows if row["variant"] == name) for field in phase_fields]
        for name in names
    ]
    ymax = max([sum(v) for v in values] + [1.0])

    width, height = 980, 580
    body, x0, y0, x1, y1 = chart_frame("variant vs median phase time", "variant", "median time (ms)", width, height)
    bar_gap = 16
    bar_w = max(28, (x1 - x0) / max(1, len(names)) - bar_gap)

    for t in range(6):
        frac = t / 5.0
        y = y0 - (y0 - y1) * frac
        value = ymax * frac
        body.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" class="grid"/>')
        body.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end">{value:.3g}</text>')

    for i, name in enumerate(names):
        cx = x0 + (i + 0.5) * (x1 - x0) / max(1, len(names))
        bottom = y0
        for pi, value in enumerate(values[i]):
            h = value / ymax * (y0 - y1)
            color = PALETTE[pi % len(PALETTE)]
            body.append(f'<rect x="{cx - bar_w / 2:.1f}" y="{bottom - h:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
            bottom -= h
        body.append(f'<text transform="translate({cx:.1f},{y0 + 26}) rotate(25)" text-anchor="start" class="legend">{html.escape(name)}</text>')

    for pi, label in enumerate(phase_labels):
        legend_y = y1 + pi * 18
        color = PALETTE[pi % len(PALETTE)]
        body.append(f'<rect x="{x1 + 24}" y="{legend_y - 9}" width="10" height="10" fill="{color}"/>')
        body.append(f'<text x="{x1 + 40}" y="{legend_y}" class="legend">{html.escape(label)}</text>')
    write_svg(out_path, body, width, height)


def threads_speedup(rows, out_path):
    by_case = defaultdict(dict)
    for row in rows:
        key = (row["variant"], row["rank"], row["nnz"], row["mode"])
        by_case[key].setdefault(int(row["thread_count"]), []).append(row["compute_ms"])

    speedups = defaultdict(list)
    for per_thread in by_case.values():
        if 1 not in per_thread:
            continue
        baseline = median(per_thread[1])
        if baseline <= 0.0 or not math.isfinite(baseline):
            continue
        for thread_count, values in per_thread.items():
            t = median(values)
            if t > 0.0:
                speedups[thread_count].append(baseline / t)

    series = {"median compute speedup": sorted((float(k), median(v)) for k, v in speedups.items())}
    ideal_x = sorted(speedups)
    if ideal_x:
        series["ideal"] = [(float(x), float(x)) for x in ideal_x]
    line_chart(series, "threads vs speedup", "threads", "speedup vs 1 thread", out_path)


def main():
    parser = argparse.ArgumentParser(description="Plot sparse-dense TTM benchmark CSV results as SVG files.")
    parser.add_argument("csv", type=Path, help="Input CSV produced by ttm_bench")
    parser.add_argument("--out-dir", type=Path, default=Path("plots"), help="Output directory")
    args = parser.parse_args()

    rows = read_rows(args.csv)
    if not rows:
        raise SystemExit("CSV has no data rows")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rank_vs_metric(rows, "total_ms", "rank vs median total_ms", "median total time (ms)", args.out_dir / "rank_vs_total_ms.svg")
    rank_vs_metric(rows, "rel_error", "rank vs median rel_error", "median relative error", args.out_dir / "rank_vs_rel_error.svg", logy=True)
    traffic_breakdown(rows, args.out_dir / "rank_vs_traffic_breakdown.svg")
    variant_phase_bars(rows, args.out_dir / "variant_vs_phase_time.svg")
    threads_speedup(rows, args.out_dir / "threads_vs_speedup.svg")
    print(f"wrote SVG plots to {args.out_dir}")


if __name__ == "__main__":
    main()
