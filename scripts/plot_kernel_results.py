#!/usr/bin/env python3
from __future__ import print_function

import csv
import math
import os
import sys


COLORS = {
    "factor_fp64_compute_fp64": "#1f77b4",
    "factor_fp32_compute_fp64": "#2ca02c",
    "factor_fp32_global_upcast_compute_fp64": "#2ca02c",
    "factor_fp32_onfly_compute_fp64": "#17becf",
    "factor_fp32_blocked_compute_fp64": "#ff7f0e",
    "factor_fp32_2dblocked_compute_fp64": "#8c564b",
    "factor_fp32_compute_fp32": "#d62728",
    "value_fp32_factor_fp64_compute_fp64": "#9467bd",
    "csr_fp64_factor_fp64_backend": "#34495e",
    "csr_factor_fp32_global_upcast_fp64_backend": "#27ae60",
    "csr_factor_fp32_tiled_accessor_fp64_backend": "#c0392b",
}


def median(values):
    vals = sorted(values)
    if not vals:
        return 0.0
    n = len(vals)
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def load_rows(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def get_float(row, key):
    if key == "kernel_ms" and key not in row:
        return float(row["upcast_prepare_ms"]) + float(row["compute_ms"])
    if key == "factor_compute_logical_read_bytes" and key not in row:
        return float(row["factor_logical_read_bytes"])
    return float(row[key])


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def svg_begin(w, h):
    return ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">' % (w, h, w, h),
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font-family:Arial,sans-serif;font-size:12px}.title{font-size:16px;font-weight:bold}</style>']


def svg_end():
    return ["</svg>"]


def write_svg(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines))


def scale(v, vmin, vmax, omin, omax):
    if vmax == vmin:
        return (omin + omax) * 0.5
    return omin + (v - vmin) * (omax - omin) / (vmax - vmin)


def grouped_median(rows, x_key, y_key):
    data = {}
    for r in rows:
        key = (r["variant"], get_float(r, x_key))
        data.setdefault(key, []).append(get_float(r, y_key))
    out = {}
    for key, vals in data.items():
        out.setdefault(key[0], []).append((key[1], median(vals)))
    for k in out:
        out[k].sort()
    return out


def grouped_median_by_label(rows, label_key, y_key):
    data = {}
    for r in rows:
        label = r.get(label_key, "")
        variant = r["variant"]
        key = (variant, label)
        data.setdefault(key, []).append(get_float(r, y_key))
    out = {}
    for key, vals in data.items():
        out.setdefault(key[0], []).append((key[1], median(vals)))
    for k in out:
        out[k].sort()
    return out


def line_plot(rows, out_path, x_key, y_key, title, y_label):
    data = grouped_median(rows, x_key, y_key)
    points = []
    for vals in data.values():
        points.extend(vals)
    if not points:
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    w, h = 820, 460
    l, r, t, b = 70, 30, 50, 60
    x0, x1 = min(xs), max(xs)
    y0, y1 = 0.0, max(ys) * 1.08 if max(ys) > 0 else 1.0
    lines = svg_begin(w, h)
    lines.append('<text class="title" x="%d" y="28">%s</text>' % (l, title))
    lines.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#222"/>' % (l, h-b, w-r, h-b))
    lines.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#222"/>' % (l, t, l, h-b))
    lines.append('<text x="%d" y="%d">%s</text>' % (w//2 - 20, h-18, x_key))
    lines.append('<text x="8" y="%d">%s</text>' % (h//2, y_label))
    for variant in sorted(data):
        vals = data[variant]
        color = COLORS.get(variant, "#333333")
        path = []
        for x, y in vals:
            px = scale(x, x0, x1, l, w-r)
            py = scale(y, y0, y1, h-b, t)
            path.append((px, py))
        if path:
            d = "M " + " L ".join(["%.2f %.2f" % p for p in path])
            lines.append('<path d="%s" fill="none" stroke="%s" stroke-width="2"/>' % (d, color))
            for px, py in path:
                lines.append('<circle cx="%.2f" cy="%.2f" r="3" fill="%s"/>' % (px, py, color))
    y_text = "%.3g" % y1
    lines.append('<text x="12" y="%d">%s</text>' % (t + 4, y_text))
    lines.append('<text x="12" y="%d">0</text>' % (h-b))
    lx, ly = w - 310, 58
    i = 0
    for variant in sorted(data):
        color = COLORS.get(variant, "#333333")
        lines.append('<rect x="%d" y="%d" width="12" height="12" fill="%s"/>' % (lx, ly + 18*i - 10, color))
        lines.append('<text x="%d" y="%d">%s</text>' % (lx + 18, ly + 18*i, variant))
        i += 1
    lines.extend(svg_end())
    write_svg(out_path, lines)


def compute_traffic_breakdown(rows, out_path):
    data = {}
    for r in rows:
        key = (get_float(r, "rank"), r["variant"])
        idx = get_float(r, "index_logical_read_bytes")
        val = get_float(r, "value_logical_read_bytes")
        fac = get_float(r, "factor_compute_logical_read_bytes")
        data.setdefault(key, []).append((idx, val, fac))
    ranks = sorted(set(k[0] for k in data))
    variants = sorted(set(k[1] for k in data))
    w, h = 900, 480
    lines = svg_begin(w, h)
    lines.append('<text class="title" x="50" y="28">rank_vs_compute_traffic_breakdown</text>')
    x = 70
    bar_w = 14
    gap = 8
    max_total = 1.0
    totals = {}
    for key, vals in data.items():
        idx = median([v[0] for v in vals])
        val = median([v[1] for v in vals])
        fac = median([v[2] for v in vals])
        totals[key] = (idx, val, fac)
        max_total = max(max_total, idx + val + fac)
    for rank in ranks:
        lines.append('<text x="%d" y="440">r=%s</text>' % (x, int(rank)))
        for variant in variants:
            idx, val, fac = totals.get((rank, variant), (0, 0, 0))
            ybase = 405
            heights = [idx / max_total * 330.0, val / max_total * 330.0, fac / max_total * 330.0]
            colors = ["#777777", "#ffbf00", COLORS.get(variant, "#333333")]
            y = ybase
            for height, color in zip(heights, colors):
                y -= height
                lines.append('<rect x="%d" y="%.2f" width="%d" height="%.2f" fill="%s"/>' % (x, y, bar_w, height, color))
            x += bar_w + gap
        x += 22
    lines.append('<text x="610" y="70">stack: index, value, factor compute reads</text>')
    lines.extend(svg_end())
    write_svg(out_path, lines)


def phase_time(rows, out_path):
    variants = sorted(set(r["variant"] for r in rows))
    w, h = 1100, 500
    lines = svg_begin(w, h)
    lines.append('<text class="title" x="50" y="28">variant_vs_phase_time</text>')
    phases = ["format_prepare_ms", "upcast_prepare_ms", "compute_ms"]
    colors = ["#999999", "#ffbf00", "#2ca02c"]
    totals = {}
    max_total = 1.0
    for v in variants:
        group = [r for r in rows if r["variant"] == v]
        vals = [median([get_float(r, p) for r in group]) for p in phases]
        totals[v] = vals
        max_total = max(max_total, sum(vals))
    x = 60
    step = 190
    for v in variants:
        y = 410
        for val, color in zip(totals[v], colors):
            height = val / max_total * 330.0
            y -= height
            lines.append('<rect x="%d" y="%.2f" width="80" height="%.2f" fill="%s"/>' % (x, y, height, color))
        lines.append('<text transform="translate(%d,445) rotate(35)">%s</text>' % (x, v))
        x += step
    lines.append('<text x="760" y="70">gray=format, yellow=upcast, green=compute</text>')
    lines.extend(svg_end())
    write_svg(out_path, lines)


def categorical_bar_plot(rows, out_path, label_key, y_key, title):
    data = grouped_median_by_label(rows, label_key, y_key)
    labels = sorted(set(label for vals in data.values() for label, _ in vals))
    variants = sorted(data)
    if not labels or not variants:
        return
    w, h = 980, 480
    l, t, b = 70, 50, 80
    max_y = 1.0
    value_map = {}
    for variant in variants:
        for label, val in data[variant]:
            value_map[(variant, label)] = val
            max_y = max(max_y, val)
    lines = svg_begin(w, h)
    lines.append('<text class="title" x="%d" y="28">%s</text>' % (l, title))
    lines.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#222"/>' % (l, h-b, w-30, h-b))
    lines.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#222"/>' % (l, t, l, h-b))
    group_w = (w - l - 60) / float(len(labels))
    bar_w = max(3.0, min(18.0, group_w / float(len(variants) + 1)))
    li = 0
    for label in labels:
        base_x = l + li * group_w + 8
        vi = 0
        for variant in variants:
            val = value_map.get((variant, label), 0.0)
            height = val / max_y * (h - b - t)
            x = base_x + vi * bar_w
            y = h - b - height
            color = COLORS.get(variant, "#333333")
            lines.append('<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" fill="%s"/>' %
                         (x, y, bar_w - 1.0, height, color))
            vi += 1
        lines.append('<text transform="translate(%.2f,%d) rotate(35)">%s</text>' %
                     (base_x, h - 55, label))
        li += 1
    lines.append('<text x="12" y="%d">%.3g</text>' % (t + 4, max_y))
    lx, ly = w - 360, 58
    vi = 0
    for variant in variants:
        color = COLORS.get(variant, "#333333")
        lines.append('<rect x="%d" y="%d" width="12" height="12" fill="%s"/>' %
                     (lx, ly + 18 * vi - 10, color))
        lines.append('<text x="%d" y="%d">%s</text>' % (lx + 18, ly + 18 * vi, variant))
        vi += 1
    lines.extend(svg_end())
    write_svg(out_path, lines)


def threads_speedup(rows, out_path):
    base = [r for r in rows if r["variant"] == "factor_fp64_compute_fp64"]
    if not base:
        return
    by_thread = {}
    for r in base:
        by_thread.setdefault(get_float(r, "thread_count"), []).append(get_float(r, "compute_ms"))
    if not by_thread:
        return
    baseline_thread = min(by_thread)
    t1 = median(by_thread[baseline_thread])
    plot_rows = []
    for t, vals in by_thread.items():
        plot_rows.append({"variant": "fp64 baseline", "thread_count": str(t), "speedup": str(t1 / median(vals) if median(vals) else 0.0)})
    line_plot(plot_rows, out_path, "thread_count", "speedup", "threads_vs_speedup", "speedup")


def main():
    if len(sys.argv) < 2:
        print("usage: plot_kernel_results.py result.csv --out-dir DIR", file=sys.stderr)
        return 2
    path = sys.argv[1]
    out_dir = "plots"
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--out-dir" and i + 1 < len(sys.argv):
            out_dir = sys.argv[i + 1]
            i += 2
        else:
            print("unknown argument: %s" % sys.argv[i], file=sys.stderr)
            return 2

    ensure_dir(out_dir)
    rows = load_rows(path)
    line_plot(rows, os.path.join(out_dir, "rank_vs_total_ms.svg"), "rank", "total_ms", "rank_vs_total_ms", "total_ms")
    line_plot(rows, os.path.join(out_dir, "rank_vs_kernel_ms.svg"), "rank", "kernel_ms", "rank_vs_kernel_ms", "kernel_ms")
    line_plot(rows, os.path.join(out_dir, "rank_vs_compute_ms.svg"), "rank", "compute_ms", "rank_vs_compute_ms", "compute_ms")
    line_plot(rows, os.path.join(out_dir, "rank_vs_rel_error.svg"), "rank", "rel_error", "rank_vs_rel_error", "rel_error")
    compute_traffic_breakdown(rows, os.path.join(out_dir, "rank_vs_compute_traffic_breakdown.svg"))
    phase_time(rows, os.path.join(out_dir, "variant_vs_phase_time.svg"))
    line_plot(rows, os.path.join(out_dir, "tile_rows_vs_kernel_ms.svg"), "tile_rows", "kernel_ms", "tile_rows_vs_kernel_ms", "kernel_ms")
    if rows and "output_block_rows" in rows[0]:
        line_plot(rows, os.path.join(out_dir, "output_block_rows_vs_kernel_ms.svg"), "output_block_rows", "kernel_ms", "output_block_rows_vs_kernel_ms", "kernel_ms")
    if rows and "layout" in rows[0]:
        categorical_bar_plot(rows, os.path.join(out_dir, "layout_vs_kernel_ms.svg"), "layout", "kernel_ms", "layout_vs_kernel_ms")
    threads_speedup(rows, os.path.join(out_dir, "threads_vs_speedup.svg"))
    print("wrote SVG plots to %s" % out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
