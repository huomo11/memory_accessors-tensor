#!/usr/bin/env python3
from __future__ import print_function

import csv
import math
import os
import sys


COLORS = {
    "factor_fp64_compute_fp64": "#1f77b4",
    "factor_fp32_compute_fp64": "#2ca02c",
    "factor_fp32_compute_fp32": "#d62728",
    "value_fp32_factor_fp64_compute_fp64": "#9467bd",
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
        key = (r["variant"], float(r[x_key]))
        data.setdefault(key, []).append(float(r[y_key]))
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


def traffic_breakdown(rows, out_path):
    data = {}
    for r in rows:
        key = (float(r["rank"]), r["variant"])
        idx = float(r["index_logical_read_bytes"])
        val = float(r["value_logical_read_bytes"])
        fac = float(r["factor_logical_read_bytes"])
        data.setdefault(key, []).append((idx, val, fac))
    ranks = sorted(set(k[0] for k in data))
    variants = sorted(set(k[1] for k in data))
    w, h = 900, 480
    lines = svg_begin(w, h)
    lines.append('<text class="title" x="50" y="28">rank_vs_traffic_breakdown</text>')
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
    lines.append('<text x="650" y="70">stack: index, value, factor</text>')
    lines.extend(svg_end())
    write_svg(out_path, lines)


def phase_time(rows, out_path):
    variants = sorted(set(r["variant"] for r in rows))
    w, h = 860, 460
    lines = svg_begin(w, h)
    lines.append('<text class="title" x="50" y="28">variant_vs_phase_time</text>')
    phases = ["format_prepare_ms", "upcast_prepare_ms", "compute_ms"]
    colors = ["#999999", "#ffbf00", "#2ca02c"]
    totals = {}
    max_total = 1.0
    for v in variants:
        group = [r for r in rows if r["variant"] == v]
        vals = [median([float(r[p]) for r in group]) for p in phases]
        totals[v] = vals
        max_total = max(max_total, sum(vals))
    x = 70
    for v in variants:
        y = 390
        for val, color in zip(totals[v], colors):
            height = val / max_total * 320.0
            y -= height
            lines.append('<rect x="%d" y="%.2f" width="80" height="%.2f" fill="%s"/>' % (x, y, height, color))
        lines.append('<text transform="translate(%d,420) rotate(35)">%s</text>' % (x, v))
        x += 180
    lines.append('<text x="610" y="70">gray=format, yellow=upcast, green=compute</text>')
    lines.extend(svg_end())
    write_svg(out_path, lines)


def threads_speedup(rows, out_path):
    base = [r for r in rows if r["variant"] == "factor_fp64_compute_fp64"]
    if not base:
        return
    by_thread = {}
    for r in base:
        by_thread.setdefault(float(r["thread_count"]), []).append(float(r["compute_ms"]))
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
    line_plot(rows, os.path.join(out_dir, "rank_vs_rel_error.svg"), "rank", "rel_error", "rank_vs_rel_error", "rel_error")
    traffic_breakdown(rows, os.path.join(out_dir, "rank_vs_traffic_breakdown.svg"))
    phase_time(rows, os.path.join(out_dir, "variant_vs_phase_time.svg"))
    threads_speedup(rows, os.path.join(out_dir, "threads_vs_speedup.svg"))
    print("wrote SVG plots to %s" % out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
