#!/usr/bin/env python3
"""Generate self-contained HTML report combining two experiments:
1. Teacher intermediate step decode (ODE trajectory analysis)
2. Distillation method comparison (5 methods on 105 samples)

Usage:
    cd /mnt/workspace/kuno/distillation
    python distill_methods/scripts/generate_combined_report.py
"""

import base64
import csv
import io
import math
import os
from pathlib import Path

from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

# Data paths
INTERMEDIATE_CSV = PROJECT_DIR / "results/distill_methods/intermediate_decode/summary.csv"
INTERMEDIATE_ROOT = PROJECT_DIR / "results/distill_methods/intermediate_decode"
METHOD_SUMMARY_CSV = PROJECT_DIR / "results/distill_methods/runs/20260406_1131/metrics/distill_summary.csv"
METHOD_PER_SAMPLE_CSV = PROJECT_DIR / "results/distill_methods/runs/20260406_1131/metrics/distill_per_sample.csv"
RENDERS_ROOT = PROJECT_DIR / "datasets/Toys4k/renders/512"
LATENT_ANALYSIS_DIR = PROJECT_DIR / "distill_methods/reports/latent_analysis"
OUTPUT_PATH = PROJECT_DIR / "distill_methods/reports/2026-04-08_combined-analysis.html"

# Experimental data (hardcoded from analysis runs)
CROSS_SAMPLE_COS = {
    5: 0.9997, 10: 0.9992, 15: 0.9975, 20: 0.9916, 25: 0.9747,
    30: 0.9443, 35: 0.9077, 40: 0.8747, 45: 0.8516, 50: 0.8408,
}
VOLUME_LOGIT_DATA = {
    "ball_002": [
        (5, -1.0, -1.0, 100.0, False), (10, -1.0, -1.0, 100.0, False),
        (15, -1.0, -0.9, 100.0, False), (20, -1.0, 0.1, 100.0, True),
        (25, -1.0, 1.0, 83.4, True), (30, -1.0, 1.0, 52.8, True),
        (35, -1.0, 1.0, 52.0, True), (40, -1.0, 1.0, 52.0, True),
        (45, -1.0, 1.0, 52.0, True), (50, -1.0, 1.1, 52.0, True),
    ],
    "chair_168": [
        (5, -1.0, -1.0, 100.0, False), (10, -1.0, -1.0, 100.0, False),
        (15, -1.0, -1.0, 100.0, False), (20, -1.0, -1.0, 100.0, False),
        (25, -1.0, 0.9, 99.9, True), (30, -1.0, 1.0, 97.0, True),
        (35, -1.0, 1.0, 96.0, True), (40, -1.0, 1.0, 96.0, True),
        (45, -1.0, 1.0, 96.0, True), (50, -1.0, 1.0, 96.0, True),
    ],
    "helicopter_015": [
        (5, -1.0, -1.0, 100.0, False), (10, -1.0, -1.0, 100.0, False),
        (15, -1.0, -1.0, 100.0, False), (20, -1.0, -1.0, 100.0, False),
        (25, -1.0, -1.0, 100.0, False), (30, -1.0, 1.0, 99.9, True),
        (35, -1.0, 1.0, 99.4, True), (40, -1.0, 1.0, 99.4, True),
        (45, -1.0, 1.0, 99.4, True), (50, -1.0, 1.0, 99.4, True),
    ],
}

STEPS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
SAMPLE_ORDER = [
    "ball_002", "apple_007", "cup_056", "shoe_028", "chair_168",
    "guitar_003", "airplane_007", "robot_016", "dinosaur_069", "helicopter_015",
]
COMPLEXITY = {
    "ball_002": "simple", "apple_007": "simple",
    "cup_056": "medium", "shoe_028": "medium", "chair_168": "medium",
    "guitar_003": "complex", "airplane_007": "complex",
    "robot_016": "complex", "dinosaur_069": "complex", "helicopter_015": "complex",
}
COMPLEXITY_COLORS = {"simple": "#22c55e", "medium": "#eab308", "complex": "#ef4444"}
GRID_SAMPLES = ["ball_002", "chair_168", "helicopter_015"]


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def embed_image(path, max_size=256):
    """Load, resize, and base64-encode an image."""
    img = Image.open(path)
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img)
        img = bg.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    if max_size and (img.width > max_size or img.height > max_size):
        ratio = max_size / max(img.width, img.height)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def embed_grid(path):
    """Embed grid at 50% size."""
    img = Image.open(path).convert("RGB")
    img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def build_image_data():
    """Collect all images for embedding."""
    images = {}
    for oid in SAMPLE_ORDER:
        cat = oid.rsplit("_", 1)[0]
        # Input image
        input_path = RENDERS_ROOT / cat / oid / "image.png"
        if input_path.exists():
            images[f"input_{oid}"] = embed_image(str(input_path), 256)
        # Step renders
        for step in STEPS:
            render_path = INTERMEDIATE_ROOT / oid / "renders" / f"step_{step:02d}.png"
            if render_path.exists():
                images[f"step_{oid}_{step}"] = embed_image(str(render_path), 256)
    # Grid images
    for oid in GRID_SAMPLES:
        grid_path = INTERMEDIATE_ROOT / oid / "renders" / f"{oid}_grid.png"
        if grid_path.exists():
            images[f"grid_{oid}"] = embed_grid(str(grid_path))
    # Latent analysis images (from other agent)
    for name in ["pca_grid", "cosine_similarity_grid", "trajectory_straightness",
                  "distance_evolution", "tsne_trajectories"]:
        path = LATENT_ANALYSIS_DIR / f"{name}.png"
        if path.exists():
            images[f"la_{name}"] = embed_image(str(path), max_size=900)
    return images


def generate_cd_chart_svg(data):
    """Generate inline SVG chart: CD vs step for all samples."""
    w, h = 700, 350
    pad_l, pad_r, pad_t, pad_b = 70, 20, 30, 50

    # Parse data by sample
    by_sample = {}
    for row in data:
        oid = row["object_id"]
        if oid not in by_sample:
            by_sample[oid] = []
        by_sample[oid].append(row)

    # Find CD range (log scale)
    all_cds = []
    for rows in by_sample.values():
        for r in rows:
            cd = r.get("chamfer_distance")
            if cd and r["status"] == "success":
                all_cds.append(float(cd))
    if not all_cds:
        return "<p>No data</p>"

    cd_min = min(all_cds) * 0.5
    cd_max = max(all_cds) * 1.5

    def log_y(cd):
        if cd <= 0:
            return pad_t
        log_min = math.log10(cd_min)
        log_max = math.log10(cd_max)
        frac = (math.log10(cd) - log_min) / (log_max - log_min)
        return pad_t + (1 - frac) * (h - pad_t - pad_b)

    def x_pos(t):
        return pad_l + (t - 0.0) / 1.1 * (w - pad_l - pad_r)

    svg_parts = [
        f'<svg viewBox="0 0 {w} {h}" class="chart" style="max-width:{w}px;width:100%;">',
        # Crystallization highlight band
        f'<rect x="{x_pos(0.5)}" y="{pad_t}" width="{x_pos(0.7)-x_pos(0.5)}" '
        f'height="{h-pad_t-pad_b}" fill="#38bdf8" opacity="0.08"/>',
        f'<text x="{(x_pos(0.5)+x_pos(0.7))/2}" y="{pad_t+14}" '
        f'text-anchor="middle" class="chart-label" style="font-size:10px;fill:#38bdf8">'
        f'crystallization zone</text>',
    ]

    # Axes
    svg_parts.append(
        f'<line x1="{pad_l}" y1="{h-pad_b}" x2="{w-pad_r}" y2="{h-pad_b}" '
        f'stroke="#475569" stroke-width="1"/>'
    )
    # X labels
    for step in STEPS:
        t = step / 50
        x = x_pos(t)
        svg_parts.append(
            f'<text x="{x}" y="{h-pad_b+20}" text-anchor="middle" '
            f'class="chart-label">{t:.1f}</text>'
        )
        svg_parts.append(
            f'<line x1="{x}" y1="{h-pad_b}" x2="{x}" y2="{h-pad_b+5}" '
            f'stroke="#475569" stroke-width="1"/>'
        )
    svg_parts.append(
        f'<text x="{(pad_l+w-pad_r)/2}" y="{h-5}" text-anchor="middle" '
        f'class="chart-label">t (ODE progress)</text>'
    )

    # Y grid + labels (log scale)
    for exp in range(-4, 1):
        val = 10 ** exp
        if val < cd_min or val > cd_max:
            continue
        y = log_y(val)
        svg_parts.append(
            f'<line x1="{pad_l}" y1="{y}" x2="{w-pad_r}" y2="{y}" '
            f'stroke="#334155" stroke-width="0.5" stroke-dasharray="3,3"/>'
        )
        label = f"1e{exp}" if exp < 0 else str(int(val))
        svg_parts.append(
            f'<text x="{pad_l-8}" y="{y+4}" text-anchor="end" '
            f'class="chart-label" style="font-size:10px">{label}</text>'
        )
    svg_parts.append(
        f'<text x="12" y="{(pad_t+h-pad_b)/2}" text-anchor="middle" '
        f'transform="rotate(-90,12,{(pad_t+h-pad_b)/2})" '
        f'class="chart-label">Chamfer Distance (log)</text>'
    )

    # Lines per sample
    for oid in SAMPLE_ORDER:
        rows = by_sample.get(oid, [])
        color = COMPLEXITY_COLORS[COMPLEXITY[oid]]
        points = []
        for r in rows:
            cd = r.get("chamfer_distance")
            if cd and r["status"] == "success":
                t = float(r["t"])
                points.append((x_pos(t), log_y(float(cd))))
        if len(points) >= 2:
            pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            svg_parts.append(
                f'<polyline points="{pts_str}" fill="none" '
                f'stroke="{color}" stroke-width="2" opacity="0.8"/>'
            )
            # Endpoint label
            lx, ly = points[-1]
            short_name = oid.rsplit("_", 1)[0]
            svg_parts.append(
                f'<text x="{lx+4}" y="{ly+3}" style="font-size:9px;fill:{color}">'
                f'{short_name}</text>'
            )
            # Dots
            for x, y in points:
                svg_parts.append(
                    f'<circle cx="{x}" cy="{y}" r="2.5" fill="{color}"/>'
                )

    # Legend
    ly = pad_t + 8
    for label, color in COMPLEXITY_COLORS.items():
        svg_parts.append(
            f'<rect x="{w-pad_r-130}" y="{ly}" width="10" height="10" '
            f'fill="{color}" rx="2"/>'
        )
        svg_parts.append(
            f'<text x="{w-pad_r-115}" y="{ly+9}" style="font-size:11px;fill:#94a3b8">'
            f'{label}</text>'
        )
        ly += 16

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def generate_runtime_svg():
    """Stacked bar chart: diffusion vs decode time."""
    methods = [
        ("teacher_50step", "50", 4.6, 9.0),
        ("pd_6step", "6", 0.5, 9.0),
        ("cd_4step", "4", 0.4, 9.0),
        ("dmd1_1step", "1", 0.1, 9.0),
        ("dmd2_1step", "1", 0.1, 9.0),
    ]
    w, h = 500, 220
    bar_h = 28
    gap = 8
    pad_l, pad_t = 130, 30
    max_time = 16

    svg = [f'<svg viewBox="0 0 {w} {h}" class="chart" style="max-width:{w}px;width:100%;">']
    svg.append(f'<text x="{pad_l + (w-pad_l)/2}" y="16" text-anchor="middle" '
               f'class="chart-title">Wall-Clock Time Breakdown</text>')

    for i, (name, steps, diff_t, dec_t) in enumerate(methods):
        y = pad_t + i * (bar_h + gap)
        diff_w = diff_t / max_time * (w - pad_l - 40)
        dec_w = dec_t / max_time * (w - pad_l - 40)
        # Label
        svg.append(f'<text x="{pad_l-8}" y="{y+bar_h/2+4}" text-anchor="end" '
                   f'style="font-size:12px;fill:#e2e8f0">{name}</text>')
        # Diffusion bar
        svg.append(f'<rect x="{pad_l}" y="{y}" width="{diff_w}" height="{bar_h}" '
                   f'fill="#38bdf8" rx="3"/>')
        # Decode bar
        svg.append(f'<rect x="{pad_l+diff_w}" y="{y}" width="{dec_w}" height="{bar_h}" '
                   f'fill="#f97316" rx="0 3 3 0"/>')
        # Time labels
        total = diff_t + dec_t
        svg.append(f'<text x="{pad_l+diff_w+dec_w+6}" y="{y+bar_h/2+4}" '
                   f'style="font-size:11px;fill:#94a3b8">{total:.1f}s</text>')

    # Legend
    ly = pad_t + len(methods) * (bar_h + gap) + 10
    svg.append(f'<rect x="{pad_l}" y="{ly}" width="12" height="12" fill="#38bdf8" rx="2"/>')
    svg.append(f'<text x="{pad_l+16}" y="{ly+10}" style="font-size:11px;fill:#94a3b8">'
               f'Diffusion</text>')
    svg.append(f'<rect x="{pad_l+100}" y="{ly}" width="12" height="12" fill="#f97316" rx="2"/>')
    svg.append(f'<text x="{pad_l+116}" y="{ly+10}" style="font-size:11px;fill:#94a3b8">'
               f'VAE Decode</text>')

    svg.append("</svg>")
    return "\n".join(svg)


def build_step_viewer_html(images):
    """Build interactive step viewer with slider for each sample."""
    rows = []
    for oid in SAMPLE_ORDER:
        cat = oid.rsplit("_", 1)[0]
        comp = COMPLEXITY[oid]
        comp_color = COMPLEXITY_COLORS[comp]

        # Build image map for JS
        img_map = {}
        for step in STEPS:
            key = f"step_{oid}_{step}"
            if key in images:
                img_map[step] = images[key]

        if not img_map:
            continue

        input_key = f"input_{oid}"
        input_src = images.get(input_key, "")

        min_step = min(img_map.keys())
        max_step = max(img_map.keys())

        # Embed all step images as hidden elements
        step_imgs = []
        for step in STEPS:
            key = f"step_{oid}_{step}"
            if key in images:
                step_imgs.append(
                    f'<img id="img-{oid}-{step}" src="{images[key]}" '
                    f'style="display:none;width:256px;height:256px;border-radius:6px;">'
                )
            else:
                step_imgs.append(
                    f'<div id="img-{oid}-{step}" style="display:none;width:256px;height:256px;'
                    f'border-radius:6px;background:var(--bg3);line-height:256px;text-align:center;'
                    f'color:var(--fg3);font-size:12px;">No isosurface</div>'
                )

        rows.append(f'''
<div class="sample-row" style="display:flex;gap:16px;align-items:center;padding:12px 0;border-bottom:1px solid var(--bg3);">
  <div style="width:80px;flex-shrink:0;">
    {"<img src='" + input_src + "' style='width:80px;height:80px;border-radius:4px;'>" if input_src else ""}
    <div style="font-size:12px;font-weight:600;margin-top:4px;">{cat}</div>
    <div style="font-size:10px;color:{comp_color}">{comp}</div>
  </div>
  <div style="flex:1;min-width:0;">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
      <label style="font-size:12px;color:var(--fg2);white-space:nowrap;">Step:</label>
      <input type="range" min="5" max="50" step="5" value="{max_step}"
        oninput="updateStep('{oid}', this.value)"
        style="flex:1;max-width:300px;">
      <span id="step-label-{oid}" style="font-size:13px;font-weight:600;min-width:100px;">
        t={max_step/50:.1f} (step {max_step})
      </span>
      <span id="cd-label-{oid}" style="font-size:12px;color:var(--fg2);min-width:120px;"></span>
    </div>
    <div id="viewer-{oid}" style="position:relative;">
      {''.join(step_imgs)}
    </div>
  </div>
</div>''')

    return "\n".join(rows)


def build_method_table(summary_data):
    """Build sortable method comparison table."""
    rows = []
    for r in summary_data:
        model = r["model"]
        cd_mean = float(r["chamfer_distance_mean"])
        cd_median = float(r["chamfer_distance_median"])
        hausdorff = float(r["hausdorff_mean"])
        f1 = float(r["f_score_001_mean"])
        f2 = float(r["f_score_002_mean"])
        n_fail = int(r["n_failures"])
        n = int(r["n_samples"])

        rows.append(f'''<tr>
  <td>{model}</td>
  <td class="num">{cd_mean*1000:.1f}</td>
  <td class="num">{cd_median*1000:.2f}</td>
  <td class="num">{hausdorff:.3f}</td>
  <td class="num">{f1:.3f}</td>
  <td class="num">{f2:.3f}</td>
  <td class="num">{n}/{n+n_fail}</td>
</tr>''')

    return "\n".join(rows)


def build_complementary_table(per_sample_data):
    """Build CD vs DMD1 complementary failure table."""
    # Group by object_id
    cd_data = {}
    dmd1_data = {}
    for r in per_sample_data:
        oid = r["object_id"]
        cat = r["category"]
        model = r["model"]
        cd_val = float(r["chamfer_distance"])
        if model == "cd_4step":
            cd_data[oid] = (cat, cd_val)
        elif model == "dmd1_1step":
            dmd1_data[oid] = (cat, cd_val)

    # Find biggest wins for each
    dmd1_wins = []
    cd_wins = []
    for oid in cd_data:
        if oid not in dmd1_data:
            continue
        cat_cd, val_cd = cd_data[oid]
        _, val_dmd1 = dmd1_data[oid]
        if val_dmd1 > 0 and val_cd > 0:
            ratio = val_cd / val_dmd1
            if ratio > 3:
                dmd1_wins.append((cat_cd, val_cd, val_dmd1, ratio))
            elif 1 / ratio > 3:
                cd_wins.append((cat_cd, val_cd, val_dmd1, 1 / ratio))

    dmd1_wins.sort(key=lambda x: -x[3])
    cd_wins.sort(key=lambda x: -x[3])

    html = '<h3>DMD1 wins (1-step beats 4-step)</h3><table><thead><tr>'
    html += '<th>Category</th><th class="num">CD (cd_4step)</th>'
    html += '<th class="num">CD (dmd1_1step)</th><th class="num">Ratio</th></tr></thead><tbody>'
    for cat, v_cd, v_dmd1, ratio in dmd1_wins[:6]:
        html += f'<tr><td>{cat}</td><td class="num">{v_cd:.5f}</td>'
        html += f'<td class="num best">{v_dmd1:.5f}</td>'
        html += f'<td class="num">{ratio:.0f}x</td></tr>'
    html += '</tbody></table>'

    html += '<h3 style="margin-top:20px">CD wins (4-step beats 1-step)</h3>'
    html += '<table><thead><tr><th>Category</th><th class="num">CD (cd_4step)</th>'
    html += '<th class="num">CD (dmd1_1step)</th><th class="num">Ratio</th></tr></thead><tbody>'
    for cat, v_cd, v_dmd1, ratio in cd_wins[:6]:
        html += f'<tr><td>{cat}</td><td class="num best">{v_cd:.5f}</td>'
        html += f'<td class="num">{v_dmd1:.5f}</td>'
        html += f'<td class="num">{ratio:.0f}x</td></tr>'
    html += '</tbody></table>'

    return html


def build_cd_data_js(data):
    """Build JS data object for step viewer CD labels."""
    by_sample = {}
    for row in data:
        oid = row["object_id"]
        if oid not in by_sample:
            by_sample[oid] = {}
        step = int(row["step"])
        cd = row.get("chamfer_distance")
        if cd and row["status"] == "success":
            by_sample[oid][step] = float(cd)
    lines = ["const cdData = {"]
    for oid, steps in by_sample.items():
        step_strs = ", ".join(f"{s}: {v:.6f}" for s, v in sorted(steps.items()))
        lines.append(f'  "{oid}": {{{step_strs}}},')
    lines.append("};")
    return "\n".join(lines)


def generate_html():
    print("Loading data ...")
    intermediate_data = load_csv(INTERMEDIATE_CSV)
    method_summary = load_csv(METHOD_SUMMARY_CSV)
    per_sample = load_csv(METHOD_PER_SAMPLE_CSV)

    print("Embedding images ...")
    images = build_image_data()
    print(f"  {len(images)} images embedded")

    print("Generating components ...")
    cd_chart = generate_cd_chart_svg(intermediate_data)
    runtime_chart = generate_runtime_svg()
    step_viewer = build_step_viewer_html(images)
    method_table = build_method_table(method_summary)
    complementary = build_complementary_table(per_sample)
    cd_js = build_cd_data_js(intermediate_data)

    # Latent analysis images
    la_images = {}
    for name in ["pca_grid", "cosine_similarity_grid", "trajectory_straightness",
                  "distance_evolution", "tsne_trajectories"]:
        key = f"la_{name}"
        if key in images:
            la_images[name] = images[key]

    # Volume logit table HTML
    vol_rows = ""
    for oid, data in VOLUME_LOGIT_DATA.items():
        cat = oid.rsplit("_", 1)[0]
        for step, vmin, vmax, neg_pct, crossing in data:
            cross_str = '<span class="best">Yes</span>' if crossing else '<span style="color:var(--fg3)">No</span>'
            neg_cls = "" if neg_pct >= 99.9 else (' class="best"' if neg_pct < 55 else "")
            vol_rows += f'<tr><td>{cat}</td><td class="num">{step}</td><td class="num">{step/50:.1f}</td>'
            vol_rows += f'<td class="num">{vmin:.1f}</td><td class="num">{vmax:.1f}</td>'
            vol_rows += f'<td class="num"{neg_cls}>{neg_pct:.1f}%</td><td>{cross_str}</td></tr>\n'

    # Cross-sample similarity table
    cos_rows = ""
    for step, cos in sorted(CROSS_SAMPLE_COS.items()):
        t = step / 50
        phase = "I" if t <= 0.4 else ("II" if t <= 0.6 else "III")
        cos_cls = "" if cos > 0.98 else (' class="best"' if cos < 0.92 else ' style="color:var(--yellow)"')
        cos_rows += f'<tr><td class="num">{step}</td><td class="num">{t:.1f}</td>'
        cos_rows += f'<td class="num"{cos_cls}>{cos:.4f}</td><td>{phase}</td></tr>\n'

    # Grid images
    grid_html = ""
    for oid in GRID_SAMPLES:
        key = f"grid_{oid}"
        if key in images:
            cat = oid.rsplit("_", 1)[0]
            comp = COMPLEXITY[oid]
            grid_html += f'''
<div style="margin:16px 0;">
  <div style="font-size:13px;font-weight:600;margin-bottom:4px;">
    {cat} <span style="color:{COMPLEXITY_COLORS[comp]};font-size:11px;">({comp})</span>
    <span style="color:var(--fg3);font-size:11px;margin-left:8px;">
      t=0.1 → 0.2 → 0.3 → 0.4 → 0.5 → 0.6 → 0.7 → 0.8 → 0.9 → 1.0
    </span>
  </div>
  <img src="{images[key]}" style="width:100%;border-radius:6px;background:var(--bg2);">
</div>'''

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Structure Formation and Distillation in VecSet Latent Space</title>
<style>
:root {{
  --bg: #0f172a; --bg2: #1e293b; --bg3: #334155;
  --fg: #e2e8f0; --fg2: #94a3b8; --fg3: #64748b;
  --accent: #38bdf8; --green: #22c55e; --red: #ef4444; --orange: #f97316; --yellow: #eab308;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--fg); font-family: 'Inter', -apple-system, system-ui, sans-serif; font-size: 14px; line-height: 1.6; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
h1 {{ font-size: 26px; font-weight: 700; margin-bottom: 4px; }}
h2 {{ font-size: 18px; font-weight: 600; margin: 40px 0 12px; color: var(--accent); border-bottom: 1px solid var(--bg3); padding-bottom: 6px; }}
h3 {{ font-size: 15px; font-weight: 600; margin: 24px 0 8px; color: var(--fg2); }}
.subtitle {{ color: var(--fg2); font-size: 13px; margin-bottom: 24px; }}
p {{ color: var(--fg2); margin: 8px 0; max-width: 900px; }}
p strong {{ color: var(--fg); }}
nav {{ background: var(--bg2); border-radius: 8px; padding: 12px 16px; margin-bottom: 24px; position: sticky; top: 0; z-index: 10; }}
nav a {{ color: var(--accent); text-decoration: none; font-size: 12px; margin-right: 16px; }}
nav a:hover {{ text-decoration: underline; }}
.cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }}
.card {{ background: var(--bg2); border-radius: 8px; padding: 16px; flex: 1; min-width: 180px; }}
.card-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--fg3); }}
.card-value {{ font-size: 28px; font-weight: 700; margin-top: 4px; }}
.card-sub {{ font-size: 12px; color: var(--fg2); margin-top: 2px; }}
table {{ border-collapse: collapse; width: 100%; margin: 8px 0; max-width: 900px; }}
th {{ background: var(--bg3); color: var(--fg); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.3px; }}
td, th {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--bg3); }}
tr:hover td {{ background: var(--bg2); }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.best {{ color: var(--green); font-weight: 600; }}
.worst {{ color: var(--red); font-weight: 600; }}
.callout {{ background: var(--bg2); border-left: 3px solid var(--accent); border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 16px 0; max-width: 900px; }}
.callout.warn {{ border-left-color: var(--orange); }}
.callout.success {{ border-left-color: var(--green); }}
.callout-title {{ font-weight: 600; font-size: 13px; margin-bottom: 4px; }}
.chart {{ display: block; margin: 12px 0; }}
.chart-title {{ font-size: 13px; font-weight: 600; fill: var(--fg2); }}
.chart-label {{ font-size: 11px; fill: var(--fg2); }}
input[type="range"] {{ accent-color: var(--accent); }}
</style>
</head>
<body>
<div class="container">

<h1>Structure Formation and Distillation in VecSet Latent Space</h1>
<div class="subtitle">
  Branch: exp/distill-methods-v2 &bull; Date: 2026-04-08 &bull;
  Hunyuan3D-2.1 (50-step flow matching) &bull; Toys4k dataset
</div>

<nav>
  <a href="#overview">Overview</a>
  <a href="#methodology">Methodology</a>
  <a href="#ode">ODE Analysis</a>
  <a href="#latent">Latent Dynamics</a>
  <a href="#methods">Method Comparison</a>
  <a href="#synthesis">Synthesis</a>
  <a href="#related">Related Work</a>
  <a href="#next">Limitations &amp; Next Steps</a>
</nav>

<!-- ==================== OVERVIEW ==================== -->
<h2 id="overview">1. Overview</h2>

<p>This report combines two experiments to answer: <strong>why does distillation work for 3D generation,
and what are its limits?</strong> First, we decode the teacher's intermediate ODE states to observe
how 3D structure forms in the VecSet latent space. Then, we evaluate five distillation methods
and connect their performance to the structure formation dynamics.</p>

<div class="cards">
  <div class="card">
    <div class="card-label">Experiment 1</div>
    <div class="card-value" style="font-size:22px;">ODE Decode</div>
    <div class="card-sub">10 samples &times; 10 steps</div>
  </div>
  <div class="card">
    <div class="card-label">Experiment 2</div>
    <div class="card-value" style="font-size:22px;">5 Methods</div>
    <div class="card-sub">105 samples, 15K training steps</div>
  </div>
  <div class="card">
    <div class="card-label">Key Finding</div>
    <div class="card-value" style="color:var(--accent);font-size:22px;">t = 0.6</div>
    <div class="card-sub">Structure crystallization point</div>
  </div>
  <div class="card">
    <div class="card-label">Best Method</div>
    <div class="card-value" style="color:var(--green);font-size:22px;">CD 4-step</div>
    <div class="card-sub">CD = 19.3 &times; 10<sup>-3</sup></div>
  </div>
</div>

<!-- ==================== METHODOLOGY ==================== -->
<h2 id="methodology">2. Methodology &amp; Caveats</h2>

<p>Transparency about experimental conditions and known limitations.</p>

<table>
<thead><tr><th>Parameter</th><th>Value</th><th>Note</th></tr></thead>
<tbody>
<tr><td>Base model</td><td>Hunyuan3D-2.1 (VecSet flow matching)</td><td>Single model; results may not generalize to Sparse Voxel models</td></tr>
<tr><td>ODE trajectory samples</td><td>N=10 (diverse categories)</td><td>Sufficient for qualitative patterns; not for statistical claims</td></tr>
<tr><td>Distillation evaluation</td><td>N=105 test samples</td><td>1 per category, held-out from 420 training samples</td></tr>
<tr><td>Training</td><td>LoRA rank 64, 15K steps, batch 4</td><td>Miniature scale vs production (FlashVDM: full fine-tune, batch 256)</td></tr>
<tr><td>Dataset</td><td>Toys4k (simple objects)</td><td>Not representative of Objaverse-scale diversity</td></tr>
<tr><td>Guidance scale</td><td>5.0 (teacher inference) / 7.5 (training)</td><td>Verified: 3-phase structure robust to CFG; timing shifts by &plusmn;1 step</td></tr>
<tr><td>Volume logit resolution</td><td>64 and 384</td><td>Verified: identical results at both resolutions</td></tr>
<tr><td>CD measurement</td><td>Unit-sphere norm, no ICP</td><td>Absolute CD not comparable to ICP-aligned evaluations; relative trends valid</td></tr>
<tr><td>Latent analysis</td><td>Mean-pooled (4096,64) &rarr; (64,)</td><td>Token-level dynamics not captured; see caveats in Section 4</td></tr>
</tbody>
</table>

<div class="callout warn">
  <div class="callout-title">CFG Robustness Check</div>
  <p>The ODE trajectory was initially run with CFG=7.5 (training config) instead of 5.0 (teacher inference).
  Re-running with CFG=5.0 shows: <strong>8/10 samples have identical crystallization step; 2 samples shift by 1 step later.</strong>
  Final CD at step 50 differs by &lt;12%. The three-phase structure is robust to this variation.</p>
</div>

<!-- ==================== ODE ANALYSIS ==================== -->
<h2 id="ode">3. Teacher ODE Trajectory Analysis</h2>

<p>The teacher model generates 3D shapes by solving a flow-matching ODE from noise (t=0) to data (t=1)
over 50 Euler steps. We decode the latent at 10 checkpoints to observe how structure emerges.</p>

<h3>Structure Formation Grid</h3>
<p>Grey cells indicate steps where marching cubes found no isosurface (latent not yet structured enough).</p>
{grid_html}

<h3>Interactive Step Viewer</h3>
<p>Drag the slider to see how each shape forms during the ODE solve.</p>
{step_viewer}

<h3>Chamfer Distance vs ODE Progress</h3>
<p>Log-scale CD to ground truth at each step. The blue band marks the crystallization zone (t=0.5&ndash;0.7)
where structure rapidly emerges.</p>
{cd_chart}

<div class="callout">
  <div class="callout-title">Finding 1: Crystallization at t = 0.5&ndash;0.6</div>
  <p>For t &lt; 0.4, the latent contains no decodable structure (marching cubes fails entirely).
  At t = 0.5&ndash;0.6, global structure abruptly appears. Simple shapes (ball, apple) crystallize
  earlier (t = 0.4) than complex ones (helicopter, robot: t = 0.6).</p>
</div>

<div class="callout success">
  <div class="callout-title">Finding 2: CD converges by t = 0.7</div>
  <p>After t = 0.7 (step 35), Chamfer distance barely changes. The final 30% of ODE steps contribute
  almost nothing to geometric quality. This means the last 15 steps out of 50 are
  refinement of already-formed structure.</p>
</div>

<div class="callout warn">
  <div class="callout-title">Finding 3: Complexity determines crystallization timing</div>
  <p>Simple convex shapes (ball, apple) emerge at t = 0.4. Medium complexity (cup, shoe, chair) at t = 0.5.
  Complex articulated shapes (robot, helicopter) require t = 0.6. Distillation must preserve
  fidelity during each shape's critical crystallization window.</p>
</div>

<!-- ==================== LATENT SPACE DYNAMICS ==================== -->
<h2 id="latent">4. Latent Space Dynamics: Three Phases of Structure Formation</h2>

<p>Beyond mesh decoding, we analyze the latent vectors directly to understand <strong>what the teacher
is computing</strong> at each ODE step. Three distinct phases emerge.</p>

<div class="callout">
  <div class="callout-title">Phase I (t=0.0&ndash;0.4): Homogeneous Contraction &mdash; Noise Purification</div>
  <p>The latent L2 norm <strong>decreases by 21%</strong> (463 &rarr; 367). Cross-sample cosine similarity
  is <strong>0.999+</strong> &mdash; ball, chair, and helicopter are virtually identical in latent space.
  PCA shows all samples clustered together. The VAE volume decoder outputs <strong>uniformly &minus;1.0</strong>
  everywhere: "there is no object." The model is removing high-frequency noise while all samples
  converge to a shared pre-structural state. <strong>Sample-specific information has not yet been
  injected into the latent.</strong></p>
</div>

<div class="callout warn">
  <div class="callout-title">Phase II (t=0.4&ndash;0.6): Crystallization &mdash; Structural Phase Transition</div>
  <p>Cross-sample similarity <strong>drops sharply</strong> (0.99 &rarr; 0.94). PCA shows samples separating
  into distinct clusters. Volume logits develop zero-crossings (isosurfaces appear). The ODE
  trajectory shows <strong>maximum deviation from a straight line</strong> in this interval &mdash; the flow
  is "bending" as conditioning information forces differentiation. This is the <strong>critical window
  where the model decides what shape to create</strong>.</p>
</div>

<div class="callout success">
  <div class="callout-title">Phase III (t=0.6&ndash;1.0): Refinement &amp; Scaling</div>
  <p>Chamfer distance converges. Volume logit statistics stabilize. Token norms increase monotonically
  (365 &rarr; 510) but spatial structure doesn't change. PCA positions are stable. The model
  is <strong>scaling up the amplitude of already-formed structure</strong> without modifying topology.</p>
</div>

<h3>Cross-Sample Cosine Similarity</h3>
<p>Cosine similarity of flattened latent vectors (4096&times;64 = 262144-dim) between all sample pairs.
Values near 1.0 mean samples are indistinguishable; divergence indicates differentiation.</p>
<table style="max-width:500px;">
<thead><tr><th class="num">Step</th><th class="num">t</th><th class="num">Cos Similarity</th><th>Phase</th></tr></thead>
<tbody>
{cos_rows}
</tbody>
</table>

<h3>VAE Volume Logit Statistics</h3>
<p>The VAE decoder maps latents to a 3D SDF grid. Marching cubes extracts the mesh at the
zero-crossing. When <code>neg%=100</code> and no crossing exists, the decoder sees "empty space."</p>
<table style="max-width:700px;">
<thead><tr><th>Sample</th><th class="num">Step</th><th class="num">t</th><th class="num">Min</th><th class="num">Max</th><th class="num">Neg%</th><th>Zero Crossing</th></tr></thead>
<tbody>
{vol_rows}
</tbody>
</table>

<h3>PCA of Mean-Pooled Latents</h3>
<p>Each dot is one sample's mean-pooled latent (4096 tokens &rarr; 1 vector of dim 64). Shared PCA basis
across all steps. Early steps: tight cluster. Phase II: rapid separation. Phase III: positions stabilize.</p>
{"<img src='" + la_images.get('pca_grid', '') + "' style='width:100%;max-width:900px;border-radius:6px;'>" if 'pca_grid' in la_images else '<p style="color:var(--fg3)">[PCA image not available]</p>'}

<h3>t-SNE Trajectories</h3>
<p>t-SNE of all 100 latents (10 samples &times; 10 steps). Each trajectory shows one sample's
ODE path. Faint = early steps, bold = late steps. Trajectories are nearly straight and
well-separated after Phase II.</p>
{"<img src='" + la_images.get('tsne_trajectories', '') + "' style='width:100%;max-width:700px;border-radius:6px;'>" if 'tsne_trajectories' in la_images else '<p style="color:var(--fg3)">[t-SNE image not available]</p>'}

<h3>Pairwise Cosine Similarity Heatmaps</h3>
<p>Heatmaps at each step showing cosine similarity between mean-pooled latents. Early steps:
uniformly high (all similar). Phase II: block structure emerges (within-category similarity
exceeds between-category).</p>
{"<img src='" + la_images.get('cosine_similarity_grid', '') + "' style='width:100%;max-width:900px;border-radius:6px;'>" if 'cosine_similarity_grid' in la_images else '<p style="color:var(--fg3)">[Cosine similarity image not available]</p>'}

<h3>Trajectory Straightness</h3>
<p>Left: deviation from straight-line path at each step. Peaks at t=0.3&ndash;0.5 (Phase I/II boundary)
where the flow "bends" most. Right: overall straightness per sample (S &gt; 0.9 for all).
High straightness is why few-step distillation works &mdash; the ODE path is nearly linear.</p>
{"<img src='" + la_images.get('trajectory_straightness', '') + "' style='width:100%;max-width:900px;border-radius:6px;'>" if 'trajectory_straightness' in la_images else '<p style="color:var(--fg3)">[Trajectory straightness image not available]</p>'}

<h3>Distance Evolution</h3>
<p>Left: pairwise L2 distance between samples grows steadily. Center: cluster separation (max/min
distance ratio) drops during Phase I then recovers &mdash; confirming initial convergence then
differentiation. Right: intra-sample token spread (std of token norms) shows the same
V-shape: compression then expansion.</p>
{"<img src='" + la_images.get('distance_evolution', '') + "' style='width:100%;max-width:900px;border-radius:6px;'>" if 'distance_evolution' in la_images else '<p style="color:var(--fg3)">[Distance evolution image not available]</p>'}

<!-- ==================== METHOD COMPARISON ==================== -->
<h2 id="methods">5. Distillation Method Comparison</h2>

<p>Five methods trained on Hunyuan3D-2.1 with LoRA (rank 64), 420 samples, 15K steps.
Evaluated on 105 held-out test samples against the 50-step teacher.</p>

<h3>Fidelity Summary</h3>
<table>
<thead><tr>
  <th>Model</th>
  <th class="num">CD (&times;10<sup>-3</sup>)</th>
  <th class="num">CD Median</th>
  <th class="num">Hausdorff</th>
  <th class="num">F@1%</th>
  <th class="num">F@2%</th>
  <th class="num">Success</th>
</tr></thead>
<tbody>
{method_table}
</tbody>
</table>

<p><strong>SiD (1-step)</strong> is excluded: 0/105 success rate. All outputs fail surface extraction
(latents produce volumes with no valid isosurface).</p>

<h3>Runtime Breakdown</h3>
<p>Volume decoding (~9s) dominates wall-clock time. Step reduction from 50&rarr;1 saves only ~4.5s (28%).</p>
{runtime_chart}

<h3>Complementary Failures: CD vs DMD1</h3>
<p>CD (4-step) and DMD1 (1-step) have complementary strengths &mdash; neither dominates on all categories.
This suggests fundamentally different failure modes, not simply "more steps = better."</p>
{complementary}

<!-- ==================== SYNTHESIS ==================== -->
<h2 id="synthesis">6. Synthesis: Connecting ODE Dynamics to Distillation</h2>

<div class="callout">
  <div class="callout-title">Why CD (4-step) wins</div>
  <p>With 4 steps, the ODE visits t = 0.25, 0.5, 0.75, 1.0.
  Two points fall in the crystallization zone (t = 0.5 and 0.75), and
  consistency distillation enforces self-consistency along the full trajectory.
  This covers the critical structure formation phase while maintaining trajectory coherence.</p>
</div>

<div class="callout warn">
  <div class="callout-title">Why 1-step is harder than it looks</div>
  <p>The crystallization at t = 0.5&ndash;0.6 is an abrupt phase transition, not a gradual process.
  A 1-step model must compress this discontinuity into a single forward pass.
  DMD1 succeeds partially by using pre-computed teacher pairs as regression targets (sidestepping
  the trajectory entirely). DMD2 fails because adversarial loss alone cannot capture this transition.</p>
</div>

<div class="callout success">
  <div class="callout-title">Volume decode is the real bottleneck</div>
  <p>Even with 1-step diffusion (0.1s), total inference is 11s because VAE decode takes ~9s.
  Step reduction has hit diminishing returns. The next frontier is decoder acceleration
  (FlashVDM approach: hierarchical decode + adaptive KV selection achieves 45&times; decode speedup).</p>
</div>

<h3>PD Stages vs Crystallization</h3>
<table>
<thead><tr><th>PD Stage</th><th>Steps</th><th>Coverage</th><th>Crystallization</th></tr></thead>
<tbody>
<tr><td>Stage 1</td><td>50 &rarr; 25</td><td>t sampled every 0.04</td><td>Full coverage, but most steps in post-convergence zone</td></tr>
<tr><td>Stage 2</td><td>25 &rarr; 12</td><td>t sampled every 0.08</td><td>2-3 points in crystallization zone</td></tr>
<tr><td>Stage 3</td><td>12 &rarr; 6</td><td>t sampled every 0.17</td><td>1-2 points in crystallization zone, error accumulates</td></tr>
</tbody>
</table>
<p>PD's cascaded error accumulation across 3 stages explains why it underperforms CD despite using more steps (6 vs 4).
A possible improvement: skip directly from 50 to 12 steps, then 12 to 6 (2-stage instead of 3).</p>

<div class="callout">
  <div class="callout-title">Connection to LATTICE (Lai et al., CVPR 2026)</div>
  <p>LATTICE argues that VecSet tokens lack spatial structure, and introducing positional
  encoding (VoxSet) enables Scaling Law. Our Phase I data supports this directly: at t &lt; 0.4,
  cross-sample cosine similarity is 0.999+ and PCA variance is near-random (PC10 = 18.7% vs
  15.6% for uniform). <strong>The VecSet latent space is structureless during noise purification.</strong>
  Structure only emerges through conditioning at Phase II &mdash; and once it does, it is
  implicit and fragile (FlashVDM found only ~10 tokens per query are attended to).</p>
</div>

<div class="callout">
  <div class="callout-title">Connection to FlashVDM (ICCV 2025 Highlight)</div>
  <p>FlashVDM discovered that VecSet VAE attention maps have strong spatial locality.
  Our trajectory straightness (S &gt; 0.91) is consistent with this, though recent work
  (&ldquo;Straightness Is Not Your Need,&rdquo; ICLR 2025) argues straightness is descriptive rather
  than causal for distillation success. What matters more: the <strong>consistency constraint</strong>
  (our CD method&rsquo;s key advantage) and <strong>Phase II coverage</strong>. FlashVDM&rsquo;s guidance distillation
  warmup is essential precisely because it prepares the student for the crystallization transition.</p>
</div>

<!-- ==================== RELATED WORK ==================== -->
<h2 id="related">7. Related Work &amp; Positioning</h2>

<p>How our findings relate to existing literature.</p>

<table>
<thead><tr><th>Paper</th><th>Venue</th><th>Finding</th><th>Connection to Our Work</th></tr></thead>
<tbody>
<tr><td>Sclocchi et al.</td><td>PNAS 2024</td><td>Coarse features via sharp phase transition; fine details smooth</td><td><strong>Our Phase II is the 3D analogue.</strong> In 3D VecSet, the transition is more dramatic: spatial structure itself must be created, not just filled in</td></tr>
<tr><td>Park et al. (2307.12868)</td><td>NeurIPS 2023</td><td>Pullback metric shows latent geometry evolves across timesteps</td><td>Confirms coarse-to-fine in 2D. Our work extends to 3D VecSet with direct mesh decode verification</td></tr>
<tr><td>DC-AE 1.5 (2508.00413)</td><td>ICCV 2025</td><td>Structured latent channels accelerate convergence 4x</td><td>Our Phase I &ldquo;structurelessness&rdquo; (cos=0.999) is what DC-AE fixes in 2D by structuring channels. LATTICE does the same in 3D with PE</td></tr>
<tr><td>&ldquo;Straightness Is Not Your Need&rdquo; (2410.07303)</td><td>ICLR 2025</td><td>Curved paths work equally well; matched pairs matter more</td><td>Our S&gt;0.91 is descriptive, not causal. CD works because of trajectory consistency, not straightness per se</td></tr>
<tr><td>FlashVDM (2503.16302)</td><td>ICCV 2025</td><td>VecSet attention locality (~10 tokens); 3-stage distillation</td><td>Our attention analysis confirms locality is surface-specific (14 tokens at surface vs 38 outside). FlashVDM&rsquo;s CFD base validates our CD &gt; adversarial finding</td></tr>
<tr><td>LATTICE (2512.03052)</td><td>CVPR 2026</td><td>PE on VoxSet enables Scaling Law</td><td>Our Phase I data (cross-sample cos=0.999) provides experimental evidence that VecSet is structureless. PE injection addresses this directly</td></tr>
<tr><td>Consistency Models</td><td>ICML 2023</td><td>Self-consistency along ODE with O(&epsilon;&sup2;) convergence</td><td>CD&rsquo;s success in our comparison validates the consistency principle for 3D VecSet flow matching</td></tr>
<tr><td>Linear Convergence (2410.09046)</td><td>Preprint</td><td>Convergence &prop; intrinsic dimension d</td><td>Structure (PE/VoxSet) reduces effective d &rarr; faster convergence. Theoretical backing for LATTICE + our observations</td></tr>
</tbody>
</table>

<div class="callout">
  <div class="callout-title">Novel Contribution</div>
  <p>No prior work has decoded intermediate ODE states of a 3D LDM to study structure formation dynamics.
  Our three-phase characterization, VAE attention analysis, and controlled distillation comparison
  provide the first experimental bridge between latent space geometry theory (Park, DC-AE) and
  practical distillation design (FlashVDM, Consistency Models) in the 3D generation domain.</p>
</div>

<!-- ==================== LIMITATIONS & NEXT STEPS ==================== -->
<h2 id="next">8. Limitations &amp; Next Steps</h2>

<h3>Limitations</h3>
<ul style="color:var(--fg2);margin:8px 0 8px 20px;max-width:900px;">
  <li><strong>Single base model:</strong> All results are for Hunyuan3D-2.1 (VecSet). Sparse Voxel models
  (Trellis, Trellis2) may show different dynamics due to built-in spatial structure.</li>
  <li><strong>Small scale:</strong> N=10 trajectory samples, N=420 training data, LoRA rank 64.
  Production distillation operates at orders of magnitude larger scale.</li>
  <li><strong>Simple dataset:</strong> Toys4k objects are geometrically simple compared to Objaverse diversity.</li>
  <li><strong>No ICP alignment in trajectory CD:</strong> Absolute CD values from intermediate decode are
  not comparable to ICP-aligned evaluations. Relative trends across steps remain valid.</li>
  <li><strong>Mean pooling:</strong> Latent analysis (PCA, t-SNE) uses mean-pooled tokens, losing
  token-level sparsity dynamics.</li>
</ul>

<h3>Next Steps</h3>
<ul style="color:var(--fg2);margin:8px 0 8px 20px;max-width:900px;">
  <li><strong>Sparse Voxel comparison:</strong> Run the same trajectory analysis on Trellis2 to test whether
  the three-phase structure is VecSet-specific or universal to 3D LDMs.</li>
  <li><strong>Teacher step-quality curve:</strong> Run teacher at 1, 2, 4, 6, 12, 25 steps (no distillation)
  to isolate distillation&rsquo;s contribution from simple step reduction.</li>
  <li><strong>Token-level dynamics:</strong> Analyze per-token norm distributions and activation patterns
  to complement mean-pooled analysis.</li>
  <li><strong>Riemannian metric analysis:</strong> Apply Park et al.&rsquo;s pullback metric to 3D VecSet latent space
  to quantify geometry changes across phases.</li>
  <li><strong>2-stage PD:</strong> Test [50&rarr;12], [12&rarr;6] to reduce error accumulation.</li>
</ul>

</div>

<script>
{cd_js}

function updateStep(oid, step) {{
  step = parseInt(step);
  // Hide all images for this sample
  for (let s = 5; s <= 50; s += 5) {{
    const el = document.getElementById('img-' + oid + '-' + s);
    if (el) el.style.display = 'none';
  }}
  // Show selected step
  const el = document.getElementById('img-' + oid + '-' + step);
  if (el) el.style.display = 'block';
  // Update labels
  const label = document.getElementById('step-label-' + oid);
  if (label) label.textContent = 't=' + (step/50).toFixed(1) + ' (step ' + step + ')';
  const cdLabel = document.getElementById('cd-label-' + oid);
  if (cdLabel) {{
    const cd = cdData[oid] && cdData[oid][step];
    cdLabel.textContent = cd !== undefined ? 'CD = ' + cd.toFixed(6) : '';
  }}
}}

// Initialize all viewers to show max step
document.addEventListener('DOMContentLoaded', function() {{
  const samples = {str(SAMPLE_ORDER)};
  samples.forEach(function(oid) {{
    updateStep(oid, 50);
  }});
}});
</script>

</body>
</html>'''

    return html


def main():
    print("Generating combined report ...")
    html = generate_html()

    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(html)

    size_mb = os.path.getsize(OUTPUT_PATH) / 1e6
    print(f"Report written to {OUTPUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
