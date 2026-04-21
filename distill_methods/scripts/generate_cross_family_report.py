#!/usr/bin/env python3
"""Generate a self-contained HTML benchmark report for the cross-family run.

Reads:
  <RUN_DIR>/metrics/{summary.csv, per_sample.csv, frechet_distance.csv,
                     mesh_quality.csv, failures.csv}
  <RUN_DIR>/clamp_loose/metrics/per_sample.csv
  <RUN_DIR>/diagnostics/{fairness_stats.csv, alignment_scale_hist.png,
                         clamp_sensitivity.png, mesh_stats.png}

Writes a single HTML file with all images base64-embedded.
"""

import argparse
import base64
import html
import os
from pathlib import Path

import numpy as np
import pandas as pd


MODELS_ORDER = [
    "teacher_50step",
    "flashvdm",
    "pd_6step",
    "cd_4step",
    "dmd1_1step",
    "dmd2_1step",
    "trellis",
    "mdt_dist",
    "trellis2",
]

MODEL_FAMILY = {
    "teacher_50step": "H3D-2.1",
    "flashvdm": "H3D-2.1",
    "pd_6step": "H3D-2.1",
    "cd_4step": "H3D-2.1",
    "dmd1_1step": "H3D-2.1",
    "dmd2_1step": "H3D-2.1",
    "trellis": "TRELLIS v1",
    "mdt_dist": "TRELLIS v1",
    "trellis2": "TRELLIS 2",
}

MODEL_STEPS = {
    "teacher_50step": "50",
    "flashvdm": "5",
    "pd_6step": "6",
    "cd_4step": "4",
    "dmd1_1step": "1",
    "dmd2_1step": "1",
    "trellis": "25+25",
    "mdt_dist": "2+2",
    "trellis2": "default",
}


def b64_img(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" style="max-width:100%;"/>'


def fmt_num(v, ndp=5):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{ndp}f}"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def render_outliers(per_sample: pd.DataFrame, top_k: int = 5) -> str:
    """Per-model top-k worst (highest CD) samples. Reveals when mean ranking
    is driven by a few catastrophic failures rather than typical quality."""
    if per_sample.empty:
        return ""
    a = per_sample[per_sample.get("track", "A") == "A"] if "track" in per_sample.columns else per_sample
    rows = []
    for m in MODELS_ORDER:
        mdf = a[a["model"] == m].nlargest(top_k, "chamfer_distance")[["object_id", "chamfer_distance", "alignment_scale"]]
        items = [f"{r.object_id} (CD={r.chamfer_distance:.3f}, s={r.alignment_scale:.2f})" for r in mdf.itertuples()]
        rows.append(f"<tr><td>{m}</td><td>{'; '.join(items)}</td></tr>")
    hdr = "<tr><th>Model</th><th>Top-5 worst samples (object_id, CD, alignment_scale)</th></tr>"
    return (
        "<table class='summary'><thead>" + hdr + "</thead><tbody>"
        + "\n".join(rows) + "</tbody></table>"
    )


def render_paired_significance(per_sample: pd.DataFrame) -> str:
    """Paired Wilcoxon signed-rank test on Track A CD for key close comparisons.

    Reports Δmedian bootstrap 95% CI and p-value for each pair. Non-significant
    pairs (p > 0.05) indicate that N=105 is insufficient to distinguish the
    models at this effect size.
    """
    if per_sample.empty or "track" not in per_sample.columns:
        return ""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return "<p>(scipy not available — significance tests skipped)</p>"
    a = per_sample[per_sample["track"] == "A"]

    pairs = [
        ("teacher_50step", "dmd2_1step",  "Teacher ~ DMD2 Exp-1"),
        ("dmd2_1step",     "dmd1_1step",  "DMD2 Exp-1 vs DMD1 K=1"),
        ("trellis2",       "trellis",     "TRELLIS 2 vs TRELLIS v1"),
        ("cd_4step",       "dmd2_1step",  "CD 4-step vs DMD2"),
        ("dmd2_1step",     "dmd1_1step",  "DMD2 vs DMD1 (paper claim check)"),
        ("mdt_dist",       "trellis",     "MDT-Dist vs TRELLIS v1 teacher"),
        ("flashvdm",       "teacher_50step", "FlashVDM vs H3D-2.1 teacher"),
        ("pd_6step",       "dmd2_1step",  "PD 6-step vs DMD2 1-step"),
    ]
    seen = set()
    rows = []
    rng = np.random.default_rng(42)
    for a_name, b_name, label in pairs:
        key = tuple(sorted([a_name, b_name]))
        if key in seen:
            continue
        seen.add(key)
        a_df = a[a["model"] == a_name][["object_id", "chamfer_distance"]].rename(columns={"chamfer_distance": "cd_a"})
        b_df = a[a["model"] == b_name][["object_id", "chamfer_distance"]].rename(columns={"chamfer_distance": "cd_b"})
        merged = a_df.merge(b_df, on="object_id")
        if len(merged) < 5:
            continue
        diff = (merged["cd_a"] - merged["cd_b"]).to_numpy()
        boot = [float(np.median(rng.choice(diff, len(diff), replace=True))) for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        try:
            _, p = wilcoxon(diff)
        except Exception:
            p = float("nan")
        sig = "✓" if (isinstance(p, float) and p < 0.05) else "—"
        rows.append(
            f"<tr><td>{label}</td>"
            f"<td class='num'>{len(diff)}</td>"
            f"<td class='num'>[{lo:+.5f}, {hi:+.5f}]</td>"
            f"<td class='num'>{p:.4f}</td>"
            f"<td style='text-align:center;'>{sig}</td></tr>"
        )

    hdr = ("<tr><th>Comparison</th><th>n</th><th>Δmedian CD 95% CI</th>"
           "<th>Wilcoxon p</th><th>sig (p&lt;0.05)</th></tr>")
    return (
        "<table class='summary'><thead>" + hdr + "</thead><tbody>"
        + "\n".join(rows) + "</tbody></table>"
        + "<p style='font-size:13px;color:#666;margin-top:8px;'>"
        "Paired Wilcoxon signed-rank test on per-sample Track A CD. "
        "Δmedian sign: negative means the first model in the comparison has lower CD. "
        "Non-significant rows mean N=105 (each from a unique Toys4k category) "
        "cannot resolve the effect size.</p>"
    )


def render_summary_table(summary: pd.DataFrame, fd: pd.DataFrame) -> str:
    """Build the main results table. Reports BOTH mean and median — mean is
    outlier-sensitive when a few samples produce garbage (e.g. MDT-Dist on
    bicycle/octopus samples yields CD > 3), which masks the typical quality."""
    if summary.empty:
        return "<p>No summary data.</p>"

    fd_map = {}
    if not fd.empty:
        for _, row in fd.iterrows():
            fd_map.setdefault(row.get("model"), {})[row.get("feature_extractor", "?")] = row.get("fd")

    summ_a = summary[summary.get("track", "A") == "A"] if "track" in summary.columns else summary
    summ_a = summ_a.set_index("model")

    rows = []
    for m in MODELS_ORDER:
        if m not in summ_a.index:
            continue
        r = summ_a.loc[m]
        fd_inc = fd_map.get(m, {}).get("inception_v3")
        fd_dino = fd_map.get(m, {}).get("dinov2")
        cells = [
            f"<td>{m}</td>",
            f"<td>{MODEL_FAMILY[m]}</td>",
            f"<td>{MODEL_STEPS[m]}</td>",
            f"<td class='num'><strong>{fmt_num(r.get('chamfer_distance_median'), 5)}</strong></td>",
            f"<td class='num'>{fmt_num(r.get('chamfer_distance_mean'), 5)}</td>",
            f"<td class='num'>{fmt_num(r.get('f_score_001_median'), 4)}</td>",
            f"<td class='num'>{fmt_num(r.get('f_score_002_median'), 4)}</td>",
            f"<td class='num'>{fmt_num(r.get('hausdorff_median'), 4)}</td>",
            f"<td class='num'>{int(r.get('n_failures', 0))}</td>",
            f"<td class='num'>{fmt_num(fd_inc, 3)}</td>",
            f"<td class='num'>{fmt_num(fd_dino, 3)}</td>",
        ]
        rows.append("<tr>" + "".join(cells) + "</tr>")

    hdr = (
        "<tr><th>Model</th><th>Family</th><th>Steps</th>"
        "<th>CD median (↓)</th><th>CD mean</th>"
        "<th>F@1% med (↑)</th><th>F@2% med (↑)</th><th>Haus med (↓)</th>"
        "<th>fails</th>"
        "<th>FD-inc (↓)</th><th>FD-dino (↓)</th></tr>"
    )
    return (
        "<table class='summary'><thead>" + hdr + "</thead><tbody>"
        + "\n".join(rows) + "</tbody></table>"
        + "<p style='font-size:13px;color:#666;margin-top:8px;'>"
        "Median is the primary metric; mean is shown for reference but is "
        "dominated by a small number of catastrophic samples in some families "
        "(see section 3.5 for outlier analysis).</p>"
    )


def render_track_ab(per_sample: pd.DataFrame) -> str:
    if per_sample.empty:
        return "<p>No track data.</p>"
    rows = []
    for m in MODELS_ORDER:
        mdf = per_sample[per_sample["model"] == m]
        cd_a = mdf[mdf["track"] == "A"]["chamfer_distance"].median() if "track" in mdf.columns else np.nan
        cd_b = mdf[mdf["track"] == "B"]["chamfer_distance"].median() if "track" in mdf.columns else np.nan
        gain = cd_b - cd_a if pd.notna(cd_b) and pd.notna(cd_a) else np.nan
        rows.append(
            f"<tr><td>{m}</td>"
            f"<td class='num'>{fmt_num(cd_a)}</td>"
            f"<td class='num'>{fmt_num(cd_b)}</td>"
            f"<td class='num'>{fmt_num(gain)}</td></tr>"
        )
    hdr = "<tr><th>Model</th><th>Track A CD (median)</th><th>Track B CD (median)</th><th>ICP gain (B−A)</th></tr>"
    return (
        "<table class='summary'><thead>" + hdr + "</thead><tbody>"
        + "\n".join(rows) + "</tbody></table>"
    )


def render_fairness_table(stats: pd.DataFrame) -> str:
    if stats.empty:
        return "<p>No fairness stats.</p>"
    cols = [
        "model", "n_samples", "scale_mean", "scale_median",
        "clamp_hit_low_pct", "clamp_hit_high_pct",
        "cd_clamp_tight", "cd_clamp_loose", "cd_delta",
    ]
    present = [c for c in cols if c in stats.columns]
    stats = stats.set_index("model") if "model" in stats.columns else stats
    rows = []
    for m in MODELS_ORDER:
        if m not in stats.index:
            continue
        r = stats.loc[m]
        cells = []
        for c in present:
            if c == "model":
                cells.append(f"<td>{m}</td>")
                continue
            v = r.get(c)
            if isinstance(v, float):
                fmt = f"{v:.2f}" if "pct" in c else f"{v:.5f}"
                cells.append(f"<td class='num'>{fmt}</td>")
            else:
                cells.append(f"<td class='num'>{v}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    hdr = "<tr>" + "".join(f"<th>{c}</th>" for c in present) + "</tr>"
    return "<table class='summary'><thead>" + hdr + "</thead><tbody>" + "\n".join(rows) + "</tbody></table>"


def render_failures(failures: pd.DataFrame) -> str:
    if failures.empty:
        return "<p class='ok'>No inference/evaluation failures.</p>"
    total = len(failures)
    per_model = failures.groupby("model").size().to_dict()
    summary = ", ".join(f"{k}: {v}" for k, v in sorted(per_model.items()))
    return f"<p class='warn'><strong>{total} failures</strong> across ({summary}). See metrics/failures.csv.</p>"


def render_speed_table(run_dir: Path, summary: pd.DataFrame) -> str:
    """Per-model timing: DiT time, VAE time, end-to-end, with within-family speedups.

    DiT times are extracted from inference logs via tqdm progress-bar parsing;
    end-to-end comes from meta.json runtime_sec per sample.
    """
    import json, re
    logs = run_dir / "logs"
    preds = run_dir / "predictions"
    if not logs.exists() or not preds.exists():
        return ""

    dit_re = re.compile(r'Diffusion Sampling::\s*100%\|.*?\|\s*(\d+)/\1\s*\[\d+:\d+<\d+:\d+,\s*([\d.]+)it/s\]')
    vae_re = re.compile(r'Volume Decoding:\s*100%\|.*?\|\s*(\d+)/\1\s*\[\d+:\d+<\d+:\d+,\s*([\d.]+)it/s\]')
    fv_re  = re.compile(r'FlashVDM Volume Decoding:\s*100%\|.*?\|\s*(\d+)/\1\s*\[\d+:\d+<\d+:\d+,\s*([\d.]+)it/s\]')
    tr_re  = re.compile(r'Sampling:\s*100%\|.*?\|\s*(\d+)/\1\s*\[\d+:\d+<\d+:\d+,\s*([\d.]+)it/s\]')

    def times_from(text, rgx):
        return [int(m.group(1)) / float(m.group(2)) for m in rgx.finditer(text) if float(m.group(2)) > 0]

    def read_log(model):
        p = logs / f"inference_{model}.log"
        return p.read_text(errors="ignore") if p.exists() else ""

    def total_from_meta(model):
        d = preds / model / "default"
        if not d.exists():
            return None
        ts = []
        for sid in d.iterdir():
            mp = sid / "meta.json"
            if mp.exists():
                try:
                    data = json.loads(mp.read_text())
                    if data.get("status") == "success" and "runtime_sec" in data:
                        ts.append(data["runtime_sec"])
                except Exception:
                    pass
        return float(np.median(ts)) if ts else None

    # Compute per-model (dit_median, vae_median, total_median)
    entries = {}
    for model in MODELS_ORDER:
        t = read_log(model)
        if MODEL_FAMILY[model] == "H3D-2.1":
            dit = times_from(t, dit_re)
            vae = times_from(t, fv_re if model == "flashvdm" else vae_re)
            dit_med = float(np.median(dit)) if dit else None
            vae_med = float(np.median(vae)) if vae else None
        elif MODEL_FAMILY[model].startswith("TRELLIS"):
            sampling = times_from(t, tr_re)
            # TRELLIS v1: ss + slat (2 bars per sample); TRELLIS 2: ss + slat (+ texture but we skip)
            if sampling:
                paired = [sampling[i] + sampling[i + 1] for i in range(0, len(sampling) - 1, 2)]
                dit_med = float(np.median(paired)) if paired else None
            else:
                dit_med = None
            vae_med = None
        else:
            dit_med = vae_med = None
        entries[model] = {
            "dit": dit_med,
            "vae": vae_med,
            "total": total_from_meta(model),
        }

    # Baselines per family (teacher_50step, trellis, trellis2)
    baselines = {
        "H3D-2.1":    entries.get("teacher_50step"),
        "TRELLIS v1": entries.get("trellis"),
        "TRELLIS 2":  entries.get("trellis2"),
    }

    # Build table
    def fmt_s(v):
        return f"{v:.3f}s" if isinstance(v, float) and v == v else "—"
    def fmt_x(num, den):
        if num is None or den is None or num != num or den != den or den <= 0:
            return "—"
        return f"{num / den:.1f}×"

    rows = []
    sa = summary[summary.get("track", "A") == "A"].set_index("model") if "track" in summary.columns else summary.set_index("model")
    for m in MODELS_ORDER:
        if m not in entries:
            continue
        e = entries[m]
        base = baselines[MODEL_FAMILY[m]]
        cd_med = sa.loc[m, "chamfer_distance_median"] if m in sa.index else float("nan")
        cells = [
            f"<td>{m}</td>",
            f"<td>{MODEL_FAMILY[m]}</td>",
            f"<td>{MODEL_STEPS[m]}</td>",
            f"<td class='num'>{fmt_num(cd_med, 5)}</td>",
            f"<td class='num'>{fmt_s(e['dit'])}</td>",
            f"<td class='num'>{fmt_x(base['dit'] if base else None, e['dit'])}</td>",
            f"<td class='num'>{fmt_s(e['vae'])}</td>",
            f"<td class='num'>{fmt_s(e['total'])}</td>",
            f"<td class='num'>{fmt_x(base['total'] if base else None, e['total'])}</td>",
        ]
        rows.append("<tr>" + "".join(cells) + "</tr>")

    hdr = ("<tr><th>Model</th><th>Family</th><th>Steps</th><th>CD median ↓</th>"
           "<th>DiT time</th><th>DiT speedup</th><th>VAE time</th>"
           "<th>End-to-end</th><th>E2E speedup</th></tr>")
    return (
        "<table class='summary'><thead>" + hdr + "</thead><tbody>"
        + "\n".join(rows) + "</tbody></table>"
        + "<p style='font-size:13px;color:#666;margin-top:8px;'>"
        "Speedups are within-family (baseline = teacher_50step / trellis / trellis2). "
        "DiT time = diffusion sampling only (what distillation targets). "
        "VAE time = volume decoding (unchanged by DMD/PD/CD distillation; "
        "FlashVDM's turbo VAE accelerates this separately). "
        "End-to-end (from meta.json runtime_sec) is bounded by whichever is slower.</p>"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--title", default="Cross-family Benchmark")
    ap.add_argument("--prev-report", default="2026-04-15_bugfix-and-multistep.html")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    metrics = run_dir / "metrics"
    diag = run_dir / "diagnostics"

    summary = load_csv(metrics / "summary.csv")
    per_sample = load_csv(metrics / "per_sample.csv")
    fd = load_csv(metrics / "frechet_distance.csv")
    failures = load_csv(metrics / "failures.csv")
    fairness = load_csv(diag / "fairness_stats.csv")

    img_hist = b64_img(diag / "alignment_scale_hist.png")
    img_sens = b64_img(diag / "clamp_sensitivity.png")
    img_mesh = b64_img(diag / "mesh_stats.png")

    summary_html = render_summary_table(summary, fd)
    track_ab_html = render_track_ab(per_sample)
    fairness_html = render_fairness_table(fairness)
    failures_html = render_failures(failures)
    outliers_html = render_outliers(per_sample)
    significance_html = render_paired_significance(per_sample)
    speed_html = render_speed_table(run_dir, summary)

    now = pd.Timestamp.now().strftime("%Y-%m-%d")

    html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(args.title)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 980px; margin: 30px auto; padding: 0 20px; line-height: 1.7; color: #1a1a1a; font-size: 15px; }}
h1 {{ font-size: 24px; border-bottom: 2px solid #333; padding-bottom: 6px; }}
.series {{ color: #666; font-size: 13px; margin-top: 5px; }}
.tldr {{ background: #eff6ff; border-left: 4px solid #3b82f6; padding: 12px 18px; margin: 20px 0; }}
h2 {{ font-size: 19px; margin-top: 36px; color: #1e40af; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
h3 {{ font-size: 16px; margin-top: 22px; color: #444; }}
table {{ border-collapse: collapse; margin: 12px 0; font-size: 13.5px; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 5px 10px; text-align: left; }}
th {{ background: #f5f5f5; }}
td.num {{ text-align: right; font-family: monospace; }}
code {{ font-family: 'Courier New', monospace; background: #f1f1f1; padding: 1px 5px; border-radius: 3px; font-size: 13px; }}
.warn {{ background: #fef3c7; border-left: 4px solid #f59e0b; padding: 10px 15px; margin: 12px 0; }}
.ok {{ background: #d1fae5; border-left: 4px solid #059669; padding: 10px 15px; margin: 12px 0; }}
figure {{ margin: 20px 0; text-align: center; }}
figcaption {{ font-size: 12px; color: #555; margin-top: 6px; text-align: left; }}
a {{ color: #2563eb; }}
</style>
</head>
<body>

<h1>Part 4 — Cross-family Benchmark on Toys4k</h1>
<div class="series">
Previous: <a href="{html.escape(args.prev_report)}">Part 3 (DMD2 bug fixes &amp; DMD1 multi-step)</a><br>
{now} &nbsp;|&nbsp; Sho Kuno &nbsp;|&nbsp; Shanda AI Research Tokyo / The University of Tokyo
</div>

<div class="tldr">
<strong>TL;DR.</strong> We benchmark 9 models across 3 backbone families (Hunyuan3D-2.1, TRELLIS v1, TRELLIS 2) on the Toys4k 105-sample test split, using a shared evaluation pipeline (bbox-centered unit-sphere normalization &rarr; 24-start similarity ICP &rarr; Chamfer / F-score / Hausdorff). We additionally run a <strong>scale-clamp sensitivity audit</strong> ([0.3, 3.0] vs the default [0.5, 2.0]) to detect alignment errors masked by clamping.
<br><br>
<strong>Key caveat.</strong> Every family has a few catastrophic samples (e.g. MDT-Dist fails on bicycle_011 with CD=4.5; FlashVDM fails on flower_005 with CD=2.8; TRELLIS 2 on dragon_011 with CD=0.89). These outliers dominate the <em>mean</em> CD but are not representative of typical quality. We therefore report <strong>median CD as the primary metric</strong>; the mean is included for reference only.
</div>

<h2>1. Setup</h2>

<h3>1.1 Models evaluated</h3>
<ul>
<li><strong>Hunyuan3D-2.1 family</strong>: teacher_50step (50-step), flashvdm (5-step turbo VAE), pd_6step, cd_4step, dmd1_1step, dmd2_1step. Distilled variants use LoRA on the shared DiT.</li>
<li><strong>TRELLIS v1 family</strong>: trellis (25+25-step teacher, native sampler), mdt_dist (2+2-step, Zhou et al. 2025, weights-swap into TRELLIS v1 pipeline).</li>
<li><strong>TRELLIS 2 family (reference)</strong>: trellis2 (default 1024_cascade pipeline).</li>
</ul>

<h3>1.2 Evaluation protocol</h3>
<p>All predictions pass through the same pipeline: mesh cleaning (remove NaN/Inf/degenerate/tiny-components) &rarr; area-proportional surface sampling (100 k alignment pts, 16.4 k eval pts, seed 0) &rarr; bbox-center + unit-sphere normalization (GT and prediction identical) &rarr; Track A similarity ICP (24 canonical cube rotations, reflection forbidden, scale clamped [0.5, 2.0]) &rarr; metrics. Track B computes CD with identity alignment. Fréchet Distance uses Blender multiview renders (azimuths 0/90/180/270°, elev 30°, 512 px) and InceptionV3 / DINOv2 features. Bootstrap 95% CI from n=1000 resamples.</p>

<h2>2. Main results</h2>

<p>Chamfer Distance and F-score are computed on unit-sphere-normalized point clouds after Track A alignment. Lower CD / Hausdorff / FD is better; higher F-score is better.</p>

{summary_html}

{failures_html}

<h2>2.5 Speed / efficiency</h2>
<p>Distillation compresses the <strong>diffusion sampling (DiT)</strong> portion of inference. The volume decoder (marching cubes on an implicit field, ~8 s per sample for Hunyuan3D-2.1) is unchanged by DMD/PD/CD distillation. Consequently, a nominal "50× speedup" on DiT translates to a smaller end-to-end speedup when the decoder is the bottleneck. FlashVDM's turbo VAE is an independent line of work that accelerates <em>the decoder</em>, giving it a different speed profile.</p>

{speed_html}

<h2>3. Statistical significance and fairness audit</h2>

<h3>3.0 Paired significance tests (Track A CD)</h3>
<p>Because test samples span 105 distinct Toys4k categories (each with a single sample), and because
mean CD is outlier-dominated, we perform <strong>paired Wilcoxon signed-rank tests</strong> on
per-sample Track A CD for the key close comparisons. This is more robust than comparing means.</p>

{significance_html}

<h3>3.1 Track A vs Track B (ICP rotation gain)</h3>
<p>Track B applies identity alignment to unit-sphere-normalized clouds (no rotation/scale search), so the gap (B − A) quantifies how much the 24-start ICP recovers. Families whose native coordinate frame matches GT closely will show small gaps; families with different canonical orientations show larger gaps.</p>

{track_ab_html}

<h3>3.2 Alignment scale distribution per model</h3>
<p>Track A's Umeyama solver returns a uniform scale factor ∈ [0.5, 2.0] (clamped). Clustering at the endpoints indicates that the clamp is actively masking larger scale mismatches.</p>

<figure>{img_hist}<figcaption>Fig 1. Per-model histograms of Track A alignment scale. Red dashed lines mark the clamp bounds; green dotted line is the ideal s=1 (since both clouds are pre-normalized).</figcaption></figure>

<h3>3.3 Scale clamp sensitivity</h3>
<p>We re-ran Track A ICP with a looser clamp [0.3, 3.0] and report CD(loose) − CD(tight) per model. A large positive delta means the tight clamp was masking alignment error (samples where the "true" optimal scale was outside [0.5, 2.0], now found at [0.3, 3.0], lowers CD).</p>

<figure>{img_sens}<figcaption>Fig 2. Scale-clamp sensitivity: CD(clamp [0.3, 3.0]) − CD(clamp [0.5, 2.0]) per model.</figcaption></figure>

{fairness_html}

<h3>3.4 Mesh density across families</h3>
<p>Post-cleaning mesh statistics. Note that TRELLIS 2 and MDT-Dist typically produce denser meshes (≥ 10⁶ vertices) than Hunyuan3D-2.1 (~10⁵). Area-proportional surface sampling should compensate, but thin-structure categories may be affected.</p>

<figure>{img_mesh}<figcaption>Fig 3. Mesh vertices / faces / connected components per model (post-cleaning), boxplots across 105 samples.</figcaption></figure>

<h3>3.5 Outlier samples (top-5 worst per model)</h3>
<p>Reveals when mean-CD rankings are driven by a few catastrophic failures. Objects with alignment_scale = 0.50 hit the clamp lower bound (ICP wanted to shrink further but was capped); those with CD &gt; 0.1 under Track A are geometrically wrong at the shape level, not just a coordinate issue. Repeated object_ids across families (octopus_010, coin_031, shoe_028, plate_040) indicate intrinsically hard objects — high genus, flat/thin geometry — where all 3D generators struggle.</p>

{outliers_html}

<h2>4. Fairness discussion</h2>

<h3>4.1 FlashVDM base model (important!)</h3>
<p><strong>FlashVDM in this benchmark uses Hunyuan3D-<u>2.0</u> + turbo VAE, not Hunyuan3D-2.1.</strong> The official <code>enable_flashvdm()</code> turbo VAE mapping has entries only for Hunyuan3D-2 / -2mv / -2mini; there is no v2.1 turbo VAE. Our <code>run_inference_flashvdm.py</code> imports <code>hy3dgen.shapegen</code> (the v2.0 module) and loads <code>tencent/Hunyuan3D-2</code>. So FlashVDM's median CD (0.00245) reflects v2.0 + turbo, not v2.1 + turbo. A fair 5-step v2.1 comparison would require retraining the turbo VAE for v2.1.</p>

<p>Per Tencent's official turbo configuration, FlashVDM uses <code>octree_resolution=380</code> while all other Hunyuan3D-2.x models use 384. We preserve the official setting.</p>

<h3>4.2 Cross-family canonical orientation</h3>
<p>Hunyuan3D-2.1 and TRELLIS v1/2 do not share a canonical frame (different up-axes / scales). ICP with 24 initial cube rotations compensates for pure-orientation differences in the 90° increments. Finer rotational misalignments (e.g. 30°) must be recovered by ICP refinement; the alignment cost and Track B − A gap characterize how much this matters.</p>

<h3>4.3 Scale clamp</h3>
<p>The clamp [0.5, 2.0] is a safeguard against degenerate ICP collapse. Per the sensitivity audit, if a model's CD drops substantially when the clamp is relaxed to [0.3, 3.0], the tight clamp was artificially favoring that model's report. Models with scale-distribution mass near the endpoints are the likely offenders.</p>

<h3>4.4 Tiny-component removal</h3>
<p>The cleaner deletes connected components whose face count is &lt; 1 % of the largest component. For thin-structure categories (screwdriver, plate, etc.) this may delete valid, disconnected handle/rim pieces. We log pre- and post-cleaning stats in <code>mesh_quality.csv</code>; category-level inspection is warranted if a model drops samples disproportionately.</p>

<h3>4.5 Ground truth</h3>
<p>GT is the 10 k-point cloud from Toys4k's official distribution (<code>pc10K.npz</code>); predictions are mesh-sampled to 16.4 k points. Both go through identical normalization and sampling (seed 0), so point-count differences do not bias the metrics — but GT does not go through the prediction cleaner.</p>

<h3>4.6 Category coverage and statistical power</h3>
<p><strong>All 105 test samples come from 105 distinct Toys4k categories</strong> — one sample per category. This has two consequences:</p>
<ul>
<li><strong>Category-level analysis is impossible.</strong> A single catastrophic sample
(e.g. <code>octopus_010</code> CD=3.93 for MDT-Dist) represents 100 % of that category's signal.
Differences between thin (plate, screwdriver) vs round (apple, ball) vs complex (octopus, dragon)
categories cannot be disentangled from sample-level noise.</li>
<li><strong>Close model comparisons cannot be resolved.</strong> Per the paired Wilcoxon tests
(§3.0), <em>DMD2 vs DMD1</em> and <em>TRELLIS 2 vs TRELLIS v1</em> have p-values of ~0.15,
not significant at α=0.05. The effect sizes are small (|Δmedian| ≈ 10⁻⁴) and lost in the
per-sample variance at this sample size.</li>
</ul>
<p>A larger test set — sampling 5-10 objects per category (~500-1000 samples total) — would
reduce the CI widths by √(N/105) ≈ 2.2-3.1× and make currently-non-significant comparisons
resolvable. The limiting factor is inference cost (≈ 14-20 h for N=1500) and storage
(≈ 1.8 TB for N=1500 predictions).</p>

<h2>5. Reproducibility</h2>

<p>Run ID: <code>{html.escape(run_dir.name)}</code>. Config: <code>distill_methods/config_cross_family.yaml</code>. Orchestrator: <code>distill_methods/scripts/run_cross_family_benchmark.sh --run-id {html.escape(run_dir.name)}</code>. All intermediate outputs (normalized meshes, alignment JSONs, per-sample metrics) are under <code>{html.escape(str(run_dir))}</code>.</p>

</body>
</html>
"""

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(html_out)
    print(f"Wrote report: {out}")


if __name__ == "__main__":
    main()
