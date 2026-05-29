#!/usr/bin/env bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=2:00:00
#$ -N dmd2-sweep-eval
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/collect_dmd2_sweep_metrics.qsub.log
#
# Phase 0a metrics: run_eval (Track A, CD/F-score/Hausdorff only) for each
# swept checkpoint, then assemble a CD/F-vs-step table. Run AFTER the
# eval_dmd2_ckpt_sweep.sh inference jobs have populated each step's
# predictions. CPU-bound (ICP); reuses the training venv ($PY).
#
#   qsub -g tga-koike-shanda collect_dmd2_sweep_metrics.sh
#   # or run interactively on a login node:
#   STEPS="step_2000 step_4000 ..." bash collect_dmd2_sweep_metrics.sh

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "collect_dmd2_sweep_metrics"
tsubame_setup_env

REPO_EVAL=/gs/fs/tga-koike-shanda2/sk/3d-distill-eval
SWEEP_ROOT=/gs/fs/tga-koike-shanda2/sk/scratch/distill_methods/m2_cd_tsubame_20260519_1629/eval_1050/ckpt_sweep
STEPS="${STEPS:-step_2000 step_4000 step_6000 step_8000 step_10000 step_12000}"
WORKERS="${WORKERS:-16}"

cd "$REPO_EVAL"
export PYTHONPATH="$REPO_EVAL/pipeline:$REPO_EVAL/distill_methods:$PYTHONPATH"

for S in $STEPS; do
    CFG=$SWEEP_ROOT/$S/config_dmd2_${S}.yaml
    SUMMARY=$SWEEP_ROOT/$S/metrics/summary.csv
    if [ ! -f "$CFG" ]; then
        echo "[skip] $S: no config ($CFG) — inference not run yet?"
        continue
    fi
    if [ -f "$SUMMARY" ]; then
        echo "[skip] $S: summary exists ($SUMMARY)"
        continue
    fi
    NPRED=$(find "$SWEEP_ROOT/$S/predictions/dmd2_1step/default" -name 'mesh_raw.obj' 2>/dev/null | wc -l)
    echo "=== eval $S ($NPRED predictions) ==="
    CUDA_VISIBLE_DEVICES=0 "$PY" \
        "$REPO_EVAL/pipeline/scripts/run_eval.py" \
        --config "$CFG" \
        --models dmd2_1step \
        --skip-track-b \
        --workers "$WORKERS" || echo "[warn] run_eval failed for $S"
done

# Assemble Track A CD/F-score vs step into one CSV + stdout table.
"$PY" - "$SWEEP_ROOT" $STEPS <<'PYEOF'
import csv, os, sys
sweep_root = sys.argv[1]
steps = sys.argv[2:]
cols = ["chamfer_distance_median", "chamfer_distance_mean", "chamfer_distance_std",
        "f_score_001_median", "f_score_002_median", "hausdorff_median", "n_failures"]
out_rows = []
for s in steps:
    p = os.path.join(sweep_root, s, "metrics", "summary.csv")
    if not os.path.isfile(p):
        print(f"[miss] {s}: {p}")
        continue
    for r in csv.DictReader(open(p)):
        if r.get("model") == "dmd2_1step" and r.get("track") == "A":
            row = {"step": int(s.split("_")[1])}
            row.update({c: r.get(c, "") for c in cols})
            out_rows.append(row)
out_rows.sort(key=lambda r: r["step"])
out_csv = os.path.join(sweep_root, "sweep_curve.csv")
if out_rows:
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step"] + cols)
        w.writeheader(); w.writerows(out_rows)
    print(f"\n=== dmd2_1step CD/F-vs-step (Track A) -> {out_csv} ===")
    hdr = ["step", "CD_med", "CD_mean", "CD_std", "F@1%", "F@2%", "HD_med", "fail"]
    print("  ".join(f"{h:>9}" for h in hdr))
    for r in out_rows:
        vals = [r["step"], r["chamfer_distance_median"], r["chamfer_distance_mean"],
                r["chamfer_distance_std"], r["f_score_001_median"],
                r["f_score_002_median"], r["hausdorff_median"], r["n_failures"]]
        def fmt(v):
            try: return f"{float(v):.5g}"
            except (TypeError, ValueError): return str(v)
        print("  ".join(f"{fmt(v):>9}" for v in vals))
else:
    print("No summary rows found — run inference + eval first.")
PYEOF

echo "================================================================"
echo "dmd2 sweep metrics done: $(date -Iseconds)"
echo "================================================================"
