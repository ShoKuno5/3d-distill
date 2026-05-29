#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=3:00:00
#$ -N xmodal-1050
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/cross_modal_1050.qsub.log
#
# ULIP-I / Uni3D-I (cross-modal) for the 1050 cross-family benchmark.
# Needs NO multiview renders (point clouds sampled from meshes), so it runs
# independently of the FD/CLIP render step. Sharded by sample across the 4
# GPUs of one node; shard CSVs merged into metrics/cross_modal.csv.
#
# Validates the cross-modal namespace-isolation fix (ULIP + Uni3D in one
# process) on real data — the previous cross_modal.csv was entirely empty.
#
# Submit: qsub -ar 7083 -g tga-koike-shanda cross_modal_1050.sh

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "cross_modal_1050"
tsubame_setup_env

REPO_EVAL=/gs/fs/tga-koike-shanda2/sk/3d-distill-eval
CFG=$REPO_EVAL/distill_methods/config_tsubame_1050_eval.yaml
EVAL_ROOT=/gs/fs/tga-koike-shanda2/sk/scratch/distill_methods/m2_cd_tsubame_20260519_1629/eval_1050
MODELS=${MODELS:-"teacher_50step flashvdm_dsw cd_4step dmd2_1step mdt_dist trellis2"}
METRICS_DIR=$EVAL_ROOT/metrics
mkdir -p "$METRICS_DIR"

echo "================================================================"
echo "cross-modal 1050: $(date -Iseconds), host=$(hostname)"
nvidia-smi -L
echo "config: $CFG"; echo "models: $MODELS"; echo "eval_root: $EVAL_ROOT"
echo "================================================================"

cd "$REPO_EVAL"
export PYTHONPATH="$REPO_EVAL/pipeline:$REPO_EVAL/distill_methods:$PYTHONPATH"

NSHARDS=4
PIDS=()
for K in 0 1 2 3; do
    LOG=/gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/cross_modal_1050_shard${K}.log
    echo "[launch] GPU $K shard $K/$NSHARDS -> $LOG"
    CUDA_VISIBLE_DEVICES=$K "$PY" \
        "$REPO_EVAL/pipeline/scripts/run_cross_modal.py" \
        --config "$CFG" \
        --models $MODELS \
        --shard-index $K --num-shards $NSHARDS \
        --out-csv "$METRICS_DIR/cross_modal_shard${K}.csv" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

EXIT_RC=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || { echo "[FAIL] shard $i exit=$?"; EXIT_RC=1; }
done

echo "===== merge shard CSVs -> cross_modal.csv ====="
"$PY" - "$METRICS_DIR" $NSHARDS <<'PY'
import csv, sys
from pathlib import Path
metrics_dir = Path(sys.argv[1]); nshards = int(sys.argv[2])
rows = []
for k in range(nshards):
    p = metrics_dir / f"cross_modal_shard{k}.csv"
    if not p.exists():
        print(f"  WARNING: missing {p}"); continue
    with open(p) as f:
        r = csv.reader(f); next(r, None)
        rows.extend(list(r))
out = metrics_dir / "cross_modal.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["object_id", "model", "ulip_i", "uni3d_i", "error"])
    w.writerows(rows)
# quick sanity: how many rows have a numeric uni3d_i
n_ok = sum(1 for x in rows if len(x) >= 4 and x[3] not in ("", "None"))
print(f"  merged {len(rows)} rows -> {out}  (uni3d_i populated: {n_ok})")
PY

echo "================================================================"
echo "cross-modal 1050 done: $(date -Iseconds), rc=$EXIT_RC"
echo "head of cross_modal.csv:"; head -3 "$METRICS_DIR/cross_modal.csv" 2>/dev/null
echo "================================================================"
exit $EXIT_RC
