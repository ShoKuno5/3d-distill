#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=2:00:00
#$ -N dmd2-sweep
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/eval_dmd2_ckpt_sweep.qsub.log
#
# Phase 0a: dmd2_1step checkpoint x metric sweep (1 node = 4 GPU, sharded).
# Runs inference for ONE checkpoint step into a per-step output dir so the
# CD/F-score-vs-step curve can be assembled. Submit one job per step:
#
#   for S in step_2000 step_4000 step_6000 step_8000 step_10000 step_12000; do
#     qsub -g tga-koike-shanda -v DMD2_CKPT_STEP=$S eval_dmd2_ckpt_sweep.sh
#   done
#
# (Submit only 2-3 at a time to leave node headroom on the shared account.)
# Metrics are computed separately (CPU) by collect_dmd2_sweep_metrics.sh after
# inference, then assembled into a CD/F-vs-step table.

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "eval_dmd2_ckpt_sweep"
tsubame_setup_env
export HY3DGEN_MODELS=/gs/fs/tga-koike-shanda2/sk/cache/hy3dgen

REPO_MAIN=/gs/fs/tga-koike-shanda2/sk/3d-distill
REPO_EVAL=/gs/fs/tga-koike-shanda2/sk/3d-distill-eval
CFG_SRC=$REPO_EVAL/distill_methods/config_tsubame_1050_eval.yaml

DMD2_RUN=/gs/fs/tga-koike-shanda2/sk/scratch/distill_methods/m2_dmd2_multinode_20260519_1022
DMD2_CKPT_STEP=${DMD2_CKPT_STEP:?set DMD2_CKPT_STEP=step_NNNN}
DMD2_LORA=$DMD2_RUN/checkpoints/dmd2/$DMD2_CKPT_STEP

# Per-step output root keeps each checkpoint's predictions/metrics separate.
SWEEP_ROOT=/gs/fs/tga-koike-shanda2/sk/scratch/distill_methods/m2_cd_tsubame_20260519_1629/eval_1050/ckpt_sweep
STEP_OUT=$SWEEP_ROOT/$DMD2_CKPT_STEP
PRED_ROOT=$STEP_OUT/predictions/dmd2_1step/default
CFG=$STEP_OUT/config_dmd2_${DMD2_CKPT_STEP}.yaml
SHARD_LOG_DIR=$STEP_OUT/inference_logs
NUM_SHARDS=${NUM_SHARDS:-4}
SHARD_OFFSET=${SHARD_OFFSET:-0}

mkdir -p "$PRED_ROOT" "$SHARD_LOG_DIR"

echo "================================================================"
echo "dmd2 ckpt sweep: $DMD2_CKPT_STEP  $(date -Iseconds), host=$(hostname)"
nvidia-smi -L
echo "lora:  $DMD2_LORA"
echo "preds: $PRED_ROOT"
echo "================================================================"

if [ ! -d "$DMD2_LORA" ]; then
    echo "ERROR: DMD2 ckpt missing: $DMD2_LORA" >&2
    exit 1
fi

# Build a per-step eval config: point dmd2_1step at this checkpoint + per-step
# predictions_root + this step's output_root, and disable the expensive metrics
# (frechet/clip/cross_modal) so the sweep only computes CD / F-score / Hausdorff.
"$PY" - <<PYEOF
import yaml
with open("$CFG_SRC") as f:
    c = yaml.safe_load(f)
c["output_root"] = "$STEP_OUT"
for m in c.get("models", []):
    if m.get("name") == "dmd2_1step":
        ip = m.setdefault("inference_params", {})
        ip["lora_path"] = "$DMD2_LORA"
        m["predictions_root"] = "$PRED_ROOT"
for key in ("frechet_distance", "clip_image", "cross_modal"):
    if key in c.get("metrics", {}):
        c["metrics"][key]["enabled"] = False
with open("$CFG", "w") as f:
    yaml.safe_dump(c, f, sort_keys=False)
print("config:", "$CFG")
PYEOF

cd "$REPO_MAIN/models/hunyuan3d21/hy3dshape"
export PYTHONPATH="$REPO_MAIN/models/hunyuan3d21/hy3dshape:$REPO_EVAL/pipeline:$REPO_EVAL/distill_methods:$PYTHONPATH"

PIDS=()
for K in 0 1 2 3; do
    SHARD_IDX=$((K + SHARD_OFFSET))
    LOG="$SHARD_LOG_DIR/shard_${SHARD_IDX}.log"
    echo "[launch] GPU $K shard $SHARD_IDX/$NUM_SHARDS -> $LOG"
    CUDA_VISIBLE_DEVICES=$K "$PY" \
        "$REPO_EVAL/distill_methods/scripts/run_inference_distilled.py" \
        --config "$CFG" \
        --model-name dmd2_1step \
        --shard-index "$SHARD_IDX" \
        --num-shards "$NUM_SHARDS" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

EXIT_RC=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[FAIL] shard $((i + SHARD_OFFSET)) exit=$rc"
        EXIT_RC=$rc
    fi
done

echo ""
echo "===== Summary ($DMD2_CKPT_STEP) ====="
echo "predictions: $(find "$PRED_ROOT" -name 'mesh_raw.obj' 2>/dev/null | wc -l) / 1050"
echo "config for metrics: $CFG"
echo "================================================================"
echo "dmd2 ckpt sweep $DMD2_CKPT_STEP done: $(date -Iseconds), rc=$EXIT_RC"
echo "================================================================"
exit $EXIT_RC
