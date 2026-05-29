#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=2:00:00
#$ -N dmd2-1050
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/eval_dmd2_inference_1050.qsub.log
#
# dmd2_1step inference on Toys4k 1050, 4 GPU intra-node sharded.
# DMD2_CKPT_STEP env var picks the lora ckpt step (default step_12000 = latest from h_rt cut).

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "eval_dmd2_inference_1050"
tsubame_setup_env
export HY3DGEN_MODELS=/gs/fs/tga-koike-shanda2/sk/cache/hy3dgen

REPO_MAIN=/gs/fs/tga-koike-shanda2/sk/3d-distill
REPO_EVAL=/gs/fs/tga-koike-shanda2/sk/3d-distill-eval
CFG_SRC=$REPO_EVAL/distill_methods/config_tsubame_1050_eval.yaml
CFG=$REPO_EVAL/distill_methods/config_tsubame_1050_eval_dmd2.yaml

DMD2_RUN=/gs/fs/tga-koike-shanda2/sk/scratch/distill_methods/m2_dmd2_multinode_20260519_1022
DMD2_CKPT_STEP=${DMD2_CKPT_STEP:-step_12000}
DMD2_LORA=$DMD2_RUN/checkpoints/dmd2/$DMD2_CKPT_STEP

EVAL_ROOT=/gs/fs/tga-koike-shanda2/sk/scratch/distill_methods/m2_cd_tsubame_20260519_1629/eval_1050
SHARD_LOG_DIR=$EVAL_ROOT/inference_logs/dmd2_1step
NUM_SHARDS=${NUM_SHARDS:-4}
SHARD_OFFSET=${SHARD_OFFSET:-0}

mkdir -p "$EVAL_ROOT" "$SHARD_LOG_DIR"

echo "================================================================"
echo "dmd2_1step 1050 inference: $(date -Iseconds), host=$(hostname)"
nvidia-smi -L
echo "lora: $DMD2_LORA"
echo "NUM_SHARDS=$NUM_SHARDS, SHARD_OFFSET=$SHARD_OFFSET"
echo "================================================================"

if [ ! -d "$DMD2_LORA" ]; then
    echo "ERROR: DMD2 ckpt missing: $DMD2_LORA" >&2
    exit 1
fi

# Build dmd2 eval config: patch lora_path
"$PY" - <<PYEOF
import yaml
with open("$CFG_SRC") as f: c = yaml.safe_load(f)
for m in c.get("models", []):
    if m.get("name") == "dmd2_1step":
        m.setdefault("inference_params", {})["lora_path"] = "$DMD2_LORA"
with open("$CFG", "w") as f: yaml.safe_dump(c, f, sort_keys=False)
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
echo "===== Summary ====="
echo "predictions: $(find $EVAL_ROOT/predictions/dmd2_1step -name 'mesh_raw.obj' 2>/dev/null | wc -l) / 1050"

echo "================================================================"
echo "dmd2_1step 1050 inference done: $(date -Iseconds), rc=$EXIT_RC"
echo "================================================================"
exit $EXIT_RC
