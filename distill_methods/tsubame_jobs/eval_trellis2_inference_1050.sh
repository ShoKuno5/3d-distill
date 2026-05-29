#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=4:00:00
#$ -N trellis2-1050
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/eval_trellis2_inference_1050.qsub.log
#
# TRELLIS.2 inference on Toys4k 1050, 4 GPU intra-node sharded.
# Requires trellis2_uv env (cumesh/o-voxel/flexgemm/nvdiffrast built).

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "eval_trellis2_inference_1050"
tsubame_setup_env

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/gs/fs/tga-koike-shanda2/sk/cache/uv
export TORCH_HOME=/gs/fs/tga-koike-shanda2/sk/cache/torch
export HF_HOME=/gs/fs/tga-koike-shanda2/sk/hf_cache

VENV_PY=/gs/fs/tga-koike-shanda2/sk/envs/trellis2_uv/bin/python
REPO_DIR=/gs/fs/tga-koike-shanda2/sk/models/trellis2
SCRIPT=/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/scripts/trellis2_inference.py

MANIFEST=/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/manifests/toys4k_eval_1000/test.csv
EVAL_ROOT=/gs/fs/tga-koike-shanda2/sk/scratch/distill_methods/m2_cd_tsubame_20260519_1629/eval_1050
TRELLIS2_OUTPUT=$EVAL_ROOT/predictions/trellis2/default
SHARD_LOG_DIR=$EVAL_ROOT/inference_logs/trellis2
NUM_SHARDS=${NUM_SHARDS:-4}
SHARD_OFFSET=${SHARD_OFFSET:-0}

mkdir -p "$TRELLIS2_OUTPUT" "$SHARD_LOG_DIR"

echo "================================================================"
echo "TRELLIS.2 1050 inference: $(date -Iseconds), host=$(hostname)"
nvidia-smi -L
echo "manifest: $MANIFEST"
echo "output: $TRELLIS2_OUTPUT"
echo "NUM_SHARDS=$NUM_SHARDS, SHARD_OFFSET=$SHARD_OFFSET"
echo "================================================================"

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
cd $REPO_DIR

PIDS=()
for K in 0 1 2 3; do
    SHARD_IDX=$((K + SHARD_OFFSET))
    LOG="$SHARD_LOG_DIR/shard_${SHARD_IDX}.log"
    echo "[launch] GPU $K shard $SHARD_IDX/$NUM_SHARDS -> $LOG"
    TRELLIS2_MANIFEST="$MANIFEST" \
    TRELLIS2_OUTPUT="$TRELLIS2_OUTPUT" \
    TRELLIS2_SHARD_INDEX="$SHARD_IDX" \
    TRELLIS2_NUM_SHARDS="$NUM_SHARDS" \
    CUDA_VISIBLE_DEVICES=$K $VENV_PY $SCRIPT > "$LOG" 2>&1 &
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
echo "predictions: $(find $TRELLIS2_OUTPUT -name 'mesh.obj' 2>/dev/null | wc -l) / 1050"

echo "================================================================"
echo "TRELLIS.2 1050 inference done: $(date -Iseconds), rc=$EXIT_RC"
echo "================================================================"
exit $EXIT_RC
