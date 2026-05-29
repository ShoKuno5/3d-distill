#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=4:00:00
#$ -N flashvdm-1050
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/eval_flashvdm_inference_1050.qsub.log
#
# FlashVDM (HY3D-2.0 turbo, DSW recipe) inference on Toys4k 1050.
# 4 GPU intra-node sharded. Uses hy3d20_uv venv (HY3D-2.0).

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "eval_flashvdm_inference_1050"
tsubame_setup_env

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/gs/fs/tga-koike-shanda2/sk/cache/uv
export TORCH_HOME=/gs/fs/tga-koike-shanda2/sk/cache/torch
export HF_HOME=/gs/fs/tga-koike-shanda2/sk/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0

VENV_PY=/gs/fs/tga-koike-shanda2/sk/envs/hy3d20_uv/bin/python
REPO_DIR=/gs/fs/tga-koike-shanda2/sk/models/hunyuan3d
INF_SCRIPT=/gs/fs/tga-koike-shanda2/sk/3d-distill-eval/pipeline/scripts/run_inference_flashvdm.py
INF_CFG=/gs/fs/tga-koike-shanda2/sk/3d-distill-eval/distill_methods/config_flashvdm_dsw_1050.yaml

EVAL_ROOT=/gs/fs/tga-koike-shanda2/sk/scratch/distill_methods/m2_cd_tsubame_20260519_1629/eval_1050
SHARD_LOG_DIR=$EVAL_ROOT/inference_logs/flashvdm_dsw
NUM_SHARDS=${NUM_SHARDS:-4}
SHARD_OFFSET=${SHARD_OFFSET:-0}

mkdir -p "$EVAL_ROOT" "$SHARD_LOG_DIR"

echo "================================================================"
echo "flashvdm_dsw 1050 inference: $(date -Iseconds), host=$(hostname)"
nvidia-smi -L
echo "inf_cfg: $INF_CFG"
echo "NUM_SHARDS=$NUM_SHARDS, SHARD_OFFSET=$SHARD_OFFSET"
echo "================================================================"

# Pre-DL HY3D-2.0 turbo weights (cached after first run)
echo "[step 0] HF DL tencent/Hunyuan3D-2 hunyuan3d-dit-v2-0-turbo subfolder"
$VENV_PY -c "
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id='tencent/Hunyuan3D-2', allow_patterns=['hunyuan3d-dit-v2-0-turbo/*'])
print('downloaded to:', p)
" 2>&1 | tail -10

cd $REPO_DIR

PIDS=()
for K in 0 1 2 3; do
    SHARD_IDX=$((K + SHARD_OFFSET))
    LOG="$SHARD_LOG_DIR/shard_${SHARD_IDX}.log"
    echo "[launch] GPU $K shard $SHARD_IDX/$NUM_SHARDS -> $LOG"
    CUDA_VISIBLE_DEVICES=$K $VENV_PY $INF_SCRIPT \
        --config $INF_CFG \
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
echo "predictions: $(find $EVAL_ROOT/predictions/flashvdm_dsw -name 'mesh_raw.obj' 2>/dev/null | wc -l) / 1050"

echo "================================================================"
echo "flashvdm_dsw 1050 inference done: $(date -Iseconds), rc=$EXIT_RC"
echo "================================================================"
exit $EXIT_RC
