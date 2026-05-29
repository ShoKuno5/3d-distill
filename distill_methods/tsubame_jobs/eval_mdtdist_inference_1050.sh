#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=4:00:00
#$ -N mdtdist-1050
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/eval_mdtdist_inference_1050.qsub.log
#
# MDT-Dist (TRELLIS v1 + 2+2-step weight swap) inference on Toys4k 1050.
# 4 GPU intra-node sharded. Requires mdtdist_uv env + .pt ckpts in
# /gs/fs/tga-koike-shanda2/sk/models/mdt_dist/ckpts/.
# NOTE: run_inference_mdt_dist.py currently has no shard arg, so we patch
# samples per shard via TRELLIS_MDT_SHARD env vars (added below).

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "eval_mdtdist_inference_1050"
tsubame_setup_env

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/gs/fs/tga-koike-shanda2/sk/cache/uv
export TORCH_HOME=/gs/fs/tga-koike-shanda2/sk/cache/torch
export HF_HOME=/gs/fs/tga-koike-shanda2/sk/hf_cache
export SPCONV_ALGO=native
export TORCH_CUDA_ARCH_LIST="9.0"

VENV_PY=/gs/fs/tga-koike-shanda2/sk/envs/mdtdist_uv/bin/python
REPO_DIR=/gs/fs/tga-koike-shanda2/sk/models/trellis_v1
INF_SCRIPT=/gs/fs/tga-koike-shanda2/sk/3d-distill/pipeline/scripts/run_inference_mdt_dist.py
CFG=/gs/fs/tga-koike-shanda2/sk/3d-distill-eval/distill_methods/config_mdt_dist_1050.yaml

EVAL_ROOT=/gs/fs/tga-koike-shanda2/sk/scratch/distill_methods/m2_cd_tsubame_20260519_1629/eval_1050
SHARD_LOG_DIR=$EVAL_ROOT/inference_logs/mdt_dist
NUM_SHARDS=${NUM_SHARDS:-4}
SHARD_OFFSET=${SHARD_OFFSET:-0}

mkdir -p "$EVAL_ROOT" "$SHARD_LOG_DIR"

echo "================================================================"
echo "MDT-Dist 1050 inference: $(date -Iseconds), host=$(hostname)"
nvidia-smi -L
echo "config: $CFG"
echo "NUM_SHARDS=$NUM_SHARDS, SHARD_OFFSET=$SHARD_OFFSET"
echo "================================================================"

# Sanity: ckpts exist
for f in ss_flow_img_dit_L_16l8_fp16.pt slat_flow_img_dit_L_64l8p2_fp16.pt; do
    if [ ! -f "/gs/fs/tga-koike-shanda2/sk/models/mdt_dist/ckpts/$f" ]; then
        echo "ERROR: mdt-dist ckpt missing: $f" >&2
        exit 1
    fi
done

export PYTHONPATH="$REPO_DIR:$REPO_DIR:/gs/fs/tga-koike-shanda2/sk/3d-distill/pipeline:$PYTHONPATH"
cd "$REPO_DIR"

PIDS=()
for K in 0 1 2 3; do
    SHARD_IDX=$((K + SHARD_OFFSET))
    LOG="$SHARD_LOG_DIR/shard_${SHARD_IDX}.log"
    echo "[launch] GPU $K shard $SHARD_IDX/$NUM_SHARDS -> $LOG"
    # run_inference_mdt_dist.py doesn't support shard args natively, so we
    # use --max-samples + a wrapper that filters. Simpler: pass shard via
    # MDT_SHARD env var and rely on a small in-script slice (added by patch).
    MDT_SHARD_INDEX="$SHARD_IDX" \
    MDT_NUM_SHARDS="$NUM_SHARDS" \
    CUDA_VISIBLE_DEVICES=$K $VENV_PY $INF_SCRIPT \
        --config $CFG \
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
echo "predictions: $(find $EVAL_ROOT/predictions/mdt_dist -name 'mesh_raw.obj' 2>/dev/null | wc -l) / 1050"

echo "================================================================"
echo "MDT-Dist 1050 inference done: $(date -Iseconds), rc=$EXIT_RC"
echo "================================================================"
exit $EXIT_RC
