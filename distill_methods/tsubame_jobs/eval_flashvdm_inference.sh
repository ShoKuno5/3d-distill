#!/usr/bin/env bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=2:00:00
#$ -N flashvdm-inference
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/eval_flashvdm_inference.qsub.log
#
# flashvdm (HY3D-2.0 turbo + flash decoder) inference on Toys4k 105 sample.
#   - DL tencent/Hunyuan3D-2 weights if not in cache (~5 GB)
#   - Use existing 3d-distill venv (FlashVDM is integrated via HY3D-2.0 turbo)
#   - Output: predictions/flashvdm/default/<oid>/mesh_raw.obj

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "eval_flashvdm_inference"
tsubame_setup_env

REPO_MAIN=/gs/fs/tga-koike-shanda2/sk/3d-distill
REPO_EVAL=/gs/fs/tga-koike-shanda2/sk/3d-distill-eval
CFG=$REPO_EVAL/distill_methods/config_tsubame_cd_eval.yaml

echo "================================================================"
echo "flashvdm inference: $(date -Iseconds), host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader -i 0
echo "================================================================"

# Pre-warm HY3D-2.0 cache (small DL if needed)
echo ""
echo "[0/2] DL tencent/Hunyuan3D-2 weights if missing"
"$PY" -c "
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id='tencent/Hunyuan3D-2', allow_patterns=['*.json', '*.safetensors', '*.fp16.ckpt', '*.txt', '*.yaml'])
print('cache root:', p)
"

# Run inference
echo ""
echo "[1/2] Run inference"
cd "$REPO_MAIN/models/hunyuan3d21/hy3dshape"
export PYTHONPATH="$REPO_MAIN/models/hunyuan3d21/hy3dshape:$REPO_EVAL/pipeline:$REPO_EVAL/distill_methods:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 "$PY" \
    "$REPO_EVAL/distill_methods/scripts/run_inference_distilled.py" \
    --config "$CFG" \
    --model-name flashvdm

echo ""
echo "[2/2] Summary"
M2=/gs/fs/tga-koike-shanda2/sk/scratch/distill_methods/m2_cd_tsubame_20260519_1629
echo "predictions: $(find $M2/predictions/flashvdm -name 'mesh_raw.obj' 2>/dev/null | wc -l) / 105"
echo "================================================================"
echo "flashvdm done: $(date -Iseconds)"
echo "================================================================"
