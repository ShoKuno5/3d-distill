#!/usr/bin/env bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=1:30:00
#$ -N flashvdm-dsw
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/eval_flashvdm_dsw.qsub.log
#
# FlashVDM (HY3D-2.0 turbo) on 105 sample, DSW-verified recipe.

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "eval_flashvdm_dsw"
tsubame_setup_env

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/gs/fs/tga-koike-shanda2/sk/cache/uv
export TORCH_HOME=/gs/fs/tga-koike-shanda2/sk/cache/torch
export HF_HOME=/gs/fs/tga-koike-shanda2/sk/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0

VENV_PY=/gs/fs/tga-koike-shanda2/sk/envs/hy3d20_uv/bin/python
REPO_DIR=/gs/fs/tga-koike-shanda2/sk/models/hunyuan3d
SCRIPT=/gs/fs/tga-koike-shanda2/sk/3d-distill-eval/pipeline/scripts/run_inference_flashvdm.py
CONFIG=/gs/fs/tga-koike-shanda2/sk/3d-distill-eval/distill_methods/config_flashvdm_dsw.yaml

echo "================================================================"
echo "FlashVDM (HY3D-2.0 turbo, DSW recipe): $(date -Iseconds), host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader -i 0
echo "================================================================"

# Pre-DL HY3D-2.0 turbo model
echo "[step 1] HF DL tencent/Hunyuan3D-2 hunyuan3d-dit-v2-0-turbo subfolder"
$VENV_PY -c "
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id='tencent/Hunyuan3D-2', allow_patterns=['hunyuan3d-dit-v2-0-turbo/*'])
print('downloaded to:', p)
" 2>&1 | tail -10

# Inference (cd into models/hunyuan3d so hy3dgen package finds local egg-info)
cd $REPO_DIR
echo "[step 2] inference (105 samples, expect ~15 min)"
$VENV_PY $SCRIPT --config $CONFIG 2>&1 | tail -200

echo "================================================================"
echo "FlashVDM-DSW done: $(date -Iseconds)"
M2=/gs/fs/tga-koike-shanda2/sk/scratch/distill_methods/m2_cd_tsubame_20260519_1629
echo "predictions: $(find $M2/predictions/flashvdm_dsw -name 'mesh_raw.obj' 2>/dev/null | wc -l) / 105"
echo "================================================================"
