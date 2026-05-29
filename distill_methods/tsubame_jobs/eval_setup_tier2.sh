#!/usr/bin/env bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=2:00:00
#$ -N eval-setup-tier2
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/eval_setup_tier2.qsub.log
#
# Day 1-B + Day 3-4 setup:
#   1. pip install open_clip_torch + huggingface_hub
#   2. DL ULIP-1 PointBERT (HF: SFXX/ulip)
#   3. DL Uni3D-Giant (HF: BAAI/Uni3D modelzoo/uni3d-g/model.pt)
#   4. DL EVA02-E-14-plus via open_clip (auto-cache + symlink to FlashVDM expected path)
#   5. pip install pointnet2_ops (CUDA build)
#   6. Sanity: import ULIP, Uni3DScore, open_clip

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "eval_setup_tier2"
tsubame_setup_env
export PATH="$HOME/.local/bin:$PATH"  # uv
export UV_CACHE_DIR=/gs/fs/tga-koike-shanda2/sk/cache/uv  # home quota avoidance
export TORCH_HOME=/gs/fs/tga-koike-shanda2/sk/cache/torch

EXTERNAL=/gs/fs/tga-koike-shanda2/sk/3d-distill-eval/pipeline/src/evaluation/external

echo "================================================================"
echo "eval-setup-tier2: $(date -Iseconds), host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader -i 0
nvcc --version 2>&1 | head -4
echo "================================================================"

# ----- 1. pip install open_clip_torch + huggingface_hub -----
echo ""
echo "[1/6] pip install open_clip_torch + huggingface_hub"
uv pip install --python "$PY" --upgrade open_clip_torch huggingface_hub timm trimesh 2>&1 | tail -5

# ----- 2. DL ULIP-1 PointBERT -----
echo ""
echo "[2/6] DL ULIP-1 PointBERT (865 MB)"
ULIP_CKPT="$EXTERNAL/ulip/ULIP/pretrained_models_ckpt_zero-shot_classification_checkpoint_pointbert.pt"
mkdir -p "$(dirname $ULIP_CKPT)"
if [ -f "$ULIP_CKPT" ]; then
    echo "  exists: $ULIP_CKPT ($(stat -c %s $ULIP_CKPT) bytes), skipping"
else
    "$PY" -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id='SFXX/ulip', repo_type='dataset',
    filename='ULIP-1/pretrained_models/ckpt_zero-sho_classification/checkpoint_pointbert.pt')
print('downloaded:', p)
import shutil; shutil.copy(p, '$ULIP_CKPT')
print('placed at:', '$ULIP_CKPT')
"
fi

# ----- 3. DL Uni3D-Giant -----
echo ""
echo "[3/6] DL Uni3D-Giant (~3 GB)"
UNI3D_CKPT="$EXTERNAL/uni3d/uni3d/model.pt"
mkdir -p "$(dirname $UNI3D_CKPT)"
if [ -f "$UNI3D_CKPT" ]; then
    echo "  exists: $UNI3D_CKPT ($(stat -c %s $UNI3D_CKPT) bytes), skipping"
else
    "$PY" -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id='BAAI/Uni3D', filename='modelzoo/uni3d-g/model.pt')
print('downloaded:', p)
import shutil; shutil.copy(p, '$UNI3D_CKPT')
print('placed at:', '$UNI3D_CKPT')
"
fi

# ----- 4. DL EVA02-E-14-plus open_clip weights -----
echo ""
echo "[4/6] DL EVA02-E-14-plus open_clip weights (~10 GB)"
OPENCLIP_CKPT="$EXTERNAL/uni3d/uni3d/open_clip_pytorch_model.bin"
if [ -f "$OPENCLIP_CKPT" ]; then
    echo "  exists: $OPENCLIP_CKPT ($(stat -c %s $OPENCLIP_CKPT) bytes), skipping"
else
    "$PY" -c "
import open_clip
# EVA02-E-14-plus default pretrained is laion2b_s9b_b144k.
m, _, _ = open_clip.create_model_and_transforms(
    'EVA02-E-14-plus', pretrained='laion2b_s9b_b144k',
    cache_dir='$HF_HOME/open_clip'
)
# Find the cached .bin file
import os, glob
candidates = glob.glob('$HF_HOME/open_clip/**/open_clip_pytorch_model.bin', recursive=True)
print('candidates:', candidates)
if candidates:
    import shutil; shutil.copy(candidates[0], '$OPENCLIP_CKPT')
    print('placed at:', '$OPENCLIP_CKPT')
else:
    print('WARN: no .bin found in cache')
"
fi

# ----- 5. Build pointnet2_ops -----
echo ""
echo "[5/6] pip install pointnet2_ops (CUDA build)"
POINTNET2=$EXTERNAL/ulip/pointnet2_ops-main
if "$PY" -c "import pointnet2_ops" 2>/dev/null; then
    echo "  already installed"
else
    cd $POINTNET2
    uv pip install --python "$PY" -e . 2>&1 | tail -10
fi

# ----- 6. Sanity -----
echo ""
echo "[6/6] Sanity imports"
cd $EXTERNAL
"$PY" -c "
import sys
sys.path.insert(0, '.')
print('cwd:', __import__('os').getcwd())
# ULIP
try:
    from ulip.ulip_score import ULIP
    print('  OK: ulip.ulip_score.ULIP importable')
except Exception as e:
    print('  FAIL ULIP import:', e)
# Uni3DScore
try:
    from uni3d.uni3d_score import Uni3DScore
    print('  OK: uni3d.uni3d_score.Uni3DScore importable')
except Exception as e:
    print('  FAIL Uni3DScore import:', e)
# open_clip EVA02
try:
    import open_clip
    print('  OK: open_clip', open_clip.__version__)
except Exception as e:
    print('  FAIL open_clip:', e)
"

echo ""
echo "=== Final ckpt status ==="
for f in "$ULIP_CKPT" "$UNI3D_CKPT" "$OPENCLIP_CKPT"; do
    if [ -f "$f" ]; then
        echo "  ✓ $f ($(stat -c %s $f) bytes)"
    else
        echo "  ✗ MISSING $f"
    fi
done

echo "================================================================"
echo "eval-setup-tier2 done: $(date -Iseconds)"
echo "================================================================"
