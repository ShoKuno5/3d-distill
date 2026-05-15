#!/usr/bin/env bash
#$ -cwd
#$ -l cpu_16
#$ -l h_rt=2:00:00
#$ -N tsubame-setup
#$ -j y
#$ -o /gs/fs/tga-koike-shanda/kuno/scratch/tsubame_logs/setup_env.qsub.log
#
# One-shot environment bootstrap for TSUBAME H100 distill jobs.
# Idempotent: rerunnable; existing artifacts are skipped.
#
# What this does:
#   1. mkdir -p $TSUBAME_ROOT/{hf_cache,data,scratch/tsubame_logs}
#   2. install uv (user-local) if missing
#   3. ensure repo is up-to-date (assumes already cloned by hand)
#   4. uv venv (Python 3.10) + torch 2.6.0+cu124 + distill deps
#   5. HF snapshot_download tencent/Hunyuan3D-2.1 → hf_cache
#
# Prereqs (done manually before submit):
#   - mkdir -p /gs/fs/tga-koike-shanda/kuno
#   - cd /gs/fs/tga-koike-shanda/kuno && git clone -b exp/scaling-experiments \
#       https://github.com/ShoKuno5/3d-distill.git
#   - echo "WANDB_API_KEY=..." > /gs/fs/tga-koike-shanda/kuno/3d-distill/.env (chmod 600)
#
# Submit (no -ar needed; cpu_16 fits in queue without consuming reservation):
#   qsub -g tga-koike-shanda /gs/fs/tga-koike-shanda/kuno/3d-distill/distill_methods/tsubame_jobs/setup_env.sh
#
# Or with reservation (faster scheduling, uses points):
#   qsub -ar 6925 -g tga-koike-shanda <path>/setup_env.sh

set -euo pipefail

TSUBAME_ROOT="${TSUBAME_ROOT:-/gs/fs/tga-koike-shanda/kuno}"
REPO="$TSUBAME_ROOT/3d-distill"
LOGDIR="$TSUBAME_ROOT/scratch/tsubame_logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/setup_$(hostname)_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "[log mirror: $LOG]"

echo "================================================================"
echo "TSUBAME env setup: $(date -Iseconds)"
echo "host: $(hostname), user: $(whoami)"
echo "================================================================"

# --- 1. mkdir ---
echo
echo "[1/5] mkdir hierarchy"
mkdir -p "$TSUBAME_ROOT"/{hf_cache,data,scratch/tsubame_logs,scratch/distill_methods}
df -h "$TSUBAME_ROOT" | tail -1

# --- 2. uv install ---
echo
echo "[2/5] uv install (user-local)"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# Source uv env (the installer writes to ~/.local/bin/env)
if [ -f "$HOME/.local/bin/env" ]; then
    . "$HOME/.local/bin/env"
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

# --- 3. Repo state ---
echo
echo "[3/5] repo state"
if [ ! -d "$REPO/.git" ]; then
    echo "ERROR: $REPO not found. Manually clone first:"
    echo "  cd $TSUBAME_ROOT && git clone -b exp/scaling-experiments https://github.com/ShoKuno5/3d-distill.git"
    exit 1
fi
cd "$REPO"
git fetch origin
git checkout exp/scaling-experiments
git pull --ff-only
git log --oneline -3

# --- 4. venv + deps ---
echo
echo "[4/5] uv venv + torch + deps"
module load cuda/12.8.0 cudnn/9.8.0 2>/dev/null || echo "[warn] module load failed (non-interactive shell?)"

if [ ! -d ".venv" ]; then
    uv venv --python 3.10 .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python --version

# Torch first (largest wheel, separate index)
if ! python -c "import torch" 2>/dev/null; then
    echo "Installing PyTorch 2.6.0+cu124 ..."
    uv pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124
fi
python -c "import torch; print(f'torch {torch.__version__}, cuda {torch.version.cuda}')"

# Core distill deps (from src/* imports)
echo "Installing core deps ..."
uv pip install \
    numpy scipy pandas matplotlib pyyaml pillow trimesh \
    scikit-learn peft transformers diffusers accelerate \
    huggingface_hub wandb einops omegaconf safetensors

# Hunyuan3D-2.1 inference helpers (slow build, do after core)
echo "Installing pymeshlab + opencv-python-headless ..."
uv pip install pymeshlab opencv-python-headless

python -c "
import torch, peft, trimesh, transformers, diffusers, accelerate, yaml
print('  torch       :', torch.__version__)
print('  peft        :', peft.__version__)
print('  trimesh     :', trimesh.__version__)
print('  transformers:', transformers.__version__)
print('  diffusers   :', diffusers.__version__)
print('  accelerate  :', accelerate.__version__)
print('OK core imports')
"

# --- 5. HF Hunyuan3D-2.1 download ---
echo
echo "[5/5] HF download tencent/Hunyuan3D-2.1 (~14 GB)"
export HF_HOME="$TSUBAME_ROOT/hf_cache"
if [ -d "$HF_HOME/hub/models--tencent--Hunyuan3D-2.1" ]; then
    echo "  hf cache already populated:"
    du -sh "$HF_HOME"
else
    python -c "
from huggingface_hub import snapshot_download
p = snapshot_download('tencent/Hunyuan3D-2.1', cache_dir='$HF_HOME/hub')
print('  downloaded to', p)
"
    du -sh "$HF_HOME"
fi

echo
echo "================================================================"
echo "Setup complete: $(date -Iseconds)"
echo "  $TSUBAME_ROOT:"
du -sh "$TSUBAME_ROOT"/* 2>/dev/null
echo "  venv: $REPO/.venv ($(du -sh $REPO/.venv | cut -f1))"
echo "================================================================"
echo
echo "Next steps:"
echo "  1. Put WANDB_API_KEY in $REPO/.env (if not done)"
echo "  2. Sync HSSD 525 latents + DMD1 pairs from sk-train to $TSUBAME_ROOT/data/"
echo "  3. Run smoke: qrsh -l node_q=1 -ar 6925 -g tga-koike-shanda -l h_rt=1:00:00"
echo "     then: bash $REPO/distill_methods/tsubame_jobs/smoke.sh"
