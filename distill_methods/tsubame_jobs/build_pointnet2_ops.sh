#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=0:30:00
#$ -N build-pn2
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/build_pointnet2_ops.qsub.log
#
# Rebuild the pointnet2_ops CUDA extension for H100 (sm_90). The shipped
# _ext*.so was built for another arch, so ULIP's PointBERT furthest-point
# sampling fails on H100 with "no kernel image is available for execution on
# the device". Build on an H100 node with CUDA 12.8 and TORCH_CUDA_ARCH_LIST=9.0.
#
# Submit: qsub -ar 7083 -g tga-koike-shanda build_pointnet2_ops.sh

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "build_pointnet2_ops"
tsubame_setup_env

# torch in the venv is built against CUDA 13.0; _common loads cuda/12.8.0, which
# torch's cpp_extension rejects (major mismatch). Switch to cuda/13.x to build.
module unload cuda/12.8.0 2>/dev/null || true
module load cuda/13.1.1 2>/dev/null || module load cuda/13 2>/dev/null || true

# _common.sh runs `set -euo pipefail`; we handle errors manually (and an `ls`
# of an already-removed .so must not abort the job), so relax it here.
set +eu

VENV=/gs/fs/tga-koike-shanda2/sk/3d-distill/.venv
PY=$VENV/bin/python
UV=$HOME/.local/bin/uv
export PATH="$VENV/bin:$PATH"   # so torch's cpp_extension finds the `ninja` binary
PN2=/gs/fs/tga-koike-shanda2/sk/3d-distill-eval/pipeline/src/evaluation/external/ulip/pointnet2_ops-main

echo "================================================================"
echo "build pointnet2_ops (sm_90): $(date -Iseconds), host=$(hostname)"
nvidia-smi -L | head -1
echo "CUDA_HOME=${CUDA_HOME:-<unset>}; nvcc: $(command -v nvcc)"
nvcc --version 2>/dev/null | tail -2
echo "================================================================"

# Build deps (compute node: uv has no login thread limit here).
VIRTUAL_ENV=$VENV "$UV" pip install setuptools wheel ninja 2>&1 | tail -3

cd "$PN2"
echo "removing old extension:"; ls -la pointnet2_ops/_ext*.so 2>/dev/null || echo "  (no existing .so)"
rm -f pointnet2_ops/_ext*.so

export TORCH_CUDA_ARCH_LIST="9.0"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
"$PY" setup.py build_ext --inplace
RC=$?
echo "build_ext rc=$RC"
ls -la pointnet2_ops/_ext*.so 2>/dev/null

echo "===== verify FPS kernel on H100 ====="
CUDA_VISIBLE_DEVICES=0 "$PY" - <<'PY'
import torch, sys
sys.path.insert(0, ".")
from pointnet2_ops import pointnet2_utils as pu
x = torch.rand(2, 1024, 3, device="cuda")
idx = pu.furthest_point_sample(x, 256)
print("FPS OK:", tuple(idx.shape), idx.dtype, "device", idx.device)
PY
VRC=$?

echo "================================================================"
echo "build pointnet2_ops done: $(date -Iseconds), build_rc=$RC verify_rc=$VRC"
echo "================================================================"
exit $(( RC || VRC ))
