#!/usr/bin/env bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=0:30:00
#$ -N eval-blender-smoke
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/eval_blender_smoke.qsub.log
#
# Blender + multiview_renderer.render_multiview smoke (Day 1 of paper-grade eval upgrade).
# 1 mesh × 4 view (azim 0/90/180/270, elev 30, res 512) to verify Blender on TSUBAME compute node.

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "eval_blender_smoke"
tsubame_setup_env

REPO_EVAL=/gs/fs/tga-koike-shanda2/sk/3d-distill-eval

echo "================================================================"
echo "Blender smoke on $(hostname): $(date -Iseconds)"
echo "================================================================"

# 1) Blender binary check
echo "[1/3] Blender --version"
/gs/fs/tga-koike-shanda2/sk/blender-3.6.18-linux-x64/blender --version 2>&1 | head -3
echo ""

# 2) Find Blender via multiview_renderer's resolver
echo "[2/3] _find_blender resolver"
cd "$REPO_EVAL"
"$PY" - <<PY
import sys
sys.path.insert(0, "pipeline")
from src.evaluation.multiview_renderer import _find_blender, _BLENDER_SCRIPT, RESOLUTION
print("  candidates considered:", "$REPO_EVAL/envs/...", "/gs/fs/.../sk/blender-3.6.18-linux-x64", "/usr/bin/blender")
print("  resolved Blender:", _find_blender())
print("  render script   :", _BLENDER_SCRIPT)
print("  resolution      :", RESOLUTION)
PY
echo ""

# 3) Run multiview render on 1 test mesh
echo "[3/3] render_multiview on apple_007"
MESH=/gs/fs/tga-koike-shanda2/sk/data/toys4k/official/toys4k_obj_files/apple/apple_007/mesh.obj
OUT=/gs/fs/tga-koike-shanda2/sk/scratch/distill_methods/_blender_smoke_apple_007
mkdir -p "$OUT"
ls -la "$MESH"

cd "$REPO_EVAL"
"$PY" - <<PY
import sys, os, time
sys.path.insert(0, "pipeline")
from src.evaluation.multiview_renderer import render_multiview
t0 = time.time()
render_multiview("$MESH", "$OUT")
dt = time.time() - t0
print(f"render time: {dt:.2f}s")
print(f"outputs in {'$OUT'}:")
for f in sorted(os.listdir("$OUT")):
    p = os.path.join("$OUT", f)
    print(f"  {f}: {os.path.getsize(p)} bytes")
PY

echo "================================================================"
echo "Blender smoke done: $(date -Iseconds)"
echo "================================================================"
