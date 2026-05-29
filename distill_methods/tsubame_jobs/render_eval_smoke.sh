#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=0:30:00
#$ -N render-eval-smoke
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/render_eval_smoke.qsub.log
#
# Smoke test for the per-mesh Blender render fix on a real GPU node: render a
# handful of meshes (1 view-set each = 1 Blender launch each) and report timing.
# Verifies _blender_render_mesh.py multi-azimuth mode + render_multiview before
# committing the full 16-GPU run.
#
# Submit: qsub -ar 7083 -g tga-koike-shanda render_eval_smoke.sh

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "render_eval_smoke"
tsubame_setup_env

export BLENDER_BIN=/gs/fs/tga-koike-shanda2/sk/blender-3.6.18-linux-x64/blender
REPO_EVAL=/gs/fs/tga-koike-shanda2/sk/3d-distill-eval
CFG=$REPO_EVAL/distill_methods/config_tsubame_1050_eval.yaml
LIMIT=${LIMIT:-8}

echo "================================================================"
echo "render eval smoke: $(date -Iseconds), host=$(hostname)"
nvidia-smi -L
echo "config: $CFG  limit: $LIMIT"
echo "================================================================"

cd "$REPO_EVAL"
export PYTHONPATH="$REPO_EVAL/pipeline:$REPO_EVAL/distill_methods:$PYTHONPATH"

# One shard, GPU 0, first LIMIT per-model meshes (--skip-gt so we render
# UNcached predicted meshes and actually exercise the per-mesh Blender launch;
# GT meshes are mostly already cached and would be skipped).
CUDA_VISIBLE_DEVICES=0 "$PY" \
    "$REPO_EVAL/pipeline/scripts/render_eval_multiview.py" \
    --config "$CFG" \
    --models teacher_50step \
    --skip-gt \
    --shard-index 0 --num-shards 1 --limit "$LIMIT"
RC=$?

echo "===== sample outputs ====="
EVAL_ROOT=/gs/fs/tga-koike-shanda2/sk/scratch/distill_methods/m2_cd_tsubame_20260519_1629/eval_1050
find "$EVAL_ROOT/multiview_renders/teacher_50step" -name 'view_*.png' 2>/dev/null | head -8
echo "================================================================"
echo "render eval smoke done: $(date -Iseconds), rc=$RC"
echo "================================================================"
exit $RC
