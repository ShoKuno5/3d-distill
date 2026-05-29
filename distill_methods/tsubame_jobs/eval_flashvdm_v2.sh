#!/usr/bin/env bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=2:00:00
#$ -N flashvdm-v2
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/eval_flashvdm_v2.qsub.log
#
# flashvdm retry: higher octree_resolution to fix marching_cubes failures (35/105 → target 100+).
# Override --num-inference-steps / --octree-resolution via env if supported, else patch config.

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "eval_flashvdm_v2"
tsubame_setup_env

REPO_MAIN=/gs/fs/tga-koike-shanda2/sk/3d-distill
REPO_EVAL=/gs/fs/tga-koike-shanda2/sk/3d-distill-eval

# Use a higher-octree config (patch in-place)
CFG_SRC=$REPO_EVAL/distill_methods/config_tsubame_cd_eval.yaml
CFG_V2=$REPO_EVAL/distill_methods/config_tsubame_cd_eval_flashvdm_v2.yaml
cp $CFG_SRC $CFG_V2

# Bump octree_resolution and switch predictions_root for separate output dir
"$PY" - <<PY
import yaml
p = "$CFG_V2"
with open(p) as f: c = yaml.safe_load(f)
for m in c.get("models", []):
    if m["name"] == "flashvdm":
        m.setdefault("inference_params", {})["octree_resolution"] = 512
        m["predictions_root"] = "/gs/fs/tga-koike-shanda2/sk/scratch/distill_methods/m2_cd_tsubame_20260519_1629/predictions/flashvdm_v2/default" 
        print("flashvdm config:", m.get("inference_params"))
with open(p, "w") as f: yaml.safe_dump(c, f, sort_keys=False)
print("wrote:", p)
PY

echo "================================================================"
echo "flashvdm v2 (octree=512): $(date -Iseconds)"
echo "================================================================"

cd "$REPO_MAIN/models/hunyuan3d21/hy3dshape"
export PYTHONPATH="$REPO_MAIN/models/hunyuan3d21/hy3dshape:$REPO_EVAL/pipeline:$REPO_EVAL/distill_methods:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 "$PY" \
    "$REPO_EVAL/distill_methods/scripts/run_inference_distilled.py" \
    --config "$CFG_V2" \
    --model-name flashvdm

M2=/gs/fs/tga-koike-shanda2/sk/scratch/distill_methods/m2_cd_tsubame_20260519_1629
echo "v2 predictions: $(find $M2/predictions/flashvdm_v2 -name 'mesh_raw.obj' 2>/dev/null | wc -l)"
echo "================================================================"
echo "flashvdm v2 done: $(date -Iseconds)"
echo "================================================================"
