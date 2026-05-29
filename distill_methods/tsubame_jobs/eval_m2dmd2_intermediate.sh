#!/usr/bin/env bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=2:00:00
#$ -N m2dmd2-step7k
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/eval_m2dmd2_intermediate.qsub.log
#
# M2 DMD2 intermediate eval: use step_7000 LoRA (latest available, training continues)
# Inference + eval (CD/F-score/Hausdorff/FD/CLIP-I/ULIP-I/Uni3D-I) on Toys4k 105.

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "eval_m2dmd2_intermediate"
tsubame_setup_env

REPO_MAIN=/gs/fs/tga-koike-shanda2/sk/3d-distill
REPO_EVAL=/gs/fs/tga-koike-shanda2/sk/3d-distill-eval
DMD2_RUN=/gs/fs/tga-koike-shanda2/sk/scratch/distill_methods/m2_dmd2_multinode_20260519_1022
CFG_SRC=$REPO_EVAL/distill_methods/config_tsubame_cd_eval.yaml
CFG=$REPO_EVAL/distill_methods/config_tsubame_dmd2_eval.yaml

echo "================================================================"
echo "M2 DMD2 step_7000 eval: $(date -Iseconds), host=$(hostname)"
echo "================================================================"

# Build DMD2 eval config: output_root → DMD2 run dir, lora_path → step_7000
"$PY" - <<PYEOF
import yaml
with open("$CFG_SRC") as f: c = yaml.safe_load(f)
c["output_root"] = "$DMD2_RUN"
for m in c.get("models", []):
    if m.get("name") == "dmd2_1step":
        m.setdefault("inference_params", {})["lora_path"] = "$DMD2_RUN/checkpoints/dmd2/step_7000"
        # Force predictions_root to a step_7000-specific dir to avoid colliding with final
        m["predictions_root"] = "$DMD2_RUN/predictions/dmd2_1step_step7000/default"
with open("$CFG", "w") as f: yaml.safe_dump(c, f, sort_keys=False)
print("config:", "$CFG")
PYEOF

# Phase 1: inference
echo ""
echo "===== Phase 1: DMD2 step_7000 inference ====="
cd "$REPO_MAIN/models/hunyuan3d21/hy3dshape"
export PYTHONPATH="$REPO_MAIN/models/hunyuan3d21/hy3dshape:$REPO_EVAL/pipeline:$REPO_EVAL/distill_methods:$PYTHONPATH"
CUDA_VISIBLE_DEVICES=0 "$PY" \
    "$REPO_EVAL/distill_methods/scripts/run_inference_distilled.py" \
    --config "$CFG" \
    --model-name dmd2_1step
echo "Phase 1 exit: $?"

# Phase 2: eval
echo ""
echo "===== Phase 2: eval all 4 model (cd_4step + teacher + flashvdm + dmd2_step7k) ====="
cd "$REPO_EVAL"
export PYTHONPATH="$REPO_EVAL/pipeline:$REPO_EVAL/distill_methods:$PYTHONPATH"
CUDA_VISIBLE_DEVICES=0 "$PY" \
    "$REPO_EVAL/pipeline/scripts/run_eval.py" \
    --config "$CFG" \
    --models dmd2_1step \
    --workers 4
echo "Phase 2 exit: $?"

# Summary
echo ""
echo "===== Summary ====="
echo "predictions: $(find $DMD2_RUN/predictions/dmd2_1step_step7000 -name 'mesh_raw.obj' 2>/dev/null | wc -l) / 105"
echo "metrics:"
ls -la $DMD2_RUN/metrics/ 2>/dev/null

echo "================================================================"
echo "M2 DMD2 step_7000 done: $(date -Iseconds)"
echo "================================================================"
