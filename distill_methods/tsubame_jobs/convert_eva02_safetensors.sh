#!/usr/bin/env bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:20:00
#$ -N eva02-convert
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/convert_eva02.qsub.log
#
# Convert EVA02-E-14-plus open_clip_model.safetensors → open_clip_pytorch_model.bin
# Places at FlashVDM expected path under external/uni3d/uni3d/

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "convert_eva02"
tsubame_setup_env

echo "================================================================"
echo "EVA02 safetensors→bin convert: $(date -Iseconds), host=$(hostname)"
echo "================================================================"

SRC=/gs/fs/tga-koike-shanda2/sk/hf_cache/open_clip/models--timm--eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k/snapshots/0162f50c2742be3c90b3a81e4e8066fc5a90e125/open_clip_model.safetensors
DST=/gs/fs/tga-koike-shanda2/sk/3d-distill-eval/pipeline/src/evaluation/external/uni3d/uni3d/open_clip_pytorch_model.bin

ls -la "$SRC"
mkdir -p "$(dirname $DST)"

"$PY" <<PYEOF
import safetensors.torch as st
import torch, os
sd = st.load_file("$SRC")
print(f"loaded {len(sd)} tensors from safetensors")
torch.save(sd, "$DST")
print(f"wrote {os.path.getsize('$DST')} bytes to {'$DST'}")
PYEOF

ls -la "$DST"
echo "================================================================"
echo "convert done: $(date -Iseconds)"
echo "================================================================"
