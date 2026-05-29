#!/usr/bin/env bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:10:00
#$ -N quick-install
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/quick_install.qsub.log

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "quick_install"
tsubame_setup_env
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/gs/fs/tga-koike-shanda2/sk/cache/uv

VENV_PY=/gs/fs/tga-koike-shanda2/sk/3d-distill/.venv/bin/python

echo "install easydict + pyyaml (extra) in 3d-distill venv"
uv pip install --python $VENV_PY easydict pyyaml munch 2>&1 | tail -5

echo ""
$VENV_PY -c "import easydict, yaml, munch; print('easydict:', easydict.__version__, 'yaml:', yaml.__version__, 'munch:', munch.__version__)"
