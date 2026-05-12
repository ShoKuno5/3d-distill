#!/usr/bin/env bash
# qzcli benchmark: measure DMD2 per-step time + GPU util for batch=4 vs batch=8
# on H100 80GB, to decide whether to (a) keep DSW-parity batch=4 / 15K steps
# or (b) use batch=8 / 7500 steps for faster wall clock.
#
# Outputs to gpfs:
#   .../qzcli_logs/bench_batch4_<host>_<ts>.log
#   .../qzcli_logs/bench_batch8_<host>_<ts>.log
#   .../qzcli_logs/bench_util_batch4_<host>_<ts>.csv  (gpu util every 500ms)
#   .../qzcli_logs/bench_util_batch8_<host>_<ts>.csv

set -euo pipefail

SK5=/inspire/qb-ilm/project/qproject-assement/zhangkaipeng-24043/sk5
LOGDIR="$SK5/scratch/qzcli_logs"
mkdir -p "$LOGDIR"
TS=$(date +%Y%m%d_%H%M%S)
HOST=$(hostname)
MAIN_LOG="$LOGDIR/bench_main_${HOST}_${TS}.log"
exec > >(tee -a "$MAIN_LOG") 2>&1
echo "[log: $MAIN_LOG]"

# Install runtime libs if missing
if ! ldconfig -p | grep -q "libGL.so.1"; then
    echo "[setup] apt install runtime libs ..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq 2>&1 | tail -3 || true
    apt-get install -y -qq libgl1 libsm6 libxrender1 libxfixes3 libxi6 libxkbcommon0 2>&1 | tail -5
fi

REPO="$SK5/repos/3d-gen-eval"
H="$REPO/models/hunyuan3d21"
cd "$H/hy3dshape"
export HF_HOME="$SK5/hf_cache"
export PYTHONPATH=".:$REPO/distill_methods/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

[ -f "$SK5/.env" ] && set -a && . "$SK5/.env" && set +a

echo
echo "================================================================"
echo "DMD2 batch-size benchmark on H100"
echo "host: $HOST"
echo "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "================================================================"

run_one_bench() {
    local BS=$1
    local NSTEPS=$2
    local OUT_ROOT="$SK5/scratch/qzcli_bench/bench_b${BS}_${TS}"
    local CFG_PATH="$OUT_ROOT/config.yaml"
    local UTIL_CSV="$LOGDIR/bench_util_batch${BS}_${HOST}_${TS}.csv"

    mkdir -p "$OUT_ROOT"

    echo
    echo "================================================================"
    echo "RUN batch=$BS  steps=$NSTEPS"
    echo "================================================================"

    # Patch smoke config for this batch
    "$H/.venv/bin/python" -c "
import yaml
cfg = yaml.safe_load(open('$REPO/distill_methods/configs/config_525_hssd.yaml'))
cfg['training']['total_steps'] = $NSTEPS
cfg['training']['batch_size'] = $BS
cfg['training']['log_interval'] = 1
cfg['training']['save_interval'] = 999999  # avoid save during bench
cfg['training']['methods']['dmd2']['replay_buffer_size'] = 32
cfg['training']['methods']['dmd2']['replay_warmup'] = 4
cfg['output_root'] = '$OUT_ROOT'
yaml.safe_dump(cfg, open('$CFG_PATH', 'w'), default_flow_style=False, sort_keys=False)
"

    # Start GPU util monitor in background
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used --format=csv,noheader -lms 500 > "$UTIL_CSV" &
    local MONPID=$!

    # Run training
    set +e
    "$H/.venv/bin/python" "$REPO/distill_methods/src/train.py" \
        --config "$CFG_PATH" --method dmd2 2>&1 | tee "$LOGDIR/bench_batch${BS}_${HOST}_${TS}.log"
    local STATUS=$?
    set -e

    kill $MONPID 2>/dev/null || true
    wait $MONPID 2>/dev/null || true

    if [ "$STATUS" -ne 0 ]; then
        echo "  FAILED batch=$BS (status $STATUS) — likely OOM, see log"
        return $STATUS
    fi

    # Compute per-step timing from log (timestamps between consecutive step= lines)
    "$H/.venv/bin/python" -c "
import re
import statistics
import sys
log_path = '$LOGDIR/bench_batch${BS}_${HOST}_${TS}.log'
util_path = '$UTIL_CSV'

# Per-step times from log timestamps. The base_distiller logs lines like
#   '2026-... [INFO] base_distiller: step=N loss=...'
step_lines = []
ts_re = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) \[INFO\] base_distiller: step=(\d+)')
import datetime
for ln in open(log_path):
    m = ts_re.match(ln)
    if m:
        t = datetime.datetime.strptime(m.group(1)+'.'+m.group(2)+'000', '%Y-%m-%d %H:%M:%S.%f')
        step_lines.append((int(m.group(3)), t))
if len(step_lines) < 3:
    print(f'  [bench] not enough step lines: {len(step_lines)}')
    sys.exit(0)
# Skip first 5 steps (warmup), use remaining
skip = 5
deltas = []
for i in range(skip+1, len(step_lines)):
    dt = (step_lines[i][1] - step_lines[i-1][1]).total_seconds()
    deltas.append(dt)
if deltas:
    print(f'  [bench batch=$BS] per-step mean={statistics.mean(deltas):.3f}s  median={statistics.median(deltas):.3f}s  N={len(deltas)} (warmup {skip} skipped)')

# GPU util from monitor csv
util_vals = []
mem_vals = []
for ln in open(util_path):
    parts = [p.strip() for p in ln.split(',')]
    if len(parts) < 4: continue
    try:
        u = int(parts[1].rstrip(' %'))
        m = int(parts[2].rstrip(' %'))
        util_vals.append(u)
        mem_vals.append(m)
    except ValueError: continue
if util_vals:
    print(f'  [bench batch=$BS] GPU-Util  mean={statistics.mean(util_vals):.1f}%  median={statistics.median(util_vals)}%  p90={sorted(util_vals)[int(len(util_vals)*0.9)]}%')
    print(f'  [bench batch=$BS] Mem-Util  mean={statistics.mean(mem_vals):.1f}%  median={statistics.median(mem_vals)}%')
"
}

# Run batch=4 first (DSW parity)
run_one_bench 4 30

# Then batch=8 (may OOM)
run_one_bench 8 30 || echo "  (batch=8 failed, likely OOM)"

echo
echo "================================================================"
echo "Benchmark complete: $(date -Iseconds)"
echo "  main log: $MAIN_LOG"
echo "  util CSVs: $LOGDIR/bench_util_batch*_${HOST}_${TS}.csv"
echo "================================================================"
