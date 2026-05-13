#!/usr/bin/env bash
# Submit all 4 M1 methods to TSUBAME at once.
#
# Each job grabs 1 H100 node_f (4 GPU). With ar_id 6925 holding 4 nodes,
# all 4 methods can run in parallel under one reservation.
#
# Run on TSUBAME login1 (or any shell with qsub):
#   bash launch_m1_4method.sh <ar_id>
# Default ar_id: 6925.

set -euo pipefail

AR_ID="${1:-6925}"
GROUP="tga-koike-shanda"
SH="$(cd "$(dirname "$0")" && pwd)"

mkdir -p /gs/fs/tga-koike-shanda/kuno/scratch/tsubame_logs

for METHOD in pd cd dmd1 dmd2; do
    echo "================================================================"
    echo "Submitting m1_${METHOD}.sh (ar=$AR_ID, group=$GROUP)"
    echo "================================================================"
    qsub -ar "$AR_ID" -g "$GROUP" "$SH/m1_${METHOD}.sh"
done

echo
echo "All 4 jobs submitted. List with: qstat"
qstat
