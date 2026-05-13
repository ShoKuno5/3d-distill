#!/usr/bin/env bash
# Launch M1 4-method 拡張: PD / CD / DMD1 を H200-3-2 group に 1 GPU ずつ投入
# Run on sk-train (cwd = sk5/):
#   bash /tmp/launch_m1_4method.sh

set -e

QZ=qzcli_venv/bin/qzcli
WS=ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6
PROJECT=project-52a1ffef-b01e-44ac-b727-031379e0c603   # Q项目-评估前沿探索
GROUP=lcg-95e38be4-4842-4155-af13-4325aa744bca         # H200-3号机房-2 (15 GPU free)
SPEC=4dd0e854-e2a4-4253-95e6-64c13f0b5117              # 1x H200
SH=/inspire/qb-ilm/project/qproject-assement/zhangkaipeng-24043/sk5/repos/3d-distill/distill_methods/qzcli_jobs

for METHOD in pd cd dmd1; do
    echo "================================================================"
    echo "Submitting m1${METHOD}..."
    echo "================================================================"
    $QZ create \
        -n "m1${METHOD}" \
        --project "$PROJECT" \
        -c "bash ${SH}/m1_${METHOD}.sh" \
        -w "$WS" \
        -g "$GROUP" \
        --instances 1 \
        --spec "$SPEC"
    echo
done

echo "All 3 jobs submitted. List:"
$QZ ls -c -r -w "$WS"
