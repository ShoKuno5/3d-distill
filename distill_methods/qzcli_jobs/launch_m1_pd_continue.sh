#!/usr/bin/env bash
# M1 PD 継続 (stage 1, 2): 既存 m1_pd_20260512_0532 出力を resume
# Run on sk-train (cwd = sk5/) after `qzcli login`:
#   bash distill_methods/qzcli_jobs/launch_m1_pd_continue.sh

set -e

QZ=qzcli_venv/bin/qzcli
WS=ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6
PROJECT=project-52a1ffef-b01e-44ac-b727-031379e0c603
GROUP=lcg-95e38be4-4842-4155-af13-4325aa744bca
SPEC=4dd0e854-e2a4-4253-95e6-64c13f0b5117

EXISTING_OUTPUT=/inspire/qb-ilm/project/qproject-assement/zhangkaipeng-24043/sk5/scratch/distill_methods/m1_pd_20260512_0532
SCRIPT=/inspire/qb-ilm/project/qproject-assement/zhangkaipeng-24043/sk5/repos/3d-distill/distill_methods/qzcli_jobs/m1_pd.sh

CMD="M1_PD_OUTPUT_ROOT=$EXISTING_OUTPUT M1_PD_RUN_NAME=m1_pd_20260512_0532 bash $SCRIPT"

$QZ create \
    -n m1pdcont \
    --project "$PROJECT" \
    -c "$CMD" \
    -w "$WS" \
    -g "$GROUP" \
    --instances 1 \
    --spec "$SPEC"
