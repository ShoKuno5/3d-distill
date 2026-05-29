#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=2:00:00
#$ -N metrics-2of6
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/eval_metrics_2of6.qsub.log

MODELS="mdt_dist trellis2" WORKERS=48 /gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/eval_metrics_1050.sh
