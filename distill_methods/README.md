# Distillation Methods Comparison (v2)

Branch: `exp/distill-methods-v2` | Hunyuan3D-2.1 | Toys4k (420 train / 105 test)

## Purpose

4つの拡散蒸留手法を比較し、ODE軌道の圧縮が構造形成の階層をどう変えるかを調べる。

## Hypothesis

- PD: 段階的圧縮により構造形成の順序を保存する
- CD: consistency制約がglobalなため時間的階層が崩れうる
- DMD1/DMD2: 1-step生成で最も攻撃的な圧縮、構造の一貫性を犠牲にしうる
- FlashVDM (公式turbo, 5-step): 産業ベースライン

## Methods

| Method | Paper | Inference Steps | Mechanism |
|--------|-------|----------------|-----------|
| PD | Salimans & Ho 2022 | 6 | teacher 2-step → student 1-step を3段階 (50→25→12→6) |
| CD | Song et al. 2023 | 4 | 任意のODE点を同一終点にマッピング |
| DMD1 | Yin et al. 2024 | 1 | KL divergence + 事前計算regression pairs |
| DMD2 | Yin et al. 2024b | 1 | KL divergence + GAN discriminator |

Baselines: Teacher 50-step, FlashVDM 5-step

## Pipeline

```
Data Prep (1回)                  Training + Eval (run.sh)
==============                   ========================

Toys4k meshes                    Phase 1: Training (4 GPU並列)
    │                                PD  (GPU0, 3 stages)
    ├─ create_manifests.py           CD  (GPU1)
    │   → manifest_{train,test}.csv  DMD1 (GPU2)
    │                                DMD2 (GPU3)
    ├─ render_batch.py                   │
    │   → input images (512x512)     Phase 2: Inference
    │                                    teacher 50-step, FlashVDM,
    ├─ prepare_training_data.py          PD 6-step, CD 4-step,
    │   → training_data/*.npz            DMD1 1-step, DMD2 1-step
    │     (VAE latent + image cond)      │
    │                                Phase 3: Evaluation
    └─ generate_dmd1_pairs.py            CD, F-score, Hausdorff, FD
        → dmd1_pairs/*.npz              │
          (noise, teacher output)    Phase 4: Report
          20,000 pairs
```

Data prep の出力は `results/distill_methods/` 直下に共有データとして置く。
Training 以降の出力は `results/distill_methods/runs/<RUN_ID>/` に分離される。

## Changes from v1 (`exp/distill-methods-comparison`)

**Bug fixes** (1dd987d): PD x-space→v-space loss, CD student/EMA逆転, DMD1 sum→MSE, DMD2 feature space mismatch

**Infrastructure**: run単位の出力分離 (`runs/<timestamp>/`), data prep分離, 4手法GPU並列

## Usage

```bash
# Data prep (個別スクリプト, 初回のみ)
scripts/create_manifests.py --target-samples 525 --seed 42
scripts/render_batch.py --num-gpus 4
scripts/prepare_training_data.py --config config.yaml --shard 0 --num-shards 4  # x4 GPU
scripts/generate_dmd1_pairs.py --config config.yaml --shard 0 --num-shards 4    # x4 GPU

# Training + eval
bash distill_methods/run.sh
bash distill_methods/run.sh --skip-training    # eval only
bash distill_methods/run.sh --run-id YYYYMMDD_HHMM  # resume
```

## Run Log

| Run ID | Status | Notes |
|--------|--------|-------|
