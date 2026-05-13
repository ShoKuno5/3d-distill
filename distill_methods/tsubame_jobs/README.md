# TSUBAME 4.0 batch jobs

H100 96GB ノード (`node_f`, 4 GPU) 上で M1 distillation 実験を回すための qsub script 群。
qzcli (SII) 上の `qzcli_jobs/` を TSUBAME UGE/SGE スケジューラ向けに移植したもの。

## 前提

- アカウント: Erwin の `uh05887`、グループ `tga-koike-shanda`
- ストレージレイアウト:
  ```
  /gs/fs/tga-koike-shanda/kuno/        (SSD 3T, code + venv + ckpt)
    ├── 3d-distill/                    (this repo)
    │   └── .venv/                     (uv venv, Python 3.10)
    ├── hf_cache/                      (~14 GB, tencent/Hunyuan3D-2.1)
    ├── data/                          (pre-encoded latents + DMD1 pairs + toys4k)
    │   ├── 525_hssd/training_data/    (~6 GB, 420 npz)
    │   ├── 525_hssd/dmd1_pairs/       (20K pair npz)
    │   └── toys4k/renders/            (105 eval renders)
    └── scratch/                       (per-run outputs + logs)
  ```
- `$REPO/.env` に `WANDB_API_KEY=...` を置く (TSUBAME は egress OK なので WANDB online)
- 予約: `ar_id 6925` (5/11–5/18, 4 nodes, state `r`)

## ジョブ一覧

| script | 用途 | GPU | wall clock |
|---|---|---:|---|
| `smoke.sh` | env 動作確認 (filesystem / CUDA / pipeline / 5-step DMD2) | 1 GPU (node_q) | ~10 min |
| `m1_pd.sh` | M1 PD (Progressive Distillation, 3 stage × 5K step) | 4 H100 (node_f) | ~6-10h |
| `m1_cd.sh` | M1 CD (Consistency Distillation, 15K step) | 4 H100 (node_f) | ~6-10h |
| `m1_dmd1.sh` | M1 DMD1 (pair gen + 15K step) | 4 H100 (node_f) | ~12-20h |
| `m1_dmd2.sh` | M1 DMD2 (15K step + GAN) | 4 H100 (node_f) | ~12-20h |
| `launch_m1_4method.sh` | 4 method を 4 ノード並列で投入 | 16 H100 | 1 reservation で完結 |

`_common.sh` は共通 setup (module load、env、config rewrite ヘルパ)。

## 推奨投入順序

```bash
# 0. interactive で smoke
qrsh -l node_q=1 -ar 6925 -g tga-koike-shanda -l h_rt=1:00:00
bash /gs/fs/tga-koike-shanda/kuno/3d-distill/distill_methods/tsubame_jobs/smoke.sh

# 1. smoke OK なら 4 method 一括投入
bash /gs/fs/tga-koike-shanda/kuno/3d-distill/distill_methods/tsubame_jobs/launch_m1_4method.sh 6925

# 2. 監視
qstat                      # ジョブ一覧
qstat -j <job_id>          # 詳細
tail -F /gs/fs/tga-koike-shanda/kuno/scratch/tsubame_logs/m1_*.log

# 3. 削除
qdel <job_id>
```

## qzcli との差分

| | qzcli | TSUBAME |
|---|---|---|
| Scheduler | qzcli create | qsub |
| GPU 単位 | 1 spec=1 H200 | node_f=4 H100 |
| GL libs install | `apt-get` 可 (root) | 不要想定 (なければ user-space conda or module で対応) |
| WANDB | offline 必須 (egress 不可) | online OK |
| .env path | `$SK5/.env` | `$REPO/.env` |
| venv | sk-train の `models/hunyuan3d21/.venv` を gpfs 共有 | `$REPO/.venv` を uv で新規構築 |
| HF cache | `$SK5/hf_cache` symlink to `/root/.cache/huggingface` | `HF_HOME=$TSUBAME_ROOT/hf_cache` 直接設定 |

## 失敗時のヒント

- `WANDB_API_KEY must be set` → `$REPO/.env` に書く
- `libGL.so.1 not found` → `module load` で対応するか、user-space で conda 経由 install
- OOM → batch_size を 4→2→1 に下げる、または bf16 を確認
- `module: command not found` → interactive node では `.bashrc` を再 source、`/etc/profile.d/modules.sh` を source
- ar が変わったら ssh config の `tsubame-reserved` ホスト名も更新 (memory: [[reference_tsubame]])
