# qzcli batch jobs

このディレクトリは qzcli ジョブ投入用の bash script を置く場所。各 script は H100 ノード上で動作することを前提とする。

## 前提

- sk-train の gpfs (`/inspire/.../sk5/`) が qzcli H100 ノードからも読める前提
- `models/hunyuan3d21/.venv/` の Python 3.10 venv をそのまま使う
- 必須環境変数: `WANDB_API_KEY` (`sk5/.env` から source される)
- HF cache: `sk5/hf_cache/` (Hunyuan3D-2.1 重み 14GB)

## ジョブ一覧

| script | 用途 | GPU | wall clock | 注 |
|---|---|---:|---|---|
| `smoke.sh` | env 動作確認 (gpfs / CUDA / pipeline / 5-step train) | 1 H100 | ~5-10 min | これが pass する前に本番は投げない |
| `m1_dmd2.sh` | M1 本番: DMD2 on HSSD 420 → Toys4k 105 eval | 1-2 H100 | ~12-25h | 仮説テスト本体 |
| `r1_dmd2.sh` | R1 reproduction: DMD2 on Toys4k 420 → Toys4k 105 | 1-2 H100 | ~12-25h | DSW env diff 計測の control |

## 推奨投入順序

```bash
# 0. cookie auth (1 回)
~/.qzcli_venv/bin/qzcli cookie -w ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6
~/.qzcli_venv/bin/qzcli avail -w ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6  # H100/H200 空き確認

# 1. SMOKE — env 動作確認
~/.qzcli_venv/bin/qzcli create \
  -n smoke-distill3d \
  -c "bash /inspire/qb-ilm/project/qproject-assement/zhangkaipeng-24043/sk5/repos/3d-distill/distill_methods/qzcli_jobs/smoke.sh" \
  -w ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6 \
  -g lcg-79b2ad0e-a375-43f3-a0b1-b4ce79710fd7 \
  --instances 1
# (compute group は cuda12.8 H100、1x H100 spec は qzcli avail で確認した値に置き換え)

# 2. SMOKE 結果見て OK なら、本番 2 並列投入
~/.qzcli_venv/bin/qzcli create \
  -n m1-dmd2 \
  -c "bash /inspire/.../sk5/repos/3d-distill/distill_methods/qzcli_jobs/m1_dmd2.sh" \
  -w ws-9dcc0e1f-... -g lcg-<H100> --instances 1 --spec <2xH100 spec>

~/.qzcli_venv/bin/qzcli create \
  -n r1-dmd2 \
  -c "bash /inspire/.../sk5/repos/3d-distill/distill_methods/qzcli_jobs/r1_dmd2.sh" \
  -w ws-9dcc0e1f-... -g lcg-<H100> --instances 1 --spec <2xH100 spec>

# 3. 監視
~/.qzcli_venv/bin/qzcli ls -c -r           # 走行中ジョブ一覧
~/.qzcli_venv/bin/qzcli watch <job-id>    # ライブログ
~/.qzcli_venv/bin/qzcli status <job-id>   # 状態
```

## compute group / spec ID (キャッシュ済み)

cache: `qzcli res -w ws-9dcc0e1f-...` で確認可。代表的なもの:

| group | GPU | lcg ID |
|---|---|---|
| cuda12.8版本H100 | H100 80GB | lcg-79b2ad0e-a375-43f3-a0b1-b4ce79710fd7 |
| cuda12.9版本H100 | H100 80GB | lcg-bc36d6bf-43e1-437a-b976-bc4d63dadf57 |
| H200-1号机房 | H200 141GB | lcg-df089db8-817a-4aa8-a164-eb1a32948564 |
| H200-2号机房 | H200 141GB | lcg-303ac8c6-aa19-4284-af03-2296592326e5 |
| H200-3号机房 | H200 141GB | lcg-a91ad10b-415d-4abd-8170-828a2feae5d2 |

spec ID (2 GPU 構成例、H200):
- 2x H200 + 30核CPU + 400GB内存: `7166bd2e-6cbe-4bd9-be38-762d11003e7f`
- 4x H200 + 60核CPU + 800GB内存: `45ab2351-fc8a-4d50-a30b-b39a5306c906`

H100 spec ID は `qzcli avail` で要確認。

## 失敗時のヒント

- WANDB key 未設定 → `sk5/.env` を H100 ノードに伝える経路を `qzcli create` の `-e WANDB_API_KEY=...` で渡す
- gpfs 不可視 → ジョブ内で github clone + HF DL + 全 re-encode する fallback (時間倍増、最終手段)
- OOM → batch_size=4 を 2 や 1 に下げる
- 途中 fail → `--resume <ckpt>` で続行 (`run.sh` 互換)
