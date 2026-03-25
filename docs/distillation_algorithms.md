# 蒸留アルゴリズム解説

Hunyuan3D-2.1 (Flow Matching DiT, 3.3B params, 50 steps) に対する4つの蒸留手法の解説資料。

## 目次

1. [前提知識: Flow Matching](#前提知識-flow-matching)
2. [手法の分類](#手法の分類)
3. [Progressive Distillation (PD)](#1-progressive-distillation-pd)
4. [Consistency Distillation (CD)](#2-consistency-distillation-cd)
5. [Distribution Matching Distillation (DMD1)](#3-distribution-matching-distillation-dmd1)
6. [DMD2](#4-dmd2)
7. [Hunyuan3D-2.1 への適応メモ](#hunyuan3d-21-への適応メモ)
8. [参考文献](#参考文献)

---

## 前提知識: Flow Matching

Flow Matching は ODE ベースの生成モデルフレームワーク。ノイズ分布からデータ分布への確率的輸送を学習する。

### Hunyuan3D-2.1 の規約 (ICPlan)

```
パス:     x_t = t · x_data + (1-t) · noise
速度場:   v = x_data - noise
t=0: ノイズ,  t=1: データ
```

- モデルは速度場 `v(x_t, t)` を予測する
- 入力: `model(x_t, t, contexts={'main': image_cond})`
- Latent shape: `[B, N, C] = [B, 4096, 4]` (VAE latent)
- 生成時は `t=0` (ノイズ) から `t=1` (データ) に向かって ODE を解く

### Euler ステップ

```
x_{t+h} = x_t + h · v(x_t, t)
```

ステップ幅 `h = 1/N` (N がステップ数)。ステップ数が多いほど精度が高いが遅い。
蒸留の目標は、少ないステップで teacher の品質を再現すること。

---

## 手法の分類

### Trajectory ベース (PD, CD)

Teacher の ODE 軌道を直接模倣する。

```
[直感] teacher が 2 歩かけて進む道のりを、student に 1 歩で歩かせる
```

- **利点**: 安定しやすい、実装がシンプル
- **欠点**: ODE 軌道への依存が強い、1-step 性能に限界がある場合も

### Distribution ベース (DMD1, DMD2)

Student の出力分布全体を teacher の出力分布に近づける。

```
[直感] 個々の軌道ではなく、最終的な出力の「分布」が一致するように訓練
```

- **利点**: 1-step 生成に強い、軌道に縛られない
- **欠点**: 実装が複雑、学習不安定になりやすい

```
            ┌──────────────────────────────────────────────────┐
            │         Trajectory ベース                        │
            │                                                  │
            │   PD: teacher 2-step → student 1-step (反復)     │
            │   CD: ODE 軌道上の自己整合性                     │
            │                                                  │
            ├──────────────────────────────────────────────────┤
            │         Distribution ベース                      │
            │                                                  │
            │   DMD1: score matching + 回帰                    │
            │   DMD2: score matching + GAN (回帰不要)          │
            │                                                  │
            └──────────────────────────────────────────────────┘
```

---

## 1. Progressive Distillation (PD)

**論文**: Salimans & Ho, "Progressive Distillation for Fast Sampling of Diffusion Models" (2022)

### 直感

Teacher が2ステップかけて到達する点を、student が1ステップで到達するように訓練する。
これを繰り返し、50→25→12→6 とステップ数を半減していく。

### 数式

```
x_mid = x_t + h · v_teacher(x_t, t)                 # teacher 1st step
x_tgt = x_mid + h · v_teacher(x_mid, t + h)         # teacher 2nd step
x_pred = x_t + 2h · v_student(x_t, t)               # student 1-step

Loss = MSE(x_pred, x_tgt)
```

### 学習パイプライン

```
┌─────────────────────────────────────────────────────────┐
│ 段階 1: 50-step teacher → 25-step student               │
│                                                         │
│   for each training step:                               │
│     1. サンプル: x_data, noise → x_t (t ~ uniform)      │
│     2. Teacher 2-step: x_t → x_mid → x_tgt             │
│     3. Student 1-step: x_t → x_pred                    │
│     4. Loss = MSE(x_pred, x_tgt)                       │
│     5. Student を更新 (LoRA)                            │
│                                                         │
│   → student は 25 step で生成可能に                     │
│                                                         │
├─────────────────────────────────────────────────────────┤
│ 段階 2: 25-step teacher (= 段階1 student) → 12-step    │
│   LoRA merge → 新しい teacher, 新しい LoRA init         │
│   同じ手順で訓練                                        │
│                                                         │
├─────────────────────────────────────────────────────────┤
│ 段階 3: 12-step → 6-step                               │
│ 段階 4: (optional) 6-step → 3-step                     │
└─────────────────────────────────────────────────────────┘
```

### ポイント

- 各段階でステップ数が半減 → 4段階で 50→6 step
- 各段階は独立: 前段階の student が次段階の teacher になる
- LoRA merge + reinit で段階を切り替え
- 最もシンプルで安定 → パイプライン検証に最適

---

## 2. Consistency Distillation (CD)

**論文**: Song et al., "Consistency Models" (2023)

### 直感

ODE 軌道上の任意の点から、同じ終点 (t=1, データ) を予測できるように訓練する。
つまり「整合性関数 f(x_t, t) = x_data の推定値」が軌道上のどの点でも一致するようにする。

### 数式

```
# 整合性関数の定義
f(x, t) = x + (1-t) · v_student(x, t)
# t=1 のとき f(x_1, 1) = x_1 (boundary condition)
# t<1 のとき f(x_t, t) ≈ x_data の推定

# Teacher 1-step で隣接点を取得
x_{t-h} = x_t - h · v_teacher(x_t, t)

# 整合性損失: 同じ軌道上の 2 点が同じ終点を予測すべき
Loss = MSE( f_student(x_t, t),  f_EMA(x_{t-h}, t-h) )
```

ここで `f_EMA` は student の EMA (指数移動平均) コピー。

### 学習パイプライン

```
┌─────────────────────────────────────────────────────────┐
│ Consistency Distillation                                │
│                                                         │
│   for each training step:                               │
│     1. サンプル: x_data, noise → x_t (t ~ uniform)      │
│     2. Teacher 1-step: x_t → x_{t-h}  (隣接点)         │
│     3. Student 予測:   f(x_t, t)                        │
│     4. EMA 予測:       f_EMA(x_{t-h}, t-h)             │
│     5. Loss = MSE(student予測, EMA予測)                 │
│     6. Student を更新, EMA を更新                       │
│                                                         │
│   推論時:                                               │
│     x_data_hat = f(x_noise, t=0)   ← 1-step 生成可能   │
│     (multi-step も可: 途中で denoise → re-noise)        │
└─────────────────────────────────────────────────────────┘
```

### ポイント

- PD と異なり段階分けが不要 → 一発で訓練
- EMA が重要: teacher の ODE に追従しつつ、自分自身の予測を安定化
- FlashVDM (Hunyuan3D-2.1 公式蒸留) はこの手法を核に使用
- 1-step 生成と multi-step 生成の両方に対応

---

## 3. Distribution Matching Distillation (DMD1)

**論文**: Yin et al., "One-step Diffusion with Distribution Matching Distillation" (2023)

### 直感

Student の出力分布と teacher の出力分布の KL ダイバージェンスを最小化する。
KL 勾配の計算には「fake score network」(student 出力分布のスコアを学習するネットワーク) を使う。
加えて、回帰損失 (teacher 出力との直接ペアリング) で安定化する。

### 数式

```
# === 事前準備: 回帰データセット ===
# noise_i → teacher 50-step ODE → x_teacher_i
# 20k 対の (noise_i, x_teacher_i) を事前生成

# === Fake Score Network の更新 ===
x_fake = student(noise)                    # student の 1-step 出力 (no grad)
t_probe = uniform(0, 1)
eps = randn_like(x_fake)
x_noised = t_probe · x_fake + (1-t_probe) · eps
# denoising score matching on student outputs
Loss_fake = MSE( fake_score(x_noised, t_probe), -eps / (1-t_probe) )

# === Student の更新 ===
x_gen = student(noise)                     # 1-step 出力 (with grad)

# 分布マッチング: KL 勾配 ≈ fake_score - teacher_score
# (teacher_score は teacher モデルで近似)
grad_kl = fake_score(x_gen, t) - teacher(x_gen, t)
L_distill = (grad_kl.detach() · x_gen).sum()

# 回帰: 事前計算した teacher 出力とのペアリング
L_regress = MSE(x_gen, x_teacher_paired)

Loss = L_distill + λ_reg · L_regress
```

### 学習パイプライン

```
┌─────────────────────────────────────────────────────────┐
│ 事前準備: 回帰データセット生成 (~6.4h)                  │
│   for i in 1..20000:                                    │
│     noise_i ~ N(0, I)                                   │
│     x_teacher_i = teacher_50step(noise_i)               │
│     save (noise_i, x_teacher_i) → NPZ                  │
├─────────────────────────────────────────────────────────┤
│ 学習ループ (~6.7h)                                      │
│                                                         │
│   3 つのネットワーク:                                    │
│     - Teacher (frozen, 元の DiT)                        │
│     - Student (LoRA)                                    │
│     - Fake score (LoRA, 別パラメータ)                   │
│                                                         │
│   for each step:                                        │
│     1. Fake score 更新 (student 出力分布を学習)         │
│     2. Student 更新 (KL + 回帰)                        │
│                                                         │
│     ┌──────────────┐    score     ┌──────────────┐      │
│     │  Fake Score   │◄───────────│  Student 出力  │     │
│     │  (student分布  │            │  (1-step)     │      │
│     │   のスコア)    │            └───────┬───────┘      │
│     └──────┬───────┘                     │              │
│            │                             │              │
│            │  KL gradient                │ regression   │
│            ▼                             ▼              │
│     ┌──────────────┐            ┌──────────────┐        │
│     │   Teacher     │            │ 回帰データ    │       │
│     │  (frozen)     │            │ (事前計算)    │       │
│     └──────────────┘            └──────────────┘        │
└─────────────────────────────────────────────────────────┘
```

### ポイント

- 3 モデル同時にメモリに載せる必要あり → LoRA で ~58GB/GPU に抑制
- 回帰損失が mode collapse を防ぐ安定化剤として機能
- 事前のデータ生成が高コスト (~6.4h)
- 1-step 生成に特化した設計

---

## 4. DMD2

**論文**: Yin et al., "Improved Distribution Matching Distillation" (2024)

### 直感

DMD1 の改良版。2つの大きな変更:
1. **回帰データセット不要**: GAN discriminator で置き換え → 事前計算コスト削除
2. **GAN 損失追加**: real/fake を判別する discriminator が student の出力品質を直接評価

### 数式

```
# === 1. Fake Score 更新 (DMD1 と同じ) ===
x_fake = student(noise).detach()
Loss_fake = score_matching_loss(fake_score, x_fake, t_probe)

# === 2. Discriminator 更新 (TTUR: 5回/student 1回) ===
x_real = replay_buffer.sample()            # teacher 出力のバッファ
x_gen = student(noise).detach()
Loss_D = hinge_loss(D(x_real), D(x_gen))

# === 3. Student 更新 ===
x_gen = student(noise)
L_distill = score_diff_loss(fake_score, teacher, x_gen, t)
L_gan = -mean(D(x_gen))                   # generator loss

Loss_G = L_distill + λ_gan · L_gan
```

### 学習パイプライン

```
┌─────────────────────────────────────────────────────────┐
│ DMD2 学習ループ (~11h)                                  │
│                                                         │
│   4 つのコンポーネント:                                  │
│     - Teacher (frozen)                                  │
│     - Student (LoRA)                                    │
│     - Fake score (LoRA)                                 │
│     - Discriminator (小さい MLP, ~5M params)            │
│                                                         │
│   Replay Buffer: teacher 出力を逐次キャッシュ           │
│     - warmup: 128 個を事前生成 (~2.5 min)               │
│     - 以降は学習中に teacher を走らせて追加              │
│                                                         │
│   for each step:                                        │
│     1. Fake score 更新                                  │
│     2. Discriminator 更新 × 5 (TTUR)                    │
│        - replay buffer から real / student から fake     │
│     3. Student 更新 (KL + GAN)                          │
│     4. (定期的に) teacher ODE → replay buffer に追加     │
│                                                         │
│     ┌──────────────┐    score     ┌──────────────┐      │
│     │  Fake Score   │◄───────────│  Student 出力  │     │
│     └──────┬───────┘            └───────┬───────┘      │
│            │ KL grad                    │ GAN loss      │
│            ▼                            ▼              │
│     ┌──────────────┐            ┌──────────────┐        │
│     │   Teacher     │            │Discriminator │        │
│     │  (frozen)     │            │  (MLP)       │       │
│     └──────┬───────┘            └──────┬───────┘        │
│            │                           │                │
│            ▼                           │                │
│     ┌──────────────┐                   │ real samples   │
│     │ Replay Buffer │◄─────────────────┘                │
│     └──────────────┘                                    │
└─────────────────────────────────────────────────────────┘
```

### DMD1 vs DMD2 比較

| 項目 | DMD1 | DMD2 |
|------|------|------|
| 回帰データセット | 必要 (事前生成 ~6.4h) | 不要 |
| GAN discriminator | なし | あり |
| Replay buffer | なし | あり |
| 安定化の仕組み | 回帰損失 | GAN + replay buffer |
| 学習コスト | ~13.5h (データ生成含む) | ~11h |
| 実装複雑度 | 中 | 高 |

---

## Hunyuan3D-2.1 への適応メモ

### 共通事項

- **Latent 空間**: VAE でエンコードされた `[B, 4096, 4]` のトークン列上で蒸留を行う
- **条件付け**: 画像条件 `image_cond` を `contexts={'main': image_cond}` で渡す
- **LoRA**: rank-64, DiT の attention 層に適用。フルモデル学習はメモリ・速度の制約で非現実的
- **精度**: bf16 (Ampere 以降の GPU で安定)
- **CFG**: teacher は `guidance_scale=7.5` を使用。蒸留時は student に CFG を distill する (teacher の CFG 出力を target とする)

### PD 固有

- Hunyuan3D-2.1 の `FlowMatchEulerDiscreteScheduler` のタイムステップスケジュールをそのまま利用
- 段階間で LoRA merge → reinit が必要 → `merge_and_unload()` + 新 LoRA 挿入

### CD 固有

- 既存の `ConsistencyFlowMatchEulerDiscreteScheduler` を参考に推論スケジュールを構築
- `LitEma` クラスが `flow_matching_sit.py` に存在 → 再利用

### DMD1 固有

- Fake score network は teacher と同じアーキテクチャ (DiT) + 別 LoRA
- 3つの LoRA (student, fake_score, teacher=frozen) が同一 DiT 上に共存
  → 実装上は LoRA adapter の切り替えで対応

### DMD2 固有

- Discriminator: DiT block 16 の中間特徴量 → mean-pool → MLP
  → DiT のフォワード途中から特徴を抜く hook が必要
- Replay buffer: `[1024, 4096, 4]` ≈ 67MB (bf16) → メモリ上で管理可能
- TTUR (Two Time-scale Update Rule): D の学習率を G の 5 倍に

---

## 参考文献

| 略称 | タイトル | リンク |
|------|---------|--------|
| PD | Progressive Distillation for Fast Sampling of Diffusion Models | https://arxiv.org/abs/2202.00512 |
| CD | Consistency Models | https://arxiv.org/abs/2303.01469 |
| DMD1 | One-step Diffusion with Distribution Matching Distillation | https://arxiv.org/abs/2311.18828 |
| DMD2 | Improved Distribution Matching Distillation | https://arxiv.org/abs/2405.14867 |
| FlashVDM | FlashVDM: Fast and Efficient 3D Generation | https://arxiv.org/abs/2503.16302 |
| Flow Matching | Flow Matching for Generative Modeling | https://arxiv.org/abs/2210.02747 |
