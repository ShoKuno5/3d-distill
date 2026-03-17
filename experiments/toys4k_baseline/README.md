# toys4k_baseline

## 目的
4モデル（trellis, trellis2, hunyuan3d, hunyuan3d21）のベースライン比較。
Toys4kサブセット（10サンプル）でgeometry metricsを計測。

## 条件
- データセット: Toys4k subset (10 samples)
- モデル: trellis 1.0, trellis2 2.0, hunyuan3d 2.0, hunyuan3d21 2.1
- メトリクス: Chamfer Distance (L2), F-score (τ=0.01,0.02), Hausdorff
- アライメント: similarity ICP (24 initial rotations)

## 実行
```bash
bash experiments/toys4k_baseline/run.sh                  # フル実行
bash experiments/toys4k_baseline/run.sh --skip-inference  # 評価のみ
bash experiments/toys4k_baseline/run.sh --skip-inference --workers=8
```

## 結果
`results/toys4k/report.md`
