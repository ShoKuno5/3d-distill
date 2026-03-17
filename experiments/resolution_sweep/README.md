# resolution_sweep

## 目的
入力画像の解像度（300/512/1024）が3D再構成品質に与える影響を調査。

## 条件
- データセット: Toys4k subset, 3解像度バリアント（300×300, 512×512, 1024×1024）
- モデル: trellis 1.0, trellis2 2.0, hunyuan3d 2.0, hunyuan3d21 2.1
- configs: `res_300.yaml`, `res_512.yaml`, `res_1024.yaml`

## 実行
```bash
bash experiments/resolution_sweep/run.sh                  # 全解像度
bash experiments/resolution_sweep/run.sh --only=512       # 特定解像度のみ
bash experiments/resolution_sweep/run.sh --render-grid    # 比較グリッド画像生成
```

## 結果
`results/resolution_sweep/comparison.md`
