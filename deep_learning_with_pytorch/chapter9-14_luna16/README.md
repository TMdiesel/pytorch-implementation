# 肺がん早期発見プロジェクト

- CTスキャン画像を用いた、肺の悪性腫瘍の検出です。

## データ
- [LUNAグランドチャレンジ](https://luna16.grand-challenge.org/download/)のデータを用います。
- `candidates.csv`
  - 結節の候補に関するデータ
  - class列は0が結節ではない、1が悪性・良性問わず結節の候補
- `annotations.csv`
    - 結節候補に関する他のデータ
- `subset*/*.raw`, `subset*/*.mhd`
    - CTデータ

## 実装
### 10章
- `candidates.csv`と`annotations.csv`を結合
- CTデータ
  - -1000～1000HUでクリップ
  - ミリメートル単位の配列をボクセルアドレスベースの座標系に変換
  - 結節の抽出
- PyTorch Datasetの作成
    - ラベル：結節 or not, 陽性 or 陰性
### 11章
- 結節候補を分類

## 参考
- [Deep Learning with PyTorch](https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf)
- [著者GitHub](https://github.com/deep-learning-with-pytorch/dlwpt-code)
- [日本語版著者GitHub](https://github.com/Gin5050/deep-learning-with-pytorch-ja)