# PyTorch Lightningを利用した画像分類
PyTorch Lightningを利用した分類用のニューラルネットワークの実装です。
MNISTデータを対象としています。

## DataModuleについて
MNISTDataModuleというPyTorchの標準のDataModuleを使うと2行でMNISTのデータを呼び出すことができます。
```
from pl_examples.basic_examples.mnist_datamodule import MNISTDataModule

dm = MNISTDataModule()
```
しかし、本リポジトリでは自前のデータセットにも対応できるように、上記のmoduleを使わずにdataset, datamoduleの実装を行います。


## 実行手順
1. MNISTデータをダウンロードする。  
    実行コマンドは以下のとおりです。
    ```
    poetry run python src/download_mnist.py
    ```
2. ニューラルネットワークを学習させる。  
    実行コマンドは以下のとおりです。
    ```
    poetry run python src/image_classifier_models.py
    ```

## 可視化
TensorBoardを起動させることで、学習曲線やネットワーク構造を可視化できます。
実行コマンドは以下のとおりです。
```
poetry run tensorboard --logdir=path/to/log
```
## To do
- TensorBoard用のDockerコンテナを作る。

## 参考
- https://github.com/PyTorchLightning/pytorch-lightning
- https://github.com/oreilly-japan/deep-learning-from-scratch
- https://github.com/YutaroOgawa/pytorch_advanced