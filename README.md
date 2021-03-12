# PyTorch
PyTorchに関連する実装です。
詳細は各ディレクトリ下のREADMEをご参照ください。

- [pytorch_classification][pytorch_classification]
    - PyTorch Lightningを利用した分類用のニューラルネットワークの実装です。
- [pytorch_autoencoder][pytorch_autoencoder]
    - PyTorch Lightningを利用したAutoEncoderの実装です。データには信号データであるMITの心電図波形データセットを用います
- [pytorch_mlflow][pytorch_mlflow]
    - MLflowを利用した実験管理の手順です。
- [deep_learning_with_pytorch][deep_learning_with_pytorch]
  - [Deep Learning with PyTorch](https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf)の実装です。
- [gan][gan]
    - DCGANによる画像生成の実装です。

[pytorch_classification]:./pytorch_classification
[pytorch_autoencoder]:./pytorch_autoencoder
[pytorch_mlflow]:./pytorch_mlflow
[deep_learning_with_pytorch]:./deep_learning_with_pytorch
[gan]:./gan

## 実行環境
下記コマンドで必要なパッケージをインストールできます。
```
poetry install
```