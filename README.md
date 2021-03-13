# PyTorch
PyTorchに関連する実装です。
詳細は各ディレクトリ下のREADMEをご参照ください。

- [pytorch_classification][pytorch_classification]
    - PyTorch Lightningを利用した分類用のニューラルネットワークの実装
- [pytorch_autoencoder][pytorch_autoencoder]
    - PyTorch Lightningを利用したAutoEncoderの実装
    - データには信号データであるMITの心電図波形データセットを用います
- [pytorch_mlflow][pytorch_mlflow]
    - MLflowを利用した実験管理手順
- [deep_learning_with_pytorch][deep_learning_with_pytorch]
  - [Deep Learning with PyTorch](https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf)の実装
- [gan][gan]
    - DCGANによる画像生成の実装
- [gan_based_anomaly_detection][gan_based_anomaly_detection]
    - GANによる異常検知アルゴリズムの実装
- [pytorch_mlflow_hydra_optuna][pytorch_mlflow_hydra_optuna]
    - MLflow, Hydra, Optunaを利用した実験管理手順

[pytorch_classification]:./pytorch_classification
[pytorch_autoencoder]:./pytorch_autoencoder
[pytorch_mlflow]:./pytorch_mlflow
[deep_learning_with_pytorch]:./deep_learning_with_pytorch
[gan]:./gan
[gan_based_anomaly_detection]:./gan_based_anomaly_detection
[pytorch_mlflow_hydra_optuna]:./pytorch_mlflow_hydra_optuna


## 実行環境
下記コマンドで必要なパッケージをインストールできます。
```
poetry install
```