# 実験管理
- MLflow, Hydra, Optunaを用いた実験管理の手順です。サンプルプロジェクトとしてPyTorchを用いたMNISTの分類を対象とします。

## ディレクトリ構成

### 説明
| ディレクトリ名 | 説明 |
| :---: | :--- | 
| `_data`|データセットや特徴量を格納します。
|`config`|学習時の設定パラメータをyamlファイルで管理します。内部は`src`の構造と対応させます。
| `diary`|分析日記をつけます。
| `logs` | PytorchLightning, MLflow, その他(loggingなど)で出力されたログを格納します。`mlruns`のartifactsにログ、設定ファイル、モデルなどをすべて保存しています。
|`outputs`| Hydraの出力ディレクトリです。
|`multirun`| Hydraの出力ディレクトリです。
|`notebooks`|EDA用のjupyter notebookを格納します。
|`scripts`|シェルスクリプトを格納します。
|`src`|Pythonスクリプトファイルを格納します。<br> ファイルは`dataset`,`feature`,`model`,`utils`というサブディレクトリで整理します。<br> `utils`以外の下にはさらに実験ごとにサブディレクトリを作ります。(後方互換性を気にしないため。)|

### 構成
```
├── _data
│   ├── data
│   ├── feature
├── config
│   ├── dataset
│   ├── feature
│   └── model
├── diary
├── logs
│   ├── lightning
│   ├── logging_
│   │   ├── dataset
│   │   ├── feature
│   │   └── model
│   └── mlruns
├── notebooks
├── scripts
├── outputs
├── multirun
└── src
    ├── dataset
    │   ├── dataset01
    │   └── ...
    ├── feature
    │   ├── feature01
    │   └── ...
    ├── model
    │   ├── model01
    │   └── ...
    └── utils

```


## 実験管理手順
### 用語
実験、runという単位は、[オレオレKaggle実験管理](https://zenn.dev/fkubota/articles/f7efe69fd2044d)と同様です。
### 手順
1番目の実験における`model`の学習を例に取ります。手順は以下のとおりです。
1. 実験ごとに`model/model1`の下にコードを置きます。
2. `config/model/model1`の下に学習パラメータ設定用の`config.yaml`を置きます。yamlファイル中のパスは作業ディレクトリからの相対パスで書きます。
3. 下記コマンドでスクリプトを実行します。
   Optunaで最適化したいパラメータをCLIで指定できます。
    ```
    env PYTHONPATH=$(pwd) poetry run python src/model/model01 image_classifier_models.py  -m 'learning_rate=interval(0.001,0.002)'                                                
    ```
4. `bash scripts/mlflow_ui.sh`でMLflow UIを立ち上げ、実験結果をまとめて見ることができます。
5. 必要に応じて、`notebooks/load_model.ipynb`でexperiment_nameとrun_idを指定してモデルをロードし、後処理を行います。




## 参考
- 実験管理
  - [オレオレKaggle実験管理](https://zenn.dev/fkubota/articles/f7efe69fd2044d)
- MLflow
   - [Python: MLflow Projects を使ってみる](https://blog.amedama.jp/entry/mlflow-projects)
   - [MLflow GitHub](https://github.com/mlflow/mlflow)
   - [MLflow Projects documentation](https://www.mlflow.org/docs/latest/projects.html)
   - [mlflowを使ってデータ分析サイクルの効率化する方法を考える](https://qiita.com/masa26hiro/items/574c48d523ed76e76a3b)
   - [MLflow tags](https://github.com/mlflow/mlflow/blob/9fd60eeee77dbda37bae0ff97bc899e2bf87605f/mlflow/utils/mlflow_tags.py#L7)
   - [MLFlowLogger source](https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/loggers/mlflow.html)
   - [MLflowのデータストアを覗いてみる](https://blog.hoxo-m.com/entry/mlflow_store)
   - [MLflow使い始めたのでメモ](https://zenn.dev/currypurin/articles/15bd449da18807b08f89)
- Hydra
   - [Hydra公式ドキュメント](https://hydra.cc/docs/intro/)
   - [Python: OmegaConf を使ってみる](https://blog.amedama.jp/entry/omega-conf)
   - [ハイパラ管理のすすめ -ハイパーパラメータをHydra+MLflowで管理しよう-](https://ymym3412.hatenablog.com/entry/2020/02/09/034644)
- Pytorch
   - [Pytorch Lightning documentation](https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.loggers.mlflow.html)
   - [Pytorch ModelCheckpoint](https://pytorch-lightning.readthedocs.io/en/1.0.2/_modules/pytorch_lightning/callbacks/model_checkpoint.html)
- Optuna
    - [supikiti/hydra-mlflow-optuna-sample](https://github.com/supikiti/hydra-mlflow-optuna-sample)
    - [Optuna Sweeper plugin](https://hydra.cc/docs/next/plugins/optuna_sweeper)