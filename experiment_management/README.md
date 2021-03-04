# 実験管理
- MLflowを用いた実験管理の手順です。サンプルプロジェクトとしてPyTorchを用いたMNISTの分類を対象とします。

## ディレクトリ構成

### 説明
| ディレクトリ名 | 説明 |
| :---: | :--- | 
| `_data`|データセットや特徴量を格納します。
|`config`|学習時の設定パラメータをyamlファイルで管理します。内部は`src`の構造と対応させます。
| `diary`|分析日記をつけます。
| `logs` | PytorchLightning, MLflow, その他(loggingなど)で出力されたログを格納します。
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
│   ├── lightning_logs
│   │   ├── version_0
│   │   ├── ...
│   ├── logging_
│   │   ├── dataset
│   │   ├── feature
│   │   └── model
│   └── mlruns
├── notebooks
│   └── template.ipynb
├── scripts
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

## 実行環境
poetryを用いてPythonパッケージを管理します。下記コマンドで必要なパッケージをインストールできます。
```
poetry install
```

## 実験管理手順
### 用語
実験、runという単位は、[オレオレKaggle実験管理](https://zenn.dev/fkubota/articles/f7efe69fd2044d)と同様です。
### 手順
1番目の実験における`model`の学習を例に取ります。手順は以下のとおりです。
1. 実験ごとに`model/model1`の下にコードを置く。
2. `config/model/model1`の下に学習パラメータ設定用の`config1.yml`を置く。同一の実験下でパラメータを変えて複数のrunを行うときは、新たに`config2.yml`...を作る。
3. `bash scripts/run_python.sh`でスクリプトを実行する。ここで、CONFIG_PATH, PYTHON_SCRIPTは必要に応じて書き換える。
4. `bash scripts/mlflow_ui.sh`でMLflow UIを立ち上げ、実験結果をまとめて見る。




## 参考
- [Python: MLflow Projects を使ってみる](https://blog.amedama.jp/entry/mlflow-projects)
- [MLflow GitHub](https://github.com/mlflow/mlflow)
- [MLflow Projects documentation](https://www.mlflow.org/docs/latest/projects.html)
- [mlflowを使ってデータ分析サイクルの効率化する方法を考える](https://qiita.com/masa26hiro/items/574c48d523ed76e76a3b)
- [オレオレKaggle実験管理](https://zenn.dev/fkubota/articles/f7efe69fd2044d)
- [Pytorch Lightning documentation](https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.loggers.mlflow.html)
- [MLflow tags](https://github.com/mlflow/mlflow/blob/9fd60eeee77dbda37bae0ff97bc899e2bf87605f/mlflow/utils/mlflow_tags.py#L7)
- [MLFlowLogger source](https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/loggers/mlflow.html)