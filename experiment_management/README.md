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



```
poetry run mlflow run  . --no-conda
```

## メモ
- Dockerコンテナで実行することもできる。
- .envファイルの読み込みが上手くいかない… 
    - `env PYTHONPATH=$(pwd) poetry run python hoge.py` としておく
- `poetry run mlflow ui`
- mlrunsの場所変更がわからないので、作業ディレクトリに置いておく
- dataclassesの中でネストする方法がわからなかったので、1重のconfig.yaml, dataclassを与える

## todo
- [x] config作成
- [x] mlflow tracking使う
- [ ] loggingでoutput作成
- [ ] 環境変数指定

## 参考
- [Python: MLflow Projects を使ってみる](https://blog.amedama.jp/entry/mlflow-projects)
- [MLflow GitHub](https://github.com/mlflow/mlflow)
- [MLflow Projects documentation](https://www.mlflow.org/docs/latest/projects.html)
- [mlflowを使ってデータ分析サイクルの効率化する方法を考える](https://qiita.com/masa26hiro/items/574c48d523ed76e76a3b)
- [オレオレKaggle実験管理](https://zenn.dev/fkubota/articles/f7efe69fd2044d)