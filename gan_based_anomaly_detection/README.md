# GANによる異常検知
- GANによる異常検知アルゴリズムの実装です。データとしてMNISTを用います。

## 実行手順
1. `notebooks/make_folders_and_data_downloads.ipynb`の各セルを実行してMNISTデータを`data/`にダウンロードする。  
2. ニューラルネットワークを学習させる。  
   実行コマンドは以下のとおりです。`config/config.yml`で各種パラメータを変更することができます。
    ```
    env PYTHONPATH=$(pwd) poetry run python src/model/image_generate.py --config_path=config/config.yml
    ```
3. mlflow uiで、学習曲線や生成画像を確認する。   
   下記コマンドでサーバーを立ち上げられます。
   ```
   poetry run mlflow ui --backend-store-uri ./logs/mlruns/
   ```

## 結果

## 参考
- GitHub: [YutaroOgawa/pytorch_advanced](https://github.com/YutaroOgawa/pytorch_advanced/tree/master/6_gan_anomaly_detection)