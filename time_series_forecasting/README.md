# Time series forecasting
- 時系列予測の実装です。
- マックスプランク生物地球化学研究所によって記録された気象時系列データセットを使用します。
このデータセットには、気温、大気圧、湿度などの14の異なる機能が含まれています。


## 実行手順
1. [TensorFlow公式ドキュメント](https://www.tensorflow.org/tutorials/structured_data/time_series#the_weather_dataset)より気象時系列データをダウンロードします。
2. 下記コマンドでモデルを学習させます。
    ```
    env PYTHONPATH=$(pwd) poetry run python src/model/train.py
    ```
3. MLflow UIを立ち上げて学習結果を確認します。
    ```
    poetry run mlflow ui  --backend-store-uri=logs/mlruns
    ```



## 参考
- [TensorFlow時系列予測](https://www.tensorflow.org/tutorials/structured_data/time_series)