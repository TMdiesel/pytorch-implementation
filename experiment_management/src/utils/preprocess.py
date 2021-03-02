"""CSV ファイルから読み込んだ DataFrame を Pickle として永続化する"""
import os
import logging
import pathlib

import pandas as pd

LOGGER = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)

    # データの置き場所
    data_dir = pathlib.Path('./data')

    # CSV ファイルを読み込む
    raw_data_file = data_dir / 'iris.csv'
    LOGGER.info(f'load data from {raw_data_file}')
    df = pd.read_csv(str(raw_data_file))

    LOGGER.info(f'data shape: {df.shape}')

    # Pickle として永続化する
    parsed_data_file = data_dir / 'iris.pickle'
    LOGGER.info(f'save data to {parsed_data_file}')
    df.to_pickle(parsed_data_file)


if __name__ == '__main__':
    main()