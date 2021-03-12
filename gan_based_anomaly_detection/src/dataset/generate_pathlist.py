# default package
import pathlib
import pickle

# third party package
import numpy as np
import pandas as pd


def make_datapath_list(data_dir:str)->(np.array):
    """
    データのパスを格納したリストを作成する。
    """
    data_dir=pathlib.Path(data_dir)
    path_list=np.sort(list(data_dir.glob("*.jpg")))

    return path_list