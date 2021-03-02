# default package
import pathlib
import pickle

# third party package
import numpy as np
import pandas as pd


def make_datapath_list(data_dir:str,label_path:str)->(np.array,np.array):
    """
    データのパスを格納したリストを作成する。
    """
    data_dir=pathlib.Path(data_dir)
    path_list=np.sort(list(data_dir.glob("*.pkl")))[:2000]

    with open(label_path,"rb") as f:
        label_list=pickle.load(f)

    return path_list,label_list