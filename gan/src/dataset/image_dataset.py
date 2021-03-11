""" 
note: 自作のdatasetを作るmodule
"""
# default package
import pickle

# third party package
import numpy as np
from PIL import Image
from skimage.transform import resize as skresize

# third party package(torch-related)
import torch
from torchvision import models,transforms
import torch.utils.data as data


class BaseTransform():
    """
    Attributes
    ----------
    mean : 各色チャネルの平均値
    std : 各色チャネルの標準偏差
    """
    def __init__(self, mean:float=0.5, std:float=0.5):
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),  
            transforms.Normalize(mean, std),
        ])

    def __call__(self, img):
        return self.base_transform(img)


class ImageDataset(data.Dataset):
    """
    Attributes
    ----------
    filepath_list : 画像のパスを格納したリスト
    transform : 前処理クラスのインスタンス
    """

    def __init__(self, filepath_list,transform=None):
        self.filepath_list = filepath_list  
        self.transform = transform  

    def __len__(self):
        return len(self.filepath_list)

    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータを取得
        '''
        path = self.filepath_list[index]
        img=Image.open(path)
        transformed = self.transform(img)  
        return transformed