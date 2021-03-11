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
    def __init__(self, mean:float=None, std:float=None):
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),  
            #transforms.Normalize(mean, std),
        ])

    def __call__(self, array:np.array):
        array=array/255
        return torch.tensor(array)


class ImageDataset(data.Dataset):
    """
    Attributes
    ----------
    filepath_list : 画像のパスを格納したリスト
    transform : 前処理クラスのインスタンス
    """

    def __init__(self, filepath_list, label_list, class_num,transform=None):
        self.filepath_list = filepath_list  
        self.label_list = label_list
        self.class_num=class_num
        self.transform = transform  

    def __len__(self):
        return len(self.filepath_list)

    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとラベルを取得
        '''
        path = self.filepath_list[index]
        with open(path,"rb") as f:
            array=pickle.load(f)
        transformed = self.transform(array)  
        
        label=self.label_list[index]
        onehot_label=torch.eye(self.class_num)[label]
        return transformed.float(), torch.tensor(label).long()