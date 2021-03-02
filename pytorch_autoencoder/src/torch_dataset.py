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

    def __init__(self, mean:float=None, std:float=None):
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),  
            #transforms.Normalize(mean, std),
        ])

    def __call__(self, array:np.array):
        return torch.tensor(array)


class Dataset(data.Dataset):

    def __init__(self, filepath, transform=None):
        self.x=np.load(filepath)
        self.transform = transform  

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        transformed = self.transform(self.x[index])  
        return transformed.float(), 0