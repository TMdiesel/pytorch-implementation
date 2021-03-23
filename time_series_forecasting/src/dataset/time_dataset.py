""" 
note: 自作のdatasetを作るmodule
"""
# default package
import pickle

# third party package
import numpy as np
import pandas as pd

# third party package(torch-related)
import torch
from torchvision import models,transforms
import torch.utils.data as data


class TimeDataset(data.Dataset):

    def __init__(
        self,
        df:pd.DataFrame,
        input_length:int,
        label_length:int,
    ):
        self.df=df
        self.input_length=input_length
        self.label_length=label_length

    def __len__(self):
        length=self.df.shape[0]-self.input_length-self.label_length+1
        return length

    def __getitem__(self, index):
        """
        - torch tensor化
        - return input and label
        """
        array=torch.tensor(self.df.values)
        input_end=index+self.input_length
        input=array[index:input_end]
        label=array[input_end+1:input_end+1+self.label_length]
        return input.float(),label.float()