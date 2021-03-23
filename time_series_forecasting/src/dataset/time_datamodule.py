"""
note: 自作のdatasetを使ったdata module
"""
# default package
import platform
import typing as t
import pathlib

# third party package
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

# my package
import src.dataset.time_dataset as time_dataset


class TimeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        input_length:int,
        label_length:int,
        num_workers:int=4,
        seed:int=1234,
        batch_size:int=16,
        array_train:np.array=None,
        array_val  :np.array=None,
        array_test :np.array=None,
        *args,
        **kwargs,
        ):

        super().__init__()
        if platform.system()=="Windows":
            num_workers=0

        self.input_length=input_length
        self.label_length=label_length
        self.num_workers = num_workers
        self.seed = seed
        self.batch_size = batch_size
        self.array_test=array_test
        self.array_train=array_train
        self.array_val=array_val

    def setup(self,stage:t.Optional[str]):
        """split the train and valid dataset"""
        self.dataset_train=time_dataset.TimeDataset(
            self.array_train,
            self.input_length,
            self.label_length,
            )
        self.dataset_val=time_dataset.TimeDataset(
            self.array_val,
            self.input_length,
            self.label_length,
            )

    def train_dataloader(self):
        loader=DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        loader=DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def test_dataloader(self):
        dataset=time_dataset.TimeDataset(
            self.array_test,
            self.input_length,
            self.label_length,
            )
        loader=DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

