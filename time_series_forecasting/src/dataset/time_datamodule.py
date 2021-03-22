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
        val_ratio:float=0.2,
        num_workers:int=4,
        seed:int=1234,
        batch_size:int=16,
        df_train:pd.DataFrame=None,
        df_test:pd.DataFrame=None,
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
        self.df_test=df_test

        val_length=int(len(df_train)*val_ratio)
        self.df_train=df_train.iloc[:-val_length]
        self.df_val=df_train.iloc[-val_length:]

    def setup(self,stage:t.Optional[str]):
        """split the train and valid dataset"""
        self.dataset_train=time_dataset.TimeDataset(
            self.df_train,
            self.input_length,
            self.label_length,
            )
        self.dataset_val=time_dataset.TimeDataset(
            self.df_val,
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
            self.df_test,
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

