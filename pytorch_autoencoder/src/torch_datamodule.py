"""
note: 自作のdatasetを使ったdata module
"""
# default package
import platform
import typing as t
import pathlib

# third party package
import numpy as np
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

# my package
import src.torch_dataset as torch_dataset


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        val_ratio:int=0.2,
        num_workers:int=4,
        seed:int=1234,
        batch_size:int=64,
        filepath_train:t.List[pathlib.Path]=None,
        filepath_test:t.List[pathlib.Path]=None,
        *args,
        **kwargs,
        ):

        super().__init__()
        if platform.system()=="Windows":
            num_workers=0

        self.val_ratio = val_ratio
        self.num_workers = num_workers
        self.seed = seed
        self.batch_size = batch_size
        self.filepath_train=filepath_train
        self.filepath_test =filepath_test
        self.dataset_train = ...
        self.dataset_val = ...

    def setup(self,stage:t.Optional[str]):
        """split the train and valid dataset"""
        dataset=torch_dataset.Dataset(self.filepath_train,
                                           torch_dataset.BaseTransform())
        train_length=len(dataset)
        val_split=int(train_length*self.val_ratio)
        self.dataset_train,self.dataset_val=random_split(
            dataset,[train_length-val_split,val_split]
        )

    def train_dataloader(self):
        loader=DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
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
        dataset=torch_dataset.Dataset(self.filepath_test,
                                           torch_dataset.BaseTransform())
        loader=DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

