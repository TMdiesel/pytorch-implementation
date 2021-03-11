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
import src.dataset.image_dataset as image_dataset


class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        val_split:int=20,
        num_workers:int=4,
        seed:int=1234,
        batch_size:int=16,
        filepath_list_train:t.List[pathlib.Path]=None,
        filepath_list_test:t.List[pathlib.Path]=None,
        label_list_train:np.array=None,
        label_list_test:np.array=None,
        *args,
        **kwargs,
        ):

        super().__init__()
        if platform.system()=="Windows":
            num_workers=0

        self.val_split = val_split
        self.num_workers = num_workers
        self.seed = seed
        self.batch_size = batch_size
        self.filepath_list_train=filepath_list_train
        self.filepath_list_test =filepath_list_test
        self.dataset_train = ...
        self.dataset_val = ...

    def setup(self,stage:t.Optional[str]):
        """split the train and valid dataset"""
        dataset=image_dataset.ImageDataset(self.filepath_list_train,
                                           image_dataset.BaseTransform())
        train_length=len(dataset)
        self.dataset_train,self.dataset_val=random_split(
            dataset,[train_length-self.val_split,self.val_split]
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
        dataset=image_dataset.ImageDataset(self.filepath_list_test,
                                           image_dataset.BaseTransform())
        loader=DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

