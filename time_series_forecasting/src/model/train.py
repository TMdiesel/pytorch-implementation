# default package
from argparse import ArgumentParser
from pprint import pprint
import os
import sys
import dataclasses as dc
import typing as t
import logging
import pathlib
import datetime
import tempfile
import glob

# third party package
import yaml
import pytz
import mlflow
import mlflow.pytorch
import hydra
from omegaconf import OmegaConf, DictConfig
import pandas as pd

import torch 
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# my package
import src.dataset.time_datamodule as time_datamodule
import src.model.network as network
import src.utils.utils as ut

# logger
logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TimeSeriesForecast(pl.LightningModule):
    def __init__(self,model=None,learning_rate=1e-3,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model=model
        self.criterion = nn.MSELoss()

    def forward(self,x):
        return self.model(x)

    def training_step(self,batch,batch_idx):
        x,y=batch
        y_hat=self(x).view(y.shape)
        loss=self.criterion(y_hat,y)
        self.logger.log_metrics({"train_loss":loss.item()},self.global_step)
        return loss

    def validation_step(self,batch,batch_idx):
        x,y=batch
        y_hat=self(x).view(y.shape)
        loss=self.criterion(y_hat,y)
        self.logger.log_metrics({"val_loss":loss.item()},self.global_step)
        self.log('val_loss',loss)
        return loss
    
    def test_step(self,batch,batch_idx):
        x,y=batch
        y_hat=self(x).view(y.shape)
        loss=self.criterion(y_hat,y)
        self.log('test_loss',loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.hparams.learning_rate)


@hydra.main(config_path="../../config/model", config_name="config")
def main(config:DictConfig):
    # logger
    cwd=pathlib.Path(hydra.utils.get_original_cwd())
    ut.init_root_logger(
        cwd.joinpath(config.log_dir),
        config.log_normal,
        config.log_error,
        )

    # pytorch setting
    pl.seed_everything(config.seed)
    gpus = [0] if torch.cuda.is_available() else None

    # data
    df_train=pd.read_csv(cwd.joinpath(config.data_path)).iloc[:,1:]
    df_train=df_train[5::6]
    dm = time_datamodule.TimeDataModule(
        input_length=config.input_length,
        label_length=config.label_length,
        df_train=df_train,
        )

    # model
    net=network.LSTM(
        input_size=df_train.shape[1],
        hidden_size=config.hidden_size,
        output_size=df_train.shape[1]*config.label_length,
    )
    model = TimeSeriesForecast(
        model=net,
        learning_rate=config.learning_rate,
        )

    # mlflow
    mlflow_tags={}
    mlflow_tags["mlflow.user"]=config.user
    mlflow_tags["mlflow.source.name"]=str(os.path.abspath(__file__)).replace("/",'\\')
    mlf_logger = MLFlowLogger(
        experiment_name=config.experiment_name,
        tracking_uri=str(cwd.joinpath(config.tracking_uri)),
        tags=mlflow_tags
        )

    # train&inference
    now=datetime.datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y-%m-%d-%H-%M-%S')
    ckpt_path=str(cwd.joinpath(f"{config.checkpoint_dir}{now}_{mlf_logger.run_id}"))
    checkpoint_callback = ModelCheckpoint(
        filepath=ckpt_path,
        monitor="val_loss",
        mode="auto",
        verbose=False,
        save_top_k=1,
        save_last=False,
        period=1,
    )

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        logger=mlf_logger,
        gpus=gpus,
        checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint=None,
        )
    trainer.fit(model, datamodule=dm)

    # save log, model, and config to mlflow
    mlf_logger.experiment.log_artifact(mlf_logger.run_id,
                                cwd.joinpath(config.log_dir,config.log_normal))
    mlf_logger.experiment.log_artifact(mlf_logger.run_id,
                                cwd.joinpath(config.log_dir,config.log_error))

    with tempfile.TemporaryDirectory() as dname:
        for ckptfile in glob.glob(f"{ckpt_path}*"):
            model=model.load_from_checkpoint(checkpoint_path=ckptfile)
            with tempfile.TemporaryDirectory() as dname:
                filepath = pathlib.Path(dname).joinpath(f"{pathlib.Path(ckptfile).stem}.pth")
                torch.save(model.state_dict(),filepath)
                mlf_logger.experiment.log_artifact(mlf_logger.run_id,filepath)

    for yamlfile in glob.glob(".hydra/*.yaml"):
        mlf_logger.experiment.log_artifact(mlf_logger.run_id,yamlfile)
    
    return checkpoint_callback.best_model_score


if __name__ == '__main__':
    main()



