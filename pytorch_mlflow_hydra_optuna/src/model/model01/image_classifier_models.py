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

import torch 
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# my package
import src.dataset.dataset01.image_datamodule as image_datamodule
import src.dataset.dataset01.generate_pathlist as generate_pathlist
import src.model.model01.image_network as image_network
import src.utils.utils as ut

# global paramter 
PROJECT_DIR=os.environ.get("PROJECT_DIR","..")

# logger
logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LitClassifier(pl.LightningModule):
    def __init__(self,model=None,learning_rate=1e-3,**kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model=model
        self.accuracy = pl.metrics.Accuracy()

    def forward(self,x):
        return self.model(x)

    def training_step(self,batch,batch_idx):
        x,y=batch
        y_hat=self(x)
        loss=F.cross_entropy(y_hat,y)
        accuracy=self.accuracy(y_hat,y)
        self.logger.log_metrics({"train_acc":accuracy.item()},self.global_step)
        self.logger.log_metrics({"train_loss":loss.item()},self.global_step)
        return loss

    def validation_step(self,batch,batch_idx):
        x,y=batch
        y_hat=self(x)
        loss=F.cross_entropy(y_hat,y)
        accuracy=self.accuracy(y_hat,y)
        self.logger.log_metrics({"val_acc":accuracy.item()},self.global_step)
        self.logger.log_metrics({"val_loss":loss.item()},self.global_step)

    def test_step(self,batch,batch_idx):
        x,y=batch
        y_hat=self(x)
        loss=F.cross_entropy(y_hat,y)
        accuracy=self.accuracy(y_hat,y)
        self.log('test_loss',loss)
        self.log('test_acc', accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.hparams.learning_rate)


@hydra.main(config_path="../../../config/model/model01", config_name="config")
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
    filepath_list_train,label_list_train=generate_pathlist.make_datapath_list(
        cwd.joinpath(config.train_dir),
        cwd.joinpath(config.train_label_path))
    filepath_list_test,label_list_test=generate_pathlist.make_datapath_list(
        cwd.joinpath(config.test_dir),
        cwd.joinpath(config.test_label_path))
    dm = image_datamodule.ImageDataModule(
        filepath_list_train=filepath_list_train,
        filepath_list_test=filepath_list_test,
        label_list_train=label_list_train,
        label_list_test=label_list_test,
        )

    # model
    net=image_network.CNN()
    model = LitClassifier(
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
        save_top_k=None,
        monitor=None,
    )

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        logger=mlf_logger,
        gpus=gpus,
        checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint=None,
        )
    trainer.fit(model, datamodule=dm)
    result = trainer.test(model, datamodule=dm)
    pprint(result)

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


if __name__ == '__main__':
    main()



