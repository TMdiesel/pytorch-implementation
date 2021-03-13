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
from omegaconf import OmegaConf

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


def main(config):
    pl.seed_everything(config.seed)
    gpus = [0] if torch.cuda.is_available() else None

    filepath_list_train,label_list_train=generate_pathlist.make_datapath_list(
        config.project_dir+config.train_dir,
        config.project_dir+config.train_label_path)
    filepath_list_test,label_list_test=generate_pathlist.make_datapath_list(
        config.project_dir+config.test_dir,
        config.project_dir+config.test_label_path)
    dm = image_datamodule.ImageDataModule(
        filepath_list_train=filepath_list_train,
        filepath_list_test=filepath_list_test,
        label_list_train=label_list_train,
        label_list_test=label_list_test,
        )

    net=image_network.CNN()
    model = LitClassifier(
        model=net,
        learning_rate=config.learning_rate,
        )

    mlflow_tags={}
    mlflow_tags["mlflow.user"]=config.user
    mlflow_tags["mlflow.source.name"]=str(os.path.abspath(__file__)).replace("/",'\\')
    mlf_logger = MLFlowLogger(
        experiment_name=config.experiment_name,
        tracking_uri=config.tracking_uri,
        tags=mlflow_tags
        )

    now=datetime.datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y-%m-%d-%H-%M-%S')
    ckpt_path=f"{config.checkpoint_dir}{now}_{mlf_logger.run_id}"
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
                                config.log_dir+"/"+config.log_normal)
    mlf_logger.experiment.log_artifact(mlf_logger.run_id,
                                config.log_dir+"/"+config.log_error)

    with tempfile.TemporaryDirectory() as dname:
        for ckptfile in glob.glob(f"{ckpt_path}*"):
            model=model.load_from_checkpoint(checkpoint_path=ckptfile)
            with tempfile.TemporaryDirectory() as dname:
                filepath = pathlib.Path(dname).joinpath(f"{pathlib.Path(ckptfile).stem}.pth")
                torch.save(model.state_dict(),filepath)
                mlf_logger.experiment.log_artifact(mlf_logger.run_id,filepath)

    with tempfile.TemporaryDirectory() as dname:
        filepath = pathlib.Path(dname).joinpath("config.yaml")
        conf = OmegaConf.create(dc.asdict(config))
        conf_yaml=OmegaConf.to_yaml(conf)
        with open(filepath,"w") as f:
            f.write(conf_yaml)
        mlf_logger.experiment.log_artifact(mlf_logger.run_id,filepath)


@dc.dataclass
class Config:
    """設定値"""
    # trainer
    max_epochs:int=2
    seed:int=1234
    learning_rate:int=1e-3

    # data
    project_dir:str="path/to/project"
    train_dir:str="path/to/train"
    train_label_path:str="path/to/train"
    test_dir:str="path/to/test"
    test_label_path:str="path/to/test"

    # logging
    log_dir:str="./logs/logging_/model/model01"
    log_normal:str="log.log"
    log_error:str="error.log"

    # mlflow
    experiment_name: str="model01"
    tracking_uri: str="logs/mlruns"
    user: str="vscode"

    # checkpoint
    checkpoint_dir:str="./logs/lightning"


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_path')
    args=parser.parse_args()
    with open(args.config_path) as f:
        config=yaml.load(f,Loader=yaml.SafeLoader)
    config=Config(**config)

    ut.init_root_logger(
        pathlib.Path(config.log_dir),
        config.log_normal,
        config.log_error,
        )

    main(config)



