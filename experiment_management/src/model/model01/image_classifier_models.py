# default package
from argparse import ArgumentParser
from pprint import pprint
import os
import sys
import dataclasses as dc
import typing as t
import logging
import pathlib

# third party package
import torch 
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter
import mlflow
import mlflow.pytorch
import yaml

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
    def __init__(self,model=None,learning_rate=1e-3):
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
        self.log('loss/train', loss)
        self.log('acc/train_acc', accuracy)
        #to plot train and val in the same figure
        self.logger.experiment.add_scalars("loss(_same_figure)",
                                         {"train_loss": loss},self.global_step) 
        return loss

    def validation_step(self,batch,batch_idx):
        x,y=batch
        y_hat=self(x)
        loss=F.cross_entropy(y_hat,y)
        accuracy=self.accuracy(y_hat,y)
        self.log('loss/val',loss)
        self.log('acc/val_acc', accuracy)
        #to plot train and val in the same figure
        self.logger.experiment.add_scalars("loss(_same_figure)",
                                         {"val_loss": loss},self.global_step) 

    def test_step(self,batch,batch_idx):
        x,y=batch
        y_hat=self(x)
        loss=F.cross_entropy(y_hat,y)
        accuracy=self.accuracy(y_hat,y)
        self.log('loss/test_loss',loss)
        self.log('acc/test_acc', accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser

    def training_epoch_end(self,outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("loss(epoch)/train",avg_loss,self.current_epoch)


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

    tb_logger = pl.loggers.TensorBoardLogger(f"{PROJECT_DIR}/experiment_management/logs",name="lightning_logs")
    sampleImg=torch.rand((1,1,28,28))
    tb_logger.experiment.add_graph(net,sampleImg)

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        logger=tb_logger,
        gpus=gpus,
        resume_from_checkpoint=None,
        )
    trainer.fit(model, datamodule=dm)
    result = trainer.test(model, datamodule=dm)
    pprint(result)

    mlflow.pytorch.autolog()


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

    logger.error("testerror")

    main(config)



