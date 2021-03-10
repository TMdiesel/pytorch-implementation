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
import yaml
import mlflow
import mlflow.pytorch
from pytorch_lightning.loggers import MLFlowLogger

# my package
import src.dataset.image_datamodule as image_datamodule
import src.dataset.generate_pathlist as generate_pathlist
import src.model.dcgan as dcgan
import src.utils.utils as ut

# global paramter 
PROJECT_DIR=os.environ.get("PROJECT_DIR","..")

# logger
logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GAN(pl.LightningModule):
    def __init__(
        self,
        image_size=64,
        z_dim=20,
        learning_rate=1e-4,
        b1=0,
        b2=0.9,
        discriminator=dcgan.Discriminator(),
        generator=dcgan.Generator(),
        device="cuda",
        criterion=None,
        ):

        super().__init__()
        self.save_hyperparameters()

        self.image_size=image_size
        self.z_dim=z_dim
        self.learning_rate=learning_rate
        self.b1=b1
        self.b2=b2
        self.discriminator=discriminator
        self.generator=generator
        self.device=device
        self.criterion=criterion

    def forward(self,z):
        return self.generator(z)

    def training_step(self,batch,batch_idx,optimizer_idx):
        imgs,_=batch
        # sample noise
        z=torch.randn(imgs.shape[0],self.z_dim).to(self.device)
        z=z.view(imgs.shape[0],self.z_dim,1,1)
        # label
        label_real=torch.ones(imgs.shape[0], 1).to(self.device)
        label_fake=torch.ones(imgs.shape[0], 0).to(self.device)

        # generator
        if optimizer_idx==0:
            fake_images=self.generator(z)
            d_out_fake=self.discriminator(fake_images)
            g_loss=self.criterion(d_out_fake.view(-1),label_real)

            self.logger.log_metrics({"train_gloss":g_loss},self.global_step)
            return g_loss

        # discriminator
        if optimizer_idx==1:
            d_out_real=self.discriminator(imgs)
            d_loss_real=self.criterion(d_out_real.view(-1),label_real)
            d_loss_fake=self.criterion(d_out_fake.view(-1),label_fake)
            d_loss=d_loss_real+d_loss_fake

            self.logger.log_metrics({"train_dloss":d_loss},self.global_step)
            return d_loss

    def validation_step(self,batch,batch_idx):
        pass

    def test_step(self,batch,batch_idx):
        x,y=batch
        y_hat=self(x)
        loss=F.cross_entropy(y_hat,y)
        accuracy=self.accuracy(y_hat,y)
        self.log('test_loss',loss)
        self.log('test_acc', accuracy)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), 
                            lr=self.learning_rate, betas=(self.b1, self.b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), 
                            lr=self.learning_rate, betas=(self.b1, self.b2))
        return [opt_g, opt_d], []


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

    discriminator=dcgan.Discriminator(),
    generator=dcgan.Generator(),
    model = GAN(
        discriminator=discriminator,
        generator=generator,
        learning_rate=config.learning_rate,
        )

    mlflow_tags={}
    mlflow_tags["mlflow.runName"]=config.run_name
    mlflow_tags["mlflow.user"]=config.user
    mlflow_tags["mlflow.source.name"]=str(os.path.abspath(__file__)).replace("/",'\\')
    mlf_logger = MLFlowLogger(
        experiment_name=config.experiment_name,
        tracking_uri=config.tracking_uri,
        tags=mlflow_tags
        )

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        logger=mlf_logger,
        gpus=gpus,
        resume_from_checkpoint=None,
        )
    trainer.fit(model, datamodule=dm)
    result = trainer.test(model, datamodule=dm)
    pprint(result)

    mlf_logger.experiment.log_artifact(mlf_logger.run_id,
                                config.log_dir+"/"+config.log_normal)
    mlf_logger.experiment.log_artifact(mlf_logger.run_id,
                                config.log_dir+"/"+config.log_error)


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
    experiment_name: str="experiment"
    tracking_uri: str="logs/mlruns"
    run_name: str="test"
    user: str="vscode"


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



