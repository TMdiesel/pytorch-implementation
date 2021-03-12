# default package
from argparse import ArgumentParser
from pprint import pprint
import os
import sys
import dataclasses as dc
import typing as t
import logging
import pathlib
import tempfile
import datetime

# third party package
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter
import yaml
import mlflow
import mlflow.pytorch
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint

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
        criterion=None,
        **kwargs
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
        self.criterion=criterion
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self,z):
        return self.generator(z)

    def training_step(self,batch,batch_idx,optimizer_idx):
        imgs=batch
        # sample noise
        z=torch.randn(imgs.shape[0],self.z_dim).to(self.dev)
        z=z.view(imgs.shape[0],self.z_dim,1,1)
        # label
        label_real=torch.full((imgs.shape[0],), 1).float().to(self.dev)
        label_fake=torch.full((imgs.shape[0],), 0).float().to(self.dev)
        # fake
        fake_images=self.generator(z)
        d_out_fake=self.discriminator(fake_images)

        # generator
        if optimizer_idx==0:
            g_loss=self.criterion(d_out_fake.view(-1),label_real)

            self.logger.log_metrics({"train_gloss":g_loss.to("cpu").detach().numpy().item()},self.global_step)
            return g_loss
        # discriminator
        if optimizer_idx==1:
            d_out_real=self.discriminator(imgs)
            d_loss_real=self.criterion(d_out_real.view(-1),label_real)
            d_loss_fake=self.criterion(d_out_fake.view(-1),label_fake)
            d_loss=d_loss_real+d_loss_fake

            self.logger.log_metrics({"train_dloss":d_loss.to("cpu").detach().numpy().item()},self.global_step)
            return d_loss

    def on_epoch_end(self):
        z=torch.randn(1,self.z_dim)
        z=z.view(1,self.z_dim,1,1).to(self.dev)
        img = self(z).to("cpu").detach().numpy()[0][0]
        fig, ax = plt.subplots(1)
        ax.imshow(img,cmap="gray")
        ax.grid(False)

        with tempfile.TemporaryDirectory() as dname:
            filepath = pathlib.Path(dname).joinpath(f"{self.current_epoch}.png")
            fig.savefig(filepath)
            plt.close()
            self.logger.experiment.log_artifact(local_path=filepath,run_id=self.logger.run_id) 

    def validation_step(self,batch,batch_idx):
        pass

    def test_step(self,batch,batch_idx):
        pass

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), 
                            lr=self.learning_rate, betas=(self.b1, self.b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), 
                            lr=self.learning_rate, betas=(self.b1, self.b2))
        return [opt_g, opt_d], []


def main(config):
    pl.seed_everything(config.seed)
    gpus = [0] if torch.cuda.is_available() else None

    filepath_list_train=generate_pathlist.make_datapath_list(
        config.project_dir+config.train_dir,
    )
    dm = image_datamodule.ImageDataModule(
        filepath_list_train=filepath_list_train,
        filepath_list_test=filepath_list_train,
        )

    discriminator=dcgan.Discriminator()
    generator=dcgan.Generator()
    criterion=nn.BCEWithLogitsLoss(reduction="mean")
    model = GAN(
        discriminator=discriminator,
        generator=generator,
        criterion=criterion,
        **dc.asdict(config),
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

    now=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    checkpoint_callback = ModelCheckpoint(
        filepath=f"{config.checkpoint_dir}{now}_{mlf_logger.run_id}",
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

    # save to mlflow
    mlf_logger.experiment.log_artifact(mlf_logger.run_id,
                                config.log_dir+"/"+config.log_normal)
    mlf_logger.experiment.log_artifact(mlf_logger.run_id,
                                config.log_dir+"/"+config.log_error)


@dc.dataclass
class Config:
    """設定値"""
    # GAN
    image_size:int=64
    z_dim:int=20
    learning_rate:int=1e-4
    b1:float=0
    b2:float=0.9

    # trainer
    max_epochs:int=2

    # seed
    seed:int=1234

    # data
    project_dir:str="path/to/project"
    train_dir:str="path/to/train"
    test_dir:str="path/to/test"

    # logging
    log_dir:str="./logs/logging_/model/model01"
    log_normal:str="log.log"
    log_error:str="error.log"

    # mlflow
    experiment_name: str="experiment"
    tracking_uri: str="logs/mlruns"
    run_name: str="test"
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



