# default package
from argparse import ArgumentParser
from pprint import pprint
import os

# third party package
import torch 
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# my package
import src.torch_datamodule as torch_datamodule
import src.torch_network as torch_network

# global paramter 
PROJECT_DIR=os.environ.get("PROJECT_DIR","./")

# graph setting
plt.switch_backend('agg')


class LitClassifier(pl.LightningModule):
    def __init__(self,model=None,learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.model=model
        self.criterion=torch.nn.MSELoss()

    def forward(self,x):
        return self.model(x)

    def training_step(self,batch,batch_idx):
        x,y=batch
        y_hat=self(x)
        loss=self.criterion(y_hat,x)
        self.log('loss/train', loss)
        #to plot train and val in the same figure
        self.logger.experiment.add_scalars("loss(_same_figure)",
                                         {"train_loss": loss},self.global_step) 
        return loss

    def validation_step(self,batch,batch_idx):
        x,y=batch
        y_hat=self(x)
        loss=self.criterion(y_hat,x)
        self.log('loss/val',loss)
        #to plot train and val in the same figure
        self.logger.experiment.add_scalars("loss(_same_figure)",
                                         {"val_loss": loss},self.global_step) 

    def test_step(self,batch,batch_idx):
        x,y=batch
        y_hat=self(x)
        loss=self.criterion(y_hat,x)
        self.log('loss/test_loss',loss)
        if batch_idx==0:
            fig=plt.figure()
            plt.plot(x[0].to("cpu"))
            plt.plot(y_hat[0].to("cpu"))
            self.logger.experiment.add_figure("signal",fig,self.global_step)

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


def main():
    pl.seed_everything(1234)
    gpus = [0] if torch.cuda.is_available() else None

    filepath_train=f"{PROJECT_DIR}_data/x_train_normal.npy"
    filepath_test=f"{PROJECT_DIR}_data/x_test_normal.npy"
    dm = torch_datamodule.DataModule(
        filepath_train=filepath_train,
        filepath_test=filepath_test,
        )

    net=torch_network.CNN()
    model = LitClassifier(
        model=net,
        learning_rate=1e-3,
        )

    tb_logger = pl.loggers.TensorBoardLogger(PROJECT_DIR,name="lightning_logs")
    #sampleImg=torch.rand((1,1,28,28))
    #tb_logger.experiment.add_graph(net,sampleImg)

    trainer = pl.Trainer(
        max_epochs=2,
        logger=tb_logger,
        gpus=gpus,
        resume_from_checkpoint=None,
        )
    trainer.fit(model, datamodule=dm)
    result = trainer.test(model, datamodule=dm)
    pprint(result)


if __name__ == '__main__':
    main()



