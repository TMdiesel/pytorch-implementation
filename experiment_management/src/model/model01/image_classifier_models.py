# default package
from argparse import ArgumentParser
from pprint import pprint
import os
import sys

# third party package
import torch 
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter

print(sys.path)
# my package
import src.dataset.dataset01.image_datamodule as image_datamodule
import src.dataset.dataset01.generate_pathlist as generate_pathlist
import src.model.model01.image_network as image_network

# global paramter 
PROJECT_DIR=os.environ.get("PROJECT_DIR","..")


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


def main():
    pl.seed_everything(1234)
    gpus = [0] if torch.cuda.is_available() else None

    filepath_list_train,label_list_train=generate_pathlist.make_datapath_list(
        f"{PROJECT_DIR}/pytorch_classification/_data/train_image",
        f"{PROJECT_DIR}/pytorch_classification/_data/train_label/label.pkl")
    filepath_list_test,label_list_test=generate_pathlist.make_datapath_list(
        f"{PROJECT_DIR}/pytorch_classification/_data/test_image",
        f"{PROJECT_DIR}/pytorch_classification/_data/test_label/label.pkl")
    dm = image_datamodule.ImageDataModule(
        filepath_list_train=filepath_list_train,
        filepath_list_test=filepath_list_test,
        label_list_train=label_list_train,
        label_list_test=label_list_test,
        )

    net=image_network.CNN()
    model = LitClassifier(
        model=net,
        learning_rate=1e-3,
        )

    tb_logger = pl.loggers.TensorBoardLogger(f"{PROJECT_DIR}/experiment_management/logs",name="lightning_logs")
    sampleImg=torch.rand((1,1,28,28))
    tb_logger.experiment.add_graph(net,sampleImg)

    trainer = pl.Trainer(
        max_epochs=1,
        logger=tb_logger,
        gpus=gpus,
        resume_from_checkpoint=None,
        )
    trainer.fit(model, datamodule=dm)
    result = trainer.test(model, datamodule=dm)
    pprint(result)


if __name__ == '__main__':
    main()



