# third party package
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
from torchinfo import summary


class FC(torch.nn.Module):
    def __init__(self,hidden_dim=10):
        super().__init__()
        self.l1=torch.nn.Linear(720,hidden_dim)
        self.l2=torch.nn.Linear(hidden_dim,720)

    def forward(self,x):
        x=self.l1(x)
        x=self.l2(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_conv1=nn.Conv1d(1,10,kernel_size=3,stride=2,padding=1)
        self.enc_conv2=nn.Conv1d(10,5,kernel_size=3,stride=2,padding=1)
        self.enc_conv3=nn.Conv1d(5,2,kernel_size=3,stride=2,padding=1)
        self.dec_conv3=nn.Conv1d(2,5,kernel_size=1,stride=1)
        self.dec_conv2=nn.ConvTranspose1d(5,10,kernel_size=2,stride=2)
        self.dec_conv1=nn.ConvTranspose1d(10,1,kernel_size=2,stride=2)
        self.dec_conv0=nn.ConvTranspose1d(1,1,kernel_size=2,stride=2)
        
    def forward(self, x):
        encoded = self._encode(x)
        decoded = self._decode(encoded)
        return  decoded
    
    def _encode(self, x):
        x = x.view(x.shape[0], 1,-1)
        x=F.relu(self.enc_conv1(x))
        x=F.relu(self.enc_conv2(x))
        x=F.relu(self.enc_conv3(x))
        return x
    
    def _decode(self, x):
        x =F.relu(self.dec_conv3(x)) 
        x =F.relu(self.dec_conv2(x)) 
        x =self.dec_conv1(x)
        x =self.dec_conv0(x)
        x = x.view(x.shape[0],-1)
        return x


def _main():
    """ネットワーク構造の確認用"""
    model=CNN()
    batch_size=1
    summary(
        model,
        input_data=[batch_size,720],
        col_names=["output_size", "num_params"]
    )


if __name__=="__main__":
    _main()
