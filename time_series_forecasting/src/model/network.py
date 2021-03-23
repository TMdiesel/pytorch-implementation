# third party package
import torch
import pytorch_lightning as pl
from torch.nn import functional as F


class LSTM(torch.nn.Module):

    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.lstm1=torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.fc1=torch.nn.Linear(hidden_size,output_size)

    def forward(self,x):
        _,(hn,_)=self.lstm1(x)
        y=F.relu(hn.view(x.shape[0],-1))
        y=self.fc1(y)
        return y

