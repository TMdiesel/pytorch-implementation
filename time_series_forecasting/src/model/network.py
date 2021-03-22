# third party package
import torch
import pytorch_lightning as pl
from torch.nn import functional as F


class FC(torch.nn.Module):
    def __init__(self,hidden_dim=128):
        super().__init__()
        self.l1=torch.nn.Linear(28*28,hidden_dim)
        self.l2=torch.nn.Linear(hidden_dim,10)

    def forward(self,x):
        x=x.view(x.size(0),-1)
        x=torch.relu(self.l1(x))
        x=torch.relu(self.l2(x))
        return x
