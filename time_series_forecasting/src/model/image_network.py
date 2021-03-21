# third party package
import torch
import pytorch_lightning as pl
from torch.nn import functional as F


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,28,kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(28,10,kernel_size=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))
        self.dropout1=torch.nn.Dropout(0.25)
        self.fc1=torch.nn.Linear(250,18)
        self.dropout2=torch.nn.Dropout(0.08)
        self.fc2=torch.nn.Linear(18,10)

    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.dropout1(x)
        x=torch.relu(self.fc1(x.view(x.size(0), -1)))
        x=F.leaky_relu(self.dropout2(x))
        return F.softmax(self.fc2(x),dim=1)


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
