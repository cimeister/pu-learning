import torch
from torch import nn
import torch.nn.functional as F

from .loss import PULoss

CONFIG_NAME = 'config.json'
WEIGHTS_NAME = 'pytorch_model.bin'



class PUModel(nn.Module):
   """
   Basic Multi-layer perceptron as described in "Positive-Unlabeled Learning with Non-Negative Risk Estimator"
   """
   def __init__(self):
        super(PUModel, self).__init__()
        self.fc1 = nn.Linear(784,300, bias=False)
        self.bn1 = nn.BatchNorm1d(300)
        self.fc2 = nn.Linear(300,300, bias=False)
        self.bn2 = nn.BatchNorm1d(300)
        self.fc3 = nn.Linear(300,300, bias=False)
        self.bn3 = nn.BatchNorm1d(300)
        self.fc4 = nn.Linear(300,300, bias=False)
        self.bn4 = nn.BatchNorm1d(300)
        self.fc5 = nn.Linear(300,1)

   def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.fc5(x)
        return x