import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional
import torchvision

class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
    
    
    
