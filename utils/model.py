import torch
from torch import nn

class Linear_nn(torch.nn.Module):
    def __init__(self):
        super(Linear_nn, self).__init__()
        self.linear1 = nn.Linear(400*400*3, 1200)
        self.linear2 = nn.Linear(1200, 120)
    
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x
    