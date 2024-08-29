import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import torchvision.models as models

class ViTModel(nn.Module):
    def __init__(self, num_classes, device, pretrained=True):
        super(ViTModel, self).__init__()
        self.model = models.vit_b_16(pretrained=pretrained)
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)
        
        self.model = self.model.to(device)
    
    def forward(self, x):
        return self.model(x)
    

    