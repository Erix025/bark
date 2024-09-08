import torch.nn as nn
import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self, num_classes, device, pretrained=True):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = self.model.fc.in_features
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(in_features, num_classes)
        
        self.model = self.model.to(device)
    
    def forward(self, x):
        return self.model(x)
    
class ResNet101(nn.Module):
    def __init__(self, num_classes, device, pretrained=True):
        super(ResNet101, self).__init__()
        self.model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT if pretrained else None)
        in_features = self.model.fc.in_features
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(in_features, num_classes)

        self.model = self.model.to(device)
    
    def forward(self, x):
        return self.model(x)

class ResNet152(nn.Module):
    def __init__(self, num_classes, device, pretrained=True):
        super(ResNet152, self).__init__()
        self.model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT if pretrained else None)
        in_features = self.model.fc.in_features
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(in_features, num_classes)
        
        self.model = self.model.to(device)
    
    def forward(self, x):
        return self.model(x)
    
class ResNet18(nn.Module):
    def __init__(self, num_classes, device, pretrained=True):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
        self.model = self.model.to(device)
    
    def forward(self, x):
        return self.model(x)

    