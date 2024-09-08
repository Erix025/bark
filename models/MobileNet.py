import torch.nn as nn
import torchvision.models as models

class MobileNetV2(nn.Module):
    def __init__(self, num_classes, device, pretrained=True):
        super(MobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Linear(in_features, num_classes),
        )
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier[1].parameters():
            param.requires_grad = True
        
        self.model = self.model.to(device)
    
    def forward(self, x):
        return self.model(x)
    
class MobileNetV3(nn.Module):
    def __init__(self, num_classes, device, pretrained=True):
        super(MobileNetV3, self).__init__()
        self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, num_classes)
        self.model = self.model.to(device)
        
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier[3].parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.model(x)
    

    