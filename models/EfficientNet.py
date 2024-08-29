import torch.nn as nn
import torchvision.models as models

class EfficientNet(nn.Module):
    def __init__(self, num_classes, device, pretrained=True):
        super(EfficientNet, self).__init__()
        self.model = models.efficientnet_v2_m(pretrained=pretrained)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
        
        self.model = self.model.to(device)
    
    def forward(self, x):
        return self.model(x)
    

    