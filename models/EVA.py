import timm
from torch import nn

class EVA02(nn.Module):
    def __init__(self, num_classes, device, pretrained=True):
        super(EVA02, self).__init__()
        self.model = timm.create_model(
            "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
            pretrained=pretrained,
            num_classes=num_classes,
        )

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.head.weight.requires_grad = True
        self.model.head.bias.requires_grad = True
        
        self.model = self.model.to(device)
        
    def forward(self, x):
        if isinstance(x, list):
            x = x[0]
        return self.model(x)