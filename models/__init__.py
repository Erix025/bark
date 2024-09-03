from . import *
from .ViT import *
from .EfficientNet import *
from .MobileNet import *
from .ResNet import *
MODEL_MAP = {
    'vit': ViTModel,
    'efficient_net': EfficientNet,
    'mobile_net_v2': MobileNetV2,
    'mobile_net_v3': MobileNetV3,
    'resnet152': ResNet152,
    'resnet101': ResNet101,
    'resnet50': ResNet50,
    'resnet18': ResNet18,
}

def get_model(model_name, num_classes, device, pretrained=True):
    if model_name not in MODEL_MAP:
        raise ValueError(f'Invalid model name: {model_name}, available models: {MODEL_MAP.keys()}')
    return MODEL_MAP[model_name](num_classes, device, pretrained)