from . import *
from .ViT import *
from .EfficientNet import *
from .MobileNet import *

MODEL_MAP = {
    'vit': ViTModel,
    'efficient_net': EfficientNet,
    'mobile_net': MobileNet
}

def get_model(model_name, num_classes, device, pretrained=True):
    if model_name not in MODEL_MAP:
        raise ValueError(f'Invalid model name: {model_name}, available models: {MODEL_MAP.keys()}')
    return MODEL_MAP[model_name](num_classes, device, pretrained)