import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2


def get_processor():

    return v2.Compose([
        v2.Resize(size=(256, 256), interpolation=v2.InterpolationMode.BILINEAR),
        v2.CenterCrop(size=(224, 224)),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_feature_extractor(n_classes: int, pretrained: bool = False):

    if pretrained:
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        model = torchvision.models.resnet50()
    model.fc = nn.Linear(in_features=2048, out_features=n_classes, bias=True)
    torch.nn.init.normal_(model.fc.weight, mean=0.0, std=0.02)

    return model