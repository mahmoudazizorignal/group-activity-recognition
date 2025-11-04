import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from helpers import Player, get_settings
from baselines import get_feature_extractor
from baselines import GroupLevelDataset


class B1ResNet50Model(nn.Module):

    def __init__(self, n_classes: int, pretrained: bool):
        super().__init__()
        self.n_classes = n_classes
        self.resnet50 = get_feature_extractor(n_classes=n_classes, pretrained=pretrained)
    
    def forward(self, x: torch.Tensor):
        x = self.resnet50(x)
        return x
 