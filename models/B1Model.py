import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional
from helpers.config import get_settings
from models.enums import TensorBoardEnums


class B1Model(nn.Module):

    def __init__(self, num_classes: int, resnet_pretrained: bool):
        super().__init__()
        self.settings = get_settings()
        self.num_classes = num_classes
        self.input_size = (
            -1, 
            self.settings.C, 
            self.settings.H, 
            self.settings.W
        )
        self.target_size = (-1,)
        self.tensorboard_path = os.path.join(
            self.settings.TENSORBOARD_PATH,
            TensorBoardEnums.B1_TENSORBOARD_DIR.value,
        )

        if resnet_pretrained:
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.model = models.resnet50()

        self.model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        torch.nn.init.normal_(self.model.fc.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        x = x.view(self.input_size)

        logits = self.model(x)
        loss = None
        if y is not None:
            y = y.view(self.target_size)
            loss = F.cross_entropy(logits, y)
            
        return logits, loss
