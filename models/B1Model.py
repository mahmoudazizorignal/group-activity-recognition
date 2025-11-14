import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional
from helpers.config import get_settings
from models.enums import TensorBoardEnums


class B1Model(nn.Module):

    def __init__(self, resnet_pretrained: bool):
        super().__init__()
        self.settings = get_settings()

        # define the expected shape of the input of the model and the output of the model 
        self.input_size = (-1, self.settings.C, self.settings.H, self.settings.W)
        self.target_size = (-1,)

        # define the tensorboard path
        self.tensorboard_path = os.path.join(
            self.settings.TENSORBOARD_PATH,
            TensorBoardEnums.B1_TENSORBOARD_DIR.value,
        )

        # wether to use the pretrained one or initialize a random one
        if resnet_pretrained:
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.model = models.resnet50()

        # replace the original resnet-50 head
        self.model.fc = nn.Linear(
            in_features=2048, 
            out_features=self.settings.GROUP_ACTION_CNT, 
            bias=True
        )

        # initialize the new head of the model
        torch.nn.init.normal_(self.model.fc.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        x = x.view(self.input_size)

        logits = self.model(x)
        loss = None
        if y is not None:
            y = y.view(self.target_size)
            loss = F.cross_entropy(logits, y)
            
        return logits, loss
