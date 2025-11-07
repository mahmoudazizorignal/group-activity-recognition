import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional


class B1Model(nn.Module):

    def __init__(self, num_classes: int, resnet_pretrained: bool):
        super().__init__()
        self.num_classes = num_classes

        if resnet_pretrained:
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.model = models.resnet50()

        self.model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        torch.nn.init.normal_(self.model.fc.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor]):
        _, _, C, H, W = x.shape
        x = x.view(-1, C, H, W)
        
        logits = self.model(x)
        loss = None
        if y is not None:
            y = y.view(-1)
            loss = F.cross_entropy(logits, y)
            
        return logits, loss
 