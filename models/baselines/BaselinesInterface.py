import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Any, Tuple
from abc import ABC, abstractmethod
from helpers.config import Settings

class BaselinesInterface(ABC, nn.Module):
    
    def __init__(self, settings: Settings, resnet_pretrained: bool):
        super().__init__()
        
        self.settings = settings
        
        if resnet_pretrained:
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.resnet = models.resnet50()
        
        # remove the classification head of the resnet-50
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
    
    @abstractmethod
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Any]:
        pass
