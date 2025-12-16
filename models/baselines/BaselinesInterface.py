import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Any, Tuple
from abc import ABC, abstractmethod
from helpers.config import Settings

class BaselinesInterface(ABC, nn.Module):
    
    def __init__(self, settings: Settings, resnet_pretrained: bool, base_finetuned: Optional[nn.Module] = None):
        super().__init__()
        self.settings = settings
        
        if base_finetuned:
            self.base = base_finetuned
            
            # freeze the base model because its already finetuned
            for param in self.base.parameters():
                param.requires_grad = False
                
        else:
            if resnet_pretrained:
                # get the pretrained resnet-50
                self.base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            else:
                # get resnet-50 with random weights
                self.base = models.resnet50()
                
            # remove the classification head of the resnet-50
            self.base = nn.Sequential(*(list(self.base.children())[:-1]))
    
    @abstractmethod
    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        pass
