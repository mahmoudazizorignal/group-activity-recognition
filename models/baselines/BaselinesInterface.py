import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Any, Tuple
from abc import ABC, abstractmethod
from helpers.config import Settings

class BaselinesInterface(ABC, nn.Module):
    
    def __init__(self, settings: Settings, resnet_pretrained: bool, resnet_finetuned: Optional[nn.Module] = None):
        super().__init__()
        self.settings = settings
        
        if resnet_finetuned:
            self.resnet = resnet_finetuned
            
            # freeze the resnet-50 because its already finetuned
            for param in self.resnet.parameters():
                param.requires_grad = False
                
        else:
            if resnet_pretrained:
                # get the pretrained resnet-50
                self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            else:
                # get resnet-50 with random weights
                self.resnet = models.resnet50()
                
            # remove the classification head of the resnet-50
            self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
    
    @abstractmethod
    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        pass
