import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Union, Tuple, List
from abc import ABC, abstractmethod
from helpers.config import Settings

class BaselinesInterface(ABC, nn.Module):
    
    def __init__(self, settings: Settings, resnet_pretrained: bool, 
                 base_finetuned: Optional[nn.Module] = None, base_freeze: Optional[bool] = True):
        super().__init__()
        self.settings = settings
        
        if base_finetuned:
            self.base = base_finetuned
            
            if base_freeze:
                # freeze the base model because its already finetuned
                for param in self.base.parameters():
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
    def forward(self, batch) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float], List[float]]:
        pass
