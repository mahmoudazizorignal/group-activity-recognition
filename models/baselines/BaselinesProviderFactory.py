import torch.nn as nn
from typing import Optional
from helpers.config import Settings
from models.baselines.providers import (B1ModelProvider, B3ModelProvider, 
                                        B4ModelProvider, PersonModelProvider,
                                        B5ModelProvider)
from models.baselines.BaselinesEnums import BaselinesEnums

class BaselinesProviderFactory:
    
    def __init__(self, settings: Settings):
        self.settings = settings
    
    def create(self, provider: BaselinesEnums, resnet_pretrained: bool, resnet_finetuned: Optional[nn.Module] = None):
        if provider.name == BaselinesEnums.B1_MODEL.name:
            return B1ModelProvider(
                settings=self.settings,
                resnet_pretrained=resnet_pretrained,
            )
        
        elif provider.name == BaselinesEnums.B3_MODEL.name:
            return B3ModelProvider(
                settings=self.settings,
                resnet_pretrained=resnet_pretrained,
                resnet_finetuned=resnet_finetuned,
            )

        elif provider.name == BaselinesEnums.B4_MODEL.name:
            return B4ModelProvider(
                settings=self.settings,
                resnet_pretrained=resnet_pretrained,
            )
        
        elif provider.name == BaselinesEnums.B5_MODEL.name:
            return B5ModelProvider(
                settings=self.settings,
                resnet_pretrained=resnet_pretrained,
            )
        
        elif provider.name == BaselinesEnums.PERSON_MODEL.name:
            return PersonModelProvider(
                settings=self.settings,
                resnet_pretrained=resnet_pretrained,
            )
