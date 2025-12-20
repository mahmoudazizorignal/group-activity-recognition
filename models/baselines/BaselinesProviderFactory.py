import torch.nn as nn
from typing import Optional, Union
from helpers.config import Settings
from models.baselines.providers import (B1ModelProvider, B3ModelProvider, 
                                        B4ModelProvider, PersonModelProvider,
                                        B5ModelProvider, B6ModelProvider, 
                                        B7ModelProvider)
from models.baselines.BaselinesEnums import BaselinesEnums

class BaselinesProviderFactory:
    
    def __init__(self, settings: Settings):
        self.settings = settings
    
    def create(
        self, 
        provider: BaselinesEnums, 
        resnet_pretrained: bool, 
        base_finetuned: Optional[PersonModelProvider] = None,
        base_freeze: bool = True,
        temporal: bool = True,
    ):
        if provider.name == BaselinesEnums.B1_MODEL.name:
            return B1ModelProvider(
                settings=self.settings,
                resnet_pretrained=resnet_pretrained,
            )
        
        elif provider.name == BaselinesEnums.B3_MODEL.name:
            assert base_finetuned, f"base model cannot be None for {provider.name}"
            return B3ModelProvider(
                settings=self.settings,
                base_finetuned=base_finetuned,
            )

        elif provider.name == BaselinesEnums.B4_MODEL.name:
            return B4ModelProvider(
                settings=self.settings,
                resnet_pretrained=resnet_pretrained,
            )
        
        elif provider.name == BaselinesEnums.B5_MODEL.name:
            assert base_finetuned, f"base model cannot be None for {provider.name}"
            return B5ModelProvider(
                settings=self.settings,
                base_finetuned=base_finetuned,
                base_freeze=base_freeze,
            )
        
        elif provider.name == BaselinesEnums.B6_MODEL.name:
            assert base_finetuned, f"base model cannot be None for {provider.name}"
            return B6ModelProvider(
                settings=self.settings,
                base_finetuned=base_finetuned,
            )
        
        elif provider.name == BaselinesEnums.B7_MODEL.name:
            assert base_finetuned, f"base model cannot be None for {provider.name}"
            return B7ModelProvider(
                settings=self.settings,
                base_finetuned=base_finetuned,
            )
            
        elif provider.name == BaselinesEnums.PERSON_MODEL.name:
            return PersonModelProvider(
                settings=self.settings,
                resnet_pretrained=resnet_pretrained,
                temporal=temporal,
            )
