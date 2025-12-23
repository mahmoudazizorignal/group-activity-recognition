"""Factory for baseline model providers.

This module exposes `BaselinesProviderFactory`, a small factory that creates
concrete baseline model provider instances (e.g., B1, B3, Person) based on a
`BaselinesEnums` value. The created providers implement the project's
`BaselinesInterface` and are initialized with the project's `Settings`.
"""

import torch.nn as nn
from typing import Optional, Union
from helpers.config import Settings
from models.baselines.providers import (B1ModelProvider, B3ModelProvider, 
                                        B4ModelProvider, PersonModelProvider,
                                        B5ModelProvider, B6ModelProvider, 
                                        B7ModelProvider, B8ModelProvider)
from models.baselines.BaselinesEnums import BaselinesEnums

class BaselinesProviderFactory:
    """Create baseline model provider instances.

    The factory centralizes construction logic for baseline models. It selects
    the correct provider implementation based on a `BaselinesEnums` value and
    forwards the relevant initialization parameters.
    """
    
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
        """Instantiate the provider requested by `provider`.

        Parameters
        ----------
        provider : BaselinesEnums
            Enum value identifying which provider to create.
        resnet_pretrained : bool
            Whether to initialize ResNet backbones with ImageNet weights.
        base_finetuned : Optional[PersonModelProvider]
            A previously fine-tuned person model used as a base where required.
        base_freeze : bool
            Whether a provided base model should be frozen.
        temporal : bool
            Whether temporal components should be enabled for person models.

        Returns
        -------
        BaselinesInterface
            A concrete provider instance implementing the baseline model.
        """
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
                base_freeze=base_freeze,
            )
        
        elif provider.name == BaselinesEnums.B7_MODEL.name:
            assert base_finetuned, f"base model cannot be None for {provider.name}"
            return B7ModelProvider(
                settings=self.settings,
                base_finetuned=base_finetuned,
                base_freeze=base_freeze,
            )
            
        elif provider.name == BaselinesEnums.B8_MODEL.name:
            assert base_finetuned, f"base model cannot be None for {provider.name}"
            return B8ModelProvider(
                settings=self.settings,
                base_finetuned=base_finetuned,
                base_freeze=base_freeze,
            )
            
        elif provider.name == BaselinesEnums.PERSON_MODEL.name:
            return PersonModelProvider(
                settings=self.settings,
                resnet_pretrained=resnet_pretrained,
                temporal=temporal,
            )
