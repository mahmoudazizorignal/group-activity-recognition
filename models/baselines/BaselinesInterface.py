"""Abstract interface for baseline model providers.

This module defines `BaselinesInterface`, an abstract base class (and
`nn.Module`) that baseline providers implement. Implementations are
responsible for constructing the model components (feature extractor, LSTMs,
classifiers, metrics) and for providing a consistent `forward` signature.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Union, Tuple, List
from abc import ABC, abstractmethod
from helpers.config import Settings

class BaselinesInterface(ABC, nn.Module):
    """Abstract base class implemented by baseline model providers.

    Conforming classes must implement `forward(batch)` and should set up the
    model components (feature extractor or `base`, optional LSTM(s),
    classifier, and metric modules). The constructor handles either using a
    provided `base_finetuned` model or initializing a ResNet-50 feature
    extractor.

    Parameters
    ----------
    settings : Settings
        Project settings containing hyperparameters and device configuration.
    resnet_pretrained : bool
        Whether to initialize ResNet weights from ImageNet.
    base_finetuned : Optional[nn.Module]
        A pre-finetuned person model to reuse as a base (if provided).
    base_freeze : Optional[bool]
        Whether to freeze the supplied `base_finetuned` model's parameters.
    """
    
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
        """Run a forward pass and return outputs and metrics.

        Implementations should accept a dataset `batch` and return a tuple of
        four lists: (logits_list, losses_list, accuracies_list, f1_scores_list).

        Parameters
        ----------
        batch : Any
            Batch returned by the project's DataLoader(s). Concrete providers
            should document the expected shape for their `forward` input.

        Returns
        -------
        Tuple[List[torch.Tensor], List[torch.Tensor], List[float], List[float]]
            A 4-tuple containing lists of logits, losses, accuracies and F1
            scores (one element per model head where applicable).
        """
        pass
