import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers.config import Settings
from typing import Optional, Any, Tuple
from models.baselines.BaselinesInterface import BaselinesInterface
from models.baselines.BaselinesEnums import TensorBoardEnums

class B3ModelProvider(BaselinesInterface):
    
    def __init__(self, settings: Settings, resnet_pretrained: bool):
        super().__init__(settings = settings,resnet_pretrained=resnet_pretrained)
        
        # define the expected shape of the input of the model and the output of the model 
        self.input_size = (-1, self.settings.C, self.settings.H, self.settings.W)
        self.target_size = (-1,)
        
        # define the tensorboard path
        self.tensorboard_path = os.path.join(
            self.settings.TENSORBOARD_PATH,
            TensorBoardEnums.B3_TENSORBOARD_DIR.value,
        )
        
        # define the architecture of b3-model
        self.model = nn.ModuleDict(dict(
            resnet=self.resnet,
            head=nn.Linear(in_features=2048, out_features=self.settings.PLAYER_ACTION_CNT)
        ))
        
        # initialize the new head of the model
        torch.nn.init.normal_(self.model.fc.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Any]:
        x = x.view(self.input_size)

        logits = self.model(x)
        loss = None
        if y is not None:
            y = y.view(self.target_size)
            loss = F.cross_entropy(logits, y)
            
        return logits, loss
