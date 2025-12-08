import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score
from helpers.config import Settings
from typing import Tuple
from models.baselines.BaselinesInterface import BaselinesInterface
from models.baselines.BaselinesEnums import TensorBoardEnums

class PersonModelProvider(BaselinesInterface):
    
    def __init__(self, settings: Settings, resnet_pretrained: bool):
        super().__init__(settings = settings,resnet_pretrained=resnet_pretrained, resnet_finetuned=None)
        
        # define the tensorboard path
        self.tensorboard_path = os.path.join(
            self.settings.TENSORBOARD_PATH,
            TensorBoardEnums.PERSON_TENSORBOARD_DIR.value,
        )

        # define the architecture of the classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Dropout(p=settings.HEAD_DROPOUT_RATE),
            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Dropout(p=settings.HEAD_DROPOUT_RATE),
            nn.Linear(in_features=512, out_features=settings.PLAYER_ACTION_CNT)
        )
        
        # settings our evaluation metrics
        self.accuracy = Accuracy(
            task="multiclass", 
            num_classes=settings.PLAYER_ACTION_CNT
        )
        
        self.f1_score = F1Score(
            task="multiclass", 
            num_classes=settings.PLAYER_ACTION_CNT, 
            average="weighted"
        )

        # disable compiling for metrics calculation
        self.accuracy.forward = torch.compiler.disable(self.accuracy.forward)
        self.f1_score.forward = torch.compiler.disable(self.f1_score.forward)

    def forward(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # get the input and output of the batch and move it to the right device
        x, y, _ = batch
        x, y = x.to(self.settings.DEVICE), y.to(self.settings.DEVICE)

        # reshape the batch
        x = x.view(-1, self.settings.C, self.settings.H, self.settings.W)
        y = y.view(-1,)

        # get the logits
        logits = self.classifier(self.resnet(x).squeeze())
        
        # calculate the cross entropy loss, accuracy, and f1-score of the batch
        loss = F.cross_entropy(logits, y)
        acc  = self.accuracy(logits.argmax(dim=1), y)
        f1   = self.f1_score(logits.argmax(dim=1), y)
            
        return logits, loss, acc, f1
