import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score
from helpers.config import Settings
from typing import Tuple, List
from models.baselines.BaselinesInterface import BaselinesInterface
from models.baselines.BaselinesEnums import TensorBoardEnums

class B1ModelProvider(BaselinesInterface):
    
    def __init__(self, settings: Settings, resnet_pretrained: bool):
        super().__init__(settings = settings, resnet_pretrained=resnet_pretrained, base_finetuned=None)

        # define classifier component
        self.classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Dropout(p=settings.HEAD_DROPOUT_RATE),
            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Dropout(p=settings.HEAD_DROPOUT_RATE),
            nn.Linear(in_features=512, out_features=settings.GROUP_ACTION_CNT)
        )

        # define the tensorboard path
        self.tensorboard_path = os.path.join(
            self.settings.TENSORBOARD_PATH,
            TensorBoardEnums.B1_TENSORBOARD_DIR.value,
        )
        
        # settings our evaluation metrics
        self.accuracy = Accuracy(
            task="multiclass", 
            num_classes=settings.GROUP_ACTION_CNT
        )
        
        self.f1_score = F1Score(
            task="multiclass", 
            num_classes=settings.GROUP_ACTION_CNT, 
            average="weighted"
        )

        # disable compiling for metrics calculation
        self.accuracy.forward = torch.compiler.disable(self.accuracy.forward)
        self.f1_score.forward = torch.compiler.disable(self.f1_score.forward)
    
    def forward(self, 
                batch: Tuple[torch.Tensor, torch.Tensor]
        ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float], List[float]]:
        # get the input and output of the batch and move it to the right device
        x, y = batch # x => (BATCH_SIZE, FRAME_CNT, C, H, W), y => (BATCH_SIZE, FRAME_CNT,)
        x, y = x.to(self.settings.DEVICE), y.to(self.settings.DEVICE)

        # reshape the batch
        x = x.view(-1, self.settings.C, self.settings.H, self.settings.W)
        y = y.view(-1,)

        # get the logits
        x      = self.resnet(x)
        logits = self.classifier(x)

        # calculate the cross entropy loss, accuracy, and f1-score of the batch
        loss = F.cross_entropy(logits, y)
        acc  = self.accuracy(logits.argmax(dim=1), y)
        f1   = self.f1_score(logits.argmax(dim=1), y)
            
        return [logits], [loss], [acc], [f1]
