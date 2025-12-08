import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score
from helpers.config import Settings
from typing import Optional, Any, Tuple
from models.baselines.BaselinesInterface import BaselinesInterface
from models.baselines.BaselinesEnums import TensorBoardEnums

class B3ModelProvider(BaselinesInterface):
    
    def __init__(self, settings: Settings, resnet_pretrained: bool, resnet_finetuned: Optional[nn.Module] = None):
        super().__init__(
            settings = settings, 
            resnet_pretrained = resnet_pretrained, 
            resnet_finetuned = resnet_finetuned,
        )

        # define the classifier architecture
        self.pooler = nn.AdaptiveMaxPool1d(output_size=1)
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
            TensorBoardEnums.B3_TENSORBOARD_DIR.value,
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

    def forward(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # get the input and the group annotations only
        x, _, y = batch
        
        # swap the frame count and player count
        x = x.permute(0, 2, 1, 3, 4, 5).contiguous().view(
            -1, 
            self.settings.PLAYER_CNT, 
            self.settings.C,
            self.settings.H,
            self.settings.W,
        )

        # move the right device
        x, y = x.to(self.settings.DEVICE), y.to(self.settings.DEVICE)

        # extract the feature representations for person crops
        x = self.resnet(x.view(-1, self.settings.C, self.settings.H, self.settings.W)).squeeze().view(
            x.shape[0],
            self.settings.PLAYER_CNT, 
            -1
        )
        
        # max pool the features across all players: (B, P, 2048) => (B, 2048)
        x = self.pooler(x.permute(0, 2, 1)).squeeze()

        # get the logits
        logits = self.classifier(x)
        
        # calculate the cross entropy loss, accuracy, and f1-score of the batch
        loss = F.cross_entropy(logits, y.view(-1))
        acc  = self.accuracy(logits.argmax(dim=1), y.view(-1))
        f1   = self.f1_score(logits.argmax(dim=1), y.view(-1))
        
        return logits, loss, acc, f1
