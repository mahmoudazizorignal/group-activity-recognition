import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score
from helpers.config import Settings
from typing import Tuple, Optional, Union
from models.baselines.providers import PersonModelProvider
from models.baselines.BaselinesInterface import BaselinesInterface
from models.baselines.BaselinesEnums import TensorBoardEnums

class B5ModelProvider(BaselinesInterface):
    def __init__(self, settings: Settings, base_finetuned: PersonModelProvider):
        super().__init__(settings=settings, resnet_pretrained=False, base_finetuned=base_finetuned)
        
        if hasattr(self.base, "classifier"):
            self.base.classifier = None
        
        # define the tensorboard path
        self.tensorboard_path = os.path.join(
            self.settings.TENSORBOARD_PATH,
            TensorBoardEnums.B5_TENSORBOARD_DIR.value,
        )
        
        # define the max pooling layer
        self.pooler = nn.AdaptiveAvgPool2d(output_size=(1, settings.NO_LSTM_HIDDEN_UNITS + 2048))
        
        # define classifier component
        self.classifier = nn.Sequential(
            nn.Linear(in_features=2048 + settings.NO_LSTM_HIDDEN_UNITS, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Dropout(p=settings.HEAD_DROPOUT_RATE),
            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Dropout(p=settings.HEAD_DROPOUT_RATE),
            nn.Linear(in_features=512, out_features=settings.GROUP_ACTION_CNT),
        )
        
        # settings our evaluation metrics
        self.accuracy = Accuracy(
            task="multiclass", 
            num_classes=settings.GROUP_ACTION_CNT,
        )
        
        self.f1_score = F1Score(
            task="multiclass", 
            num_classes=settings.GROUP_ACTION_CNT, 
            average="weighted",
        )

        # disable compiling for metrics calculation
        self.accuracy.forward = torch.compiler.disable(self.accuracy.forward)
        self.f1_score.forward = torch.compiler.disable(self.f1_score.forward)
    
    def forward(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # get the input and the group annotations only
        x, _, y = batch # x => (B, P, Fr, C, H, W), y => (B, Fr)
        B, P, Fr, C, H, W = x.shape
        
        # move to the right device
        y = y[:, -1] # we only need the final frame for each clip
        x, y = x.to(self.settings.DEVICE), y.to(self.settings.DEVICE)
        
        # extract feature representation for each player in each frame
        x = x.view(B * P * Fr, C, H, W)
        x1 = self.base.base(x)
        x1 = x1.view(B * P, F, 2048)
        
        # apply the features to lstm 
        x2, (_, _) = self.base.lstm(x1) # (B * P, Fr, H)
        x = torch.concat([x1, x2], dim=2) # (B * P, Fr, 2048 + H)
        x = x[:, -1, :] # (B * P, 2048 + H)
        x = x.view(B, P, 2048 + self.settings.NO_LSTM_HIDDEN_UNITS)
        
        # max pooling all the players in a clip
        x = self.pooler(x).view(B, 2048 + self.settings.NO_LSTM_HIDDEN_UNITS) # (B, 2048 + H)
        
        # apply the classifier
        logits = self.classifier(x)
        
        # calculate the cross entropy loss, accuracy, and f1-score of the batch
        loss = F.cross_entropy(logits, y)
        acc  = self.accuracy(logits.argmax(dim=1), y)
        f1   = self.f1_score(logits.argmax(dim=1), y)
        
        return logits, loss, acc, f1
