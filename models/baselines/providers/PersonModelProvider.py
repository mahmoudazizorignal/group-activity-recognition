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
    
    def __init__(self, settings: Settings, resnet_pretrained: bool, temporal: bool = True):
        super().__init__(settings = settings, resnet_pretrained=resnet_pretrained, base_finetuned=None)
        
        # define the tensorboard path
        self.tensorboard_path = os.path.join(
            self.settings.TENSORBOARD_PATH,
            TensorBoardEnums.PERSON_TENSORBOARD_DIR.value,
        )
        self.lstm = None
        
        if temporal:
            # define lstm component
            self.lstm = nn.LSTM(
                input_size=2048,
                hidden_size=settings.NO_LSTM_HIDDEN_UNITS,
                num_layers=settings.NO_LSTM_LAYERS,
                batch_first=True,
                dropout=settings.LSTM_DROPOUT_RATE,
            )

        # define the architecture of the classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features=2048 + settings.NO_LSTM_HIDDEN_UNITS * temporal, out_features=1024),
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
        # get the input and the group annotations only
        x, y, _ = batch # x => (B, P, F, C, H, W), y => (B, P, F)
        B, P, Fr, C, H, W = x.shape
        y = y.view(B * P * Fr, )
        
        # move to the right device
        x, y = x.to(self.settings.DEVICE), y.to(self.settings.DEVICE)
        
        # extract feature representation for each player in each frame
        x = x.view(B * P * Fr, C, H, W)
        x1 = self.base(x)
        x1 = x1.view(B * P, Fr, 2048)
        
        if self.lstm:
            # apply the features to lstm         
            x2, (_, _) = self.lstm(x1) # (B * P, F, H)
            x = torch.concat([x1, x2], dim=2) # (B * P, F, 2048 + H)
            x = x.view(B * P * Fr, 2048 + self.settings.NO_LSTM_HIDDEN_UNITS)
        else:
            # else the model is not temporal
            x = x1
        
        # apply the classifier
        logits = self.classifier(x)
        
        # calculate the cross entropy loss, accuracy, and f1-score of the batch
        loss = F.cross_entropy(logits, y)
        acc  = self.accuracy(logits.argmax(dim=1), y)
        f1   = self.f1_score(logits.argmax(dim=1), y)
        return logits, loss, acc, f1
