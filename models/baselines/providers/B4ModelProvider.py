import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score
from helpers.config import Settings
from typing import Tuple
from models.baselines.BaselinesInterface import BaselinesInterface
from models.baselines.BaselinesEnums import TensorBoardEnums


class B4ModelProvider(BaselinesInterface):

    def __init__(self, settings: Settings, resnet_pretrained: bool):
        super().__init__(settings=settings, resnet_pretrained=resnet_pretrained, base_finetuned=None)

        # define the tensorboard path
        self.tensorboard_path = os.path.join(
            self.settings.TENSORBOARD_PATH,
            TensorBoardEnums.B4_TENSORBOARD_DIR.value,
        )

        # define lstm component
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=settings.NO_LSTM_HIDDEN_UNITS,
            num_layers=settings.NO_LSTM_LAYERS,
            batch_first=True,
            dropout=settings.LSTM_DROPOUT_RATE,
        )
        
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

    def forward(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # get the input and output of the batch and move it to the right device
        x, y = batch # x => (BATCH_SIZE, FRAME_CNT, C, H, W), y => (BATCH_SIZE, FRAME_CNT,)
        y = y[:, -1] # we only need one frame annotation per clip
        x, y = x.to(self.settings.DEVICE), y.to(self.settings.DEVICE)

        # do the forward path
        batch_size = x.shape[0]
        x = x.view(batch_size * self.settings.FRAME_CNT, self.settings.C, self.settings.H, self.settings.W)
        x1 = self.base(x).squeeze() # (BATCH_SIZE * FRAME_CNT, 2048)
        x1 = x1.view(batch_size, self.settings.FRAME_CNT, 2048) # (BATCH_SIZE, FRAME_CNT, 2048)
        x2, (_, _) = self.lstm(x1) # (BATCH_SIZE, FRAME_CNT, NO_LSTM_HIDDEN_UNITS)
        x = torch.concat([x1, x2], dim=2) # (BATCH_SIZE, FRAME_CNT, 2048 + NO_LSTM_HIDDEN_UNITS)
        x = x[:, -1, :] # (BATCH_SIZE, 2048 + NO_LSTM_HIDDEN_UNITS)
        logits = self.classifier(x) # (BATCH_SIZE, GROUP_ACTION_CNT)
        
        # calculate the cross entropy loss, accuracy, and f1-score of the batch
        loss = F.cross_entropy(logits, y)
        acc  = self.accuracy(logits.argmax(dim=1), y)
        f1   = self.f1_score(logits.argmax(dim=1), y)

        return logits, loss, acc, f1
