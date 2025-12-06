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
        super().__init__(settings=settings, resnet_pretrained=resnet_pretrained, resnet_finetuned=None)

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
            proj_size=settings.GROUP_ACTION_CNT,
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
        x, y = batch # x => (Batch_Size, FRAME_CNT, C, H, W), y => (Batch_Size, FRAME_CNT,)
        x, y = x.to(self.settings.DEVICE), y.to(self.settings.DEVICE)

        # do the forward path
        x = self.resnet(x.view(-1, self.settings.C, self.settings.H, self.settings.W)).squeeze()
        logits, (_, _) = self.lstm(x.view(-1, self.settings.FRAME_CNT, 2048))
        
        # we only concerned with the final hidden state for instance
        logits = logits[:, -1, :] 
        y = y[:, -1]

        # calculate the cross entropy loss, accuracy, and f1-score of the batch
        loss = F.cross_entropy(logits, y)
        acc  = self.accuracy(logits.argmax(dim=1), y)
        f1   = self.f1_score(logits.argmax(dim=1), y)

        return logits, loss, acc, f1
