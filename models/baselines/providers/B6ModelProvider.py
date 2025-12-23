import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score
from helpers.config import Settings
from typing import Tuple, List
from models.baselines.providers import PersonModelProvider
from models.baselines.BaselinesInterface import BaselinesInterface
from models.baselines.BaselinesEnums import TensorBoardEnums, BaselinesEnums

class B6ModelProvider(BaselinesInterface):
    
    def __init__(self, settings: Settings, base_finetuned: PersonModelProvider, base_freeze: bool):
        super().__init__(
            settings = settings, 
            resnet_pretrained = False, 
            base_finetuned = base_finetuned,
            base_freeze=base_freeze,
        )
        # we only need the feature extractor that fine-tuned on person-level images
        assert not hasattr(self.base, "lstm"), f"the base model for {BaselinesEnums.B6_MODEL} cannot be temporal"
        
        # define the max pooling layer
        self.pooler = nn.AdaptiveAvgPool2d(output_size=(1, 2048))
        
        # define lstm component
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=settings.NO_LSTM_HIDDEN_UNITS2,
            num_layers=settings.NO_LSTM_LAYERS2,
            batch_first=True,
            dropout=settings.LSTM_DROPOUT_RATE2,
        )
        
        # define classifier component
        self.classifier = nn.Sequential(
            nn.Linear(in_features=2048 + settings.NO_LSTM_HIDDEN_UNITS2, out_features=1024),
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
            TensorBoardEnums.B6_TENSORBOARD_DIR.value,
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
                batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float], List[float]]:
        """Forward pass for B6.

        Processes person crops per-frame to produce both player-level and
        group-level predictions via pooling and LSTM-based temporal fusion.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            x: (B, P, Fr, C, H, W), y1: (B, P, Fr), y2: (B, Fr)

        Returns
        -------
        Tuple[List[torch.Tensor], List[torch.Tensor], List[float], List[float]]
            ([logits1, logits2], [loss1, loss2], [accs], [f1s])
        """
        # get the input and the group annotations only
        x, y1, y2 = batch # x => (B, P, Fr, C, H, W), y1 => (B, P, Fr), y2 => (B, Fr)
        B, P, Fr, C, H, W = x.shape
        
        # move to the right device
        y1 = y1.view(B * P * Fr,)
        y2 = y2[:, -1] # we only need the final frame for each clip
        x, y1, y2 = x.to(self.settings.DEVICE), y1.to(self.settings.DEVICE), y2.to(self.settings.DEVICE)
        
        # extract feature representation for each player in each frame
        x = x.view(B * P * Fr, C, H, W)
        x1 = self.base.resnet(x)
        x1 = x1.view(B, P, Fr, 2048)
        
        # get the logits of the player activities
        logits1 = self.base.classifier(x1.view(B * P * Fr, 2048))
        
        # max pooling all the players in a clip
        x1 = x1.permute(0, 2, 1, 3) # (B, Fr, P, 2048)
        x1 = self.pooler(x1).view(B, Fr, 2048) # (B, Fr, 2048)
        
        # apply the features to lstm 
        x2, (_, _) = self.lstm(x1) # (B, Fr, Hi2)
        x = torch.concat([x1, x2], dim=2) # (B, Fr, 2048 + Hi2)
        x = x[:, -1, :] # (B, Hi2)
        
        # apply the classifier
        logits2 = self.classifier(x)
        
        # calculate the cross entropy loss, accuracy, and f1-score of the batch
        logits = [logits1, logits2]
        losses = [F.cross_entropy(logits1, y1), F.cross_entropy(logits2, y2)]
        accs  = [self.base.accuracy(logits1.argmax(dim=1), y1), self.accuracy(logits2.argmax(dim=1), y2)]
        f1s  = [self.base.f1_score(logits1.argmax(dim=1), y1), self.f1_score(logits2.argmax(dim=1), y2)]
        
        return logits, losses, accs, f1s
