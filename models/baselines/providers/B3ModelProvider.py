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

class B3ModelProvider(BaselinesInterface):
    
    def __init__(self, settings: Settings, base_finetuned: PersonModelProvider):
        super().__init__(
            settings = settings, 
            resnet_pretrained = False, 
            base_finetuned = base_finetuned,
        )
        # we only need the feature extractor that fine-tuned on person-level images
        assert not hasattr(self.base, "lstm"), f"the base model for {BaselinesEnums.B3_MODEL} cannot be temporal"
        if hasattr(self.base, "classifier"):
            self.base.classifier = None
            
        # define the max pooling layer
        self.pooler = nn.AdaptiveAvgPool2d(output_size=(1, 2048))
        
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

    def forward(self, 
                batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float], List[float]]:
        """Forward pass for B3.

        Expects grouped person crops and group annotations. Uses the provided
        person base to extract per-player features, pools across players and
        classifies the resulting group representation.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            x: (B, P, Fr, C, H, W), _ (unused person labels), y: (B, Fr)

        Returns
        -------
        Tuple[List[torch.Tensor], List[torch.Tensor], List[float], List[float]]
            ([logits], [loss], [acc], [f1])
        """
        # get the input and the group annotations only
        x, _, y = batch
        
        # move the right device
        x, y = x.to(self.settings.DEVICE), y.to(self.settings.DEVICE)
        
        # swap the frame count and player count
        x = x.permute(0, 2, 1, 3, 4, 5)
        x = x.reshape(-1, self.settings.PLAYER_CNT, self.settings.C, self.settings.H, self.settings.W)

        # extract the feature representations for person crops
        batch_size = x.shape[0]
        x = x.view(-1, self.settings.C, self.settings.H, self.settings.W)
        x = self.base.resnet(x)
        x = x.view(batch_size, self.settings.PLAYER_CNT, 2048)
        
        # max pool the features across all players: (B, P, 2048) => (B, 2048)
        x = self.pooler(x).squeeze()

        # get the logits
        logits = self.classifier(x)
        
        # calculate the cross entropy loss, accuracy, and f1-score of the batch
        loss  = F.cross_entropy(logits, y.view(-1))
        preds = logits.argmax(dim=1)
        acc   = self.accuracy(preds, y.view(-1))
        f1    = self.f1_score(preds, y.view(-1))
        
        return [logits], [loss], [acc], [f1]
