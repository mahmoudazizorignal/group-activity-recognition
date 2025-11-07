import math
from tqdm.auto import tqdm
from typing import Any, Type

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score
from helpers import get_settings, Settings
from controllers.enums import LREnums


class CosineLR:
    def __init__(self, max_steps, warmup_steps, max_lr, min_lr):
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr

    def get_lr(self, it: int) -> float:
        if it < self.warmup_steps:
            return self.max_lr * (it + 1) / self.warmup_steps
        if it > self.max_steps:
            return self.min_lr
        decay_ratio = (it - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)


class ExponentialLR:

    def __init__(self, initial_lr: float, beta: float):
        assert 0 <= beta <= 1.0, f"expected beta to be in range [0, 1], instead got: {beta}"
        self.lr = initial_lr
        self.beta = beta
    
    def get_lr(self, it=None) -> float:
        cur_lr = self.beta * self.lr
        self.lr *= self.beta
        return cur_lr

class IdentityLR:

    def __init__(self, initial_lr: float):
        self.lr = initial_lr

    def get_lr(self, it=None) -> float:
        return self.lr


class TrainerController:
    
    def __init__(self, 
                 Model: Type[nn.Module], 
                 train_loader: DataLoader, 
                 val_loader: DataLoader, 
                 num_classes: int,
                 resnet_pretrained: bool,
                 tensorboard_path: str):

        self.settings = get_settings()

        # sets the internal precision of float32 matrix multiplications.
        torch.set_float32_matmul_precision(self.settings.MATMUL_PRECISION)

        self.model = Model(num_classes=num_classes, resnet_pretrained=resnet_pretrained)
        self.model.compile()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            fused=torch.cuda.is_available()
        )

        # settings our evaluation metrics
        self.accuracy = Accuracy(
            task="multiclass", 
            num_classes=num_classes
        ).to(self.settings.DEVICE)

        self.f1_score = F1Score(
            task="multiclass", 
            num_classes=num_classes, 
            average="weighted"
        ).to(self.settings.DEVICE)

        self.train_loader = train_loader
        self.val_loader = val_loader

        if self.settings.LR_SCHEDULER == LREnums.COSINE_LR.value:
            self.scheduler = CosineLR(
                max_steps=self.settings.NUM_EPOCHS * len(train_loader) / self.settings.GRAD_ACCUM_STEPS,
                warmup_steps=self.settings.WARMUP_STEPS,
                max_lr=self.settings.MAX_LR,
                min_lr=self.settings.MIN_LR,
            )
        
        elif self.settings.LR_SCHEDULER == LREnums.EXPONENTIAL_LR.value:
            self.scheduler = ExponentialLR(
                initial_lr=self.settings.MAX_LR,
                beta=self.settings.BETA,
            )
        else:
            self.scheduler = IdentityLR(
                initial_lr=self.settings.MAX_LR,
            )
        
    def eval_model(self) -> tuple[float | Any, float | Any, float | Any]:
    
        self.model.eval()
        with torch.inference_mode():
            running_loss, running_acc, running_f1 = 0.0, 0.0, 0.0
            for xb, yb, in self.val_loader:
                xb, yb = xb.to(self.settings.DEVICE), yb.to(self.settings.DEVICE)
                
                with torch.autocast(device_type=self.settings.DEVICE, dtype=torch.bfloat16):
                    logits, loss = self.model(xb, yb)
                
                running_loss += loss.item()
                running_acc  += self.accuracy(logits.argmax(dim=1), yb).item()
                running_f1   += self.f1_score(logits.argmax(dim=1), yb).item()
            
            running_loss /= len(self.val_loader)
            running_acc  /= len(self.val_loader)
            running_f1   /= len(self.val_loader)

        self.model.train()
        return running_loss, running_acc, running_f1

    def fit(self) -> nn.Module:

        step = 0
        for epoch in range(self.settings.NUM_EPOCHS):

            running_loss, running_acc, running_f1 = 0.0, 0.0, 0.0
            for i, (xb, yb) in tqdm(enumerate(self.train_loader)):

                xb, yb = xb.to(self.settings.DEVICE), yb.to(self.settings.DEVICE)
                with torch.autocast(device_type=self.settings.DEVICE, dtype=torch.bfloat16):
                    logits, loss = self.model(xb, yb)
                loss.backward()

                if (i + 1) % self.settings.GRAD_ACCUM_STEPS == 0 or i + 1 == len(self.train_loader):

                    lr = self.scheduler.get_lr(step)
                    for param in self.optimizer.param_groups:
                        param["lr"] = lr

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    torch.cuda.synchronize()

                    if (step + 1) % self.settings.EVAL_STEPS:
                        val_loss, val_acc, val_f1 = self.eval_model()

                    step += 1     

                running_loss += loss.item()
                running_acc  += self.accuracy(logits, yb)
                running_f1   += self.f1_score(logits, yb)

            running_loss /= len(self.train_loader)
            running_acc  /= len(self.train_loader)
            running_f1   /= len(self.train_loader)

            print(f"Epoch [{epoch + 1}/{self.settings.NUM_EPOCHS}]: train_loss: {running_loss:.4f}, train_acc: {running_acc:.3f}, train_f1: {running_f1:.3f}, val_loss: {val_loss:.4f} val_acc: {val_acc:.3f}, val_f1: {val_f1:.3f}")

        return self.model
