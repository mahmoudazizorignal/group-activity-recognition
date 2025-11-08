import math
from tqdm.auto import tqdm
from typing import Any, Type

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from torchmetrics import Accuracy, F1Score
from helpers import get_settings
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
                 resnet_pretrained: bool):

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
            for xb, yb, in tqdm(self.val_loader):
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

        with SummaryWriter(log_dir=self.model.tensorboard_path) as writer:
            
            # getting one batch of the data for visualization
            one_batch = next(iter(self.train_loader))
            one_batch = one_batch.view(self.model.input_size)

            # visualize the structure of the model and its input
            writer.add_graph(
                model=self.model, 
                input_to_model=one_batch, 
                verbose=True,
            )

            # visualizing images in the batch
            writer.add_images(
                tag="batch_example",
                img_tensor=v2.ToDtype(torch.uint8, scale=True)(one_batch),
                global_step=0,
            )
            
            step = 0
            for epoch in range(self.settings.NUM_EPOCHS):

                running_loss, loss_accum, running_acc, running_f1 = 0.0, 0.0, 0.0, 0.0
                for i, (xb, yb) in tqdm(enumerate(self.train_loader)):

                    xb, yb = xb.to(self.settings.DEVICE), yb.to(self.settings.DEVICE)
                    with torch.autocast(device_type=self.settings.DEVICE, dtype=torch.bfloat16):
                        logits, loss = self.model(xb, yb)
                    loss.backward()

                    loss_accum += loss.item()

                    if (i + 1) % self.settings.GRAD_ACCUM_STEPS == 0 or i + 1 == len(self.train_loader):

                        # getting the updated learning rate
                        lr = self.scheduler.get_lr(step)

                        # tracking learning rate values
                        writer.add_scalar(
                            tag="lr_scheduler", 
                            scalar_value=lr, 
                            global_step=step,
                        )

                        # tracking the loss of each gradient accumulation step in the train
                        writer.add_scalar(
                            tag="grad_accum_steps/train/loss", 
                            scalar_value=loss_accum, 
                            global_step=step,
                        )

                        # zeroing loss accumulation after using it
                        loss_accum = 0.0

                        # updating our learning rate
                        for param in self.optimizer.param_groups:
                            param["lr"] = lr

                        # tracking the ditributions of the gradients in the model
                        for name, param in self.model.named_parameters():
                            if param is not None:
                                writer.add_histogram(
                                    tag=f"{name}.grad",
                                    values=param.grad,
                                    global_step=step,
                                )

                        # tracking the loss of each gradient accumulation step in the val
                        if (step + 1) % self.settings.EVAL_INTERVALS == 0:
                            val_accum_loss, _, _ = self.eval_model()
                            writer.add_scalar(
                                tag="grad_accum_steps/val/loss", 
                                scalar_value=val_accum_loss, 
                                global_step=step,
                            )

                        step += 1
                        
                        # cliping gradients to avoid exploading gradients
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                        # weights updates
                        self.optimizer.step()

                        # zeroing all the gradients
                        self.optimizer.zero_grad()    

                    running_loss += loss.item()
                    running_acc  += self.accuracy(logits.argmax(dim=1), yb).item()
                    running_f1   += self.f1_score(logits.argmax(dim=1), yb).item()

                running_loss /= len(self.train_loader)
                running_acc  /= len(self.train_loader)
                running_f1   /= len(self.train_loader)

                val_loss, val_acc, val_f1 = self.eval_model()

                # tracking losses and metrics values
                writer.add_scalar(tag="loss/train", scalar_value=running_loss, global_step=epoch)
                writer.add_scalar(tag="loss/val", scalar_value=val_loss, global_step=epoch)
                writer.add_scalar(tag="accuracy/train", scalar_value=running_acc, global_step=epoch)
                writer.add_scalar(tag="accuracy/val", scalar_value=val_acc, global_step=epoch)
                writer.add_scalar(tag="f1_score/train", scalar_value=running_f1, global_step=epoch)
                writer.add_scalar(tag="f1_score/val", scalar_value=val_f1, global_step=epoch)

                # wait for all cuda kernels to finish
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                print(f"Epoch [{epoch + 1}/{self.settings.NUM_EPOCHS}]: train_loss: {running_loss:.4f}, train_acc: {running_acc:.3f}, train_f1: {running_f1:.3f}, val_loss: {val_loss:.4f} val_acc: {val_acc:.3f}, val_f1: {val_f1:.3f}")

        return self.model
