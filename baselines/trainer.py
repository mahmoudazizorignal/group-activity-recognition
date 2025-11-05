import os
import math
import json
from time import time
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score
from helpers import Settings


def get_lr(it: int, settings: Settings):
    # 1) linear warmup for warmup_iters steps
    if it < settings.WARMUP_STEPS:
        return settings.MAX_LR * (it + 1) / settings.WARMUP_STEPS
    # 2) if it > lr_decay_iters, return min learning rate
    if it > settings.MAX_LR:
        return settings.MIN_LR
    # 3) in between, use cosine decay rate down to min learning rate
    decay_ratio = (it - settings.WARMUP_STEPS) / (settings.MAX_STEPS - settings.WARMUP_STEPS)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return settings.MIN_LR + coeff * (settings.MAX_LR - settings.MIN_LR)



def eval_model(model, loader: DataLoader, settings: Settings):
    
    model.eval()
    accuracy = Accuracy(task="multiclass", num_classes=settings.GROUP_ACTION_CNT).to(settings.DEVICE)
    f1_score = F1Score(task="multiclass", num_classes=settings.GROUP_ACTION_CNT, average="weighted").to(settings.DEVICE)
    
    with torch.inference_mode():
        total_batches = 0
        acc, f1, loss = 0.0, 0.0, 0.0
        for xb, yb, in tqdm(loader):
            _, _, C, H, W = xb.shape
            xb = xb.view(-1, C, H, W).to(settings.DEVICE)
            yb = yb.view(-1).to(settings.DEVICE)
            logits = model(xb)
            loss += F.cross_entropy(logits, yb).item()
            acc += accuracy(logits.argmax(dim=1), yb).item()
            f1 += f1_score(logits.argmax(dim=1), yb).item()
            total_batches += 1
        
        loss /= total_batches
        acc /= total_batches
        f1 /= total_batches

    model.train()
    return loss, acc, f1


def train(model, train_loader: DataLoader, val_loader: DataLoader, settings: Settings, save_path: str):
    
    model.compile()
    accuracy = Accuracy(task="multiclass", num_classes=settings.GROUP_ACTION_CNT).to(settings.DEVICE)
    f1_score = F1Score(task="multiclass", num_classes=settings.GROUP_ACTION_CNT, average="weighted").to(settings.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), fused=torch.cuda.is_available())
    iterator = iter(train_loader)

    training_args = settings.model_dump()
    training_args["train_loss"] = []
    training_args["train_accuracy"] = []
    training_args["train_f1"] = []
    training_args["val_loss"] = []
    training_args["eval_accuracy"] = []
    training_args["eval_f1"] = []
    
    for step in range(settings.MAX_STEPS):
        start = time()
        optimizer.zero_grad()
        loss_accum = 0.0
        step_accuracy = 0.0
        step_f1 = 0.0
        for _ in range(settings.GRAD_ACCUM_STEPS):
        
            try:
                xb, yb = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                xb, yb = next(iterator)
            
            _, _, C, H, W = xb.shape
            xb = xb.view(-1, C, H, W).to(settings.DEVICE)
            yb = yb.view(-1).to(settings.DEVICE)

            with torch.autocast(device_type=settings.DEVICE, dtype=torch.bfloat16):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
        
            loss /= settings.GRAD_ACCUM_STEPS
            step_accuracy += accuracy(logits.argmax(dim=1), yb).item() / settings.GRAD_ACCUM_STEPS
            step_f1 += f1_score(logits.argmax(dim=1), yb).item() / settings.GRAD_ACCUM_STEPS
            loss_accum += loss.detach()
            loss.backward()

        lr = get_lr(step, settings)
        for param in optimizer.param_groups:
            param["lr"] = lr

        optimizer.step()
        torch.cuda.synchronize()
        end = time()

        training_args["train_loss"].append(loss_accum.item())
        training_args["train_accuracy"].append(step_accuracy)
        training_args["train_f1"].append(step_f1)
        print(f"step: {step + 1:4d} | lr: {lr:.4f} | loss: {loss_accum.item():.4f} | accuracy: {step_accuracy:.4f} | f1: {step_f1:.4f} | dt: {end - start: .4f}sec")

        if (step + 1) % settings.EVAL_STEPS == 0:
            start = time()
            eval_loss, eval_acc, eval_f1 = eval_model(model, val_loader, settings)
            torch.cuda.synchronize()
            end = time()

            training_args["val_loss"].append(eval_loss)
            training_args["eval_accuracy"].append(eval_acc)
            training_args["eval_f1"].append(eval_f1)
            print(f"==> eval_loss: {eval_loss:.4f} | eval_acc: {eval_acc:.4f} | eval_f1: {eval_f1:.4f} | dt: {end - start:.4f}sec")

    # saving checkpoint
    exp_folder = time()
    save_path = os.path.join(save_path, f"{exp_folder}")
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "results.json"), "w") as file:
        json.dump(training_args, file)

    torch.save(
        obj=model.state_dict(), 
        f=os.path.join(save_path, "model.pth"),
    )
