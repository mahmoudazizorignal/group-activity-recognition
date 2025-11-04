import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from helpers import Player, get_settings
from baselines import GroupLevelDataset, B1ResNet50Model

torch.manual_seed(13)
torch.cuda.manual_seed(13)

device = "cuda" if torch.cuda.is_available() else "cpu"
settings = get_settings()
total_batch_size = 256
batch_size = 32
grad_accum_steps = total_batch_size // batch_size

max_lr = 0.003
min_lr = max_lr * 0.1
warmup_steps = 5
max_steps = 30
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay rate down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


if __name__ == "__main__":
    
    group_dataset = GroupLevelDataset(
        videos_split=settings.TRAIN_VIDEOS,
        settings=settings,
    )

    train_loader = DataLoader(
        dataset=group_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
    )
    iterator = iter(train_loader)

    torch.set_float32_matmul_precision('high')
    model = B1ResNet50Model(n_classes=settings.GROUP_ACTION_CNT, pretrained=True).to(device)
    accuracy = Accuracy(task="multiclass", num_classes=settings.GROUP_ACTION_CNT).to(device)
    model.compile()
    optimizer = torch.optim.AdamW(model.parameters(), fused=torch.cuda.is_available())

    for step in range(max_steps):
        start = time()
        optimizer.zero_grad()
        loss_accum = 0.0
        step_accuracy = 0.0
        for _ in range(grad_accum_steps):
        
            try:
                xb, yb = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                xb, yb = next(iterator)
            
            _, _, C, H, W = xb.shape
            xb = xb.view(-1, C, H, W).to(device)
            yb = yb.view(-1).to(device)

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
        
            loss /= grad_accum_steps
            step_accuracy += accuracy(logits.argmax(dim=1), yb) / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        
        lr = get_lr(step)
        for param in optimizer.param_groups:
            param["lr"] = lr

        optimizer.step()
        torch.cuda.synchronize()
        end = time()
        print(f"step: {step:4d} | lr: {lr:.4f} | loss: {loss_accum.item():.4f} | accuracy: {step_accuracy:.4f} | dt: {(end - start) * 1000: .2f}ms")

