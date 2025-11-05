import os
import torch
from torch.utils.data import DataLoader
from helpers import Player, get_settings, Settings
from baselines import GroupLevelDataset, B1ResNet50Model, train


if __name__ == "__main__":
    
    settings = get_settings()

    torch.manual_seed(settings.SEED)
    torch.cuda.manual_seed(settings.SEED)

    group_train_dataset = GroupLevelDataset(
        videos_split=settings.TRAIN_VIDEOS,
        settings=settings,
    )

    group_val_dataset = GroupLevelDataset(
        videos_split=settings.VALIDATION_VIDEOS,
        settings=settings,
    )

    train_loader = DataLoader(
        dataset=group_train_dataset,
        batch_size=settings.MINI_BATCH,
        shuffle=True,
        pin_memory=bool(settings.PIN_MEMORY),
        num_workers=settings.NUM_WORKERS_TRAIN,
    )
    val_loader = DataLoader(
        dataset=group_val_dataset,
        batch_size=settings.MINI_BATCH,
        shuffle=False,
        pin_memory=bool(settings.PIN_MEMORY),
        num_workers=settings.NUM_WORKERS_EVAL,
    )

    torch.set_float32_matmul_precision(settings.MATMUL_PRECISION)
    model = B1ResNet50Model(n_classes=settings.GROUP_ACTION_CNT, pretrained=True).to(settings.DEVICE)
    save_path = os.path.join(os.path.dirname(__file__), "checkpoints")

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        settings=settings,
        save_path=save_path,
    )
