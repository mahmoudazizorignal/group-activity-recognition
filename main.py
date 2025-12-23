import torch
from torch.utils.data import DataLoader
from helpers.config import get_settings
from models.lr.LREnums import LREnums
from models.baselines.providers import PersonModelProvider, B5ModelProvider, B6ModelProvider, B7ModelProvider
from models.baselines.BaselinesEnums import BaselinesEnums
from models.datasets import DatasetProviderFactory, DatasetEnums
from controllers import AnnotationController, TrainerController

if __name__ == "__main__":
    settings = get_settings()
    torch.manual_seed(settings.SEED)
    torch.cuda.manual_seed(settings.SEED)
    annotations = AnnotationController.get_annotations(settings=settings)
    
    
    
    
    
    
    import code; code.interact(local=locals())
    base = PersonModelProvider(settings=settings, resnet_pretrained=False, temporal=True)
    
    # prepare the datasets of the train and val
    train_dataset = DatasetProviderFactory(settings=settings, annotations=annotations).create(
        provider=DatasetEnums.DatasetEnums.PERSON,
        videos_split=settings.TRAIN_VIDEOS,
    )
    
    # prepare the train and val loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=settings.MINI_BATCH,
        shuffle=True,
    )
    
    # define our trainer
    trainer = TrainerController(
        baseline=BaselinesEnums.B8_MODEL,
        lr_scheduler=LREnums.COSINE,
        settings=settings,
        train_loader=train_loader,
        val_loader=train_loader,
        resnet_pretrained=False,
        base_finetuned=base,
        base_freeze=True,
        person_temporal=True,
        compile=False,
        group_only=False,
        tensorboard_track=False,
    )

    # fit the model
    model = trainer.fit()
