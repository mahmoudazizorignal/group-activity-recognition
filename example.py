import torch
from torch.utils.data import DataLoader
from helpers.config import get_settings
from models.lr.LREnums import LREnums
from models.baselines.providers import PersonModelProvider, B8ModelProvider
from models.baselines.BaselinesEnums import BaselinesEnums
from models.datasets import DatasetProviderFactory, DatasetEnums
from controllers import AnnotationController, TrainerController

if __name__ == "__main__":
    # get the configuration set in the `.env` file
    settings = get_settings()

    # set the seed for reproducibility
    torch.manual_seed(settings.SEED)
    torch.cuda.manual_seed(settings.SEED)

    # process the annotations and get them
    annotator = AnnotationController(settings=settings)
    annotations = annotator.process_annotations()
    # annotator.save_annotations() # you can optionally save them so you can process them only once

    # prepare the datasets of the train and val
    train_dataset = DatasetProviderFactory(settings=settings, annotations=annotations).create(
        provider=DatasetEnums.DatasetEnums.PERSON,
        videos_split=settings.TRAIN_VIDEOS,
    )
    val_dataset = DatasetProviderFactory(settings=settings, annotations=annotations).create(
        provider=DatasetEnums.DatasetEnums.PERSON,
        videos_split=settings.VALIDATION_VIDEOS,
    )
    test_dataset = DatasetProviderFactory(settings=settings, annotations=annotations).create(
        provider=DatasetEnums.DatasetEnums.PERSON,
        videos_split=settings.TEST_VIDEOS,
    )


    # prepare the train and val loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=settings.MINI_BATCH,
        shuffle=True,
        pin_memory=bool(settings.PIN_MEMORY),
        num_workers=settings.NUM_WORKERS_TRAIN,
        persistent_workers=bool(settings.PERSISTANT_WORKERS),
        drop_last=True,  # to avoid batch norm errors
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=settings.MINI_BATCH,
        shuffle=False,
        pin_memory=bool(settings.PIN_MEMORY),
        num_workers=settings.NUM_WORKERS_EVAL,
        persistent_workers=bool(settings.PERSISTANT_WORKERS),
        drop_last=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=settings.MINI_BATCH,
        shuffle=False,
        pin_memory=bool(settings.PIN_MEMORY),
        num_workers=settings.NUM_WORKERS_EVAL,
        persistent_workers=bool(settings.PERSISTANT_WORKERS),
        drop_last=True,
    )

    # setup the base model
    base_model = PersonModelProvider(
        settings=settings, 
        resnet_pretrained=True, 
        temporal=True, # B8 expects the base model to be temporal
    )

    # define our trainer
    trainer = TrainerController(
        baseline=BaselinesEnums.B8_MODEL,
        lr_scheduler=LREnums.COSINE,
        settings=settings,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        resnet_pretrained=False,
        base_finetuned=base_model,
        base_freeze=False,
        compile=True,
        group_only=False,
        tensorboard_track=True,
    )

    # fit the model
    model = trainer.fit()
