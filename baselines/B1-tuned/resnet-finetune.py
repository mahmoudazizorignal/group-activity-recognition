import os
import pickle
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2


base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def load_annotations_pickle():
    
    with open(f"{base_path}/dataset/annotations.pkl", "rb") as file:
        annotations = pickle.load(file=file)
    
    return annotations


def load_model():

    processor = v2.Compose([
        v2.Resize(size=232, interpolation=v2.InterpolationMode.BILINEAR),
        v2.CenterCrop(size=224),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(in_features=2048, out_features=8, bias=True)

    return model, processor



if __name__ == "__main__":

    # Check wether you're running as a script of as a module to handle ImportError
    if __package__ is None:
        import sys
        sys.path.append(base_path)
        from helpers import Player, get_settings
    
    else:
        from helpers import Player, get_settings

    settings = get_settings()
    annotations = load_annotations_pickle()
    model, processor = load_model()

    import code; code.interact(local=locals())