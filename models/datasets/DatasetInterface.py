import os
import torch
from torch import Tensor
from typing import Tuple, Union
from torch.utils.data import Dataset
from helpers.config import Settings
from abc import ABC, abstractmethod

class DatasetInterface(ABC, Dataset):
    
    def __init__(self, videos_split: list, annotations: dict, settings: Settings):
        # load splits and annotatations
        self.settings = settings
        self.videos_split = videos_split
        self.annotations = annotations
        
        # get the dataset path
        self.dataset_path = os.path.join(
            self.settings.BASE_PATH,
            self.settings.DATASET_PATH,
        )
        
        # specify the number of frames before and after the target
        self.no_frames_before = self.settings.CNT_BEFORE_TARGET
        self.no_frames_after = self.settings.CNT_AFTER_TARGET

        # loop through all the clips in the datasets
        self.clips = []
        for video_no in videos_split:
            for clip_no in self.annotations[f"{video_no}"].keys():
                # get the current clip path
                clip_dir = os.path.join(
                    self.dataset_path,
                    str(video_no),
                    str(clip_no),
                )
                # if this clip exist in the data, we are taking its frames
                if os.path.exists(clip_dir):
                    self.clips.append(clip_dir)
        
        # sort the clips
        self.clips.sort()
        
        
    @abstractmethod
    def __len__(self) -> int:
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]:
        pass
