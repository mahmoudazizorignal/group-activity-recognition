import os
import pickle
import torch
from torchvision.io import decode_image
from torch.utils.data import DataLoader, Dataset
from baselines import get_processor
from helpers import Player, get_settings, Settings
from abc import ABC, abstractmethod


class BaseDataset(ABC, Dataset):
    
    def __init__(self, videos_split: list, settings: Settings):
        
        self.videos_split = videos_split
        self.settings = settings
        
        self.__base_path = self.settings.BASE_PATH
        self.__annotations_pkl_path = f"{self.__base_path}/{settings.ANNOTATION_PATH}"
        self.__dataset_path = f"{self.__base_path}/{settings.DATASET_PATH}"
        
        self.no_frames_before = self.settings.CNT_BEFORE_TARGET
        self.no_frames_after = self.settings.CNT_AFTER_TARGET

        with open(self.__annotations_pkl_path, "rb") as file:
            self.annotations = pickle.load(file)    

        self.clips = []
        for video_no in videos_split:
            for clip_no in self.annotations[f"{video_no}"].keys():
                self.clips.append(f"{self.__dataset_path}/{video_no}/{clip_no}")
        self.clips.sort()

        self.__processor = get_processor()


class GroupLevelDataset(BaseDataset):

    def __init__(self, videos_split: list, settings: Settings):
        super().__init__(videos_split, settings)
        self.activity_to_id = self.settings.GROUP_ACTION_TO_ID
    
    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx: int):
        clip_path = self.clips[idx]
        video_no, middle_frame = clip_path.split("/")[-2:]

        cur_clip_folder = f"{self._BaseDataset__dataset_path}/{video_no}/{middle_frame}"
        clip_frames = []
        for frame in range(int(middle_frame) - self.no_frames_before, int(middle_frame) + self.no_frames_after + 1):
            cur_frame_path = f"{cur_clip_folder}/{frame}.jpg"
            clip_frames.append(decode_image(cur_frame_path))

        clip_frames = torch.stack(clip_frames) # (F, C, H, W)
        x = self._BaseDataset__processor(clip_frames)

        activity_id = self.activity_to_id[self.annotations[video_no][middle_frame]["group_activity"]]
        y = torch.tensor([activity_id])
        y = y.repeat(x.shape[0])
        return x, y


class PersonLevelDataset(BaseDataset):
    pass



if __name__ == "__main__":

    settings = get_settings()
    group_dataset = GroupLevelDataset(
        videos_split=settings.TRAIN_VIDEOS, 
        settings=settings
    )
    batch = group_dataset[0]
