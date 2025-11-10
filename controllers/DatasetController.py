import os
import torch
from abc import ABC
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import v2
from helpers.config import get_settings, Settings


class BaseDataset(ABC, Dataset):
    
    def __init__(self, videos_split: list, annotations: dict):
        self.settings = get_settings()
        self.videos_split = videos_split
        self.annotations = annotations
        
        self.__annotations_pkl_path = os.path.join(
            self.settings.BASE_PATH,
            self.settings.ANNOTATION_PATH,
        )
        self.__dataset_path = os.path.join(
            self.settings.BASE_PATH,
            self.settings.DATASET_PATH,
        )
        
        self.no_frames_before = self.settings.CNT_BEFORE_TARGET
        self.no_frames_after = self.settings.CNT_AFTER_TARGET

        self.clips = []
        for video_no in videos_split:
            for clip_no in self.annotations[f"{video_no}"].keys():
                self.clips.append(f"{self.__dataset_path}/{video_no}/{clip_no}")
        self.clips.sort()


class GroupLevelDataset(BaseDataset):

    def __init__(self, videos_split: list, annotations: dict):
        super().__init__(videos_split, annotations)
        self.activity_to_id = self.settings.GROUP_ACTION_TO_ID
        self.transform = v2.Compose([
            v2.Resize(size=(256, 256), interpolation=v2.InterpolationMode.BILINEAR),
            v2.CenterCrop(size=(224, 224)),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(mean=self.settings.NORM_MEAN, std=self.settings.NORM_STD),
        ])
    
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
        x = self.transform(clip_frames)

        activity_id = self.activity_to_id[self.annotations[video_no][middle_frame]["group_activity"]]
        y = torch.tensor([activity_id])
        y = y.repeat(x.shape[0])
        return x, y


class PersonLevelDataset(BaseDataset):

    def __init__(self, videos_split: list, annotations: dict):
        super().__init__(videos_split, annotations)
        self.activity_to_id = self.settings.PLAYER_ACTION_TO_ID
        self.transform = v2.Compose([
            v2.Resize(size=(224, 224), interpolation=v2.InterpolationMode.BILINEAR),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(mean=self.settings.NORM_MEAN, std=self.settings.NORM_STD),
        ])

