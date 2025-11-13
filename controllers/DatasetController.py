import os
import torch
from abc import ABC
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import v2
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
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
                clip_dir = os.path.join(
                    self.__dataset_path,
                    str(video_no),
                    str(clip_no),
                )
                self.clips.append(clip_dir)
        self.clips.sort()
    
    def __len__(self) -> int:
        return len(self.clips)

    def get_dataset_path(self) -> str:
        return self.__dataset_path


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
    

    def __getitem__(self, idx: int):

        # get current clip path
        video_no, middle_frame = self.clips[idx].split("/")[-2:]

        cur_clip_folder = os.path.join(
            self.get_dataset_path(),
            video_no,
            middle_frame,
        )

        # load specified number of frames before and after the middle
        clip_frames = []

        for frame in range(int(middle_frame) - self.no_frames_before, int(middle_frame) + self.no_frames_after + 1):
            
            cur_frame_path = os.path.join(
                cur_clip_folder,
                f"{frame}.jpg",
            )

            clip_frames.append(decode_image(cur_frame_path))

        clip_frames = torch.stack(clip_frames) # (F, C, H, W)

        # resizing, crop, scaling, and normalizing the frames
        x = self.transform(clip_frames)

        # get the corresponding group label
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


    
    def __getitem__(self, idx):

        # get current clip path
        video_no, middle_frame = self.clips[idx].split("/")[-2:]

        cur_clip_folder = os.path.join(
            self.get_dataset_path(),
            video_no,
            middle_frame,
        )

        # preload the frames in each clip
        frames_imgs = {}
        for frame_id in range(int(middle_frame) - self.no_frames_before, int(middle_frame) + self.no_frames_after + 1):
            
            cur_frame_path = os.path.join(
                cur_clip_folder,
                f"{frame_id}.jpg",
            )

            frames_imgs[str(frame_id)] = decode_image(cur_frame_path)

        # load specified number of frames before and after the middle for each player
        x = []
        y = []
        players = self.annotations[video_no][middle_frame]["players"]
        for player in players:
            
            # if player is missing in this clip, make images of zeros, and label 9 for `missing`
            if len(player) == 0:

                x.append(torch.zeros((
                    self.settings.FRAME_CNT, 
                    self.settings.C, 
                    self.settings.H, 
                    self.settings.W
                )))

                y.append(torch.full(
                    size=(self.settings.FRAME_CNT,),
                    fill_value=9,
                ))

                continue
            
            # else, get a specified number of frames before and after along with the corresponding labels
            x.append([])
            y.append([])
            frames = player[10 - self.no_frames_before: 10 + self.no_frames_after + 1]
            for frame in frames:
                
                # retrieve bouding box details
                xx, yy, h, w, frame_id, activity = frame

                # get the corresponding frame
                frame_img = frames_imgs[str(frame_id)]

                # crop the frame
                cropped_frame = F.crop(img=frame_img, top=yy, left=xx, height=w, width=h)

                # resizing, crop, scaling, and normalizing the frames
                processesed_frame = self.transform(cropped_frame)

                # append to x, y
                x[-1].append(processesed_frame)
                y[-1].append(torch.tensor(self.activity_to_id[activity]))
            
            # x[-1] => (FRAME_CNT, C, H, W), y[-1] => (FRAME_CNT, )
            x[-1] = torch.stack(x[-1])
            y[-1] = torch.stack(y[-1])
        
        # x => (PLAYER_CNT, FRAME_CNT, C, H, W), y => (PLAYER_CNT, FRAME_CNT, )
        x = torch.stack(x)
        y = torch.stack(y)

        return x, y
