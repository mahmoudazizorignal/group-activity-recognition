import os
import torch
from helpers.config import Settings
from torchvision.transforms import v2
from torchvision.io import decode_image
from models.datasets.DatasetInterface import DatasetInterface

class GroupDatasetProvider(DatasetInterface):
    
    def __init__(self, videos_split: list, annotations: dict, settings: Settings):
        super().__init__(videos_split=videos_split, annotations=annotations, settings=settings)
        self.activity_to_id = self.settings.GROUP_ACTION_TO_ID
        self.transform = v2.Compose([
            v2.Resize(size=(256, 256), interpolation=v2.InterpolationMode.BILINEAR),
            v2.CenterCrop(size=(224, 224)),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(mean=self.settings.NORM_MEAN, std=self.settings.NORM_STD),
        ])
    
    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int):

        # get current clip path
        video_no, middle_frame = self.clips[idx].split("/")[-2:]

        cur_clip_folder = os.path.join(
            self.dataset_path,
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
