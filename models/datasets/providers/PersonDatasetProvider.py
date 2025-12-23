import os
import torch
from torch import Tensor
from typing import Tuple
from helpers.config import Settings
from torchvision.transforms import v2
from torchvision.io import decode_image
import torchvision.transforms.functional as F
from models.datasets.DatasetInterface import DatasetInterface

class PersonDatasetProvider(DatasetInterface):
    """
    A PyTorch Dataset implementation for loading individual person-level video clips with
    activity annotations.
    
    This dataset provider loads video frames from disk, extracts individual player bounding
    boxes, applies transformations, and returns tensors suitable for training person-level
    activity recognition models. Each sample consists of temporal sequences of cropped player
    regions centered around a middle frame, along with corresponding individual action labels
    and group activity labels.
    
    Attributes:
        group_player_to_id (dict): Mapping from group activity names to integer identifiers.
        player_activity_to_id (dict): Mapping from individual player action names to integer
            identifiers.
        transform (v2.Compose): Composed transformation pipeline for preprocessing cropped
            player regions, including resizing, dtype conversion, and normalization.
    
    Args:
        videos_split (list): List of video identifiers included in this dataset split.
        annotations (dict): Dictionary containing player bounding boxes, individual actions,
            and group activity annotations for each video and frame.
        settings (Settings): Configuration object containing dataset parameters, paths, and
            action mappings.
    """
    def __init__(self, videos_split: list, annotations: dict, settings: Settings):
        super().__init__(videos_split=videos_split, annotations=annotations, settings=settings)
        self.group_player_to_id = self.settings.GROUP_ACTION_TO_ID
        self.player_activity_to_id = self.settings.PLAYER_ACTION_TO_ID
        self.transform = v2.Compose([
            v2.Resize(size=(224, 224), interpolation=v2.InterpolationMode.BILINEAR),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(mean=self.settings.NORM_MEAN, std=self.settings.NORM_STD),
        ])
    
        
    def __len__(self) -> int:
        """
        Return the total number of clips in the dataset.
        
        Returns:
            int: The number of video clips available in this dataset split.
        """
        return len(self.clips)
    
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Retrieve and preprocess a video clip with all players and their corresponding labels
        at the given index.
        
        This method loads a temporal sequence of frames centered around a middle frame,
        extracts individual player bounding boxes from each frame, applies the transformation
        pipeline to each cropped region, and returns the processed player sequences along with
        their individual action labels and group activity label. Missing players are represented
        with zero tensors and assigned a special 'missing' label (9).
        
        Args:
            idx (int): Index of the clip to retrieve from the dataset.
        
        Returns:
            tuple: A tuple containing:
                - x (torch.Tensor): Preprocessed player frame sequences with shape
                  (PLAYER_CNT, FRAME_CNT, C, H, W), where PLAYER_CNT is the number of
                  players in the clip, FRAME_CNT is the total number of frames, C is the
                  number of channels (3 for RGB), H is height (224), and W is width (224).
                - y1 (torch.Tensor): Individual player action labels with shape
                  (PLAYER_CNT, FRAME_CNT), where each element contains the action ID for
                  the corresponding player at each frame. Missing players have label 9.
                - y2 (torch.Tensor): Group activity labels with shape (FRAME_CNT,), where
                  each element contains the same group activity ID for the entire clip.
        """
        # get current clip path
        video_no, middle_frame = self.clips[idx].split("/")[-2:]

        cur_clip_folder = os.path.join(
            self.dataset_path,
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
        y1 = []
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

                y1.append(torch.full(
                    size=(self.settings.FRAME_CNT,),
                    fill_value=9,
                ))

                continue
            
            # else, get a specified number of frames before and after along with the corresponding labels
            x.append([])
            y1.append([])
            frames = player[9 - self.no_frames_before: 9 + self.no_frames_after + 1]
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
                y1[-1].append(torch.tensor(self.player_activity_to_id[activity]))
            
            # x[-1] => (FRAME_CNT, C, H, W), y[-1] => (FRAME_CNT, )
            x[-1] = torch.stack(x[-1])
            y1[-1] = torch.stack(y1[-1])
        
        # x => (PLAYER_CNT, FRAME_CNT, C, H, W), y => (PLAYER_CNT, FRAME_CNT, )
        x = torch.stack(x)
        y1 = torch.stack(y1)

        # Get the group activity of that clip
        group_activity = self.group_player_to_id[self.annotations[video_no][middle_frame]["group_activity"]]
        y2 = torch.tensor([group_activity for _ in range(self.settings.FRAME_CNT)])
        return x, y1, y2
