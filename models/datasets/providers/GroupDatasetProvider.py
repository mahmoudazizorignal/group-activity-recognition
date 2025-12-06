import os
import torch
from helpers.config import Settings
from torchvision.transforms import v2
from torchvision.io import decode_image
from models.datasets.DatasetInterface import DatasetInterface

class GroupDatasetProvider(DatasetInterface):
    """
    A PyTorch Dataset implementation for loading video clips with group activity annotations.
    
    This dataset provider loads video frames from disk, applies transformations, and returns
    tensors suitable for training group activity recognition models. Each sample consists of
    a temporal sequence of frames centered around a middle frame, along with corresponding
    group activity labels.
    
    Attributes:
        activity_to_id (dict): Mapping from group activity names to integer identifiers.
        transform (v2.Compose): Composed transformation pipeline for preprocessing frames,
            including resizing, center cropping, dtype conversion, and normalization.
    
    Args:
        videos_split (list): List of video identifiers included in this dataset split.
        annotations (dict): Dictionary containing group activity annotations for each video
            and frame.
        settings (Settings): Configuration object containing dataset parameters and paths.
    """
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
        """
        Return the total number of clips in the dataset.
        
        Returns:
            int: The number of video clips available in this dataset split.
        """
        return len(self.clips)

    def __getitem__(self, idx: int):
        """
        Retrieve and preprocess a video clip and its corresponding label at the given index.
        
        This method loads a temporal sequence of frames centered around a middle frame,
        applies the transformation pipeline, and returns the processed frames along with
        their group activity labels. The number of frames loaded is determined by
        `no_frames_before` and `no_frames_after` attributes.
        
        Args:
            idx (int): Index of the clip to retrieve from the dataset.
        
        Returns:
            tuple: A tuple containing:
                - x (torch.Tensor): Preprocessed video frames with shape 
                  (FRAME_CNT, C, H, W), where FRAME_CNT is the total number of frames,
                  C is the number of channels (3 for RGB), H is height (224), and W is
                  width (224).
                - y (torch.Tensor): Group activity labels with shape (FRAME_CNT,), where
                  each element contains the same activity ID corresponding to the clip's
                  group activity.
        """
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
