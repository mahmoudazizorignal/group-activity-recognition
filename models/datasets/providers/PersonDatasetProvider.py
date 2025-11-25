import os
import torch
from helpers.config import Settings
from torchvision.transforms import v2
from torchvision.io import decode_image
import torchvision.transforms.functional as F
from models.datasets.DatasetInterface import DatasetInterface

class PersonDatasetProvider(DatasetInterface):
    
    def __init__(self, videos_split: list, annotations: dict, settings: Settings):
        super().__init__(videos_split=videos_split, annotations=annotations, settings=settings)
        self.activity_to_id = self.settings.PLAYER_ACTION_TO_ID
        self.transform = v2.Compose([
            v2.Resize(size=(224, 224), interpolation=v2.InterpolationMode.BILINEAR),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(mean=self.settings.NORM_MEAN, std=self.settings.NORM_STD),
        ])
    
        
    def __len__(self) -> int:
        return len(self.clips)
    
    
    def __getitem__(self, idx):

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
