import os
import pickle
from helpers import get_settings, Settings

class AnnotationController:

    def __init__(self, ):
        self.settings = get_settings()
        self.tracking_annotation_path = os.path.join(
            self.settings.BASE_PATH, 
            self.settings.TRACKING_ANNOTATION,
        )
        self.volleyball_annotation_path = os.path.join(
            self.settings.BASE_PATH,
            self.settings.DATASET_PATH,
        )
        self.save_path = os.path.join(
            self.settings.BASE_PATH,
            self.settings.ANNOTATION_PATH,
        )
        self.annotations = {}

    def _load_tracking_annotation(self, tracking_annotation_path: str):

        players = [[] for _ in range(12)]
        with open(tracking_annotation_path, "r", encoding="utf-8") as file:
            
            for line in file:

                info = line.split()

                if len(info) == 0 or int(info[0]) > 11: continue

                players[int(info[0])].append([
                    x := int(info[1]),
                    y := int(info[2]),
                    h := int(info[3]) - int(info[1]),
                    w := int(info[4]) - int(info[2]),
                    frame_id := int(info[5]),
                    activity := info[-1],
                ])
        
        for player in players:
            player = player[9 - self.settings.CNT_BEFORE_TARGET: 9 + self.settings.CNT_AFTER_TARGET + 1]
        
        return players

    def _load_group_annotations(self, group_annotation_path: str, target_frames: dict = dict()):

        with open(group_annotation_path, "r", encoding="utf-8") as file:
            
            for line in file:
                info = line.split()
                if len(info) == 0: continue
                target_frame, group_activity = info[:2]
                target_frame = target_frame.replace(".jpg", "")
                target_frames[target_frame] = group_activity
        
        return target_frames

    def process_annotations(self):
        target_frames = {}
        annotations = {}
        for dirpath, dirnames, _ in os.walk(self.volleyball_annotation_path):

            if dirnames:
                
                group_annotation_path = os.path.join(dirpath, "annotations.txt")
                
                if not os.path.exists(group_annotation_path): continue

                print(f"processing group annotations of video number {dirpath.split("/")[-1]} .....")

                target_frames = self._load_group_annotations(
                    group_annotation_path=group_annotation_path, 
                    target_frames=target_frames,
                )

                continue
            
            video_no, clip_no = dirpath.split("/")[-2:]
            
            if annotations.get(video_no, -1) == -1: 
                print(f"processing frames annotations in video number {video_no} ......")
                annotations[video_no] = {}

            if annotations[video_no].get(clip_no, -1) == -1: 
                annotations[video_no][clip_no] = {}

            annotations[video_no][clip_no] = {
                "group_activity": target_frames[clip_no],
                "players": self._load_tracking_annotation(
                    os.path.join(self.tracking_annotation_path, video_no, clip_no, f"{clip_no}.txt")
                ),
            }

        self.annotations = annotations        
        return self.annotations

    @classmethod
    def get_annotations(cls):
        settings = get_settings()
        annotation_path = os.path.join(
            settings.BASE_PATH,
            settings.ANNOTATION_PATH,
        )
        assert os.path.exists(annotation_path), "annotations need to be processed first before you can get it!"
        with open(annotation_path, "rb") as file:
            annotations = pickle.load(file)
        return annotations

    def save_annotations(self):
        assert not self.annotations == {}, "annotations need to be processed first before you can save it!"
        with open(self.save_path, "wb") as file:
            pickle.dump(
                obj = self.annotations, file = file,
            )
        
        return self.save_path
