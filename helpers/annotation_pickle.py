import os
import pickle
from time import time


base_path =  f"{os.path.dirname(os.path.dirname(__file__))}/dataset"
tracking_annotation_path = os.path.join(base_path, "volleyball/volleyball_tracking_annotation/volleyball_tracking_annotation")
volleyball_annotation_path = os.path.join(base_path, "volleyball/volleyball_/videos")


class Player:

    def __init__(self, x: int, y: int, h: int, w: int, frame_id: int, activity: str):
        self.x = x
        self.y = y
        self.h = h
        self.w = w
        self.frame_id = frame_id
        self.activity = activity


def load_group_annotations(group_annotation_path: str, target_frames: dict = dict()):

    with open(group_annotation_path, "r", encoding="utf-8") as file:
        
        for line in file:
            info = line.split()
            if len(info) == 0: continue
            target_frame, group_activity = info[:2]
            target_frame = target_frame.replace(".jpg", "")
            target_frames[target_frame] = group_activity
    
    return target_frames


def load_tracking_annotation(tracking_annotation_path: str):

    players = [[] for player in range(12)]
    with open(tracking_annotation_path, "r", encoding="utf-8") as file:
        
        for line in file:

            info = line.split()

            if len(info) == 0 or int(info[0]) > 11: continue

            players[int(info[0])].append(Player(
                x = int(info[1]),
                y = int(info[2]),
                h = int(info[3]) - int(info[1]),
                w = int(info[4]) - int(info[2]),
                frame_id = int(info[5]),
                activity=info[-1]
            ))
    
    for player in players:
        player = player[5:]
        player = player[:-6]
    
    return players


def load_annotations(tracking_annotation_path: str, volleyball_annotation_path: str):
    
    target_frames = {}
    annotations = {}
    for dirpath, dirnames, _ in os.walk(volleyball_annotation_path):

        if dirnames:
            
            group_annotation_path = os.path.join(dirpath, "annotations.txt")
            
            if not os.path.exists(group_annotation_path): continue

            print(f"processing group annotations of video number {dirpath.split("/")[-1]} .....")

            target_frames = load_group_annotations(
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
            "players": load_tracking_annotation(os.path.join(tracking_annotation_path, video_no, clip_no, f"{clip_no}.txt")),
        }
                
    return annotations


if __name__ == "__main__":

    assert os.path.exists(tracking_annotation_path) and os.path.exists(volleyball_annotation_path)
    
    start = time()

    annotations = load_annotations(
        tracking_annotation_path=tracking_annotation_path, 
        volleyball_annotation_path=volleyball_annotation_path
    )

    with open(os.path.join(base_path, "annotations.pkl"), "wb") as file:
        pickle.dump(
            obj=annotations,
            file=file
        )

    end = time()
    print(f"processing of annotations takes: {end - start}")
    print(f"annotations saved at: {base_path}/annotations.pkl")
