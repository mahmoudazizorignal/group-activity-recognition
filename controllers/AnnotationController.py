import os
import pickle
from helpers.config import Settings

class AnnotationController:
    """
    Controller class for processing, loading, and saving video dataset annotations.
    
    This class handles the processing of tracking annotations (player bounding boxes and
    individual actions) and group activity annotations from the volleyball dataset. It
    consolidates these annotations into a unified structure and provides utilities for
    saving and loading the processed annotations.
    
    Attributes:
        settings (Settings): Configuration object containing paths and parameters.
        tracking_annotation_path (str): Path to the directory containing tracking annotations
            with player bounding boxes and individual actions.
        volleyball_annotation_path (str): Path to the volleyball dataset directory containing
            group activity annotations.
        save_path (str): Path where processed annotations will be saved.
        annotations (dict): Nested dictionary storing processed annotations with structure:
            {video_no: {frame_no: {"group_activity": str, "players": list}}}
    
    Args:
        settings (Settings): Configuration object containing dataset paths and parameters.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
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
        """
        Load tracking annotations for individual players from a file.
        
        Parses a tracking annotation file containing bounding box coordinates, frame IDs,
        and activity labels for up to 12 players. Each player's annotations are stored as
        a list of frames with their corresponding bounding box information and activity.
        
        Args:
            tracking_annotation_path (str): Path to the tracking annotation text file.
        
        Returns:
            list: A list of 12 sublists, one per player. Each sublist contains frame
                annotations as [x, y, h, w, frame_id, activity], where:
                - x (int): Left coordinate of bounding box
                - y (int): Top coordinate of bounding box
                - h (int): Height of bounding box
                - w (int): Width of bounding box
                - frame_id (int): Frame number
                - activity (str): Individual player activity label
        """
        players = [[] for _ in range(12)]
        with open(tracking_annotation_path, "r", encoding="utf-8") as file:
            
            frames = {}
            for line in file:

                info = line.split()

                if len(info) == 0 or int(info[0]) > 11: continue
                
                if frames.get(int(info[5]), -1) == -1:
                    frames[int(info[5])] = []
                    
                frames[info[5]].append([
                    x        := int(info[1]),
                    y        := int(info[2]),
                    h        := int(info[3]) - int(info[1]),
                    w        := int(info[4]) - int(info[2]),
                    activity := info[-1],        
                ])
            
            players = [[] for _ in range(12)]
            for frame in frames.keys():
                
                # sort the bounding boxes by the top left corner
                frames[frame].sort()
                
                # appending the players movements through the whole clip
                for i, bbox in enumerate(frames[frame]):
                    players[i].append([
                        x        := bbox[0],
                        y        := bbox[1],
                        h        := bbox[2],
                        w        := bbox[3],
                        frame_id := frame,
                        activity := bbox[4],
                    ])
        
        for player in players:
            player = player[9 - self.settings.CNT_BEFORE_TARGET: 9 + self.settings.CNT_AFTER_TARGET + 1]
        
        return players

    def _load_group_annotations(self, group_annotation_path: str, target_frames: dict = dict()):
        """
        Load group activity annotations from a file.
        
        Parses a group annotation file that contains frame identifiers and their
        corresponding group activity labels. Updates the provided target_frames
        dictionary with the parsed annotations.
        
        Args:
            group_annotation_path (str): Path to the group annotation text file.
            target_frames (dict, optional): Existing dictionary to update with new
                annotations. Defaults to an empty dictionary.
        
        Returns:
            dict: Dictionary mapping frame identifiers (str) to group activity labels (str).
        """
        with open(group_annotation_path, "r", encoding="utf-8") as file:
            
            for line in file:
                info = line.split()
                if len(info) == 0: continue
                target_frame, group_activity = info[:2]
                target_frame = target_frame.replace(".jpg", "")
                target_frames[target_frame] = group_activity
        
        return target_frames

    def process_annotations(self):
        """
        Process all annotations from the volleyball dataset directory.
        
        Traverses the dataset directory structure to load both group activity annotations
        and player tracking annotations. Consolidates all annotations into a nested
        dictionary structure organized by video number and frame number. The processed
        annotations are stored in the `annotations` attribute.
        
        The method processes:
        - Group activity labels for each target frame
        - Player bounding boxes and individual action labels for each frame
        
        Returns:
            dict: Nested dictionary with structure:
                {video_no: {frame_no: {
                    "group_activity": str,
                    "players": list of player annotations
                }}}
        """
        target_frames = {}
        annotations = {}
        for dirpath, dirnames, _ in os.walk(self.volleyball_annotation_path):

            if dirnames:
                
                group_annotation_path = os.path.join(dirpath, "annotations.txt")
                
                if not os.path.exists(group_annotation_path): continue

                print(f"processing group annotations of video number {dirpath.split('/')[-1]} .....")

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

            import code; code.interact(local=locals())
            annotations[video_no][clip_no] = {
                "group_activity": target_frames[clip_no],
                "players": self._load_tracking_annotation(
                    os.path.join(self.tracking_annotation_path, video_no, clip_no, f"{clip_no}.txt")
                ),
            }

        self.annotations = annotations        
        return self.annotations

    @classmethod
    def get_annotations(cls, settings: Settings):
        """
        Load previously saved annotations from disk.
        
        Class method to retrieve annotations that have been processed and saved using
        the `save_annotations` method. This provides a convenient way to load annotations
        without needing to reprocess them.
        
        Args:
            settings (Settings): Configuration object containing the base path and
                annotation file path.
        
        Returns:
            dict: The loaded annotations dictionary with the same structure as produced
                by `process_annotations`.
        
        Raises:
            AssertionError: If the annotation file does not exist at the specified path,
                indicating that annotations need to be processed first.
        """
        settings = settings
        annotation_path = os.path.join(
            settings.BASE_PATH,
            settings.ANNOTATION_PATH,
        )
        assert os.path.exists(annotation_path), "annotations need to be processed first before you can get it!"
        with open(annotation_path, "rb") as file:
            annotations = pickle.load(file)
        return annotations

    def save_annotations(self):
        """
        Save processed annotations to disk using pickle serialization.
        
        Serializes the processed annotations dictionary and saves it to the path specified
        in the settings. This allows for efficient loading of annotations in future runs
        without reprocessing.
        
        Returns:
            str: The file path where annotations were saved.
        
        Raises:
            AssertionError: If the annotations dictionary is empty, indicating that
                `process_annotations` needs to be called first.
        """
        assert not self.annotations == {}, "annotations need to be processed first before you can save it!"
        with open(self.save_path, "wb") as file:
            pickle.dump(
                obj = self.annotations, file = file,
            )
        
        return self.save_path
