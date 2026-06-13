import csv
import os
from datetime import datetime
import cv2

class ResultLogger:
    def __init__(self, output_folder="", test_name = "", fieldnames=None):
        """
        Initialize a logger for test results

        Args:
            output_folder (str, optional): _description_. Defaults to "".
            fieldnames (_type_, optional): _description_. Defaults to None.
        """
        now = datetime.now()
        os.makedirs(output_folder, exist_ok=True)
        self.filename = os.path.join(output_folder, test_name + now.strftime("%Y%m%d_%H%M%S") + ".csv")
        self.fieldnames = fieldnames or ["trial_id", "instruction", "lifted", "success", "end_time"]
        self.records = []
        
    def log(self, **kwargs):
        """
            record one trial results
        """
        record = {k: kwargs.get(k, None) for k in self.fieldnames}
        self.records.append(record)
        
    def save(self):
        with open(self.filename, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.records)
        print(f"[ResultLogger] Results log saved in {self.filename}, there are {len(self.records)} records.")
        
        
class VideoLogger:
    def __init__(self, output_folder, fps=10):
        """
        Initialize a logger for test results

        Args:
            output_folder (str, optional): _description_. Defaults to "".
            fieldnames (_type_, optional): _description_. Defaults to None.
        """
        self.output_folder = os.path.join(output_folder, f"videos")
        os.makedirs(self.output_folder, exist_ok=True)
        self.records = []
        self.fps = fps
        
    
    def save_video(self, frames, test_name, trial_id, success):
        if not frames:
            print(f"[ExperimentLogger] trial {trial_id}: no frames to save!")
            return
        height, width, _ = frames[0].shape
        tag = "sccuess" if success else "fail"
        filename = os.path.join(self.output_folder,  f"{test_name}_trial_{trial_id}_{tag}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = out = cv2.VideoWriter(filename, fourcc, self.fps, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()
        print(f"[VideoLogger] save video: {filename}, there are {len(frames)} frames.")