import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2 

from shotlab.extract import landmarks_path
from shotlab.pose import DEFAULT_MODEL, draw_poses, points_from_table
from shotlab.roi import get_roi
from shotlab.video import play
from shotlab.config import SHOTS_PATH


def load_points(video_path, model=DEFAULT_MODEL):
    df = pd.read_parquet(landmarks_path(video_path, model))
    return points_from_table(df)

def shot_mask(video_file, n_frames):
    shots = pd.read_csv(SHOTS_PATH)
    shots = shots[shots["video_file"] == video_file]
    mask = np.zeros(n_frames, dtype=bool)
    for s in shots.itertuples():
        mask[s.start_frame:s.end_frame] = True
    return mask
if __name__ == "__main__":
    video_path = sys.argv[1]
    start_s =float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

    points = load_points(video_path)
    roi = get_roi(Path(video_path).name)
    in_shot = shot_mask(Path(video_path).name, len(points))

    def overlay(frame, index):
        if index < len(points) and not np.isnan(points[index, 0, 0]):
            draw_poses(frame, [points[index]], roi)
        if index < len(in_shot) and in_shot[index]:
            cv2.putText(frame, "SHOT", (40,200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)

    play(video_path, start_s, overlay)