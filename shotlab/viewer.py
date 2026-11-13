import sys
from pathlib import Path

import numpy as np
import pandas as pd

from shotlab.extract import landmarks_path
from shotlab.pose import DEFAULT_MODEL, draw_poses
from shotlab.roi import get_roi
from shotlab.video import play


def load_points(video_path, model=DEFAULT_MODEL):
    df = pd.read_parquet(landmarks_path(video_path, model))
    return df.iloc[:, 2:].to_numpy().reshape(len(df), 33, 3)

if __name__ == "__main__":
    video_path = sys.argv[1]
    start_s =float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

    points = load_points(video_path)
    roi = get_roi(Path(video_path).name)

    def overlay(frame, index):
        if index < len(points) and not np.isnan(points[index, 0, 0]):
            draw_poses(frame, [points[index]], roi)
    play(video_path, start_s, overlay)