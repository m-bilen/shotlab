import sys
from pathlib import Path

import pandas as pd

from shotlab.ball import detect_in_plan, window_plan
from shotlab.config import BALL_ALL_DIR, SHOTS_PATH
from shotlab.release_eval import add_release_window

MODEL = "yolo26n_ball"


def part_path(video_file):
    return BALL_ALL_DIR / f"{Path(video_file).stem}.parquet"


def detect_video(video_shots):
    return detect_in_plan(window_plan(video_shots), [(MODEL, 640)])


if __name__ == "__main__":
    shots = add_release_window(pd.read_csv(SHOTS_PATH))
    if len(sys.argv) > 1:
        result = detect_video(shots.head(int(sys.argv[1])))
        print(result.groupby("start_frame")["conf"].apply(lambda c: c.notna().mean()))
        sys.exit()

    BALL_ALL_DIR.mkdir(exist_ok=True)
    for video_file, video_shots in shots.groupby("video_file"):
        path = part_path(video_file)
        if path.exists():
            print(f"{video_file}: already done")
            continue
        detect_video(video_shots).to_parquet(path, index=False)