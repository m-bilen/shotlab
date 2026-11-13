import time
from pathlib import Path

import numpy as np
import pandas as pd

from shotlab.pose import DEFAULT_MODEL, create_landmarker, detect_poses, vision
from shotlab.roi import get_roi
from shotlab.video import read_frames

OUTPUT_DIR = Path("data/landmarks")
LANDMARK_NAMES = [lm.name.lower() for lm in vision.PoseLandmark]
VALUE_COLUMNS = [f"{name}_{axis}" for name in LANDMARK_NAMES for axis in ("x", "y","v")]

def landmarks_path(video_file, model=DEFAULT_MODEL):
    return OUTPUT_DIR / f"{Path(video_file).stem}_{model}.parquet"


def extract_landmarks(video_path, roi, max_frames=None, model=DEFAULT_MODEL):   
    frames, times, values = [], [], []
    empty = np.full(len(VALUE_COLUMNS), np.nan, dtype=np.float32)
    last_ts = -1
    start = time.time()
    with create_landmarker(vision.RunningMode.VIDEO, num_poses=1, model=model) as landmarker:
        for index, t_ms, frame in read_frames(video_path):
            if max_frames is not None and index >= max_frames:
                break
            ts = max(int(t_ms), last_ts +1)
            last_ts = ts
            poses = detect_poses(landmarker, frame, roi, ts)
            frames.append(index)
            times.append(t_ms)
            values.append(poses[0].ravel().astype(np.float32) if poses else empty)

            if index % 1000 == 0:
                print(f"frame {index} {time.time() - start:.0f} s")
    
    print(f"Done in {time.time() - start:.0f} s")
    df = pd.DataFrame(np.vstack(values), columns=VALUE_COLUMNS)
    df.insert(0, "t_ms", times)
    df.insert(0, "frame", frames)
    return df

if __name__ == "__main__":
    import sys
    video_path = Path(sys.argv[1])
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else None

    roi = get_roi(video_path.name)
    df = extract_landmarks(str(video_path), roi, max_frames)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = landmarks_path(video_path)
    df.to_parquet(out_path, index=False)

    missing = df["nose_x"].isna().mean() * 100
    print(f"Saved {len(df)} frames to {out_path}")
    print(f"No pose in {missing:.1f}% of frames")