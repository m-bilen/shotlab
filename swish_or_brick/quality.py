import glob
from pathlib import Path

import numpy as np
import pandas as pd
from swish_or_brick.config import LANDMARKS_DIR, QUALITY_REPORT_PATH
from swish_or_brick.pose import DEFAULT_MODEL

KEY_POINTS = ["right_shoulder", "right_elbow", "right_wrist", "right_hip", "right_knee", "right_ankle"]
JUMP_SPEED = 30


def frame_interval_stats(df):
    dt = df["t_ms"].diff().dropna()
    median = dt.median()
    return {
        "median_dt_ms": round(median, 2),
        "frame_gaps": int((dt > 1.5 * median).sum()),
        "max_gap_ms": round(dt.max(), 1),
    }

def torso_length(df):
    shoulder = df[["right_shoulder_x", "right_shoulder_y"]].to_numpy()
    hip = df[["right_hip_x", "right_hip_y"]].to_numpy()
    return float(np.nanmedian(np.linalg.norm(shoulder - hip, axis=1)))

def speeds(df, name):
    xy = df[[f"{name}_x", f"{name}_y"]].to_numpy()
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    dt_s = np.diff(df["t_ms"].to_numpy()) / 1000
    return step / torso_length(df) / dt_s



def jump_rate(df, name, threshold=JUMP_SPEED):
    speed = speeds(df, name)
    valid = np.isfinite(speed)
    return float((speed[valid] > threshold).mean() * 100)

def video_report(path):
    df = pd.read_parquet(path)
    row = {
        "frames": len(df),
        "no_pose_pct": round(df["nose_x"].isna().mean() * 100, 1),
    }

    row.update(frame_interval_stats(df))
    for name in KEY_POINTS:
        row[f"jump_{name}"] = round(jump_rate(df,name), 2)
    return row

if __name__ == "__main__":
    files = sorted(glob.glob(str(LANDMARKS_DIR / f"*_{DEFAULT_MODEL}.parquet")))
    report = pd.DataFrame({Path(f).stem: video_report(f) for f in files})
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)
    print(report)
    report.to_csv(QUALITY_REPORT_PATH)

