import glob
from pathlib import Path

import numpy as np
import pandas as pd

from shotlab.extract import OUTPUT_DIR

KEY_POINTS = ["right_shoulder", "right_elbow", "right_wrist", "right_hip", "right_knee", "right_ankle"]


def frame_interval_stats(df):
    dt = df["t_ms"].diff().dropna()
    return {
        "median_dt_ms": round(dt.median(), 2),
        "gaps_over_25ms": int((dt > 25).sum()),
        "max_gap_ms": round(dt.max(), 1),
    }

def jump_rate(df, name, threshold_px=50):
    xy = df[[f"{name}_x", f"{name}_y"]].to_numpy()
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    valid = ~np.isnan(step)
    return float((step[valid] > threshold_px).mean() * 100)

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
    files = sorted(glob.glob(str(OUTPUT_DIR / "*.parquet")))
    report = pd.DataFrame({Path(f).stem: video_report(f) for f in files})
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)
    print(report)
    report.to_csv("data/quality_report.csv")

