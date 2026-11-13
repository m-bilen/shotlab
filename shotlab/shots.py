from pathlib import Path

import numpy as np
import pandas as pd

from shotlab.extract import landmarks_path

SESSIONS_PATH = Path("data/sessions.csv")
MANUAL_PATH = Path("data/release_frames_manual.csv")
SHOTS_PATH = Path("data/shots.csv")

MIN_VISIBILITY = 0.5
MAX_GAP = 6
MIN_LENGTH = 15

def find_runs(mask):
    padded = np.concatenate([[False], mask, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(s), int(e)) for s, e in zip(changes[::2], changes[1::2])]

def merge_runs(runs, max_gap):
    merged = []
    for start, end in runs:
        if merged and start - merged [-1][1] <= max_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged

def arm_up_mask(df):
    wrist_y = df["right_wrist_y"].to_numpy()
    nose_y = df["nose_y"].to_numpy()
    visible = df["right_wrist_v"].to_numpy() > MIN_VISIBILITY
    return (wrist_y < nose_y) & visible

def detect_shots(df):
    runs = merge_runs(find_runs(arm_up_mask(df)), MAX_GAP)
    runs = [(s, e) for s, e in runs if e - s >= MIN_LENGTH]

    wrist_y = df["right_wrist_y"].to_numpy()
    rows = []
    for start, end in runs:
        top = start + int(np.nanargmin(wrist_y[start:end]))
        rows.append({
            "start_frame": int(df["frame"].iloc[start]),
            "end_frame": int(df["frame"].iloc[end - 1]) + 1,
            "wrist_top_frame": int(df["frame"].iloc[top]),
        })
    return pd.DataFrame(rows, columns=["start_frame", "end_frame", "wrist_top_frame"])

def shots_for_session(row):
    columns = ["frame", "nose_y", "right_wrist_y", "right_wrist_v"]
    df = pd.read_parquet(landmarks_path(row.video_file), columns=columns)
    part = df[(df["frame"] >= row.start_frame) & (df["frame"] < row.end_frame)]
    shots = detect_shots(part.reset_index(drop=True))
    shots.insert(0, "shooter", row.shooter)
    shots.insert(0, "session_id", row.session_id)
    shots.insert(0, "video_file", row.video_file)
    return shots

def manual_inside_shots(shots, manual):
    inside = 0
    for r in manual.itertuples():
        same_video = shots["video_file"] == r.video_file
        covers = (shots["start_frame"] <= r.release_frame) & (shots["end_frame"] > r.release_frame)
        inside += int((same_video & covers).any())
    return inside

if __name__ == "__main__":
    sessions = pd.read_csv(SESSIONS_PATH)
    shots = pd.concat([shots_for_session(r) for r in sessions.itertuples()], ignore_index=True)
    shots["duration_s"] = ((shots["end_frame"] - shots["start_frame"]) / 60).round(2)
    group = shots.groupby(["video_file", "shooter"])
    shots["gap_s"] = (group["start_frame"].diff() / 60).round(2)    
    shots.to_csv(SHOTS_PATH, index=False)

    summary = sessions.set_index(["session_id", "shooter"])[["shots"]]
    summary["found"] = shots.groupby(["session_id", "shooter"]).size()
    print(summary)
    stats = shots.groupby(["session_id", "shooter"]).agg(
        median_duration=("duration_s", "median"),
        median_gap=("gap_s", "median"),
        short_gaps=("gap_s", lambda g: int((g < 3).sum())),
    )
    print(stats)    

    manual = pd.read_csv(MANUAL_PATH)
    print(f"Manual release frames inside a shot: {manual_inside_shots(shots, manual)}/{len(manual)}")
