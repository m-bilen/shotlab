

import numpy as np
import pandas as pd

from shotlab.extract import landmarks_path
from shotlab.geometry import angle
from shotlab.config import MANUAL_PATH, SESSIONS_PATH, SHOTS_PATH

MIN_VISIBILITY = 0.5
MAX_GAP_MS = 120
MIN_LENGTH_MS = 250
MIN_ELBOW_MAX = 150
SHORT_GAP_S = 3

def find_runs(mask):
    padded = np.concatenate([[False], mask, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(s), int(e)) for s, e in zip(changes[::2], changes[1::2])]

def merge_runs(runs, t, max_gap_ms):
    merged = []
    for start, end in runs:
        if merged and t[start] - t[merged[-1][1] - 1] <= max_gap_ms:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged

def arm_up_mask(df):
    wrist_y = df["right_wrist_y"].to_numpy()
    nose_y = df["nose_y"].to_numpy()
    visible = df["right_wrist_v"].to_numpy() > MIN_VISIBILITY
    return (wrist_y < nose_y) & visible

def xy(df, name):
    return df[[f"{name}_x", f"{name}_y"]].to_numpy()

def elbow_at(df, frames):
    rows = df.set_index("frame").loc[frames]
    return angle(xy(rows, "right_shoulder"), xy(rows, "right_elbow"), xy(rows, "right_wrist"))

def keep_real_shots(shots):
    return shots[shots["elbow_max"] >= MIN_ELBOW_MAX].reset_index(drop=True)

def add_timing(shots):
    shots = shots.sort_values(["video_file", "start_ms"]).reset_index(drop=True)
    shots["duration_s"] = ((shots["end_ms"] - shots["start_ms"]) / 1000).round(2)
    group = shots.groupby(["video_file", "shooter"])
    shots["gap_s"] = (group["start_ms"].diff() / 1000).round(2)
    return shots

def isolated_shots(shots):
    shots = shots.sort_values(["video_file", "start_ms"]).reset_index(drop=True)
    next_gap = shots.groupby(["video_file", "shooter"])["gap_s"].shift(-1)
    near_other = (shots["gap_s"] < SHORT_GAP_S) | (next_gap < SHORT_GAP_S)
    return shots[~near_other].reset_index(drop=True)

def summarize(sessions, shots): 
    keys = ["session_id", "shooter"]
    summary = sessions.groupby(keys)[["shots"]].sum()
    found = shots.groupby(keys).size()
    summary["found"] = found.reindex(summary.index, fill_value=0)
    stats = shots.groupby(keys).agg(
        median_duration=("duration_s", "median"),
        median_gap=("gap_s", "median"),
        short_gaps=("gap_s", lambda g: int((g < SHORT_GAP_S).sum())),
    )
    return summary.join(stats)
def detect_shots(df):
    t = df["t_ms"].to_numpy()
    runs = merge_runs(find_runs(arm_up_mask(df)), t, MAX_GAP_MS)
    runs = [(s, e) for s, e in runs if t[e - 1] - t[s] >= MIN_LENGTH_MS]

    wrist_y = df["right_wrist_y"].to_numpy()
    frame = df["frame"].to_numpy()
    rows = []
    for start, end in runs:
        top = start + int(np.nanargmin(wrist_y[start:end]))
        rows.append({
            "wrist_top_ms": float(t[top]),
            "start_frame": int(frame[start]),
            "end_frame": int(frame[end - 1]) + 1,
            "wrist_top_frame": int(frame[top]),
            "start_ms": float(t[start]),
            "end_ms": float(t[end - 1]),
        })
    columns = ["start_frame", "end_frame", "wrist_top_frame", "start_ms", "end_ms", "wrist_top_ms"]
    return pd.DataFrame(rows, columns=columns)

def shots_for_session(row):
    columns = ["frame", "t_ms", "nose_y", "right_wrist_y", "right_wrist_v","right_shoulder_x", "right_shoulder_y", "right_elbow_x", "right_elbow_y", "right_wrist_x"]
    df = pd.read_parquet(landmarks_path(row.video_file), columns=columns)
    part = df[(df["frame"] >= row.start_frame) & (df["frame"] < row.end_frame)]
    shots = detect_shots(part.reset_index(drop=True))
    shots["elbow_top"] = elbow_at(part, shots["wrist_top_frame"]).round(1)
    elbow = angle(xy(part, "right_shoulder"), xy(part, "right_elbow"), xy(part, "right_wrist"))
    frames = part["frame"].to_numpy()
    shots["elbow_max"] = [
        round(float(np.nanmax(elbow[(frames >= s) & (frames < e)])), 1)
        for s, e in zip(shots["start_frame"], shots["end_frame"])
    ]
    shots.insert(0, "shooter", row.shooter)
    shots.insert(0, "session_id", row.session_id)
    shots.insert(0, "video_file", row.video_file)
    return shots

def covering_counts(shots, manual):
    counts = []
    for r in manual.itertuples():
        same_video = shots["video_file"] == r.video_file
        covers = (shots["start_frame"] <= r.release_frame) & (shots["end_frame"] > r.release_frame)
        counts.append(int((same_video & covers).sum()))
    return np.array(counts)

if __name__ == "__main__":
    sessions = pd.read_csv(SESSIONS_PATH)
    shots = pd.concat([shots_for_session(r) for r in sessions.itertuples()], ignore_index=True)
    shots = add_timing(keep_real_shots(shots))
    shots.to_csv(SHOTS_PATH, index=False)

    pd.set_option("display.width", 200)
    print(summarize(sessions, shots))

    counts = covering_counts(shots, pd.read_csv(MANUAL_PATH))
    print(f"Manual release frames inside a shot: {(counts > 0).sum()}/{len(counts)}")
    print(f"Inside more than one shot: {(counts > 1).sum()}")
