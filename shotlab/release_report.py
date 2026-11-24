import numpy as np
import pandas as pd
import sys

from shotlab.config import MANUAL_PATH, release_ball_path, REVIEWED_PATH, SESSIONS_PATH, SHOTS_PATH
from shotlab.extract import landmarks_path
from shotlab.release import release_time, shot_distance
from shotlab.release_eval import manual_shots, reviewed_shots

THRESHOLDS = np.round(np.arange(0.5, 2.01, 0.05), 2)
MAX_ERROR_MS = 100
NOT_BLIND = [
    ("VID_20261112_104955.mp4", 90049),
    ("VID_20261112_104955.mp4", 41220),
    ("VID_20261112_104955.mp4", 10139),
]

def load_eval_frames(model):
    ball = pd.read_parquet(release_ball_path(model))
    parts = []
    for video_file, video_ball in ball.groupby("video_file"):
        columns = ["frame", "right_index_x", "right_index_y"]
        landmarks = pd.read_parquet(landmarks_path(video_file), columns=columns)
        parts.append(video_ball.merge(landmarks, on="frame"))
    return pd.concat(parts, ignore_index=True)

def releases(frames, threshold):
    rows = []
    for (video_file, start_frame), shot in frames.groupby(["video_file", "start_frame"]):
        shot = shot.sort_values("t_ms")
        found = release_time(shot["t_ms"], shot_distance(shot), threshold)
        rows.append({
            "video_file": video_file,
            "start_frame": start_frame,
            "release_ms": np.nan if found is None else found,
        })
    return pd.DataFrame(rows)

def manual_errors(manual, found, frames):
    times = frames[["video_file", "frame", "t_ms"]].drop_duplicates()
    result = manual.merge(times, left_on=["video_file", "release_frame"], right_on=["video_file", "frame"])
    result = result.merge(found, on=["video_file", "start_frame"])
    result["error_ms"] = result["release_ms"] - result["t_ms"]
    return result


def score(errors):
    return errors["error_ms"].abs().fillna(MAX_ERROR_MS).clip(upper=MAX_ERROR_MS).mean()

def choose_threshold(errors):
    return min(errors, key=lambda thr: score(errors[thr]))


def leave_one_session_out(manual, frames):
    errors = {thr: manual_errors(manual, releases(frames, thr), frames) for thr in THRESHOLDS}
    tested = []
    for session in sorted(manual["session_id"].unique()):
        train = {thr: score(e[e["session_id"] != session]) for thr, e in errors.items()}
        best = min(train, key=train.get)
        test = errors[best]
        tested.append(test[test["session_id"] == session].assign(threshold=best))
    return pd.concat(tested, ignore_index=True), errors

def session_summary(tested, fps):
    tested = tested.assign(error_frames=tested["error_ms"] * tested["video_file"].map(fps) / 1000)
    return tested.groupby("session_id").agg(
        threshold=("threshold", "first"),
        shots=("error_ms", "size"),
        missed=("error_ms", lambda e: int(e.isna().sum())),
        median_frames=("error_frames", "median"),
        within_2=("error_frames", lambda e: round((e.abs() <= 2).mean() * 100)),
    )


def release_check(reviewed, frames, threshold):
    check = reviewed.merge(releases(frames, threshold), on=["video_file", "start_frame"])
    check = check.assign(release_found=check["release_ms"].notna())
    return check.groupby("label")["release_found"].agg(["sum", "size"])

def blind_only(tested):
    keys = zip(tested["video_file"], tested["release_frame"])
    return tested[[key not in NOT_BLIND for key in keys]]


def overall(tested, fps):
    error_frames = tested["error_ms"] * tested["video_file"].map(fps) / 1000
    return {
        "shots": len(tested),
        "missed": int(tested["error_ms"].isna().sum()),
        "within_2": int((error_frames.abs() <= 2).sum()),
    }

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "yolo26m"
    shots = pd.read_csv(SHOTS_PATH)
    frames = load_eval_frames(model)
    manual = manual_shots(shots, pd.read_csv(MANUAL_PATH))
    reviewed = reviewed_shots(shots, pd.read_csv(REVIEWED_PATH))
    fps = pd.read_csv(SESSIONS_PATH).drop_duplicates("video_file").set_index("video_file")["fps"]

    tested, errors = leave_one_session_out(manual, frames)
    final = choose_threshold(errors)

    pd.set_option("display.width", 200)
    print(f"Model {model}: median {frames['ms'].median():.0f} ms per frame on this CPU")
    print(session_summary(tested, fps))
    print(f"All shots: {overall(tested, fps)}")
    print(f"Blind labels only: {overall(blind_only(tested), fps)}")
    print(f"\nThreshold chosen on all sessions: {final}")
    print(release_check(reviewed, frames, final))