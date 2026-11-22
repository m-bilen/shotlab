import time

import pandas as pd

from shotlab.ball import ball_region, detect_ball, window_frames
from shotlab.config import BALL_PRETRAINED_PATH, MODELS_DIR, SHOTS_PATH, VIDEO_DIR
from shotlab.extract import landmarks_path
from shotlab.roi import get_roi
from shotlab.shots import isolated_shots
from shotlab.video import find_video, read_selected_frames

CONFIGS = [("yolo26n", 640), ("yolo26s", 640), ("yolo26m", 640), ("yolo26s", 1088)]
SHOTS_PER_GROUP = 10
SEED = 0

def plan_frames(shots):
    groups = isolated_shots(shots).groupby(["session_id", "shooter"])
    sample = groups.sample(n=SHOTS_PER_GROUP, random_state=SEED)
    rows = []
    for video_file, video_shots in sample.groupby("video_file"):
        times = pd.read_parquet(landmarks_path(video_file), columns=["frame", "t_ms"])
        for shot in video_shots.itertuples():
            for frame in window_frames(times["frame"], times["t_ms"], shot.wrist_top_frame):
                rows.append({
                    "video_file": video_file,
                    "session_id": shot.session_id,
                    "shooter": shot.shooter,
                    "wrist_top_frame": shot.wrist_top_frame,
                    "frame": frame,
                })
    return pd.DataFrame(rows)

def run(plan):
    from ultralytics import YOLO

    models = {(name, imgsz): YOLO(str(MODELS_DIR / f"{name}.pt")) for name, imgsz in CONFIGS}
    rows = []
    for video_file, video_plan in plan.groupby("video_file"):
        region = ball_region(get_roi(video_file))
        path = str(find_video(video_file, VIDEO_DIR))
        start = time.time()
        for index, t_ms, frame in read_selected_frames(path, video_plan["frame"]):
            for (name, imgsz), model in models.items():
                t0 = time.perf_counter()
                ball = detect_ball(model, frame, region, imgsz)
                rows.append({
                    "video_file": video_file,
                    "frame": index,
                    "t_ms": t_ms,
                    "model": name,
                    "imgsz": imgsz,
                    "ms": (time.perf_counter() - t0) * 1000,
                    **(ball or {}),
                })
        print(f"{video_file}: {len(video_plan)} frames in {time.time() - start:.0f} s")
    return plan.merge(pd.DataFrame(rows), on=["video_file", "frame"])


if __name__ == "__main__":
    shots = pd.read_csv(SHOTS_PATH)
    plan = plan_frames(shots)
    print(f"{len(plan)} frames from {plan.groupby(['video_file', 'wrist_top_frame']).ngroups} shots")
    result = run(plan)
    result.to_parquet(BALL_PRETRAINED_PATH, index=False)
    print(f"Saved {len(result)} rows to {BALL_PRETRAINED_PATH}")