import time

import numpy as np
import pandas as pd

from swish_or_brick.config import MODELS_DIR, VIDEO_DIR
from swish_or_brick.extract import landmarks_path
from swish_or_brick.roi import get_roi
from swish_or_brick.video import find_video, read_selected_frames

BALL_NAMES = ("sports ball", "ball")
IMGSZ = 1088
MIN_CONF = 0.25


def ball_class(model):
    for index, name in model.names.items():
        if name in BALL_NAMES:
            return index
    raise ValueError(f"No ball class in model: {model.names}")


def frames_between(frames, t_ms, start_ms, end_ms):
    frames = np.asarray(frames)
    t_ms = np.asarray(t_ms)
    inside = (t_ms >= start_ms) & (t_ms <= end_ms)
    return frames[inside].tolist()

def ball_region(roi):
    x, y, size, _ = roi
    top = max(0, y - size)
    return (x, top, size, y + size - top)

def detect_ball(model, frame_bgr, region, imgsz=IMGSZ, conf=0.1):
    x, y, w, h = region
    crop = frame_bgr[y:y + h, x:x + w]
    result = model.predict(crop, imgsz=imgsz, classes=[ball_class(model)], conf=conf, verbose=False)[0]
    if len(result.boxes) == 0:
        return None
    best = int(result.boxes.conf.argmax())
    x1, y1, x2, y2 = result.boxes.xyxy[best].tolist()
    return {
        "conf": float(result.boxes.conf[best]),
        "ball_x": x + (x1 + x2) / 2,
        "ball_y": y + (y1 + y2) / 2,
        "ball_size": ((x2 - x1) + (y2 - y1)) / 2,
    }

def window_plan(shots):
    rows = []
    for video_file, video_shots in shots.groupby("video_file"):
        times = pd.read_parquet(landmarks_path(video_file), columns=["frame", "t_ms"])
        for shot in video_shots.itertuples():
            for frame in frames_between(times["frame"], times["t_ms"], shot.window_start_ms, shot.window_end_ms):
                rows.append({
                   "start_frame": shot.start_frame,
                    "video_file": video_file,
                    "session_id": shot.session_id,
                    "shooter": shot.shooter,
                    "wrist_top_frame": shot.wrist_top_frame,
                    "frame": frame,
                })
    return pd.DataFrame(rows)

def detect_in_plan(plan, configs):
    from ultralytics import YOLO

    models = {(name, imgsz): YOLO(str(MODELS_DIR / f"{name}.pt")) for name, imgsz in configs}
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
    import sys
    from pathlib import Path

    import cv2
    from ultralytics import YOLO

    from swish_or_brick.config import MODELS_DIR
    from swish_or_brick.roi import get_roi
    from swish_or_brick.video import read_frame, resize_for_display

    video_path, frame_index = sys.argv[1], int(sys.argv[2])
    model_name = sys.argv[3] if len(sys.argv) > 3 else "yolo26s"

    frame = read_frame(video_path, frame_index)
    region = ball_region(get_roi(Path(video_path).name))
    model = YOLO(str(MODELS_DIR / f"{model_name}.pt"))
    ball = detect_ball(model, frame, region)
    print(ball)

    x, y, w, h = region
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
    if ball is not None:
        center = (int(ball["ball_x"]), int(ball["ball_y"]))
        cv2.circle(frame, center, int(ball["ball_size"] / 2), (0, 255, 255), 3)
    cv2.imshow("Swish or Brick - ball", resize_for_display(frame)[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()