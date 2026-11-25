from pathlib import Path

import cv2
import pandas as pd

from swish_or_brick.ball import ball_region, detect_ball
from swish_or_brick.config import BALL_CROPS_DIR, BALL_PSEUDO_PATH, BALL_TRAIN_PLAN_PATH, MODELS_DIR, VIDEO_DIR
from swish_or_brick.roi import get_roi
from swish_or_brick.video import find_video, read_selected_frames

TEACHER = "yolo26m"
TEACHER_IMGSZ = 640


def crop_name(video_file, frame):
    return f"{Path(video_file).stem}_{frame}.jpg"


def save_crops(plan, model):
    rows = []
    for video_file, video_plan in plan.groupby("video_file"):
        x, y, w, h = ball_region(get_roi(video_file))
        split_of = video_plan.set_index("frame")["split"]
        path = str(find_video(video_file, VIDEO_DIR))
        for index, _, frame in read_selected_frames(path, video_plan["frame"]):
            crop = frame[y:y + h, x:x + w]
            image = f"{split_of[index]}/{crop_name(video_file, index)}"
            cv2.imwrite(str(BALL_CROPS_DIR / image), crop)
            ball = detect_ball(model, crop, (0, 0, w, h), TEACHER_IMGSZ)
            rows.append({"image": image, "video_file": video_file, "frame": index,
                         "split": split_of[index], **(ball or {})})
        print(f"{video_file}: {len(video_plan)} crops")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from ultralytics import YOLO

    plan = pd.read_csv(BALL_TRAIN_PLAN_PATH)
    for split in plan["split"].unique():
        (BALL_CROPS_DIR / split).mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODELS_DIR / f"{TEACHER}.pt"))
    labels = save_crops(plan, model)
    labels.to_csv(BALL_PSEUDO_PATH, index=False)
    print(f"Saved {len(labels)} crops, ball found in {labels['conf'].notna().mean():.0%}")