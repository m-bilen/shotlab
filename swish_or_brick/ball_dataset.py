import shutil

import cv2
import pandas as pd

from swish_or_brick.config import BALL_CROPS_DIR, BALL_DATASET_DIR, BALL_LABELS_PATH

COLAB_PATH = "/content/ball_dataset"


def yolo_line(ball_x, ball_y, ball_size, width, height):
    half = ball_size / 2
    x1, x2 = max(0.0, ball_x - half), min(width, ball_x + half)
    y1, y2 = max(0.0, ball_y - half), min(height, ball_y + half)
    cx = (x1 + x2) / 2 / width
    cy = (y1 + y2) / 2 / height
    return f"0 {cx:.6f} {cy:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"

def label_text(row, width, height):
    if not row.ball:
        return ""
    return yolo_line(row.ball_x, row.ball_y, row.ball_size, width, height) + "\n"


def data_yaml(path=COLAB_PATH):
    return f"path: {path}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n"

def export(labels, out_dir):
    for row in labels.itertuples():
        split, name = row.image.split("/")
        image_dir = out_dir / "images" / split
        label_dir = out_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        source = BALL_CROPS_DIR / row.image
        height, width = cv2.imread(str(source)).shape[:2]
        shutil.copy(source, image_dir / name)
        (label_dir / name.replace(".jpg", ".txt")).write_text(label_text(row, width, height))
    (out_dir / "data.yaml").write_text(data_yaml())


if __name__ == "__main__":
    labels = pd.read_csv(BALL_LABELS_PATH)
    if BALL_DATASET_DIR.exists():
        shutil.rmtree(BALL_DATASET_DIR)
    export(labels, BALL_DATASET_DIR)
    archive = shutil.make_archive(str(BALL_DATASET_DIR), "zip", BALL_DATASET_DIR)
    print(f"{len(labels)} images, {int(labels['ball'].sum())} with a ball -> {archive}")