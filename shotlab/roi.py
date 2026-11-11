from pathlib import Path

import cv2
import pandas as pd

from shotlab.geometry import square_roi
from shotlab.video import read_frame

ROIS_PATH = Path("data/rois.csv")

def select_roi(frame):
    window = "Select shooter area"
    scale = frame.shape[1] / 960
    small = cv2.resize(frame, (960, 540))
    x, y, w, h = cv2.selectROI(window, small)
    cv2.destroyWindow(window)
    roi = (int(x * scale), int(y * scale), int(w * scale), int(h * scale))
    return square_roi(roi, frame.shape[1], frame.shape[0])

def load_rois(path=ROIS_PATH):
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame(columns=["video_file", "x", "y", "size"])

def save_roi(video_file, roi, path=ROIS_PATH):
    rois = load_rois(path)
    rois = rois[rois["video_file"] != video_file]
    x, y, size, _ = roi
    new_row = pd.DataFrame([{"video_file": video_file, "x": x, "y": y, "size": size}])
    rois = new_row if rois.empty else pd.concat([rois, new_row], ignore_index=True)
    rois.to_csv(path, index=False)

def get_roi(video_file, path=ROIS_PATH):
    rois = load_rois(path)
    row = rois[rois["video_file"] == video_file]
    if row.empty:
        raise KeyError(f"No ROI for {video_file}. Run: python -m shotlab.roi <video> <frame>")
    r = row.iloc[0]
    return (int(r["x"]), int(r["y"]), int(r["size"]), int(r["size"]))

if __name__ == "__main__":
    import sys

    video_path = Path(sys.argv[1])
    frame_index = int(sys.argv[2])

    frame = read_frame(str(video_path), frame_index)
    roi = select_roi(frame)
    save_roi(video_path.name, roi)
    print(f"Saved ROI {roi} for {video_path.name}")