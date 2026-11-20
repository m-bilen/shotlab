import cv2 
import mediapipe as mp
import numpy as np 
from shotlab.video import read_frame, resize_for_display
from pathlib import Path
from shotlab.config import MODELS_DIR

from shotlab.roi import get_roi

MODELS = {
    "full": str(MODELS_DIR / "pose_landmarker_full.task"),
    "heavy": str(MODELS_DIR / "pose_landmarker_heavy.task"),
}
DEFAULT_MODEL = "heavy"
vision = mp.tasks.vision
CONNECTIONS = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]
RIGHT_SIDE = {12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}
LANDMARK_NAMES = [lm.name.lower() for lm in vision.PoseLandmark]
AXES = ("x", "y", "v")
VALUE_COLUMNS = [f"{name}_{axis}" for name in LANDMARK_NAMES for axis in AXES]


def create_landmarker(running_mode=vision.RunningMode.IMAGE, num_poses=2, model=DEFAULT_MODEL):
    options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODELS[model]),
        running_mode = running_mode,
        num_poses=num_poses,
    )
    return vision.PoseLandmarker.create_from_options(options)

def detect_poses(landmarker, frame_bgr, roi=None, timestamp_ms=None):
    x0, y0 = 0, 0
    if roi is not None:
        x0, y0, w, h = roi
        frame_bgr = frame_bgr[y0:y0 + h, x0:x0 + w]
    img_h, img_w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    if timestamp_ms is None:
        result = landmarker.detect(image)
    else:
        result = landmarker.detect_for_video(image, timestamp_ms)

    poses = []
    for landmarks in result.pose_landmarks:
        rows = []
        for lm in landmarks:
            values = {"x": lm.x * img_w + x0, "y": lm.y * img_h + y0, "v": lm.visibility}
            rows.append([values[axis] for axis in AXES])
        poses.append(np.array(rows))
    return poses

def draw_poses(frame_bgr, poses, roi=None):
    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (255, 0, 0), 3)

    for points in poses:
        for start, end in CONNECTIONS:
            p1 = (int(points[start, 0]), int(points[start, 1]))
            p2 = (int(points[end, 0]), int(points[end, 1]))
            right = start in RIGHT_SIDE and end in RIGHT_SIDE
            color = (0, 255, 0) if right else (255, 255, 255)
            cv2.line(frame_bgr, p1, p2, color, 3)
        for px, py, _ in points:
            cv2.circle(frame_bgr, (int(px), int(py)), 6, (0, 0, 255), -1)
    return frame_bgr

def points_from_table(df):
    return df[VALUE_COLUMNS].to_numpy().reshape(len(df), len(LANDMARK_NAMES), len(AXES))

if __name__ == "__main__":
    import sys

    video_path = sys.argv[1]
    frame_index = int(sys.argv[2])

    frame = read_frame(video_path, frame_index)
    roi = get_roi(Path(video_path).name)
    print(f"ROI: {roi}")
    with create_landmarker() as landmarker:
        poses = detect_poses(landmarker, frame, roi)
    print(f"Detected {len(poses)} pose(s)")
    draw_poses(frame, poses, roi)
    cv2.imshow("ShotLab - pose", resize_for_display(frame)[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows
