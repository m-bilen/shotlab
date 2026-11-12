import cv2 
import mediapipe as mp
import numpy as np 
from shotlab.video import read_frame
from pathlib import Path

from shotlab.roi import get_roi

MODELS = {
    "full": "models/pose_landmarker_full.task",
    "heavy": "models/pose_landmarker_heavy.task",
}
DEFAULT_MODEL = "heavy"
vision = mp.tasks.vision
CONNECTIONS = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]


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
        points = np.array([
            [lm.x * img_w + x0, lm.y * img_h + y0, lm.visibility]
            for lm in landmarks
        ])
        poses.append(points)
    return poses

def draw_poses(frame_bgr, poses, roi=None):
    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (255, 0, 0), 3)

    for points in poses:
        for start, end in CONNECTIONS:
            p1 = (int(points[start, 0]), int(points[start, 1]))
            p2 = (int(points[end, 0]), int(points[end, 1]))
            cv2.line(frame_bgr, p1, p2, (255, 255, 255), 3)
        for px, py, _ in points:
            cv2.circle(frame_bgr, (int(px), int(py)), 6, (0, 0, 255), -1)
    return frame_bgr


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
    cv2.imshow("ShotLab - pose", cv2.resize(frame, (960, 540)))
    cv2.waitKey(0)
    cv2.destroyAllWindows
