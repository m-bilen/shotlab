import numpy as np

WINDOW_BEFORE_MS = 400
WINDOW_AFTER_MS = 200
BALL_CLASS = 32  # "sports ball" in COCO
IMGSZ = 1088

def window_frames(frames, t_ms, top_frame, before_ms=WINDOW_BEFORE_MS, after_ms=WINDOW_AFTER_MS):
    frames = np.asarray(frames)
    t_ms = np.asarray(t_ms)
    t_top = t_ms[frames == top_frame][0]
    inside = (t_ms >= t_top - before_ms) & (t_ms <= t_top + after_ms)
    return frames[inside].tolist()

def ball_region(roi):
    x, y, size, _ = roi
    top = max(0, y - size)
    return (x, top, size, y + size - top)

def detect_ball(model, frame_bgr, region, imgsz=IMGSZ, conf=0.1):
    x, y, w, h = region
    crop = frame_bgr[y:y + h, x:x + w]
    result = model.predict(crop, imgsz=imgsz, classes=[BALL_CLASS], conf=conf, verbose=False)[0]
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

if __name__ == "__main__":
    import sys
    from pathlib import Path

    import cv2
    from ultralytics import YOLO

    from shotlab.config import MODELS_DIR
    from shotlab.roi import get_roi
    from shotlab.video import read_frame, resize_for_display

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
    cv2.imshow("ShotLab - ball", resize_for_display(frame)[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()