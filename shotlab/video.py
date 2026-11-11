import cv2


def video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_s": frame_count / fps,
    }

def read_frames(path, start_frame = 0):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES , start_frame)
    index = start_frame
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield index, frame 
            index += 1
    finally: 
        cap.release()

def read_frame(path, index):
    for _, frame in read_frames(path, start_frame=index):
        return frame
    raise ValueError(f"Frame {index} not found in { path }")


def play(path, start_s = 0.0):
    fps = video_info(path)["fps"]
    start_frame = int(start_s * fps)
    delay_ms = max(1, int(1000 / fps))
    paused = False
    for index, frame in read_frames(path, start_frame):
        small = cv2.resize(frame, (960,540))
        text = f"frame {index} t = {index / fps:.2f}s"
        cv2.putText(small, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 255), 2)
        cv2.imshow("ShotLab", small)
        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys

    video_path = sys.argv[1]
    start_s = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

    for key, value in video_info(video_path).items():
        print(f"{key}: {value}")

    play(video_path, start_s)