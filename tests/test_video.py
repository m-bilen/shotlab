import cv2
import numpy as np

from shotlab.video import read_frame

def make_test_video(path, n_frames):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64,48))
    for i in range(n_frames):
        frame = np.full((48,64,3), i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()

def test_read_frame_returns_requested_frame(tmp_path):
    path = tmp_path / "test.avi"
    make_test_video(path, 20)

    frame = read_frame(str(path), 7)
    assert abs(frame.mean()- 70) < 3
    