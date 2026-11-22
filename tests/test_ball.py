import numpy as np
import pandas as pd

from shotlab.ball import window_frames, ball_region
from shotlab.ball_report import add_phase, detection_rates

def test_window_frames_60_fps():
    frames = np.arange(100)
    t_ms = frames * 1000 / 60
    result = window_frames(frames, t_ms, top_frame=30)
    assert result[0] == 6
    assert result[-1] == 42


def test_window_frames_same_time_at_30_fps():
    frames = np.arange(50)
    t_ms = frames * 1000 / 30
    result = window_frames(frames, t_ms, top_frame=15)
    assert result[0] == 3
    assert result[-1] == 21

def test_ball_region_extends_up():
    assert ball_region((300, 500, 600, 600)) == (300, 0, 600, 1100)
    assert ball_region((300, 800, 200, 200)) == (300, 600, 200, 400)


def test_add_phase_uses_time_from_wrist_top():
    result = pd.DataFrame({
        "video_file": "a.mp4",
        "wrist_top_frame": 30,
        "frame": [6, 24, 30, 42],
        "t_ms": [100.0, 400.0, 500.0, 700.0],
    })
    assert add_phase(result)["phase"].tolist() == ["hand", "release", "flight", "flight"]


def test_detection_rates_counts_missing_as_not_found():
    result = pd.DataFrame({
        "session_id": "S01",
        "model": "yolo26n",
        "imgsz": 640,
        "conf": [0.9, 0.1, None, 0.5],
    })
    assert detection_rates(result, ["session_id"]).iloc[0, 0] == 50.0