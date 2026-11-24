import numpy as np
import pandas as pd
import pytest

from shotlab.ball import frames_between, ball_region, ball_class
from shotlab.ball_report import add_phase, detection_rates

def test_frames_between_uses_time():
    frames = np.arange(100)
    assert frames_between(frames, frames * 1000 / 60, 100, 700) == list(range(6, 43))
    assert frames_between(frames, frames * 1000 / 30, 100, 700) == list(range(3, 22))

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

class FakeModel:
    def __init__(self, names):
        self.names = names


def test_ball_class_finds_ball_in_any_model():
    assert ball_class(FakeModel({0: "person", 32: "sports ball"})) == 32
    assert ball_class(FakeModel({0: "ball"})) == 0
    with pytest.raises(ValueError):
        ball_class(FakeModel({0: "person"}))