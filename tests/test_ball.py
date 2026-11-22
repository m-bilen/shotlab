import numpy as np

from shotlab.ball import window_frames


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
