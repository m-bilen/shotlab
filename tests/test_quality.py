import pandas as pd
import pytest
import numpy as np
from shotlab.quality import frame_interval_stats, jump_rate

def test_frame_interval_stats_finds_gap():
    df = pd.DataFrame({"t_ms": [0, 16.7, 33.3, 66.7, 83.3]})
    stats = frame_interval_stats(df)
    assert stats["frame_gaps"] == 1
    assert stats["max_gap_ms"] == pytest.approx(33.4)



def make_track(fps, scale=1.0, glitch=False):
    t_ms = np.arange(fps) * 1000 / fps
    torso = 100 * scale
    wrist_x = 10 * torso * t_ms / 1000
    if glitch:
        wrist_x[fps // 2] += 5 * torso
    return pd.DataFrame({
        "t_ms": t_ms,
        "right_shoulder_x": 0.0, "right_shoulder_y": 0.0,
        "right_hip_x": 0.0, "right_hip_y": torso,
        "right_wrist_x": wrist_x, "right_wrist_y": 0.0,
    })

def test_fast_real_motion_is_not_a_jump():
    for fps in (30, 60):
        assert jump_rate(make_track(fps), "right_wrist") == 0.0

def test_glitch_is_a_jump_at_both_fps():
    for fps in (30, 60):
        assert jump_rate(make_track(fps, glitch=True), "right_wrist") > 0

def test_jump_rate_ignores_scale():
    small = jump_rate(make_track(60, scale=1.0, glitch=True), "right_wrist")
    big = jump_rate(make_track(60, scale=2.0, glitch=True), "right_wrist")
    assert small == pytest.approx(big)

def test_frame_interval_stats_at_30_fps():
    df = pd.DataFrame({"t_ms": np.arange(10) * 1000 / 30})
    assert frame_interval_stats(df)["frame_gaps"] == 0