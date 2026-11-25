import numpy as np
import pandas as pd
import pytest

from swish_or_brick.shots import (add_timing, covering_counts, detect_shots, find_runs,
                           keep_real_shots, merge_runs, summarize, isolated_shots)


def test_find_runs():
    mask = np.array([False, True, True, False, True])
    assert find_runs(mask) == [(1, 3), (4, 5)]


def test_merge_runs_joins_small_gaps():
    t = np.arange(50) * 10.0
    assert merge_runs([(0, 5), (8, 10), (30, 40)], t, 40) == [(0, 10), (30, 40)]


def test_detect_shots():
    df = pd.DataFrame({
        "frame": np.arange(100),
        "t_ms": np.arange(100) * 1000 / 60,
        "nose_y": 500.0,
        "right_wrist_y": 600.0,
        "right_wrist_v": 0.9,
    })
    df.loc[20:49, "right_wrist_y"] = 400.0
    df.loc[30:32, "right_wrist_y"] = 600.0
    df.loc[40, "right_wrist_y"] = 350.0
    df.loc[70:74, "right_wrist_y"] = 400.0

    shots = detect_shots(df)
    assert len(shots) == 1
    assert shots.iloc[0][["start_frame", "end_frame", "wrist_top_frame"]].tolist() == [20, 50, 40]

def test_detect_shots_same_at_30_fps():
    df = pd.DataFrame({
        "frame": np.arange(50),
        "t_ms": np.arange(50) * 1000 / 30,
        "nose_y": 500.0,
        "right_wrist_y": 600.0,
        "right_wrist_v": 0.9
    })
    df.loc[10:24, "right_wrist_y"] = 400.0
    df.loc[16, "right_wrist_y"] = 600.0
    df.loc[35:37, "right_wrist_y"] = 400.0
    shots = detect_shots(df)
    assert len(shots) == 1
    assert shots.iloc[0][["start_frame", "end_frame"]].tolist() == [10,25] 

def test_keep_real_shots():
    shots = pd.DataFrame({"elbow_max": [175.0, 120.0, 150.0, np.nan]})
    assert keep_real_shots(shots)["elbow_max"].tolist() == [175.0, 150.0]

def test_add_timing_sorts_before_gaps():
    shots = pd.DataFrame({
        "video_file": ["a.mp4"] * 3,
        "shooter": ["mehmet"] * 3,
        "start_ms": [20000.0, 0.0, 10000.0],
        "end_ms": [21000.0, 1000.0, 11500.0],
    })
    result = add_timing(shots)
    assert result["start_ms"].tolist() == [0.0, 10000.0, 20000.0]
    assert result["duration_s"].tolist() == pytest.approx([1.0, 1.5, 1.0])
    assert result["gap_s"].tolist()[1:] == pytest.approx([10.0, 10.0])

def test_summarize_shows_zero_for_empty_session():
    sessions = pd.DataFrame({
        "session_id": ["S01", "S02"],
        "shooter": ["mehmet", "mehmet"],
        "shots": [2, 1],
    })
    shots = add_timing(pd.DataFrame({
        "video_file": ["a.mp4", "a.mp4"],
        "session_id": ["S01", "S01"],
        "shooter": ["mehmet", "mehmet"],
        "start_ms": [0.0, 10000.0],
        "end_ms": [1000.0, 11000.0],
    }))
    summary = summarize(sessions, shots)
    assert summary.loc[("S01", "mehmet"), "found"] == 2
    assert summary.loc[("S02", "mehmet"), "found"] == 0

def test_covering_counts():
    shots = pd.DataFrame({
        "video_file": ["a.mp4", "a.mp4", "b.mp4"],
        "start_frame": [10, 15, 10],
        "end_frame": [20, 30, 20],
    })
    manual = pd.DataFrame({
        "video_file": ["a.mp4", "a.mp4", "a.mp4", "b.mp4"],
        "release_frame": [12, 17, 20, 25],
    })
    assert covering_counts(shots, manual).tolist() == [1, 2, 1, 0]

def test_isolated_shots_drops_both_sides_of_short_gap():
    shots = add_timing(pd.DataFrame({
        "video_file": ["a.mp4"] * 4,
        "shooter": ["mehmet"] * 4,
        "start_ms": [0.0, 10000.0, 12000.0, 22000.0],
        "end_ms": [1000.0, 11000.0, 13000.0, 23000.0],
    }))
    assert isolated_shots(shots)["start_ms"].tolist() == [0.0, 22000.0]