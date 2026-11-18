import numpy as np
import pandas as pd

from shotlab.shots import detect_shots, find_runs, merge_runs


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