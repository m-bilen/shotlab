import numpy as np
import pandas as pd

from shotlab.shots import detect_shots, find_runs, merge_runs


def test_find_runs():
    mask = np.array([False, True, True, False, True])
    assert find_runs(mask) == [(1, 3), (4, 5)]


def test_merge_runs_joins_small_gaps():
    assert merge_runs([(0, 5), (8, 10), (30, 40)], 3) == [(0, 10), (30, 40)]


def test_detect_shots():
    df = pd.DataFrame({
        "frame": np.arange(100),
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
    assert shots.iloc[0].tolist() == [20, 50, 40]