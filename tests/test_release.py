import numpy as np
import pandas as pd
import pytest


from shotlab.release import release_time, shot_distance
from shotlab.release_report import choose_threshold, score, session_summary, blind_only, overall
from shotlab.releases import release_counts


def test_release_time_interpolates_between_frames():
    t = [0, 10, 20, 30, 40]
    d = [0.5, 0.6, 1.0, 1.6, 2.0]
    assert release_time(t, d, 1.2) == pytest.approx(23.333, abs=0.01)


def test_release_time_skips_missing_frames():
    t = [0, 10, 20, 30]
    d = [0.5, np.nan, 2.0, 2.2]
    assert release_time(t, d, 1.25) == pytest.approx(10.0)


def test_release_time_none_when_ball_comes_to_hand():
    t = [0, 10, 20, 30, 40]
    d = [3.0, 2.0, 1.0, 0.5, 0.5]
    assert release_time(t, d, 1.5) is None


def test_release_time_ignores_single_spike():
    t = [0, 10, 20, 30, 40, 50]
    d = [0.5, 3.0, 0.5, 0.5, 2.0, 2.2]
    assert release_time(t, d, 1.5, hold_ms=30) == pytest.approx(36.667, abs=0.01)


def test_shot_distance_uses_median_ball_size():
    df = pd.DataFrame({
        "conf": [0.9, 0.9, 0.1],
        "ball_x": [10.0, 30.0, 50.0],
        "ball_y": [0.0, 0.0, 0.0],
        "ball_size": [10.0, 20.0, 99.0],
        "right_index_x": [0.0, 0.0, 0.0],
        "right_index_y": [0.0, 0.0, 0.0],
    })
    dist = shot_distance(df)
    assert dist[:2] == pytest.approx([10 / 15, 30 / 15])
    assert np.isnan(dist[2])

def test_score_caps_errors_and_counts_missing():
    errors = pd.DataFrame({"error_ms": [10.0, -30.0, np.nan, 500.0]})
    assert score(errors) == pytest.approx(60.0)

def test_choose_threshold_picks_lowest_score():
    errors = {
        1.0: pd.DataFrame({"error_ms": [50.0, 50.0]}),
        1.5: pd.DataFrame({"error_ms": [5.0, -5.0]}),
    }
    assert choose_threshold(errors) == 1.5


def test_session_summary_counts_frames_by_fps():
    tested = pd.DataFrame({
        "session_id": ["S01", "S01", "S05", "S05"],
        "video_file": ["a.mp4", "a.mp4", "b.mp4", "b.mp4"],
        "threshold": 1.0,
        "error_ms": [30.0, np.nan, 30.0, 90.0],
    })
    fps = pd.Series({"a.mp4": 60, "b.mp4": 30})
    summary = session_summary(tested, fps)
    assert summary.loc["S01", "missed"] == 1
    assert summary.loc["S01", "within_2"] == 50
    assert summary.loc["S05", "within_2"] == 50


def test_blind_only_drops_relabeled_shots():
    tested = pd.DataFrame({
        "video_file": ["VID_20261112_104955.mp4", "VID_20261112_104955.mp4"],
        "release_frame": [41220, 5000],
    })
    assert blind_only(tested)["release_frame"].tolist() == [5000]


def test_overall_counts_within_two_frames():
    tested = pd.DataFrame({
        "video_file": ["a.mp4", "a.mp4", "a.mp4"],
        "error_ms": [10.0, 50.0, np.nan],
    })
    fps = pd.Series({"a.mp4": 60})
    assert overall(tested, fps) == {"shots": 3, "missed": 1, "within_2": 1}

def test_release_counts_per_session():
    sessions = pd.DataFrame({"session_id": ["S01", "S02"], "shooter": "mehmet", "shots": [3, 2]})
    result = pd.DataFrame({
        "session_id": ["S01", "S01", "S01", "S01"],
        "shooter": "mehmet",
        "release_ms": [100.0, 200.0, np.nan, 300.0],
    })
    table = release_counts(sessions, result)
    assert table.loc[("S01", "mehmet")].tolist() == [3, 4, 3]
    assert table.loc[("S02", "mehmet")].tolist() == [2, 0, 0]