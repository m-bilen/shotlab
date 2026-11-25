import pandas as pd

from swish_or_brick.ball_train_data import HOLDOUT_VIDEO, split_shots, spread_index, sample_shots, train_window

def test_spread_index():
    assert spread_index(10, 4) == [0, 3, 6, 9]
    assert spread_index(2, 5) == [0, 1]


def test_split_shots_drops_exam_and_holds_out_video():
    shots = pd.DataFrame({
        "video_file": ["a.mp4", "a.mp4", HOLDOUT_VIDEO, HOLDOUT_VIDEO],
        "start_frame": [10, 20, 30, 40],
    })
    exam = pd.DataFrame({"video_file": ["a.mp4", HOLDOUT_VIDEO], "start_frame": [20, 40]})
    train, val = split_shots(shots, exam)
    assert train["start_frame"].tolist() == [10]
    assert val["start_frame"].tolist() == [30]

def test_sample_shots_counts_per_group():
    train = pd.DataFrame({"session_id": ["S01"] * 20 + ["S02"] * 20, "shooter": "mehmet", "start_frame": range(40)})
    val = pd.DataFrame({"session_id": "S08", "shooter": ["mehmet"] * 25 + ["father"] * 25, "start_frame": range(50)})
    sampled = sample_shots(train, val)
    assert sampled[sampled["split"] == "train"].groupby("session_id").size().tolist() == [14, 14]
    assert sampled[sampled["split"] == "val"].groupby("shooter").size().tolist() == [20, 20]

def test_train_window_follows_shot_start():
    shots = pd.DataFrame({"start_ms": [1000.0], "end_ms": [3000.0]})
    window = train_window(shots)
    assert window[["window_start_ms", "window_end_ms"]].values.tolist() == [[700.0, 1500.0]]