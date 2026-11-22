import pandas as pd
import pytest

from shotlab.release_eval import manual_shots, reviewed_shots


def test_manual_shots_finds_covering_shot():
    shots = pd.DataFrame({"video_file": ["a.mp4", "a.mp4"], "start_frame": [10, 50], "end_frame": [30, 70]})
    manual = pd.DataFrame({"video_file": ["a.mp4"], "release_frame": [55]})
    result = manual_shots(shots, manual)
    assert result[["start_frame", "release_frame"]].values.tolist() == [[50, 55]]


def test_reviewed_shots_fails_on_unknown_shot():
    shots = pd.DataFrame({"video_file": ["a.mp4"], "start_frame": [10]})
    reviewed = pd.DataFrame({"video_file": ["a.mp4"], "start_frame": [99], "label": ["rebound"]})
    with pytest.raises(ValueError):
        reviewed_shots(shots, reviewed)