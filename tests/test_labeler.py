import numpy as np
import pandas as pd
import pytest

from swish_or_brick.labeler import add_label, create_app, read_labels, shots_to_label, undo_last,AFTER_RELEASE_S,read_label_rows


def make_releases():
    return pd.DataFrame({
        "video_file": ["b.mp4", "a.mp4", "a.mp4"],
        "start_frame": [5, 30, 10],
        "start_ms": [100.0, 3000.0, 1000.0],
        "release_ms": [200.0, np.nan, 1100.0],
    })


def test_shots_to_label_keeps_released_shots_in_order():
    assert shots_to_label(make_releases())["shot_id"].tolist() == ["a.mp4:10", "b.mp4:5"]


def test_labels_round_trip(tmp_path):
    path = tmp_path / "labels.csv"
    add_label(path, "a.mp4:10", "made")
    add_label(path, "b.mp4:5", "missed")
    assert undo_last(path) == "b.mp4:5"
    assert read_labels(path)["shot_id"].tolist() == ["a.mp4:10"]


def test_api_serves_next_unlabeled_shot(tmp_path):
    client = create_app(shots_to_label(make_releases()), tmp_path / "labels.csv", {}).test_client()
    first = client.get("/api/next").get_json()
    assert first["shot_id"] == "a.mp4:10"
    assert (first["start_s"], first["end_s"]) == pytest.approx((0.0, 1.1 + AFTER_RELEASE_S))
    after = client.post("/api/label", json={"shot_id": "a.mp4:10", "label": "made"}).get_json()
    assert after["shot_id"] == "b.mp4:5"
    assert client.post("/api/label", json={"shot_id": "a.mp4:10", "label": "maybe"}).status_code == 400

def test_read_labels_keeps_last_label(tmp_path):
    path = tmp_path / "labels.csv"
    add_label(path, "a.mp4:10", "made")
    add_label(path, "a.mp4:10", "not_a_shot")
    labels = read_labels(path)
    assert len(labels) == 1
    assert labels["label"].iloc[0] == "not_a_shot"

def test_undo_keeps_earlier_rows(tmp_path):
    path = tmp_path / "labels.csv"
    add_label(path, "a.mp4:10", "made")
    add_label(path, "a.mp4:10", "not_a_shot")
    add_label(path, "a.mp4:20", "missed")
    undo_last(path)
    assert len(read_label_rows(path)) == 2