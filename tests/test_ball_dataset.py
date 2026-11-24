import pandas as pd

from shotlab.ball_dataset import label_text, yolo_line


def test_yolo_line_normalizes_box():
    assert yolo_line(50, 100, 20, 200, 400) == "0 0.250000 0.250000 0.100000 0.050000"


def test_yolo_line_clips_box_at_edge():
    assert yolo_line(195, 100, 20, 200, 400) == "0 0.962500 0.250000 0.075000 0.050000"


def test_label_text_is_empty_without_ball():
    row = pd.Series({"ball": False, "ball_x": None, "ball_y": None, "ball_size": None})
    assert label_text(row, 200, 400) == ""