import pytest

from swish_or_brick.ball_speed import speed_summary


def test_speed_summary_uses_medians():
    medians, ratio = speed_summary({"yolo26m": [300, 310, 290], "yolo26n_ball": [50, 60, 40]})
    assert medians == {"yolo26m": 300.0, "yolo26n_ball": 50.0}
    assert ratio == pytest.approx(6.0)