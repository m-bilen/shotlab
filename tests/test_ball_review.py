import pytest
import pandas as pd

from swish_or_brick.ball_review import box_from_roi, remaining


def test_box_from_roi_undoes_display_scale():
    assert box_from_roi((10, 20, 30, 40), 0.5) == pytest.approx((50, 80, 70))

def test_remaining_skips_labeled_images():
    pseudo = pd.DataFrame({"image": ["train/a.jpg", "train/b.jpg", "val/c.jpg"]})
    labels = pd.DataFrame({"image": ["train/b.jpg"]})
    assert remaining(pseudo, labels)["image"].tolist() == ["train/a.jpg", "val/c.jpg"]