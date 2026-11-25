import numpy as np
import pandas as pd

from swish_or_brick.pose import AXES, LANDMARK_NAMES, VALUE_COLUMNS, points_from_table

def test_value_columns_layout():
    assert len(VALUE_COLUMNS) == len(LANDMARK_NAMES) * len(AXES)
    assert VALUE_COLUMNS[:3] == ["nose_x", "nose_y", "nose_v"]

def test_points_from_table_matches_column_names():
    values = np.arange(2 * len(VALUE_COLUMNS), dtype=float).reshape(2, -1)
    df = pd.DataFrame(values, columns=VALUE_COLUMNS)
    df.insert(0, "t_ms", [0.0, 16.7])
    df.insert(0, "frame", [0, 1])

    points = points_from_table(df)
    wrist = LANDMARK_NAMES.index("right_wrist")
    assert points[1, wrist, AXES.index("y")] == df.loc[1, "right_wrist_y"]