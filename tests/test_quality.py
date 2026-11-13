import pandas as pd

from shotlab.quality import frame_interval_stats, jump_rate

def test_frame_interval_stats_finds_gap():
    df = pd.DataFrame({"t_ms": [0, 16.7, 33.3, 66.7, 83.3]})
    stats = frame_interval_stats(df)
    assert stats["gaps_over_25ms"] == 1
    assert stats["max_gap_ms"] == 33.4


def test_jump_rate_counts_big_steps():
    df = pd.DataFrame({
        "right_wrist_x": [0, 1, 2, 100, 101],
        "right_wrist_y": [0, 0, 0, 0, 0]
    })
    assert jump_rate(df, "right_wrist") == 25.0