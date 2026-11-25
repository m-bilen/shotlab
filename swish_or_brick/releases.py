import pandas as pd

from swish_or_brick.config import BALL_ALL_DIR, RELEASES_PATH, SESSIONS_PATH, SHOTS_PATH
from swish_or_brick.release import add_fingertip
from swish_or_brick.release_report import releases

RELEASE_THRESHOLD = 1.15


def release_counts(sessions, result):
    keys = ["session_id", "shooter"]
    table = sessions.groupby(keys)[["shots"]].sum()
    table["detected"] = result.groupby(keys).size()
    table["released"] = result.groupby(keys)["release_ms"].count()
    return table.fillna(0).astype(int)


if __name__ == "__main__":
    ball = pd.concat([pd.read_parquet(p) for p in sorted(BALL_ALL_DIR.glob("*.parquet"))], ignore_index=True)
    found = releases(add_fingertip(ball), RELEASE_THRESHOLD)
    result = pd.read_csv(SHOTS_PATH).merge(found, on=["video_file", "start_frame"], how="left")
    result.to_csv(RELEASES_PATH, index=False)
    print(release_counts(pd.read_csv(SESSIONS_PATH), result))