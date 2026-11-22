import pandas as pd

from shotlab.ball import detect_in_plan, window_plan
from shotlab.config import MANUAL_PATH, RELEASE_BALL_PATH, REVIEWED_PATH, SHOTS_PATH
from shotlab.shots import covering_mask

CONFIGS = [("yolo26m", 640)]
MARGIN_MS = 200


def manual_shots(shots, manual):
    rows = []
    for r in manual.itertuples():
        shot = shots[covering_mask(shots, r.video_file, r.release_frame)].iloc[0]
        rows.append({**shot.to_dict(), "release_frame": r.release_frame})
    return pd.DataFrame(rows)


def reviewed_shots(shots, reviewed):
    result = shots.merge(reviewed, on=["video_file", "start_frame"])
    if len(result) != len(reviewed):
        raise ValueError("Some reviewed shots are not in shots.csv")
    return result

def eval_shots(shots, manual, reviewed):
    selected = pd.concat([manual_shots(shots, manual), reviewed_shots(shots, reviewed)], ignore_index=True)
    selected = selected.drop_duplicates(["video_file", "start_frame"])
    return selected.assign(
        window_start_ms=selected["start_ms"] - MARGIN_MS,
        window_end_ms=selected["end_ms"] + MARGIN_MS,
    )


if __name__ == "__main__":
    shots = pd.read_csv(SHOTS_PATH)
    selected = eval_shots(shots, pd.read_csv(MANUAL_PATH), pd.read_csv(REVIEWED_PATH))
    plan = window_plan(selected)
    print(f"{len(plan)} frames from {len(selected)} shots")
    result = detect_in_plan(plan, CONFIGS)
    result.to_parquet(RELEASE_BALL_PATH, index=False)
    print(f"Saved {len(result)} rows to {RELEASE_BALL_PATH}")