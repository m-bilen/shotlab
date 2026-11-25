import pandas as pd

from swish_or_brick.ball import detect_in_plan, window_plan
from swish_or_brick.config import BALL_PRETRAINED_PATH, SHOTS_PATH
from swish_or_brick.shots import isolated_shots

WINDOW_BEFORE_MS = 400
WINDOW_AFTER_MS = 200
CONFIGS = [("yolo26n", 640), ("yolo26s", 640), ("yolo26m", 640), ("yolo26s", 1088)]
SHOTS_PER_GROUP = 10
SEED = 0

def plan_frames(shots):
    groups = isolated_shots(shots).groupby(["session_id", "shooter"])
    sample = groups.sample(n=SHOTS_PER_GROUP, random_state=SEED)
    sample = sample.assign(
        window_start_ms=sample["wrist_top_ms"] - WINDOW_BEFORE_MS,
        window_end_ms=sample["wrist_top_ms"] + WINDOW_AFTER_MS,
    )
    return window_plan(sample)



if __name__ == "__main__":
    shots = pd.read_csv(SHOTS_PATH)
    plan = plan_frames(shots)
    print(f"{len(plan)} frames from {plan.groupby(['video_file', 'wrist_top_frame']).ngroups} shots")
    result = detect_in_plan(plan, CONFIGS)
    result.to_parquet(BALL_PRETRAINED_PATH, index=False)
    print(f"Saved {len(result)} rows to {BALL_PRETRAINED_PATH}")