import numpy as np
import pandas as pd

from swish_or_brick.ball import window_plan
from swish_or_brick.config import BALL_TRAIN_PLAN_PATH, MANUAL_PATH, REVIEWED_PATH, SHOTS_PATH
from swish_or_brick.release_eval import  eval_shots
from swish_or_brick.shots import isolated_shots

HOLDOUT_VIDEO = "VID_20261115_111103.mp4"
TRAIN_SHOTS_PER_GROUP = 14
VAL_SHOTS_PER_SHOOTER = 20
FRAMES_PER_SHOT = 8
SEED = 1
TRAIN_BEFORE_MS = 300
TRAIN_AFTER_MS = 500






def split_shots(shots, exclude):
    keys = exclude[["video_file", "start_frame"]].drop_duplicates()
    merged = shots.merge(keys, on=["video_file", "start_frame"], how="left", indicator=True)
    free = merged[merged["_merge"] == "left_only"].drop(columns="_merge")
    train = free[free["video_file"] != HOLDOUT_VIDEO]
    val = free[free["video_file"] == HOLDOUT_VIDEO]
    return train, val

def spread_index(length, n):
    if length <= n:
        return list(range(length))
    return np.linspace(0, length - 1, n).round().astype(int).tolist()


def thin_plan(plan, n=FRAMES_PER_SHOT):
    parts = [shot.iloc[spread_index(len(shot), n)] for _, shot in plan.groupby(["video_file", "start_frame"])]
    return pd.concat(parts, ignore_index=True)


def sample_shots(train, val, seed=SEED):
    train = train.groupby(["session_id", "shooter"]).sample(n=TRAIN_SHOTS_PER_GROUP, random_state=seed)
    val = val.groupby("shooter").sample(n=VAL_SHOTS_PER_SHOOTER, random_state=seed)
    return pd.concat([train.assign(split="train"), val.assign(split="val")], ignore_index=True)

def train_window(shots):
    return shots.assign(
        window_start_ms=shots["start_ms"] - TRAIN_BEFORE_MS,
        window_end_ms=shots["start_ms"] + TRAIN_AFTER_MS,
    )

def training_plan(sampled):
    parts = []
    for split, shots in sampled.groupby("split"):
        parts.append(thin_plan(window_plan(train_window(shots))).assign(split=split))
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    shots = pd.read_csv(SHOTS_PATH)
    exam = eval_shots(shots, pd.read_csv(MANUAL_PATH), pd.read_csv(REVIEWED_PATH))
    train, val = split_shots(isolated_shots(shots), exam)
    plan = training_plan(sample_shots(train, val))
    plan.to_csv(BALL_TRAIN_PLAN_PATH, index=False)
    print(plan.groupby(["split", "session_id"]).size())