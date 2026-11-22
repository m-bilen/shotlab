import pandas as pd

from shotlab.config import BALL_PRETRAINED_PATH
from shotlab.ball import MIN_CONF

PHASE_BINS = [-400.5, -150, 0, 200.5]
PHASE_NAMES = ["hand", "release", "flight"]

def add_phase(result):
    top = result[result["frame"] == result["wrist_top_frame"]]
    top = top[["video_file", "wrist_top_frame", "t_ms"]].drop_duplicates(["video_file", "wrist_top_frame"])
    result = result.merge(top.rename(columns={"t_ms": "t_top_ms"}), on=["video_file", "wrist_top_frame"])
    dt = result["t_ms"] - result["t_top_ms"]
    result["phase"] = pd.cut(dt, bins=PHASE_BINS, labels=PHASE_NAMES, right=False)
    return result

def detection_rates(result, by, conf=MIN_CONF):
    found = result["conf"].fillna(0) >= conf
    rates = result.assign(found=found).groupby(by + ["model", "imgsz"], observed=True)["found"].mean()
    return (rates * 100).round(1).unstack(["model", "imgsz"])

if __name__ == "__main__":
    result = add_phase(pd.read_parquet(BALL_PRETRAINED_PATH))
    pd.set_option("display.width", 200)

    print("Median ms per frame")
    print(result.groupby(["model", "imgsz"])["ms"].median().round(1))
    print("\nDetection rate (%) by phase")
    print(detection_rates(result, ["phase"]))
    print("\nDetection rate (%) by session and shooter")
    print(detection_rates(result, ["session_id", "shooter"]))