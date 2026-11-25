from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

SESSIONS_PATH = DATA_DIR / "sessions.csv"
ROIS_PATH = DATA_DIR / "rois.csv"
SHOTS_PATH = DATA_DIR / "shots.csv"
MANUAL_PATH = DATA_DIR / "release_frames_manual.csv"
QUALITY_REPORT_PATH = DATA_DIR / "quality_report.csv"
LANDMARKS_DIR = DATA_DIR / "landmarks"
VIDEO_DIR = Path(os.environ.get("SHOTLAB_VIDEO_DIR", "C:/Shotlab_Video/ana_veri"))
BALL_PRETRAINED_PATH = DATA_DIR / "ball_pretrained.parquet"
REVIEWED_PATH = DATA_DIR / "reviewed_shots.csv"

def release_ball_path(model):
    return DATA_DIR / f"ball_release_eval_{model}.parquet"

BALL_TRAIN_PLAN_PATH = DATA_DIR / "ball_train_plan.csv"
BALL_CROPS_DIR = DATA_DIR / "ball_crops"
BALL_PSEUDO_PATH = DATA_DIR / "ball_pseudo_labels.csv"
BALL_LABELS_PATH = DATA_DIR / "ball_labels.csv"
BALL_DATASET_DIR = DATA_DIR / "ball_dataset"
BALL_ALL_DIR = DATA_DIR / "ball_all"
RELEASES_PATH = DATA_DIR / "releases.csv"

