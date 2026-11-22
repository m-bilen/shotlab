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
RELEASE_BALL_PATH = DATA_DIR / "ball_release_eval.parquet"

