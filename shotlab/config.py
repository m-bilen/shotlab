from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

SESSIONS_PATH = DATA_DIR / "sessions.csv"
ROIS_PATH = DATA_DIR / "rois.csv"
SHOTS_PATH = DATA_DIR / "shots.csv"
MANUAL_PATH = DATA_DIR / "release_frames_manual.csv"
QUALITY_REPORT_PATH = DATA_DIR / "quality_report.csv"
LANDMARKS_DIR = DATA_DIR / "landmarks"
