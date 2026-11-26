from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Flask, abort, jsonify, request, send_file

from swish_or_brick.config import MAKE_MISS_PATH, RELEASES_PATH, VIDEO_DIR
from swish_or_brick.video import find_video

LABELS = {"made", "missed", "not_a_shot", "cant_tell", "conflict"}
BEFORE_S = 1.0
AFTER_RELEASE_S = 3.0
COLUMNS = ["shot_id", "label", "labeled_at"]


def shot_id(video_file, start_frame):
    return f"{video_file}:{start_frame}"

def shots_to_label(releases):
    shots = releases[releases["release_ms"].notna()]
    shots = shots.sort_values(["video_file", "start_ms"]).reset_index(drop=True)
    ids = [shot_id(v,f) for v, f in zip(shots["video_file"], shots["start_frame"])]
    return shots.assign(shot_id=ids)

def read_labels(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=COLUMNS)

def remaining(shots, labels):
    return shots[~shots["shot_id"].isin(labels["shot_id"])]

def add_label(path, shot, label):
    time = datetime.now().isoformat(timespec="seconds")
    row = pd.DataFrame([{"shot_id": shot, "label": label, "labeled_at": time}])
    row.to_csv(path, mode="a", header=not path.exists(), index=False)

def undo_last(path):
    labels = read_labels(path)
    if labels.empty:
        return None
    labels.iloc[:-1].to_csv(path, index=False)
    return labels["shot_id"].iloc[-1]

def clip(shot):
    return {
        "shot_id": shot["shot_id"],
        "video_file": shot["video_file"],
        "start_s": shot["start_ms"] / 1000 - BEFORE_S,
        "end_s": shot["release_ms"] / 1000 + AFTER_RELEASE_S,
    }


def create_app(shots, labels_path, video_paths):
    app = Flask(__name__)
    ids = set(shots["shot_id"])

    def next_shot():
        todo = remaining(shots, read_labels(labels_path))
        if todo.empty:
            return jsonify(done=True)
        return jsonify(done=False, left=len(todo), total=len(shots), **clip(todo.iloc[0]))

    @app.get("/")
    def page():
        return send_file(Path(__file__).parent / "labeler.html")

    @app.get("/api/next")
    def get_next():
        return next_shot()

    @app.post("/api/label")
    def label():
        data = request.get_json()
        if data["shot_id"] not in ids or data["label"] not in LABELS:
            abort(400)
        add_label(labels_path, data["shot_id"], data["label"])
        return next_shot()

    @app.post("/api/undo")
    def undo():
        undo_last(labels_path)
        return next_shot()

    @app.get("/video/<name>")
    def video(name):
        if name not in video_paths:
            abort(404)
        return send_file(video_paths[name], conditional=True)

    return app


if __name__ == "__main__":
    shots = shots_to_label(pd.read_csv(RELEASES_PATH))
    video_paths = {v: find_video(v, VIDEO_DIR) for v in shots["video_file"].unique()}
    print("Open http://127.0.0.1:5000")
    create_app(shots, MAKE_MISS_PATH, video_paths).run(port=5000)