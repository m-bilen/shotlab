import time

import cv2
import numpy as np

from swish_or_brick.ball import detect_ball
from swish_or_brick.config import BALL_CROPS_DIR, MODELS_DIR

MODELS = ["yolo26m", "yolo26n_ball"]
N_IMAGES = 200
WARMUP = 10


def time_ms(func):
    start = time.perf_counter()
    func()
    return (time.perf_counter() - start) * 1000


def benchmark(models, images, warmup=WARMUP):
    times = {name: [] for name in models}
    for i, image in enumerate(images):
        height, width = image.shape[:2]
        for name, model in models.items():
            ms = time_ms(lambda: detect_ball(model, image, (0, 0, width, height), 640))
            if i >= warmup:
                times[name].append(ms)
    return times


def speed_summary(times, slow="yolo26m", fast="yolo26n_ball"):
    medians = {name: float(np.median(values)) for name, values in times.items()}
    return medians, medians[slow] / medians[fast]


if __name__ == "__main__":
    from ultralytics import YOLO

    paths = sorted((BALL_CROPS_DIR / "val").glob("*.jpg"))[:N_IMAGES]
    images = [cv2.imread(str(path)) for path in paths]
    models = {name: YOLO(str(MODELS_DIR / f"{name}.pt")) for name in MODELS}
    medians, ratio = speed_summary(benchmark(models, images))
    print(f"{len(images) - WARMUP} images, median ms per image: {medians}")
    print(f"yolo26n_ball is {ratio:.1f}x faster than yolo26m on this CPU")