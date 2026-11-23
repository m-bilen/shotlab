import cv2
import pandas as pd

from shotlab.config import BALL_CROPS_DIR, BALL_LABELS_PATH, BALL_PSEUDO_PATH

DISPLAY_HEIGHT = 700
WINDOW = "ShotLab - ball review"
ENTER = 13
COLUMNS = ["image", "ball", "ball_x", "ball_y", "ball_size", "source"]


def load_labels(path=BALL_LABELS_PATH):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=COLUMNS)


def box_from_roi(roi, scale):
    x, y, w, h = (v / scale for v in roi)
    return x + w / 2, y + h / 2, (w + h) / 2


def teacher_label(row):
    return {"image": row.image, "ball": True, "ball_x": row.ball_x, "ball_y": row.ball_y,
            "ball_size": row.ball_size, "source": "teacher"}


def no_ball_label(row):
    return {"image": row.image, "ball": False, "ball_x": None, "ball_y": None,
            "ball_size": None, "source": "none"}


def manual_label(row, roi, scale):
    ball_x, ball_y, ball_size = box_from_roi(roi, scale)
    return {"image": row.image, "ball": True, "ball_x": ball_x, "ball_y": ball_y,
            "ball_size": ball_size, "source": "manual"}

def remaining(pseudo, labels):
    return pseudo[~pseudo["image"].isin(labels["image"])]

def draw(image, row, scale):
    small = cv2.resize(image, None, fx=scale, fy=scale)
    if pd.notna(row.conf):
        center = (int(row.ball_x * scale), int(row.ball_y * scale))
        cv2.circle(small, center, int(row.ball_size * scale / 2), (0, 255, 255), 2)
        text, color = f"teacher {row.conf:.2f}", (0, 255, 255)
    else:
        text, color = "teacher: no ball", (0, 0, 255)
    cv2.putText(small, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return small


def ask(row):
    image = cv2.imread(str(BALL_CROPS_DIR / row.image))
    scale = DISPLAY_HEIGHT / image.shape[0]
    shown = draw(image, row, scale)
    cv2.namedWindow(WINDOW)
    cv2.moveWindow(WINDOW, 0, 0)    
    cv2.imshow(WINDOW, shown)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            return None
        if key == ENTER and pd.notna(row.conf):
            return teacher_label(row)
        if key == ord("x"):
            return no_ball_label(row)
        if key == ord("r"):
            roi = cv2.selectROI(WINDOW, shown, showCrosshair=False)
            if roi[2] > 0 and roi[3] > 0:
                return manual_label(row, roi, scale)
            cv2.imshow(WINDOW, shown)


if __name__ == "__main__":
    pseudo = pd.read_csv(BALL_PSEUDO_PATH)
    labels = load_labels()
    todo = remaining(pseudo, labels)
    print(f"{len(labels)} done, {len(todo)} left")
    for row in todo.itertuples():
        label = ask(row)
        if label is None:
            break
        labels = pd.concat([labels, pd.DataFrame([label])], ignore_index=True)
        labels.to_csv(BALL_LABELS_PATH, index=False)
    cv2.destroyAllWindows()
    print(f"{len(labels)} done")