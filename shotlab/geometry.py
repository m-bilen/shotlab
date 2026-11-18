import numpy as np



def square_roi(roi, frame_w, frame_h):
    x,y,w,h = roi
    side = min(max(w,h), frame_w, frame_h)
    cx = x + w / 2
    cy = y + h / 2

    x = int(round(cx - side / 2))
    y = int(round(cy -side / 2))
    x = max(0, min(x, frame_w - side))
    y = max(0, min(y, frame_h - side))
    return (x, y, side, side)

def angle(a, b, c):
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    cos = np.sum(ba * bc, axis=-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
