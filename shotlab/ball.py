import numpy as np

WINDOW_BEFORE_MS = 400
WINDOW_AFTER_MS = 200

def window_frames(frames, t_ms, top_frame, before_ms=WINDOW_BEFORE_MS, after_ms=WINDOW_AFTER_MS):
    frames = np.asarray(frames)
    t_ms = np.asarray(t_ms)
    t_top = t_ms[frames == top_frame][0]
    inside = (t_ms >= t_top - before_ms) & (t_ms <= t_top + after_ms)
    return frames[inside].tolist()