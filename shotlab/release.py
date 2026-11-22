import numpy as np

from shotlab.ball import MIN_CONF

HOLD_MS = 100

def shot_distance(df, min_conf=MIN_CONF):
    confident = df["conf"].fillna(0) >= min_conf
    size = df.loc[confident, "ball_size"].median()
    dx = df["ball_x"] - df["right_index_x"]
    dy = df["ball_y"] - df["right_index_y"]
    dist = np.hypot(dx, dy) / size
    return dist.where(confident).to_numpy()

def release_time(t_ms, dist, threshold, hold_ms=HOLD_MS):
    t_ms = np.asarray(t_ms, dtype=float)
    dist = np.asarray(dist, dtype=float)
    valid = ~np.isnan(dist)
    t, d = t_ms[valid], dist[valid]
    for i in range(1, len(d)):
        if d[i - 1] < threshold <= d[i]:
            after = d[(t >= t[i]) & (t <= t[i] + hold_ms)]
            if (after >= threshold).all():
                return t[i - 1] + (threshold - d[i - 1]) / (d[i] - d[i - 1]) * (t[i] - t[i - 1])
    return None