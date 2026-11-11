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

