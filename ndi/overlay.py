import math

import numpy as np

GREEN = np.array((0, 255, 0), np.float32)  # BGR
# The gaze path's shades, BGR: pale green for the soonest step to deep green for the last one.
PATH_SOON = np.array((170, 235, 170), np.float32)
PATH_LAST = np.array((40, 190, 40), np.float32)


def draw_gaze(frame: np.ndarray, uv) -> np.ndarray:
    """Draw the gaze prediction at the chosen step (ring + centre dot) on a BGRA frame (in-place).

    `uv` is the gaze point in [0, 1] frame coordinates (origin top-left), or None to draw nothing.
    """
    if uv is None:
        return frame
    h, w = frame.shape[:2]
    cx, cy = uv[0] * (w - 1), uv[1] * (h - 1)
    radius = max(6, min(h, w) // 36)  # 15 px at 960x544, 60 at 4K
    thickness = max(2, radius // 6)

    # only the square around the ring (the whole frame took 24 ms at 4K)
    reach = radius + thickness
    x0, x1 = max(0, math.floor(cx - reach)), min(w, math.ceil(cx + reach) + 1)
    y0, y1 = max(0, math.floor(cy - reach)), min(h, math.ceil(cy + reach) + 1)
    if x0 >= x1 or y0 >= y1:
        return frame
    ys, xs = np.ogrid[y0:y1, x0:x1]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    ring = np.abs(dist - radius) <= thickness / 2
    dot = dist <= thickness
    frame[y0:y1, x0:x1][ring | dot] = (0, 255, 0, 255)  # green (BGRA)
    return frame


def draw_fovea(frame: np.ndarray, box) -> np.ndarray:
    """Draw the circle the SR model upscales (foveated model) as a thin dashed outline on a
    BGRA frame (in-place). `box` is its square (x0, y0, x1, y1) in pixels, or None."""
    if box is None:
        return frame
    x0, y0, x1, y1 = box
    cx, cy, r = (x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2, (x1 - x0) / 2
    region = frame[y0:y1, x0:x1]
    ys, xs = np.ogrid[y0:y1, x0:x1]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    angle = np.arctan2(ys - cy, xs - cx)
    ring = (np.abs(dist - r) <= 1.5) & (np.sin(angle * 24) > 0)  # 24 dashes
    region[ring] = (255, 255, 0, 255)  # cyan (BGRA)
    return frame


def draw_gaze_path(frame: np.ndarray, path) -> np.ndarray:
    """Draw all predicted steps (a list of (u, v) in [0, 1], step 1 first) on a BGRA frame
    (in-place): a line through small dots in shades of green, pale for the soonest step to deep
    green for the last, which is drawn just stronger (opaque, a pixel larger). None draws nothing."""
    if not path:
        return frame
    h, w = frame.shape[:2]
    pts = np.array([(u * (w - 1), v * (h - 1)) for u, v in path], np.float32)
    n = len(pts)
    shades = PATH_SOON + (PATH_LAST - PATH_SOON) * (np.arange(n) / max(n - 1, 1))[:, None]
    r = max(2, min(h, w) // 100)  # dot radius: 5 px at 960x544

    def blend(ys, xs, colour, a):
        ok = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
        ys, xs = ys[ok], xs[ok]
        px = frame[ys, xs, :3].astype(np.float32)
        frame[ys, xs, :3] = (px + (colour - px) * a + 0.5).astype(np.uint8)

    for i in range(n - 1):  # line segments, 2 px, in the shade of the step they lead to
        (x0, y0), (x1, y1) = pts[i], pts[i + 1]
        steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        t = np.linspace(0, 1, steps)
        xs = np.round(x0 + (x1 - x0) * t).astype(int)
        ys = np.round(y0 + (y1 - y0) * t).astype(int)
        blend(ys, xs, shades[i + 1], 0.45)
        if abs(x1 - x0) >= abs(y1 - y0):  # second pixel across the line
            blend(ys + 1, xs, shades[i + 1], 0.45)
        else:
            blend(ys, xs + 1, shades[i + 1], 0.45)

    def disc(radius):
        oy, ox = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        inside = (ox ** 2 + oy ** 2) <= radius * radius
        return oy[inside], ox[inside]

    dot, last = disc(r), disc(r + 1)
    for i, (x, y) in enumerate(pts):
        oy, ox = last if i == n - 1 else dot
        blend(round(y) + oy, round(x) + ox, shades[i], 1.0 if i == n - 1 else 0.8)
    return frame


def draw_gaze_prediction(frame: np.ndarray, uv, path=None) -> np.ndarray:
    """The gaze prediction: all predicted steps (`path`) and the ring on the chosen step (`uv`)."""
    draw_gaze_path(frame, path)
    return draw_gaze(frame, uv)
