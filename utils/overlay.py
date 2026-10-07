import math

import numpy as np


def _pixelate_roi(roi: np.ndarray, factor: int) -> np.ndarray:
    """Return a pixelated copy of ROI using pure numpy nearest-upsample."""
    h, w = roi.shape[:2]
    factor = max(1, int(factor))
    small = roi[::factor, ::factor]
    up = np.repeat(np.repeat(small, factor, axis=0), factor, axis=1)
    return up[:h, :w]


def draw_square(frame: np.ndarray, frame_idx: int) -> np.ndarray:
    """Draw a bouncing white square overlay on a BGRA frame (in-place)."""
    h, w = frame.shape[:2]
    size = max(8, min(h, w) // 6)

    # Two sine waves with different periods produce a smooth non-repeating path.
    cx = int((w - size) / 2 * (1 + math.sin(frame_idx * 2 * math.pi / 300))) + size // 2
    cy = int((h - size) / 2 * (1 + math.sin(frame_idx * 2 * math.pi / 220))) + size // 2
    x1, x2 = max(0, cx - size // 2), min(w, cx + size // 2)
    y1, y2 = max(0, cy - size // 2), min(h, cy + size // 2)

    if x2 <= x1 or y2 <= y1:
        return frame

    roi = frame[y1:y2, x1:x2]
    factor = max(1, min(8, min(roi.shape[0], roi.shape[1]) // 12))
    frame[y1:y2, x1:x2] = _pixelate_roi(roi, factor)

    # Draw white border (BGRA).
    border = 3
    frame[y1:y1 + border, x1:x2] = (255, 255, 255, 255)
    frame[y2 - border:y2, x1:x2] = (255, 255, 255, 255)
    frame[y1:y2, x1:x1 + border] = (255, 255, 255, 255)
    frame[y1:y2, x2 - border:x2] = (255, 255, 255, 255)
    return frame
