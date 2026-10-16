"""Gaze-guided ("foveated") 4x upscaling: the SR model where the viewer looks, bicubic elsewhere.

Every frame is upscaled with bicubic interpolation; the SR model then runs on a square LR
crop centred on the predicted gaze point, and its 4x result is blended back in with a
round, soft-edged mask (full SR in the centre, fading to bicubic at the edge), so no seam
shows. Only the SR's fine detail is used: on a small crop the SR model shifts the
colours (5-10 levels brighter in testing; on a whole frame it keeps them), so the crop's
low frequencies are taken from bicubic instead, and colour and brightness match exactly.
The SR model runs on every frame (k = 1): RIFE cannot interpolate between crops taken
at different positions.

Same push()/flush() interface as the other upscalers, plus the gaze points: push(frames,
gazes) returns every frame at once, and `last_boxes` holds each frame's fovea region in
output (HR) pixels, for drawing on the -Gaze source.
"""

import numpy as np
import torch
import torch.nn.functional as F

from upscale.bicubic import BicubicStream

SCALE = 4
# Share of the fovea's radius over which SR fades into bicubic.
FEATHER = 0.35
# Low-pass for the colour match: downsample by this factor (area) and back up (bilinear).
# 8 at 4x output is half the LR resolution, so the detail kept is finer than the LR frame holds.
LOWPASS = 8


def lowpass(images):
    """(N, H, W, 3) float32 -> blurred (N, H, W, 3): area-downsample by LOWPASS, bilinear back up."""
    t = torch.from_numpy(images).permute(0, 3, 1, 2)
    small = F.interpolate(t, scale_factor=1 / LOWPASS, mode="area")
    return F.interpolate(small, size=t.shape[-2:], mode="bilinear", align_corners=False).permute(0, 2, 3, 1).numpy()


def fovea_box(uv, width, height, size):
    """Square LR crop (x0, y0, x1, y1) of `size` pixels centred on gaze point `uv` in [0, 1]
    (frame centre when None), moved inside the frame at the edges."""
    u, v = uv if uv is not None else (0.5, 0.5)
    x0 = round(u * width - size / 2)
    y0 = round(v * height - size / 2)
    x0 = min(max(x0, 0), width - size)
    y0 = min(max(y0, 0), height - size)
    return x0, y0, x0 + size, y0 + size


def fovea_mask(size, feather=FEATHER):
    """(size, size, 1) float32 blend weights: 1 inside the inner circle, smoothly down to 0 at
    the circle touching the square's edges, 0 in the corners."""
    c = (size - 1) / 2
    y, x = np.mgrid[:size, :size]
    r = np.hypot(x - c, y - c) / (size / 2)  # 0 at the centre, 1 at the edge midpoints
    t = np.clip((1.0 - r) / feather, 0.0, 1.0)
    return (t * t * (3 - 2 * t)).astype(np.float32)[..., None]  # smoothstep


class FoveatedStream:
    """Bicubic on the whole frame + `sr` (a VSRPipeline built with k=1) on a `fovea` x `fovea`
    LR square around each frame's gaze point."""

    def __init__(self, sr, fovea=96, device="cpu"):
        if fovea < 16:
            raise ValueError(f"fovea must be at least 16 LR pixels, got {fovea}")
        self.sr = sr
        self.fovea = fovea
        self.base = BicubicStream(scale=SCALE, device=device)
        self._mask = None
        self.last_boxes = []

    def push(self, images, gazes=None, base=None) -> list[np.ndarray]:
        """Upscales a batch of uint8 (H, W, 3) frames, the SR model at each frame's gaze point
        (`gazes`: one (u, v) or None per frame). `base`: the frames already upscaled with
        bicubic (writable copies, blended into in place), to avoid computing them twice."""
        if not images:
            self.last_boxes = []
            return []
        gazes = gazes if gazes is not None else [None] * len(images)
        out = base if base is not None else self.base.push(images)
        h, w = images[0].shape[:2]
        size = min(self.fovea, h, w)
        if self._mask is None or self._mask.shape[0] != size * SCALE:
            self._mask = fovea_mask(size * SCALE)
        boxes = [fovea_box(g, w, h, size) for g in gazes]
        crops = [np.ascontiguousarray(img[y0:y1, x0:x1]) for img, (x0, y0, x1, y1) in zip(images, boxes)]
        sr = np.stack(self.sr(crops)).astype(np.float32)
        hr = [(x0 * SCALE, y0 * SCALE, x1 * SCALE, y1 * SCALE) for x0, y0, x1, y1 in boxes]
        base = np.stack([frame[Y0:Y1, X0:X1] for frame, (X0, Y0, X1, Y1) in zip(out, hr)]).astype(np.float32)
        # SR detail on bicubic's colours: base_low + (sr - sr_low)
        fused = sr - lowpass(sr) + lowpass(base)
        blended = base + (fused - base) * self._mask
        for frame, region, (X0, Y0, X1, Y1) in zip(out, blended, hr):
            frame[Y0:Y1, X0:X1] = np.clip(region + 0.5, 0, 255).astype(np.uint8)
        self.last_boxes = hr
        return out

    def flush(self) -> list[np.ndarray]:
        """Nothing is buffered."""
        self.last_boxes = []
        return []
