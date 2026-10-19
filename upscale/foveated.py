"""Gaze-guided ("foveated") 4x upscaling: the SR model where the viewer looks, bicubic elsewhere.

Every frame is upscaled with bicubic interpolation; the SR model then runs on a square LR
crop centred on the predicted gaze point, and its 4x result is blended back in with a
round, soft-edged mask (full SR in the centre, fading to bicubic at the edge), so no seam
shows. Only the SR's fine detail is used: on a small crop the SR model shifts the
colours (5-10 levels brighter in testing; on a whole frame it keeps them), so the crop's
low frequencies are taken from bicubic instead, and colour and brightness match exactly.
k = 1 (default): the SR model runs on every frame's crop, at that frame's gaze point.

k > 1: the SR model runs only on every k-th frame (keyframes), and RIFE makes the crops in
between from the two keyframe crops around them. RIFE can only blend crops cut from the
same place, so the crop stays put for each stretch of k frames: frames are held until the
stretch's closing keyframe has been decoded, and the crop is centred on the mean of the
stretch's gaze points, so it covers where the gaze is during the stretch rather than
lagging behind it. The previous stretch's closing keyframe opens the next one; if the crop
moves by more than a quarter of the fovea, that keyframe is upscaled again at the new
position (2 SR calls for that stretch), otherwise its SR crop is reused (1 call). Output
lags input by up to k-1 frames, as with the vsr source.

Same push()/flush() interface as the other upscalers, plus the gaze points: push(frames,
gazes) returns the frames that are ready (all of them at k = 1), and `last_boxes` holds each
returned frame's fovea region in output (HR) pixels, for drawing on the -Foveated source.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F

from upscale import SCALE
from upscale.bicubic import BicubicStream
from upscale.vsr import frames as frame_io

# Share of the fovea's radius over which SR fades into bicubic.
FEATHER = 0.35
# Low-pass for the colour match: downsample by this factor (area) and back up (bilinear).
# 8 at 4x output is half the LR resolution, so the detail kept is finer than the LR frame holds.
LOWPASS = 8
# k > 1: a stretch keeps the previous stretch's crop (and reuses its closing keyframe's SR) when
# the new crop would move by at most this share of the fovea.
STICKY = 0.25


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


def mean_gaze(gazes):
    """Mean of the (u, v) gaze points that are not None, or None if there are none."""
    points = [g for g in gazes if g is not None]
    if not points:
        return None
    return tuple(float(x) for x in np.mean(points, axis=0))


class FoveatedStream:
    """Bicubic on the whole frame + `sr` (a VSRPipeline) on a `fovea` x `fovea` LR square
    around the gaze point: on every frame (k = 1), or on every k-th frame with RIFE in
    between (k > 1; `sr` must then have its interpolation model, i.e. be built with k > 1)."""

    def __init__(self, sr, fovea=96, device="cpu", k=1):
        if fovea < 16:
            raise ValueError(f"fovea must be at least 16 LR pixels, got {fovea}")
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        if k > 1 and getattr(sr, "interpolation", None) is None:
            raise ValueError(f"k={k} needs the interpolation model: build the VSRPipeline with k > 1")
        self.sr = sr
        self.fovea = fovea
        self.k = k
        self.base = BicubicStream(scale=SCALE, device=device)
        self._mask = None
        self.last_boxes = []
        # Time in each stage (SR on the keyframe crops, RIFE for the crops in between, blending
        # into bicubic), crops each made, and frames done; cumulative, for performance figures.
        self.sr_s = self.rife_s = self.blend_s = 0.0
        self.sr_frames = self.rife_frames = self.frames = 0
        self.reset()

    def reset(self):
        """Forgets frames held for the next keyframe (k > 1). Call between clips."""
        self._index = 0
        self._pending = []  # (index, LR frame, gaze, bicubic frame) after the last keyframe
        self._previous = None  # the last keyframe: (LR frame, LR box, SR crop tensor)

    def push(self, images, gazes=None, base=None) -> list[np.ndarray]:
        """Upscales a batch of uint8 (H, W, 3) frames (`gazes`: one (u, v) or None per frame).
        `base`: the frames already upscaled with bicubic (writable copies, blended into in
        place), to avoid computing them twice. Returns the frames that are ready, in order."""
        self.last_boxes = []
        if not images:
            return []
        gazes = gazes if gazes is not None else [None] * len(images)
        bases = base if base is not None else self.base.push(images)
        if self.k == 1:
            return self._every_frame(images, gazes, bases)
        for image, gaze, frame in zip(images, gazes, bases):
            self._pending.append((self._index, image, gaze, frame))
            self._index += 1
        return self._drain(final=False)

    def flush(self) -> list[np.ndarray]:
        """Returns the frames still held (k > 1), the last one taken as a keyframe, and resets."""
        self.last_boxes = []
        out = self._drain(final=True) if self.k > 1 else []
        self.reset()
        return out

    # --- k = 1: SR on every frame's crop ---------------------------------------------------
    def _every_frame(self, images, gazes, bases):
        h, w = images[0].shape[:2]
        size = min(self.fovea, h, w)
        boxes = [fovea_box(g, w, h, size) for g in gazes]
        crops = [np.ascontiguousarray(img[y0:y1, x0:x1]) for img, (x0, y0, x1, y1) in zip(images, boxes)]
        start = time.perf_counter()
        sr = np.stack(self.sr(crops)).astype(np.float32)  # numpy out, so the GPU work is done
        self.sr_s += time.perf_counter() - start
        self.sr_frames += len(crops)
        return self._blend(bases, sr, boxes)

    # --- k > 1: SR on keyframe crops, RIFE in between ----------------------------------------
    def _drain(self, final):
        """Emits every held frame whose closing keyframe has arrived (or all of them if final)."""
        out, boxes = [], []
        while self._pending:
            ends = [i for i, (index, *_) in enumerate(self._pending) if index % self.k == 0]
            if not ends:
                if not final:
                    break
                ends = [len(self._pending) - 1]  # end of the clip: the last frame closes the stretch
            stretch, self._pending = self._pending[:ends[0] + 1], self._pending[ends[0] + 1:]
            frames, frame_boxes = self._stretch(stretch)
            out += frames
            boxes += frame_boxes
        self.last_boxes = boxes
        return out

    def _stretch(self, stretch):
        """Upscales the crops of one stretch: the frames after the previous keyframe up to and
        including the closing keyframe, all at one crop position."""
        image = stretch[-1][1]
        h, w = image.shape[:2]
        size = min(self.fovea, h, w)
        box = fovea_box(mean_gaze([g for _, _, g, _ in stretch]), w, h, size)
        start = time.perf_counter()
        if self._previous is None:  # the clip's first keyframe opens nothing: emit it as is
            end_sr = self._sr_crops([image], box)[0]
            self.sr_s += time.perf_counter() - start
            self._previous = (image, box, end_sr)
            return self._blend([stretch[-1][3]], self._numpy([end_sr]), [box]), [_hr(box)]
        prev_image, prev_box, prev_sr = self._previous
        if max(abs(a - b) for a, b in zip(prev_box, box)) <= STICKY * size:
            box = prev_box  # close enough: keep the crop, and the previous keyframe's SR
            end_sr = self._sr_crops([image], box)[0]
        else:  # the crop moves: the opening keyframe is upscaled again at the new place
            prev_sr, end_sr = self._sr_crops([prev_image, image], box)
        self.sr_s += time.perf_counter() - start
        # RIFE: the crops between the two keyframes, at their place in time
        start = time.perf_counter()
        n = len(stretch)
        mids = [self.sr.interpolation(prev_sr, end_sr, timestep=i / n) for i in range(1, n)]
        crops = self._numpy(mids + [end_sr])
        self.rife_s += time.perf_counter() - start
        self.rife_frames += len(mids)
        self._previous = (image, box, end_sr)
        return self._blend([frame for *_, frame in stretch], crops, [box] * n), [_hr(box)] * n

    def _sr_crops(self, images, box):
        """SR of each image's `box` crop, as (1, C, 4h, 4w) tensors (for RIFE)."""
        x0, y0, x1, y1 = box
        crops = [np.ascontiguousarray(img[y0:y1, x0:x1]) for img in images]
        batch = frame_io.to_batch(crops, device=self.sr.device, dtype=self.sr.dtype)
        self.sr_frames += len(crops)
        out = self.sr.run_sr(batch)
        self._sync()  # so the SR time is not counted as RIFE's
        return out

    def _sync(self):
        """Waits for queued GPU work."""
        device = torch.device(self.sr.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    def _numpy(self, tensors):
        """(1, C, H, W) tensors -> (N, H, W, 3) float32 (waits for the GPU)."""
        return np.stack([frame_io.to_numpy(t)[0] for t in tensors]).astype(np.float32)

    # --- both ------------------------------------------------------------------------------------
    def _blend(self, bases, sr, boxes):
        """Blends SR crops (N, 4h, 4w, 3 float32) into the bicubic frames at the LR `boxes`:
        SR detail on bicubic's colours, through the round soft-edged mask."""
        start = time.perf_counter()
        size = boxes[0][2] - boxes[0][0]
        if self._mask is None or self._mask.shape[0] != size * SCALE:
            self._mask = fovea_mask(size * SCALE)
        hr = [_hr(b) for b in boxes]
        base = np.stack([frame[Y0:Y1, X0:X1] for frame, (X0, Y0, X1, Y1) in zip(bases, hr)]).astype(np.float32)
        # SR detail on bicubic's colours: base_low + (sr - sr_low)
        fused = sr - lowpass(sr) + lowpass(base)
        blended = base + (fused - base) * self._mask
        for frame, region, (X0, Y0, X1, Y1) in zip(bases, blended, hr):
            frame[Y0:Y1, X0:X1] = np.clip(region + 0.5, 0, 255).astype(np.uint8)
        self.blend_s += time.perf_counter() - start
        self.frames += len(bases)
        self.last_boxes = hr
        return list(bases)


def _hr(box):
    """LR box -> the same region in output (HR) pixels."""
    return tuple(v * SCALE for v in box)
