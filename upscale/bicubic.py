"""Bicubic 4x upscaling with the same push()/flush() interface as VSRStream.

A baseline to compare the SR model against: no model, no weights, and every
frame is returned as soon as it is pushed (no keyframe buffering).
"""

import numpy as np
import torch
import torch.nn.functional as F

from upscale import SCALE
from upscale.vsr import frames as frame_io


def _bicubic(batch, scale):
    return F.interpolate(batch, scale_factor=scale, mode="bicubic", align_corners=False)


class BicubicStream:
    def __init__(self, scale: int = SCALE, device: str = "cpu"):
        self.scale = scale
        self.device = torch.device(device)

    @torch.no_grad()
    def upscale(self, images) -> torch.Tensor:
        """Upscales a batch of uint8 (H, W, 3) frames to an (N, 3, H, W) float32 tensor on the
        device, not clamped: for callers that keep working on the frames there."""
        batch = frame_io.to_batch(images, device=self.device, dtype=torch.float32)
        return frame_io.compiled(_bicubic, self.device)(batch, self.scale)

    @torch.no_grad()
    def upscale_bgra(self, images):
        """Upscales a batch of uint8 (H, W, 3) frames straight to the BGRA NDI sends. Returns the LR
        batch ((N, 3, h, w) float16 on a GPU, float32 on the CPU; for upscale_region) and the (N, H, W, 4) uint8 BGRA frames,
        both on the device. The float frames at the output size (100 MB each at 4K) are not kept."""
        # float16 on a GPU: half the memory traffic (bicubic + BGRA 4.2 -> 3.6 ms per 4K frame on an
        # M4); 3% of the values round one level differently from float32, none by more
        dtype = torch.float16 if self.device.type in ("mps", "cuda") else torch.float32
        lr = frame_io.to_batch(images, device=self.device, dtype=dtype)
        return lr, frame_io.to_bgra_tensor(frame_io.compiled(_bicubic, self.device)(lr, self.scale))

    @torch.no_grad()
    def upscale_region(self, lr, box):
        """Bicubic of the LR `box` (x0, y0, x1, y1) of an (N, 3, h, w) LR batch: an (N, 3, 4 (y1 - y0),
        4 (x1 - x0)) float tensor with the values that region has in the whole upscaled frame. It
        upscales the box plus a 2 px margin, all that the bicubic kernel reaches."""
        x0, y0, x1, y1 = box
        h, w = lr.shape[-2:]
        ax0, ay0, ax1, ay1 = max(x0 - 2, 0), max(y0 - 2, 0), min(x1 + 2, w), min(y1 + 2, h)
        up, s = _bicubic(lr[..., ay0:ay1, ax0:ax1], self.scale), self.scale
        return up[..., (y0 - ay0) * s:(y1 - ay0) * s, (x0 - ax0) * s:(x1 - ax0) * s]

    def push(self, images) -> list[np.ndarray]:
        """Upscales a batch of uint8 (H, W, 3) frames and returns them all."""
        if not images:
            return []
        # to_numpy clamps to [0, 1], which also clips bicubic overshoot at edges.
        return frame_io.to_numpy(self.upscale(images))

    def flush(self) -> list[np.ndarray]:
        """Nothing is ever buffered, so there is nothing to drain."""
        return []
