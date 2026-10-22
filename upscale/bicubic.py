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

    def push(self, images) -> list[np.ndarray]:
        """Upscales a batch of uint8 (H, W, 3) frames and returns them all."""
        if not images:
            return []
        # to_numpy clamps to [0, 1], which also clips bicubic overshoot at edges.
        return frame_io.to_numpy(self.upscale(images))

    def flush(self) -> list[np.ndarray]:
        """Nothing is ever buffered, so there is nothing to drain."""
        return []
