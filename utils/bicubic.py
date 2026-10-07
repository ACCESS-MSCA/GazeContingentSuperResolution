"""Bicubic 4x upscaling with the same push()/flush() interface as VSRStream.

A baseline to compare the SR model against: no model, no weights, and every
frame is returned as soon as it is pushed (no keyframe buffering).
"""

import numpy as np
import torch
import torch.nn.functional as F

from vsr import frames as frame_io


class BicubicStream:
    def __init__(self, scale: int = 4, device: str = "cpu"):
        self.scale = scale
        self.device = torch.device(device)

    @torch.no_grad()
    def push(self, images) -> list[np.ndarray]:
        """Upscales a batch of uint8 (H, W, 3) frames and returns them all."""
        if not images:
            return []
        batch = frame_io.to_batch(images, device=self.device, dtype=torch.float32)
        upscaled = F.interpolate(
            batch, scale_factor=self.scale, mode="bicubic", align_corners=False
        )
        # to_numpy clamps to [0, 1], which also clips bicubic overshoot at edges.
        return frame_io.to_numpy(upscaled)

    def flush(self) -> list[np.ndarray]:
        """Nothing is ever buffered, so there is nothing to drain."""
        return []
