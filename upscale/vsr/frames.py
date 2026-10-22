"""Conversion between the public numpy frame format and internal tensors.

Public format: uint8 RGB arrays of shape (H, W, 3). Float arrays in [0, 1] and
torch tensors are also accepted on input as a convenience; output is always
uint8 (H, W, 3) so callers can hand frames straight to imageio/OpenCV/PIL.
"""

import numpy as np
import torch

_CHANNEL_COUNTS = (1, 3, 4)


def _array_to_chw(array: np.ndarray) -> np.ndarray:
    """Normalizes an HWC or CHW array to CHW, rejecting ambiguous shapes."""
    if array.ndim != 3:
        raise ValueError(f"expected a 3-dimensional frame, got shape {array.shape}")

    hwc = array.shape[-1] in _CHANNEL_COUNTS
    chw = array.shape[0] in _CHANNEL_COUNTS

    if hwc and chw:
        # e.g. (3, 3, 3) -- genuinely ambiguous, so pick the documented layout.
        return np.transpose(array, (2, 0, 1))
    if hwc:
        return np.transpose(array, (2, 0, 1))
    if chw:
        return array
    raise ValueError(
        f"cannot infer channel axis for frame of shape {array.shape}; "
        f"expected (H, W, C) or (C, H, W) with C in {_CHANNEL_COUNTS}"
    )


def _to_float_chw_tensor(frame) -> torch.Tensor:
    """Converts one frame to a (C, H, W) float tensor in [0, 1]."""
    if isinstance(frame, torch.Tensor):
        tensor = frame.detach()
        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise ValueError(
                    f"batched tensor frames must have batch size 1, got {tensor.shape}"
                )
            tensor = tensor.squeeze(0)
        if tensor.dtype == torch.uint8:
            tensor = tensor.float().div_(255.0)
        else:
            tensor = tensor.float()
        return tensor

    array = np.asarray(frame)
    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0
    else:
        array = array.astype(np.float32)
        if array.size and float(array.max()) > 1.5:
            raise ValueError(
                "float frames must be in [0, 1]; got values above 1.0. "
                "Pass a uint8 array instead if these are 0-255 pixels."
            )

    return torch.from_numpy(np.ascontiguousarray(_array_to_chw(array)))


def to_batch(frames, device, dtype) -> torch.Tensor:
    """Stacks a list of frames into one (N, C, H, W) tensor on `device`.

    All frames must share a resolution -- a video pipeline has no meaningful
    behaviour for a ragged clip, and silently resizing would corrupt output.
    """
    if not frames:
        raise ValueError("expected at least one frame")

    if all(isinstance(f, np.ndarray) and f.dtype == np.uint8 and f.ndim == 3 and f.shape[-1] == 3 for f in frames):
        # The usual case, decoded RGB frames: uploaded as uint8 and converted on the device,
        # 4x less to copy and no float conversion on the CPU (2.0 -> 0.5 ms at 960x540).
        if len({f.shape for f in frames}) != 1:
            raise ValueError(f"all frames must share a shape, got {sorted({f.shape for f in frames})}")
        # contiguous: a channels-last layout made bicubic 5x slower on MPS
        batch = torch.from_numpy(np.stack(frames)).to(device).permute(0, 3, 1, 2).contiguous()
        return batch.to(torch.float32).div_(255.0).to(dtype)

    tensors = [_to_float_chw_tensor(f) for f in frames]

    shapes = {tuple(t.shape) for t in tensors}
    if len(shapes) != 1:
        raise ValueError(f"all frames must share a shape, got {sorted(shapes)}")

    channels = tensors[0].shape[0]
    if channels != 3:
        raise ValueError(
            f"expected 3-channel RGB frames, got {channels} channels. "
            "Convert grayscale to RGB and drop any alpha channel before calling."
        )

    return torch.stack(tensors).to(device=device, dtype=dtype)


def to_numpy(tensor: torch.Tensor) -> list[np.ndarray]:
    """Converts a (N, C, H, W) or (1, C, H, W) tensor to uint8 (H, W, C) arrays."""
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    array = (
        tensor.detach()
        .to(torch.float32)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )
    return [np.ascontiguousarray(frame) for frame in array]


_COMPILED = {}


def compiled(fn, device: torch.device):
    """`fn` compiled with torch.compile on a GPU (MPS, CUDA), as is on the CPU. For chains of
    elementwise ops over whole output frames, which run as one kernel when compiled instead
    of a pass over the frame each (4K BGRA conversion on an M4: 14 -> 3 ms). The first call
    per shape compiles (about a second); TORCHDYNAMO_DISABLE=1 turns it off."""
    if torch.device(device).type == "cpu":
        return fn
    if fn not in _COMPILED:
        _COMPILED[fn] = torch.compile(fn)
    return _COMPILED[fn]


def _bgra(tensor: torch.Tensor) -> torch.Tensor:
    rgb = tensor.to(torch.float32).clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
    alpha = torch.full_like(rgb[:, :1], 255)
    return torch.cat([rgb.flip(1), alpha], dim=1).permute(0, 2, 3, 1).contiguous()


def to_bgra_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Converts an (N, 3, H, W) RGB tensor in [0, 1] to a uint8 (N, H, W, 4) BGRA tensor, the
    layout NDI sends, on the tensor's device."""
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return compiled(_bgra, tensor.device)(tensor.detach())


def to_bgra(tensor: torch.Tensor) -> list[np.ndarray]:
    """Converts an (N, 3, H, W) RGB tensor in [0, 1] to uint8 (H, W, 4) BGRA arrays, the
    format NDI sends. The conversion runs on the tensor's device and the result is copied
    off it once: at 4K output, converting on the CPU took longer than upscaling."""
    return list(to_bgra_tensor(tensor).cpu().numpy())
