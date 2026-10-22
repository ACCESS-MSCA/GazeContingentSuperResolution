"""UniSal on the Apple Neural Engine, through Core ML (macOS, needs coremltools).

The Neural Engine is separate from the GPU, so saliency no longer competes with the SR model
for it: 3.8 ms per frame on an M4, against 11 ms compiled on MPS. It computes in float16, so
the output is close to PyTorch's but not identical: over real frames with head motion the
predicted gaze moved by 0.09 deg on average and 0.24 deg at most (one LR pixel of a 240 px
wide frame is 0.45 deg; the model's own error is 8.5 deg at 333 ms).

The model is converted from unisal.pth on first use (a few seconds) and cached in
~/.cache/gaze/, named after the weights file's hash and the input size.
"""

import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "gaze"


def available():
    """Whether Core ML can be used here (macOS with coremltools installed)."""
    if sys.platform != "darwin":
        return False
    try:
        import coremltools  # noqa: F401
    except ImportError:
        return False
    return True


class _Static(nn.Module):
    """UniSal on one frame: (1, 3, H, W) -> (1, 1, H, W) saliency, the graph Core ML converts."""

    def __init__(self, unisal):
        super().__init__()
        self.unisal = unisal

    def forward(self, x):
        return self.unisal(x.unsqueeze(1)).squeeze(1)


def _cached_path(sal_path, size):
    digest = hashlib.sha256(Path(sal_path).read_bytes()).hexdigest()[:16]
    return CACHE_DIR / f"unisal_{digest}_{size[0]}x{size[1]}_fp16.mlpackage"


def _convert(sal_path, size, path):
    import coremltools as ct

    from .inference import load_unisal_model

    width, height = size
    model = _Static(load_unisal_model(sal_path, torch.device("cpu"))).eval()
    example = torch.rand(1, 3, height, width)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    ml = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=example.shape)], outputs=[ct.TensorType(name="sal")],
                    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15,
                    convert_to="mlprogram")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:  # written whole, then moved in place
        ml.save(os.path.join(tmp, path.name))
        shutil.move(os.path.join(tmp, path.name), path)


class CoreMLUniSal:
    """Called like the UniSal module on static input: (N, 1, 3, H, W) float tensor (H, W = `size`
    reversed) -> (N, 1, 1, H, W) saliency on the CPU, run on the Neural Engine."""

    def __init__(self, sal_path, size):
        import coremltools as ct

        path = _cached_path(sal_path, size)
        if not path.exists():
            _convert(sal_path, size, path)
        self._model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)

    def __call__(self, x):
        out = [self._model.predict({"x": frame[None].cpu().numpy()})["sal"] for frame in x[:, 0]]
        return torch.from_numpy(np.concatenate(out))[:, None].float()
