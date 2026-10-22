"""The SR model on the Apple Neural Engine, through Core ML (macOS, needs coremltools).

Used for the foveated source's crops: a fixed input size, which the Neural Engine needs. On an
M4 a 64x64 crop takes 41 ms there, against 92 ms on MPS, and leaves the GPU to RIFE and the
full-frame work. It computes in float16, so the output differs from PyTorch's in textured areas
(equally sharp, slightly different fine texture; 35-57 dB PSNR on the detail the blend keeps).

Each input size is converted once (about 20 s; torch.export, then coremltools), compiled, and cached
in ~/.cache/vsr/coreml/ as a compiled model only, named after the checkpoint files and the size. The model then runs in a
child process, which takes about 8 s to load it onto the Neural Engine: coremltools' predict
holds Python's GIL for the whole call (~40 ms), which in this process would stop every other
thread, the stream thread included, while it runs. Waiting on the child's pipe does not.
"""

import copy
import hashlib
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import threading
import traceback
from pathlib import Path

import numpy as np
import torch

CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "vsr" / "coreml"


def available():
    """Whether Core ML can be used here (macOS with coremltools installed)."""
    if sys.platform != "darwin":
        return False
    try:
        import coremltools  # noqa: F401
    except ImportError:
        return False
    return True


def _name(ckpt_paths, height, width):
    key = hashlib.sha256()
    for path in ckpt_paths:
        stat = os.stat(path)
        key.update(f"{os.path.abspath(path)}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return f"sr_{key.hexdigest()[:16]}_{height}x{width}_fp16"


def _convert(model, height, width, package):
    import coremltools as ct

    cpu_model = copy.deepcopy(model).to("cpu", torch.float32).eval()
    example = torch.rand(1, 3, height, width)
    with torch.no_grad():
        # torch.export: tracing (torch.jit.trace) left an int cast coremltools could not convert
        exported = torch.export.export(cpu_model, (example,)).run_decompositions({})
    ml = ct.convert(exported, compute_precision=ct.precision.FLOAT16,
                    minimum_deployment_target=ct.target.macOS15, convert_to="mlprogram")
    package.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=package.parent) as tmp:  # written whole, then moved in place
        ml.save(os.path.join(tmp, package.name))
        shutil.move(os.path.join(tmp, package.name), package)


def _serve(conn, compiled, shape):
    """Child process: loads the compiled model onto the Neural Engine, then upscales each batch
    of (B, 3, H, W) float32 frames it is sent, in order, until it gets None or the pipe closes."""
    try:
        import coremltools as ct

        model = ct.models.CompiledMLModel(compiled, compute_units=ct.ComputeUnit.CPU_AND_NE)
        model.predict({"x": np.zeros(shape, np.float32)})  # the first call sets up the Neural Engine
    except Exception:  # noqa: BLE001 -- reported to the parent, which raises it
        conn.send(("error", traceback.format_exc()))
        return
    conn.send(("ready", None))
    while True:
        try:
            frames = conn.recv()
        except EOFError:  # parent went away
            return
        if frames is None:
            return
        try:
            out = [next(iter(model.predict({"x": frame[None]}).values())) for frame in frames]
            conn.send(("ok", np.concatenate(out)))
        except Exception:  # noqa: BLE001 -- reported to the parent, which raises it
            conn.send(("error", traceback.format_exc()))


class CoreMLSR:
    """`model` (the SR network) for (B, 3, height, width) inputs, run on the Neural Engine in a
    child process. Returns (B, 3, 4 height, 4 width) float32 tensors on the CPU. One call at a
    time; any thread may call it."""

    def __init__(self, model, ckpt_paths, height, width, log=print):
        import coremltools as ct

        base = CACHE_DIR / _name(ckpt_paths, height, width)
        package, compiled = base.with_suffix(".mlpackage"), base.with_suffix(".mlmodelc")
        if not compiled.exists():
            if not package.exists():
                log(f"[vsr] converting the SR model for the Neural Engine ({width}x{height}), once: ~20 s")
                _convert(model, height, width, package)
            shutil.move(ct.models.utils.compile_model(str(package)), compiled)
            shutil.rmtree(package)  # only the compiled model is loaded (each is ~0.9 GB)
        log("[vsr] loading the SR model onto the Neural Engine...")
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_serve, args=(child, str(compiled), (1, 3, height, width)),
                                 name="sr-neural-engine", daemon=True)
        self._proc.start()
        child.close()
        self._lock = threading.Lock()
        self._reply()

    def __call__(self, batch):
        frames = np.ascontiguousarray(batch.detach().to("cpu", torch.float32).numpy())
        with self._lock:
            self._conn.send(frames)
            return torch.from_numpy(self._reply())

    def close(self):
        if self._proc.is_alive():
            try:
                self._conn.send(None)
            except OSError:
                pass
            self._proc.join(timeout=5)
        if self._proc.is_alive():
            self._proc.terminate()
        self._conn.close()

    def _reply(self):
        try:
            kind, value = self._conn.recv()
        except EOFError:
            raise RuntimeError(f"Neural Engine SR process exited (code {self._proc.exitcode})") from None
        if kind == "error":
            raise RuntimeError(f"Neural Engine SR process failed:\n{value}")
        return value
