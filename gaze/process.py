"""The gaze model in its own process (ProcessPredictor).

PyTorch's MPS backend is not thread-safe: running the gaze model on MPS in a thread
while the stream thread upscales on MPS corrupts MPS's kernel cache and can hang the
process. A child process has its own MPS context, so the two never share it. Frames go
over a pipe (about 100 KB per sample at 240x136), which is cheap at 30 Hz.

ProcessPredictor has the same push()/reset() interface as gaze.GazePredictor, so
ThreadedGazeTracker uses it unchanged: its worker thread waits on the pipe while the
child computes, and the stream thread never waits at all.
"""

import multiprocessing as mp
import time
import traceback

import numpy as np

# One full model window (15 frames) plus one prediction: compiles every kernel the model uses.
WARMUP_SAMPLES = 16


def _mps_mb():
    """MPS memory allocated by this (child) process, MB (None without MPS)."""
    import torch

    return torch.mps.current_allocated_memory() / 1e6 if torch.backends.mps.is_available() else None


def _serve(conn, factory, kwargs):
    """Child process: builds the predictor, then answers push/reset requests in order. Each
    answer carries the child's timing for the performance view: model time for this
    request, total CPU time of the child process and its MPS memory."""
    try:
        predictor = factory(**kwargs)
        # Warm-up: the first calls on MPS compile kernels and take far longer than the
        # ~20 ms steady state, so the stream's first second of samples would queue up and
        # be dropped. A full window of blank frames here, while the parent waits for
        # "ready", moves that cost to loading.
        blank = np.zeros((136, 240, 3), np.uint8)
        for _ in range(WARMUP_SAMPLES):
            predictor.push(blank, (0.0, 0.0))
        predictor.reset()
    except Exception:  # noqa: BLE001 -- reported to the parent, which raises it
        conn.send(("error", traceback.format_exc()))
        return
    conn.send(("ready", None))
    while True:
        try:
            msg = conn.recv()
        except EOFError:  # parent went away
            return
        if msg[0] == "stop":
            return
        if msg[0] == "reset":
            predictor.reset()
            continue
        try:
            start = time.perf_counter()
            out = predictor.push(msg[1], msg[2])
            stats = {"model_s": time.perf_counter() - start, "cpu_s": time.process_time(), "mps_mb": _mps_mb()}
            if out is not None:
                probs = out.get("task_probs")
                out = {"offsets": np.asarray(out["offsets"], np.float32),
                       "task_probs": None if probs is None else np.asarray(probs, np.float32)}
            conn.send(("ok", out, stats))
        except Exception:  # noqa: BLE001 -- reported to the parent, which raises it
            conn.send(("error", traceback.format_exc()))


class ProcessPredictor:
    """Runs a gaze predictor (by default gaze.GazePredictor(**kwargs)) in a child process.

    `factory` must be importable by the child (a module-level class or function). The
    constructor waits until the model is loaded. Only one thread may call push/reset.
    After each push, `last_stats` holds the child's timing (see _serve).
    """

    def __init__(self, factory=None, **kwargs):
        if factory is None:
            from gaze.inference import GazePredictor as factory
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_serve, args=(child, factory, kwargs), name="gaze-model", daemon=True)
        self._proc.start()
        child.close()
        self.last_stats = None
        self._check(self._recv())

    def push(self, frame, pose):
        self._conn.send(("push", np.asarray(frame), (float(pose[0]), float(pose[1]))))
        reply = self._recv()
        out = self._check(reply)
        self.last_stats = reply[2]
        return out

    def reset(self):
        self._conn.send(("reset",))

    def close(self):
        if self._proc.is_alive():
            try:
                self._conn.send(("stop",))
            except OSError:
                pass
            self._proc.join(timeout=5)
        if self._proc.is_alive():
            self._proc.terminate()
        self._conn.close()

    def _recv(self):
        try:
            return self._conn.recv()
        except EOFError:
            raise RuntimeError(f"gaze model process exited (code {self._proc.exitcode})") from None

    @staticmethod
    def _check(reply):
        kind, value = reply[0], reply[1]
        if kind == "error":
            raise RuntimeError(f"gaze model process failed:\n{value}")
        return value
