"""Predicted gaze per decoded frame, for the gaze overlay (--gaze).

The gaze model runs at 30 Hz on the LR frames and a head pose. Head poses come from a
recorded trace (CSV with columns time, azimuth, elevation in seconds and degrees), a
live pose set from outside (LiveHead, e.g. the control page), or a still head looking
at the centre of the screen.

The video frame is treated as the headset's field of view: a predicted offset from the
head direction maps to a point relative to the frame centre, scaled by the field of view.
"""

import csv
import queue
import threading
import time

import numpy as np

from .spec import FUTURE, MODEL_HZ  # the model's rate and number of predicted steps

# Field of view of the headset the gaze model was trained with (HTC Vive, EHTask), degrees.
DEFAULT_FOV = (108.77, 111.53)


class HeadTrace:
    """Head azimuth/elevation (degrees) over time, linearly interpolated."""

    def __init__(self, times, azimuth, elevation):
        self.times = np.asarray(times, dtype=np.float64)
        # Unwrap so interpolation across ±180° takes the short way round.
        self.azimuth = np.degrees(np.unwrap(np.radians(np.asarray(azimuth, dtype=np.float64))))
        self.elevation = np.asarray(elevation, dtype=np.float64)

    @classmethod
    def from_csv(cls, path):
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"No head poses in {path}")
        return cls(
            [float(r["time"]) for r in rows],
            [float(r["azimuth"]) for r in rows],
            [float(r["elevation"]) for r in rows],
        )

    @classmethod
    def still(cls):
        return cls([0.0], [0.0], [0.0])

    def pose_at(self, t):
        """Pose at time t (seconds); held constant outside the recorded range."""
        az = float(np.interp(t, self.times, self.azimuth))
        el = float(np.interp(t, self.times, self.elevation))
        return (az + 180) % 360 - 180, el


class LiveHead:
    """Head pose set from another thread (control page, later a headset); the gaze model
    samples the latest pose at each 30 Hz tick."""

    def __init__(self, azimuth=0.0, elevation=0.0):
        self._lock = threading.Lock()
        self._pose = (float(azimuth), float(elevation))

    def set(self, azimuth, elevation):
        pose = ((float(azimuth) + 180) % 360 - 180, min(max(float(elevation), -90.0), 90.0))
        with self._lock:
            self._pose = pose

    def pose_at(self, t):
        with self._lock:
            return self._pose


def offset_to_uv(offset, fov=DEFAULT_FOV):
    """Gaze offset from the head direction (azimuth, elevation in degrees) -> (u, v) in
    [0, 1] frame coordinates, origin top-left. Points outside the field of view are clamped."""
    # Plain floats: the model's float32 values are not JSON-serialisable (control page status).
    u = 0.5 + float(offset[0]) / fov[0]
    v = 0.5 - float(offset[1]) / fov[1]
    return min(max(u, 0.0), 1.0), min(max(v, 0.0), 1.0)


class GazeTracker:
    """Feeds decoded frames to the gaze model at 30 Hz and returns the predicted gaze
    point for each frame.

    Frames arriving faster than 30 Hz are subsampled; slower ones are repeated, so the
    model always sees 30 Hz input. `step` picks which of the 20 predicted steps (1 =
    the next 30 Hz sample, 20 = 666 ms ahead) is shown.
    """

    def __init__(self, fps, trace=None, step=1, fov=DEFAULT_FOV, start=0.0, device=None, predictor=None):
        if not 1 <= step <= FUTURE:
            raise ValueError(f"step must be between 1 and {FUTURE}, got {step}")
        if predictor is None:
            from gaze.inference import GazePredictor  # loads UniSal + the gaze model

            predictor = GazePredictor(device=device)
        self.predictor = predictor
        self.trace = trace or HeadTrace.still()
        self.fps = float(fps)
        self.step = step
        self.fov = fov
        self.start = start
        self.reset()

    def reset(self):
        """Starts over, e.g. when the video loops: clears the model's 15-frame window."""
        self.frame_idx = 0
        self.next_tick = 0
        self._restart()

    def set_source(self, source):
        """Switches the head pose source (a HeadTrace or LiveHead) mid-stream. The model's
        window restarts so the jump between sources is not read as head motion."""
        self.trace = source
        self._restart()

    def update(self, frame):
        """Takes the next decoded LR frame (uint8 RGB) and returns its gaze point (u, v),
        or None until the model has 15 samples (0.5 s). `path` then holds the points of
        all 20 predicted steps, step 1 first, and `info` the latest prediction for display
        (path, the shown step's offset in degrees, task probabilities)."""
        t = self.frame_idx / self.fps
        # Every 30 Hz tick up to this frame's time gets this frame (the latest one shown).
        while self.next_tick / MODEL_HZ <= t + 1e-9:
            pose = self.trace.pose_at(self.start + self.next_tick / MODEL_HZ)
            self._push(frame, pose)
            self.next_tick += 1
        self.frame_idx += 1
        return self.latest()[0]

    def latest(self):
        """(gaze point, info) of the latest prediction."""
        return self.uv, self.info

    def _restart(self):
        self.predictor.reset()
        self.uv = None
        self.path = None
        self.info = None

    def _push(self, frame, pose):
        self._store(self.predictor.push(frame, pose))

    def _store(self, out):
        if out is None:
            return
        self.path = [offset_to_uv(o, self.fov) for o in out["offsets"]]
        self.uv = self.path[self.step - 1]
        probs = out.get("task_probs")
        self.info = {
            "path": self.path,
            "step": self.step,
            "offset": [float(x) for x in out["offsets"][self.step - 1]],
            "task_probs": None if probs is None else [float(p) for p in probs],
        }


class ThreadedGazeTracker(GazeTracker):
    """GazeTracker with the model (UniSal + Conv-LSTM, ~19 ms per update on MPS) in its own
    thread, so decoding, upscaling and sending do not wait for it.

    update() hands each 30 Hz sample to the worker and returns at once with the latest
    finished prediction, so a frame carries a prediction up to about one update older
    than the frame. If the worker falls more than MAX_QUEUED samples behind, new samples
    are dropped and the model's window restarts once it catches up, so the gap is not read
    as head motion. Call close() when done.

    The predictor must not share a torch MPS context with the stream thread: PyTorch's MPS
    backend is not thread-safe and concurrent use can hang the process. The stream gives
    it a gaze.process.ProcessPredictor, which runs the model in a child process.
    """

    MAX_QUEUED = 30  # 1 s of samples
    _RESET, _STOP = object(), object()

    def __init__(self, *args, **kwargs):
        self._jobs = queue.Queue()
        self._lock = threading.Lock()
        self._error = None
        self._gap = False
        self.dropped = 0
        # Cumulative, for stats(): predictions made, seconds in the model (child process) and
        # waiting for it (round trip), CPU seconds of the worker thread and of the child.
        self._counters = {"predictions": 0, "model_s": 0.0, "roundtrip_s": 0.0, "worker_cpu_s": 0.0,
                          "child_cpu_s": None, "child_mps_mb": None}
        self._result_tick = None
        super().__init__(*args, **kwargs)
        self._worker = threading.Thread(target=self._run, name="gaze-model", daemon=True)
        self._worker.start()

    def latest(self):
        with self._lock:
            return self.uv, self.info

    def stats(self):
        """Counters for the performance view, plus the current queue depth, samples dropped
        and lag: how many 30 Hz samples the latest prediction is behind the newest sample."""
        with self._lock:
            out = dict(self._counters)
            tick = self._result_tick
        out["queued"] = self._jobs.qsize()
        out["dropped"] = self.dropped
        out["lag_samples"] = None if tick is None else max(0, self.next_tick - 1 - tick)
        return out

    def wait(self):
        """Blocks until every queued sample has been processed (tests, end of a clip)."""
        self._jobs.join()

    def close(self):
        self._jobs.put(self._STOP)
        self._worker.join(timeout=5)

    def _restart(self):
        with self._lock:
            self.uv = self.path = self.info = None
        self._gap = False
        self._jobs.put(self._RESET)

    def _push(self, frame, pose):
        if self._error is not None:
            raise RuntimeError("gaze model failed") from self._error
        if self._jobs.qsize() >= self.MAX_QUEUED:
            self.dropped += 1
            self._gap = True
            return
        if self._gap:  # samples were dropped: start a fresh window
            self._gap = False
            self._jobs.put(self._RESET)
        self._jobs.put((frame, pose, self.next_tick))

    def _run(self):
        while True:
            job = self._jobs.get()
            try:
                if job is self._STOP:
                    return
                if job is self._RESET:
                    self.predictor.reset()
                    with self._lock:
                        self.uv = self.path = self.info = None
                        self._result_tick = None
                elif self._error is None:
                    frame, pose, tick = job
                    start = time.perf_counter()
                    out = self.predictor.push(frame, pose)
                    roundtrip = time.perf_counter() - start
                    child = getattr(self.predictor, "last_stats", None) or {}
                    with self._lock:
                        self._store(out)
                        c = self._counters
                        c["predictions"] += 1
                        c["roundtrip_s"] += roundtrip
                        c["model_s"] += child.get("model_s", roundtrip)
                        c["worker_cpu_s"] = time.thread_time()
                        c["child_cpu_s"] = child.get("cpu_s")
                        c["child_mps_mb"] = child.get("mps_mb")
                        if out is not None:
                            self._result_tick = tick
            except Exception as exc:  # noqa: BLE001 -- raised in the stream thread by the next update()
                self._error = exc
            finally:
                self._jobs.task_done()
