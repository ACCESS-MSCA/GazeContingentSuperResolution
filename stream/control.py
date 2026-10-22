"""Shared state between the stream and its controls (command line or control page).

StreamOptions are fixed for one run (changing them restarts the stream). LiveSettings
change while streaming: the stream reads them every frame. StreamStatus is written by
the stream and read by the control page.
"""

import math
import threading
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from gaze.spec import FUTURE, MODEL_HZ
from gaze.tracker import HeadTrace, LiveHead

ROOT = Path(__file__).resolve().parent.parent
VIDEOS = ROOT / "Videos"

# Pre-downscaled 8x (1920x1080 -> 240x136), so no scaling is done live and the
# 4x output (960x544) fits in Apple GPU (MPS) memory.
DEFAULT_VIDEO = VIDEOS / "big_buck_bunny_240x136.mp4"

# The NDI sources a run can send, in this order, all from the same decoded frames and in
# lockstep (each enabled source gets every frame, so a run goes at its slowest source's pace).
SOURCES = {
    "source": "Source",      # the decoded video as is (LR)
    "vsr": "VSR",            # the SR model on the whole frame (+ RIFE when k > 1)
    "bicubic": "Bicubic",    # bicubic 4x
    "foveated": "Foveated",  # bicubic 4x + the gaze prediction, and the SR model around it (fovea_sr)
}
GAZE_SOURCES = ("foveated",)  # need the gaze model
CROPPED_SOURCES = ("source", "bicubic", "vsr")  # the video pipeline, which vsr_crop applies to
DEFAULT_SOURCES = ["source"]  # the video as is: real time, no models loaded
DEVICES = ("cuda", "mps", "cpu")
HEAD_MODES = ("trace", "live")


@dataclass
class StreamOptions:
    """Options fixed for one run of the stream."""

    video: str = str(DEFAULT_VIDEO)
    # NDI sources to send (keys of SOURCES), each named "<name>-<label>".
    sources: list = field(default_factory=lambda: list(DEFAULT_SOURCES))
    # (vsr only) k=1: diffusion SR on every frame. k>1: SR on every k-th frame (keyframes),
    # RIFE interpolates the frames in between (faster, adds up to k-1 frames delay).
    k: int = 3
    # (the video pipeline: source, bicubic, vsr) side of a centred LR square that those sources
    # send (upscaled 4x by bicubic and vsr), so they stay comparable pixel for pixel and VSR's speed
    # can be measured at other sizes without re-encoding the video; None = whole frame.
    vsr_crop: int | None = None
    # (foveated only) whether the SR model runs around the gaze (off: bicubic + the gaze
    # prediction, real time), the side of the LR square it upscales, whether that region is
    # outlined, and whether the gaze prediction is drawn.
    fovea_sr: bool = True
    fovea: int = 64
    # (foveated only) 1: SR on every frame's crop; k > 1: SR on every k-th frame's crop, RIFE in
    # between, with the crop held still for each stretch of k frames (see upscale/foveated.py).
    fovea_k: int = 5
    fovea_outline: bool = True
    fovea_gaze: bool = True
    # (foveated only) run the SR model on the Apple Neural Engine (macOS with coremltools, on
    # MPS), off the GPU; otherwise, or off, on the device (upscale/vsr/super_resolution/coreml.py).
    fovea_ane: bool = True
    # (foveated, Neural Engine, fovea_k > 1) upscale the next stretch on the Neural Engine while
    # the GPU finishes the current one: faster, but frames are held fovea_k frames longer.
    fovea_overlap: bool = True
    batch: int = 2
    start: float = 0.0
    frames: int | None = None
    device: str | None = None
    name: str = "StreamNDI"

    @property
    def gaze(self):
        """Whether the gaze model runs (a source that needs it is enabled)."""
        return any(s in self.sources for s in GAZE_SOURCES)

    def source_names(self):
        """{source key: NDI name} for the enabled sources, in SOURCES order. Cropped sources
        are named "<name>-<label>-Cropped", so they are not mistaken for the whole frame."""
        return {k: f"{self.name}-{label}" + ("-Cropped" if k in CROPPED_SOURCES and self.vsr_crop else "")
                for k, label in SOURCES.items() if k in self.sources}

    @classmethod
    def from_dict(cls, data):
        """Builds options from the control page's JSON, with type and range checks.
        Videos must be files in Videos/, since the page may be reachable from the LAN."""
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown options: {', '.join(sorted(unknown))}")
        opts = cls(**data)
        video = VIDEOS / Path(opts.video).name
        if Path(opts.video).name not in list_files(VIDEOS, ".mp4", ".mov", ".mkv"):
            raise ValueError(f"Video not found in Videos/: {opts.video}")
        opts.video = str(video)
        opts.sources = normalize_sources(opts.sources)
        if opts.device not in (None, "", *DEVICES):
            raise ValueError(f"device must be one of {DEVICES} or empty for auto")
        opts.device = opts.device or None
        opts.k, opts.batch, opts.fovea, opts.fovea_k = int(opts.k), int(opts.batch), int(opts.fovea), int(opts.fovea_k)
        opts.start = float(opts.start)
        opts.frames = int(opts.frames) if opts.frames not in (None, "") else None
        opts.vsr_crop = int(opts.vsr_crop) if opts.vsr_crop not in (None, "", 0) else None
        if opts.vsr_crop is not None and opts.vsr_crop < 16:
            raise ValueError("crop must be at least 16 LR pixels (or empty for the whole frame)")
        if opts.k < 1 or opts.fovea_k < 1 or opts.batch < 1 or opts.start < 0 or (opts.frames is not None and opts.frames < 1):
            raise ValueError("k, foveated k, batch and frames must be at least 1; start at least 0")
        if opts.fovea < 16:
            raise ValueError("fovea must be at least 16 LR pixels")
        opts.fovea_outline, opts.fovea_gaze = bool(opts.fovea_outline), bool(opts.fovea_gaze)
        opts.fovea_sr, opts.fovea_ane, opts.fovea_overlap = bool(opts.fovea_sr), bool(opts.fovea_ane), bool(opts.fovea_overlap)
        opts.name = str(opts.name).strip() or cls.name
        return opts

    def to_dict(self):
        d = asdict(self)
        d["video"] = Path(self.video).name
        d["gaze"] = self.gaze
        d["source_names"] = self.source_names()
        return d


def stretch_horizon(k, fps):
    """The foveated source's horizon: the gaze step (1-FUTURE, at MODEL_HZ) at the end of a stretch of
    k video frames at `fps`, rounded up, so the predicted gaze that places the crop (and the ring)
    is where the gaze will be when the stretch's last frame is shown (k = 1 -> step 1, the next
    sample; k = 3 at 60 fps -> step 2, 67 ms)."""
    return min(FUTURE, max(1, math.ceil(k * MODEL_HZ / float(fps))))


def normalize_sources(sources):
    """A list (or comma-separated string) of source keys -> the enabled keys in SOURCES order.
    Raises ValueError for unknown keys or an empty selection."""
    if isinstance(sources, str):
        sources = [s for s in (p.strip() for p in sources.split(",")) if s]
    sources = list(sources or [])
    unknown = [s for s in sources if s not in SOURCES]
    if unknown:
        raise ValueError(f"Unknown sources: {', '.join(unknown)} (choose from {', '.join(SOURCES)})")
    if not sources:
        raise ValueError("Enable at least one NDI source")
    return [k for k in SOURCES if k in sources]


def list_files(folder, *suffixes):
    """Names of the files in `folder` with one of `suffixes`, sorted."""
    if not folder.is_dir():
        return []
    return sorted(p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in suffixes)


class LiveSettings:
    """Settings that take effect on the next frame: the head pose source, which the stream polls;
    the page calls set_* from the server's threads."""

    def __init__(self, trace_path=None):
        self._lock = threading.Lock()
        self.live_head = LiveHead()
        # Live by default: until a pose is sent, the head stays at the centre.
        self.head_mode = "live"
        self.head_trace = None
        self.head_source = self.live_head
        if trace_path:
            self.set_head("trace", trace_path)

    def set_head(self, mode, trace_path=None):
        """Switches the head pose source: 'trace' (CSV path) or 'live' (set_pose)."""
        if mode not in HEAD_MODES:
            raise ValueError(f"head mode must be one of {HEAD_MODES}")
        if mode == "trace":
            if not trace_path:
                raise ValueError("trace mode needs a head trace file")
            source = HeadTrace.from_csv(trace_path)  # raises if missing or malformed
        else:
            source, trace_path = self.live_head, None
        with self._lock:
            self.head_mode, self.head_trace, self.head_source = mode, trace_path, source

    def set_pose(self, azimuth, elevation):
        self.live_head.set(azimuth, elevation)

    def snapshot(self):
        with self._lock:
            trace = Path(self.head_trace).name if self.head_trace else None
            return {"head_mode": self.head_mode, "head_trace": trace,
                    "head_pose": self.live_head.pose_at(0)}


class StreamStatus:
    """Progress of the current run: state is idle, loading, streaming, stopped, done or error."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data = {"state": "idle", "error": None, "sent": 0, "fps": 0.0, "gaze": None, "gaze_info": None, "head": None, "perf": None,
                      "options": None, "size": None}

    def update(self, **values):
        with self._lock:
            self._data.update(values)

    def snapshot(self):
        with self._lock:
            return dict(self._data)
