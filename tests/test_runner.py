"""run_stream end to end with stub models and outputs: no video file, NDI, GPU or weights.

Frames are 64x48 (LR), so the upscaled sources are 256x192. The SR and RIFE stubs are the
routing tests' (SR repeats pixels 4x, RIFE blends linearly) and the gaze model is the tracker
tests' fake, which predicts once it has 15 samples.
"""

from fractions import Fraction

import numpy as np
import pytest

pytest.importorskip("torch")

import ndi.output
import stream.runner as runner
from stream.control import LiveSettings, StreamOptions, StreamStatus
from test_gaze_tracker import FakePredictor
from test_pipeline_routing import _build

W, H = 64, 48


class FakeOutput:
    """Stands in for NDIOutput: records the sources it was given and every frame sent."""

    instance = None

    def __init__(self, sources, fps, audio_data=None):
        self.sources = sources  # {key: (name, width, height)}
        self.sent = []
        self.idle_s = 0.0
        self.dropped_timing_count = 0
        FakeOutput.instance = self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def send(self, frames):
        self.sent.append({k: f.shape for k, f in frames.items()})

    def restart(self):
        pass


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    """Patches the video, audio, NDI and models; returns the stub SR pipeline."""
    pipeline = _build(1)
    pipeline.load_interpolation = lambda *a: None  # the stub already has RIFE
    monkeypatch.setattr(runner, "probe_video", lambda path: (W, H, Fraction(30), None))
    monkeypatch.setattr(runner, "read_rgb_frames", lambda path, max_frames=None, start=0.0: (
        np.full((H, W, 3), i, np.uint8) for i in range(max_frames or 30)))
    monkeypatch.setattr(runner, "decode_audio_to_array", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_vsr_pipeline", lambda cache, device, k: pipeline)
    monkeypatch.setattr(ndi.output, "NDIOutput", FakeOutput)
    import gaze.process
    monkeypatch.setattr(gaze.process, "ProcessPredictor", lambda **kw: FakePredictor())
    return pipeline


def _run(**options):
    status = StreamStatus()
    opts = StreamOptions(video="clip.mp4", once=True, device="cpu", **options)
    runner.run_stream(opts, LiveSettings(), status)
    return FakeOutput.instance, status.snapshot()


def test_crop_applies_to_the_whole_video_pipeline(stubbed):
    out, status = _run(sources=["source", "bicubic", "vsr"], vsr_crop=16, k=3, frames=9)
    assert {k: v[0] for k, v in out.sources.items()} == {
        "source": "StreamNDI-Source-Cropped", "bicubic": "StreamNDI-Bicubic-Cropped", "vsr": "StreamNDI-VSR-Cropped"}
    assert {k: v[1:] for k, v in out.sources.items()} == {"source": (16, 16), "bicubic": (64, 64), "vsr": (64, 64)}
    # every frame goes out on every source, at the size its NDI source was opened with
    assert len(out.sent) == 9 and status["state"] == "done"
    assert all(f == {"source": (16, 16, 4), "bicubic": (64, 64, 4), "vsr": (64, 64, 4)} for f in out.sent)
    assert status["sizes"]["vsr"] == [64, 64]


def test_no_crop_sends_whole_frames(stubbed):
    out, _ = _run(sources=["source", "bicubic", "vsr"], k=2, frames=6)
    assert {k: v[1:] for k, v in out.sources.items()} == {"source": (W, H), "bicubic": (4 * W, 4 * H), "vsr": (4 * W, 4 * H)}
    assert len(out.sent) == 6


def test_foveated_with_rife_sends_every_frame_in_order(stubbed):
    # k = 3: SR on every 3rd frame's crop, RIFE in between, frames held until their keyframe
    out, status = _run(sources=["foveated"], fovea=16, fovea_k=3, frames=20)
    assert out.sources == {"foveated": ("StreamNDI-Foveated", 4 * W, 4 * H)}  # never cropped
    assert len(out.sent) == 20 and all(f == {"foveated": (4 * H, 4 * W, 4)} for f in out.sent)
    assert status["state"] == "done"
    # keyframes 0, 3, ..., 18 and the clip's last frame 19: 8 keyframe crops (one more SR if the
    # crop moves when the first prediction arrives); RIFE makes the other 12
    assert sum(stubbed.sr.calls) >= 8 and len(stubbed.interpolation.calls) == 12


def test_foveated_without_sr_is_bicubic_with_the_gaze(stubbed):
    out, _ = _run(sources=["foveated"], fovea_sr=False, frames=10)
    assert len(out.sent) == 10
    assert stubbed.sr.calls == []  # the SR model never runs


def test_one_sr_pipeline_whatever_k(monkeypatch):
    """A new k reuses the cached SR model (loading RIFE once) instead of building another."""
    built, rife = [], []

    class FakePipeline:
        def __init__(self, k, device):
            built.append((k, device))

        def load_interpolation(self):
            rife.append(True)

    import upscale.vsr
    monkeypatch.setattr(upscale.vsr, "VSRPipeline", FakePipeline)
    cache = {}
    pipelines = [runner._vsr_pipeline(cache, "cpu", k) for k in (1, 3, 5, 1, 2)]
    assert len(built) == 1 and all(p is pipelines[0] for p in pipelines)
    assert len(rife) == 3  # asked for k = 3, 5 and 2; load_interpolation itself loads only once
