import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from control_page.server import make_server
from gaze.tracker import GazeTracker, HeadTrace, LiveHead
from stream import control
from stream.control import LiveSettings, StreamOptions, StreamStatus


@pytest.fixture
def videos(tmp_path, monkeypatch):
    (tmp_path / "clip.mp4").write_bytes(b"")
    (tmp_path / "head.csv").write_text("time,azimuth,elevation\n0,0,0\n1,10,5\n")
    monkeypatch.setattr(control, "VIDEOS", tmp_path)
    monkeypatch.setattr("control_page.server.VIDEOS", tmp_path)
    return tmp_path


class FakeController:
    def __init__(self):
        self.live = LiveSettings()
        self.status = StreamStatus()
        self.started = []
        self.stopped = 0

    def start(self, opts):
        self.started.append(opts)

    def stop(self):
        self.stopped += 1


@pytest.fixture
def server(videos):
    controller = FakeController()
    controller.monitor_opened = 0

    def monitor():
        controller.monitor_opened += 1

    def sources():
        return [{"host": "OTHER-PC", "stream": "Camera 1"}, {"host": "HERE.LOCAL", "stream": "StreamNDI-SR"}]

    srv = make_server(controller, StreamOptions(video=str(videos / "clip.mp4")), port=0, monitor=monitor,
                      sources=sources)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"

    def call(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as res:
                return res.status, res.read()
        except urllib.error.HTTPError as err:
            return err.code, err.read()

    yield controller, call
    srv.shutdown()
    srv.server_close()


def test_page_and_state(server):
    _, call = server
    code, page = call("/")
    assert code == 200 and b"Stream control" in page
    code, body = call("/api/state")
    state = json.loads(body)
    assert state["videos"] == ["clip.mp4"] and state["traces"] == ["head.csv"]
    assert state["options"]["video"] == "clip.mp4"
    assert state["live"]["head_mode"] == "live" and state["status"]["state"] == "idle"
    assert StreamOptions().sources == ["source"] and StreamOptions().gaze is False
    assert StreamOptions().fovea_outline is True
    assert StreamOptions().fovea_gaze is True


def test_start_validates_and_converts_options(server, videos):
    controller, call = server
    code, _ = call("/api/start", {"options": {"video": "clip.mp4", "sources": ["foveated", "source", "vsr"], "k": "2",
                                              "frames": "", "device": "", "fovea_outline": 1}})
    assert code == 200
    opts = controller.started[-1]
    assert opts.video == str(videos / "clip.mp4") and opts.k == 2 and opts.frames is None and opts.device is None
    assert opts.sources == ["source", "vsr", "foveated"]  # canonical order
    assert opts.gaze is True and opts.fovea_outline is True
    assert opts.source_names() == {"source": "StreamNDI-Source", "foveated": "StreamNDI-Foveated", "vsr": "StreamNDI-VSR"}


def test_vsr_crop_is_optional(server, videos):
    controller, call = server
    assert call("/api/start", {"options": {"video": "clip.mp4", "vsr_crop": ""}})[0] == 200
    assert controller.started[-1].vsr_crop is None  # whole frame
    assert call("/api/start", {"options": {"video": "clip.mp4", "sources": ["vsr"], "vsr_crop": "64"}})[0] == 200
    assert controller.started[-1].vsr_crop == 64
    assert controller.started[-1].source_names()["vsr"] == "StreamNDI-VSR-Cropped"
    assert call("/api/start", {"options": {"video": "clip.mp4", "sources": ["source", "bicubic", "foveated"], "vsr_crop": "64"}})[0] == 200
    assert controller.started[-1].source_names() == {  # the whole video pipeline is cropped, not foveated
        "source": "StreamNDI-Source-Cropped", "bicubic": "StreamNDI-Bicubic-Cropped", "foveated": "StreamNDI-Foveated"}


def test_foveated_sr_can_be_off(server, videos):
    controller, call = server
    assert call("/api/start", {"options": {"video": "clip.mp4", "sources": ["foveated"], "fovea_sr": False}})[0] == 200
    opts = controller.started[-1]
    assert opts.fovea_sr is False and opts.gaze is True  # the gaze model still runs
    assert StreamOptions().fovea_sr is True


@pytest.mark.parametrize("options", [
    {"video": "../secret.mp4"}, {"video": "missing.mp4"}, {"video": "clip.mp4", "sources": ["x"]},
    {"video": "clip.mp4", "k": 0}, {"video": "clip.mp4", "device": "tpu"}, {"video": "clip.mp4", "nope": 1},
    {"video": "clip.mp4", "sources": []}, {"video": "clip.mp4", "model": "vsr"}, {"video": "clip.mp4", "fovea": 4},
    {"video": "clip.mp4", "vsr_crop": 8},
])
def test_start_rejects_bad_options(server, options):
    controller, call = server
    code, body = call("/api/start", {"options": options})
    assert code == 400 and "error" in json.loads(body)
    assert controller.started == []



def test_live_settings_and_head_pose(server):
    controller, call = server
    assert call("/api/live", {"head_mode": "trace", "head_trace": "head.csv"})[0] == 200
    assert isinstance(controller.live.head_source, HeadTrace)
    assert controller.live.head_source.pose_at(0.5) == pytest.approx((5.0, 2.5))
    assert call("/api/live", {"head_mode": "trace", "head_trace": "../x.csv"})[0] == 400

    assert call("/api/live", {"head_mode": "live"})[0] == 200
    assert call("/api/live", {"head_mode": "still"})[0] == 400  # removed: Live with no input is the still head
    assert call("/api/head", {"azimuth": 190, "elevation": 100})[0] == 200
    assert controller.live.head_source.pose_at(123.0) == pytest.approx((-170.0, 90.0))
    assert call("/api/head", {"azimuth": "left"})[0] == 400


def test_stop_and_unknown_paths(server):
    controller, call = server
    assert call("/api/stop", {})[0] == 200 and controller.stopped == 1
    assert call("/api/nope", {})[0] == 404
    assert call("/nope")[0] == 404
    assert call("/api/live", [1, 2])[0] == 400


def test_live_head_is_sampled_by_tracker_and_switching_restarts_window():
    class Recorder:
        def __init__(self):
            self.poses = []

        def reset(self):
            self.poses.append("reset")

        def push(self, frame, pose):
            self.poses.append(pose)

    rec = Recorder()
    head = LiveHead()
    tracker = GazeTracker(30, HeadTrace.still(), predictor=rec)
    frame = np.zeros((4, 4, 3), np.uint8)
    tracker.update(frame)
    tracker.set_source(head)
    head.set(20, -5)
    tracker.update(frame)
    head.set(25, -5)
    tracker.update(frame)
    assert rec.poses == ["reset", (0.0, 0.0), "reset", (20.0, -5.0), (25.0, -5.0)]


def test_monitor_button_opens_monitor(server):
    controller, call = server
    assert call("/api/monitor", {})[0] == 200
    assert controller.monitor_opened == 1


def test_monitor_error_is_reported(videos):
    def broken():
        raise ValueError("Could not open NDI Video Monitor (is NDI Tools installed?)")

    srv = make_server(FakeController(), StreamOptions(video=str(videos / "clip.mp4")), port=0, monitor=broken)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    req = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}/api/monitor", data=b"{}")
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req)
    assert err.value.code == 400 and b"NDI Tools" in err.value.read()
    srv.shutdown()
    srv.server_close()


def test_state_lists_ndi_sources_and_marks_this_computer(server, monkeypatch):
    monkeypatch.setattr("socket.gethostname", lambda: "here.local")
    _, call = server
    ndi = json.loads(call("/api/state")[1])["ndi"]
    assert ndi == [{"host": "OTHER-PC", "stream": "Camera 1", "local": False},
                   {"host": "HERE.LOCAL", "stream": "StreamNDI-SR", "local": True}]



def test_state_survives_failing_ndi_discovery(videos):
    def broken():
        raise RuntimeError("Invalid source")

    srv = make_server(FakeController(), StreamOptions(video=str(videos / "clip.mp4")), port=0, sources=broken)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with urllib.request.urlopen(f"http://127.0.0.1:{srv.server_address[1]}/api/state") as res:
        assert json.loads(res.read())["ndi"] == []
    srv.shutdown()
    srv.server_close()


def test_gaze_model_runs_only_for_sources_that_need_it():
    assert StreamOptions(sources=["source", "bicubic", "vsr"]).gaze is False
    assert StreamOptions(sources=["foveated"]).gaze is True
    assert StreamOptions(sources=["foveated"]).to_dict()["gaze"] is True


def test_horizon_follows_foveated_k():
    from stream.control import stretch_horizon
    # k video frames at 60 fps -> gaze steps at 30 Hz, rounded up
    assert [stretch_horizon(k, 60) for k in (1, 2, 3, 5)] == [1, 1, 2, 3]
    assert stretch_horizon(3, 30) == 3 and stretch_horizon(100, 30) == 20  # capped at the last step
