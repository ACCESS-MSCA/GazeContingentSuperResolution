import threading
import time

import numpy as np
import pytest

from gaze.tracker import (
    DEFAULT_FOV,
    GazeTracker,
    HeadTrace,
    ThreadedGazeTracker,
    offset_to_uv,
)
from ndi.overlay import draw_gaze


class FakePredictor:
    """Records pushes; predicts an offset equal to the latest head pose after 15 samples."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.poses = []

    def push(self, frame, pose):
        self.poses.append(pose)
        if len(self.poses) < 15:
            return None
        offsets = np.tile(np.array(pose, np.float32), (20, 1))
        offsets[:, 0] += np.arange(20)  # step k -> azimuth + (k - 1)
        return {"offsets": offsets}


def _run(fps, num_frames, **kwargs):
    predictor = FakePredictor()
    tracker = GazeTracker(fps, predictor=predictor, **kwargs)
    frame = np.zeros((8, 8, 3), np.uint8)
    return predictor, [tracker.update(frame) for _ in range(num_frames)]


@pytest.mark.parametrize("fps, num_frames, ticks", [(30, 30, 30), (60, 60, 30), (50, 50, 30), (24, 24, 29)])
def test_model_runs_at_30_hz_for_any_fps(fps, num_frames, ticks):
    predictor, _ = _run(fps, num_frames)
    assert len(predictor.poses) == ticks


def test_no_gaze_until_window_is_full():
    _, uvs = _run(30, 20)
    assert all(uv is None for uv in uvs[:14])
    assert all(uv is not None for uv in uvs[14:])


def test_still_head_looks_at_centre():
    _, uvs = _run(30, 20)
    assert uvs[-1] == pytest.approx((0.5, 0.5))


def test_step_selects_prediction_horizon():
    _, uvs = _run(30, 20, step=11)
    assert uvs[-1] == pytest.approx(offset_to_uv((10.0, 0.0)))


def test_reset_restarts_window():
    predictor = FakePredictor()
    tracker = GazeTracker(30, predictor=predictor)
    frame = np.zeros((8, 8, 3), np.uint8)
    for _ in range(20):
        tracker.update(frame)
    tracker.reset()
    assert tracker.update(frame) is None
    assert len(predictor.poses) == 1


def test_trace_is_sampled_at_video_time_with_start_offset():
    trace = HeadTrace([0.0, 10.0], [0.0, 100.0], [0.0, -50.0])
    predictor, _ = _run(30, 30, trace=trace, start=2.0)
    assert predictor.poses[0] == pytest.approx((20.0, -10.0))
    assert predictor.poses[-1] == pytest.approx((20.0 + 29 / 3, -10.0 - 29 / 6))


def test_trace_interpolates_across_wraparound():
    trace = HeadTrace([0.0, 1.0], [170.0, -170.0], [0.0, 0.0])
    assert trace.pose_at(0.5)[0] == pytest.approx(-180.0)
    assert trace.pose_at(0.25)[0] == pytest.approx(175.0)


def test_trace_from_csv(tmp_path):
    path = tmp_path / "head.csv"
    path.write_text("time,azimuth,elevation\n0,0,0\n1,10,5\n")
    assert HeadTrace.from_csv(path).pose_at(0.5) == pytest.approx((5.0, 2.5))


def test_offset_to_uv_maps_field_of_view_to_frame():
    assert offset_to_uv((0, 0)) == (0.5, 0.5)
    assert offset_to_uv((DEFAULT_FOV[0] / 2, DEFAULT_FOV[1] / 2)) == pytest.approx((1.0, 0.0))
    assert offset_to_uv((500, -500)) == (1.0, 1.0)  # clamped


def test_step_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        GazeTracker(30, step=21, predictor=FakePredictor())


def test_draw_gaze_marks_point_and_skips_none():
    frame = np.zeros((100, 200, 4), np.uint8)
    draw_gaze(frame, None)
    assert not frame.any()
    draw_gaze(frame, (0.25, 0.5))
    assert tuple(frame[50, 50]) == (0, 255, 0, 255)  # centre dot
    assert not frame[:, 120:].any()  # nothing far from the point
    draw_gaze(frame, (1.0, 1.0))  # corner: must not raise


def test_gaze_point_is_plain_floats_for_json():
    uv = offset_to_uv(np.array([3.0, -2.0], np.float32))
    assert all(type(x) is float for x in uv)


def test_path_holds_all_steps_and_matches_shown_step():
    predictor = FakePredictor()
    tracker = GazeTracker(30, predictor=predictor, step=5)
    frame = np.zeros((8, 8, 3), np.uint8)
    for _ in range(15):
        tracker.update(frame)
    assert len(tracker.path) == 20
    assert tracker.path[4] == tracker.uv
    assert tracker.path[0] == pytest.approx(offset_to_uv((0.0, 0.0)))
    assert tracker.path[19] == pytest.approx(offset_to_uv((19.0, 0.0)))
    tracker.reset()
    assert tracker.path is None


def test_info_has_shown_step_offset_and_task_probs():
    class WithTask(FakePredictor):
        def push(self, frame, pose):
            out = super().push(frame, pose)
            if out is not None:
                out["task_probs"] = np.array([0.1, 0.2, 0.6, 0.1], np.float32)
            return out

    tracker = GazeTracker(30, predictor=WithTask(), step=3)
    frame = np.zeros((8, 8, 3), np.uint8)
    for _ in range(15):
        tracker.update(frame)
    info = tracker.info
    assert info["step"] == 3 and info["offset"] == pytest.approx([2.0, 0.0])
    assert info["task_probs"] == pytest.approx([0.1, 0.2, 0.6, 0.1])
    assert all(type(p) is float for p in info["task_probs"])  # JSON for the control page
    assert info["path"] is tracker.path


# --- ThreadedGazeTracker ---------------------------------------------------------



def test_threaded_matches_synchronous_once_caught_up():
    frame = np.zeros((8, 8, 3), np.uint8)
    trace = HeadTrace([0.0, 2.0], [0.0, 40.0], [0.0, 10.0])
    sync = GazeTracker(30, trace, step=7, predictor=FakePredictor())
    threaded = ThreadedGazeTracker(30, trace, step=7, predictor=FakePredictor())
    try:
        for _ in range(40):
            expected = sync.update(frame)
            threaded.update(frame)
            threaded.wait()  # the worker keeps up, as at 30 Hz in a real stream
        assert threaded.dropped == 0
        assert threaded.latest()[0] == pytest.approx(expected)
        assert threaded.latest()[1]["path"] == pytest.approx(sync.path)
    finally:
        threaded.close()


class SlowPredictor(FakePredictor):
    def __init__(self, delay=0.0, gate=None):
        super().__init__()
        self.delay, self.gate, self.resets = delay, gate, 0

    def reset(self):
        super().reset()
        self.resets = getattr(self, "resets", 0) + 1

    def push(self, frame, pose):
        if self.gate is not None:
            self.gate.wait()
        time.sleep(self.delay)
        return super().push(frame, pose)


def test_update_does_not_wait_for_the_model():
    tracker = ThreadedGazeTracker(30, predictor=SlowPredictor(delay=0.05))
    try:
        frame = np.zeros((8, 8, 3), np.uint8)
        start = time.perf_counter()
        for _ in range(10):
            tracker.update(frame)
        assert time.perf_counter() - start < 0.05  # 10 model calls would take 0.5 s
    finally:
        tracker.close()


def test_falling_behind_drops_samples_and_restarts_window():
    gate = threading.Event()
    predictor = SlowPredictor(gate=gate)
    tracker = ThreadedGazeTracker(30, predictor=predictor)
    try:
        frame = np.zeros((8, 8, 3), np.uint8)
        for _ in range(ThreadedGazeTracker.MAX_QUEUED + 10):
            tracker.update(frame)
        assert tracker.dropped >= 10
        gate.set()
        tracker.wait()
        resets_before = predictor.resets
        tracker.update(frame)  # first sample after the gap starts a fresh window
        tracker.wait()
        assert predictor.resets == resets_before + 1
        assert len(predictor.poses) == 1
    finally:
        gate.set()
        tracker.close()


def test_model_error_surfaces_in_update():
    class Broken(FakePredictor):
        def push(self, frame, pose):
            raise ValueError("boom")

    tracker = ThreadedGazeTracker(30, predictor=Broken())
    try:
        frame = np.zeros((8, 8, 3), np.uint8)
        tracker.update(frame)
        tracker.wait()
        with pytest.raises(RuntimeError, match="gaze model failed"):
            tracker.update(frame)
    finally:
        tracker.close()


def test_close_stops_worker():
    tracker = ThreadedGazeTracker(30, predictor=FakePredictor())
    tracker.close()
    assert not tracker._worker.is_alive()


# --- ProcessPredictor (model in a child process) ---------------------------------

def fake_factory():
    """Importable by the child process."""
    return FakePredictor()


def failing_factory():
    raise ValueError("no weights")


def test_process_predictor_matches_in_process_predictor():
    from gaze.process import ProcessPredictor

    remote, local = ProcessPredictor(fake_factory), FakePredictor()
    try:
        frame = np.zeros((8, 8, 3), np.uint8)
        for i in range(20):
            a, b = remote.push(frame, (i, -i)), local.push(frame, (i, -i))
            assert (a is None) == (b is None)
            if b is not None:
                np.testing.assert_allclose(a["offsets"], b["offsets"])
        remote.reset()
        local.reset()
        assert remote.push(frame, (0, 0)) is None  # window restarted in the child too
    finally:
        remote.close()


def test_process_predictor_reports_load_errors():
    from gaze.process import ProcessPredictor

    with pytest.raises(RuntimeError, match="no weights"):
        ProcessPredictor(failing_factory)


def test_threaded_tracker_with_process_predictor():
    from gaze.process import ProcessPredictor

    predictor = ProcessPredictor(fake_factory)
    tracker = ThreadedGazeTracker(30, predictor=predictor, step=20)
    try:
        frame = np.zeros((8, 8, 3), np.uint8)
        for _ in range(20):
            tracker.update(frame)
            tracker.wait()
        assert tracker.latest()[0] == pytest.approx(offset_to_uv((19.0, 0.0)))
    finally:
        tracker.close()
        predictor.close()


def test_threaded_stats_count_predictions_and_lag():
    tracker = ThreadedGazeTracker(30, predictor=FakePredictor())
    try:
        frame = np.zeros((8, 8, 3), np.uint8)
        for _ in range(20):
            tracker.update(frame)
            tracker.wait()
        st = tracker.stats()
        assert st["predictions"] == 20
        assert st["lag_samples"] == 0 and st["queued"] == 0 and st["dropped"] == 0
        assert st["roundtrip_s"] >= 0 and st["model_s"] >= 0
        tracker.reset()
        tracker.wait()
        assert tracker.stats()["lag_samples"] is None  # no prediction since the restart
    finally:
        tracker.close()


def test_process_predictor_reports_child_stats():
    from gaze.process import ProcessPredictor

    p = ProcessPredictor(fake_factory)
    try:
        p.push(np.zeros((8, 8, 3), np.uint8), (0, 0))
        assert set(p.last_stats) == {"model_s", "cpu_s", "mps_mb"}
        assert p.last_stats["model_s"] >= 0 and p.last_stats["cpu_s"] > 0
    finally:
        p.close()


def test_draw_gaze_path_blends_dots_stronger_for_later_steps():
    from ndi.overlay import draw_gaze_path, draw_gaze_prediction

    frame = np.zeros((200, 400, 4), np.uint8)
    path = [(0.1 + 0.04 * i, 0.5) for i in range(20)]
    draw_gaze_path(frame, path)
    first, last = frame[100, int(0.1 * 399), 1], frame[100, int(0.86 * 399), 1]
    assert 0 < first < last == 255  # green channel: faint first step, full last step
    assert frame[:50].max() == 0  # nothing far from the path
    draw_gaze_path(frame, None)  # no prediction yet: no-op
    draw_gaze_prediction(np.zeros((200, 400, 4), np.uint8), (0.5, 0.5), path)  # ring + path, must not raise
    draw_gaze_path(frame, [(1.0, 1.0), (1.2, -0.1)])  # off-frame points are clipped, must not raise
