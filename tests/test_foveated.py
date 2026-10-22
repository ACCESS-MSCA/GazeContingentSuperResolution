import numpy as np
import pytest

torch = pytest.importorskip("torch")

from upscale.foveated import FoveatedStream, fovea_box, fovea_mask


class FakeSR:
    """Stands in for VSRPipeline(k=1): fine detail (a 1-pixel checkerboard, 100/200) at a
    shifted brightness (mean 150), like the real model on small crops."""

    def __init__(self):
        self.calls = []

    def __call__(self, crops):
        self.calls.append([c.shape for c in crops])
        out = []
        for c in crops:
            h, w = c.shape[0] * 4, c.shape[1] * 4
            board = ((np.indices((h, w)).sum(0) % 2) * 100 + 100).astype(np.uint8)
            out.append(np.repeat(board[..., None], 3, axis=2))
        return out


def test_box_is_centred_on_gaze_and_kept_inside():
    assert fovea_box((0.5, 0.5), 240, 136, 96) == (72, 20, 168, 116)
    assert fovea_box(None, 240, 136, 96) == (72, 20, 168, 116)  # no gaze yet: centre
    assert fovea_box((0.0, 0.0), 240, 136, 96) == (0, 0, 96, 96)
    assert fovea_box((1.0, 1.0), 240, 136, 96) == (144, 40, 240, 136)


def test_mask_is_full_in_centre_and_zero_at_edges():
    m = fovea_mask(64)
    assert m.shape == (64, 64, 1)
    assert m[32, 32, 0] == pytest.approx(1.0, abs=1e-3)
    assert m[0, 0, 0] == 0 and m[32, 0, 0] < 0.05  # corner, edge midpoint
    assert np.all(np.diff(m[32, 32:, 0]) <= 1e-6)  # falls off outwards


def test_sr_detail_on_bicubic_colours_at_the_gaze_only():
    sr = FakeSR()
    stream = FoveatedStream(sr, fovea=32)
    frames = [np.full((64, 96, 3), 60, np.uint8) for _ in range(2)]  # flat: bicubic stays 60
    out = stream.push(frames, [(0.25, 0.5), None])
    assert len(out) == 2 and out[0].shape == (256, 384, 3)
    assert sr.calls == [[(32, 32, 3), (32, 32, 3)]]
    x0, y0, x1, y1 = stream.last_boxes[0]
    assert (x0, y0, x1, y1) == (32, 64, 160, 192)  # LR (8, 16)-(40, 48) x4
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    centre = out[0][cy - 16:cy + 16, cx - 16:cx + 16].astype(np.float32)
    assert centre.std() > 30  # the SR's detail is there...
    assert centre.mean() == pytest.approx(60, abs=2)  # ...on bicubic's brightness, not the SR's 150
    assert np.all(out[0][:, 200:] == 60)  # bicubic elsewhere
    assert stream.last_boxes[1] == (128, 64, 256, 192)  # second frame: centre
    assert stream.flush() == []


def test_fovea_is_capped_at_the_frame():
    sr = FakeSR()
    stream = FoveatedStream(sr, fovea=200)
    stream.push([np.zeros((40, 60, 3), np.uint8)], [(0.5, 0.5)])
    assert sr.calls == [[(40, 40, 3)]]


def test_rejects_tiny_fovea():
    with pytest.raises(ValueError):
        FoveatedStream(FakeSR(), fovea=8)


def test_stage_timers_count_frames():
    stream = FoveatedStream(FakeSR(), fovea=32)
    frames = [np.full((64, 64, 3), 80, np.uint8) for _ in range(3)]
    stream.push(frames, [(0.5, 0.5)] * 3)
    assert stream.frames == 3
    assert stream.sr_s > 0 and stream.blend_s > 0


# --- k > 1: SR on keyframe crops, RIFE in between (stub models from the routing tests) ---

def _push_all(stream, n, gaze_of, batch=2):
    """Pushes n flat 64x64 frames (frame i all i) in batches; returns (outputs, boxes)."""
    frames = [np.full((64, 64, 3), i, np.uint8) for i in range(n)]
    out, boxes = [], []
    for s in range(0, n, batch):
        out += stream.push(frames[s:s + batch], [gaze_of(i) for i in range(s, min(s + batch, n))])
        boxes += stream.last_boxes
    out += stream.flush()
    boxes += stream.last_boxes
    return out, boxes


def test_keyframes_and_rife_with_a_still_gaze():
    from test_pipeline_routing import _build
    pipeline = _build(3)
    stream = FoveatedStream(pipeline, fovea=32, k=3)
    out, boxes = _push_all(stream, 9, lambda i: (0.5, 0.5))
    assert [o[0, 0, 0] for o in out] == list(range(9))  # every frame, in order (bicubic outside the fovea)
    assert len(boxes) == 9 and len(set(boxes)) == 1
    # keyframes 0, 3, 6 and the clip's last frame 8; the crop never moves, so each SR'd once
    assert stream.sr_frames == 4 and stream.rife_frames == 5 and stream.frames == 9
    # one batched call per stretch, each crop at its place in time
    assert [t for _, t in pipeline.interpolation.calls] == [[1 / 3, 2 / 3], [1 / 3, 2 / 3], [1 / 2]]


def test_frames_wait_for_their_closing_keyframe():
    from test_pipeline_routing import _build
    stream = FoveatedStream(_build(3), fovea=32, k=3)
    frames = [np.full((64, 64, 3), i, np.uint8) for i in range(4)]
    assert len(stream.push(frames[:2], [None, None])) == 1  # keyframe 0 only; 1 waits for 3
    assert len(stream.push(frames[2:], [None, None])) == 3  # 1, 2 and keyframe 3


def test_crop_moves_per_stretch_and_reupscales_the_opening_keyframe():
    from test_pipeline_routing import _build
    stream = FoveatedStream(_build(3), fovea=16, k=3)
    # gaze jumps from the left to the right edge after frame 3
    out, boxes = _push_all(stream, 7, lambda i: (0.1, 0.5) if i <= 3 else (0.9, 0.5))
    assert len(out) == 7
    assert len(set(boxes[1:4])) == 1 and len(set(boxes[4:7])) == 1 and boxes[1] != boxes[4]
    assert stream.sr_frames == 4  # keyframes 0, 3, 6, plus 3 again at the new crop


def test_k_above_one_needs_the_interpolation_model():
    with pytest.raises(ValueError):
        FoveatedStream(FakeSR(), fovea=32, k=2)


def test_bgra_output_blends_into_the_shared_bicubic_frames():
    from upscale.bicubic import BicubicStream
    stream = FoveatedStream(FakeSR(), fovea=32, bgra=True)
    frames = [np.full((64, 96, 3), (10, 20, 30), np.uint8)]
    lr, bgra = base = BicubicStream().upscale_bgra(frames)
    sent_by_bicubic = bgra[0].numpy().copy()  # what the bicubic source sends, before foveated
    out = stream.push(frames, [(0.5, 0.5)], base=base)
    assert out[0].shape == (256, 384, 4) and out[0].dtype == np.uint8
    assert tuple(out[0][0, 0]) == (30, 20, 10, 255)  # B, G, R, A outside the fovea
    assert np.array_equal(bgra[0].numpy(), out[0])  # blended into the shared frame in place
    assert not np.array_equal(sent_by_bicubic, out[0])  # (the bicubic source copied it first)


def test_region_bicubic_matches_the_whole_frame():
    from upscale.bicubic import BicubicStream
    up = BicubicStream()
    lr = torch.rand(1, 3, 40, 60)
    whole = torch.nn.functional.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)
    for x0, y0, x1, y1 in ((10, 8, 26, 24), (0, 0, 16, 16), (44, 24, 60, 40)):  # inside, and at the corners
        region = up.upscale_region(lr, (x0, y0, x1, y1))
        assert torch.allclose(region, whole[..., 4 * y0:4 * y1, 4 * x0:4 * x1], atol=1e-5)


def test_ane_sends_the_crops_to_the_neural_engine():
    from test_pipeline_routing import _build
    pipeline = _build(3)
    stream = FoveatedStream(pipeline, fovea=40, k=3, ane=True)
    stream.prepare(64, 48)  # loads it for the crop size, padded to a multiple of 16
    assert pipeline.sr.neural_engine_sizes == [(48, 48)]
    out, _ = _push_all(stream, 7, lambda i: (0.5, 0.5))
    assert len(out) == 7
    assert pipeline.sr.neural_engine_calls == len(pipeline.sr.calls) == 3  # keyframes 0, 3, 6


def test_without_ane_nothing_goes_to_the_neural_engine():
    from test_pipeline_routing import _build
    pipeline = _build(3)
    stream = FoveatedStream(pipeline, fovea=32, k=3)
    stream.prepare(64, 64)
    _push_all(stream, 4, lambda i: None)
    assert pipeline.sr.neural_engine_calls == 0 and pipeline.sr.neural_engine_sizes == []


def test_overlap_gives_the_same_frames_one_stretch_later():
    from test_pipeline_routing import _build
    gaze_of = lambda i: (0.1, 0.5) if i <= 4 else (0.9, 0.5)  # the crop moves once
    plain = FoveatedStream(_build(3), fovea=16, k=3, ane=True)
    overlapped = FoveatedStream(_build(3), fovea=16, k=3, ane=True, overlap=True)
    out_plain, boxes_plain = _push_all(plain, 10, gaze_of)
    out_overlap, boxes_overlap = _push_all(overlapped, 10, gaze_of)
    assert boxes_overlap == boxes_plain
    assert all(np.array_equal(a, b) for a, b in zip(out_overlap, out_plain)) and len(out_overlap) == 10
    assert overlapped.sr_frames == plain.sr_frames
    assert overlapped.sr.sr.neural_engine_calls == plain.sr.sr.neural_engine_calls


def test_overlap_holds_frames_until_the_next_stretch_has_started():
    from test_pipeline_routing import _build
    stream = FoveatedStream(_build(3), fovea=32, k=3, ane=True, overlap=True)
    frames = [np.full((64, 64, 3), i, np.uint8) for i in range(7)]
    assert stream.push(frames[:2], [None, None]) == []  # keyframe 0 started, not finished
    assert len(stream.push(frames[2:4], [None, None])) == 1  # keyframe 3 started: 0 is done
    assert len(stream.push(frames[4:7], [None] * 3)) == 3  # keyframe 6 started: 1-3 are done
    assert len(stream.flush()) == 3  # 4-6


def test_overlap_needs_the_neural_engine():
    from test_pipeline_routing import _build
    with pytest.raises(ValueError):
        FoveatedStream(_build(3), fovea=32, k=3, overlap=True)
