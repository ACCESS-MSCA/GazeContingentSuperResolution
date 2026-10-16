import numpy as np
import pytest

pytest.importorskip("torch")

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
