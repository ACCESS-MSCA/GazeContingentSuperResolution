import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gaze import FUTURE, PAST, TASK_HORIZONS, TASK_NAMES, GazePredictor


def window():
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (90, 120, 3), dtype=np.uint8) for _ in range(PAST)]
    hmd = np.cumsum(rng.normal(0, 2, (PAST, 2)), 0) + [170, 10]
    return frames, hmd


@pytest.fixture(scope="module")
def predictor():
    return GazePredictor(device="cpu")


def test_matches_original_pipeline(predictor):
    # Reference values from the original GazePrediction code (core.models.Conv_LSTM + WrappedUNISAL).
    out = predictor.predict(*window())
    np.testing.assert_allclose(out["offsets"][[0, 9, 19]],
                               [[4.0593, -7.2503], [9.2592, -1.9435], [13.477, 1.9898]], atol=1e-3)
    np.testing.assert_allclose(out["task_probs"], [0.0748, 0.0115, 0.8917, 0.0219], atol=1e-4)
    assert out["task"] == "Saliency"
    assert out["horizon"] == TASK_HORIZONS[TASK_NAMES.index("Saliency")]


def test_absolute_gaze_is_head_plus_offset_wrapped(predictor):
    frames, hmd = window()
    out = predictor.predict(frames, hmd)
    assert out["offsets"].shape == out["gaze"].shape == (FUTURE, 2)
    assert np.all((out["gaze"][:, 0] >= -180) & (out["gaze"][:, 0] < 180))
    diff = (out["gaze"][:, 0] - hmd[-1, 0] - out["offsets"][:, 0] + 180) % 360 - 180
    np.testing.assert_allclose(diff, 0, atol=1e-3)


def test_push_matches_predict(predictor):
    frames, hmd = window()
    predictor.reset()
    results = [predictor.push(f, h) for f, h in zip(frames, hmd)]
    assert all(r is None for r in results[:-1])
    np.testing.assert_allclose(results[-1]["offsets"], predictor.predict(frames, hmd)["offsets"], atol=1e-5)


def test_rejects_wrong_window_length(predictor):
    frames, hmd = window()
    with pytest.raises(ValueError):
        predictor.predict(frames[1:], hmd[1:])
