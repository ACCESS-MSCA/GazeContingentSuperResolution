"""Gaze prediction without eye tracking: UniSal saliency + HMD motion -> gaze for the next 666 ms.

The model API (predict_gaze, GazePredictor, ...) loads lazily, so the stream-side modules
gaze.tracker (head pose, 30 Hz scheduling, threaded tracker) and gaze.process (model in a
child process) can be imported without loading torch.
"""

_MODEL_API = ("FUTURE", "PAST", "TASK_HORIZONS", "TASK_NAMES", "GazePredictor", "predict_gaze")

__all__ = list(_MODEL_API)


def __getattr__(name):
    if name in _MODEL_API:
        from . import inference

        return getattr(inference, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
