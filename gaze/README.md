# Gaze prediction

Predicts where the viewer will look over the next 666 ms (20 frames at 30 Hz) from what they see and how their head moves, without eye tracking. Inference only, taken from the GazePrediction journal extension, *Behaviour-Aware Gaze Prediction in Virtual Reality Without Eye Tracking*.

## Usage

```python
from gaze import predict_gaze, GazePredictor

out = predict_gaze(frames, hmd)  # last 15 viewport frames + (15, 2) HMD azimuth/elevation in degrees
out["gaze"][out["horizon"] - 1]  # absolute gaze (azimuth, elevation) at the horizon for the predicted task

p = GazePredictor()              # streaming: one frame at a time, saliency computed once per frame
out = p.push(frame, (az, el))    # None until 15 frames are buffered
```

Inputs:
- `frames`: 15 viewport (HMD field-of-view) frames, oldest first, each `(H, W, 3)` RGB uint8, any size (resized to 384×288). The model was trained on the HMD field of view cropped from 360° video, not on full equirectangular frames.
- `hmd`: `(15, 2)` HMD azimuth and elevation in degrees for the same frames, sampled at 30 Hz.

Output dict:

| Key | Shape | Content |
|---|---|---|
| `offsets` | `(20, 2)` | gaze minus head direction, degrees |
| `gaze` | `(20, 2)` | absolute gaze: latest HMD sample + offsets, azimuth wrapped to [-180, 180) |
| `task_probs` | `(4,)` | Free Viewing, Visual Search, Saliency, Track |
| `task` | str | most likely task |
| `horizon` | int | steps ahead to use for that task (13 / 15 / 10 / 20), selected on validation data with a 10° tolerance |

The model was trained relative to the head direction at the first predicted frame, which is not known at inference time, so `gaze` uses the latest HMD sample (33 ms earlier).

Device defaults to CUDA, then MPS, then CPU. One pushed frame takes about 19 ms on MPS (M-series Mac) and 210 ms on CPU, mostly UniSal.

## Contents

| Path | Content |
|---|---|
| `inference.py` | `predict_gaze`, `GazePredictor`, Conv-LSTM definition, preprocessing |
| `tracker.py` | for the stream: head pose sources (trace, live), 30 Hz scheduling per decoded frame, `ThreadedGazeTracker` (model off the stream thread) |
| `process.py` | `ProcessPredictor`: the model in a child process, since PyTorch's MPS backend is not thread-safe |
| `weights/behaviour_aware_lambda20_seed0.pth` / `.json` | Conv-LSTM with gaze and task heads (24,862 parameters), task-loss weight λ = 20, seed 0, and its training config |
| `weights/unisal.pth` | frozen UniSal saliency model |
| `unisal/` | UniSal network definition (Apache 2.0, see `unisal/LICENSE`) |

The λ = 20 model is the best of the trained configurations on the EHTask test set: 8.48° endpoint error at 333 ms and 11.99° at 666 ms, 68.6% task accuracy (mean over three seeds). Seed 0 has the lowest validation objective of the three. `tests/test_gaze.py` checks the output against reference values from the original code.
