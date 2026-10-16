"""Gaze prediction inference with the behaviour-aware model of the GazePrediction journal extension
(*Behaviour-Aware Gaze Prediction in Virtual Reality Without Eye Tracking*).

Uses the best fine-tuned model, `weights/behaviour_aware_lambda20_seed0.pth` (Conv-LSTM with gaze and task
heads, task-loss weight λ = 20; lowest validation objective of its three seeds and the seed used for the
paper's single-seed analyses), together with the frozen UniSal saliency model (`weights/unisal.pth`).

Only torch, numpy and Pillow are needed. The Conv-LSTM has the same layers and parameter names as
core.models.Conv_LSTM of the original repo and is loaded with strict=True; `unisal/` holds the UniSal
network only.

Example:
    from gaze import predict_gaze

    out = predict_gaze(frames, hmd)  # 15 viewport frames (RGB uint8) and (15, 2) HMD angles in degrees
    out["gaze"][out["horizon"] - 1]  # absolute gaze (azimuth, elevation) at the task-dependent horizon

Preprocessing mirrors revision/build_sal_cache.py and revision/cached_data.py of the original repo, the
pipeline used to train and evaluate the model.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from .unisal.model import UNISAL

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "weights")
MODEL_PATH = os.path.join(WEIGHTS, "behaviour_aware_lambda20_seed0.pth")  # config: same name, .json
SAL_PATH = os.path.join(WEIGHTS, "unisal.pth")

FPS = 30
PAST = 15     # input frames (500 ms at 30 Hz)
FUTURE = 20   # predicted frames (666 ms)
TASK_NAMES = ["Free Viewing", "Visual Search", "Saliency", "Track"]
# Prediction horizon (steps) per predicted task, selected on validation data with a 10° tolerance (seed 0).
TASK_HORIZONS = [13, 15, 10, 20]

UNISAL_SIZE = (384, 288)  # (width, height) of the UniSal input
RGB_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
RGB_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


class Conv_LSTM(nn.Module):
    """Inference copy of core.models.Conv_LSTM (original repo) with the task head (use_task_output=True)."""

    def __init__(self, num_time_features, conv_output_embeddings, sequence_length, prediction_length,
                 prediction_features, num_classes=4, image_kernel_size=5, image_padding=0,
                 lstm_hidden_size=128, lstm_num_layers=1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=image_kernel_size, padding=image_padding)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=image_kernel_size, padding=image_padding)
        self.bn2 = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(8 * 3 * 3, conv_output_embeddings)
        self.Dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(input_size=conv_output_embeddings + num_time_features, hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers, batch_first=True)
        self.fc_output = nn.Linear(lstm_hidden_size, prediction_features * prediction_length)
        self.future_steps = prediction_length
        self.prediction_features = prediction_features
        self.task_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, image, time_series):
        batch_size, num_images, _, _, H, W = image.shape
        x = F.interpolate(image.view(batch_size * num_images, 1, H, W), size=(25, 25))
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.Dropout(F.relu(self.fc1(x.view(batch_size * num_images, -1))))
        combined_features = torch.cat((time_series, x.view(batch_size, num_images, -1)), dim=2)
        lstm_out, (h_n, _) = self.lstm(combined_features)
        gaze_output = self.fc_output(h_n[-1])
        task_output = self.task_head(torch.mean(lstm_out, dim=1))
        return gaze_output.view(-1, self.future_steps, self.prediction_features), task_output


def load_unisal_model(sal_path, device):
    # The ImageNet backbone weights would be overwritten by unisal.pth, so they are not loaded.
    model = UNISAL(cnn_cfg={"pretrained": False})
    model.load_state_dict(torch.load(sal_path, map_location="cpu"))
    return model.to(device).eval()


def load_gaze_model(model_path, device):
    with open(os.path.splitext(model_path)[0] + ".json") as f:
        cfg = json.load(f)
    model = Conv_LSTM(num_time_features=cfg["num_time_features"], conv_output_embeddings=cfg["conv_output_embeddings"],
                      sequence_length=cfg["past_timepoints"], prediction_length=cfg["future_timepoints"],
                      prediction_features=cfg["prediction_features"], num_classes=cfg.get("n_classes", 4),
                      lstm_hidden_size=cfg["lstm_hidden_states"], lstm_num_layers=cfg["lstm_layers"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model.to(device).eval()


def motion_features(hmd, dt=1.0 / FPS):
    """(T, 2) HMD azimuth/elevation (deg) -> (T, 4) [az vel, el vel (deg/s), az diff, el diff (deg)].
    Same as revision/cached_data.motion_features of the original repo: wrapped first differences, first value duplicated."""
    d_az = np.diff(hmd[:, 0])
    d_az = np.where(d_az > 180, d_az - 360, d_az)
    d_az = np.where(d_az < -180, d_az + 360, d_az)
    d_el = np.diff(hmd[:, 1])
    d_az = np.concatenate([d_az[:1], d_az])
    d_el = np.concatenate([d_el[:1], d_el])
    return np.stack([d_az / dt, d_el / dt, d_az, d_el], axis=1).astype(np.float32)


def default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class GazePredictor:
    """Holds the UniSal and gaze models. `predict` takes a full 15-frame window; `push` feeds one frame at a
    time (saliency is computed once per frame) and returns a prediction once 15 frames are buffered."""

    def __init__(self, model_path=MODEL_PATH, sal_path=SAL_PATH, device=None):
        self.device = torch.device(device) if device is not None else default_device()
        self.unisal = load_unisal_model(sal_path, self.device)
        self.model = load_gaze_model(model_path, self.device)
        self._sal = []
        self._hmd = []

    @torch.no_grad()
    def saliency(self, frames):
        """Viewport frames (N, H, W, 3) RGB uint8, any size -> (N, 25, 25) saliency maps as the model consumes
        them (UniSal static mode, per-image standardisation, scaled by 2 and shifted by -1)."""
        imgs = np.stack([np.asarray(Image.fromarray(np.asarray(f, np.uint8)).resize(UNISAL_SIZE, Image.LANCZOS))
                         for f in frames])
        x = torch.as_tensor(imgs).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        x = (x - RGB_MEAN.to(self.device)) / RGB_STD.to(self.device)
        s = self.unisal(x.unsqueeze(1)).squeeze(1)  # (N, 1, H, W)
        mean = s.mean(dim=(1, 2, 3), keepdim=True)
        var = s.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
        s = (s - mean) / torch.sqrt(var + 1e-5) * 2 - 1
        return F.interpolate(s, size=(25, 25))[:, 0]

    @torch.no_grad()
    def predict_from_saliency(self, sal, hmd):
        """sal: (15, 25, 25) tensor; hmd: (15, 2) HMD azimuth/elevation in degrees at 30 Hz."""
        hmd = np.asarray(hmd, np.float32)
        ts = torch.from_numpy(motion_features(hmd)).to(self.device)
        gaze, logits = self.model(sal.to(self.device)[None, :, None, None], ts[None])
        offsets = gaze[0].cpu().numpy()
        probs = torch.softmax(logits, 1)[0].cpu().numpy()
        task = int(probs.argmax())
        # The model predicts offsets from the head direction at the first predicted frame, which is not yet
        # known at inference time; the latest HMD sample is used instead (one frame, 33 ms, earlier).
        az = (hmd[-1, 0] + offsets[:, 0] + 180) % 360 - 180
        el = np.clip(hmd[-1, 1] + offsets[:, 1], -90, 90)
        return {
            "offsets": offsets,                    # (20, 2) gaze minus head direction, degrees
            "gaze": np.stack([az, el], 1),         # (20, 2) absolute gaze, degrees
            "task_probs": probs,                   # (4,) in TASK_NAMES order
            "task": TASK_NAMES[task],
            "horizon": TASK_HORIZONS[task],        # steps ahead to use for this task (1-based)
        }

    def predict(self, frames, hmd):
        """frames: 15 viewport frames (H, W, 3) RGB uint8, oldest first; hmd: (15, 2) HMD azimuth/elevation
        in degrees for the same frames, sampled at 30 Hz."""
        if len(frames) != PAST or len(hmd) != PAST:
            raise ValueError(f"Expected {PAST} frames and HMD samples, got {len(frames)} and {len(hmd)}")
        return self.predict_from_saliency(self.saliency(frames), hmd)

    def push(self, frame, hmd_sample):
        """Adds one frame and its HMD (azimuth, elevation); returns a prediction, or None for the first 14."""
        self._sal.append(self.saliency([frame])[0])
        self._hmd.append(hmd_sample)
        if len(self._sal) > PAST:
            self._sal.pop(0)
            self._hmd.pop(0)
        if len(self._sal) < PAST:
            return None
        return self.predict_from_saliency(torch.stack(self._sal), self._hmd)

    def reset(self):
        self._sal, self._hmd = [], []


_PREDICTOR = None


def predict_gaze(frames, hmd, device=None):
    """Predicts gaze for the next 20 frames (666 ms) from the last 15 viewport frames and HMD angles.

    Args:
        frames: 15 viewport (HMD field-of-view) frames, oldest first, each (H, W, 3) RGB uint8.
        hmd: (15, 2) HMD azimuth and elevation in degrees for the same frames, sampled at 30 Hz.
        device: torch device; default CUDA, then MPS, then CPU. Models are loaded on the first call.

    Returns dict with "offsets" and "gaze" (20, 2) in degrees, "task_probs" (4,), "task" and "horizon".
    """
    global _PREDICTOR
    if _PREDICTOR is None or (device is not None and _PREDICTOR.device != torch.device(device)):
        _PREDICTOR = GazePredictor(device=device)
    return _PREDICTOR.predict(frames, hmd)
