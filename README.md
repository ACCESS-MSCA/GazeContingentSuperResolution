# Gaze-Contingent Super-Resolution

Super-resolution that adapts to where the viewer is looking: full-quality upscaling at the gaze point, cheaper processing in the periphery.

## Status

Early setup: a video is decoded, upscaled 4x (the `vsr/` SR model, or bicubic as a baseline) and streamed over NDI in one process. No gaze handling yet.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and ffmpeg (`brew install uv ffmpeg`).

```bash
uv sync
```

This creates `.venv/` with Python 3.12 and the dependencies from `pyproject.toml` (locked in `uv.lock`). Model weights (several GB, including Stable Diffusion 2.1 base) download from Hugging Face on the first run into `~/.cache`.

## Usage

`main.py` decodes a video, upscales every frame 4x and streams the result as an NDI source on the LAN, with audio, paced to the video's fps. It loops until Ctrl-C.

```bash
uv run python main.py                                           # Videos/big_buck_bunny_240x136.mp4 -> 960x544
uv run python main.py --start 10                                # skip the black fade-in at the start
uv run python main.py Videos/testfile_50fps_240x136.mp4 --dual  # extra "-Square" source with a moving overlay
uv run python main.py --start 10 --frames 60 --once             # short test run
uv run python main.py --start 10 --model bicubic                # bicubic baseline to compare against
```

| Option | Default | Notes |
|---|---|---|
| `video` | `Videos/big_buck_bunny_240x136.mp4` | Used at its own resolution; output is 4x that. |
| `--model` | `vsr` | `vsr`: the SR model (+ RIFE). `bicubic`: plain bicubic interpolation, a fast baseline for comparison (no weights, near real time). |
| `--k` | `3` | vsr only. `1`: diffusion SR on every frame. `>1`: SR on every k-th frame, RIFE interpolates the rest (faster, up to k-1 frames delay). See `vsr/README.md`. |
| `--batch` | `2` | Frames fed to the upscaler per call. |
| `--start` | `0` | Start offset in seconds. |
| `--frames` | all | Stop after N frames in total. |
| `--once` | off | Play once instead of looping. |
| `--device` | auto | `cuda`, `mps` or `cpu`. Auto picks CUDA, then MPS (Apple GPU), then CPU. |
| `--name` | `StreamNDI-SR` | NDI source name. |
| `--dual` | off | Adds a second source, `<name>-Square`, with a moving overlay. |

### Viewing the stream

Open an NDI receiver on any machine on the same network, e.g. **NDI Video Monitor** (free, part of [NDI Tools](https://ndi.video/tools/)) or OBS with the DistroAV plugin, and pick `<computer name> (StreamNDI-SR)`. If the source does not appear, check that the firewall allows incoming connections for Python.

### Input videos

Test videos go in `Videos/` (ignored by git). Videos are not scaled live, so downscale them beforehand, e.g. 8x:

```bash
ffmpeg -i Videos/big_buck_bunny.mp4 -vf "scale=iw/8:-2:flags=lanczos" -c:v libx264 -crf 14 -c:a copy Videos/big_buck_bunny_240x136.mp4
```

`-2` keeps the height even, which the encoder requires (1080 / 8 = 135 becomes 136).

## Upscaling model

`vsr/` upscales with an AdcSR-style compressed Stable Diffusion 2.1 UNet: a single forward pass per frame (no denoising loop), channels pruned to 75%, text conditioning and VAE encoder removed, half-size decoder. It is about 456M parameters, and RIFE handles frame interpolation when `k > 1`.

Performance on an Apple M4 (MPS):

- 240x136 input (960x544 output): about 6 fps, so the stream plays in slow motion against the 60 fps source. Bicubic (`--model bicubic`) runs at about 55 fps on the same input.
- 480x270 input (1920x1080 output): the SR model runs out of memory. The model's self-attention builds a very large buffer on MPS; NVIDIA GPUs avoid this with memory-efficient attention.

Real-time output needs an NVIDIA GPU or a lighter model.

## Layout

```
main.py             video -> 4x upscaling -> NDI stream (audio, --dual)
utils/
  ffmpeg.py         video/audio decoding
  ndi.py            NDI output: senders, per-frame audio, pacing, overlay source
  overlay.py        moving square for the --dual source
  bicubic.py        bicubic upscaler (--model bicubic)
vsr/                4x video super-resolution (SR model + RIFE), see vsr/README.md
tests/              vsr tests: uv run pytest (no GPU needed)
Videos/             test videos (not in git)
```
