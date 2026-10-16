# Gaze-Contingent Super-Resolution

Super-resolution that adapts to where the viewer is looking: full-quality upscaling at the gaze point, cheaper processing in the periphery.

## Status

A video is decoded once and sent over NDI as up to five sources, from the original to gaze-guided super-resolution: the video as is, bicubic 4x, bicubic with the gaze prediction drawn on it (`gaze/`), foveated (the SR model only around the predicted gaze point, bicubic elsewhere) and the SR model on the whole frame (`upscale/vsr/`). The sources are sent in lockstep, frame for frame, so they can be compared side by side.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and ffmpeg (`brew install uv ffmpeg`).

```bash
uv sync
```

This creates `.venv/` with Python 3.12 and the dependencies from `pyproject.toml` (locked in `uv.lock`). Model weights (several GB, including Stable Diffusion 2.1 base) download from Hugging Face on the first run into `~/.cache`.

## Usage

`main.py` decodes a video and sends each enabled source (`--sources`) as its own NDI source on the LAN, with audio, paced to the video's fps. It loops until Ctrl-C.

| Source | NDI name | Size | Content |
|---|---|---|---|
| `source` | `<name>-Source` | LR (240x136) | the decoded video as is |
| `vsr` | `<name>-VSR` | 4x | the SR model on the whole frame (+ RIFE when `--k` > 1) |
| `bicubic` | `<name>-Bicubic` | 4x | bicubic interpolation |
| `gaze` | `<name>-Gaze` | 4x | bicubic + the gaze prediction: all 20 predicted steps as dots (fainter = sooner) and a ring on the chosen step, as on the control page |
| `foveated` | `<name>-Foveated` | 4x | the SR model around the predicted gaze, bicubic elsewhere (`--fovea`; the region is outlined unless `--no-fovea-outline`) |

All sources come from the same decoded frames and are sent in **lockstep**: each frame goes out on every enabled source once all of them have it, so a run goes at the pace of its slowest source. `source` (the default), `bicubic` and `gaze` are real time; adding `foveated` (~3 fps) or `vsr` slows every source down to match, which keeps them frame-for-frame comparable. The gaze model only runs when `gaze` or `foveated` is enabled.

```bash
uv run python main.py                                           # Videos/big_buck_bunny_240x136.mp4 as StreamNDI-Source
uv run python main.py --start 10                                # skip the black fade-in at the start
uv run python main.py --start 10 --frames 60 --once             # short test run
uv run python main.py --start 10 --sources source,bicubic,gaze  # all three real-time sources
uv run python main.py --start 10 --sources bicubic,foveated     # compare bicubic with gaze-guided SR (slow motion)
uv run python main.py --start 10 --sources foveated --no-fovea-outline  # without the dashed circle
uv run python main.py --start 10 --sources bicubic,vsr --k 3    # compare with SR on the whole frame
uv run python main.py --ui                                      # control page instead (see below)
```

| Option | Default | Notes |
|---|---|---|
| `video` | `Videos/big_buck_bunny_240x136.mp4` | Used at its own resolution; output is 4x that. |
| `--sources` | `source` | NDI sources to send, comma-separated: `source`, `vsr`, `bicubic`, `gaze`, `foveated` (see above). |
| `--k` | `3` | vsr only. `1`: diffusion SR on every frame. `>1`: SR on every k-th frame, RIFE interpolates the rest (faster, up to k-1 frames delay). See `upscale/vsr/README.md`. |
| `--fovea` | `96` | foveated only. Side of the LR square around the gaze that the SR model upscales (×4 in the output). |
| `--fovea-outline` / `--no-fovea-outline` | on | foveated only. Draws the upscaled region (dashed circle) on the foveated source. |
| `--fovea-gaze` / `--no-fovea-gaze` | on | foveated only. Also draws the gaze prediction (steps and ring, as on the `gaze` source) on the foveated source. |
| `--batch` | `2` | Frames fed to the upscaler per call. |
| `--start` | `0` | Start offset in seconds. |
| `--frames` | all | Stop after N frames in total. |
| `--once` | off | Play once instead of looping. |
| `--device` | auto | `cuda`, `mps` or `cpu`. Auto picks CUDA, then MPS (Apple GPU), then CPU. |
| `--name` | `StreamNDI` | NDI name prefix: sources are `<name>-Source`, `<name>-Bicubic`, ... |
| `--gaze-trace` | head at the centre | Head pose CSV, columns `time,azimuth,elevation` (seconds from the video start, degrees). |
| `--gaze-horizon` | `20` | gaze source: which predicted step it draws the ring at, 1–20 at 30 Hz (1 = next frame, 20 = 666 ms ahead). |
| `--fovea-horizon` | `20` | foveated source: which predicted step the SR region (and its ring) follows, 1–20. |
| `--ui` | off | Opens the control page instead of streaming; the options above become its starting values. |
| `--ui-host` / `--ui-port` | `127.0.0.1` / `8000` | `--ui-host 0.0.0.0` makes the page reachable from other devices on the LAN. |

### Control page

`uv run python main.py --ui` opens `http://localhost:8000/` in the browser:

- **Pipeline** (right, under Performance): a diagram of how each NDI source is made from the decoded frames (decode → source / bicubic / foveated / VSR → NDI; head pose + 30 Hz samples → gaze model process, which places the foveated SR area; "+ gaze prediction" and "+ outline" labels where those are drawn on a source), with live values in each stage: the video's resolution and frame rate (read with ffprobe, also before streaming), ms per frame for each upscaler, gaze model time and lag, NDI names. The rates are chips, not subtitles: each processing box (the upscalers, the gaze model) has an fps chip in its top-right corner (the rate it could reach from its time per frame), and each NDI box its send rate next to its name, which is the run's actual pace. Decode has no chip: ffmpeg decodes ahead in its own process and is not the bottleneck. The Samples box has a grey **30 fps** chip: the gaze model's input rate (fixed by its training), which its own chip must keep up with, not the video's. It is green when the step keeps up with the video (30 Hz for the gaze model) and amber when it is the bottleneck. Parts no enabled source uses are greyed out.
- **Stream** (left panel, top): video (files in `Videos/`), the NDI sources to send (checkboxes; Foveated shows its fovea size, *outline* and *gaze prediction* nested under it, VSR its k, when ticked), start and NDI name; batch, frames, device and play once are under **More**. The buttons in the card's title row: **▶ Start** applies them, restarting the stream if it is running; **■ Stop** stops it. Changing an option while streaming does not interrupt the stream: ▶ is highlighted and a note lists what changed (e.g. *sources +Source, −Gaze · fovea 96 → 128*) until you press ▶ to restart with them, or undo the change. Models stay loaded between runs, so only what changed is reloaded.
- **NDI streams** (left panel, below Stream): every NDI source on the network, refreshed 4×/s. This run's own sources are highlighted and tagged with their source (source, vsr, bicubic, gaze, foveated). The **screen** button in its title row launches NDI Video Monitor on the computer running the stream (macOS, NDI Tools); it cannot preselect a source, so pick the stream in it. A source appears about 2 s after a stream starts and disappears when it stops.
- **Head (live)** (left panel, below NDI streams), applied on the next frame:
  - head pose source: **Live** (default) or **Trace** (a CSV in `Videos/`, e.g. `head_sweep_demo.csv`). A Live head nobody moves stays at the centre;
  - live head control: drag the pad (azimuth ±90°, elevation ±45°), or arrow keys (Shift for 10° steps). Either switches the head to Live. Switching the head source restarts the gaze model's 0.5 s window, so the jump is not read as head motion.
- **Horizon** (nested under *Bicubic + gaze prediction* and under *Foveated* in the Stream card): which of the 20 predicted steps that source uses, from 1 (next frame, 33 ms) to 20 (667 ms ahead). Each source has its own, and changing it applies on the next frame, without a restart. The model predicts all 20 steps every frame, so the sources just read different points of the same prediction.
- **Status** (header): state, frames sent, fps (polled 4×/s).
- **Performance** (top right), updated twice a second:
  - stream: send rate vs the video's fps, upscaling time per frame, frame budget used (share of each second not spent waiting for the next frame slot; 100% means it cannot keep up), late frames;
  - gaze model: predictions per second (input is 30 Hz), model time in its process, round trip, lag of the shown prediction behind the newest frame, queue and dropped samples;
  - threads and compute: CPU per thread and process as a share of one core (stream thread, gaze thread, whole stream process, model process), GPU (MPS) memory per process, peak RAM. GPU utilisation is not shown: macOS only exposes it to root (`sudo powermetrics`);
  - predicted task with its probability.

Each card's ⓘ icon (top-right corner) shows its explanation in a popover (Escape or a click elsewhere closes it).

The page talks to a small JSON API (`/api/state`, `/api/start`, `/api/stop`, `/api/live`, `/api/head`, see `control_page/server.py`). Live head poses are sent at up to 30 per second over one keep-alive connection.

The gaze model runs in a child process, driven by its own thread: each decoded frame carries the latest finished prediction (at most about one 30 Hz update behind), and if the model falls more than 1 s behind, samples are dropped and its window restarts (reported at the end of the run). A separate process rather than a thread because PyTorch's MPS backend is not thread-safe: running both on MPS in one process hung it.

Head pose is sampled at the video's 30 Hz ticks as frames are decoded. When upscaling runs slower than real time (the SR model at ~6 fps), video time runs slow too, so live head motion reaches the model as if it were faster than it was. Use bicubic for live head control until the pipeline runs in real time.

### Viewing the stream

Open an NDI receiver on any machine on the same network, e.g. **NDI Video Monitor** (free, part of [NDI Tools](https://ndi.video/tools/)) or OBS with the DistroAV plugin, and pick `<computer name> (StreamNDI-Source)` (or another source). If the source does not appear, check that the firewall allows incoming connections for Python.

### Input videos

Test videos go in `Videos/` (ignored by git). Videos are not scaled live, so downscale them beforehand, e.g. 8x:

```bash
ffmpeg -i Videos/big_buck_bunny.mp4 -vf "scale=iw/8:-2:flags=lanczos" -c:v libx264 -crf 14 -c:a copy Videos/big_buck_bunny_240x136.mp4
```

`-2` keeps the height even, which the encoder requires (1080 / 8 = 135 becomes 136).

## Upscaling model

`upscale/vsr/` upscales with an AdcSR-style compressed Stable Diffusion 2.1 UNet: a single forward pass per frame (no denoising loop), channels pruned to 75%, text conditioning and VAE encoder removed, half-size decoder. It is about 456M parameters, and RIFE handles frame interpolation when `k > 1`.

Performance on an Apple M4 (MPS):

- 240x136 input (960x544 output): about 6 fps, so the stream plays in slow motion against the 60 fps source. Bicubic keeps up with the 60 fps source, with or without the gaze source.
- 480x270 input (1920x1080 output): the SR model runs out of memory. The model's self-attention builds a very large buffer on MPS; NVIDIA GPUs avoid this with memory-efficient attention.

Real-time output needs an NVIDIA GPU or a lighter model.

### Gaze-guided upscaling (the `foveated` source)

`upscale/foveated.py`: every frame is upscaled with bicubic; the SR model then runs on a `--fovea` × `--fovea` LR square centred on the predicted gaze point at its own horizon (`--fovea-horizon`, or *Horizon* under Foveated on the page), and the result is blended in with a round, soft-edged mask. The foveated source shows that region as a dashed cyan circle by default (`--no-fovea-outline`, or untick *outline* on the page, for the clean result). With no prediction yet (the first 15 gaze samples), the fovea sits at the frame centre.

- **Colour match.** On a small crop the SR model shifts the colours (5–10 levels brighter in testing; on a whole frame it keeps them), which showed as a light disc. Only the SR's fine detail is used: the crop's low frequencies come from bicubic, so colours match exactly.
- **SR on every frame** (k = 1): RIFE cannot interpolate between crops taken at different positions, so frames come back without buffering and stay matched to their gaze points.
- **Speed** (Apple M4, MPS, 96 px fovea): about 3.1 fps, against about 0.4 fps for SR on the whole frame every frame (`--sources vsr --k 1`). The cost depends on the fovea size, not the input: 480x270 input (1920x1080 output), where whole-frame SR runs out of memory, also runs at about 3.1 fps.
- **Slow motion.** Below real time the video plays slowly and gaze is sampled in video time, so gaze updates come slowly too (about 1.6 per second at 3 fps on a 60 fps video; the first prediction after about 10 s). Recorded head traces stay consistent; live head motion reaches the model faster than it was.

## Gaze prediction

`gaze/` predicts where the viewer will look over the next 666 ms from the HMD viewport frames and head motion, without eye tracking (UniSal saliency + a Conv-LSTM, from the GazePrediction journal extension). `main.py` draws it on the `gaze` source and uses it to place the SR region of the `foveated` source; it runs only when one of those is enabled. See `gaze/README.md`.

The video frame is treated as the headset's field of view (108.8° × 111.5°), so a predicted offset from the head direction maps to a point relative to the frame centre. With a head that does not move (no trace, nothing sent from the page) the prediction barely moves: the model relies mostly on head motion, and a still head is outside its training data (360° video viewed in a headset).

```python
from gaze import predict_gaze
out = predict_gaze(frames, hmd)  # 15 viewport frames (RGB uint8) + (15, 2) HMD azimuth/elevation in degrees, 30 Hz
```

## Layout

```
main.py             command line: options -> run_stream, or the control page (--ui)
stream/             the stream loop and its helpers
  ffmpeg.py         video/audio decoding
  perf.py           per-second rates and memory for the Performance card
  runner.py         the stream: decode -> gaze -> upscale -> NDI (run_stream)
  control.py        stream options, live settings (gaze step, head source) and status
control_page/       the control page (main.py --ui)
  server.py         stdlib http.server and JSON API
  page.html         the page
  controller.py     runs the stream in a background thread for the page
ndi/
  output.py         NDI output: one sender per source, shared audio and pacing
  discovery.py      NDI sources on the network (control page's NDI streams panel)
  monitor.py        opens NDI Video Monitor (control page)
  overlay.py        gaze prediction: steps + ring (-Gaze, -Foveated); fovea outline (-Foveated)
upscale/            4x upscalers with the same push()/flush() interface
  vsr/              video super-resolution (SR model + RIFE), see upscale/vsr/README.md
  bicubic.py        bicubic upscaler, the fast baseline
  foveated.py       gaze-guided: SR on a square around the gaze, bicubic elsewhere (foveated source)
gaze/               gaze prediction from viewport + head motion, see gaze/README.md
  tracker.py        head pose (trace or live) + 30 Hz model updates per decoded frame, threaded
  process.py        runs the model in a child process (PyTorch's MPS backend is not thread-safe)
tests/              vsr, gaze, control and NDI tests: uv run pytest (no GPU needed)
Videos/             test videos (not in git)
```
