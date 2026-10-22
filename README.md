# Gaze-Contingent Super-Resolution

Super-resolution that adapts to where the viewer is looking: full-quality upscaling at the gaze point, cheaper processing in the periphery.

## Status

A video is decoded once and sent over NDI as up to four sources, from the original to gaze-guided super-resolution: the video as is, bicubic 4x, the SR model on the whole frame (`upscale/vsr/`), and foveated: bicubic with the gaze prediction (`gaze/`) drawn on it and, unless SR is switched off, the SR model only around the predicted gaze point. The sources are sent in lockstep, frame for frame, so they can be compared side by side.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and ffmpeg (`brew install uv ffmpeg`).

```bash
uv sync
```

This creates `.venv/` with Python 3.12 and the dependencies from `pyproject.toml` (locked in `uv.lock`); on macOS that includes `coremltools`, for the Apple Neural Engine. Model weights (several GB, including Stable Diffusion 2.1 base) download from Hugging Face on the first run into `~/.cache`. The Neural Engine versions of the SR model (one per fovea size, ~0.9 GB each) and of UniSal are converted on first use and cached in `~/.cache/vsr/coreml/` and `~/.cache/gaze/`.

## Usage

`main.py` decodes a video and sends each enabled source (`--sources`) as its own NDI source on the LAN, with audio, paced to the video's fps. It plays the video once (from `--start`, up to `--frames` frames), or until Ctrl-C.

| Source | NDI name | Size | Content |
|---|---|---|---|
| `source` | `<name>-Source` | LR (240x136) | the decoded video as is |
| `vsr` | `<name>-VSR` (`-VSR-Cropped` with `--vsr-crop`) | 4x | the SR model on the whole frame (+ RIFE when `--k` > 1) |
| `bicubic` | `<name>-Bicubic` | 4x | bicubic interpolation |
| `foveated` | `<name>-Foveated` | 4x | bicubic + the gaze prediction (all 20 predicted steps as dots in shades of green, pale for the soonest, the last one a little stronger, and a small ring on the chosen step; `--no-fovea-gaze` hides it), and the SR model around the predicted gaze (`--fovea`, `--fovea-k`; the region is outlined unless `--no-fovea-outline`). `--no-fovea-sr`: no SR, bicubic + the gaze prediction only (real time) |

All sources come from the same decoded frames and are sent in **lockstep**: each frame goes out on every enabled source once all of them have it, so a run goes at the pace of its slowest source. `source` (the default), `bicubic` and `foveated` with `--no-fovea-sr` are real time, up to 4K output; `foveated` with SR at its defaults (64 px fovea, k = 5, SR on the Neural Engine overlapped with the GPU) is real time on 30 fps videos up to 4K and on 60 fps videos up to 1080p output, and about 50 fps at 4K (M4, see *Speed* below). Slower settings (a larger fovea, k = 1) or `vsr` slow every source down to match, which keeps them frame-for-frame comparable. Frames are sent, paced to the video's fps, from a thread of their own, so the next frames are upscaled while earlier ones wait for their slot. The gaze model only runs when `foveated` is enabled.

```bash
uv run python main.py                                           # Videos/big_buck_bunny_240x136.mp4 as StreamNDI-Source
uv run python main.py --start 10                                # skip the black fade-in at the start
uv run python main.py --start 10 --frames 60                    # short test run
uv run python main.py --start 10 --sources bicubic,foveated --no-fovea-sr  # real time: bicubic, and bicubic + gaze prediction
uv run python main.py --start 10 --sources bicubic,foveated     # compare bicubic with gaze-guided SR
uv run python main.py --start 10 --sources foveated --no-fovea-outline  # without the dashed circle
uv run python main.py --start 10 --sources bicubic,vsr --k 3    # compare with SR on the whole frame
uv run python main.py --start 10 --sources vsr --vsr-crop 64    # VSR speed on a 64x64 LR square (256x256 out)
uv run python main.py --start 10 --sources source,bicubic,vsr --vsr-crop 64  # all three on the same 64x64 square
uv run python main.py --ui                                      # control page instead (see below)
```

| Option | Default | Notes |
|---|---|---|
| `video` | `Videos/big_buck_bunny_240x136.mp4` | Used at its own resolution; output is 4x that. |
| `--sources` | `source` | NDI sources to send, comma-separated: `source`, `vsr`, `bicubic`, `foveated` (see above). |
| `--k` | `3` | vsr only. `1`: diffusion SR on every frame. `>1`: SR on every k-th frame, RIFE interpolates the rest (faster, up to k-1 frames delay). See `upscale/vsr/README.md`. |
| `--vsr-crop` | whole frame | The video pipeline (`source`, `bicubic`, `vsr`): side of a centred LR square those sources send, cut from each decoded frame (`source` as is, `bicubic` and `vsr` ×4, e.g. 64 → 256x256), so they stay comparable pixel for pixel and VSR's speed can be measured at other sizes without re-encoding the video. Cut down to fit the frame. They are then named `<name>-<label>-Cropped`. `foveated` always uses the whole frame. |
| `--fovea-sr` / `--no-fovea-sr` | on | foveated only. Runs the SR model around the gaze; off: bicubic + the gaze prediction only, in real time. |
| `--fovea-k` | `5` | foveated only. SR on every k-th frame's crop, RIFE in between (see *Gaze-guided upscaling* below). |
| `--fovea` | `64` | foveated only. Side of the LR square around the gaze that the SR model upscales (×4 in the output). |
| `--fovea-outline` / `--no-fovea-outline` | on | foveated only. Draws the upscaled region (dashed circle) on the foveated source. |
| `--fovea-gaze` / `--no-fovea-gaze` | on | foveated only. Also draws the gaze prediction (steps and ring) on the foveated source. |
| `--fovea-ane` / `--no-fovea-ane` | on | foveated only. Runs the SR model on the Apple Neural Engine (macOS on MPS, with coremltools) in its own process, off the GPU: a 64 px crop in ~41 ms instead of ~92 ms. Float16, so fine texture differs slightly from the GPU's. Each crop size is converted once (~20 s, cached in `~/.cache/vsr/coreml/`); loading takes ~8 s per run. Off, or without a Neural Engine: on the device. |
| `--fovea-overlap` / `--no-fovea-overlap` | on | foveated, Neural Engine, `--fovea-k` > 1. Upscales the next stretch on the Neural Engine while the GPU does RIFE and the blend for the current one: faster, but frames are held one stretch (k frames) longer, so the fovea uses the gaze predicted one stretch further ahead. |
| `--batch` | `2` | Frames fed to the upscaler per call. |
| `--start` | `0` | Start offset in seconds. |
| `--frames` | all | Stop after N frames in total. |
| `--device` | auto | `cuda`, `mps` or `cpu`. Auto picks CUDA, then MPS (Apple GPU), then CPU. |
| `--name` | `StreamNDI` | NDI name prefix: sources are `<name>-Source`, `<name>-Bicubic`, ... |
| `--gaze-trace` | head at the centre | Head pose CSV, columns `time,azimuth,elevation` (seconds from the video start, degrees). |
| `--ui` | off | Opens the control page instead of streaming; `video`, `--sources`, the fovea options, `--k`, `--vsr-crop` and `--name` become its starting values. The page always plays the whole video once from 0 s, with batch 2 and device auto. |
| `--ui-host` / `--ui-port` | `127.0.0.1` / `8000` | `--ui-host 0.0.0.0` makes the page reachable from other devices on the LAN. |

### Control page

`uv run python main.py --ui` opens `http://localhost:8000/` in the browser. The header holds the run's state, frames sent and fps, and on the right **NDI streams**; below it **Performance**, with **Head (live)** on its right, then **Pipelines** across the whole page. Everything about a stream is set in Pipelines.

- **Pipelines**: one diagram per area, **Video Super Resolution** (blue; Source, Bicubic, VSR use only the video) and **Gaze-Contingent Video Super Resolution** (amber; Foveated also uses the head pose, through the gaze model). Each shows how its NDI sources are made from the decoded frames, with live values in each box: the video's resolution and frame rate (read with ffprobe, also before streaming), ms per frame for each step, the gaze model's times and lag, NDI names. Each processing box has an **fps chip** (the rate it could reach from its time per frame: green when it keeps up with the video, or 30 Hz for the gaze model, amber when it is the bottleneck), each NDI box its send rate, which is the run's actual pace. The bottom row of every box shows **where it runs**: a CPU, GPU or NPU (Apple Neural Engine) chip, and the thread or process (*VSR Thread*, *SR Thread*, *Gaze Thread*, *NDI Thread*, *FFmpeg Process*, see *Threads and processes* below). While a source streams, the parts it does not use are greyed out.
  - **Keyframes**: both **SR ×4** boxes pick the frames SR runs on as *t and [t+k (k=…)]*: SR on a stretch's two keyframes, t and t+k, and **RIFE** makes the *frames between t and t+k*; at k = 1 just *t (k=1)*, every frame, and RIFE is off. With overlap, the foveated SR box shows *t+k and [t+2k]*: while RIFE fills one stretch, SR is already on the next.
  - **Gaze prediction** (dashed frame): Head pose, **UniSal** (each 30 Hz sample's saliency map, under Decode) and **LSTM** (20 gaze steps from the last 15 maps and head poses), each with its own time and fps chip. The LSTM places the SR crop; the foveated SR box shows which gaze span that covers, in frames, e.g. *gaze: t and t+5 future*, or *t+5 and t+10* with overlap.
- **Playing a source and its options**: each NDI box has a **▶** that streams that source alone (**■** stops it); a run plays the video once. The video is picked in either diagram's **Video** box (files in `Videos/`); *Crop* is in the Video Super Resolution diagram's Decode box (a centred square for Source, Bicubic and VSR; empty or × for the whole frame). In the gaze-contingent diagram: *SR area* in the SR ×4 box (`--fovea`), the **VSR** checkbox on the dashed frame's top-right corner (Foveated's SR on or off; off, the blend passes bicubic as is), **Neural Engine** and **overlap with GPU** under it (`--fovea-ane`, `--fovea-overlap`; greyed out where they cannot apply), and the overlays on their own arrows (*Overlay: outline*, *Overlay: full horizon gaze prediction*). Changing an option while streaming does not interrupt the stream: a note lists what changed until you stop and play again. Models stay loaded between runs.
- **Horizon** (no setting): the foveated source places its crop with the gaze predicted for when its frames are shown: the end of the stretch, ⌈k × 30 / video fps⌉ 30 Hz steps ahead, or 2k frames with overlap, which holds frames one stretch longer. StreamNDI-Foveated's box shows it (*horizon 10 · 333 ms* at k = 5, 30 fps, overlap).
- **NDI streams** (header): every NDI source on the network, refreshed 4×/s; this run's own are highlighted. The **screen** button launches NDI Video Monitor on the computer running the stream (macOS, NDI Tools; pick the stream in it). A source appears about 2 s after a stream starts.
- **Head (live)** (beside Performance, as tall), applied on the next frame: **Live** (default) or **Trace** (a CSV in `Videos/`, e.g. `head_sweep_demo.csv`) and the trace in its title row; drag the pad (azimuth ±90°, elevation ±45°) or use the arrow keys (Shift for 10° steps), which switches the head to Live. A Live head nobody moves stays at the centre. Switching the head source restarts the gaze model's 0.5 s window, so the jump is not read as head motion.
- **Performance**, updated twice a second: send rate vs the video's fps, upscaling time per frame, frame budget used (share of each second not spent waiting for room in the NDI send queue; 100% means it cannot keep up), late frames; the gaze model's predictions per second, model time, round trip, lag and dropped samples; CPU per thread and process (share of one core), GPU (MPS) memory per process, peak RAM (GPU utilisation needs root on macOS); the predicted task.

Each card's ⓘ icon shows its explanation in a popover (Escape or a click elsewhere closes it). Text in the diagrams stays the size of the rest of the page however wide the window is.

The page talks to a small JSON API (`/api/state`, `/api/start`, `/api/stop`, `/api/live` (head source), `/api/head`, see `control_page/server.py`). Live head poses are sent at up to 30 per second over one keep-alive connection.

Head pose is sampled at the video's 30 Hz ticks as frames are decoded. When upscaling runs slower than real time, video time runs slow too, so live head motion reaches the model as if it were faster than it was.

### Threads and processes

A run uses one Python process with up to four threads, and helper processes:

| Thread / process | What it does | Runs on |
|---|---|---|
| **VSR Thread** (the stream thread) | reads decoded frames, samples the head pose, does all the GPU work: bicubic straight to BGRA, SR on the GPU (VSR, or foveated without the Neural Engine), RIFE, the blend into the BGRA frame; hands frames to the NDI thread | GPU |
| **NDI Thread** | draws the overlays, converts Source and VSR frames to BGRA, sends them, and paces the sends to the video's fps | CPU |
| **Gaze Thread** | hands each 30 Hz sample to the gaze process and keeps the latest prediction; the stream thread never waits for it | CPU (waiting) |
| **SR Thread** | with overlap: has the Neural Engine upscale the next stretch's crop while the stream thread does RIFE and the blend | CPU (waiting) |
| FFmpeg Process | decodes the video into a pipe | CPU |
| gaze process | UniSal, then the LSTM, one after the other per sample | NPU (UniSal), GPU (LSTM) |
| SR process | the SR model on the fovea crop (`--fovea-ane`) | NPU |

Only the stream thread uses the GPU in the main process: PyTorch's MPS backend is not thread-safe (the gaze model on MPS in a second thread hung the process), so other GPU work has its own process. Core ML runs in processes too: its predict holds Python's GIL for the whole call (~40 ms), which in the stream's process would stop every thread. Sending runs in its own thread so the GPU keeps working while frames wait for their slot.

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

- 240x136 input (960x544 output): about 1.6 fps at k = 3 and 0.5 fps at k = 1, so the stream plays in slow motion against the 60 fps source. Bicubic keeps up with the 60 fps source, as does foveated with SR off.
- 480x270 input (1920x1080 output): the SR model runs out of memory (it asks for a 20 GB buffer). The model's self-attention builds a very large buffer on MPS; NVIDIA GPUs avoid this with memory-efficient attention.

On MPS the model runs in float16 (float32 is about 25% slower for the same output). On the whole frame the Neural Engine is only about 14% faster than the GPU (its large attention layers suit it badly), so `vsr` stays on the GPU; the Neural Engine is used for the foveated crop. Real-time whole-frame output needs an NVIDIA GPU or a lighter model.

### Gaze-guided upscaling (the `foveated` source)

`upscale/foveated.py`: every frame is upscaled with bicubic; the SR model then runs on a `--fovea` × `--fovea` LR square centred on the gaze predicted for when its frames are shown (see *Horizon* above), and the result is blended in with a round, soft-edged mask. The foveated source shows that region as a dashed cyan circle by default (`--no-fovea-outline`, or untick *outline* on the page, for the clean result). With no prediction yet (the first 15 gaze samples), the fovea sits at the frame centre.

- **Colour match.** On a small crop the SR model shifts the colours (5–10 levels brighter in testing; on a whole frame it keeps them), which showed as a light disc. Only the SR's fine detail is used: the crop's low frequencies come from bicubic, so colours match exactly.
- **`--fovea-k`** (default 5). At 1, the SR model runs on every frame's crop, at that frame's gaze point, with no buffering. At k > 1 it runs only on every k-th frame's crop (keyframes), and RIFE makes the crops in between. RIFE can only blend crops cut from the same place, so the crop stays still for each stretch of k frames. Frames are held until the stretch's closing keyframe is decoded (up to k−1 frames, as with vsr). The crop is centred on the mean of the stretch's gaze points, so it covers where the gaze is during the stretch. If the crop moves by more than a quarter of the fovea, the stretch's opening keyframe is upscaled again at the new place; otherwise its SR is reused. RIFE makes all the crops of a stretch in one batched call. The tradeoff is that the fovea moves in steps, once per stretch.
- **Neural Engine and overlap** (`--fovea-ane`, `--fovea-overlap`, on by default on a Mac). The SR model runs on the Neural Engine through Core ML, in its own process (`upscale/vsr/super_resolution/coreml.py`): a 64 px crop in ~41 ms instead of ~92 ms on the GPU. It computes in float16, so fine texture in busy areas differs slightly from the GPU's (equally sharp; 35–57 dB PSNR on the detail the blend keeps). With overlap, the Neural Engine upscales the next stretch's keyframe while the GPU does RIFE and the blend for the current one, which hides SR almost entirely; frames are then held one stretch longer, and the crop uses the gaze predicted one stretch further ahead.
- **Full frames stay on the GPU.** Bicubic goes straight to the BGRA frame NDI sends, in compiled GPU code, and only the fovea region is blended in float, from the LR crop upscaled again; each frame is copied off the GPU once.
- **Speed** (Apple M4; 64 px fovea, k = 5, Neural Engine + overlap; steady state):

  | Video | Output | Stream |
  |---|---|---|
  | 240x136, 30 fps | 960x544 | 30 fps (real time) |
  | 480x270, 30 fps | 1920x1080 | 30 fps (real time) |
  | 960x540, 30 fps | 3840x2160 | 30 fps (real time) |
  | 240x136, 60 fps | 960x544 | ~60 fps |
  | 480x270, 60 fps | 1920x1080 | 60 fps (real time) |
  | 960x540, 60 fps | 3840x2160 | ~50 fps |

  At 4K the stream thread spends per frame ~10 ms on RIFE, ~6 ms on bicubic + BGRA and ~3 ms on the blend and copy; SR waits ~0.2 ms. The cost of SR depends on the fovea size, not the input: a 96 px crop takes ~240 ms on the GPU (k = 1 runs at about 4 fps), against ~0.5 fps for SR on the whole 240x136 frame every frame (`--sources vsr --k 1`).
- **Slow motion.** Below real time (e.g. k = 1 with a large fovea) the video plays slowly and gaze is sampled in video time, so gaze updates come slowly too (about 2 per second at 4 fps on a 60 fps video; the first prediction after about 7.5 s). Recorded head traces stay consistent; live head motion reaches the model faster than it was.

## Gaze prediction

`gaze/` predicts where the viewer will look over the next 666 ms from the HMD viewport frames and head motion, without eye tracking (UniSal saliency + a Conv-LSTM, from the GazePrediction journal extension). `main.py` draws it on the `foveated` source and uses it to place that source's SR region; it runs only when `foveated` is enabled, in its own process. On a Mac, UniSal runs on the Neural Engine through Core ML (`gaze/coreml.py`; ~4 ms per frame, ~12 ms while the SR model shares the Neural Engine) and the LSTM on the GPU (~6 ms); elsewhere UniSal is compiled on the GPU. See `gaze/README.md`.

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
  output.py         NDI output: one sender per source, shared audio and pacing, in its own thread
  discovery.py      NDI sources on the network (control page's NDI streams panel)
  monitor.py        opens NDI Video Monitor (control page)
  overlay.py        gaze prediction: steps + ring; fovea outline (-Foveated)
upscale/            4x upscalers with the same push()/flush() interface
  vsr/              video super-resolution (SR model + RIFE), see upscale/vsr/README.md
  bicubic.py        bicubic upscaler, the fast baseline
  foveated.py       gaze-guided: SR on a square around the gaze, bicubic elsewhere (foveated source)
  vsr/super_resolution/coreml.py  the SR model on the Apple Neural Engine (Core ML, own process)
gaze/               gaze prediction from viewport + head motion, see gaze/README.md
  tracker.py        head pose (trace or live) + 30 Hz model updates per decoded frame, threaded
  process.py        runs the model in a child process (PyTorch's MPS backend is not thread-safe)
  coreml.py         UniSal on the Apple Neural Engine (Core ML)
tests/              vsr, gaze, control and NDI tests: uv run pytest (no GPU needed)
Videos/             test videos (not in git)
```
