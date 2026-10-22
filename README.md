# Gaze-Contingent Super-Resolution

Super-resolution that adapts to where the viewer is looking: full-quality upscaling at the gaze point, cheaper processing in the periphery.

## Status

A video is decoded once and sent over NDI as up to four sources, from the original to gaze-guided super-resolution: the video as is, bicubic 4x, the SR model on the whole frame (`upscale/vsr/`), and foveated: bicubic with the gaze prediction (`gaze/`) drawn on it and, unless SR is switched off, the SR model only around the predicted gaze point. The sources are sent in lockstep, frame for frame, so they can be compared side by side.

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
| `vsr` | `<name>-VSR` (`-VSR-Cropped` with `--vsr-crop`) | 4x | the SR model on the whole frame (+ RIFE when `--k` > 1) |
| `bicubic` | `<name>-Bicubic` | 4x | bicubic interpolation |
| `foveated` | `<name>-Foveated` | 4x | bicubic + the gaze prediction (all 20 predicted steps as dots, fainter = sooner, and a ring on the chosen step; `--no-fovea-gaze` hides it), and the SR model around the predicted gaze (`--fovea`, `--fovea-k`; the region is outlined unless `--no-fovea-outline`). `--no-fovea-sr`: no SR, bicubic + the gaze prediction only (real time) |

All sources come from the same decoded frames and are sent in **lockstep**: each frame goes out on every enabled source once all of them have it, so a run goes at the pace of its slowest source. `source` (the default), `bicubic` and `foveated` with `--no-fovea-sr` are real time; `foveated` with SR (~4 fps at `--fovea-k 1`) or `vsr` slows every source down to match, which keeps them frame-for-frame comparable. Frames are sent, paced to the video's fps, from a thread of their own, so the next frames are upscaled while earlier ones wait for their slot. The gaze model only runs when `foveated` is enabled.

```bash
uv run python main.py                                           # Videos/big_buck_bunny_240x136.mp4 as StreamNDI-Source
uv run python main.py --start 10                                # skip the black fade-in at the start
uv run python main.py --start 10 --frames 60 --once             # short test run
uv run python main.py --start 10 --sources bicubic,foveated --no-fovea-sr  # real time: bicubic, and bicubic + gaze prediction
uv run python main.py --start 10 --sources bicubic,foveated     # compare bicubic with gaze-guided SR (slow motion)
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
| `--fovea-k` | `1` | foveated only. SR on every k-th frame's crop, RIFE in between (see *Gaze-guided upscaling* below). |
| `--fovea` | `96` | foveated only. Side of the LR square around the gaze that the SR model upscales (×4 in the output). |
| `--fovea-outline` / `--no-fovea-outline` | on | foveated only. Draws the upscaled region (dashed circle) on the foveated source. |
| `--fovea-gaze` / `--no-fovea-gaze` | on | foveated only. Also draws the gaze prediction (steps and ring) on the foveated source. |
| `--batch` | `2` | Frames fed to the upscaler per call. |
| `--start` | `0` | Start offset in seconds. |
| `--frames` | all | Stop after N frames in total. |
| `--once` | off | Play once instead of looping. |
| `--device` | auto | `cuda`, `mps` or `cpu`. Auto picks CUDA, then MPS (Apple GPU), then CPU. |
| `--name` | `StreamNDI` | NDI name prefix: sources are `<name>-Source`, `<name>-Bicubic`, ... |
| `--gaze-trace` | head at the centre | Head pose CSV, columns `time,azimuth,elevation` (seconds from the video start, degrees). |
| `--ui` | off | Opens the control page instead of streaming; `video`, `--sources`, the fovea options, `--k`, `--vsr-crop` and `--name` become its starting values. The page always plays the whole video from 0 s in a loop, with batch 2 and device auto. |
| `--ui-host` / `--ui-port` | `127.0.0.1` / `8000` | `--ui-host 0.0.0.0` makes the page reachable from other devices on the LAN. |

### Control page

`uv run python main.py --ui` opens `http://localhost:8000/` in the browser. NDI streams and Head (live) are on the left, Performance and Pipelines on the right. The pipelines come in two areas: **Video Super Resolution** (blue; Source, Bicubic, VSR use only the video) and **Gaze-Contingent Video Super Resolution** (amber; Foveated also uses the head pose through the gaze model). Everything about a stream is set in Pipelines.

- **Pipelines** (right, under Performance): one diagram per area, of how each NDI source is made from the decoded frames. Each starts with Video → Decode in its first row; the video is decoded once, and bicubic computed once, for both (decode → source / bicubic / foveated / VSR → NDI, with VSR in two stages: SR on keyframes, then RIFE for the frames in between, greyed out at k = 1; head pose + 30 Hz samples → gaze model (its own process), which places the foveated SR area; "+ gaze prediction" and "+ outline" labels where those are drawn on a source), with live values in each stage: the video's resolution and frame rate (read with ffprobe, also before streaming), ms per frame for each upscaler (for VSR, per stage: ms per keyframe for SR, per interpolated frame for RIFE; for Foveated, SR on the crop, RIFE for the crops in between (k > 1) and blending into bicubic, per frame sent), gaze model time and lag, NDI names. The rates are chips, not subtitles: each processing box (the upscalers, the gaze model) has an fps chip in its top-right corner (the rate it could reach from its time per frame), and each NDI box its send rate next to its name, which is the run's actual pace. Decode has no chip: ffmpeg decodes ahead in its own process and is not the bottleneck. The Gaze model box takes the last 15 LR frames sampled at 30 Hz (the *15 LR frames @ 30 Hz* arrow: every 2nd frame of a 60 fps video) and the head pose, sampled at 30 Hz (its input rate, fixed by its training), so its chip must keep up with 30 Hz, not the video's rate; the *gaze point* arrow shows it placing the foveated crop (labelled *current (t) gaze point* at k = 1, *current (t) + t+k future gaze point* at k > 1, where the crop follows the mean gaze of the stretch). A chip is green when the step keeps up with the video (30 Hz for the gaze model) and amber when it is the bottleneck. While a source streams, the parts it does not use are greyed out.
- **Playing a source and its options** (Pipelines): each NDI box has a **▶** button that streams that source alone; the other buttons are disabled while it runs. While it streams, the button is **■** and stops the stream. The video is picked in the **Video** box of either diagram (files in `Videos/`). In both diagrams, the **SR ×4** box has a dropdown for the frames SR runs on (*every frame*, *every 2nd frame*, …: k). The other options sit in the diagrams too: *Crop* in the Decode box of the Video Super Resolution diagram (it is cut from each decoded frame) (a centred square for Source, Bicubic and VSR; × or an empty field for the whole frame) in Video Super Resolution; Foveated's SR on/off as a checkbox beside the dashed *VSR* frame in its diagram, *SR area* in its SR ×4 box, right of the title (the side of the square crop around the gaze in px, `--fovea`), and the overlays as checkboxes on their own arrows in the diagram (*Overlay: outline* and *Overlay: full horizon gaze prediction*, all 20 predicted steps drawn; an unticked overlay's arrow is greyed out) in Gaze-Contingent Video Super Resolution (with SR off, Foveated Blend passes the bicubic frame as is and the SR stages are greyed out). Changing an option while streaming does not interrupt the stream: a note lists what changed (e.g. *fovea 96 → 128*) until you stop and play again. Models stay loaded between runs, so only what changed is reloaded.
- **NDI streams** (left panel, below Stream): every NDI source on the network, refreshed 4×/s. This run's own sources are highlighted and tagged with their source (source, vsr, bicubic, gaze, foveated). The **screen** button in its title row launches NDI Video Monitor on the computer running the stream (macOS, NDI Tools); it cannot preselect a source, so pick the stream in it. A source appears about 2 s after a stream starts and disappears when it stops.
- **Head (live)** (left panel, below NDI streams), applied on the next frame:
  - head pose source: **Live** (default) or **Trace** (a CSV in `Videos/`, e.g. `head_sweep_demo.csv`). A Live head nobody moves stays at the centre;
  - live head control: drag the pad (azimuth ±90°, elevation ±45°), or arrow keys (Shift for 10° steps). Either switches the head to Live. Switching the head source restarts the gaze model's 0.5 s window, so the jump is not read as head motion.
- **Horizon** (no setting): the foveated source uses the predicted gaze step at the end of its stretch of k frames, ⌈k × 30 / video fps⌉ (k counts video frames, steps are 30 Hz gaze samples): k = 1 → step 1 (33 ms ahead), k = 3 at 60 fps → step 2 (67 ms). That step places the SR crop and the ring of the gaze prediction overlay; StreamNDI-Foveated's box shows it (*horizon 2 · 67 ms*).
- **Status** (header): state, frames sent, fps (polled 4×/s).
- **Performance** (top right), updated twice a second:
  - stream: send rate vs the video's fps, upscaling time per frame, frame budget used (share of each second not spent waiting for room in the NDI send queue; 100% means it cannot keep up), late frames;
  - gaze model: predictions per second (input is 30 Hz), model time in its process, round trip, lag of the shown prediction behind the newest frame, queue and dropped samples;
  - threads and compute: CPU per thread and process as a share of one core (stream thread, gaze thread, whole stream process, model process), GPU (MPS) memory per process, peak RAM. GPU utilisation is not shown: macOS only exposes it to root (`sudo powermetrics`);
  - predicted task with its probability.

Each card's ⓘ icon (top-right corner) shows its explanation in a popover (Escape or a click elsewhere closes it).

The page talks to a small JSON API (`/api/state`, `/api/start`, `/api/stop`, `/api/live` (head source), `/api/head`, see `control_page/server.py`). Live head poses are sent at up to 30 per second over one keep-alive connection.

The gaze model runs in a child process, driven by its own thread: each decoded frame carries the latest finished prediction (at most about one 30 Hz update behind), and if the model falls more than 1 s behind, samples are dropped and its window restarts (reported at the end of the run). A separate process rather than a thread because PyTorch's MPS backend is not thread-safe: running both on MPS in one process hung it.

Head pose is sampled at the video's 30 Hz ticks as frames are decoded. When upscaling runs slower than real time (foveated SR at 4–30 fps), video time runs slow too, so live head motion reaches the model as if it were faster than it was. Use bicubic for live head control until the pipeline runs in real time.

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

On MPS the model runs in float16 (float32 is about 25% slower for the same output). Real-time output needs an NVIDIA GPU or a lighter model.

### Gaze-guided upscaling (the `foveated` source)

`upscale/foveated.py`: every frame is upscaled with bicubic; the SR model then runs on a `--fovea` × `--fovea` LR square centred on the predicted gaze point at the step where the gaze will be at the end of its stretch of k frames (see *Horizon* above), and the result is blended in with a round, soft-edged mask. The foveated source shows that region as a dashed cyan circle by default (`--no-fovea-outline`, or untick *outline* on the page, for the clean result). With no prediction yet (the first 15 gaze samples), the fovea sits at the frame centre.

- **Colour match.** On a small crop the SR model shifts the colours (5–10 levels brighter in testing; on a whole frame it keeps them), which showed as a light disc. Only the SR's fine detail is used: the crop's low frequencies come from bicubic, so colours match exactly.
- **`--fovea-k`** (default 1). At 1, the SR model runs on every frame's crop, at that frame's gaze point, with no buffering. At k > 1 it runs only on every k-th frame's crop (keyframes), and RIFE makes the crops in between. RIFE can only blend crops cut from the same place, so the crop stays still for each stretch of k frames. Frames are held until the stretch's closing keyframe is decoded (up to k−1 frames, as with vsr). The crop is centred on the mean of the stretch's gaze points, so it covers where the gaze is during the stretch. If the crop moves by more than a quarter of the fovea, the stretch's opening keyframe is upscaled again at the new place; otherwise its SR is reused. On an M4 (96 px fovea, head still): k = 1 runs at 4.0 fps (SR 237 ms per frame), k = 3 at 9.4 fps (SR 80 + RIFE 13 + blend 2 ms), k = 5 at 13.7 fps (SR 49 + RIFE 14 + blend 2 ms). The tradeoff is that the fovea moves in steps, once per stretch.
- **Speed** (Apple M4, MPS, 96 px fovea): about 4 fps, against about 0.5 fps for SR on the whole frame every frame (`--sources vsr --k 1`). The cost depends on the fovea size, not the input: 480x270 input (1920x1080 output), where whole-frame SR runs out of memory, also runs at about 3.9 fps. A smaller fovea is much cheaper: 64 px at k = 5 runs at about 29 fps (SR 95 ms per crop, 19 ms per frame).
- **Slow motion.** Below real time the video plays slowly and gaze is sampled in video time, so gaze updates come slowly too (about 2 per second at 4 fps on a 60 fps video; the first prediction after about 7.5 s). Recorded head traces stay consistent; live head motion reaches the model faster than it was.

## Gaze prediction

`gaze/` predicts where the viewer will look over the next 666 ms from the HMD viewport frames and head motion, without eye tracking (UniSal saliency + a Conv-LSTM, from the GazePrediction journal extension). `main.py` draws it on the `foveated` source and uses it to place that source's SR region; it runs only when `foveated` is enabled. See `gaze/README.md`.

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
  overlay.py        gaze prediction: steps + ring; fovea outline (-Foveated)
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
