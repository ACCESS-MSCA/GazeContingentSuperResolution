"""The stream itself: decode -> (gaze) -> upscale -> NDI, for one set of StreamOptions.

run_stream() is used by the command line (main.py) and, in a background thread, by the
control page (control_page.controller.StreamController).
"""

import itertools
import threading
import time
from collections import deque

import numpy as np
import torch

from stream.control import CROPPED_SOURCES, SOURCES, LiveSettings, StreamStatus, stretch_horizon
from stream.ffmpeg import decode_audio_to_array, probe_video, read_rgb_frames
from stream.perf import Rates, mps_mb, peak_rss_mb

# Fixed by the SR model: every output frame is 4x the input in each dimension.
SCALE = 4


def default_device() -> str:
    """Pick the fastest available backend: NVIDIA GPU, then Apple GPU, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _vsr_pipeline(cache, device, k):
    """The SR pipeline for `device`, one per device whatever k is (vsr and foveated share it):
    k only decides how often SR runs, so a new k loads RIFE once (k > 1) instead of another
    copy of the SR model."""
    from upscale.vsr import VSRPipeline

    pipeline = _cached(cache, ("vsr", device), lambda: VSRPipeline(k=1, device=device))
    if k > 1:
        pipeline.load_interpolation()
    return pipeline


def _cached(cache, key, factory):
    """Models are kept between runs started from the control page, so a restart with
    other options only reloads what changed."""
    if cache is None:
        return factory()
    if key not in cache:
        cache[key] = factory()
    return cache[key]


def run_stream(opts, live=None, status=None, stop=None, cache=None):
    """Streams one video with `opts` until it ends (--once / --frames), Ctrl-C, or `stop` is set.

    Every enabled NDI source (opts.sources) is made from the same decoded frames and they
    are sent in lockstep: a frame goes out on all sources once each has it, so the run goes
    at the pace of its slowest source (bicubic + gaze is real time; foveated and vsr are not).

    live: LiveSettings read every frame (head source); status: StreamStatus
    to report progress to; stop: threading.Event; cache: dict to keep models between runs.
    """
    from gaze.tracker import ThreadedGazeTracker
    from ndi.output import AUDIO_CHANNELS, AUDIO_SAMPLE_RATE, NDIOutput, rgb_to_bgra
    from ndi.overlay import draw_fovea, draw_gaze_prediction
    from upscale.bicubic import BicubicStream
    from upscale.foveated import FoveatedStream

    live = live or LiveSettings()
    status = status or StreamStatus()
    stop = stop or threading.Event()
    status.update(state="loading", error=None, sent=0, fps=0.0, gaze=None, gaze_info=None, head=None, perf=None,
                  options=opts.to_dict())
    enabled = set(opts.sources)

    # --- Sizes and device -----------------------------------------------------
    # The video is used at its own resolution (LR); upscaled sources are 4x that (HR).
    lr_w, lr_h, fps, _ = probe_video(opts.video)
    hr_w, hr_h = lr_w * SCALE, lr_h * SCALE
    device = opts.device or default_device()
    # (the video pipeline: source, bicubic, vsr) a centred LR square, cut down to fit the frame,
    # cut from each decoded frame; None = the whole frame. Foveated always uses the whole frame.
    crop = min(opts.vsr_crop, lr_w, lr_h) if opts.vsr_crop and enabled & set(CROPPED_SOURCES) else None
    crop_box = None if crop is None else ((lr_w - crop) // 2, (lr_h - crop) // 2, (lr_w + crop) // 2, (lr_h + crop) // 2)
    hr_box = None if crop is None else tuple(v * SCALE for v in crop_box)
    sizes = {k: (hr_w, hr_h) for k in SOURCES}
    sizes["source"] = (lr_w, lr_h)
    if crop:
        sizes.update(source=(crop, crop), bicubic=(crop * SCALE, crop * SCALE), vsr=(crop * SCALE, crop * SCALE))
    status.update(size=[hr_w, hr_h], lr_size=[lr_w, lr_h], sizes={k: list(v) for k, v in sizes.items()})

    print(f"Source  : {opts.video}")
    print(f"Size    : LR {lr_w}x{lr_h} -> HR {hr_w}x{hr_h} @ {float(fps):.3f} fps")
    print(f"Sources : {', '.join(opts.sources)} (batch={opts.batch}, device={device})")

    # --- Upscalers ------------------------------------------------------------
    # Each exposes push()/flush() and returns frames in order. Bicubic is computed once and
    # shared by the bicubic and foveated sources.
    upscalers = {}
    if enabled & {"bicubic", "foveated"}:
        upscalers["bicubic"] = BicubicStream(scale=SCALE, device=device)
    if "foveated" in enabled and not opts.fovea_sr:
        print("Foveated: SR off, bicubic" + (" + gaze prediction" if opts.fovea_gaze else ""))
    if "foveated" in enabled and opts.fovea_sr:
        # The SR model on a square around the gaze, on top of bicubic: on every frame (fovea_k=1),
        # or on every fovea_k-th frame with RIFE in between (the pipeline then loads RIFE too).
        sr = _vsr_pipeline(cache, device, opts.fovea_k)
        upscalers["foveated"] = FoveatedStream(sr, fovea=opts.fovea, device=device, k=opts.fovea_k)
        extras = [x for x, on in (("outline", opts.fovea_outline), ("gaze prediction", opts.fovea_gaze)) if on]
        print(f"Foveated: SR on {opts.fovea}x{opts.fovea} LR px at the gaze"
              + (f", every {opts.fovea_k}{'nd' if opts.fovea_k == 2 else 'rd' if opts.fovea_k == 3 else 'th'} frame + RIFE" if opts.fovea_k > 1 else "") + (f" + {', '.join(extras)}" if extras else ""))
    if "vsr" in enabled:
        # The SR model on the whole frame (and RIFE when k > 1); weights download on first
        # run. The stream keeps keyframe spacing across push() calls, so frames can be fed
        # in small batches, but it returns them up to k-1 frames late.
        pipeline = _vsr_pipeline(cache, device, opts.k)
        upscalers["vsr"] = pipeline.stream(k=opts.k)
        print(f"VSR     : k={opts.k}")
    if crop:
        print(f"Crop    : source, bicubic, vsr: centre {crop}x{crop} LR px ({crop * SCALE}x{crop * SCALE} upscaled)")

    # --- Gaze ----------------------------------------------------------------
    # The gaze model runs in its own process, driven from its own thread (ThreadedGazeTracker), so
    # the stream never waits for it. Each decoded frame takes the latest finished prediction
    # (all 20 predicted steps), queued until that frame is sent. The foveated source uses the step
    # at the end of its stretch of fovea_k frames (stretch_horizon); the head pose source is a live
    # setting, picked up every frame.
    tracker = None
    if opts.gaze:
        from gaze.process import ProcessPredictor

        # In its own process: PyTorch's MPS backend is not thread-safe (see gaze/process.py).
        predictor = _cached(cache, ("gaze", device), lambda: ProcessPredictor(device=device))
        head_source = live.head_source
        # The tracker's own step does not matter here: the source picks its step from the full path.
        tracker = ThreadedGazeTracker(fps, head_source, step=20, start=opts.start, predictor=predictor)
        fov_h = stretch_horizon(opts.fovea_k, fps)
        print(f"Gaze    : head {live.head_mode}, foveated horizon {fov_h} (k={opts.fovea_k})")

    # --- NDI output -----------------------------------------------------------
    # Audio is decoded whole, then trimmed so it lines up with --start.
    audio_data = decode_audio_to_array(opts.video, AUDIO_SAMPLE_RATE, AUDIO_CHANNELS)
    if audio_data is not None:
        audio_data = audio_data[:, int(opts.start * AUDIO_SAMPLE_RATE):]

    # One NDI source per enabled source; NDIOutput attaches the same audio block to each per
    # frame and paces the sends to the video's fps.
    names = opts.source_names()
    output = NDIOutput({k: (n, *sizes[k]) for k, n in names.items()}, fps, audio_data=audio_data)
    print("Press Ctrl-C to stop.\n")

    started = time.monotonic()
    sent = 0

    # --- Lockstep queues ----------------------------------------------------------
    # Per decoded frame: the LR frame and its gaze; per upscaler: its frames as they come
    # back (vsr up to k-1 frames late). A frame is sent once every queue it needs has it.
    pending = deque()  # (LR frame, gaze info, {source: gaze point at its horizon})
    ready = {k: deque() for k in upscalers}  # foveated entries are (frame, fovea box)

    # --- Performance view (control page), updated twice a second ---------------
    rates = Rates()
    upscaling = {k: {"s": 0.0, "frames": 0} for k in upscalers}
    next_perf = started

    def upscale(batch=None, gazes=None):
        """Pushes a batch through every upscaler (flushes without a batch), timed per upscaler."""
        bic = None
        for key, up in upscalers.items():
            start = time.perf_counter()
            if batch is None:
                out = up.flush()
            elif key == "foveated":
                # blends into copies of bicubic's frames instead of upscaling them again
                out = up.push(batch, gazes, base=[f.copy() for f in bic] if bic is not None else None)
            elif key == "vsr" and crop_box:
                x0, y0, x1, y1 = crop_box
                out = up.push([np.ascontiguousarray(f[y0:y1, x0:x1]) for f in batch])
            else:
                out = up.push(batch)
            upscaling[key]["s"] += time.perf_counter() - start
            upscaling[key]["frames"] += len(out)
            if key == "bicubic":
                bic = out
            ready[key].extend(zip(out, up.last_boxes) if key == "foveated" else out)
        send_ready()

    def report_perf(now):
        g = tracker.stats() if tracker else {}
        counters = {f"up_{k}_s": v["s"] for k, v in upscaling.items()}
        counters.update({f"up_{k}_n": v["frames"] for k, v in upscaling.items()})
        if "vsr" in upscalers:  # the VSR source's two stages: SR on keyframes, RIFE in between
            vsr = upscalers["vsr"]
            counters.update(vsr_sr_s=vsr.sr_s, vsr_sr_n=vsr.sr_frames, vsr_rife_s=vsr.rife_s, vsr_rife_n=vsr.rife_frames)
        if "foveated" in upscalers:  # the foveated source's two stages: SR on the crop, blend into bicubic
            fov = upscalers["foveated"]
            counters.update(fov_sr_s=fov.sr_s, fov_rife_s=fov.rife_s, fov_blend_s=fov.blend_s, fov_n=fov.frames)
        r = rates.update(
            now, sent=sent, idle_s=output.idle_s, stream_cpu_s=time.thread_time(), process_cpu_s=time.process_time(),
            predictions=g.get("predictions"), model_s=g.get("model_s"), roundtrip_s=g.get("roundtrip_s"),
            worker_cpu_s=g.get("worker_cpu_s"), child_cpu_s=g.get("child_cpu_s"), **counters,
        )
        if r is None:
            return

        def ms_per(total, count):  # e.g. seconds of model time per prediction, in ms
            return 1000 * r[total] / r[count] if r.get(count) else None

        lag = g.get("lag_samples")
        status.update(perf={
            "stream": {"fps": r["sent"], "video_fps": float(fps),
                       "upscale_ms": {k: ms_per(f"up_{k}_s", f"up_{k}_n") for k in upscalers},
                       # share of each second not spent waiting for the next frame slot
                       "busy": max(0.0, 1.0 - r["idle_s"]), "late": output.dropped_timing_count,
                       # per stage: ms per frame it makes, and ms per frame sent (its share of the VSR source's cost)
                       "vsr_stages": None if "vsr" not in upscalers else {
                           stage: {"ms": ms_per(f"vsr_{stage}_s", f"vsr_{stage}_n"), "ms_out": ms_per(f"vsr_{stage}_s", "up_vsr_n")}
                           for stage in ("sr", "rife")},
                       "foveated_stages": None if "foveated" not in upscalers else {
                           stage: ms_per(f"fov_{stage}_s", "fov_n") for stage in ("sr", "rife", "blend")}},
            "gaze": None if not tracker else {
                "rate": r.get("predictions"), "model_ms": ms_per("model_s", "predictions"),
                "roundtrip_ms": ms_per("roundtrip_s", "predictions"),
                "lag_ms": None if lag is None else lag * 1000 / 30, "queued": g["queued"], "dropped": g["dropped"]},
            # CPU: seconds of CPU per second (1.0 = one core); the process figure includes all its threads
            "compute": {"stream_thread": r["stream_cpu_s"], "gaze_thread": r.get("worker_cpu_s"),
                        "process": r["process_cpu_s"], "model_process": r.get("child_cpu_s"),
                        "mps_mb": mps_mb(), "model_mps_mb": g.get("child_mps_mb"), "rss_mb": peak_rss_mb()},
        })

    def send_ready():
        """Sends every frame that all its sources have, oldest first, and reports throughput."""
        nonlocal sent, next_perf
        while pending and all(ready[k] for k in upscalers):
            lr, info, points = pending.popleft()
            path = info["path"] if info else None
            ups = {k: q.popleft() for k, q in ready.items()}
            frames = {}
            if "source" in enabled:
                frames["source"] = rgb_to_bgra(lr if crop_box is None else lr[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]])
            if "bicubic" in enabled:
                bic = ups["bicubic"]  # the whole frame, shared with foveated: cut after upscaling
                frames["bicubic"] = rgb_to_bgra(bic if hr_box is None else bic[hr_box[1]:hr_box[3], hr_box[0]:hr_box[2]])
            if "foveated" in enabled:
                if opts.fovea_sr:
                    fov, box = ups["foveated"]
                    frames["foveated"] = rgb_to_bgra(fov)
                    if opts.fovea_outline:
                        draw_fovea(frames["foveated"], box)
                else:  # SR off: the bicubic frame, with the gaze prediction drawn on it
                    frames["foveated"] = rgb_to_bgra(ups["bicubic"])
                if opts.fovea_gaze:
                    draw_gaze_prediction(frames["foveated"], points.get("foveated"), path)
            if "vsr" in enabled:
                frames["vsr"] = rgb_to_bgra(ups["vsr"])
            output.send(frames)
            sent += 1
            elapsed = time.monotonic() - started
            status.update(sent=sent, fps=sent / elapsed, gaze=points.get("foveated"), gaze_info=info)
            now = time.monotonic()
            if now >= next_perf:
                report_perf(now)
                next_perf = now + 0.5
            if sent % 30 == 0:
                print(f"{sent} frames sent ({sent / elapsed:.2f} fps)")

    # --- Main loop: decode -> upscale -> send ---------------------------------
    # Each pass plays the clip once; without --once it loops forever.
    passes = range(1) if opts.once else itertools.count()
    remaining = opts.frames
    status.update(state="streaming")

    with output:
        try:
            for _ in passes:
                batch, batch_gazes = [], []
                read = 0
                if tracker:
                    tracker.reset()
                for frame in read_rgb_frames(opts.video, remaining, opts.start):
                    if stop.is_set():
                        break
                    read += 1
                    batch.append(frame)
                    info, points = None, {}
                    if tracker:
                        if live.head_source is not head_source:
                            head_source = live.head_source
                            tracker.set_source(head_source)
                        tracker.update(frame)
                        info = tracker.latest()[1]
                        if info:  # the foveated source's point: the step at the end of its stretch
                            points = {"foveated": info["path"][fov_h - 1]}
                        status.update(head=tracker.trace.pose_at(opts.start + tracker.frame_idx / float(fps)))
                    pending.append((frame, info, points))
                    batch_gazes.append(points.get("foveated"))
                    if len(batch) == opts.batch:
                        # vsr's push() returns only the frames that are ready: a frame
                        # between keyframes waits until the next keyframe is upscaled.
                        upscale(batch, batch_gazes)
                        batch, batch_gazes = [], []
                if stop.is_set():
                    break
                if read == 0:
                    print("No frames decoded (is --start past the end of the video?).")
                    break
                if batch:
                    upscale(batch, batch_gazes)
                # End of the clip: drain the tail, then rewind audio for the next pass.
                upscale()
                output.restart()

                # --frames is a total across passes; stop once it is reached.
                if opts.frames is not None:
                    remaining = opts.frames - sent
                    if remaining <= 0:
                        break

            elapsed = time.monotonic() - started
            print(f"Done: {sent} frames sent in {elapsed:.1f}s ({sent / max(elapsed, 1e-9):.2f} fps)")
            status.update(state="stopped" if stop.is_set() else "done")

        except KeyboardInterrupt:
            print("\nStopped by user.")
            status.update(state="stopped")
        finally:
            if tracker:
                tracker.close()
                if tracker.dropped:
                    print(f"[warn] gaze model fell behind: {tracker.dropped} samples dropped")
