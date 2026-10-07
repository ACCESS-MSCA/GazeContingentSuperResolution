"""
Stream a video over NDI, upscaling every frame 4x before it is sent.

One process: ffmpeg decodes a pre-downscaled video as is, VSRStream upscales
it, and the upscaled frames go out as an NDI source with the video's audio,
paced to its fps. --dual adds a second source with a moving overlay. The video
loops until Ctrl-C (or plays once with --once). If upscaling is slower than
real time the stream simply runs slow.

Usage:
    python main.py
    python main.py <path_to_video> [--k 3] [--batch 2] [--start 0]
                   [--frames N] [--once] [--device mps] [--name NAME] [--dual]
"""

import argparse
import itertools
import time
from pathlib import Path

import torch

from utils.ffmpeg import decode_audio_to_array, probe_video, read_rgb_frames
from utils.ndi import AUDIO_CHANNELS, AUDIO_SAMPLE_RATE, NDIOutput, rgb_to_bgra
from vsr import VSRPipeline

ROOT = Path(__file__).resolve().parent

# Pre-downscaled 8x (1920x1080 -> 240x136), so no scaling is done live and the
# 4x output (960x544) fits in Apple GPU (MPS) memory.
DEFAULT_VIDEO = ROOT / "Videos" / "big_buck_bunny_240x136.mp4"

# Fixed by the SR model: every output frame is 4x the input in each dimension.
SCALE = 4


def default_device() -> str:
    """Pick the fastest available backend: NVIDIA GPU, then Apple GPU, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    # --- Command-line options -------------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("video", nargs="?", default=str(DEFAULT_VIDEO))
    # k=1: diffusion SR on every frame. k>1: SR on every k-th frame (keyframes),
    # RIFE interpolates the frames in between (faster, adds up to k-1 frames delay).
    parser.add_argument("--k", type=int, default=3, help="keyframe stride")
    parser.add_argument("--batch", type=int, default=2, help="frames per push()")
    parser.add_argument("--start", type=float, default=0.0, help="start offset in seconds")
    parser.add_argument("--frames", type=int, default=None, help="stop after N frames")
    parser.add_argument("--once", action="store_true", help="play once instead of looping")
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: auto)")
    parser.add_argument("--name", default="StreamNDI-SR", help="NDI source name")
    parser.add_argument("--dual", action="store_true", help="add an overlay source")
    args = parser.parse_args()

    # --- Sizes and device -----------------------------------------------------
    # The video is used at its own resolution (LR); the NDI stream is 4x that (HR).
    lr_w, lr_h, fps, _ = probe_video(args.video)
    hr_w, hr_h = lr_w * SCALE, lr_h * SCALE
    device = args.device or default_device()

    print(f"Source  : {args.video}")
    print(f"Size    : LR {lr_w}x{lr_h} -> HR {hr_w}x{hr_h} @ {float(fps):.3f} fps")
    print(f"VSR     : k={args.k}, batch={args.batch}, device={device}")

    # --- Upscaler -------------------------------------------------------------
    # Loads the SR model (and RIFE when k > 1); weights download on first run.
    # The stream keeps keyframe spacing consistent across push() calls, so frames
    # can be fed in small batches as they are decoded.
    pipeline = VSRPipeline(k=args.k, device=device)
    stream = pipeline.stream()

    # --- NDI output -----------------------------------------------------------
    # Audio is decoded whole, then trimmed so it lines up with --start.
    audio_data = decode_audio_to_array(args.video, AUDIO_SAMPLE_RATE, AUDIO_CHANNELS)
    if audio_data is not None:
        audio_data = audio_data[:, int(args.start * AUDIO_SAMPLE_RATE):]

    # One NDI source at HR size (plus the "-Square" overlay source with --dual).
    # NDIOutput attaches one audio block per frame and paces sends to the fps.
    output = NDIOutput(args.name, hr_w, hr_h, fps, audio_data=audio_data, dual=args.dual)
    print("Press Ctrl-C to stop.\n")

    started = time.monotonic()
    sent = 0

    def send_all(hr_frames):
        """Send upscaled frames over NDI and print throughput every 30 frames."""
        nonlocal sent
        for hr in hr_frames:
            # The model outputs RGB; NDI is configured for BGRA.
            output.send(rgb_to_bgra(hr))
            sent += 1
            if sent % 30 == 0:
                elapsed = time.monotonic() - started
                print(f"{sent} frames sent ({sent / elapsed:.2f} fps)")

    # --- Main loop: decode -> upscale -> send ---------------------------------
    # Each pass plays the clip once; without --once it loops forever.
    passes = range(1) if args.once else itertools.count()
    remaining = args.frames

    with output:
        try:
            for _ in passes:
                batch = []
                read = 0
                for frame in read_rgb_frames(args.video, remaining, args.start):
                    read += 1
                    batch.append(frame)
                    if len(batch) == args.batch:
                        # push() returns only the frames that are ready: a frame
                        # between keyframes waits until the next keyframe is upscaled,
                        # so this can return nothing or several frames at once.
                        send_all(stream.push(batch))
                        batch = []
                if read == 0:
                    print("No frames decoded (is --start past the end of the video?).")
                    break
                if batch:
                    send_all(stream.push(batch))
                # End of the clip: drain the tail, then rewind audio for the next pass.
                send_all(stream.flush())
                output.restart()

                # --frames is a total across passes; stop once it is reached.
                if args.frames is not None:
                    remaining = args.frames - sent
                    if remaining <= 0:
                        break

            elapsed = time.monotonic() - started
            print(f"Done: {sent} frames sent in {elapsed:.1f}s ({sent / elapsed:.2f} fps)")

        except KeyboardInterrupt:
            print("\nStopped by user.")


if __name__ == "__main__":
    main()
