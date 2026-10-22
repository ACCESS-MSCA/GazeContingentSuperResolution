import json
import subprocess
import sys
from fractions import Fraction

import numpy as np


DEFAULT_FPS = Fraction(30, 1)


def _parse_fps(value: str) -> Fraction:
    if not value or value in {"0", "0/0"}:
        return DEFAULT_FPS
    try:
        return Fraction(value).limit_denominator(1001)
    except Exception:
        return DEFAULT_FPS


def probe_video(video_path: str):
    """Return (width, height, fps_fraction, total_frames) using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames",
        "-of",
        "json",
        video_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        print("Error: ffprobe not found. Install ffmpeg/ffprobe first.")
        sys.exit(1)

    if result.returncode != 0:
        print(f"Error: ffprobe failed for '{video_path}'.")
        sys.exit(1)

    try:
        payload = json.loads(result.stdout or "{}")
        stream = (payload.get("streams") or [])[0]
    except Exception:
        stream = None

    if not stream:
        print(f"Error: no video stream found in '{video_path}'.")
        sys.exit(1)

    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        print(f"Error: invalid video size from '{video_path}'.")
        sys.exit(1)

    fps = _parse_fps(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))

    nb_frames_raw = stream.get("nb_frames")
    try:
        total_frames = int(nb_frames_raw) if nb_frames_raw not in (None, "N/A") else 0
    except ValueError:
        total_frames = 0

    return width, height, fps, total_frames


def read_exact(pipe, num_bytes: int):
    """Read exactly num_bytes from pipe, or None on EOF/failure."""
    chunks = []
    remaining = num_bytes
    while remaining > 0:
        chunk = pipe.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def decode_audio_to_array(video_path: str, sample_rate: int, num_channels: int):
    """Decode full media audio to float32 array shaped (channels, samples)."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-vn",
        "-map",
        "0:a:0?",
        "-ac",
        str(num_channels),
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        print("[warn] ffmpeg not found; audio disabled.")
        return None

    raw = result.stdout or b""
    if not raw:
        return None

    audio = np.frombuffer(raw, dtype=np.float32)
    if audio.size < num_channels:
        return None

    usable = (audio.size // num_channels) * num_channels
    if usable != audio.size:
        audio = audio[:usable]

    # ffmpeg raw PCM is interleaved, NDI expects (channels, samples).
    return audio.reshape(-1, num_channels).T.copy()

def read_rgb_frames(video_path: str, max_frames: int | None = None, start: float = 0.0):
    """Yield uint8 (H, W, 3) RGB frames at the video's own resolution.

    Starts `start` seconds in and stops after `max_frames` frames (None: to the end).
    """
    width, height, _, _ = probe_video(video_path)
    frame_bytes = width * height * 3

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(start),
        "-i",
        video_path,
        "-an",
    ]
    if max_frames is not None:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("Error: ffmpeg not found. Install ffmpeg first.")
        sys.exit(1)

    try:
        while True:
            raw = read_exact(proc.stdout, frame_bytes)
            if raw is None:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
    finally:
        proc.stdout.close()
        proc.wait()
