import time
from fractions import Fraction

import numpy as np
from cyndilib.audio_frame import AudioSendFrame
from cyndilib.sender import Sender
from cyndilib.video_frame import VideoSendFrame
from cyndilib.wrapper.ndi_structs import FourCC

from utils.overlay import draw_square


def make_sender(source_name: str, width: int, height: int, fps: Fraction) -> tuple:
    """Create and configure a VideoSendFrame + Sender pair for one NDI output."""
    vf = VideoSendFrame()
    vf.set_resolution(width, height)
    vf.set_frame_rate(fps)  # must be a Fraction, not a float
    vf.set_fourcc(FourCC.BGRA)

    # NDIOutput paces frames manually, so disable internal NDI clocks
    # to avoid double rate-limiting/jitter.
    s = Sender(source_name, clock_video=False, clock_audio=False)
    s.set_video_frame(vf)
    return s, vf


def rgb_to_bgra(rgb):
    """Convert a uint8 (H, W, 3) RGB frame to a writable (H, W, 4) BGRA frame for NDI."""
    h, w = rgb.shape[:2]
    bgra = np.empty((h, w, 4), dtype=np.uint8)
    bgra[..., :3] = rgb[..., ::-1]
    bgra[..., 3] = 255
    return bgra


# NDI audio settings
AUDIO_SAMPLE_RATE = 48000
AUDIO_CHANNELS = 2


class NDIOutput:
    """Paced NDI output for one video: per-frame audio and an optional overlay source.

    Use as a context manager and call send() once per BGRA frame. Audio is sliced
    per frame from `audio_data`, so it stays in step with the video even when
    frames arrive slower than real time.
    """

    def __init__(self, source_name, width, height, fps, audio_data=None, dual=False):
        fps_float = float(fps)
        self.frame_duration = 1.0 / fps_float

        samples_per_frame_exact = AUDIO_SAMPLE_RATE / max(fps_float, 1.0)
        self.audio_samples_per_frame = max(1, int(round(samples_per_frame_exact)))

        # Create main sender
        self.sender_plain, _ = make_sender(source_name, width, height, fps)
        print(f"NDI name: '{source_name}'")
        print(
            f"Audio   : {AUDIO_SAMPLE_RATE} Hz, {AUDIO_CHANNELS} ch, "
            f"{self.audio_samples_per_frame} samples/frame"
        )

        if abs(samples_per_frame_exact - self.audio_samples_per_frame) > 1e-6:
            print(
                f"[warn] fractional audio/frame ({samples_per_frame_exact:.6f}); "
                "using fixed-size blocks for channel stability."
            )

        self.audio_enabled = audio_data is not None and audio_data.shape[1] > 0
        self.audio_data = audio_data
        self.total_audio_samples = int(audio_data.shape[1]) if self.audio_enabled else 0

        if self.audio_enabled:
            self._attach_audio(self.sender_plain)
        else:
            print("[warn] could not start audio decoder; sending video only.")

        # Optional second sender with overlay
        self.sender_overlay = None
        if dual:
            overlay_name = f"{source_name}-Square"
            self.sender_overlay, _ = make_sender(overlay_name, width, height, fps)
            print(f"NDI name: '{overlay_name}'")
            if self.audio_enabled:
                self._attach_audio(self.sender_overlay)

        self.frame_idx = 0
        self.dropped_timing_count = 0
        self.restart()

    def _attach_audio(self, sender):
        af = AudioSendFrame(max_num_samples=self.audio_samples_per_frame)
        af.sample_rate = AUDIO_SAMPLE_RATE
        af.num_channels = AUDIO_CHANNELS
        sender.set_audio_frame(af)

    def restart(self):
        """Rewinds audio and resets the pacing clock, e.g. after the source loops."""
        self.audio_pos = 0
        # Monotonic scheduled clock:
        # avoids accumulating drift better than "sleep(frame_duration - elapsed)"
        self.next_frame_time = time.monotonic()

    def __enter__(self):
        self.sender_plain.__enter__()
        if self.sender_overlay is not None:
            self.sender_overlay.__enter__()
        self.restart()
        return self

    def __exit__(self, *exc):
        if self.sender_overlay is not None:
            self.sender_overlay.__exit__(*exc)
        self.sender_plain.__exit__(*exc)

    def _next_audio_block(self):
        end_pos = self.audio_pos + self.audio_samples_per_frame
        if end_pos <= self.total_audio_samples:
            audio_frame = self.audio_data[:, self.audio_pos:end_pos].copy()
            self.audio_pos = end_pos
            if self.audio_pos >= self.total_audio_samples:
                self.audio_pos = 0
        else:
            first = self.audio_data[:, self.audio_pos:self.total_audio_samples]
            remain = end_pos - self.total_audio_samples
            second = self.audio_data[:, 0:remain]
            audio_frame = np.concatenate((first, second), axis=1).copy()
            self.audio_pos = remain
        return audio_frame

    def send(self, bgra: np.ndarray):
        """Sends one writable (H, W, 4) BGRA frame, then sleeps until the next frame slot."""
        plain_frame = bgra.ravel()

        overlay_frame = None
        if self.sender_overlay is not None:
            # Copy only for overlay path so the plain frame stays untouched.
            bgra_sq = np.array(bgra, copy=True)
            draw_square(bgra_sq, self.frame_idx)
            overlay_frame = bgra_sq.ravel()

        if self.audio_enabled and self.total_audio_samples > 0:
            audio_frame = self._next_audio_block()
            self.sender_plain.write_video_and_audio(plain_frame, audio_frame)
            if overlay_frame is not None:
                self.sender_overlay.write_video_and_audio(overlay_frame, audio_frame)
        else:
            # Async send reduces chance the Python loop blocks on I/O.
            self.sender_plain.write_video_async(plain_frame)
            if overlay_frame is not None:
                self.sender_overlay.write_video_async(overlay_frame)

        self.frame_idx += 1

        # Schedule next frame using an accumulated target time.
        self.next_frame_time += self.frame_duration
        now = time.monotonic()
        sleep_for = self.next_frame_time - now

        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            self.dropped_timing_count += 1
            if self.dropped_timing_count % 120 == 0:
                print(
                    f"[warn] timing late {self.dropped_timing_count} times "
                    f"(latest overrun: {-sleep_for*1000:.2f} ms)"
                )
            self.next_frame_time = now
