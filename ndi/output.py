import queue
import threading
import time
from fractions import Fraction

import numpy as np
from cyndilib.audio_frame import AudioSendFrame
from cyndilib.sender import Sender
from cyndilib.video_frame import VideoSendFrame
from cyndilib.wrapper.ndi_structs import FourCC


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
    """Paced NDI output: one or more sources sent together, with per-frame audio.

    `sources` maps a key to (NDI name, width, height). Use as a context manager and call
    send() once per frame step with a {key: BGRA frame} dict: every source gets its frame
    and the same audio block, one frame slot apart. Audio is sliced per frame from
    `audio_data`, so it stays in step with the video even when frames arrive slower than
    real time.

    The sends and the pacing run in a thread of their own, so the caller can upscale the
    next frames while earlier ones wait for their slot: upscalers that return frames in
    bursts (k frames per stretch) would otherwise sleep through k-1 slots with the GPU
    idle. send() only queues the frame; it blocks while `queue_frames` frames are already
    waiting, so the stream adds at most that many frames of delay.
    """

    def __init__(self, sources, fps, audio_data=None, queue_frames=1):
        fps_float = float(fps)
        self.frame_duration = 1.0 / fps_float

        samples_per_frame_exact = AUDIO_SAMPLE_RATE / max(fps_float, 1.0)
        self.audio_samples_per_frame = max(1, int(round(samples_per_frame_exact)))

        self.audio_enabled = audio_data is not None and audio_data.shape[1] > 0
        self.audio_data = audio_data
        self.total_audio_samples = int(audio_data.shape[1]) if self.audio_enabled else 0

        self.senders = {}
        for key, (name, width, height) in sources.items():
            sender, _ = make_sender(name, width, height, fps)
            if self.audio_enabled:
                self._attach_audio(sender)
            self.senders[key] = sender
            print(f"NDI name: '{name}' ({width}x{height})")
        if self.audio_enabled:
            print(
                f"Audio   : {AUDIO_SAMPLE_RATE} Hz, {AUDIO_CHANNELS} ch, "
                f"{self.audio_samples_per_frame} samples/frame"
            )
            if abs(samples_per_frame_exact - self.audio_samples_per_frame) > 1e-6:
                print(
                    f"[warn] fractional audio/frame ({samples_per_frame_exact:.6f}); "
                    "using fixed-size blocks for channel stability."
                )
        else:
            print("[warn] could not start audio decoder; sending video only.")

        self.dropped_timing_count = 0
        self.idle_s = 0.0  # time send() spent waiting for a free place in the queue (headroom)
        self._queue = queue.Queue(maxsize=max(1, queue_frames))
        self._thread = None
        self._error = None
        self._restart()

    def _attach_audio(self, sender):
        af = AudioSendFrame(max_num_samples=self.audio_samples_per_frame)
        af.sample_rate = AUDIO_SAMPLE_RATE
        af.num_channels = AUDIO_CHANNELS
        sender.set_audio_frame(af)

    def restart(self):
        """Rewinds audio and resets the pacing clock, e.g. after the source loops (in order
        with the frames already queued)."""
        self._put(None)

    def _restart(self):
        self.audio_pos = 0
        # Monotonic scheduled clock:
        # avoids accumulating drift better than "sleep(frame_duration - elapsed)"
        self.next_frame_time = time.monotonic()

    def __enter__(self):
        for sender in self.senders.values():
            sender.__enter__()
        self._restart()
        self._thread = threading.Thread(target=self._run, name="ndi-send", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._thread is not None:
            if exc[0] is not None:  # interrupted: drop the queued frames
                self._drain()
            if self._thread.is_alive():
                self._queue.put(StopIteration)  # sends what is still queued, then stops
            self._thread.join()
            self._thread = None
        for sender in reversed(list(self.senders.values())):
            sender.__exit__(*exc)
        if exc[0] is None:
            self._raise_error()

    def _drain(self):
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _put(self, item):
        if self._thread is None:  # not started: nothing is queued, apply it now
            if item is None:
                self._restart()
            else:
                self._send_now(item)
            return
        self._raise_error()
        start = time.monotonic()
        self._queue.put(item)
        self.idle_s += time.monotonic() - start

    def _raise_error(self):
        if self._error is not None:
            error, self._error = self._error, None
            raise RuntimeError("NDI sender thread failed") from error

    def _run(self):
        """Sender thread: sends the queued frames, one per frame slot, in order."""
        while True:
            item = self._queue.get()
            if item is StopIteration:
                return
            try:
                if item is None:
                    self._restart()
                else:
                    self._send_now(item)
            except Exception as error:  # noqa: BLE001 -- raised in the caller on its next send
                self._error = error
                self._drain()
                return

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

    def send(self, frames):
        """Queues {key: (H, W, 4) BGRA frame} for the matching sources (see the class docstring).
        `frames` can also be a function returning that dict, called in the sender thread, to
        take the conversion to BGRA off the caller's thread too."""
        self._put(frames)

    def _send_now(self, frames):
        """Sends {key: (H, W, 4) BGRA frame} on the matching sources, then sleeps until the next frame slot."""
        if callable(frames):
            frames = frames()
        if self.audio_enabled and self.total_audio_samples > 0:
            audio_frame = self._next_audio_block()
            for key, bgra in frames.items():
                self.senders[key].write_video_and_audio(bgra.ravel(), audio_frame)
        else:
            # Async send reduces chance the Python loop blocks on I/O.
            for key, bgra in frames.items():
                self.senders[key].write_video_async(bgra.ravel())

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
