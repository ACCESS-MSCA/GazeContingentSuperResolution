"""Runs the stream in a background thread for the control page."""

import threading
import traceback

from stream.control import StreamStatus
from stream.runner import run_stream


class StreamController:
    """Runs run_stream() in a background thread for the control page: one run at a time,
    models cached between runs."""

    def __init__(self, live):
        self.live = live
        self.status = StreamStatus()
        self._cache = {}
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def start(self, opts):
        """Stops the current run, if any, and starts a new one with `opts`."""
        with self._lock:
            self._halt()
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._run, args=(opts, self._stop), daemon=True)
            self._thread.start()

    def stop(self):
        with self._lock:
            self._halt()

    def _halt(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None

    def _run(self, opts, stop):
        try:
            run_stream(opts, self.live, self.status, stop, self._cache)
        except Exception as exc:  # noqa: BLE001 -- reported on the page; the server keeps running
            traceback.print_exc()
            self.status.update(state="error", error=f"{type(exc).__name__}: {exc}")
