"""Control page server (main.py --ui): pick options, start/stop, live gaze and head control.

A stdlib ThreadingHTTPServer serving one HTML page and a small JSON API:

    GET  /              the page (control_page/page.html)
    GET  /api/state     options, live settings, status, available videos and head traces,
                        NDI sources on the network
    POST /api/start     {options}: (re)starts the stream with these StreamOptions
    POST /api/stop      stops the stream
    POST /api/live      {head_mode?, head_trace?}: applied on the next frame
    POST /api/head      {azimuth, elevation}: live head pose (head_mode "live"), sent ~30x/s
    POST /api/monitor   opens NDI Video Monitor on the computer running the stream (macOS)

HTTP/1.1 keep-alive is on, so the page's 30 Hz head updates reuse one connection.
"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from gaze.spec import FUTURE, MODEL_HZ
from ndi.discovery import is_local_host
from ndi.monitor import open_monitor
from stream.control import VIDEOS, StreamOptions, list_files
from stream.ffmpeg import probe_video

PAGE = Path(__file__).with_name("page.html")
VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv")
_VIDEO_INFO = {}  # (name, mtime) -> [width, height, fps], from ffprobe


def video_info(name):
    """[width, height, fps] of a video in Videos/ (probed once, cached until the file changes), or None."""
    path = VIDEOS / name
    try:
        key = (name, path.stat().st_mtime)
    except OSError:
        return None
    if key not in _VIDEO_INFO:
        try:
            w, h, fps, _ = probe_video(str(path))
            _VIDEO_INFO[key] = [w, h, float(fps)]
        except (Exception, SystemExit):  # noqa: BLE001 -- probe_video exits on unreadable files; just show no size
            _VIDEO_INFO[key] = None
    return _VIDEO_INFO[key]


_HARDWARE = None


def hardware():
    """The device runs use by default and whether the Apple Neural Engine can be used (macOS with
    coremltools), for the page's CPU / GPU / NPU chips before a run starts. Computed once."""
    global _HARDWARE
    if _HARDWARE is None:
        from stream.runner import default_device
        from upscale.vsr.super_resolution import coreml

        _HARDWARE = {"device": default_device(), "neural_engine": coreml.available()}
    return _HARDWARE


def make_server(controller, options, host="127.0.0.1", port=8000, monitor=open_monitor, sources=list):
    """Builds the server. `controller` is a control_page.controller.StreamController (or anything with start/stop,
    live and status); `options` the StreamOptions the page starts with.
    `sources` returns the NDI sources on the network ([{"host", "stream"}], e.g. SourceFinder.sources)."""
    state = {"options": options}

    def ndi_sources():
        try:
            return [dict(src, local=is_local_host(src["host"])) for src in sources()]
        except Exception:  # noqa: BLE001 -- discovery must not take the rest of the status down
            return []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # keep the console for the stream's output
            pass

        def _send(self, code, body, content_type="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _json_body(self):
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise TypeError("Expected a JSON object")
            return data

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._send(200, {
                    "options": state["options"].to_dict(),
                    "live": controller.live.snapshot(),
                    "status": controller.status.snapshot(),
                    "videos": list_files(VIDEOS, *VIDEO_SUFFIXES),
                    "video_info": {v: video_info(v) for v in list_files(VIDEOS, *VIDEO_SUFFIXES)},
                    "traces": list_files(VIDEOS, ".csv"),
                    # the gaze model's fixed rate and number of predicted steps (gaze/spec.py)
                    "gaze_model": {"hz": MODEL_HZ, "steps": FUTURE},
                    "hardware": hardware(),
                    "ndi": ndi_sources(),
                })
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                body = self._json_body()
                if self.path == "/api/start":
                    opts = StreamOptions.from_dict(body.get("options", {}))
                    state["options"] = opts
                    controller.start(opts)
                elif self.path == "/api/stop":
                    controller.stop()
                elif self.path == "/api/live":
                    live = controller.live
                    if "head_mode" in body:
                        trace = body.get("head_trace")
                        if trace is not None:
                            if trace not in list_files(VIDEOS, ".csv"):
                                raise ValueError(f"Head trace not found in Videos/: {trace}")
                            trace = str(VIDEOS / trace)
                        live.set_head(body["head_mode"], trace)
                elif self.path == "/api/head":
                    controller.live.set_pose(float(body["azimuth"]), float(body["elevation"]))
                elif self.path == "/api/monitor":
                    monitor()
                else:
                    self._send(404, {"error": "not found"})
                    return
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self._send(400, {"error": str(exc)})
                return
            self._send(200, {"ok": True})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
