"""
Stream a video over NDI as several sources, from the original to gaze-guided super-resolution.

One process: ffmpeg decodes a pre-downscaled video as is, and each enabled source
(--sources) is sent as its own NDI source with the video's audio, paced to its fps:
  source    the decoded video as is
  vsr       the SR model on the whole frame (+ RIFE when --k > 1)
  bicubic   bicubic 4x
  gaze      bicubic 4x with the gaze prediction drawn on it (gaze/ model on the LR frames)
  foveated  the SR model around the predicted gaze, bicubic elsewhere (--fovea, --fovea-outline)
The sources are sent in lockstep, frame for frame, so a run goes at the pace of its
slowest source: bicubic,gaze is real time; foveated and vsr play in slow motion. The
video loops until Ctrl-C (or plays once with --once).

--ui opens a control page instead: pick the options there, start and stop the stream,
and change the gaze step and head pose (drag or arrow keys) live.

Usage:
    python main.py
    python main.py <path_to_video> [--sources bicubic,gaze] [--k 3] [--fovea 96] [--no-fovea-outline] [--no-fovea-gaze]
                   [--batch 2] [--start 0] [--frames N] [--once] [--device mps] [--name NAME]
                   [--gaze-trace head.csv] [--gaze-horizon 20] [--fovea-horizon 20]
    python main.py --ui [--ui-host 0.0.0.0] [--ui-port 8000]
"""

import argparse
import webbrowser

from stream.control import (
    DEFAULT_SOURCES,
    DEFAULT_VIDEO,
    SOURCES,
    LiveSettings,
    StreamOptions,
    normalize_sources,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("video", nargs="?", default=str(DEFAULT_VIDEO))
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES),
                        help=f"comma-separated NDI sources to send, from: {', '.join(SOURCES)}")
    # (vsr only) k=1: diffusion SR on every frame. k>1: SR on every k-th frame (keyframes),
    # RIFE interpolates the frames in between (faster, adds up to k-1 frames delay).
    parser.add_argument("--k", type=int, default=3, help="keyframe stride")
    parser.add_argument("--fovea", type=int, default=96,
                        help="foveated only: side of the LR square around the gaze upscaled by the SR model")
    parser.add_argument("--fovea-outline", action=argparse.BooleanOptionalAction, default=True,
                        help="foveated only: outline the region the SR model upscaled (default: on)")
    parser.add_argument("--fovea-gaze", action=argparse.BooleanOptionalAction, default=True,
                        help="foveated only: draw the gaze prediction too (default: on)")
    parser.add_argument("--batch", type=int, default=2, help="frames per push()")
    parser.add_argument("--start", type=float, default=0.0, help="start offset in seconds")
    parser.add_argument("--frames", type=int, default=None, help="stop after N frames")
    parser.add_argument("--once", action="store_true", help="play once instead of looping")
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: auto)")
    parser.add_argument("--name", default="StreamNDI", help="NDI name prefix: sources are <name>-Source, -Bicubic, ...")
    parser.add_argument("--gaze-trace", default=None,
                        help="head pose CSV (time,azimuth,elevation; s, deg); default: head at the centre")
    parser.add_argument("--gaze-horizon", type=int, default=20,
                        help="gaze source: predicted step it draws, 1-20 at 30 Hz (20 = 666 ms ahead)")
    parser.add_argument("--fovea-horizon", type=int, default=20,
                        help="foveated source: predicted step where the SR region goes, 1-20 at 30 Hz")
    # Control page: the options above become the page's starting values.
    parser.add_argument("--ui", action="store_true", help="open the control page instead of streaming")
    parser.add_argument("--ui-host", default="127.0.0.1", help="0.0.0.0 to reach the page from the LAN")
    parser.add_argument("--ui-port", type=int, default=8000)
    args = parser.parse_args()
    try:
        args.sources = normalize_sources(args.sources)
    except ValueError as exc:
        parser.error(str(exc))
    for flag, step in (("--gaze-horizon", args.gaze_horizon), ("--fovea-horizon", args.fovea_horizon)):
        if not 1 <= step <= 20:
            parser.error(f"{flag} must be between 1 and 20, got {step}")
    return args


def main():
    args = parse_args()
    opts = StreamOptions(
        video=args.video, sources=args.sources, k=args.k, fovea=args.fovea, fovea_outline=args.fovea_outline, fovea_gaze=args.fovea_gaze,
        batch=args.batch, start=args.start, frames=args.frames, once=args.once, device=args.device, name=args.name,
    )
    live = LiveSettings(horizons={"gaze": args.gaze_horizon, "foveated": args.fovea_horizon},
                        trace_path=args.gaze_trace)

    # Imported here so that --help does not load torch.
    if not args.ui:
        from stream.runner import run_stream

        run_stream(opts, live)
        return

    from control_page.controller import StreamController
    from control_page.server import make_server
    from ndi.discovery import SourceFinder

    controller = StreamController(live)
    finder = SourceFinder()
    server = make_server(controller, opts, args.ui_host, args.ui_port, sources=finder.sources)
    host = "localhost" if args.ui_host in ("127.0.0.1", "0.0.0.0") else args.ui_host
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"Control page: {url}" + ("  (also on this machine's LAN address)" if args.ui_host == "0.0.0.0" else ""))
    print("Press Ctrl-C to quit.")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nQuitting.")
    finally:
        controller.stop()
        finder.close()
        server.server_close()


if __name__ == "__main__":
    main()
