"""Opening NDI Video Monitor (NDI Tools) from the control page."""

import subprocess
import sys

MONITOR_APP = "NDI Video Monitor"


def open_monitor():
    """Opens NDI Video Monitor (NDI Tools). It has no option to preselect a source, so the
    stream is picked in the app."""
    if sys.platform != "darwin":
        raise ValueError(f"Opening {MONITOR_APP} is only supported on macOS")
    result = subprocess.run(["open", "-a", MONITOR_APP], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ValueError(f"Could not open {MONITOR_APP} (is NDI Tools installed?)")
