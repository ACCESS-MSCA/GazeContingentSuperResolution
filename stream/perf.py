"""Performance numbers for the control page: per-second rates of cumulative counters, and
process memory. GPU utilisation is not available on macOS without root (powermetrics);
MPS memory per process is shown instead."""

import resource
import sys


class Rates:
    """Turns cumulative counters into per-second rates over the interval since the last call."""

    def __init__(self):
        self._last = None

    def update(self, now, **counters):
        """Per-second rates of each counter since the previous call (None on the first call).
        Counters that are None are skipped."""
        prev, self._last = self._last, (now, counters)
        if prev is None or now <= prev[0]:
            return None
        dt = now - prev[0]
        return {k: (v - prev[1][k]) / dt for k, v in counters.items()
                if v is not None and prev[1].get(k) is not None}


def peak_rss_mb():
    """Peak resident memory of this process, MB (ru_maxrss is bytes on macOS, KB on Linux)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e6 if sys.platform == "darwin" else rss / 1e3


def mps_mb():
    """MPS memory allocated by this process, MB (None without MPS)."""
    import torch

    return torch.mps.current_allocated_memory() / 1e6 if torch.backends.mps.is_available() else None
