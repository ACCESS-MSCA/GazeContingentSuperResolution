"""Finding the NDI sources on the network (control page's NDI streams panel)."""

import socket
import threading


def is_local_host(ndi_host):
    """Whether an NDI source's host name is this computer (NDI upper-cases it, e.g. MAC-MINI.LOCAL)."""
    def bare(name):
        return name.lower().removesuffix(".local")
    return bare(ndi_host) == bare(socket.gethostname())


def split_source_name(name):
    """NDI full source name "HOST (STREAM)" -> {"host": HOST, "stream": STREAM}."""
    host, sep, stream = name.partition(" (")
    if not sep or not stream.endswith(")"):
        return {"host": "", "stream": name}
    return {"host": host, "stream": stream[:-1]}


class SourceFinder:
    """NDI sources currently on the network, this machine's streams included (control page).

    Sources appear about 2 s after a sender starts and disappear when it stops.
    """

    def __init__(self):
        from cyndilib.finder import (
            Finder,  # here, so the helpers above work without the NDI runtime
        )

        self._lock = threading.Lock()
        self._finder = Finder()
        self._finder.open()

    def sources(self):
        """[{"host": ..., "stream": ...}] for every source found, sorted by host and name."""
        with self._lock:
            self._finder.update_sources()
            # Names ("HOST (STREAM)") rather than Source objects: a Source that disappears
            # while being read raises "Invalid source".
            names = list(self._finder.get_source_names())
        return sorted((split_source_name(n) for n in names), key=lambda s: (s["host"], s["stream"]))

    def close(self):
        with self._lock:
            self._finder.close()
