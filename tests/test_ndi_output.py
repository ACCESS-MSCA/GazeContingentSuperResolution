"""NDIOutput's sender thread: frames go out in order, one per slot, off the caller's thread."""

import threading
from fractions import Fraction

import numpy as np
import pytest

from ndi import output as ndi_output


class FakeSender:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def write_video_async(self, data):
        self.log.append((int(data[0]), threading.current_thread().name))


@pytest.fixture
def make_output(monkeypatch):
    log = []
    monkeypatch.setattr(ndi_output, "make_sender", lambda *a: (FakeSender(log), None))

    def make(**kwargs):
        return ndi_output.NDIOutput({"a": ("A", 2, 2)}, Fraction(1000), **kwargs), log
    return make


def frame(i):
    return {"a": np.full((2, 2, 4), i, np.uint8)}


def test_frames_are_sent_in_order_from_the_sender_thread(make_output):
    out, log = make_output(queue_frames=3)
    with out:
        for i in range(10):
            out.send(frame(i) if i % 2 else (lambda i=i: frame(i)))  # dicts and callables
        out.restart()
        out.send(frame(10))
    assert [i for i, _ in log] == list(range(11))
    assert {name for _, name in log} == {"ndi-send"}


def test_a_failed_send_is_raised_in_the_caller(make_output):
    out, _ = make_output()

    def broken():
        raise ValueError("boom")

    with pytest.raises(RuntimeError, match="sender thread failed"):
        with out:
            out.send(broken)
            for i in range(5):
                out.send(frame(i))


def test_without_the_thread_send_is_immediate(make_output):
    out, log = make_output()
    out.send(frame(7))
    assert log == [(7, threading.current_thread().name)]
