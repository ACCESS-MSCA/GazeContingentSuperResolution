import pytest

from stream.perf import Rates, peak_rss_mb


def test_rates_per_second_since_last_update():
    rates = Rates()
    assert rates.update(10.0, frames=0, busy_s=0.0) is None  # first call: nothing to compare
    assert rates.update(10.5, frames=30, busy_s=0.25) == pytest.approx({"frames": 60.0, "busy_s": 0.5})
    assert rates.update(11.5, frames=90, busy_s=0.25) == pytest.approx({"frames": 60.0, "busy_s": 0.0})


def test_rates_skip_missing_counters():
    rates = Rates()
    rates.update(0.0, a=0, b=None)
    assert rates.update(1.0, a=5, b=3) == {"a": 5.0}  # b had no previous value
    assert rates.update(1.0, a=9) is None  # no time passed


def test_peak_rss_is_plausible():
    assert 1 < peak_rss_mb() < 100_000
