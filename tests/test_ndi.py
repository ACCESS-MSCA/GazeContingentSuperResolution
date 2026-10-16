import pytest

from ndi.discovery import is_local_host, split_source_name


@pytest.mark.parametrize("name, host, stream", [
    ("HARRISS-MAC-MINI.LOCAL (StreamNDI-SR)", "HARRISS-MAC-MINI.LOCAL", "StreamNDI-SR"),
    ("STUDIO (Cam (left))", "STUDIO", "Cam (left)"),
    ("no-parens", "", "no-parens"),
])
def test_split_source_name(name, host, stream):
    assert split_source_name(name) == {"host": host, "stream": stream}


@pytest.mark.parametrize("ndi_host, hostname, local", [
    ("HARRISS-MAC-MINI.LOCAL", "Harriss-Mac-mini.local", True),
    ("HARRISS-MAC-MINI.LOCAL", "Harriss-Mac-mini", True),
    ("STUDIO-PC", "Harriss-Mac-mini.local", False),
])
def test_is_local_host(monkeypatch, ndi_host, hostname, local):
    monkeypatch.setattr("socket.gethostname", lambda: hostname)
    assert is_local_host(ndi_host) is local
