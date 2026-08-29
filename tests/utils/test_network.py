"""Tests for the BRAHE_NETWORK_MODE binding."""

import pytest

import brahe as bh


def test_network_mode_default_is_online(monkeypatch):
    monkeypatch.delenv("BRAHE_NETWORK_MODE", raising=False)
    assert bh.network_mode() == "online"
    assert bh.utils.network_mode() == "online"


@pytest.mark.parametrize(
    "value, expected",
    [
        ("online", "online"),
        ("OFFLINE", "offline"),
        (" offline-strict ", "offline-strict"),
        ("", "online"),
    ],
)
def test_network_mode_parses(monkeypatch, value, expected):
    monkeypatch.setenv("BRAHE_NETWORK_MODE", value)
    assert bh.network_mode() == expected


def test_network_mode_rejects_unknown(monkeypatch):
    monkeypatch.setenv("BRAHE_NETWORK_MODE", "maybe")
    with pytest.raises(RuntimeError, match="unrecognized value"):
        bh.network_mode()


def test_celestrak_offline_miss_raises_without_request(monkeypatch, tmp_path):
    monkeypatch.setenv("BRAHE_CACHE", str(tmp_path))
    monkeypatch.setenv("BRAHE_NETWORK_MODE", "offline")
    client = bh.celestrak.CelestrakClient(
        base_url="https://brahe-network-mode-test.invalid"
    )
    with pytest.raises(
        bh.BraheError, match="BRAHE_NETWORK_MODE is offline; Celestrak request"
    ):
        client.get_gp(catnr=25544)
