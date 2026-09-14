"""Tests for the brahe.clients namespace, the Python mirror of brahe::clients."""

import importlib

import brahe as bh


def test_clients_namespace_importable():
    clients = importlib.import_module("brahe.clients")
    assert clients is bh.clients


def test_clients_namespace_aliases_flat_modules():
    assert bh.clients.celestrak is bh.celestrak
    assert bh.clients.spacetrack is bh.spacetrack
    assert bh.clients.gcat is bh.datasets.gcat
    assert bh.clients.RateLimitConfig is bh.spacetrack.RateLimitConfig


def test_clients_namespace_exports():
    assert set(bh.clients.__all__) == {
        "RateLimitConfig",
        "celestrak",
        "gcat",
        "spacetrack",
    }
    assert "clients" in bh.__all__
