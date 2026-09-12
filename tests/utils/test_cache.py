"""Tests for cache directory management utilities."""

from pathlib import Path

import brahe as bh


def test_get_celestrak_cache_dir(monkeypatch, tmp_path):
    """Rust: test_get_celestrak_cache_dir"""
    monkeypatch.setenv("BRAHE_CACHE", str(tmp_path))
    path = Path(bh.get_celestrak_cache_dir())
    assert path == tmp_path / "celestrak"
    assert path.is_dir()
    assert bh.utils.get_celestrak_cache_dir() == str(path)


def test_get_starlink_cache_dir(monkeypatch, tmp_path):
    """Rust: test_get_starlink_cache_dir"""
    monkeypatch.setenv("BRAHE_CACHE", str(tmp_path))
    path = Path(bh.get_starlink_cache_dir())
    assert path == tmp_path / "starlink"
    assert path.is_dir()
    assert bh.utils.get_starlink_cache_dir() == str(path)
