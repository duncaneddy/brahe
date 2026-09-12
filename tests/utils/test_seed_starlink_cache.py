"""Tests for scripts/seed_starlink_cache.py."""

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "seed_starlink_cache.py"
ASSETS = REPO / "test_assets" / "starlink"


def test_seed_starlink_cache_copies_fixtures(tmp_path):
    env = {**os.environ, "BRAHE_CACHE": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    target = tmp_path / "starlink"
    names = sorted(p.name for p in ASSETS.iterdir() if p.suffix == ".txt")
    assert sorted(p.name for p in target.iterdir()) == names
    assert (target / "MANIFEST.txt").read_text() == (
        ASSETS / "MANIFEST.txt"
    ).read_text()
    assert "Seeded" in result.stdout
    assert time.time() - (target / "MANIFEST.txt").stat().st_mtime < 300


def test_seed_starlink_cache_refuses_to_replace_a_live_manifest(tmp_path):
    target = tmp_path / "starlink"
    target.mkdir(parents=True)
    live = "MEME_45000_STARLINK-1_2540142_Operational_1473385380_UNCLASSIFIED.txt\n"
    (target / "MANIFEST.txt").write_text(live)
    env = {**os.environ, "BRAHE_CACHE": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "live Starlink" in result.stderr
    assert (target / "MANIFEST.txt").read_text() == live
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--force"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (target / "MANIFEST.txt").read_text() == (
        ASSETS / "MANIFEST.txt"
    ).read_text()
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_seed_starlink_cache_removes_stale_manifest_state(tmp_path):
    target = tmp_path / "starlink"
    target.mkdir(parents=True)
    (target / "MANIFEST.meta.json").write_text("{}")
    (target / "MANIFEST.previous.txt").write_text("old")
    env = {**os.environ, "BRAHE_CACHE": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not (target / "MANIFEST.meta.json").exists()
    assert not (target / "MANIFEST.previous.txt").exists()


def test_seed_starlink_cache_explicit_dir(tmp_path):
    target = tmp_path / "elsewhere"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--cache-dir", str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (target / "MANIFEST.txt").exists()
