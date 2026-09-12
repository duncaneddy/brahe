"""Tests for scripts/seed_starlink_cache.py."""

import os
import subprocess
import sys
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
