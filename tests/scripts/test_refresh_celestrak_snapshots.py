"""Tests for scripts/refresh_celestrak_snapshots.py's install and pruning rules.

The network fetch itself is not exercised; what matters here is that a refresh
never destroys committed snapshots it did not just replace, since that is the
failure mode a maintenance command must not have.
"""

import importlib.util
import os
import shutil
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "refresh_celestrak_snapshots.py"
)
_spec = importlib.util.spec_from_file_location("refresh_celestrak_snapshots", _SCRIPT)
refresh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(refresh)


def _install(snapshot_dir: Path, fetched: list[Path], partial: bool) -> None:
    """Mirror the script's install block against a caller-supplied directory."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for src in fetched:
        staged = snapshot_dir / f".{src.name}.incoming"
        shutil.copy2(src, staged)
        os.replace(staged, snapshot_dir / src.name)
    if not partial:
        keep = {src.name for src in fetched}
        for stale in snapshot_dir.glob("*"):
            if stale.name not in keep:
                stale.unlink()


@pytest.fixture
def snapshots(tmp_path):
    """Seven committed snapshots, as the repository carries them."""
    d = tmp_path / "celestrak"
    d.mkdir()
    for name in ["active", "starlink", "gps_ops", "c1408", "f1c", "i33", "c2251"]:
        (d / f"snap_{name}").write_text("ORIGINAL")
    return d


def test_manifest_groups_reads_celestrak_entries():
    groups = refresh.manifest_groups()
    assert "active" in groups
    assert all(":" not in g for g in groups)


def test_partial_refresh_leaves_other_snapshots_untouched(snapshots, tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "snap_active").write_text("REFRESHED")

    _install(snapshots, [scratch / "snap_active"], partial=True)

    assert len(list(snapshots.glob("*"))) == 7
    assert (snapshots / "snap_active").read_text() == "REFRESHED"
    assert (snapshots / "snap_starlink").read_text() == "ORIGINAL"


def test_partial_refresh_leaves_no_staging_files(snapshots, tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "snap_active").write_text("REFRESHED")

    _install(snapshots, [scratch / "snap_active"], partial=True)

    assert not list(snapshots.glob(".*incoming"))


def test_full_refresh_prunes_groups_absent_from_the_manifest(snapshots, tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    for name in ["active", "starlink"]:
        (scratch / f"snap_{name}").write_text("REFRESHED")

    _install(snapshots, sorted(scratch.glob("*")), partial=False)

    assert sorted(p.name for p in snapshots.glob("*")) == [
        "snap_active",
        "snap_starlink",
    ]


def test_undersized_response_is_rejected(tmp_path):
    """A throttled Celestrak reply is a short HTML body, not usable data."""
    tiny = tmp_path / "snap_active"
    tiny.write_text("<html>rate limited</html>")
    assert tiny.stat().st_size < refresh.MIN_BYTES
