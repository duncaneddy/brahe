"""Tests for scripts/warm_data_cache.py's manifest handling and CLI selectors."""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "warm_data_cache.py"
_spec = importlib.util.spec_from_file_location("warm_data_cache", _SCRIPT)
warm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(warm)


def test_entry_family_classifies_manifest_forms():
    assert warm.entry_family("de440s") == "kernel"
    assert warm.entry_family("icgem:moon:GRGM660PRIM") == "icgem"
    assert warm.entry_family("horizons:Ceres:2015-12-01:2016-03-01") == "horizons"


def test_entry_family_skips_snapshot_backed_prefixes():
    """Celestrak groups are committed fixtures, so they are skipped, not warmed."""
    assert warm.entry_family("celestrak:group:active") is None


def test_entry_family_unknown_prefix_raises():
    with pytest.raises(ValueError, match="unknown prefix"):
        warm.entry_family("spacetrack:gp")


def test_parse_args_only():
    args = warm.parse_args(["--only", "icgem", "--only", "kernel"])
    assert args.only == ["icgem", "kernel"]
    assert warm.parse_args([]).only == []


def test_parse_args_rejects_snapshot_backed_family():
    """`celestrak` is no longer a warmable family."""
    with pytest.raises(SystemExit):
        warm.parse_args(["--only", "celestrak"])


def test_parse_args_rejects_unknown_family():
    with pytest.raises(SystemExit):
        warm.parse_args(["--only", "textures"])


def test_main_only_filters_entries(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("de440s\nicgem:moon:GRGM660PRIM\n")
    monkeypatch.setattr(warm, "MANIFEST_PATH", manifest)
    warmed = []
    monkeypatch.setattr(
        warm, "_warm_with_retries", lambda entry: warmed.append(entry) or "ok"
    )
    warm.main(["--only", "icgem"])
    assert warmed == ["icgem:moon:GRGM660PRIM"]


def test_main_skips_celestrak_entries_without_failing(monkeypatch, tmp_path):
    """Snapshot-backed entries are neither warmed nor treated as unknown prefixes."""
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("de440s\ncelestrak:group:active\n")
    monkeypatch.setattr(warm, "MANIFEST_PATH", manifest)
    warmed = []
    monkeypatch.setattr(
        warm, "_warm_with_retries", lambda entry: warmed.append(entry) or "ok"
    )
    warm.main([])
    assert warmed == ["de440s"]


def test_main_reports_failures_and_exits_nonzero(monkeypatch, tmp_path, capsys):
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("de440s\nicgem:moon:GRGM660PRIM\n")
    monkeypatch.setattr(warm, "MANIFEST_PATH", manifest)

    def fake(entry):
        if entry.startswith("icgem"):
            raise RuntimeError("mirror down")
        return "ok"

    monkeypatch.setattr(warm, "_warm_with_retries", fake)
    with pytest.raises(SystemExit) as exc:
        warm.main([])
    assert exc.value.code == 1
    assert "1 failed" in capsys.readouterr().err


def test_main_only_reports_unknown_prefix_as_failure(monkeypatch, tmp_path, capsys):
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("de440s\nspacetrack:gp\n")
    monkeypatch.setattr(warm, "MANIFEST_PATH", manifest)
    warmed = []
    monkeypatch.setattr(
        warm, "_warm_with_retries", lambda entry: warmed.append(entry) or "ok"
    )

    with pytest.raises(SystemExit) as exc:
        warm.main(["--only", "kernel"])

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "spacetrack:gp" in err
    assert "unknown prefix" in err
    assert warmed == ["de440s"]


def test_main_no_only_reports_unknown_prefix_as_failure(monkeypatch, tmp_path, capsys):
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("de440s\nspacetrack:gp\n")
    monkeypatch.setattr(warm, "MANIFEST_PATH", manifest)
    warmed = []
    monkeypatch.setattr(
        warm, "_warm_with_retries", lambda entry: warmed.append(entry) or "ok"
    )

    with pytest.raises(SystemExit) as exc:
        warm.main([])

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "spacetrack:gp" in err
    assert "unknown prefix" in err
    assert warmed == ["de440s"]
