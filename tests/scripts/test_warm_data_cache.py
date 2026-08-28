"""Tests for scripts/warm_data_cache.py's manifest handling and CLI selectors."""

import importlib.util
from pathlib import Path
from unittest import mock

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "warm_data_cache.py"
_spec = importlib.util.spec_from_file_location("warm_data_cache", _SCRIPT)
warm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(warm)


def test_entry_family_classifies_manifest_forms():
    assert warm.entry_family("de440s") == "kernel"
    assert warm.entry_family("icgem:moon:GRGM660PRIM") == "icgem"
    assert warm.entry_family("horizons:Ceres:2015-12-01:2016-03-01") == "horizons"
    assert warm.entry_family("celestrak:group:active") == "celestrak"


def test_entry_family_unknown_prefix_raises():
    with pytest.raises(ValueError, match="unknown prefix"):
        warm.entry_family("spacetrack:gp")


def test_parse_args_only_and_refresh():
    args = warm.parse_args(["--only", "celestrak", "--only", "kernel", "--refresh"])
    assert args.only == ["celestrak", "kernel"]
    assert args.refresh is True
    assert warm.parse_args([]).only == []


def test_parse_args_rejects_unknown_family():
    with pytest.raises(SystemExit):
        warm.parse_args(["--only", "textures"])


def test_warm_celestrak_uses_long_ttl_client_and_group_query():
    fake_client = mock.MagicMock()
    fake_client.get_gp.return_value = [object(), object()]
    with mock.patch.object(
        warm.bh.celestrak, "CelestrakClient", return_value=fake_client
    ) as ctor:
        result = warm._warm_celestrak("group:active")
    ctor.assert_called_once_with(cache_max_age=60 * 86400)
    fake_client.get_gp.assert_called_once_with(group="active")
    assert result == "2 records"


def test_warm_celestrak_rejects_non_group_spec():
    with pytest.raises(ValueError, match="celestrak:group:<name>"):
        warm._warm_celestrak("catnr:25544")


def test_main_only_filters_entries(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("de440s\ncelestrak:group:active\n")
    monkeypatch.setattr(warm, "MANIFEST_PATH", manifest)
    warmed = []
    monkeypatch.setattr(
        warm, "_warm_with_retries", lambda entry: warmed.append(entry) or "ok"
    )
    warm.main(["--only", "celestrak"])
    assert warmed == ["celestrak:group:active"]


def test_main_refresh_deletes_celestrak_cache(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("celestrak:group:active\n")
    monkeypatch.setattr(warm, "MANIFEST_PATH", manifest)
    cache = tmp_path / "celestrak"
    cache.mkdir()
    (cache / "stale").write_text("x")
    monkeypatch.setattr(warm.bh, "get_celestrak_cache_dir", lambda: str(cache))
    monkeypatch.setattr(warm, "_warm_with_retries", lambda entry: "ok")
    warm.main(["--only", "celestrak", "--refresh"])
    assert not (cache / "stale").exists()


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
