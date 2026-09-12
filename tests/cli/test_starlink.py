"""Tests for the brahe starlink CLI."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from brahe.cli.__main__ import app

runner = CliRunner()


def _entry(norad_cat_id, object_name):
    entry = MagicMock()
    entry.norad_cat_id = norad_cat_id
    entry.object_name = object_name
    entry.category = "Operational"
    entry.ephemeris_start = "2026-09-11 01:42:00.000 UTC"
    entry.ephemeris_stop = "2026-09-14 01:42:42.000 UTC"
    entry.file_name_string.return_value = f"MEME_{norad_cat_id}_{object_name}_2540142_Operational_1473385380_UNCLASSIFIED.txt"
    return entry


def _manifest(entries):
    manifest = MagicMock()
    manifest.entries.return_value = entries
    manifest.__len__.return_value = len(entries)
    manifest.find_by_norad_id.side_effect = lambda i: next(
        (e for e in entries if e.norad_cat_id == i), None
    )
    manifest.find_by_object_name.side_effect = lambda n: next(
        (e for e in entries if e.object_name == n), None
    )
    return manifest


def test_help():
    result = runner.invoke(app, ["starlink", "--help"])
    assert result.exit_code == 0
    assert "manifest" in result.stdout
    assert "download" in result.stdout


def test_manifest_help():
    result = runner.invoke(app, ["starlink", "manifest", "--help"])
    assert result.exit_code == 0
    assert "--norad-id" in result.stdout
    assert "--name" in result.stdout
    assert "--limit" in result.stdout


@patch("brahe.cli.starlink.bh.starlink.StarlinkClient")
def test_manifest_lists_entries(mock_client_cls):
    client = MagicMock()
    mock_client_cls.return_value = client
    client.get_manifest.return_value = _manifest(
        [_entry(100001, "STARLINK-38128"), _entry(100002, "STARLINK-37711")]
    )
    result = runner.invoke(app, ["starlink", "manifest"])
    assert result.exit_code == 0
    assert "STARLINK-38128" in result.stdout
    assert "STARLINK-37711" in result.stdout
    assert "2 of 2" in result.stdout


@patch("brahe.cli.starlink.bh.starlink.StarlinkClient")
def test_manifest_limit_and_filters(mock_client_cls):
    client = MagicMock()
    mock_client_cls.return_value = client
    client.get_manifest.return_value = _manifest(
        [_entry(100001, "STARLINK-38128"), _entry(100002, "STARLINK-37711")]
    )
    result = runner.invoke(app, ["starlink", "manifest", "--limit", "1"])
    assert result.exit_code == 0
    assert "STARLINK-38128" in result.stdout
    assert "STARLINK-37711" not in result.stdout
    result = runner.invoke(app, ["starlink", "manifest", "--norad-id", "100002"])
    assert result.exit_code == 0
    assert "STARLINK-37711" in result.stdout
    assert "STARLINK-38128" not in result.stdout
    result = runner.invoke(app, ["starlink", "manifest", "--name", "STARLINK-38128"])
    assert result.exit_code == 0
    assert "100001" in result.stdout
    result = runner.invoke(app, ["starlink", "manifest", "--norad-id", "1"])
    assert result.exit_code == 1
    assert "not in the manifest" in result.stdout


@patch("brahe.cli.starlink.bh.starlink.StarlinkClient")
def test_manifest_error(mock_client_cls):
    client = MagicMock()
    mock_client_cls.return_value = client
    client.get_manifest.side_effect = RuntimeError("offline")
    result = runner.invoke(app, ["starlink", "manifest"])
    assert result.exit_code == 1
    assert "ERROR" in result.stdout


@patch("brahe.cli.starlink.bh.starlink.StarlinkClient")
def test_download_to_cache(mock_client_cls):
    client = MagicMock()
    mock_client_cls.return_value = client
    client.download_ephemeris.side_effect = lambda i: f"/cache/starlink/{i}.txt"
    result = runner.invoke(app, ["starlink", "download", "100001", "100002"])
    assert result.exit_code == 0
    assert client.download_ephemeris.call_count == 2
    assert "/cache/starlink/100001.txt" in result.stdout


@patch("brahe.cli.starlink.bh.starlink.StarlinkClient")
def test_download_with_output(mock_client_cls, tmp_path):
    client = MagicMock()
    mock_client_cls.return_value = client
    client.save_ephemeris.side_effect = lambda i, d: f"{d}/{i}.txt"
    result = runner.invoke(
        app, ["starlink", "download", "100001", "--output", str(tmp_path)]
    )
    assert result.exit_code == 0
    client.save_ephemeris.assert_called_once_with(100001, str(tmp_path))
    assert str(tmp_path) in result.stdout


@patch("brahe.cli.starlink.bh.starlink.StarlinkClient")
def test_download_error(mock_client_cls):
    client = MagicMock()
    mock_client_cls.return_value = client
    client.download_ephemeris.side_effect = RuntimeError(
        "No Starlink ephemeris for NORAD ID 1"
    )
    result = runner.invoke(app, ["starlink", "download", "1"])
    assert result.exit_code == 1
    assert "ERROR" in result.stdout
