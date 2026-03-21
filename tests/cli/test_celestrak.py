"""Tests for CelesTrak CLI commands (top-level brahe celestrak)"""

from unittest.mock import patch, MagicMock

import pytest
import brahe as bh
from typer.testing import CliRunner
from brahe.cli.celestrak import app, _format_to_celestrak

runner = CliRunner()


def test_help():
    """Test top-level celestrak help lists all subcommands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "gp" in result.stdout
    assert "sup-gp" in result.stdout
    assert "satcat" in result.stdout
    assert "groups" in result.stdout
    assert "download" in result.stdout


def test_gp_help():
    """Test gp command help shows all options."""
    result = runner.invoke(app, ["gp", "--help"])
    assert result.exit_code == 0
    assert "--group" in result.stdout
    assert "--name" in result.stdout
    assert "--catnr" in result.stdout
    assert "--intdes" in result.stdout
    assert "--filter" in result.stdout
    assert "--limit" in result.stdout
    assert "--order-by" in result.stdout
    assert "--descending" in result.stdout
    assert "--columns" in result.stdout
    assert "--output-format" in result.stdout
    assert "--output-file" in result.stdout


def test_gp_requires_selector():
    """Test gp command requires at least one selector."""
    result = runner.invoke(app, ["gp"])
    assert result.exit_code == 1
    assert (
        "at least one" in result.stdout.lower() or "required" in result.stdout.lower()
    )


def test_sup_gp_help():
    """Test sup-gp command help."""
    result = runner.invoke(app, ["sup-gp", "--help"])
    assert result.exit_code == 0
    assert "source" in result.stdout.lower()
    assert "--filter" in result.stdout
    assert "--limit" in result.stdout
    assert "--output-format" in result.stdout


def test_sup_gp_invalid_source():
    """Test sup-gp rejects invalid source names."""
    result = runner.invoke(app, ["sup-gp", "nonexistent_source"])
    assert result.exit_code == 1
    assert "unknown source" in result.stdout.lower() or "error" in result.stdout.lower()


def test_satcat_help():
    """Test satcat command help."""
    result = runner.invoke(app, ["satcat", "--help"])
    assert result.exit_code == 0
    assert "--catnr" in result.stdout
    assert "--active" in result.stdout
    assert "--payloads" in result.stdout
    assert "--on-orbit" in result.stdout
    assert "--output-format" in result.stdout


def test_satcat_requires_filter():
    """Test satcat command requires at least one filter option."""
    result = runner.invoke(app, ["satcat"])
    assert result.exit_code == 1


def test_groups():
    """Test groups command lists satellite groups (no network)."""
    result = runner.invoke(app, ["groups"])
    assert result.exit_code == 0
    assert "active" in result.stdout
    assert "stations" in result.stdout
    assert "starlink" in result.stdout
    assert "gnss" in result.stdout
    assert "gps-ops" in result.stdout


def test_download_help():
    """Test download command help."""
    result = runner.invoke(app, ["download", "--help"])
    assert result.exit_code == 0
    assert "filepath" in result.stdout.lower()
    assert "--group" in result.stdout
    assert "--content-format" in result.stdout
    assert "--file-format" in result.stdout


# Integration tests (require network)


@pytest.mark.ci
def test_gp_by_catnr():
    """Test querying GP by NORAD catalog number."""
    result = runner.invoke(app, ["gp", "--catnr", "25544", "--output-format", "json"])
    assert result.exit_code == 0
    assert "ISS" in result.stdout or "ZARYA" in result.stdout


@pytest.mark.ci
def test_gp_by_group_with_limit():
    """Test querying GP by group with limit."""
    result = runner.invoke(
        app, ["gp", "--group", "stations", "--limit", "3", "--output-format", "json"]
    )
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Unit tests with mocked client (no network required)
# ---------------------------------------------------------------------------


# --- Helper function tests ---


def test_format_to_celestrak_tle():
    """Test _format_to_celestrak returns TLE for 'tle'."""
    result = _format_to_celestrak("tle")
    assert result == bh.celestrak.CelestrakOutputFormat.TLE


def test_format_to_celestrak_3le():
    """Test _format_to_celestrak returns THREE_LE for '3le'."""
    result = _format_to_celestrak("3le")
    assert result == bh.celestrak.CelestrakOutputFormat.THREE_LE


def test_format_to_celestrak_invalid():
    """Test _format_to_celestrak raises ValueError for invalid format."""
    with pytest.raises(ValueError, match="Invalid content format"):
        _format_to_celestrak("xml")


# --- GP command execution tests (mocked client) ---


@patch("brahe.cli.celestrak.format_gp_records")
@patch("brahe.cli.celestrak.bh.celestrak.CelestrakClient")
def test_gp_by_group_mock(mock_client_cls, mock_format):
    """Test gp command with --group invokes client and format_gp_records."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.query.return_value = [MagicMock()]

    result = runner.invoke(app, ["gp", "--group", "stations"])
    assert result.exit_code == 0
    mock_client_cls.assert_called_once()
    mock_client.query.assert_called_once()
    mock_format.assert_called_once()


@patch("brahe.cli.celestrak.format_gp_records")
@patch("brahe.cli.celestrak.bh.celestrak.CelestrakClient")
def test_gp_by_name_mock(mock_client_cls, mock_format):
    """Test gp command with --name invokes client."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.query.return_value = [MagicMock()]

    result = runner.invoke(app, ["gp", "--name", "ISS"])
    assert result.exit_code == 0
    mock_client_cls.assert_called_once()
    mock_client.query.assert_called_once()
    mock_format.assert_called_once()


@patch("brahe.cli.celestrak.format_gp_records")
@patch("brahe.cli.celestrak.bh.celestrak.CelestrakClient")
def test_gp_by_catnr_with_options_mock(mock_client_cls, mock_format):
    """Test gp command with --catnr and additional options."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.query.return_value = [MagicMock()]

    result = runner.invoke(
        app,
        [
            "gp",
            "--catnr",
            "25544",
            "--output-format",
            "json",
            "--limit",
            "5",
            "--order-by",
            "EPOCH",
            "--desc",
        ],
    )
    assert result.exit_code == 0
    mock_client_cls.assert_called_once()
    mock_client.query.assert_called_once()
    mock_format.assert_called_once()


@patch("brahe.cli.celestrak.format_gp_records")
@patch("brahe.cli.celestrak.bh.celestrak.CelestrakClient")
def test_gp_by_intdes_mock(mock_client_cls, mock_format):
    """Test gp command with --intdes invokes client."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.query.return_value = [MagicMock()]

    result = runner.invoke(app, ["gp", "--intdes", "1998-067A"])
    assert result.exit_code == 0
    mock_client_cls.assert_called_once()
    mock_client.query.assert_called_once()
    mock_format.assert_called_once()


@patch("brahe.cli.celestrak.format_gp_records")
@patch("brahe.cli.celestrak.bh.celestrak.CelestrakClient")
def test_gp_query_error_mock(mock_client_cls, mock_format):
    """Test gp command handles query error gracefully."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.query.side_effect = RuntimeError("Connection failed")

    result = runner.invoke(app, ["gp", "--group", "stations"])
    assert result.exit_code == 1
    assert (
        "connection failed" in result.stdout.lower() or "error" in result.stdout.lower()
    )
    mock_format.assert_not_called()


# --- Sup-GP command execution tests (mocked client) ---


@patch("brahe.cli.celestrak.format_gp_records")
@patch("brahe.cli.celestrak.bh.celestrak.CelestrakClient")
def test_sup_gp_valid_source_mock(mock_client_cls, mock_format):
    """Test sup-gp command with valid source invokes client."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.query.return_value = [MagicMock()]

    result = runner.invoke(app, ["sup-gp", "starlink"])
    assert result.exit_code == 0
    mock_client_cls.assert_called_once()
    mock_client.query.assert_called_once()
    mock_format.assert_called_once()


@patch("brahe.cli.celestrak.format_gp_records")
@patch("brahe.cli.celestrak.bh.celestrak.CelestrakClient")
def test_sup_gp_with_options_mock(mock_client_cls, mock_format):
    """Test sup-gp command with additional options."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.query.return_value = [MagicMock()]

    result = runner.invoke(
        app,
        [
            "sup-gp",
            "planet",
            "--limit",
            "10",
            "--order-by",
            "EPOCH",
            "--output-format",
            "json",
        ],
    )
    assert result.exit_code == 0
    mock_client_cls.assert_called_once()
    mock_client.query.assert_called_once()
    mock_format.assert_called_once()


@patch("brahe.cli.celestrak.format_gp_records")
@patch("brahe.cli.celestrak.bh.celestrak.CelestrakClient")
def test_sup_gp_query_error_mock(mock_client_cls, mock_format):
    """Test sup-gp command handles query error gracefully."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.query.side_effect = RuntimeError("Connection failed")

    result = runner.invoke(app, ["sup-gp", "starlink"])
    assert result.exit_code == 1
    assert (
        "connection failed" in result.stdout.lower() or "error" in result.stdout.lower()
    )
    mock_format.assert_not_called()


# --- SATCAT command execution tests (mocked client) ---


@patch("brahe.cli.celestrak.format_satcat_records")
@patch("brahe.cli.celestrak.bh.celestrak.CelestrakClient")
def test_satcat_by_catnr_mock(mock_client_cls, mock_format):
    """Test satcat command with --catnr invokes client."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.query_satcat.return_value = [MagicMock()]

    result = runner.invoke(app, ["satcat", "--catnr", "25544"])
    assert result.exit_code == 0
    mock_client_cls.assert_called_once()
    mock_client.query_satcat.assert_called_once()
    mock_format.assert_called_once()


@patch("brahe.cli.celestrak.format_satcat_records")
@patch("brahe.cli.celestrak.bh.celestrak.CelestrakClient")
def test_satcat_active_payloads_mock(mock_client_cls, mock_format):
    """Test satcat command with --active and --payloads flags."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.query_satcat.return_value = [MagicMock()]

    result = runner.invoke(app, ["satcat", "--active", "--payloads"])
    assert result.exit_code == 0
    mock_client_cls.assert_called_once()
    mock_client.query_satcat.assert_called_once()
    mock_format.assert_called_once()


@patch("brahe.cli.celestrak.format_satcat_records")
@patch("brahe.cli.celestrak.bh.celestrak.CelestrakClient")
def test_satcat_on_orbit_mock(mock_client_cls, mock_format):
    """Test satcat command with --on-orbit flag."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.query_satcat.return_value = [MagicMock()]

    result = runner.invoke(app, ["satcat", "--on-orbit"])
    assert result.exit_code == 0
    mock_client_cls.assert_called_once()
    mock_client.query_satcat.assert_called_once()
    mock_format.assert_called_once()


@patch("brahe.cli.celestrak.format_satcat_records")
@patch("brahe.cli.celestrak.bh.celestrak.CelestrakClient")
def test_satcat_error_mock(mock_client_cls, mock_format):
    """Test satcat command handles query error gracefully."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.query_satcat.side_effect = RuntimeError("Connection failed")

    result = runner.invoke(app, ["satcat", "--catnr", "25544"])
    assert result.exit_code == 1
    assert (
        "connection failed" in result.stdout.lower() or "error" in result.stdout.lower()
    )
    mock_format.assert_not_called()


# --- Download command execution tests (mocked client) ---


@patch("brahe.cli.celestrak.bh.celestrak.CelestrakClient")
def test_download_default_mock(mock_client_cls, tmp_path):
    """Test download command with default options."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    filepath = tmp_path / "data.txt"
    result = runner.invoke(app, ["download", str(filepath)])
    assert result.exit_code == 0
    assert "downloaded" in result.stdout.lower()
    mock_client_cls.assert_called_once()
    mock_client.download.assert_called_once()


@patch("brahe.cli.celestrak.bh.celestrak.CelestrakClient")
def test_download_with_formats_mock(mock_client_cls, tmp_path):
    """Test download command with explicit group and format options."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    filepath = tmp_path / "gnss.json"
    result = runner.invoke(
        app,
        [
            "download",
            str(filepath),
            "--group",
            "gnss",
            "--content-format",
            "tle",
            "--file-format",
            "json",
        ],
    )
    assert result.exit_code == 0
    assert "downloaded" in result.stdout.lower()
    mock_client_cls.assert_called_once()
    mock_client.download.assert_called_once()


@patch("brahe.cli.celestrak.bh.celestrak.CelestrakClient")
def test_download_error_mock(mock_client_cls, tmp_path):
    """Test download command handles error gracefully."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.download.side_effect = RuntimeError("Download failed")

    filepath = tmp_path / "data.txt"
    result = runner.invoke(app, ["download", str(filepath)])
    assert result.exit_code == 1
