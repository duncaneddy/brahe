"""Tests for SpaceTrack CLI commands (top-level brahe spacetrack)"""

import os
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner
from rich.console import Console

from brahe.cli.spacetrack import app, _get_client, _apply_common_options

runner = CliRunner()


def test_help():
    """Test top-level spacetrack help lists all subcommands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "gp" in result.stdout
    assert "gp-history" in result.stdout
    assert "satcat" in result.stdout


def test_gp_help():
    """Test gp command help shows all options."""
    result = runner.invoke(app, ["gp", "--help"])
    assert result.exit_code == 0
    assert "--catnr" in result.stdout
    assert "--name" in result.stdout
    assert "--epoch-range" in result.stdout
    assert "--filter" in result.stdout
    assert "--limit" in result.stdout
    assert "--order-by" in result.stdout
    assert "--descending" in result.stdout
    assert "--columns" in result.stdout
    assert "--output-format" in result.stdout
    assert "--output-file" in result.stdout


def test_gp_requires_selector():
    """Test gp requires at least one selector (and fails without credentials)."""
    result = runner.invoke(app, ["gp"])
    # Should fail - either missing selector or missing credentials
    assert result.exit_code == 1


def test_gp_history_help():
    """Test gp-history command help."""
    result = runner.invoke(app, ["gp-history", "--help"])
    assert result.exit_code == 0
    assert "catnr" in result.stdout.lower()
    assert "--epoch-range" in result.stdout
    assert "--limit" in result.stdout
    assert "--output-format" in result.stdout


def test_satcat_help():
    """Test satcat command help shows all options."""
    result = runner.invoke(app, ["satcat", "--help"])
    assert result.exit_code == 0
    assert "--catnr" in result.stdout
    assert "--name" in result.stdout
    assert "--country" in result.stdout
    assert "--object-type" in result.stdout
    assert "--limit" in result.stdout
    assert "--output-format" in result.stdout


def test_satcat_requires_selector():
    """Test satcat requires at least one selector."""
    result = runner.invoke(app, ["satcat"])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Helper function tests (direct calls, no CliRunner)
# ---------------------------------------------------------------------------


def test_get_client_missing_env():
    """_get_client with no env vars raises Exit and mentions SPACETRACK_USER."""
    from click.exceptions import Exit as ClickExit

    console = Console(file=open(os.devnull, "w"))
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ClickExit):
            _get_client(console)


@patch("brahe.cli.spacetrack.bh.SpaceTrackClient")
def test_get_client_success(mock_client_cls):
    """_get_client with env vars set creates client with correct credentials."""
    console = Console(file=open(os.devnull, "w"))
    with patch.dict(
        os.environ, {"SPACETRACK_USER": "user@test.com", "SPACETRACK_PASS": "s3cret"}
    ):
        client = _get_client(console)
    mock_client_cls.assert_called_once_with("user@test.com", "s3cret")
    assert client is mock_client_cls.return_value


def test_apply_common_options_filters():
    """_apply_common_options calls query.filter() for each filter pair."""
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query

    filters = [("INCLINATION", ">50"), ("ECCENTRICITY", "<0.01")]
    result = _apply_common_options(
        mock_query, filters, order_by=None, descending=False, limit=None
    )

    assert mock_query.filter.call_count == 2
    mock_query.filter.assert_any_call("INCLINATION", ">50")
    mock_query.filter.assert_any_call("ECCENTRICITY", "<0.01")
    assert result is mock_query


def test_apply_common_options_order_by():
    """_apply_common_options calls query.order_by() when order_by is set."""
    mock_query = MagicMock()
    mock_query.order_by.return_value = mock_query

    with patch("brahe.cli.spacetrack.bh.SortOrder") as mock_sort:
        mock_sort.ASC = "ASC"
        mock_sort.DESC = "DESC"
        _apply_common_options(
            mock_query, filters=[], order_by="EPOCH", descending=False, limit=None
        )
        mock_query.order_by.assert_called_once_with("EPOCH", "ASC")

        mock_query.reset_mock()
        _apply_common_options(
            mock_query, filters=[], order_by="EPOCH", descending=True, limit=None
        )
        mock_query.order_by.assert_called_once_with("EPOCH", "DESC")


def test_apply_common_options_limit():
    """_apply_common_options calls query.limit() when limit is set."""
    mock_query = MagicMock()
    mock_query.limit.return_value = mock_query

    result = _apply_common_options(
        mock_query, filters=[], order_by=None, descending=False, limit=10
    )
    mock_query.limit.assert_called_once_with(10)
    assert result is mock_query


# ---------------------------------------------------------------------------
# GP command execution tests (CliRunner with mocked client)
# ---------------------------------------------------------------------------


@patch("brahe.cli.spacetrack.format_gp_records")
@patch("brahe.cli.spacetrack.bh.SpaceTrackClient")
@patch("brahe.cli.spacetrack.bh.SpaceTrackQuery")
@patch.dict(os.environ, {"SPACETRACK_USER": "u", "SPACETRACK_PASS": "p"})
def test_gp_by_catnr_mock(mock_query_cls, mock_client_cls, mock_format):
    """GP command with --catnr invokes query and formats results."""
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query_cls.return_value = mock_query

    mock_client = MagicMock()
    mock_client.query_gp.return_value = [MagicMock()]
    mock_client_cls.return_value = mock_client

    result = runner.invoke(app, ["gp", "--catnr", "25544"])
    assert result.exit_code == 0
    mock_client.query_gp.assert_called_once()
    mock_format.assert_called_once()


@patch("brahe.cli.spacetrack.format_gp_records")
@patch("brahe.cli.spacetrack.bh.SpaceTrackClient")
@patch("brahe.cli.spacetrack.bh.SpaceTrackQuery")
@patch.dict(os.environ, {"SPACETRACK_USER": "u", "SPACETRACK_PASS": "p"})
def test_gp_by_name_mock(mock_query_cls, mock_client_cls, mock_format):
    """GP command with --name invokes query and formats results."""
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query_cls.return_value = mock_query

    mock_client = MagicMock()
    mock_client.query_gp.return_value = [MagicMock()]
    mock_client_cls.return_value = mock_client

    result = runner.invoke(app, ["gp", "--name", "ISS"])
    assert result.exit_code == 0
    mock_client.query_gp.assert_called_once()
    mock_format.assert_called_once()


@patch("brahe.cli.spacetrack.format_gp_records")
@patch("brahe.cli.spacetrack.bh.SpaceTrackClient")
@patch("brahe.cli.spacetrack.bh.SpaceTrackQuery")
@patch.dict(os.environ, {"SPACETRACK_USER": "u", "SPACETRACK_PASS": "p"})
def test_gp_query_error_mock(mock_query_cls, mock_client_cls, mock_format):
    """GP command exits 1 when client.query_gp raises RuntimeError."""
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query_cls.return_value = mock_query

    mock_client = MagicMock()
    mock_client.query_gp.side_effect = RuntimeError("connection failed")
    mock_client_cls.return_value = mock_client

    result = runner.invoke(app, ["gp", "--catnr", "25544"])
    assert result.exit_code == 1
    mock_format.assert_not_called()


def test_gp_missing_credentials_mock():
    """GP command exits 1 when credentials are missing."""
    with patch.dict(os.environ, {}, clear=True):
        result = runner.invoke(app, ["gp", "--catnr", "25544"])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# GP History command execution tests
# ---------------------------------------------------------------------------


@patch("brahe.cli.spacetrack.format_gp_records")
@patch("brahe.cli.spacetrack.bh.SpaceTrackClient")
@patch("brahe.cli.spacetrack.bh.SpaceTrackQuery")
@patch.dict(os.environ, {"SPACETRACK_USER": "u", "SPACETRACK_PASS": "p"})
def test_gp_history_basic_mock(mock_query_cls, mock_client_cls, mock_format):
    """GP history command invokes query for a catalog number."""
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query_cls.return_value = mock_query

    mock_client = MagicMock()
    mock_client.query_gp.return_value = [MagicMock()]
    mock_client_cls.return_value = mock_client

    result = runner.invoke(app, ["gp-history", "25544"])
    assert result.exit_code == 0
    mock_client.query_gp.assert_called_once()
    mock_format.assert_called_once()


@patch("brahe.cli.spacetrack.format_gp_records")
@patch("brahe.cli.spacetrack.bh.SpaceTrackClient")
@patch("brahe.cli.spacetrack.bh.SpaceTrackQuery")
@patch.dict(os.environ, {"SPACETRACK_USER": "u", "SPACETRACK_PASS": "p"})
def test_gp_history_with_epoch_range_mock(mock_query_cls, mock_client_cls, mock_format):
    """GP history command with --epoch-range applies the epoch filter."""
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query_cls.return_value = mock_query

    mock_client = MagicMock()
    mock_client.query_gp.return_value = [MagicMock()]
    mock_client_cls.return_value = mock_client

    result = runner.invoke(
        app, ["gp-history", "25544", "--epoch-range", "2024-01-01--2024-01-31"]
    )
    assert result.exit_code == 0
    # Should have been called at least twice: once for NORAD_CAT_ID, once for EPOCH
    assert mock_query.filter.call_count >= 2
    mock_format.assert_called_once()


@patch("brahe.cli.spacetrack.format_gp_records")
@patch("brahe.cli.spacetrack.bh.SpaceTrackClient")
@patch("brahe.cli.spacetrack.bh.SpaceTrackQuery")
@patch.dict(os.environ, {"SPACETRACK_USER": "u", "SPACETRACK_PASS": "p"})
def test_gp_history_error_mock(mock_query_cls, mock_client_cls, mock_format):
    """GP history command exits 1 when client raises."""
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query_cls.return_value = mock_query

    mock_client = MagicMock()
    mock_client.query_gp.side_effect = RuntimeError("server error")
    mock_client_cls.return_value = mock_client

    result = runner.invoke(app, ["gp-history", "25544"])
    assert result.exit_code == 1
    mock_format.assert_not_called()


# ---------------------------------------------------------------------------
# SATCAT command execution tests
# ---------------------------------------------------------------------------


@patch("brahe.cli.spacetrack.format_satcat_records")
@patch("brahe.cli.spacetrack.bh.SpaceTrackClient")
@patch("brahe.cli.spacetrack.bh.SpaceTrackQuery")
@patch.dict(os.environ, {"SPACETRACK_USER": "u", "SPACETRACK_PASS": "p"})
def test_satcat_by_catnr_mock(mock_query_cls, mock_client_cls, mock_format):
    """SATCAT command with --catnr queries and formats results."""
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query_cls.return_value = mock_query

    mock_client = MagicMock()
    mock_client.query_satcat.return_value = [MagicMock()]
    mock_client_cls.return_value = mock_client

    result = runner.invoke(app, ["satcat", "--catnr", "25544"])
    assert result.exit_code == 0
    mock_client.query_satcat.assert_called_once()
    mock_format.assert_called_once()


@patch("brahe.cli.spacetrack.format_satcat_records")
@patch("brahe.cli.spacetrack.bh.SpaceTrackClient")
@patch("brahe.cli.spacetrack.bh.SpaceTrackQuery")
@patch.dict(os.environ, {"SPACETRACK_USER": "u", "SPACETRACK_PASS": "p"})
def test_satcat_by_name_mock(mock_query_cls, mock_client_cls, mock_format):
    """SATCAT command with --name queries and formats results."""
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query_cls.return_value = mock_query

    mock_client = MagicMock()
    mock_client.query_satcat.return_value = [MagicMock()]
    mock_client_cls.return_value = mock_client

    result = runner.invoke(app, ["satcat", "--name", "ISS"])
    assert result.exit_code == 0
    mock_client.query_satcat.assert_called_once()
    mock_format.assert_called_once()


@patch("brahe.cli.spacetrack.format_satcat_records")
@patch("brahe.cli.spacetrack.bh.SpaceTrackClient")
@patch("brahe.cli.spacetrack.bh.SpaceTrackQuery")
@patch.dict(os.environ, {"SPACETRACK_USER": "u", "SPACETRACK_PASS": "p"})
def test_satcat_by_country_and_type_mock(mock_query_cls, mock_client_cls, mock_format):
    """SATCAT command with --country, --object-type, and --limit."""
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query.limit.return_value = mock_query
    mock_query_cls.return_value = mock_query

    mock_client = MagicMock()
    mock_client.query_satcat.return_value = [MagicMock()]
    mock_client_cls.return_value = mock_client

    result = runner.invoke(
        app, ["satcat", "--country", "US", "--object-type", "PAYLOAD", "--limit", "5"]
    )
    assert result.exit_code == 0
    mock_client.query_satcat.assert_called_once()
    mock_format.assert_called_once()


@patch("brahe.cli.spacetrack.format_satcat_records")
@patch("brahe.cli.spacetrack.bh.SpaceTrackClient")
@patch("brahe.cli.spacetrack.bh.SpaceTrackQuery")
@patch.dict(os.environ, {"SPACETRACK_USER": "u", "SPACETRACK_PASS": "p"})
def test_satcat_error_mock(mock_query_cls, mock_client_cls, mock_format):
    """SATCAT command exits 1 when client raises."""
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query_cls.return_value = mock_query

    mock_client = MagicMock()
    mock_client.query_satcat.side_effect = RuntimeError("timeout")
    mock_client_cls.return_value = mock_client

    result = runner.invoke(app, ["satcat", "--catnr", "25544"])
    assert result.exit_code == 1
    mock_format.assert_not_called()
