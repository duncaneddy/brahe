"""Tests for CelesTrak CLI commands"""

from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
from brahe.cli.datasets.celestrak import app

runner = CliRunner()


def test_help():
    """Test help command"""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "download" in result.stdout
    assert "lookup" in result.stdout
    assert "show" in result.stdout
    assert "search" in result.stdout
    assert "list-groups" in result.stdout


def test_list_groups():
    """Test list-groups command"""
    result = runner.invoke(app, ["list-groups"])
    assert result.exit_code == 0
    assert "active" in result.stdout
    assert "stations" in result.stdout
    assert "starlink" in result.stdout
    assert "gnss" in result.stdout


def test_search_help():
    """Test search command help"""
    result = runner.invoke(app, ["search", "--help"])
    assert result.exit_code == 0
    assert "pattern" in result.stdout.lower()
    assert "table" in result.stdout.lower()
    assert "columns" in result.stdout.lower()
    assert "minimal" in result.stdout.lower()
    assert "default" in result.stdout.lower()
    assert "all" in result.stdout.lower()


# Network tests require internet connection
# Mark as network tests to allow skipping in CI/CD


def test_search_simple_format_no_network():
    """Test search simple format - requires network"""
    # This test would require mocking or network access
    # For now, just test that command accepts correct arguments
    result = runner.invoke(app, ["search", "--help"])
    assert result.exit_code == 0


def test_search_table_format_no_network():
    """Test search table format - requires network"""
    # This test would require mocking or network access
    # For now, just test that command accepts correct arguments
    result = runner.invoke(app, ["search", "--help"])
    assert result.exit_code == 0


def test_download_help():
    """Test download command help"""
    result = runner.invoke(app, ["download", "--help"])
    assert result.exit_code == 0
    assert "filepath" in result.stdout.lower()
    assert "group" in result.stdout.lower()
    assert "content" in result.stdout.lower() and "format" in result.stdout.lower()
    assert "file" in result.stdout.lower() and "format" in result.stdout.lower()


def test_lookup_help():
    """Test lookup command help"""
    result = runner.invoke(app, ["lookup", "--help"])
    assert result.exit_code == 0
    assert "name" in result.stdout.lower()
    assert "group" in result.stdout.lower()


def test_show_help():
    """Test show command help"""
    result = runner.invoke(app, ["show", "--help"])
    assert result.exit_code == 0
    assert "identifier" in result.stdout.lower()
    assert "group" in result.stdout.lower()
    assert "compact" in result.stdout.lower()
    assert "simple" in result.stdout.lower()


# =============================================================================
# Mocked search command tests
# =============================================================================


# Sample GP record mock for search tests
def _make_gp_record(**overrides):
    """Create a mock GP record with default values."""
    rec = MagicMock()
    defaults = {
        "object_name": "ISS (ZARYA)",
        "norad_cat_id": 25544,
        "epoch": "2024-06-15 12:00:00",
        "mean_motion": 15.5,
        "eccentricity": 0.0001,
        "inclination": 51.64,
        "ra_of_asc_node": 120.0,
        "arg_of_pericenter": 90.0,
        "mean_anomaly": 45.0,
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(rec, k, v)
    return rec


@patch("brahe.cli.datasets.celestrak.bh.celestrak")
def test_search_simple_format(mock_celestrak):
    """Test search with simple (default) output format."""
    mock_client = MagicMock()
    mock_celestrak.CelestrakClient.return_value = mock_client
    mock_client.query.return_value = [
        _make_gp_record(),
        _make_gp_record(object_name="ISS DEB", norad_cat_id=99999),
    ]
    # Mock the query builder chain
    mock_query = MagicMock()
    mock_celestrak.CelestrakQuery.gp.group.return_value.filter.return_value = mock_query

    result = runner.invoke(app, ["search", "ISS"])
    assert result.exit_code == 0
    assert "ISS (ZARYA)" in result.stdout
    assert "ISS DEB" in result.stdout
    assert "25544" in result.stdout
    assert "Found 2 satellite(s)" in result.stdout


@patch("brahe.cli.datasets.celestrak.bh.celestrak")
def test_search_table_format(mock_celestrak):
    """Test search with table output format."""
    mock_client = MagicMock()
    mock_celestrak.CelestrakClient.return_value = mock_client
    mock_client.query.return_value = [_make_gp_record()]
    mock_celestrak.CelestrakQuery.gp.group.return_value.filter.return_value = (
        MagicMock()
    )

    result = runner.invoke(app, ["search", "ISS", "--table"])
    assert result.exit_code == 0
    assert "ISS" in result.stdout


@patch("brahe.cli.datasets.celestrak.bh.celestrak")
def test_search_table_minimal_columns(mock_celestrak):
    """Test search with minimal column preset."""
    mock_client = MagicMock()
    mock_celestrak.CelestrakClient.return_value = mock_client
    mock_client.query.return_value = [_make_gp_record()]
    mock_celestrak.CelestrakQuery.gp.group.return_value.filter.return_value = (
        MagicMock()
    )

    result = runner.invoke(app, ["search", "ISS", "--table", "--columns", "minimal"])
    assert result.exit_code == 0
    assert "ISS" in result.stdout


@patch("brahe.cli.datasets.celestrak.bh.celestrak")
def test_search_no_results(mock_celestrak):
    """Test search with no matching results."""
    mock_client = MagicMock()
    mock_celestrak.CelestrakClient.return_value = mock_client
    mock_client.query.return_value = []
    mock_celestrak.CelestrakQuery.gp.group.return_value.filter.return_value = (
        MagicMock()
    )

    result = runner.invoke(app, ["search", "NONEXISTENT"])
    assert result.exit_code == 0
    assert "No satellites found" in result.stdout


@patch("brahe.cli.datasets.celestrak.bh.celestrak")
def test_search_table_invalid_columns(mock_celestrak):
    """Test search with invalid column names."""
    mock_client = MagicMock()
    mock_celestrak.CelestrakClient.return_value = mock_client
    mock_client.query.return_value = [_make_gp_record()]
    mock_celestrak.CelestrakQuery.gp.group.return_value.filter.return_value = (
        MagicMock()
    )

    result = runner.invoke(app, ["search", "ISS", "--table", "--columns", "bad_column"])
    assert result.exit_code == 1
    assert "Invalid column" in result.stdout


@patch("brahe.cli.datasets.celestrak.bh.celestrak")
def test_search_error_handling(mock_celestrak):
    """Test search handles API errors gracefully."""
    mock_client = MagicMock()
    mock_celestrak.CelestrakClient.return_value = mock_client
    mock_client.query.side_effect = RuntimeError("Connection failed")
    mock_celestrak.CelestrakQuery.gp.group.return_value.filter.return_value = (
        MagicMock()
    )

    result = runner.invoke(app, ["search", "ISS"])
    assert result.exit_code == 1


# =============================================================================
# Mocked lookup command tests
# =============================================================================

SAMPLE_3LE_NAME = "ISS (ZARYA)"
SAMPLE_TLE_LINE1 = (
    "1 25544U 98067A   24166.50000000  .00016717  00000-0  10270-3 0  9002"
)
SAMPLE_TLE_LINE2 = (
    "2 25544  51.6400 120.0000 0001000  90.0000  45.0000 15.50000000    05"
)


@patch("brahe.cli.datasets.celestrak._fetch_3le_by_name")
def test_lookup_by_name(mock_fetch):
    """Test lookup command by satellite name."""
    mock_fetch.return_value = [(SAMPLE_3LE_NAME, SAMPLE_TLE_LINE1, SAMPLE_TLE_LINE2)]

    result = runner.invoke(app, ["lookup", "ISS"])
    assert result.exit_code == 0
    assert "ISS (ZARYA)" in result.stdout
    assert "25544" in result.stdout
    assert SAMPLE_TLE_LINE1 in result.stdout
    assert SAMPLE_TLE_LINE2 in result.stdout


@patch("brahe.cli.datasets.celestrak._fetch_3le_by_name")
def test_lookup_error_handling(mock_fetch):
    """Test lookup handles errors gracefully."""
    mock_fetch.side_effect = RuntimeError("No satellite found")

    result = runner.invoke(app, ["lookup", "NONEXISTENT"])
    assert result.exit_code == 1


# =============================================================================
# Mocked show command tests
# =============================================================================


@patch("brahe.cli.datasets.celestrak._fetch_3le_by_id")
def test_show_compact_by_id(mock_fetch):
    """Test show --compact with numeric NORAD ID."""
    mock_fetch.return_value = (SAMPLE_3LE_NAME, SAMPLE_TLE_LINE1, SAMPLE_TLE_LINE2)

    result = runner.invoke(app, ["show", "25544", "--compact"])
    assert result.exit_code == 0
    assert SAMPLE_TLE_LINE1 in result.stdout
    assert SAMPLE_TLE_LINE2 in result.stdout


@patch("brahe.cli.datasets.celestrak._fetch_3le_by_id")
@patch("brahe.cli.datasets.celestrak.bh.keplerian_elements_from_tle")
def test_show_simple_by_id(mock_kep, mock_fetch):
    """Test show --simple with numeric NORAD ID."""
    import brahe as bh

    mock_fetch.return_value = (SAMPLE_3LE_NAME, SAMPLE_TLE_LINE1, SAMPLE_TLE_LINE2)
    mock_epoch = bh.Epoch.from_datetime(2024, 6, 15, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    # [a, e, i, raan, argp, ma] in meters and degrees
    mock_kep.return_value = (mock_epoch, [6878137.0, 0.0001, 51.64, 120.0, 90.0, 45.0])

    result = runner.invoke(app, ["show", "25544", "--simple"])
    assert result.exit_code == 0
    assert "ISS (ZARYA)" in result.stdout
    assert "Orbital Elements" in result.stdout
    assert "Semi-major axis" in result.stdout
    assert "Eccentricity" in result.stdout
    assert "Inclination" in result.stdout


@patch("brahe.cli.datasets.celestrak._fetch_3le_by_name")
@patch("brahe.cli.datasets.celestrak.bh.keplerian_elements_from_tle")
def test_show_by_name(mock_kep, mock_fetch):
    """Test show with name identifier (non-numeric)."""
    import brahe as bh

    mock_fetch.return_value = [(SAMPLE_3LE_NAME, SAMPLE_TLE_LINE1, SAMPLE_TLE_LINE2)]
    mock_epoch = bh.Epoch.from_datetime(2024, 6, 15, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    mock_kep.return_value = (mock_epoch, [6878137.0, 0.0001, 51.64, 120.0, 90.0, 45.0])

    result = runner.invoke(app, ["show", "ISS", "--simple"])
    assert result.exit_code == 0
    assert "ISS (ZARYA)" in result.stdout


@patch("brahe.cli.datasets.celestrak._fetch_3le_by_id")
def test_show_error_handling(mock_fetch):
    """Test show handles errors gracefully."""
    mock_fetch.side_effect = RuntimeError("Not found")

    result = runner.invoke(app, ["show", "99999"])
    assert result.exit_code == 1


# =============================================================================
# Mocked download command tests
# =============================================================================


@patch("brahe.cli.datasets.celestrak.bh.celestrak")
def test_download_basic(mock_celestrak, tmp_path):
    """Test download command with default options."""
    mock_client = MagicMock()
    mock_celestrak.CelestrakClient.return_value = mock_client
    mock_celestrak.CelestrakOutputFormat.THREE_LE = "3le"
    mock_query = MagicMock()
    mock_celestrak.CelestrakQuery.gp.group.return_value.format.return_value = mock_query

    outfile = tmp_path / "test.txt"
    result = runner.invoke(app, ["download", str(outfile)])
    assert result.exit_code == 0
    assert "Downloaded" in result.stdout


@patch("brahe.cli.datasets.celestrak.bh.celestrak")
def test_download_error_handling(mock_celestrak, tmp_path):
    """Test download handles errors gracefully."""
    mock_client = MagicMock()
    mock_celestrak.CelestrakClient.return_value = mock_client
    mock_celestrak.CelestrakOutputFormat.THREE_LE = "3le"
    mock_celestrak.CelestrakQuery.gp.group.return_value.format.return_value = (
        MagicMock()
    )
    mock_client.download.side_effect = RuntimeError("Network error")

    outfile = tmp_path / "test.txt"
    result = runner.invoke(app, ["download", str(outfile)])
    assert result.exit_code == 1


# =============================================================================
# _parse_3le_text utility tests
# =============================================================================


def test_parse_3le_text():
    """Test internal 3LE text parsing."""
    from brahe.cli.datasets.celestrak import _parse_3le_text

    raw = f"{SAMPLE_3LE_NAME}\n{SAMPLE_TLE_LINE1}\n{SAMPLE_TLE_LINE2}\n"
    results = _parse_3le_text(raw)
    assert len(results) == 1
    name, line1, line2 = results[0]
    assert name == SAMPLE_3LE_NAME
    assert line1 == SAMPLE_TLE_LINE1
    assert line2 == SAMPLE_TLE_LINE2


def test_parse_3le_text_multiple():
    """Test parsing multiple 3LE entries."""
    from brahe.cli.datasets.celestrak import _parse_3le_text

    raw = (
        f"{SAMPLE_3LE_NAME}\n{SAMPLE_TLE_LINE1}\n{SAMPLE_TLE_LINE2}\n"
        f"SAT 2\n{SAMPLE_TLE_LINE1}\n{SAMPLE_TLE_LINE2}\n"
    )
    results = _parse_3le_text(raw)
    assert len(results) == 2


def test_parse_3le_text_empty():
    """Test parsing empty 3LE text."""
    from brahe.cli.datasets.celestrak import _parse_3le_text

    results = _parse_3le_text("")
    assert len(results) == 0


def test_format_to_celestrak():
    """Test CLI content format mapping."""
    from brahe.cli.datasets.celestrak import _format_to_celestrak

    import brahe as bh

    assert _format_to_celestrak("tle") == bh.celestrak.CelestrakOutputFormat.TLE
    assert _format_to_celestrak("3le") == bh.celestrak.CelestrakOutputFormat.THREE_LE
