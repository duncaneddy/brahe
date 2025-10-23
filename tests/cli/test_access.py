"""
Tests for brahe access CLI commands.
"""

import json
import tempfile
from pathlib import Path
from typer.testing import CliRunner
from brahe.cli.__main__ import app

runner = CliRunner()


def test_access_compute_basic():
    """Test basic access computation with ISS over NYC."""
    result = runner.invoke(
        app,
        [
            "access",
            "compute",
            "25544",  # ISS
            "40.7128",  # NYC lat
            "--",
            "-74.0060",  # NYC lon (negative, use -- separator)
            "--start-time",
            "2024-01-01T00:00:00",
            "--duration",
            "1",
        ],
    )

    # Should succeed
    assert result.exit_code == 0, f"Command failed with output: {result.stdout}"
    # Should mention ISS
    assert "ISS" in result.stdout or "ZARYA" in result.stdout
    # Should show location info
    assert "40.7128" in result.stdout
    assert "74.0060" in result.stdout


def test_access_compute_simple_format():
    """Test simple output format."""
    result = runner.invoke(
        app,
        [
            "access",
            "compute",
            "25544",
            "40.7128",
            "--",
            "-74.0060",
            "--start-time",
            "2024-01-01T00:00:00",
            "--duration",
            "1",
            "--output-format",
            "simple",
        ],
    )

    assert result.exit_code == 0
    # Simple format should include pipes and angle symbols
    assert "|" in result.stdout
    assert "°" in result.stdout


def test_access_compute_with_altitude():
    """Test with non-zero altitude."""
    result = runner.invoke(
        app,
        [
            "access",
            "compute",
            "25544",
            "40.7128",
            "--",
            "-74.0060",
            "--alt",
            "100",  # 100m altitude
            "--start-time",
            "2024-01-01T00:00:00",
            "--duration",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "100 m alt" in result.stdout


def test_access_compute_custom_elevation():
    """Test with custom minimum elevation."""
    result = runner.invoke(
        app,
        [
            "access",
            "compute",
            "25544",
            "40.7128",
            "--",
            "-74.0060",
            "--start-time",
            "2024-01-01T00:00:00",
            "--duration",
            "1",
            "--min-elevation",
            "20",
        ],
    )

    assert result.exit_code == 0
    assert "20.0°" in result.stdout


def test_access_compute_max_results():
    """Test limiting number of results."""
    result = runner.invoke(
        app,
        [
            "access",
            "compute",
            "25544",
            "40.7128",
            "--",
            "-74.0060",
            "--start-time",
            "2024-01-01T00:00:00",
            "--duration",
            "7",
            "--max-results",
            "3",
        ],
    )

    assert result.exit_code == 0
    # Should mention finding windows
    assert "window" in result.stdout.lower()


def test_access_compute_json_export():
    """Test JSON export functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "passes.json"

        result = runner.invoke(
            app,
            [
                "access",
                "compute",
                "25544",
                "40.7128",
                "--",
                "-74.0060",
                "--start-time",
                "2024-01-01T00:00:00",
                "--duration",
                "1",
                "--output-file",
                str(output_path),
            ],
        )

        assert result.exit_code == 0
        assert output_path.exists()

        # Validate JSON structure
        with open(output_path) as f:
            data = json.load(f)

        assert "satellite" in data
        assert "location" in data
        assert "constraint" in data
        assert "windows" in data

        assert data["satellite"]["norad_id"] == 25544
        assert data["location"]["latitude_deg"] == 40.7128
        assert data["location"]["longitude_deg"] == -74.0060
        assert data["constraint"]["min_elevation_deg"] == 10.0

        # Each window should have required fields
        if len(data["windows"]) > 0:
            window = data["windows"][0]
            assert "window_open" in window
            assert "window_close" in window
            assert "duration_sec" in window
            assert "properties" in window
            assert "azimuth_open_deg" in window["properties"]
            assert "elevation_max_deg" in window["properties"]


def test_access_compute_invalid_latitude():
    """Test error handling for invalid latitude."""
    result = runner.invoke(
        app,
        [
            "access",
            "compute",
            "25544",
            "95.0",  # Invalid latitude
            "--",
            "-74.0060",
        ],
    )

    assert result.exit_code == 1
    assert "Latitude must be between -90 and 90" in result.stdout


def test_access_compute_invalid_longitude():
    """Test error handling for invalid longitude."""
    result = runner.invoke(
        app,
        [
            "access",
            "compute",
            "25544",
            "40.7128",
            "--",
            "-185.0",  # Invalid longitude
        ],
    )

    assert result.exit_code == 1
    assert "Longitude must be between -180 and 180" in result.stdout


def test_access_compute_invalid_time_range():
    """Test error handling for invalid time range."""
    result = runner.invoke(
        app,
        [
            "access",
            "compute",
            "25544",
            "40.7128",
            "--",
            "-74.0060",
            "--start-time",
            "2024-01-02T00:00:00",
            "--end-time",
            "2024-01-01T00:00:00",  # Before start time
        ],
    )

    assert result.exit_code != 0  # Should fail (could be 1 or 2)
    assert "End time must be after start time" in result.stdout


def test_access_compute_end_without_start():
    """Test error handling for end time without start time."""
    result = runner.invoke(
        app,
        [
            "access",
            "compute",
            "25544",
            "40.7128",
            "--",
            "-74.0060",
            "--end-time",
            "2024-01-02T00:00:00",  # No start time
        ],
    )

    assert result.exit_code != 0  # Should fail (could be 1 or 2)
    assert "Cannot specify --end-time without --start-time" in result.stdout


def test_access_compute_invalid_norad_id():
    """Test error handling for invalid NORAD ID."""
    result = runner.invoke(
        app,
        [
            "access",
            "compute",
            "99999999",  # Very unlikely to exist
            "40.7128",
            "--",
            "-74.0060",
            "--start-time",
            "2024-01-01T00:00:00",
            "--duration",
            "1",
        ],
    )

    # Should fail gracefully
    assert result.exit_code == 1
    assert "Error" in result.stdout


def test_access_compute_no_windows_found():
    """Test case where no access windows are found."""
    # Use a very high elevation constraint to ensure no windows
    result = runner.invoke(
        app,
        [
            "access",
            "compute",
            "25544",
            "40.7128",
            "--",
            "-74.0060",
            "--start-time",
            "2024-01-01T00:00:00",
            "--duration",
            "0.01",  # Very short duration
            "--min-elevation",
            "89",  # Nearly overhead required
        ],
    )

    # Should succeed but report no windows
    assert result.exit_code == 0
    assert (
        "No access windows found" in result.stdout or "0 access window" in result.stdout
    )
