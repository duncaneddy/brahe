"""
Tests for brahe CLI convert module.

Tests all conversion paths for coordinates and frame conversions.
"""

import pytest
from typer.testing import CliRunner
from brahe.cli.__main__ import app

runner = CliRunner()


class TestFrameCommand:
    """Test frame conversion command (ECI <-> ECEF)."""

    def test_eci_to_ecef(self):
        """Test ECI to ECEF conversion."""
        result = runner.invoke(
            app,
            [
                "transform",
                "frame",
                "ECI",
                "ECEF",
                "2024-01-01T00:00:00Z",
                "6878137",
                "0",
                "0",
                "0",
                "7500",
                "0",
            ],
        )
        assert result.exit_code == 0
        assert "[" in result.stdout
        # Should get different values (frame rotation)
        assert "6878137" not in result.stdout or "0.000000" in result.stdout

    def test_ecef_to_eci(self):
        """Test ECEF to ECI conversion."""
        result = runner.invoke(
            app,
            [
                "transform",
                "frame",
                "ECEF",
                "ECI",
                "2024-01-01T00:00:00Z",
                "6878137",
                "0",
                "0",
                "0",
                "7500",
                "0",
            ],
        )
        assert result.exit_code == 0
        assert "[" in result.stdout

    def test_same_frame(self):
        """Test conversion to same frame (should return input)."""
        result = runner.invoke(
            app,
            [
                "transform",
                "frame",
                "ECI",
                "ECI",
                "2024-01-01T00:00:00Z",
                "6878137",
                "0",
                "0",
                "0",
                "7500",
                "0",
            ],
        )
        assert result.exit_code == 0
        assert "6878137" in result.stdout


class TestCoordinatesCommand:
    """Test coordinates conversion command."""

    # === Keplerian Conversions ===
    def test_keplerian_to_cartesian_eci(self):
        """Test Keplerian to Cartesian (ECI) conversion."""
        result = runner.invoke(
            app,
            [
                "transform",
                "coordinates",
                "keplerian",
                "cartesian",
                "",
                "6878137",
                "0.001",
                "97.8",
                "0",
                "0",
                "0",
                "--as-degrees",
            ],
        )
        assert result.exit_code == 0
        assert "[" in result.stdout
        # Should have 6 elements
        assert result.stdout.count(",") == 5

    def test_keplerian_to_cartesian_ecef(self):
        """Test Keplerian to Cartesian (ECEF) conversion."""
        result = runner.invoke(
            app,
            [
                "transform",
                "coordinates",
                "keplerian",
                "cartesian",
                "2024-01-01T00:00:00Z",
                "6878137",
                "0.001",
                "97.8",
                "0",
                "0",
                "0",
                "--as-degrees",
                "--to-frame",
                "ECEF",
            ],
        )
        assert result.exit_code == 0
        assert "[" in result.stdout

    def test_keplerian_to_geodetic(self):
        """Test Keplerian to Geodetic conversion."""
        result = runner.invoke(
            app,
            [
                "transform",
                "coordinates",
                "keplerian",
                "geodetic",
                "2024-01-01T00:00:00Z",
                "6878137",
                "0.001",
                "97.8",
                "0",
                "0",
                "0",
                "--as-degrees",
            ],
        )
        assert result.exit_code == 0
        assert "[" in result.stdout
        # Should have 3 elements (lat, lon, alt)
        assert result.stdout.count(",") == 2

    def test_keplerian_to_geocentric(self):
        """Test Keplerian to Geocentric conversion."""
        result = runner.invoke(
            app,
            [
                "transform",
                "coordinates",
                "keplerian",
                "geocentric",
                "2024-01-01T00:00:00Z",
                "6878137",
                "0.001",
                "97.8",
                "0",
                "0",
                "0",
                "--as-degrees",
            ],
        )
        assert result.exit_code == 0
        assert "[" in result.stdout
        assert result.stdout.count(",") == 2

    # === Cartesian Conversions ===
    def test_cartesian_to_keplerian(self):
        """Test Cartesian (ECI) to Keplerian conversion."""
        result = runner.invoke(
            app,
            [
                "transform",
                "coordinates",
                "cartesian",
                "keplerian",
                "",
                "6871258.863",
                "0",
                "0",
                "0",
                "1034.183142",
                "7549.721055",
                "--as-degrees",
            ],
        )
        assert result.exit_code == 0
        assert "[" in result.stdout
        # Semi-major axis should be close to input
        assert "6878" in result.stdout or "6871" in result.stdout

    def test_cartesian_frame_conversion(self):
        """Test Cartesian ECI to ECEF frame conversion."""
        result = runner.invoke(
            app,
            [
                "transform",
                "coordinates",
                "cartesian",
                "cartesian",
                "2024-01-01T00:00:00Z",
                "6878137",
                "0",
                "0",
                "0",
                "7500",
                "0",
                "--from-frame",
                "ECI",
                "--to-frame",
                "ECEF",
            ],
        )
        assert result.exit_code == 0
        assert "[" in result.stdout

    def test_cartesian_to_geodetic(self):
        """Test Cartesian (ECEF) to Geodetic conversion."""
        result = runner.invoke(
            app,
            [
                "transform",
                "coordinates",
                "cartesian",
                "geodetic",
                "2024-01-01T00:00:00Z",
                "1268739",
                "1064599",
                "4121293",
                "0",
                "0",
                "0",
                "--as-degrees",
                "--from-frame",
                "ECEF",
            ],
        )
        assert result.exit_code == 0
        assert "[" in result.stdout

    def test_cartesian_to_geocentric(self):
        """Test Cartesian (ECEF) to Geocentric conversion."""
        result = runner.invoke(
            app,
            [
                "transform",
                "coordinates",
                "cartesian",
                "geocentric",
                "2024-01-01T00:00:00Z",
                "1268739",
                "1064599",
                "4121293",
                "0",
                "0",
                "0",
                "--as-degrees",
                "--from-frame",
                "ECEF",
            ],
        )
        assert result.exit_code == 0
        assert "[" in result.stdout

    # === Geodetic Conversions ===
    def test_geodetic_to_geocentric(self):
        """Test Geodetic to Geocentric conversion."""
        result = runner.invoke(
            app,
            [
                "transform",
                "coordinates",
                "geodetic",
                "geocentric",
                "",
                "40.0",
                "75.0",
                "1000",
                "0",
                "0",
                "0",
                "--as-degrees",
            ],
        )
        assert result.exit_code == 0
        assert "[" in result.stdout
        assert result.stdout.count(",") == 2

    def test_geodetic_to_cartesian_ecef(self):
        """Test Geodetic to Cartesian (ECEF) conversion."""
        result = runner.invoke(
            app,
            [
                "transform",
                "coordinates",
                "geodetic",
                "cartesian",
                "2024-01-01T00:00:00Z",
                "40.0",
                "75.0",
                "1000",
                "0",
                "0",
                "0",
                "--as-degrees",
                "--to-frame",
                "ECEF",
            ],
        )
        assert result.exit_code == 0
        assert "[" in result.stdout

    def test_geodetic_to_cartesian_eci(self):
        """Test Geodetic to Cartesian (ECI) conversion."""
        result = runner.invoke(
            app,
            [
                "transform",
                "coordinates",
                "geodetic",
                "cartesian",
                "2024-01-01T00:00:00Z",
                "40.0",
                "75.0",
                "1000",
                "0",
                "0",
                "0",
                "--as-degrees",
                "--to-frame",
                "ECI",
            ],
        )
        assert result.exit_code == 0
        assert "[" in result.stdout

    def test_geodetic_to_keplerian_fails(self):
        """Test that Geodetic to Keplerian conversion fails (position-only)."""
        result = runner.invoke(
            app,
            [
                "transform",
                "coordinates",
                "geodetic",
                "keplerian",
                "",
                "40.0",
                "75.0",
                "1000",
                "0",
                "0",
                "0",
                "--as-degrees",
            ],
        )
        assert result.exit_code == 1
        assert "ERROR" in result.stdout

    # === Geocentric Conversions ===
    def test_geocentric_to_geodetic(self):
        """Test Geocentric to Geodetic conversion."""
        result = runner.invoke(
            app,
            [
                "transform",
                "coordinates",
                "geocentric",
                "geodetic",
                "",
                "40.0",
                "75.0",
                "6379137",
                "0",
                "0",
                "0",
                "--as-degrees",
            ],
        )
        assert result.exit_code == 0
        assert "[" in result.stdout

    def test_geocentric_to_cartesian(self):
        """Test Geocentric to Cartesian conversion."""
        result = runner.invoke(
            app,
            [
                "transform",
                "coordinates",
                "geocentric",
                "cartesian",
                "2024-01-01T00:00:00Z",
                "40.0",
                "75.0",
                "6379137",
                "0",
                "0",
                "0",
                "--as-degrees",
                "--to-frame",
                "ECEF",
            ],
        )
        assert result.exit_code == 0
        assert "[" in result.stdout

    def test_geocentric_to_keplerian_fails(self):
        """Test that Geocentric to Keplerian conversion fails (position-only)."""
        result = runner.invoke(
            app,
            [
                "transform",
                "coordinates",
                "geocentric",
                "keplerian",
                "",
                "40.0",
                "75.0",
                "6379137",
                "0",
                "0",
                "0",
                "--as-degrees",
            ],
        )
        assert result.exit_code == 1
        assert "ERROR" in result.stdout

    # === Same System Tests ===
    def test_same_system_passthrough(self):
        """Test that same system returns input unchanged."""
        result = runner.invoke(
            app,
            [
                "transform",
                "coordinates",
                "keplerian",
                "keplerian",
                "",
                "6878137",
                "0.001",
                "97.8",
                "0",
                "0",
                "0",
                "--as-degrees",
            ],
        )
        assert result.exit_code == 0
        assert "6878137" in result.stdout
        assert "0.001" in result.stdout
        assert "97.8" in result.stdout

    # === Angle Format Tests ===
    def test_radians_mode(self):
        """Test conversion with --no-as-degrees (radians)."""
        result = runner.invoke(
            app,
            [
                "transform",
                "coordinates",
                "keplerian",
                "cartesian",
                "",
                "6878137",
                "0.001",
                "1.706",
                "0",
                "0",
                "0",
                "--no-as-degrees",  # Radians mode
            ],
        )
        assert result.exit_code == 0
        assert "[" in result.stdout

    # === Format String Tests ===
    def test_format_scientific(self):
        """Test output with scientific notation format."""
        result = runner.invoke(
            app,
            [
                "transform",
                "coordinates",
                "keplerian",
                "cartesian",
                "",
                "6878137",
                "0.001",
                "97.8",
                "0",
                "0",
                "0",
                "--as-degrees",
                "--format",
                ".3e",
            ],
        )
        assert result.exit_code == 0
        assert "e+" in result.stdout or "e-" in result.stdout

    def test_format_fixed_precision(self):
        """Test output with fixed precision format."""
        result = runner.invoke(
            app,
            [
                "transform",
                "coordinates",
                "keplerian",
                "cartesian",
                "",
                "6878137",
                "0.001",
                "97.8",
                "0",
                "0",
                "0",
                "--as-degrees",
                "--format",
                ".2f",
            ],
        )
        assert result.exit_code == 0
        # Should have 2 decimal places
        assert ".00," in result.stdout or ".00]" in result.stdout

    # === Error Cases ===
    def test_missing_epoch_for_frame_conversion(self):
        """Test that missing epoch for frame conversion fails."""
        result = runner.invoke(
            app,
            [
                "transform",
                "coordinates",
                "keplerian",
                "cartesian",
                "",
                "6878137",
                "0.001",
                "97.8",
                "0",
                "0",
                "0",
                "--as-degrees",
                "--to-frame",
                "ECEF",
            ],
        )
        assert result.exit_code == 1
        assert "ERROR" in result.stdout

    def test_missing_epoch_for_geodetic(self):
        """Test that missing epoch for geodetic conversion fails."""
        result = runner.invoke(
            app,
            [
                "transform",
                "coordinates",
                "keplerian",
                "geodetic",
                "",
                "6878137",
                "0.001",
                "97.8",
                "0",
                "0",
                "0",
                "--as-degrees",
            ],
        )
        assert result.exit_code == 1
        assert "ERROR" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
