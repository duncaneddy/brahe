"""
Tests for SGP4 propagator functionality in brahe.

These tests mirror the Rust test suite to ensure Python bindings work correctly.
"""

import pytest
import numpy as np
import brahe


@pytest.fixture(scope="session", autouse=True)
def setup_eop_provider():
    """Set up EOP provider for tests that require ECEF frame conversions."""
    provider = brahe.StaticEOPProvider.from_zero()
    brahe.set_global_eop_provider_from_static_provider(provider)


@pytest.fixture
def iss_classic_tle():
    """ISS TLE in classic format for testing."""
    return (
        "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992",
        "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003"
    )


@pytest.fixture
def alpha5_tle():
    """TLE with alpha-5 NORAD ID format for testing."""
    return (
        "1 A0001U 23001A   23001.00000000  .00000000  00000-0  00000-0 0  9993",
        "2 A0001  00.0000 000.0000 0000000  00.0000 000.0000 01.00000000000004"
    )


class TestTLEUtilities:
    """Test TLE utility functions that mirror Rust tests."""

    def test_calculate_tle_line_checksum(self):
        """Test TLE line checksum calculation."""
        line1 = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  999"
        expected_checksum = 2
        actual_checksum = brahe.calculate_tle_line_checksum(line1)
        assert actual_checksum == expected_checksum

    def test_validate_tle_line(self):
        """Test TLE line validation."""
        valid_line = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992"
        invalid_line = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9999"

        assert brahe.validate_tle_line(valid_line) == True
        assert brahe.validate_tle_line(invalid_line) == False

    def test_validate_tle_lines(self):
        """Test TLE lines validation."""
        line1_valid = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992"
        line2_valid = "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003"
        line1_invalid = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9999"

        assert brahe.validate_tle_lines(line1_valid, line2_valid) == True
        assert brahe.validate_tle_lines(line1_invalid, line2_valid) == False

    def test_extract_tle_norad_id_classic(self):
        """Test classic NORAD ID extraction."""
        classic_id = "25544"
        expected_id = 25544
        actual_id = brahe.extract_tle_norad_id(classic_id)
        assert actual_id == expected_id

    def test_extract_tle_norad_id_alpha5(self):
        """Test Alpha-5 NORAD ID extraction."""
        alpha5_id = "A0001"
        expected_id = 100001  # A=10, 0001=1 -> 10*10000 + 1
        actual_id = brahe.extract_tle_norad_id(alpha5_id)
        assert actual_id == expected_id


class TestSGPPropagatorCreation:
    """Test SGP propagator creation that mirrors Rust tests."""

    def test_from_tle_basic(self, iss_classic_tle):
        """Test basic SGP propagator creation from TLE."""
        sgp = brahe.SGPPropagator.from_tle(iss_classic_tle[0], iss_classic_tle[1])
        assert sgp.norad_id == 25544

    def test_alpha5_norad_id(self, alpha5_tle):
        """Test SGP propagator creation with alpha-5 NORAD ID format."""
        sgp = brahe.SGPPropagator.from_tle(alpha5_tle[0], alpha5_tle[1])
        assert sgp.norad_id == 100001

    def test_invalid_tle_lines(self):
        """Test SGP propagator creation with invalid TLE lines."""
        invalid_line1 = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9999"
        valid_line2 = "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003"

        with pytest.raises(RuntimeError):
            brahe.SGPPropagator.from_tle(invalid_line1, valid_line2)

    def test_mismatched_norad_ids(self):
        """Test SGP propagator creation with mismatched NORAD IDs."""
        line1 = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992"
        line2 = "2 12345  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003"

        with pytest.raises(RuntimeError):
            brahe.SGPPropagator.from_tle(line1, line2)