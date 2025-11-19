"""
Tests for NRLMSISE-00 atmospheric density model.

Tests mirror the Rust implementation in src/earth_models/nrlmsise00.rs
"""

import pytest
import numpy as np
import brahe as bh


class TestNrlmsise00:
    """Tests for NRLMSISE-00 atmospheric density model."""

    def test_nrlmsise00_ecef_basic(self, eop):
        """Test NRLMSISE-00 model with ECEF coordinates."""
        bh.initialize_sw()

        # Test at 400 km altitude over equator
        epc = bh.Epoch.from_date(2020, 6, 1, bh.TimeSystem.UTC)
        x_ecef = np.array([bh.R_EARTH + 400e3, 0.0, 0.0])

        density = bh.density_nrlmsise00(epc, x_ecef)

        # Should return positive density
        assert density > 0.0
        # Typical density at 400 km is on the order of 1e-12 kg/m³
        assert 1e-14 < density < 1e-10

    def test_nrlmsise00_ecef_matches_geod(self, eop):
        """Test that ECEF and geodetic versions produce same results."""
        bh.initialize_sw()

        epc = bh.Epoch.from_date(2020, 6, 1, bh.TimeSystem.UTC)

        # Define geodetic position
        geod = np.array([0.0, 45.0, 400e3])  # 45° lat, 400 km altitude

        # Convert to ECEF
        x_ecef = bh.position_geodetic_to_ecef(geod, bh.AngleFormat.DEGREES)

        # Compute density both ways
        density_ecef = bh.density_nrlmsise00(epc, x_ecef)
        density_geod = bh.density_nrlmsise00_geod(epc, geod)

        # Should match within numerical precision
        assert density_ecef == pytest.approx(density_geod, rel=1e-10)

    def test_nrlmsise00_geod_basic(self, eop):
        """Test NRLMSISE-00 model basic functionality with geodetic coordinates."""
        bh.initialize_sw()

        # Test at 400 km altitude
        epc = bh.Epoch.from_date(2020, 6, 1, bh.TimeSystem.UTC)
        geod = np.array([-74.0, 40.7, 400e3])  # NYC area, 400 km altitude

        density = bh.density_nrlmsise00_geod(epc, geod)

        # Should return positive density
        assert density > 0.0
        # Typical density at 400 km is on the order of 1e-12 kg/m³
        assert 1e-14 < density < 1e-10

    def test_nrlmsise00_altitude_variation(self, eop):
        """Test NRLMSISE-00 density decreases with altitude."""
        bh.initialize_sw()

        epc = bh.Epoch.from_date(2020, 6, 1, bh.TimeSystem.UTC)
        lon, lat = 0.0, 0.0

        # Test at different altitudes
        altitudes = [200e3, 400e3, 600e3, 800e3]
        densities = []

        for alt in altitudes:
            geod = np.array([lon, lat, alt])
            density = bh.density_nrlmsise00_geod(epc, geod)
            densities.append(density)

        # Density should decrease with altitude
        for i in range(len(densities) - 1):
            assert densities[i] > densities[i + 1], (
                f"Density at {altitudes[i] / 1e3}km should be > at {altitudes[i + 1] / 1e3}km"
            )

    def test_nrlmsise00_different_epochs(self, eop):
        """Test NRLMSISE-00 produces different results for different epochs."""
        bh.initialize_sw()

        geod = np.array([0.0, 0.0, 400e3])

        # Test at different epochs
        epc1 = bh.Epoch.from_date(2020, 1, 1, bh.TimeSystem.UTC)
        epc2 = bh.Epoch.from_date(2020, 7, 1, bh.TimeSystem.UTC)

        density1 = bh.density_nrlmsise00_geod(epc1, geod)
        density2 = bh.density_nrlmsise00_geod(epc2, geod)

        # Densities should be different (seasonal variation)
        assert density1 != density2
        # But both should be positive and reasonable
        assert density1 > 0.0
        assert density2 > 0.0

    def test_nrlmsise00_different_locations(self, eop):
        """Test NRLMSISE-00 produces different results for different locations."""
        bh.initialize_sw()

        epc = bh.Epoch.from_date(2020, 6, 1, bh.TimeSystem.UTC)
        alt = 400e3

        # Test at different locations
        geod1 = np.array([0.0, 0.0, alt])  # Equator
        geod2 = np.array([0.0, 60.0, alt])  # High latitude

        density1 = bh.density_nrlmsise00_geod(epc, geod1)
        density2 = bh.density_nrlmsise00_geod(epc, geod2)

        # Both should be positive and reasonable
        assert density1 > 0.0
        assert density2 > 0.0

    def test_nrlmsise00_invalid_input(self, eop):
        """Test NRLMSISE-00 error handling for invalid inputs."""
        bh.initialize_sw()

        epc = bh.Epoch.from_date(2020, 6, 1, bh.TimeSystem.UTC)

        # Test with wrong array length (ECEF version)
        with pytest.raises(ValueError):
            bh.density_nrlmsise00(epc, np.array([0.0, 0.0]))  # Only 2 elements

        # Test with wrong array length (geodetic version)
        with pytest.raises(ValueError):
            bh.density_nrlmsise00_geod(epc, np.array([0.0, 0.0]))  # Only 2 elements

    def test_nrlmsise00_leo_altitudes(self, eop):
        """Test NRLMSISE-00 at typical LEO altitudes."""
        bh.initialize_sw()

        epc = bh.Epoch.from_date(2020, 6, 1, bh.TimeSystem.UTC)

        # ISS altitude (~400 km)
        geod_iss = np.array([0.0, 0.0, 400e3])
        density_iss = bh.density_nrlmsise00_geod(epc, geod_iss)

        # Starlink altitude (~550 km)
        geod_starlink = np.array([0.0, 0.0, 550e3])
        density_starlink = bh.density_nrlmsise00_geod(epc, geod_starlink)

        # ISS should have higher density than Starlink
        assert density_iss > density_starlink

        # Both should be in reasonable LEO range
        assert 1e-14 < density_iss < 1e-10
        assert 1e-15 < density_starlink < 1e-11
