"""
Tests for mean-osculating Keplerian element conversions.

Tests the Brouwer-Lyddane first-order J2 perturbation mapping between
mean and osculating orbital elements.
"""

import pytest
import numpy as np
import brahe


class TestMeanOsculatingConversions:
    """Tests for mean-osculating Keplerian element conversions."""

    def test_round_trip_mean_to_osc_to_mean_radians(self):
        """Test round-trip: mean -> osc -> mean (radians)."""
        # Define mean elements for a typical LEO satellite
        mean = np.array(
            [
                brahe.R_EARTH + 500e3,  # a = ~6878 km
                0.01,  # e = 0.01 (slightly eccentric)
                np.radians(45.0),  # i = 45 degrees
                np.radians(30.0),  # Ω = 30 degrees
                np.radians(60.0),  # ω = 60 degrees
                np.radians(90.0),  # M = 90 degrees
            ]
        )

        # Convert mean -> osc -> mean
        osc = brahe.state_koe_mean_to_osc(mean, brahe.AngleFormat.RADIANS)
        mean_recovered = brahe.state_koe_osc_to_mean(osc, brahe.AngleFormat.RADIANS)

        # Check that recovered elements are close to original
        # Note: First-order approximation means small errors of order J2² are expected
        # Tolerances match Rust tests accounting for this limitation
        assert mean[0] == pytest.approx(mean_recovered[0], abs=100.0)  # 100 m for SMA
        assert mean[1] == pytest.approx(
            mean_recovered[1], abs=1e-4
        )  # Eccentricity (J2² errors)
        assert mean[2] == pytest.approx(
            mean_recovered[2], abs=0.01
        )  # Inclination (~0.6 deg)
        assert mean[3] == pytest.approx(mean_recovered[3], abs=0.01)  # RAAN (~0.6 deg)
        assert mean[4] == pytest.approx(mean_recovered[4], abs=0.01)  # AoP (~0.6 deg)
        assert mean[5] == pytest.approx(
            mean_recovered[5], abs=0.01
        )  # Mean anomaly (~0.6 deg)

    def test_round_trip_mean_to_osc_to_mean_degrees(self):
        """Test round-trip: mean -> osc -> mean (degrees)."""
        # Define mean elements for a typical LEO satellite (angles in degrees)
        mean = np.array(
            [
                brahe.R_EARTH + 500e3,  # a = ~6878 km
                0.01,  # e = 0.01 (slightly eccentric)
                45.0,  # i = 45 degrees
                30.0,  # Ω = 30 degrees
                60.0,  # ω = 60 degrees
                90.0,  # M = 90 degrees
            ]
        )

        # Convert mean -> osc -> mean
        osc = brahe.state_koe_mean_to_osc(mean, brahe.AngleFormat.DEGREES)
        mean_recovered = brahe.state_koe_osc_to_mean(osc, brahe.AngleFormat.DEGREES)

        # Check that recovered elements are close to original
        # Tolerances in degrees (~0.01 radians = ~0.6 degrees)
        assert mean[0] == pytest.approx(mean_recovered[0], abs=100.0)  # 100 m for SMA
        assert mean[1] == pytest.approx(mean_recovered[1], abs=1e-4)  # Eccentricity
        assert mean[2] == pytest.approx(mean_recovered[2], abs=0.6)  # Inclination
        assert mean[3] == pytest.approx(mean_recovered[3], abs=0.6)  # RAAN
        assert mean[4] == pytest.approx(mean_recovered[4], abs=0.6)  # AoP
        assert mean[5] == pytest.approx(mean_recovered[5], abs=0.6)  # Mean anomaly

    def test_round_trip_osc_to_mean_to_osc(self):
        """Test round-trip: osc -> mean -> osc ≈ original."""
        # Define osculating elements for a typical LEO satellite
        osc = np.array(
            [
                brahe.R_EARTH + 600e3,  # a = ~6978 km
                0.02,  # e = 0.02
                np.radians(60.0),  # i = 60 degrees
                np.radians(45.0),  # Ω = 45 degrees
                np.radians(120.0),  # ω = 120 degrees
                np.radians(180.0),  # M = 180 degrees
            ]
        )

        # Convert osc -> mean -> osc
        mean = brahe.state_koe_osc_to_mean(osc, brahe.AngleFormat.RADIANS)
        osc_recovered = brahe.state_koe_mean_to_osc(mean, brahe.AngleFormat.RADIANS)

        # Check that recovered elements are close to original
        # Tolerances match Rust tests accounting for J2² first-order approximation errors
        assert osc[0] == pytest.approx(osc_recovered[0], abs=100.0)  # 100 m for SMA
        assert osc[1] == pytest.approx(
            osc_recovered[1], abs=1e-4
        )  # Eccentricity (J2² errors)
        assert osc[2] == pytest.approx(
            osc_recovered[2], abs=0.01
        )  # Inclination (~0.6 deg)
        assert osc[3] == pytest.approx(osc_recovered[3], abs=0.01)  # RAAN (~0.6 deg)
        assert osc[4] == pytest.approx(osc_recovered[4], abs=0.01)  # AoP (~0.6 deg)
        assert osc[5] == pytest.approx(
            osc_recovered[5], abs=0.01
        )  # Mean anomaly (~0.6 deg)

    def test_near_circular_orbit(self):
        """Test near-circular orbit (small eccentricity)."""
        mean = np.array(
            [
                brahe.R_EARTH + 400e3,
                0.0001,  # Very small eccentricity
                np.radians(28.5),
                0.0,
                0.0,
                0.0,
            ]
        )

        # Should not raise
        osc = brahe.state_koe_mean_to_osc(mean, brahe.AngleFormat.RADIANS)
        mean_recovered = brahe.state_koe_osc_to_mean(osc, brahe.AngleFormat.RADIANS)

        # Semi-major axis should be close
        assert mean[0] == pytest.approx(mean_recovered[0], abs=100.0)

    def test_sun_synchronous_orbit(self):
        """Test sun-synchronous orbit (high inclination)."""
        mean = np.array(
            [
                brahe.R_EARTH + 700e3,
                0.001,
                np.radians(98.0),  # Sun-synchronous inclination
                np.radians(45.0),
                np.radians(90.0),
                np.radians(270.0),
            ]
        )

        osc = brahe.state_koe_mean_to_osc(mean, brahe.AngleFormat.RADIANS)
        mean_recovered = brahe.state_koe_osc_to_mean(osc, brahe.AngleFormat.RADIANS)

        # Note: For small eccentricity (0.001), the relative J2² error can be larger
        # Tolerances match Rust tests
        assert mean[0] == pytest.approx(mean_recovered[0], abs=100.0)
        assert mean[1] == pytest.approx(
            mean_recovered[1], abs=1e-3
        )  # Larger tolerance for very small e
        assert mean[2] == pytest.approx(mean_recovered[2], abs=1e-4)  # Inclination

    def test_osc_differs_from_mean(self):
        """Test that osculating elements differ from mean elements."""
        mean = np.array(
            [
                brahe.R_EARTH + 500e3,
                0.01,
                np.radians(45.0),
                np.radians(30.0),
                np.radians(60.0),
                np.radians(90.0),
            ]
        )

        osc = brahe.state_koe_mean_to_osc(mean, brahe.AngleFormat.RADIANS)

        # Osculating and mean should differ (J2 perturbation effect)
        # The semi-major axis should be different
        assert abs(osc[0] - mean[0]) > 1.0  # Should differ by more than 1 meter

    def test_various_mean_anomalies(self):
        """Test various mean anomaly values."""
        base_mean = np.array(
            [
                brahe.R_EARTH + 500e3,
                0.01,
                np.radians(45.0),
                np.radians(30.0),
                np.radians(60.0),
                0.0,  # Will be varied
            ]
        )

        for m_deg in [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]:
            mean = base_mean.copy()
            mean[5] = np.radians(m_deg)

            osc = brahe.state_koe_mean_to_osc(mean, brahe.AngleFormat.RADIANS)
            mean_recovered = brahe.state_koe_osc_to_mean(osc, brahe.AngleFormat.RADIANS)

            # Check semi-major axis recovery
            assert mean[0] == pytest.approx(mean_recovered[0], abs=100.0), (
                f"Failed at mean anomaly {m_deg} degrees"
            )

    def test_geo_orbit(self):
        """Test GEO orbit."""
        # Geostationary orbit parameters
        mean = np.array(
            [
                42164e3,  # GEO radius
                0.0001,  # Near-circular
                np.radians(0.1),  # Near-equatorial
                np.radians(45.0),
                0.0,
                0.0,
            ]
        )

        osc = brahe.state_koe_mean_to_osc(mean, brahe.AngleFormat.RADIANS)
        mean_recovered = brahe.state_koe_osc_to_mean(osc, brahe.AngleFormat.RADIANS)

        # At GEO altitude, J2 effects are smaller
        assert mean[0] == pytest.approx(mean_recovered[0], abs=10.0)

    def test_numpy_array_input(self):
        """Test that numpy arrays are properly accepted."""
        # Test with numpy array
        mean_np = np.array(
            [
                brahe.R_EARTH + 500e3,
                0.01,
                np.radians(45.0),
                np.radians(30.0),
                np.radians(60.0),
                np.radians(90.0),
            ]
        )

        osc = brahe.state_koe_mean_to_osc(mean_np, brahe.AngleFormat.RADIANS)
        assert isinstance(osc, np.ndarray)
        assert len(osc) == 6

    def test_list_input(self):
        """Test that Python lists are properly accepted."""
        # Test with Python list
        mean_list = [
            brahe.R_EARTH + 500e3,
            0.01,
            np.radians(45.0),
            np.radians(30.0),
            np.radians(60.0),
            np.radians(90.0),
        ]

        osc = brahe.state_koe_mean_to_osc(mean_list, brahe.AngleFormat.RADIANS)
        assert isinstance(osc, np.ndarray)
        assert len(osc) == 6

    def test_return_type(self):
        """Test that return type is numpy array."""
        mean = np.array(
            [
                brahe.R_EARTH + 500e3,
                0.01,
                np.radians(45.0),
                0.0,
                0.0,
                0.0,
            ]
        )

        osc = brahe.state_koe_mean_to_osc(mean, brahe.AngleFormat.RADIANS)
        mean_back = brahe.state_koe_osc_to_mean(osc, brahe.AngleFormat.RADIANS)

        assert isinstance(osc, np.ndarray)
        assert isinstance(mean_back, np.ndarray)
        assert osc.dtype == np.float64
        assert mean_back.dtype == np.float64

    def test_invalid_array_length(self):
        """Test that invalid array lengths raise ValueError."""
        # Test with wrong length
        invalid = np.array([1.0, 2.0, 3.0])  # Only 3 elements

        with pytest.raises(ValueError):
            brahe.state_koe_mean_to_osc(invalid, brahe.AngleFormat.RADIANS)

        with pytest.raises(ValueError):
            brahe.state_koe_osc_to_mean(invalid, brahe.AngleFormat.RADIANS)

    def test_moderate_eccentricity(self):
        """Test with moderate eccentricity."""
        mean = np.array(
            [
                brahe.R_EARTH + 500e3,
                0.1,  # Moderate eccentricity
                np.radians(45.0),
                np.radians(30.0),
                np.radians(60.0),
                np.radians(90.0),
            ]
        )

        osc = brahe.state_koe_mean_to_osc(mean, brahe.AngleFormat.RADIANS)
        mean_recovered = brahe.state_koe_osc_to_mean(osc, brahe.AngleFormat.RADIANS)

        # Should still recover reasonably well
        assert mean[0] == pytest.approx(mean_recovered[0], abs=200.0)
        assert mean[1] == pytest.approx(mean_recovered[1], abs=1e-5)

    def test_degrees_consistency(self):
        """Test that degrees input produces degrees output."""
        # Input in degrees
        mean_deg = np.array(
            [
                brahe.R_EARTH + 500e3,
                0.01,
                45.0,  # degrees
                30.0,  # degrees
                60.0,  # degrees
                90.0,  # degrees
            ]
        )

        # Convert to osculating with degrees format
        osc_deg = brahe.state_koe_mean_to_osc(mean_deg, brahe.AngleFormat.DEGREES)

        # Verify output angles are in reasonable degree range (not radian range)
        assert osc_deg[2] >= 7.0 and osc_deg[2] < 180.0
        assert osc_deg[3] >= 7.0 and osc_deg[3] < 360.0  # RAAN should be in degrees
        assert osc_deg[4] >= 7.0 and osc_deg[4] < 360.0  # AoP should be in degrees
        assert osc_deg[5] >= 7.0 and osc_deg[5] < 360.0  # M should be in degrees
