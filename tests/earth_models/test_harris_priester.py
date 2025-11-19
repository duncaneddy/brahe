"""
Tests for Harris-Priester atmospheric density model.

Tests mirror the Rust implementation in src/earth_models/harris_priester.rs
"""

import pytest
import numpy as np
import brahe as bh


class TestHarrisPriester:
    """Tests for Harris-Priester atmospheric density model."""

    @pytest.mark.parametrize(
        "rsun_x, rsun_y, rsun_z, r_x, r_y, r_z, rho_expected",
        [
            # Cross-validation test cases from Rust (lines 408-416)
            (
                24622331959.580,
                -133060326832.922,
                -57688711921.833,
                3249419.145,
                -3249419.145,
                4565130.155,
                1.11289e-07,
            ),
            (
                24622331959.580,
                -133060326832.922,
                -57688711921.833,
                3299419.145,
                -3299419.145,
                4635840.833,
                2.01966e-10,
            ),
            (
                24622331959.580,
                -133060326832.922,
                -57688711921.833,
                3349419.145,
                -3349419.145,
                4706551.511,
                1.89075e-11,
            ),
            (
                24622331959.580,
                -133060326832.922,
                -57688711921.833,
                3399419.145,
                -3399419.145,
                4777262.189,
                3.38104e-12,
            ),
            (
                24622331959.580,
                -133060326832.922,
                -57688711921.833,
                3449419.145,
                -3449419.145,
                4847972.867,
                8.11538e-13,
            ),
            (
                24622331959.580,
                -133060326832.922,
                -57688711921.833,
                3499419.145,
                -3499419.145,
                4918683.545,
                2.32578e-13,
            ),
            (
                24622331959.580,
                -133060326832.922,
                -57688711921.833,
                3549419.145,
                -3549419.145,
                4989394.224,
                7.61632e-14,
            ),
            (
                24622331959.580,
                -133060326832.922,
                -57688711921.833,
                3599419.145,
                -3599419.145,
                5060104.902,
                2.83105e-14,
            ),
            (
                24622331959.580,
                -133060326832.922,
                -57688711921.833,
                3649419.145,
                -3649419.145,
                5130815.580,
                1.25646e-14,
            ),
        ],
    )
    def test_harris_priester_cross_validation(
        self, rsun_x, rsun_y, rsun_z, r_x, r_y, r_z, rho_expected
    ):
        """Test Harris-Priester model with cross-validation cases."""
        r_sun = np.array([rsun_x, rsun_y, rsun_z])
        r = np.array([r_x, r_y, r_z])

        rho = bh.density_harris_priester(r, r_sun)

        assert rho == pytest.approx(rho_expected, abs=1.0e-12)

    def test_harris_priester_bounds(self):
        """Test Harris-Priester model boundary conditions."""
        r_sun = np.array([24622331959.580, -133060326832.922, -57688711921.833])

        # Test below 100 km threshold
        r_low = bh.position_geodetic_to_ecef(
            np.array([0.0, 0.0, 50.0e3]), bh.AngleFormat.DEGREES
        )
        rho_low = bh.density_harris_priester(r_low, r_sun)
        assert rho_low == 0.0

        # Test above 1000 km threshold
        r_high = bh.position_geodetic_to_ecef(
            np.array([0.0, 0.0, 1100.0e3]), bh.AngleFormat.DEGREES
        )
        rho_high = bh.density_harris_priester(r_high, r_sun)
        assert rho_high == 0.0

    def test_harris_priester_valid_range(self):
        """Test Harris-Priester model within valid altitude range."""
        r_sun = np.array([24622331959.580, -133060326832.922, -57688711921.833])

        # Test at 200 km (within valid range)
        r = bh.position_geodetic_to_ecef(
            np.array([0.0, 0.0, 200.0e3]), bh.AngleFormat.DEGREES
        )
        rho = bh.density_harris_priester(r, r_sun)

        # Should return a positive density value
        assert rho > 0.0
        # Typical density at 200 km is on the order of 1e-10 kg/m³
        assert 1e-12 < rho < 1e-8

    def test_harris_priester_at_400km(self):
        """Test Harris-Priester model at typical LEO altitude (400 km)."""
        r_sun = np.array([24622331959.580, -133060326832.922, -57688711921.833])

        # Test at 400 km (ISS altitude)
        r = bh.position_geodetic_to_ecef(
            np.array([0.0, 0.0, 400.0e3]), bh.AngleFormat.DEGREES
        )
        rho = bh.density_harris_priester(r, r_sun)

        # Should return a positive density value
        assert rho > 0.0
        # Typical density at 400 km is on the order of 1e-12 kg/m³
        assert 1e-14 < rho < 1e-10
