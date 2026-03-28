"""
Tests for IGRF-14 and WMMHR-2025 magnetic field models.

Tests mirror the Rust implementation in src/earth_models/magnetic_field/.
Validation against official test data from ppigrf and WMMHR reference implementations.
"""

import pytest
import numpy as np
import brahe as bh


class TestIGRF:
    """Tests for IGRF-14 magnetic field model."""

    def test_igrf_geodetic_enz_known_values(self):
        """Test IGRF at (lon=0, lat=80, h=0) at 2025.0 — should produce reasonable field values."""
        epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
        x_geod = np.array([0.0, 80.0, 0.0])
        b = bh.igrf_geodetic_enz(epc, x_geod, bh.AngleFormat.DEGREES)

        # IGRF is lower resolution than WMMHR, so we check approximate magnitudes
        assert abs(b[0]) < 500.0, f"B_east should be small, got {b[0]}"
        assert b[1] > 5000.0, f"B_north should be > 5000 nT, got {b[1]}"
        assert b[2] < -50000.0, f"B_zenith should be strongly negative, got {b[2]}"

    def test_igrf_epoch_out_of_range(self):
        """Test that out-of-range epochs raise an error."""
        x_geod = np.array([0.0, 45.0, 0.0])

        epc = bh.Epoch.from_date(1899, 1, 1, bh.TimeSystem.UTC)
        with pytest.raises(RuntimeError):
            bh.igrf_geodetic_enz(epc, x_geod, bh.AngleFormat.DEGREES)

        epc = bh.Epoch.from_date(2031, 1, 1, bh.TimeSystem.UTC)
        with pytest.raises(RuntimeError):
            bh.igrf_geodetic_enz(epc, x_geod, bh.AngleFormat.DEGREES)

    def test_igrf_radians_input(self):
        """Test that degrees and radians input produce the same result."""
        epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)

        x_deg = np.array([10.0, 45.0, 100000.0])
        x_rad = np.array([np.radians(10.0), np.radians(45.0), 100000.0])

        b_deg = bh.igrf_geodetic_enz(epc, x_deg, bh.AngleFormat.DEGREES)
        b_rad = bh.igrf_geodetic_enz(epc, x_rad, bh.AngleFormat.RADIANS)

        np.testing.assert_allclose(b_deg, b_rad, atol=1e-6)

    def test_igrf_ecef_enz_magnitude_consistency(self):
        """ECEF and ENZ outputs should have equal magnitude (rotation preserves norm)."""
        epc = bh.Epoch.from_date(2020, 6, 15, bh.TimeSystem.UTC)
        x_geod = np.array([15.0, 50.0, 500000.0])

        b_enz = bh.igrf_geodetic_enz(epc, x_geod, bh.AngleFormat.DEGREES)
        b_ecef = bh.igrf_ecef(epc, x_geod, bh.AngleFormat.DEGREES)

        assert abs(np.linalg.norm(b_enz) - np.linalg.norm(b_ecef)) < 1e-6

    def test_igrf_geocentric_enz(self):
        """Test geocentric ENZ output is different from geodetic ENZ (due to oblateness)."""
        epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
        x_geod = np.array([10.0, 60.0, 0.0])

        b_geod = bh.igrf_geodetic_enz(epc, x_geod, bh.AngleFormat.DEGREES)
        b_geoc = bh.igrf_geocentric_enz(epc, x_geod, bh.AngleFormat.DEGREES)

        # At non-zero latitude, the two should differ due to oblateness
        assert not np.allclose(b_geod, b_geoc, atol=1.0)
        # But magnitudes should be very close
        assert abs(np.linalg.norm(b_geod) - np.linalg.norm(b_geoc)) < 1.0


class TestWMMHR:
    """Tests for WMMHR-2025 magnetic field model.

    Reference values from WMMHR2025_TEST_VALUE_TABLE_FOR_REPORT.txt.
    Convention: test file has X(north), Y(east), Z(down).
    Our ENZ: B_east=Y, B_north=X, B_zenith=-Z.
    """

    def test_wmmhr_geodetic_enz_known_values_1(self):
        """2025.0, h=0 km, lat=80, lon=0."""
        epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
        x_geod = np.array([0.0, 80.0, 0.0])
        b = bh.wmmhr_geodetic_enz(epc, x_geod, bh.AngleFormat.DEGREES)

        assert b[0] == pytest.approx(144.8, abs=1.0)  # B_east = Y
        assert b[1] == pytest.approx(6517.4, abs=1.0)  # B_north = X
        assert b[2] == pytest.approx(-54701.3, abs=1.0)  # B_zenith = -Z

    def test_wmmhr_geodetic_enz_equator(self):
        """2025.0, h=0 km, lat=0, lon=120."""
        epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
        x_geod = np.array([120.0, 0.0, 0.0])
        b = bh.wmmhr_geodetic_enz(epc, x_geod, bh.AngleFormat.DEGREES)

        assert b[0] == pytest.approx(-100.3, abs=1.0)  # B_east
        assert b[1] == pytest.approx(39643.1, abs=1.0)  # B_north
        assert b[2] == pytest.approx(10580.7, abs=1.0)  # B_zenith = -Z_down

    def test_wmmhr_geodetic_enz_south(self):
        """2025.0, h=0 km, lat=-80, lon=240."""
        epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
        x_geod = np.array([240.0, -80.0, 0.0])
        b = bh.wmmhr_geodetic_enz(epc, x_geod, bh.AngleFormat.DEGREES)

        assert b[0] == pytest.approx(15740.2, abs=1.0)  # B_east
        assert b[1] == pytest.approx(6136.3, abs=1.0)  # B_north
        assert b[2] == pytest.approx(52096.7, abs=1.0)  # B_zenith

    def test_wmmhr_at_altitude(self):
        """2025.0, h=100 km, lat=80, lon=0."""
        epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
        x_geod = np.array([0.0, 80.0, 100000.0])  # 100 km in meters
        b = bh.wmmhr_geodetic_enz(epc, x_geod, bh.AngleFormat.DEGREES)

        assert b[0] == pytest.approx(81.8, abs=1.0)
        assert b[1] == pytest.approx(6218.6, abs=1.0)
        assert b[2] == pytest.approx(-52567.3, abs=1.0)

    def test_wmmhr_secular_variation(self):
        """2027.5, h=0 km, lat=80, lon=0 — tests time interpolation with secular variation."""
        epc = bh.Epoch.from_date(2027, 7, 2, bh.TimeSystem.UTC)
        x_geod = np.array([0.0, 80.0, 0.0])
        b = bh.wmmhr_geodetic_enz(epc, x_geod, bh.AngleFormat.DEGREES)

        assert b[0] == pytest.approx(293.5, abs=2.0)
        assert b[1] == pytest.approx(6494.8, abs=2.0)
        assert b[2] == pytest.approx(-54779.0, abs=2.0)

    def test_wmmhr_nmax_truncation(self):
        """Verify nmax parameter changes results."""
        epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
        x_geod = np.array([0.0, 45.0, 0.0])

        b_full = bh.wmmhr_geodetic_enz(epc, x_geod, bh.AngleFormat.DEGREES)
        b_low = bh.wmmhr_geodetic_enz(epc, x_geod, bh.AngleFormat.DEGREES, nmax=13)

        # Overall magnitudes should be similar
        mag_full = np.linalg.norm(b_full)
        mag_low = np.linalg.norm(b_low)
        assert abs(mag_full - mag_low) < 500.0

        # But they should not be identical
        assert np.linalg.norm(b_full - b_low) > 0.1

    def test_wmmhr_invalid_nmax(self):
        """Invalid nmax should raise an error."""
        epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
        x_geod = np.array([0.0, 45.0, 0.0])

        with pytest.raises(RuntimeError):
            bh.wmmhr_geodetic_enz(epc, x_geod, bh.AngleFormat.DEGREES, nmax=0)

        with pytest.raises(RuntimeError):
            bh.wmmhr_geodetic_enz(epc, x_geod, bh.AngleFormat.DEGREES, nmax=134)

    def test_wmmhr_ecef_enz_magnitude_consistency(self):
        """ECEF and ENZ outputs should have equal magnitude."""
        epc = bh.Epoch.from_date(2025, 6, 15, bh.TimeSystem.UTC)
        x_geod = np.array([15.0, 50.0, 500000.0])

        b_enz = bh.wmmhr_geodetic_enz(epc, x_geod, bh.AngleFormat.DEGREES)
        b_ecef = bh.wmmhr_ecef(epc, x_geod, bh.AngleFormat.DEGREES)

        assert abs(np.linalg.norm(b_enz) - np.linalg.norm(b_ecef)) < 1e-6

    def test_wmmhr_geocentric_enz(self):
        """Test geocentric ENZ output differs from geodetic ENZ."""
        epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
        x_geod = np.array([10.0, 60.0, 0.0])

        b_geod = bh.wmmhr_geodetic_enz(epc, x_geod, bh.AngleFormat.DEGREES)
        b_geoc = bh.wmmhr_geocentric_enz(epc, x_geod, bh.AngleFormat.DEGREES)

        # At 60 degrees latitude they should differ due to oblateness
        assert not np.allclose(b_geod, b_geoc, atol=1.0)
        # But magnitudes should be very close
        assert abs(np.linalg.norm(b_geod) - np.linalg.norm(b_geoc)) < 1.0
