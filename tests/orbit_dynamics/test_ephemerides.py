"""
Tests for ephemerides module

Tests low-precision analytical and high-precision DE ephemeris functions
for sun and moon positions.
"""

import pytest
import numpy as np
import brahe as bh


class TestAnalyticalEphemerides:
    """Tests for low-precision analytical ephemeris functions"""

    def test_sun_position_returns_vector(self):
        """Test that sun_position returns a 3-element position vector"""
        epc = bh.Epoch.from_date(2024, 2, 25, bh.TimeSystem.UTC)
        r_sun = bh.sun_position(epc)

        assert isinstance(r_sun, np.ndarray)
        assert r_sun.shape == (3,)
        # Sun should be ~1 AU away
        distance = np.linalg.norm(r_sun)
        assert 1.4e11 < distance < 1.6e11  # m

    def test_moon_position_returns_vector(self):
        """Test that moon_position returns a 3-element position vector"""
        epc = bh.Epoch.from_date(2024, 2, 25, bh.TimeSystem.UTC)
        r_moon = bh.moon_position(epc)

        assert isinstance(r_moon, np.ndarray)
        assert r_moon.shape == (3,)
        # Moon should be ~384,000 km away
        distance = np.linalg.norm(r_moon)
        assert 3.5e8 < distance < 4.1e8  # m

    @pytest.mark.parametrize(
        "year,month,day",
        [
            (2024, 1, 1),
            (2024, 4, 15),
            (2024, 7, 1),
            (2024, 10, 15),
            (2024, 12, 31),
        ],
    )
    def test_sun_position_throughout_year(self, year, month, day):
        """Test sun position at different times of year"""
        epc = bh.Epoch.from_date(year, month, day, bh.TimeSystem.UTC)
        r_sun = bh.sun_position(epc)

        distance = np.linalg.norm(r_sun)
        # Sun distance varies throughout year (perihelion ~0.983 AU, aphelion ~1.017 AU)
        assert 1.45e11 < distance < 1.53e11  # m

    @pytest.mark.parametrize(
        "year,month,day",
        [
            (2024, 1, 1),
            (2024, 4, 15),
            (2024, 7, 1),
            (2024, 10, 15),
            (2024, 12, 31),
        ],
    )
    def test_moon_position_throughout_year(self, year, month, day):
        """Test moon position at different times of year"""
        epc = bh.Epoch.from_date(year, month, day, bh.TimeSystem.UTC)
        r_moon = bh.moon_position(epc)

        distance = np.linalg.norm(r_moon)
        # Moon distance varies (perigee ~363,000 km, apogee ~405,000 km)
        assert 3.5e8 < distance < 4.1e8  # m


class TestDEEphemerides:
    """Tests for high-precision DE ephemeris functions"""

    @pytest.fixture(autouse=True)
    def setup_ephemeris(self):
        """Initialize ephemeris before each test"""
        # Initialize with test data if available
        try:
            import os
            from pathlib import Path

            # Try to use test asset if it exists
            test_asset = (
                Path(os.environ.get("CARGO_MANIFEST_DIR", "."))
                / "test_assets"
                / "de440s.bsp"
            )
            if test_asset.exists():
                # Copy to cache using the same approach as Rust tests
                cache_dir = bh.utils.get_brahe_cache_dir_with_subdir("naif")
                cache_path = Path(cache_dir) / "de440s.bsp"
                if not cache_path.exists():
                    import shutil

                    shutil.copy(test_asset, cache_path)

            # Initialize ephemeris
            bh.initialize_ephemeris()
        except Exception as e:
            pytest.skip(f"Could not initialize ephemeris: {e}")

    def test_initialize_ephemeris(self):
        """Test that initialize_ephemeris completes without error"""
        # Re-initialization should be safe
        bh.initialize_ephemeris()

    def test_sun_position_de_returns_vector(self):
        """Test that sun_position_de returns a 3-element position vector"""
        epc = bh.Epoch.from_date(2024, 2, 25, bh.TimeSystem.UTC)
        r_sun = bh.sun_position_de(epc, bh.EphemerisSource.DE440s)

        assert isinstance(r_sun, np.ndarray)
        assert r_sun.shape == (3,)
        # Sun should be ~1 AU away
        distance = np.linalg.norm(r_sun)
        assert 1.4e11 < distance < 1.6e11  # m

    def test_moon_position_de_returns_vector(self):
        """Test that moon_position_de returns a 3-element position vector"""
        epc = bh.Epoch.from_date(2024, 2, 25, bh.TimeSystem.UTC)
        r_moon = bh.moon_position_de(epc, bh.EphemerisSource.DE440s)

        assert isinstance(r_moon, np.ndarray)
        assert r_moon.shape == (3,)
        # Moon should be ~384,000 km away
        distance = np.linalg.norm(r_moon)
        assert 3.5e8 < distance < 4.1e8  # m

    @pytest.mark.parametrize(
        "year,month,day",
        [
            (2025, 1, 1),
            (2025, 2, 15),
            (2025, 3, 30),
            (2025, 5, 15),
            (2025, 7, 1),
            (2025, 8, 15),
            (2025, 10, 1),
            (2025, 11, 15),
            (2025, 12, 31),
        ],
    )
    def test_sun_position_de_vs_analytical(self, year, month, day):
        """Test that DE440s sun position is close to analytical (within ~0.1 degrees)"""
        epc = bh.Epoch.from_date(year, month, day, bh.TimeSystem.UTC)
        r_sun_analytical = bh.sun_position(epc)
        r_sun_de = bh.sun_position_de(epc, bh.EphemerisSource.DE440s)

        # Compute angle between vectors
        dot_product = np.dot(r_sun_analytical, r_sun_de) / (
            np.linalg.norm(r_sun_analytical) * np.linalg.norm(r_sun_de)
        )
        angle = np.arccos(dot_product) * (180.0 / np.pi)

        # Should agree to within 0.1 degrees
        assert angle < 0.1

    @pytest.mark.parametrize(
        "year,month,day",
        [
            (2025, 1, 1),
            (2025, 2, 15),
            (2025, 3, 30),
            (2025, 5, 15),
            (2025, 7, 1),
            (2025, 8, 15),
            (2025, 10, 1),
            (2025, 11, 15),
            (2025, 12, 31),
        ],
    )
    def test_moon_position_de_vs_analytical(self, year, month, day):
        """Test that DE440s moon position is close to analytical (within ~0.1 degrees)"""
        epc = bh.Epoch.from_date(year, month, day, bh.TimeSystem.UTC)
        r_moon_analytical = bh.moon_position(epc)
        r_moon_de = bh.moon_position_de(epc, bh.EphemerisSource.DE440s)

        # Compute angle between vectors
        dot_product = np.dot(r_moon_analytical, r_moon_de) / (
            np.linalg.norm(r_moon_analytical) * np.linalg.norm(r_moon_de)
        )
        angle = np.arccos(dot_product) * (180.0 / np.pi)

        # Should agree to within 0.1 degrees
        assert angle < 0.1

    @pytest.mark.parametrize(
        "year,month,day",
        [
            (2025, 1, 1),
            (2025, 2, 15),
            (2025, 3, 30),
            (2025, 5, 15),
            (2025, 7, 1),
            (2025, 8, 15),
            (2025, 10, 1),
            (2025, 11, 15),
            (2025, 12, 31),
        ],
    )
    def test_mercury_position_de(self, year, month, day):
        """Test that mercury_position_de returns valid position vector"""
        epc = bh.Epoch.from_date(year, month, day, bh.TimeSystem.UTC)
        r_mercury = bh.mercury_position_de(epc, bh.EphemerisSource.DE440s)

        assert isinstance(r_mercury, np.ndarray)
        assert r_mercury.shape == (3,)

    @pytest.mark.parametrize(
        "year,month,day",
        [
            (2025, 1, 1),
            (2025, 2, 15),
            (2025, 3, 30),
            (2025, 5, 15),
            (2025, 7, 1),
            (2025, 8, 15),
            (2025, 10, 1),
            (2025, 11, 15),
            (2025, 12, 31),
        ],
    )
    def test_venus_position_de(self, year, month, day):
        """Test that venus_position_de returns valid position vector"""
        epc = bh.Epoch.from_date(year, month, day, bh.TimeSystem.UTC)
        r_venus = bh.venus_position_de(epc, bh.EphemerisSource.DE440s)

        assert isinstance(r_venus, np.ndarray)
        assert r_venus.shape == (3,)

    @pytest.mark.parametrize(
        "year,month,day",
        [
            (2025, 1, 1),
            (2025, 2, 15),
            (2025, 3, 30),
            (2025, 5, 15),
            (2025, 7, 1),
            (2025, 8, 15),
            (2025, 10, 1),
            (2025, 11, 15),
            (2025, 12, 31),
        ],
    )
    def test_mars_position_de(self, year, month, day):
        """Test that mars_position_de returns valid position vector"""
        epc = bh.Epoch.from_date(year, month, day, bh.TimeSystem.UTC)
        r_mars = bh.mars_position_de(epc, bh.EphemerisSource.DE440s)

        assert isinstance(r_mars, np.ndarray)
        assert r_mars.shape == (3,)

    @pytest.mark.parametrize(
        "year,month,day",
        [
            (2025, 1, 1),
            (2025, 2, 15),
            (2025, 3, 30),
            (2025, 5, 15),
            (2025, 7, 1),
            (2025, 8, 15),
            (2025, 10, 1),
            (2025, 11, 15),
            (2025, 12, 31),
        ],
    )
    def test_jupiter_position_de(self, year, month, day):
        """Test that jupiter_position_de returns valid position vector"""
        epc = bh.Epoch.from_date(year, month, day, bh.TimeSystem.UTC)
        r_jupiter = bh.jupiter_position_de(epc, bh.EphemerisSource.DE440s)

        assert isinstance(r_jupiter, np.ndarray)
        assert r_jupiter.shape == (3,)

    @pytest.mark.parametrize(
        "year,month,day",
        [
            (2025, 1, 1),
            (2025, 2, 15),
            (2025, 3, 30),
            (2025, 5, 15),
            (2025, 7, 1),
            (2025, 8, 15),
            (2025, 10, 1),
            (2025, 11, 15),
            (2025, 12, 31),
        ],
    )
    def test_saturn_position_de(self, year, month, day):
        """Test that saturn_position_de returns valid position vector"""
        epc = bh.Epoch.from_date(year, month, day, bh.TimeSystem.UTC)
        r_saturn = bh.saturn_position_de(epc, bh.EphemerisSource.DE440s)

        assert isinstance(r_saturn, np.ndarray)
        assert r_saturn.shape == (3,)

    @pytest.mark.parametrize(
        "year,month,day",
        [
            (2025, 1, 1),
            (2025, 2, 15),
            (2025, 3, 30),
            (2025, 5, 15),
            (2025, 7, 1),
            (2025, 8, 15),
            (2025, 10, 1),
            (2025, 11, 15),
            (2025, 12, 31),
        ],
    )
    def test_uranus_position_de(self, year, month, day):
        """Test that uranus_position_de returns valid position vector"""
        epc = bh.Epoch.from_date(year, month, day, bh.TimeSystem.UTC)
        r_uranus = bh.uranus_position_de(epc, bh.EphemerisSource.DE440s)

        assert isinstance(r_uranus, np.ndarray)
        assert r_uranus.shape == (3,)

    @pytest.mark.parametrize(
        "year,month,day",
        [
            (2025, 1, 1),
            (2025, 2, 15),
            (2025, 3, 30),
            (2025, 5, 15),
            (2025, 7, 1),
            (2025, 8, 15),
            (2025, 10, 1),
            (2025, 11, 15),
            (2025, 12, 31),
        ],
    )
    def test_neptune_position_de(self, year, month, day):
        """Test that neptune_position_de returns valid position vector"""
        epc = bh.Epoch.from_date(year, month, day, bh.TimeSystem.UTC)
        r_neptune = bh.neptune_position_de(epc, bh.EphemerisSource.DE440s)

        assert isinstance(r_neptune, np.ndarray)
        assert r_neptune.shape == (3,)

    @pytest.mark.parametrize(
        "year,month,day",
        [
            (2025, 1, 1),
            (2025, 2, 15),
            (2025, 3, 30),
            (2025, 5, 15),
            (2025, 7, 1),
            (2025, 8, 15),
            (2025, 10, 1),
            (2025, 11, 15),
            (2025, 12, 31),
        ],
    )
    def test_solar_system_barycenter_position_de(self, year, month, day):
        """Test that solar_system_barycenter_position_de returns valid position vector"""
        epc = bh.Epoch.from_date(year, month, day, bh.TimeSystem.UTC)
        r_ssb = bh.solar_system_barycenter_position_de(epc, bh.EphemerisSource.DE440s)

        assert isinstance(r_ssb, np.ndarray)
        assert r_ssb.shape == (3,)

    @pytest.mark.parametrize(
        "year,month,day",
        [
            (2025, 1, 1),
            (2025, 2, 15),
            (2025, 3, 30),
            (2025, 5, 15),
            (2025, 7, 1),
            (2025, 8, 15),
            (2025, 10, 1),
            (2025, 11, 15),
            (2025, 12, 31),
        ],
    )
    def test_ssb_position_de(self, year, month, day):
        """Test that ssb_position_de (alias) returns valid position vector"""
        epc = bh.Epoch.from_date(year, month, day, bh.TimeSystem.UTC)
        r_ssb = bh.ssb_position_de(epc, bh.EphemerisSource.DE440s)

        assert isinstance(r_ssb, np.ndarray)
        assert r_ssb.shape == (3,)

    @pytest.mark.order(1)
    def test_de_performance_benefit(self):
        """Test that subsequent calls don't re-load the kernel (performance test)"""
        import time

        epc = bh.Epoch.from_date(2024, 2, 25, bh.TimeSystem.UTC)

        # First call (kernel should already be loaded from fixture)
        start = time.perf_counter()
        _ = bh.sun_position_de(epc, bh.EphemerisSource.DE440s)
        first_call_time = time.perf_counter() - start

        # Subsequent calls should be much faster (no kernel loading)
        start = time.perf_counter()
        for _ in range(10):
            _ = bh.sun_position_de(epc, bh.EphemerisSource.DE440s)
        avg_time = (time.perf_counter() - start) / 10

        # Average time should be similar to or faster than first call
        # (not a strict requirement, just documenting expected behavior)
        assert avg_time < first_call_time
