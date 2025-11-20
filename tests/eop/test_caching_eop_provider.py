"""Tests for CachingEOPProvider Python bindings."""

import pytest
import tempfile
import os
import time
import brahe


@pytest.mark.ci
def test_caching_provider_with_explicit_filepath():
    """Test CachingEOPProvider with explicitly provided filepath."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_eop.txt")

        # Create provider with explicit filepath
        provider = brahe.CachingEOPProvider(
            eop_type="StandardBulletinA",
            max_age_seconds=7 * 86400,
            auto_refresh=False,
            interpolate=True,
            extrapolate="Hold",
            filepath=filepath,
        )

        # Verify file was created
        assert os.path.exists(filepath)

        # Verify provider is initialized
        assert provider.is_initialized()
        assert provider.eop_type() == "StandardBulletinA"
        assert provider.extrapolation() == "Hold"
        assert provider.interpolation() is True
        assert provider.len() > 0


@pytest.mark.ci
def test_caching_provider_with_default_filepath():
    """Test CachingEOPProvider with default cache filepath."""
    provider = brahe.CachingEOPProvider(
        eop_type="StandardBulletinA",
        max_age_seconds=7 * 86400,
        auto_refresh=False,
        interpolate=True,
        extrapolate="Hold",
    )

    # Verify provider is initialized
    assert provider.is_initialized()
    assert provider.eop_type() == "StandardBulletinA"
    assert provider.len() > 0

    # Verify file was created in cache directory
    cache_dir = brahe.get_brahe_cache_dir()
    expected_path = os.path.join(cache_dir, "eop", "finals.all.iau2000.txt")
    assert os.path.exists(expected_path)


@pytest.mark.ci
def test_caching_provider_c04_default_filepath():
    """Test CachingEOPProvider with default C04 filepath."""
    provider = brahe.CachingEOPProvider(
        eop_type="C04",
        max_age_seconds=7 * 86400,
        auto_refresh=False,
        interpolate=True,
        extrapolate="Hold",
    )

    # Verify provider is initialized
    assert provider.is_initialized()
    assert provider.eop_type() == "C04"
    assert provider.len() > 0

    # Verify file was created in cache directory
    cache_dir = brahe.get_brahe_cache_dir()
    expected_path = os.path.join(cache_dir, "eop", "EOP_20_C04_one_file_1962-now.txt")
    assert os.path.exists(expected_path)


def test_caching_provider_with_existing_file(iau2000_standard_filepath):
    """Test CachingEOPProvider with existing file (no network needed)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy test file to temporary location
        import shutil

        dest_path = os.path.join(tmpdir, "test_eop.txt")
        shutil.copy(iau2000_standard_filepath, dest_path)

        # Create provider with large max age (file should be used as-is)
        provider = brahe.CachingEOPProvider(
            eop_type="StandardBulletinA",
            max_age_seconds=100 * 365 * 86400,  # 1 year
            auto_refresh=False,
            interpolate=True,
            extrapolate="Hold",
            filepath=dest_path,
        )

        assert provider.is_initialized()
        assert provider.eop_type() == "StandardBulletinA"
        assert provider.len() > 0


def test_caching_provider_file_age(iau2000_standard_filepath):
    """Test file_age() method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        import shutil

        dest_path = os.path.join(tmpdir, "test_eop.txt")
        shutil.copy(iau2000_standard_filepath, dest_path)

        provider = brahe.CachingEOPProvider(
            eop_type="StandardBulletinA",
            max_age_seconds=100 * 365 * 86400,
            auto_refresh=False,
            interpolate=True,
            extrapolate="Hold",
            filepath=dest_path,
        )

        # File was just loaded, age should be very small
        age = provider.file_age()
        assert age >= 0.0
        assert age < 2.0  # Should be less than 2 seconds

        # Wait a bit and check age increased
        time.sleep(0.1)
        age2 = provider.file_age()
        assert age2 > age


def test_caching_provider_file_epoch(iau2000_standard_filepath):
    """Test file_epoch() method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        import shutil

        dest_path = os.path.join(tmpdir, "test_eop.txt")
        shutil.copy(iau2000_standard_filepath, dest_path)

        provider = brahe.CachingEOPProvider(
            eop_type="StandardBulletinA",
            max_age_seconds=100 * 365 * 86400,
            auto_refresh=False,
            interpolate=True,
            extrapolate="Hold",
            filepath=dest_path,
        )

        # Get file epoch
        epoch = provider.file_epoch()
        assert epoch is not None
        assert epoch.time_system == brahe.TimeSystem.UTC

        # File should have been loaded very recently - check MJD is reasonable
        mjd = epoch.mjd()
        assert mjd > 60000.0  # After 2023


def test_caching_provider_refresh(iau2000_standard_filepath):
    """Test refresh() method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        import shutil

        dest_path = os.path.join(tmpdir, "test_eop_refresh.txt")
        shutil.copy(iau2000_standard_filepath, dest_path)

        provider = brahe.CachingEOPProvider(
            eop_type="StandardBulletinA",
            max_age_seconds=100 * 365 * 86400,
            auto_refresh=False,
            interpolate=True,
            extrapolate="Hold",
            filepath=dest_path,
        )

        original_len = provider.len()

        # Refresh should succeed (no download needed due to large max age)
        provider.refresh()

        # Length should be unchanged
        assert provider.len() == original_len


def test_caching_provider_eop_data_retrieval(iau2000_standard_filepath):
    """Test EOP data retrieval methods (EarthOrientationProvider trait)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        import shutil

        dest_path = os.path.join(tmpdir, "test_eop_delegation.txt")
        shutil.copy(iau2000_standard_filepath, dest_path)

        provider = brahe.CachingEOPProvider(
            eop_type="StandardBulletinA",
            max_age_seconds=100 * 365 * 86400,
            auto_refresh=False,
            interpolate=True,
            extrapolate="Hold",
            filepath=dest_path,
        )

        # Test basic properties
        assert provider.is_initialized()
        assert provider.eop_type() == "StandardBulletinA"
        assert provider.extrapolation() == "Hold"
        assert provider.interpolation() is True
        assert provider.mjd_min() == 41684.0
        assert provider.mjd_max() >= 60672.0

        # Test data retrieval
        mjd = 59569.0

        ut1_utc = provider.get_ut1_utc(mjd)
        assert ut1_utc == -0.1079939

        pm_x, pm_y = provider.get_pm(mjd)
        assert pm_x > 0.0
        assert pm_y > 0.0

        dx, dy = provider.get_dxdy(mjd)
        assert dx != 0.0 or dy != 0.0

        lod = provider.get_lod(mjd)
        assert lod != 0.0

        # Test get_eop
        # NOTE: Actual order is (pm_x, pm_y, ut1_utc, dx, dy, lod) despite docstring
        pm_x_eop, pm_y_eop, ut1_utc_eop, dx_eop, dy_eop, lod_eop = provider.get_eop(mjd)
        assert ut1_utc_eop == -0.1079939
        assert pm_x_eop > 0.0
        assert pm_y_eop > 0.0


def test_caching_provider_extrapolation_modes(iau2000_standard_filepath):
    """Test different extrapolation modes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        import shutil

        # Test Hold extrapolation
        dest_path = os.path.join(tmpdir, "test_eop_hold.txt")
        shutil.copy(iau2000_standard_filepath, dest_path)

        provider_hold = brahe.CachingEOPProvider(
            eop_type="StandardBulletinA",
            max_age_seconds=100 * 365 * 86400,
            auto_refresh=False,
            interpolate=True,
            extrapolate="Hold",
            filepath=dest_path,
        )

        mjd_future = 99999.0
        ut1_utc_hold = provider_hold.get_ut1_utc(mjd_future)
        assert ut1_utc_hold != 0.0  # Should hold last value

        # Test Zero extrapolation
        dest_path_zero = os.path.join(tmpdir, "test_eop_zero.txt")
        shutil.copy(iau2000_standard_filepath, dest_path_zero)

        provider_zero = brahe.CachingEOPProvider(
            eop_type="StandardBulletinA",
            max_age_seconds=100 * 365 * 86400,
            auto_refresh=False,
            interpolate=True,
            extrapolate="Zero",
            filepath=dest_path_zero,
        )

        ut1_utc_zero = provider_zero.get_ut1_utc(mjd_future)
        assert ut1_utc_zero == 0.0  # Should return zero


def test_caching_provider_interpolation(iau2000_standard_filepath):
    """Test interpolation behavior."""
    with tempfile.TemporaryDirectory() as tmpdir:
        import shutil

        # Create provider with interpolation enabled
        dest_path = os.path.join(tmpdir, "test_eop_interp.txt")
        shutil.copy(iau2000_standard_filepath, dest_path)

        provider = brahe.CachingEOPProvider(
            eop_type="StandardBulletinA",
            max_age_seconds=100 * 365 * 86400,
            auto_refresh=False,
            interpolate=True,
            extrapolate="Hold",
            filepath=dest_path,
        )

        # Test interpolating between data points
        mjd_between = 59569.5
        ut1_utc = provider.get_ut1_utc(mjd_between)

        # Should be interpolated value (between two known points)
        # Known values: 59569.0 = -0.1079939, 59570.0 = -0.1075984
        expected = (-0.1079939 + -0.1075984) / 2.0
        assert ut1_utc == pytest.approx(expected, abs=1e-10)


def test_caching_provider_unknown_type_error():
    """Test that creating provider with Unknown EOPType raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dest_path = os.path.join(tmpdir, "test_eop_unknown.txt")

        with pytest.raises(Exception):
            brahe.CachingEOPProvider(
                eop_type="Unknown",
                max_age_seconds=100 * 365 * 86400,
                auto_refresh=False,
                interpolate=True,
                extrapolate="Hold",
                filepath=dest_path,
            )


def test_caching_provider_mjd_last_lod(iau2000_standard_filepath):
    """Test mjd_last_lod() method delegation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        import shutil

        dest_path = os.path.join(tmpdir, "test_eop_last_lod.txt")
        shutil.copy(iau2000_standard_filepath, dest_path)

        provider = brahe.CachingEOPProvider(
            eop_type="StandardBulletinA",
            max_age_seconds=10 * 365 * 86400,  # 10 years - prevent download
            auto_refresh=False,
            interpolate=True,
            extrapolate="Hold",
            filepath=dest_path,
        )

        mjd_last_lod = provider.mjd_last_lod()
        assert mjd_last_lod == 60298.0


def test_caching_provider_mjd_last_dxdy(iau2000_standard_filepath):
    """Test mjd_last_dxdy() method delegation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        import shutil

        dest_path = os.path.join(tmpdir, "test_eop_last_dxdy.txt")
        shutil.copy(iau2000_standard_filepath, dest_path)

        provider = brahe.CachingEOPProvider(
            eop_type="StandardBulletinA",
            max_age_seconds=10 * 365 * 86400,  # 10 years - prevent download
            auto_refresh=False,
            interpolate=True,
            extrapolate="Hold",
            filepath=dest_path,
        )

        mjd_last_dxdy = provider.mjd_last_dxdy()
        assert mjd_last_dxdy == 60373.0
