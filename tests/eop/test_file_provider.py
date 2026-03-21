import brahe


def test_from_c04_file(iau2000_c04_20_filepath):
    eop = brahe.FileEOPProvider.from_c04_file(iau2000_c04_20_filepath, True, "Hold")

    assert eop.is_initialized() is True
    assert eop.len() == 22605
    assert eop.eop_type() == "C04"
    assert eop.extrapolation() == "Hold"
    assert eop.interpolation() is True
    assert eop.mjd_min() == 37665.0
    assert eop.mjd_max() == 60269.0
    assert eop.mjd_last_lod() == 60269.0
    assert eop.mjd_last_dxdy() == 60269.0


def test_from_default_c04():
    eop = brahe.FileEOPProvider.from_default_c04(False, "Zero")

    assert eop.is_initialized() is True
    assert eop.len() >= 22605
    assert eop.eop_type() == "C04"
    assert eop.extrapolation() == "Zero"
    assert eop.interpolation() is False
    assert eop.mjd_min() == 37665.0
    assert eop.mjd_max() >= 60269.0
    assert eop.mjd_last_lod() >= 60269.0
    assert eop.mjd_last_dxdy() >= 60269.0


def test_from_standard_file(iau2000_standard_filepath):
    eop = brahe.FileEOPProvider.from_standard_file(
        iau2000_standard_filepath, True, "Hold"
    )

    assert eop.is_initialized() is True
    assert eop.len() == 18989
    assert eop.eop_type() == "StandardBulletinA"
    assert eop.extrapolation() == "Hold"
    assert eop.interpolation() is True
    assert eop.mjd_min() == 41684.0
    assert eop.mjd_max() == 60672.0
    assert eop.mjd_last_lod() == 60298.0
    assert eop.mjd_last_dxdy() == 60373.0


def test_from_default_standard():
    eop = brahe.FileEOPProvider.from_default_standard(True, "Hold")

    assert eop.is_initialized() is True
    assert eop.len() >= 18989
    assert eop.eop_type() == "StandardBulletinA"
    assert eop.extrapolation() == "Hold"
    assert eop.interpolation() is True
    assert eop.mjd_min() == 41684.0
    assert eop.mjd_max() >= 60672.0
    assert eop.mjd_last_lod() >= 60298.0
    assert eop.mjd_last_dxdy() >= 60373.0


def test_extrapolate_before_min_zero(iau2000_standard_filepath):
    """Test extrapolation with Zero mode for mjd < mjd_min"""
    eop = brahe.FileEOPProvider.from_standard_file(
        iau2000_standard_filepath, True, "Zero"
    )
    mjd_before_min = 40000.0  # Before mjd_min = 41684.0

    # Test get_ut1_utc
    ut1_utc = eop.get_ut1_utc(mjd_before_min)
    assert ut1_utc == 0.0

    # Test get_pm
    pm_x, pm_y = eop.get_pm(mjd_before_min)
    assert pm_x == 0.0
    assert pm_y == 0.0

    # Test get_dxdy
    dx, dy = eop.get_dxdy(mjd_before_min)
    assert dx == 0.0
    assert dy == 0.0

    # Test get_lod
    lod = eop.get_lod(mjd_before_min)
    assert lod == 0.0

    # Test get_eop
    eop_data = eop.get_eop(mjd_before_min)
    pm_x, pm_y, ut1_utc, dx, dy, lod = eop_data
    assert pm_x == 0.0
    assert pm_y == 0.0
    assert ut1_utc == 0.0
    assert dx == 0.0
    assert dy == 0.0
    assert lod == 0.0


def test_extrapolate_before_min_hold(iau2000_standard_filepath):
    """Test extrapolation with Hold mode for mjd < mjd_min"""
    eop = brahe.FileEOPProvider.from_standard_file(
        iau2000_standard_filepath, True, "Hold"
    )
    mjd_before_min = 40000.0  # Before mjd_min = 41684.0

    # Get the first values in the table for comparison
    ut1_utc_first = eop.get_ut1_utc(41684.0)
    pm_x_first, pm_y_first = eop.get_pm(41684.0)
    dx_first, dy_first = eop.get_dxdy(41684.0)
    lod_first = eop.get_lod(41684.0)

    # Test get_ut1_utc holds first value
    ut1_utc = eop.get_ut1_utc(mjd_before_min)
    assert ut1_utc == ut1_utc_first

    # Test get_pm holds first value
    pm_x, pm_y = eop.get_pm(mjd_before_min)
    assert pm_x == pm_x_first
    assert pm_y == pm_y_first

    # Test get_dxdy holds first value
    dx, dy = eop.get_dxdy(mjd_before_min)
    assert dx == dx_first
    assert dy == dy_first

    # Test get_lod holds first value
    lod = eop.get_lod(mjd_before_min)
    assert lod == lod_first

    # Test get_eop holds first values
    eop_data = eop.get_eop(mjd_before_min)
    pm_x, pm_y, ut1_utc, dx, dy, lod = eop_data
    assert pm_x == pm_x_first
    assert pm_y == pm_y_first
    assert ut1_utc == ut1_utc_first
    assert dx == dx_first
    assert dy == dy_first
    assert lod == lod_first


def test_file_eop_provider_repr_str(iau2000_standard_filepath):
    """Test FileEOPProvider __repr__ and __str__ methods."""
    eop = brahe.FileEOPProvider.from_standard_file(
        iau2000_standard_filepath, True, "Hold"
    )
    repr_str = repr(eop)
    str_str = str(eop)

    assert len(repr_str) > 0
    assert len(str_str) > 0


def test_file_eop_provider_default_constructor():
    """Test FileEOPProvider default (uninitialized) constructor."""
    eop = brahe.FileEOPProvider()

    # Should not be initialized
    assert eop.is_initialized() is False


def test_file_eop_provider_from_file(iau2000_standard_filepath):
    """Test FileEOPProvider.from_file() auto-detection."""
    eop = brahe.FileEOPProvider.from_file(iau2000_standard_filepath, True, "Hold")

    assert eop.is_initialized() is True
    assert eop.len() > 0
    assert eop.eop_type() in ("StandardBulletinA", "C04")


def test_file_eop_provider_from_default_file():
    """Test FileEOPProvider.from_default_file() with eop_type."""
    eop = brahe.FileEOPProvider.from_default_file("StandardBulletinA", True, "Hold")

    assert eop.is_initialized() is True
    assert eop.eop_type() == "StandardBulletinA"
    assert eop.len() > 0


def test_file_eop_provider_data_retrieval(iau2000_standard_filepath):
    """Test FileEOPProvider data retrieval methods."""
    eop = brahe.FileEOPProvider.from_standard_file(
        iau2000_standard_filepath, True, "Hold"
    )

    mjd = 59569.0

    # Test individual getters
    ut1_utc = eop.get_ut1_utc(mjd)
    assert ut1_utc == -0.1079939

    pm_x, pm_y = eop.get_pm(mjd)
    assert pm_x > 0.0
    assert pm_y > 0.0

    dx, dy = eop.get_dxdy(mjd)
    assert dx != 0.0 or dy != 0.0

    lod = eop.get_lod(mjd)
    assert lod != 0.0

    # Test get_eop
    result = eop.get_eop(mjd)
    assert len(result) == 6
    pm_x_eop, pm_y_eop, ut1_utc_eop, dx_eop, dy_eop, lod_eop = result
    assert ut1_utc_eop == -0.1079939


def test_extrapolate_before_min_error(iau2000_standard_filepath):
    """Test extrapolation with Error mode for mjd < mjd_min"""
    import pytest

    eop = brahe.FileEOPProvider.from_standard_file(
        iau2000_standard_filepath, True, "Error"
    )
    mjd_before_min = 40000.0  # Before mjd_min = 41684.0

    # Test get_ut1_utc raises error
    with pytest.raises(Exception):
        eop.get_ut1_utc(mjd_before_min)

    # Test get_pm raises error
    with pytest.raises(Exception):
        eop.get_pm(mjd_before_min)

    # Test get_dxdy raises error
    with pytest.raises(Exception):
        eop.get_dxdy(mjd_before_min)

    # Test get_lod raises error
    with pytest.raises(Exception):
        eop.get_lod(mjd_before_min)

    # Test get_eop raises error
    with pytest.raises(Exception):
        eop.get_eop(mjd_before_min)
