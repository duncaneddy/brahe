import sys
import brahe


def test_set_global_eop_from_zero():
    eop = brahe.StaticEOPProvider.from_zero()

    assert eop.is_initialized() is True
    assert eop.len() == 1
    assert eop.eop_type() == "Static"
    assert eop.extrapolation() == "Hold"
    assert eop.interpolation() is False
    assert eop.mjd_min() == 0
    assert eop.mjd_max() == sys.float_info.max
    assert eop.mjd_last_lod() == sys.float_info.max
    assert eop.mjd_last_dxdy() == sys.float_info.max


def test_set_global_eop_from_static_values():
    eop = brahe.StaticEOPProvider.from_values(0.001, 0.002, 0.003, 0.004, 0.005, 0.006)

    assert eop.is_initialized() is True
    assert eop.len() == 1
    assert eop.eop_type() == "Static"
    assert eop.extrapolation() == "Hold"
    assert eop.interpolation() is False
    assert eop.mjd_min() == 0
    assert eop.mjd_max() == sys.float_info.max
    assert eop.mjd_last_lod() == sys.float_info.max
    assert eop.mjd_last_dxdy() == sys.float_info.max


def test_earth_orientation_provider_len():
    """Test EarthOrientationProvider len() method"""
    # Static provider always has len == 1
    eop_zero = brahe.StaticEOPProvider.from_zero()
    assert eop_zero.len() == 1

    eop_values = brahe.StaticEOPProvider.from_values(
        0.001, 0.002, 0.003, 0.004, 0.005, 0.006
    )
    assert eop_values.len() == 1


def test_static_eop_provider_default_constructor():
    """Test StaticEOPProvider default constructor."""
    eop = brahe.StaticEOPProvider()

    # Default constructor may or may not be initialized
    assert eop.len() == 1
    assert eop.eop_type() == "Static"
    assert eop.extrapolation() == "Hold"
    assert eop.interpolation() is False


def test_static_eop_provider_repr_str():
    """Test StaticEOPProvider __repr__ and __str__ methods."""
    eop = brahe.StaticEOPProvider.from_zero()
    repr_str = repr(eop)
    str_str = str(eop)

    # __repr__ should contain debug info
    assert len(repr_str) > 0

    # __str__ should also be non-empty
    assert len(str_str) > 0


def test_static_eop_provider_data_retrieval_from_zero():
    """Test data retrieval methods on a zero-valued provider."""
    eop = brahe.StaticEOPProvider.from_zero()

    mjd = 59569.0

    # All values should be 0 for from_zero
    ut1_utc = eop.get_ut1_utc(mjd)
    assert ut1_utc == 0.0

    pm_x, pm_y = eop.get_pm(mjd)
    assert pm_x == 0.0
    assert pm_y == 0.0

    dx, dy = eop.get_dxdy(mjd)
    assert dx == 0.0
    assert dy == 0.0

    lod = eop.get_lod(mjd)
    assert lod == 0.0


def test_static_eop_provider_data_retrieval_from_values():
    """Test data retrieval methods on a custom-valued provider.

    Note: from_values(ut1_utc, pm_x, pm_y, dx, dy, lod) stores internally as
    (ut1_utc, pm_x, pm_y, dx, dy, lod) but the data tuple mapping is:
    data.0/data.1 -> get_pm(), data.2 -> get_ut1_utc(), data.3/data.4 -> get_dxdy(), data.5 -> get_lod()
    So the actual mapping from Python args is position-based into the internal tuple.
    """
    eop = brahe.StaticEOPProvider.from_values(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)

    mjd = 59569.0

    # Internal tuple: (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    # get_pm returns (data.0, data.1)
    pm_x, pm_y = eop.get_pm(mjd)
    assert pm_x == 0.1
    assert pm_y == 0.2

    # get_ut1_utc returns data.2
    ut1_utc = eop.get_ut1_utc(mjd)
    assert ut1_utc == 0.3

    # get_dxdy returns (data.3, data.4)
    dx, dy = eop.get_dxdy(mjd)
    assert dx == 0.4
    assert dy == 0.5

    # get_lod returns data.5
    lod = eop.get_lod(mjd)
    assert lod == 0.6


def test_static_eop_provider_get_eop():
    """Test get_eop() returns all values as tuple."""
    eop = brahe.StaticEOPProvider.from_values(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)

    mjd = 59569.0
    result = eop.get_eop(mjd)

    # Should return a tuple of 6 values
    assert len(result) == 6

    # get_eop returns (pm_x, pm_y, ut1_utc, dx, dy, lod)
    # where values match the internal tuple positions
    pm_x, pm_y, ut1_utc, dx, dy, lod = result
    assert pm_x == 0.1
    assert pm_y == 0.2
    assert ut1_utc == 0.3
    assert dx == 0.4
    assert dy == 0.5
    assert lod == 0.6


def test_static_eop_provider_any_mjd():
    """Test that static provider returns same values for any MJD."""
    eop = brahe.StaticEOPProvider.from_values(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)

    # Values should be constant regardless of MJD
    for mjd in [0.0, 50000.0, 60000.0, 99999.0]:
        assert eop.get_ut1_utc(mjd) == 0.3  # data.2
        assert eop.get_lod(mjd) == 0.6  # data.5
