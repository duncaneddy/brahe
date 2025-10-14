import pytest
import sys
import brahe


def test_set_global_eop_from_static_zeros():
    eop = brahe.StaticEOPProvider.from_zero()

    brahe.set_global_eop_provider_from_static_provider(eop)

    assert brahe.get_global_eop_initialization() is True
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

    brahe.set_global_eop_provider_from_static_provider(eop)

    assert brahe.get_global_eop_initialization() is True
    assert eop.is_initialized() is True
    assert eop.len() == 1
    assert eop.eop_type() == "Static"
    assert eop.extrapolation() == "Hold"
    assert eop.interpolation() is False
    assert eop.mjd_min() == 0
    assert eop.mjd_max() == sys.float_info.max
    assert eop.mjd_last_lod() == sys.float_info.max
    assert eop.mjd_last_dxdy() == sys.float_info.max


def test_set_global_eop_from_default_c04_file():
    eop = brahe.FileEOPProvider.from_default_c04(True, "Hold")

    brahe.set_global_eop_provider_from_file_provider(eop)
    assert brahe.get_global_eop_initialization() is True


def test_set_global_eop_from_c04_file(iau2000_c04_20_filepath):
    eop = brahe.FileEOPProvider.from_c04_file(iau2000_c04_20_filepath, True, "Hold")

    brahe.set_global_eop_provider_from_file_provider(eop)
    assert brahe.get_global_eop_initialization() is True


def test_set_global_eop_from_standard_file(iau2000_standard_filepath):
    eop = brahe.FileEOPProvider.from_standard_file(
        iau2000_standard_filepath, True, "Hold"
    )

    brahe.set_global_eop_provider_from_file_provider(eop)
    assert brahe.get_global_eop_initialization() is True


def test_set_global_eop_from_default_standard_file():
    eop = brahe.FileEOPProvider.from_default_standard(True, "Hold")

    brahe.set_global_eop_provider_from_file_provider(eop)
    assert brahe.get_global_eop_initialization() is True


def test_get_ut1_utc(iau2000_standard_filepath):
    eop = brahe.FileEOPProvider.from_standard_file(
        iau2000_standard_filepath, True, "Hold"
    )
    brahe.set_global_eop_provider_from_file_provider(eop)

    # Test getting exact point in table
    assert brahe.get_global_ut1_utc(59569.0) == -0.1079939

    # Test interpolating within table
    assert brahe.get_global_ut1_utc(59569.5) == (-0.1079939 + -0.1075984) / 2.0

    # Test extrapolation hold
    assert brahe.get_global_ut1_utc(99999.0) == 0.0420038

    # Test extrapolation zero
    eop = brahe.FileEOPProvider.from_standard_file(
        iau2000_standard_filepath, True, "Zero"
    )
    brahe.set_global_eop_provider_from_file_provider(eop)
    assert brahe.get_global_ut1_utc(99999.0) == 0.0


def test_get_pm_xy(iau2000_standard_filepath):
    eop = brahe.FileEOPProvider.from_standard_file(
        iau2000_standard_filepath, True, "Hold"
    )
    brahe.set_global_eop_provider_from_file_provider(eop)

    # Test getting exact point in table
    pm_x, pm_y = brahe.get_global_pm(59569.0)
    assert pm_x == 0.075382 * brahe.AS2RAD
    assert pm_y == 0.263451 * brahe.AS2RAD

    # Test interpolating within table
    pm_x, pm_y = brahe.get_global_pm(59569.5)
    assert pm_x == (0.075382 * brahe.AS2RAD + 0.073157 * brahe.AS2RAD) / 2.0
    assert pm_y == (0.263451 * brahe.AS2RAD + 0.264273 * brahe.AS2RAD) / 2.0

    # Test extrapolation hold
    pm_x, pm_y = brahe.get_global_pm(99999.0)
    assert pm_x == 0.173369 * brahe.AS2RAD
    assert pm_y == 0.266914 * brahe.AS2RAD

    # Test extrapolation zero
    eop = brahe.FileEOPProvider.from_standard_file(
        iau2000_standard_filepath, True, "Zero"
    )
    brahe.set_global_eop_provider_from_file_provider(eop)
    pm_x, pm_y = brahe.get_global_pm(99999.0)
    assert pm_x == 0.0
    assert pm_y == 0.0


def test_get_dxdy(iau2000_standard_filepath):
    eop = brahe.FileEOPProvider.from_standard_file(
        iau2000_standard_filepath, True, "Hold"
    )
    brahe.set_global_eop_provider_from_file_provider(eop)

    # Test getting exact point in table
    dX, dY = brahe.get_global_dxdy(59569.0)
    assert dX == pytest.approx(0.265 * brahe.AS2RAD * 1.0e-3, abs=1e-12)
    assert dY == pytest.approx(-0.067 * brahe.AS2RAD * 1.0e-3, abs=1e-12)

    # Test interpolating within table
    dX, dY = brahe.get_global_dxdy(59569.5)
    assert dX == pytest.approx(
        (0.265 * brahe.AS2RAD + 0.268 * brahe.AS2RAD) / 2.0 * 1.0e-3, abs=1e-12
    )
    assert dY == pytest.approx(
        (-0.067 * brahe.AS2RAD + -0.067 * brahe.AS2RAD) / 2.0 * 1.0e-3, abs=1e-12
    )

    # Test extrapolation hold
    dX, dY = brahe.get_global_dxdy(99999.0)
    assert dX == pytest.approx(0.006 * brahe.AS2RAD * 1.0e-3, abs=1e-12)
    assert dY == pytest.approx(-0.118 * brahe.AS2RAD * 1.0e-3, abs=1e-12)

    # Test extrapolation zero
    eop = brahe.FileEOPProvider.from_standard_file(
        iau2000_standard_filepath, True, "Zero"
    )
    brahe.set_global_eop_provider_from_file_provider(eop)
    dX, dY = brahe.get_global_dxdy(99999.0)
    assert dX == 0.0
    assert dY == 0.0


def test_get_lod(iau2000_standard_filepath):
    eop = brahe.FileEOPProvider.from_standard_file(
        iau2000_standard_filepath, True, "Hold"
    )
    brahe.set_global_eop_provider_from_file_provider(eop)

    # Test getting exact point in table
    assert brahe.get_global_lod(59569.0) == -0.3999 * 1.0e-3

    # Test interpolating within table
    assert brahe.get_global_lod(59569.5) == (-0.3999 + -0.3604) / 2.0 * 1.0e-3

    # Test extrapolation hold
    assert brahe.get_global_eop_extrapolation() == "Hold"
    assert brahe.get_global_lod(99999.0) == 0.7706 * 1.0e-3

    # Test extrapolation zero
    eop = brahe.FileEOPProvider.from_standard_file(
        iau2000_standard_filepath, True, "Zero"
    )
    brahe.set_global_eop_provider_from_file_provider(eop)
    assert brahe.get_global_lod(99999.0) == 0.0
