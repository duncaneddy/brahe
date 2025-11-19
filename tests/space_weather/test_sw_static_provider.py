import sys
import brahe


def test_static_sw_from_zero():
    sw = brahe.StaticSpaceWeatherProvider.from_zero()

    assert sw.is_initialized() is True
    assert sw.len() == 1
    assert sw.sw_type() == "Static"
    assert sw.extrapolation() == "Hold"
    assert sw.mjd_min() == 0
    assert sw.mjd_max() == sys.float_info.max
    assert sw.mjd_last_observed() == sys.float_info.max


def test_static_sw_from_values():
    sw = brahe.StaticSpaceWeatherProvider.from_values(
        kp=3.0, ap=15.0, f107=150.0, f107a=145.0, s=100
    )

    assert sw.is_initialized() is True
    assert sw.len() == 1
    assert sw.sw_type() == "Static"
    assert sw.extrapolation() == "Hold"
    assert sw.mjd_min() == 0
    assert sw.mjd_max() == sys.float_info.max
    assert sw.mjd_last_observed() == sys.float_info.max


def test_static_sw_provider_len():
    """Test StaticSpaceWeatherProvider len() method"""
    sw_zero = brahe.StaticSpaceWeatherProvider.from_zero()
    assert sw_zero.len() == 1

    sw_values = brahe.StaticSpaceWeatherProvider.from_values(
        kp=3.0, ap=15.0, f107=150.0, f107a=145.0, s=100
    )
    assert sw_values.len() == 1


def test_static_sw_get_kp():
    """Test getting Kp values from static provider"""
    sw = brahe.StaticSpaceWeatherProvider.from_values(
        kp=3.0, ap=15.0, f107=150.0, f107a=145.0, s=100
    )

    # Should return same value for any MJD
    mjd = 60000.0
    kp = sw.get_kp(mjd)
    assert kp == 3.0

    # All 8 Kp values should be the same
    kp_all = sw.get_kp_all(mjd)
    assert len(kp_all) == 8
    for val in kp_all:
        assert val == 3.0

    # Daily average should also be the same
    kp_daily = sw.get_kp_daily(mjd)
    assert kp_daily == 3.0


def test_static_sw_get_ap():
    """Test getting Ap values from static provider"""
    sw = brahe.StaticSpaceWeatherProvider.from_values(
        kp=3.0, ap=15.0, f107=150.0, f107a=145.0, s=100
    )

    mjd = 60000.0
    ap = sw.get_ap(mjd)
    assert ap == 15.0

    ap_all = sw.get_ap_all(mjd)
    assert len(ap_all) == 8
    for val in ap_all:
        assert val == 15.0

    ap_daily = sw.get_ap_daily(mjd)
    assert ap_daily == 15.0


def test_static_sw_get_f107():
    """Test getting F10.7 values from static provider"""
    sw = brahe.StaticSpaceWeatherProvider.from_values(
        kp=3.0, ap=15.0, f107=150.0, f107a=145.0, s=100
    )

    mjd = 60000.0
    f107_obs = sw.get_f107_observed(mjd)
    assert f107_obs == 150.0

    f107_adj = sw.get_f107_adjusted(mjd)
    assert f107_adj == 145.0  # Returns f107a value

    f107_avg = sw.get_f107_obs_avg81(mjd)
    assert f107_avg == 150.0  # Observed average uses f107 value


def test_static_sw_get_sunspot():
    """Test getting sunspot number from static provider"""
    sw = brahe.StaticSpaceWeatherProvider.from_values(
        kp=3.0, ap=15.0, f107=150.0, f107a=145.0, s=100
    )

    mjd = 60000.0
    isn = sw.get_sunspot_number(mjd)
    assert isn == 100


def test_static_sw_uninitialized():
    """Test uninitialized static provider"""
    sw = brahe.StaticSpaceWeatherProvider()

    # Should not be initialized
    assert sw.is_initialized() is False
