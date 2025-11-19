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


def test_static_sw_mjd_last_daily_predicted():
    """Test getting last MJD with daily predicted data from static provider"""
    sw = brahe.StaticSpaceWeatherProvider.from_values(
        kp=3.0, ap=15.0, f107=150.0, f107a=145.0, s=100
    )

    # Static provider returns max float for all MJD boundaries
    assert sw.mjd_last_daily_predicted() == sys.float_info.max


def test_static_sw_mjd_last_monthly_predicted():
    """Test getting last MJD with monthly predicted data from static provider"""
    sw = brahe.StaticSpaceWeatherProvider.from_values(
        kp=3.0, ap=15.0, f107=150.0, f107a=145.0, s=100
    )

    # Static provider returns max float for all MJD boundaries
    assert sw.mjd_last_monthly_predicted() == sys.float_info.max


def test_static_sw_get_f107_adj_avg81():
    """Test getting 81-day average adjusted F10.7 from static provider"""
    sw = brahe.StaticSpaceWeatherProvider.from_values(
        kp=3.0, ap=15.0, f107=150.0, f107a=145.0, s=100
    )

    mjd = 60000.0
    # Static provider returns the f107a value for adjusted average
    f107_adj_avg = sw.get_f107_adj_avg81(mjd)
    assert f107_adj_avg == 145.0


def test_static_sw_get_last_kp():
    """Test getting last N 3-hourly Kp values from static provider"""
    sw = brahe.StaticSpaceWeatherProvider.from_values(
        kp=3.0, ap=15.0, f107=150.0, f107a=145.0, s=100
    )

    mjd = 60000.0
    # Get last 5 values - all should be the same for static provider
    kp_values = sw.get_last_kp(mjd, 5)
    assert len(kp_values) == 5
    for kp in kp_values:
        assert kp == 3.0


def test_static_sw_get_last_ap():
    """Test getting last N 3-hourly Ap values from static provider"""
    sw = brahe.StaticSpaceWeatherProvider.from_values(
        kp=3.0, ap=15.0, f107=150.0, f107a=145.0, s=100
    )

    mjd = 60000.0
    # Get last 5 values - all should be the same for static provider
    ap_values = sw.get_last_ap(mjd, 5)
    assert len(ap_values) == 5
    for ap in ap_values:
        assert ap == 15.0


def test_static_sw_get_last_daily_kp():
    """Test getting last N daily average Kp values from static provider"""
    sw = brahe.StaticSpaceWeatherProvider.from_values(
        kp=3.0, ap=15.0, f107=150.0, f107a=145.0, s=100
    )

    mjd = 60000.0
    # Get last 3 daily values
    daily_kp = sw.get_last_daily_kp(mjd, 3)
    assert len(daily_kp) == 3
    for kp in daily_kp:
        assert kp == 3.0


def test_static_sw_get_last_daily_ap():
    """Test getting last N daily average Ap values from static provider"""
    sw = brahe.StaticSpaceWeatherProvider.from_values(
        kp=3.0, ap=15.0, f107=150.0, f107a=145.0, s=100
    )

    mjd = 60000.0
    # Get last 3 daily values
    daily_ap = sw.get_last_daily_ap(mjd, 3)
    assert len(daily_ap) == 3
    for ap in daily_ap:
        assert ap == 15.0


def test_static_sw_get_last_f107():
    """Test getting last N daily F10.7 values from static provider"""
    sw = brahe.StaticSpaceWeatherProvider.from_values(
        kp=3.0, ap=15.0, f107=150.0, f107a=145.0, s=100
    )

    mjd = 60000.0
    # Get last 3 daily values
    f107 = sw.get_last_f107(mjd, 3)
    assert len(f107) == 3
    for val in f107:
        assert val == 150.0


def test_static_sw_get_last_kpap_epochs():
    """Test getting epochs for last N 3-hourly Kp/Ap intervals from static provider"""
    sw = brahe.StaticSpaceWeatherProvider.from_values(
        kp=3.0, ap=15.0, f107=150.0, f107a=145.0, s=100
    )

    mjd = 60000.0
    # Get last 5 epochs
    epochs = sw.get_last_kpap_epochs(mjd, 5)
    assert len(epochs) == 5

    # Verify they are Epoch objects
    for epoch in epochs:
        assert hasattr(epoch, "mjd")

    # Verify epochs are in ascending order (oldest first)
    for i in range(len(epochs) - 1):
        assert epochs[i].mjd() < epochs[i + 1].mjd()


def test_static_sw_get_last_daily_epochs():
    """Test getting epochs for last N daily values from static provider"""
    sw = brahe.StaticSpaceWeatherProvider.from_values(
        kp=3.0, ap=15.0, f107=150.0, f107a=145.0, s=100
    )

    mjd = 60000.0
    # Get last 3 daily epochs
    epochs = sw.get_last_daily_epochs(mjd, 3)
    assert len(epochs) == 3

    # Verify they are Epoch objects
    for epoch in epochs:
        assert hasattr(epoch, "mjd")

    # Verify epochs are in ascending order (oldest first)
    for i in range(len(epochs) - 1):
        assert epochs[i].mjd() < epochs[i + 1].mjd()
