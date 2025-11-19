import brahe


def test_caching_provider_default():
    """Test creating a caching provider with default cache location"""
    sw = brahe.CachingSpaceWeatherProvider(
        max_age_seconds=7 * 86400,  # 7 days
        auto_refresh=False,
        extrapolate="Hold",
    )

    assert sw.is_initialized() is True
    assert sw.len() >= 24000
    assert sw.sw_type() == "CssiSpaceWeather"
    assert sw.extrapolation() == "Hold"


def test_caching_provider_file_epoch():
    """Test getting file epoch from caching provider"""
    sw = brahe.CachingSpaceWeatherProvider(
        max_age_seconds=7 * 86400, auto_refresh=False, extrapolate="Hold"
    )

    epoch = sw.file_epoch()
    # Should return an Epoch object
    assert hasattr(epoch, "mjd")
    # File epoch should be reasonable (after 2020)
    assert epoch.mjd() > 58849.0  # 2020-01-01


def test_caching_provider_file_age():
    """Test getting file age from caching provider"""
    sw = brahe.CachingSpaceWeatherProvider(
        max_age_seconds=7 * 86400, auto_refresh=False, extrapolate="Hold"
    )

    age = sw.file_age()
    # Age should be non-negative
    assert age >= 0.0
    # Should be less than a year old (for a working system)
    assert age < 365 * 86400


def test_caching_provider_get_kp():
    """Test getting Kp values from caching provider"""
    sw = brahe.CachingSpaceWeatherProvider(
        max_age_seconds=7 * 86400, auto_refresh=False, extrapolate="Hold"
    )

    mjd = 60000.0
    kp = sw.get_kp(mjd)
    assert 0.0 <= kp <= 9.0

    kp_all = sw.get_kp_all(mjd)
    assert len(kp_all) == 8

    kp_daily = sw.get_kp_daily(mjd)
    assert 0.0 <= kp_daily <= 9.0


def test_caching_provider_get_ap():
    """Test getting Ap values from caching provider"""
    sw = brahe.CachingSpaceWeatherProvider(
        max_age_seconds=7 * 86400, auto_refresh=False, extrapolate="Hold"
    )

    mjd = 60000.0
    ap = sw.get_ap(mjd)
    assert ap >= 0.0

    ap_all = sw.get_ap_all(mjd)
    assert len(ap_all) == 8

    ap_daily = sw.get_ap_daily(mjd)
    assert ap_daily >= 0.0


def test_caching_provider_get_f107():
    """Test getting F10.7 values from caching provider"""
    sw = brahe.CachingSpaceWeatherProvider(
        max_age_seconds=7 * 86400, auto_refresh=False, extrapolate="Hold"
    )

    mjd = 60000.0
    f107_obs = sw.get_f107_observed(mjd)
    assert f107_obs > 0.0

    f107_adj = sw.get_f107_adjusted(mjd)
    assert f107_adj >= 0.0

    f107_avg81 = sw.get_f107_obs_avg81(mjd)
    assert f107_avg81 > 0.0


def test_caching_provider_refresh():
    """Test manual refresh of caching provider"""
    sw = brahe.CachingSpaceWeatherProvider(
        max_age_seconds=7 * 86400, auto_refresh=False, extrapolate="Hold"
    )

    # Should not raise an error
    sw.refresh()

    # Data should still be valid
    assert sw.is_initialized() is True
    assert sw.len() >= 24000


def test_caching_provider_extrapolation_modes():
    """Test different extrapolation modes"""
    # Hold mode
    sw_hold = brahe.CachingSpaceWeatherProvider(
        max_age_seconds=7 * 86400, auto_refresh=False, extrapolate="Hold"
    )
    assert sw_hold.extrapolation() == "Hold"

    # Zero mode
    sw_zero = brahe.CachingSpaceWeatherProvider(
        max_age_seconds=7 * 86400, auto_refresh=False, extrapolate="Zero"
    )
    assert sw_zero.extrapolation() == "Zero"
