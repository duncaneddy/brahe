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


def test_caching_provider_mjd_min():
    """Test getting minimum MJD from caching provider"""
    sw = brahe.CachingSpaceWeatherProvider(
        max_age_seconds=7 * 86400, auto_refresh=False, extrapolate="Hold"
    )

    # First data point is 1957-10-01 (MJD 36112)
    assert sw.mjd_min() == 36112.0


def test_caching_provider_mjd_max():
    """Test getting maximum MJD from caching provider"""
    sw = brahe.CachingSpaceWeatherProvider(
        max_age_seconds=7 * 86400, auto_refresh=False, extrapolate="Hold"
    )

    # Should have data through recent dates
    assert sw.mjd_max() >= 60000.0


def test_caching_provider_mjd_last_observed():
    """Test getting last observed MJD from caching provider"""
    sw = brahe.CachingSpaceWeatherProvider(
        max_age_seconds=7 * 86400, auto_refresh=False, extrapolate="Hold"
    )

    # Should have recent observed data
    assert sw.mjd_last_observed() >= 60000.0


def test_caching_provider_mjd_last_daily_predicted():
    """Test getting last MJD with daily predicted data from caching provider"""
    sw = brahe.CachingSpaceWeatherProvider(
        max_age_seconds=7 * 86400, auto_refresh=False, extrapolate="Hold"
    )

    mjd_last_daily_predicted = sw.mjd_last_daily_predicted()
    # Should be greater than or equal to last observed
    assert mjd_last_daily_predicted >= sw.mjd_last_observed()
    # Should be reasonable value (after 2020)
    assert mjd_last_daily_predicted > 58849.0


def test_caching_provider_mjd_last_monthly_predicted():
    """Test getting last MJD with monthly predicted data from caching provider"""
    sw = brahe.CachingSpaceWeatherProvider(
        max_age_seconds=7 * 86400, auto_refresh=False, extrapolate="Hold"
    )

    mjd_last_monthly_predicted = sw.mjd_last_monthly_predicted()
    # Should be greater than or equal to daily predicted
    assert mjd_last_monthly_predicted >= sw.mjd_last_daily_predicted()
    # Should be reasonable value (after 2020)
    assert mjd_last_monthly_predicted > 58849.0


def test_caching_provider_get_f107_adj_avg81():
    """Test getting 81-day average adjusted F10.7 from caching provider"""
    sw = brahe.CachingSpaceWeatherProvider(
        max_age_seconds=7 * 86400, auto_refresh=False, extrapolate="Hold"
    )

    mjd = 60000.0
    f107_adj_avg = sw.get_f107_adj_avg81(mjd)
    # Should be positive
    assert f107_adj_avg > 0.0
    # Should be within typical solar flux range
    assert 50.0 < f107_adj_avg < 400.0


def test_caching_provider_get_sunspot_number():
    """Test getting sunspot number from caching provider"""
    sw = brahe.CachingSpaceWeatherProvider(
        max_age_seconds=7 * 86400, auto_refresh=False, extrapolate="Hold"
    )

    mjd = 60000.0
    isn = sw.get_sunspot_number(mjd)
    # Sunspot number is non-negative
    assert isn >= 0
    # Should be within reasonable range (historical max is around 250)
    assert isn < 500


def test_caching_provider_get_last_kp():
    """Test getting last N 3-hourly Kp values from caching provider"""
    sw = brahe.CachingSpaceWeatherProvider(
        max_age_seconds=7 * 86400, auto_refresh=False, extrapolate="Hold"
    )

    mjd = 60000.0
    kp_values = sw.get_last_kp(mjd, 5)
    assert len(kp_values) == 5

    # All values should be valid Kp (0-9)
    for kp in kp_values:
        assert 0.0 <= kp <= 9.0


def test_caching_provider_get_last_ap():
    """Test getting last N 3-hourly Ap values from caching provider"""
    sw = brahe.CachingSpaceWeatherProvider(
        max_age_seconds=7 * 86400, auto_refresh=False, extrapolate="Hold"
    )

    mjd = 60000.0
    ap_values = sw.get_last_ap(mjd, 5)
    assert len(ap_values) == 5

    # All values should be non-negative
    for ap in ap_values:
        assert ap >= 0.0


def test_caching_provider_get_last_daily_kp():
    """Test getting last N daily average Kp values from caching provider"""
    sw = brahe.CachingSpaceWeatherProvider(
        max_age_seconds=7 * 86400, auto_refresh=False, extrapolate="Hold"
    )

    mjd = 60000.0
    daily_kp = sw.get_last_daily_kp(mjd, 3)
    assert len(daily_kp) == 3

    # All values should be valid Kp
    for kp in daily_kp:
        assert 0.0 <= kp <= 9.0


def test_caching_provider_get_last_daily_ap():
    """Test getting last N daily average Ap values from caching provider"""
    sw = brahe.CachingSpaceWeatherProvider(
        max_age_seconds=7 * 86400, auto_refresh=False, extrapolate="Hold"
    )

    mjd = 60000.0
    daily_ap = sw.get_last_daily_ap(mjd, 3)
    assert len(daily_ap) == 3

    # All values should be non-negative
    for ap in daily_ap:
        assert ap >= 0.0


def test_caching_provider_get_last_f107():
    """Test getting last N daily F10.7 values from caching provider"""
    sw = brahe.CachingSpaceWeatherProvider(
        max_age_seconds=7 * 86400, auto_refresh=False, extrapolate="Hold"
    )

    mjd = 60000.0
    f107 = sw.get_last_f107(mjd, 3)
    assert len(f107) == 3

    # All values should be positive
    for val in f107:
        assert val > 0.0


def test_caching_provider_get_last_kpap_epochs():
    """Test getting epochs for last N 3-hourly Kp/Ap intervals from caching provider"""
    sw = brahe.CachingSpaceWeatherProvider(
        max_age_seconds=7 * 86400, auto_refresh=False, extrapolate="Hold"
    )

    mjd = 60000.0
    epochs = sw.get_last_kpap_epochs(mjd, 5)
    assert len(epochs) == 5

    # Verify they are Epoch objects
    for epoch in epochs:
        assert hasattr(epoch, "mjd")

    # Verify epochs are in ascending order (oldest first)
    for i in range(len(epochs) - 1):
        assert epochs[i].mjd() < epochs[i + 1].mjd()


def test_caching_provider_get_last_daily_epochs():
    """Test getting epochs for last N daily values from caching provider"""
    sw = brahe.CachingSpaceWeatherProvider(
        max_age_seconds=7 * 86400, auto_refresh=False, extrapolate="Hold"
    )

    mjd = 60000.0
    epochs = sw.get_last_daily_epochs(mjd, 3)
    assert len(epochs) == 3

    # Verify they are Epoch objects
    for epoch in epochs:
        assert hasattr(epoch, "mjd")

    # Verify epochs are in ascending order (oldest first)
    for i in range(len(epochs) - 1):
        assert epochs[i].mjd() < epochs[i + 1].mjd()


def test_caching_provider_with_url(tmp_path):
    """Test creating caching provider with custom URL (using pre-cached file)"""
    import shutil

    # Copy the test space weather file to the temp directory as if it was downloaded
    # This avoids making network calls during testing
    source_file = "./data/space_weather/sw19571001.txt"
    cache_dir = tmp_path / "sw_cache"
    cache_dir.mkdir()
    dest_file = cache_dir / "sw19571001.txt"
    shutil.copy(source_file, dest_file)

    # Create provider with a dummy URL - since file exists and is fresh, no download occurs
    sw = brahe.CachingSpaceWeatherProvider.with_url(
        url="https://example.com/sw19571001.txt",
        max_age_seconds=365 * 86400,  # 1 year max age
        auto_refresh=False,
        extrapolate="Hold",
        cache_dir=str(cache_dir),
    )

    # Verify provider was created successfully
    assert sw.is_initialized() is True
    assert sw.len() >= 24000
    assert sw.sw_type() == "CssiSpaceWeather"
    assert sw.extrapolation() == "Hold"

    # Verify data can be retrieved
    mjd = 60000.0
    kp = sw.get_kp(mjd)
    assert 0.0 <= kp <= 9.0

    ap = sw.get_ap_daily(mjd)
    assert ap >= 0.0

    f107 = sw.get_f107_observed(mjd)
    assert f107 > 0.0
