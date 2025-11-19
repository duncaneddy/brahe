import sys
import pytest
import brahe


def test_set_global_sw_from_test_file(sw_test_filepath):
    """Test setting global provider from test file with exact assertions"""
    sw = brahe.FileSpaceWeatherProvider.from_file(sw_test_filepath, "Hold")
    brahe.set_global_space_weather_provider(sw)

    assert brahe.get_global_sw_initialization() is True
    assert brahe.get_global_sw_type() == "CssiSpaceWeather"
    assert brahe.get_global_sw_extrapolation() == "Hold"
    # First data point is 1957-10-01 (MJD 36112)
    assert brahe.get_global_sw_mjd_min() == 36112.0


def test_get_global_known_values(sw_test_filepath):
    """Test getting known values from global provider using test file"""
    sw = brahe.FileSpaceWeatherProvider.from_file(sw_test_filepath, "Hold")
    brahe.set_global_space_weather_provider(sw)

    # MJD 36112.0 = 1957-10-01
    mjd = 36112.0

    # Known Kp value at 00:00 should be 4+1/3 (from raw value 43)
    assert brahe.get_global_kp(mjd) == pytest.approx(4.0 + 1.0 / 3.0, abs=1e-10)

    # Known Ap daily average: 21
    assert brahe.get_global_ap_daily(mjd) == pytest.approx(21.0, abs=1e-10)

    # Known F10.7 adjusted: 269.8
    assert brahe.get_global_f107_observed(mjd) == pytest.approx(269.8, abs=1e-10)

    # Known International Sunspot Number: 334
    assert brahe.get_global_sunspot_number(mjd) == 334


def test_get_global_kp_all_known_values(sw_test_filepath):
    """Test getting all Kp values from global provider using test file"""
    sw = brahe.FileSpaceWeatherProvider.from_file(sw_test_filepath, "Hold")
    brahe.set_global_space_weather_provider(sw)

    # MJD 36112.0 = 1957-10-01
    # Kp values use 1/3 increments: 43->4+1/3, 40->4.0, 30->3.0, etc.
    kp_all = brahe.get_global_kp_all(36112.0)
    assert len(kp_all) == 8
    assert kp_all[0] == pytest.approx(4.0 + 1.0 / 3.0, abs=1e-10)  # 43
    assert kp_all[1] == pytest.approx(4.0, abs=1e-10)  # 40
    assert kp_all[2] == pytest.approx(3.0, abs=1e-10)  # 30
    assert kp_all[3] == pytest.approx(2.0, abs=1e-10)  # 20
    assert kp_all[4] == pytest.approx(3.0 + 2.0 / 3.0, abs=1e-10)  # 37
    assert kp_all[5] == pytest.approx(2.0 + 1.0 / 3.0, abs=1e-10)  # 23
    assert kp_all[6] == pytest.approx(4.0 + 1.0 / 3.0, abs=1e-10)  # 43
    assert kp_all[7] == pytest.approx(3.0 + 2.0 / 3.0, abs=1e-10)  # 37


def test_get_global_ap_all_known_values(sw_test_filepath):
    """Test getting all Ap values from global provider using test file"""
    sw = brahe.FileSpaceWeatherProvider.from_file(sw_test_filepath, "Hold")
    brahe.set_global_space_weather_provider(sw)

    # MJD 36112.0 = 1957-10-01
    ap_all = brahe.get_global_ap_all(36112.0)
    assert len(ap_all) == 8
    assert ap_all[0] == pytest.approx(32.0, abs=1e-10)
    assert ap_all[1] == pytest.approx(27.0, abs=1e-10)
    assert ap_all[2] == pytest.approx(15.0, abs=1e-10)
    assert ap_all[3] == pytest.approx(7.0, abs=1e-10)
    assert ap_all[4] == pytest.approx(22.0, abs=1e-10)
    assert ap_all[5] == pytest.approx(9.0, abs=1e-10)
    assert ap_all[6] == pytest.approx(32.0, abs=1e-10)
    assert ap_all[7] == pytest.approx(22.0, abs=1e-10)


def test_set_global_sw_from_static_zeros():
    """Test setting global provider with static zeros"""
    sw = brahe.StaticSpaceWeatherProvider.from_zero()

    brahe.set_global_space_weather_provider(sw)

    assert brahe.get_global_sw_initialization() is True
    assert brahe.get_global_sw_len() == 1
    assert brahe.get_global_sw_type() == "Static"
    assert brahe.get_global_sw_extrapolation() == "Hold"
    assert brahe.get_global_sw_mjd_min() == 0
    assert brahe.get_global_sw_mjd_max() == sys.float_info.max


def test_set_global_sw_from_static_values():
    """Test setting global provider with static values"""
    sw = brahe.StaticSpaceWeatherProvider.from_values(
        kp=3.0, ap=15.0, f107=150.0, f107a=145.0, s=100
    )

    brahe.set_global_space_weather_provider(sw)

    assert brahe.get_global_sw_initialization() is True
    assert brahe.get_global_sw_len() == 1
    assert brahe.get_global_sw_type() == "Static"


def test_set_global_sw_from_default_file():
    """Test setting global provider from default file"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()

    brahe.set_global_space_weather_provider(sw)
    assert brahe.get_global_sw_initialization() is True
    assert brahe.get_global_sw_type() == "CssiSpaceWeather"


def test_set_global_sw_from_caching_provider():
    """Test setting global provider from caching provider"""
    sw = brahe.CachingSpaceWeatherProvider(
        max_age_seconds=7 * 86400, auto_refresh=False, extrapolate="Hold"
    )

    brahe.set_global_space_weather_provider(sw)
    assert brahe.get_global_sw_initialization() is True
    assert brahe.get_global_sw_type() == "CssiSpaceWeather"


def test_get_global_kp():
    """Test getting Kp from global provider"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()
    brahe.set_global_space_weather_provider(sw)

    mjd = 60000.0
    kp = brahe.get_global_kp(mjd)
    assert 0.0 <= kp <= 9.0


def test_get_global_kp_all():
    """Test getting all Kp values from global provider"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()
    brahe.set_global_space_weather_provider(sw)

    mjd = 60000.0
    kp_all = brahe.get_global_kp_all(mjd)
    assert len(kp_all) == 8
    for val in kp_all:
        assert 0.0 <= val <= 9.0


def test_get_global_kp_daily():
    """Test getting daily Kp from global provider"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()
    brahe.set_global_space_weather_provider(sw)

    mjd = 60000.0
    kp_daily = brahe.get_global_kp_daily(mjd)
    assert 0.0 <= kp_daily <= 9.0


def test_get_global_ap():
    """Test getting Ap from global provider"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()
    brahe.set_global_space_weather_provider(sw)

    mjd = 60000.0
    ap = brahe.get_global_ap(mjd)
    assert ap >= 0.0


def test_get_global_ap_all():
    """Test getting all Ap values from global provider"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()
    brahe.set_global_space_weather_provider(sw)

    mjd = 60000.0
    ap_all = brahe.get_global_ap_all(mjd)
    assert len(ap_all) == 8
    for val in ap_all:
        assert val >= 0.0


def test_get_global_ap_daily():
    """Test getting daily Ap from global provider"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()
    brahe.set_global_space_weather_provider(sw)

    mjd = 60000.0
    ap_daily = brahe.get_global_ap_daily(mjd)
    assert ap_daily >= 0.0


def test_get_global_f107_observed():
    """Test getting observed F10.7 from global provider"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()
    brahe.set_global_space_weather_provider(sw)

    mjd = 60000.0
    f107 = brahe.get_global_f107_observed(mjd)
    assert f107 > 0.0


def test_get_global_f107_adjusted():
    """Test getting adjusted F10.7 from global provider"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()
    brahe.set_global_space_weather_provider(sw)

    mjd = 60000.0
    f107 = brahe.get_global_f107_adjusted(mjd)
    assert f107 >= 0.0


def test_get_global_f107_avg81():
    """Test getting 81-day average F10.7 from global provider"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()
    brahe.set_global_space_weather_provider(sw)

    mjd = 60000.0
    f107_obs_avg = brahe.get_global_f107_obs_avg81(mjd)
    assert f107_obs_avg > 0.0

    f107_adj_avg = brahe.get_global_f107_adj_avg81(mjd)
    assert f107_adj_avg > 0.0


def test_get_global_sunspot_number():
    """Test getting sunspot number from global provider"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()
    brahe.set_global_space_weather_provider(sw)

    mjd = 60000.0
    isn = brahe.get_global_sunspot_number(mjd)
    assert isn >= 0


def test_get_global_sw_info():
    """Test getting global provider info"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()
    brahe.set_global_space_weather_provider(sw)

    assert brahe.get_global_sw_initialization() is True
    assert brahe.get_global_sw_len() >= 24000
    assert brahe.get_global_sw_type() == "CssiSpaceWeather"
    assert brahe.get_global_sw_extrapolation() == "Hold"
    assert brahe.get_global_sw_mjd_min() == 36112.0
    assert brahe.get_global_sw_mjd_max() >= 60000.0
    assert brahe.get_global_sw_mjd_last_observed() >= 60000.0


def test_initialize_sw():
    """Test the convenience initialization function"""
    brahe.initialize_sw()

    assert brahe.get_global_sw_initialization() is True
    assert brahe.get_global_sw_type() == "CssiSpaceWeather"
    assert brahe.get_global_sw_len() >= 24000


def test_global_sw_with_static_values():
    """Test that static values are returned correctly from global provider"""
    sw = brahe.StaticSpaceWeatherProvider.from_values(
        kp=5.0, ap=25.0, f107=180.0, f107a=175.0, s=120
    )
    brahe.set_global_space_weather_provider(sw)

    mjd = 60000.0
    assert brahe.get_global_kp(mjd) == 5.0
    assert brahe.get_global_ap_daily(mjd) == 25.0
    assert brahe.get_global_f107_observed(mjd) == 180.0
    assert brahe.get_global_sunspot_number(mjd) == 120


def test_get_global_last_kp():
    """Test getting last N 3-hourly Kp values from global provider"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()
    brahe.set_global_space_weather_provider(sw)

    mjd = 60000.0
    kp_values = brahe.get_global_last_kp(mjd, 5)
    assert len(kp_values) == 5

    # All values should be valid Kp
    for kp in kp_values:
        assert 0.0 <= kp <= 9.0


def test_get_global_last_ap():
    """Test getting last N 3-hourly Ap values from global provider"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()
    brahe.set_global_space_weather_provider(sw)

    mjd = 60000.0
    ap_values = brahe.get_global_last_ap(mjd, 5)
    assert len(ap_values) == 5

    # All values should be non-negative
    for ap in ap_values:
        assert ap >= 0.0


def test_get_global_last_daily_kp():
    """Test getting last N daily average Kp values from global provider"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()
    brahe.set_global_space_weather_provider(sw)

    mjd = 60000.0
    daily_kp = brahe.get_global_last_daily_kp(mjd, 3)
    assert len(daily_kp) == 3

    # Verify values match direct call
    assert pytest.approx(daily_kp[2], abs=1e-10) == brahe.get_global_kp_daily(mjd)


def test_get_global_last_daily_ap():
    """Test getting last N daily average Ap values from global provider"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()
    brahe.set_global_space_weather_provider(sw)

    mjd = 60000.0
    daily_ap = brahe.get_global_last_daily_ap(mjd, 3)
    assert len(daily_ap) == 3

    # Verify values match direct call
    assert pytest.approx(daily_ap[2], abs=1e-10) == brahe.get_global_ap_daily(mjd)


def test_get_global_last_f107():
    """Test getting last N daily F10.7 values from global provider"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()
    brahe.set_global_space_weather_provider(sw)

    mjd = 60000.0
    f107 = brahe.get_global_last_f107(mjd, 3)
    assert len(f107) == 3

    # Verify values match direct call
    assert pytest.approx(f107[2], abs=1e-10) == brahe.get_global_f107_observed(mjd)


def test_get_global_last_kpap_epochs():
    """Test getting epochs for last N 3-hourly Kp/Ap intervals from global provider"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()
    brahe.set_global_space_weather_provider(sw)

    mjd = 60000.0
    epochs = brahe.get_global_last_kpap_epochs(mjd, 5)
    assert len(epochs) == 5

    # Verify they are Epoch objects
    for epoch in epochs:
        assert hasattr(epoch, "mjd")

    # Verify epochs are in ascending order (oldest first)
    for i in range(len(epochs) - 1):
        assert epochs[i].mjd() < epochs[i + 1].mjd()

    # Last epoch should be for 00:00 on base_mjd day
    assert pytest.approx(epochs[4].mjd(), abs=1e-10) == mjd

    # Epochs should be 3 hours apart (0.125 days)
    for i in range(len(epochs) - 1):
        diff = epochs[i + 1].mjd() - epochs[i].mjd()
        assert pytest.approx(diff, abs=1e-10) == 0.125


def test_get_global_last_daily_epochs():
    """Test getting epochs for last N daily values from global provider"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()
    brahe.set_global_space_weather_provider(sw)

    mjd = 60000.0
    epochs = brahe.get_global_last_daily_epochs(mjd, 3)
    assert len(epochs) == 3

    # Verify they are Epoch objects
    for epoch in epochs:
        assert hasattr(epoch, "mjd")

    # Verify epochs are in ascending order (oldest first)
    for i in range(len(epochs) - 1):
        assert epochs[i].mjd() < epochs[i + 1].mjd()

    # Epochs should be at 00:00 UT for each day
    assert pytest.approx(epochs[0].mjd(), abs=1e-10) == mjd - 2.0
    assert pytest.approx(epochs[1].mjd(), abs=1e-10) == mjd - 1.0
    assert pytest.approx(epochs[2].mjd(), abs=1e-10) == mjd


def test_get_global_epoch_kp_alignment():
    """Test that global epochs align with Kp values"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()
    brahe.set_global_space_weather_provider(sw)

    mjd = 60000.0

    # Get both epochs and Kp values
    epochs = brahe.get_global_last_kpap_epochs(mjd, 5)
    kp_values = brahe.get_global_last_kp(mjd, 5)

    # Verify we can use the epoch to retrieve the same Kp value
    for epoch, expected_kp in zip(epochs, kp_values):
        retrieved_kp = brahe.get_global_kp(epoch.mjd())
        assert pytest.approx(retrieved_kp, abs=1e-10) == expected_kp
