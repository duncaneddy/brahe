import pytest
import brahe


def test_from_default_file():
    """Test loading from the default packaged space weather file"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()

    assert sw.is_initialized() is True
    assert sw.len() >= 24000  # Should have lots of data points
    assert sw.sw_type() == "CssiSpaceWeather"
    assert sw.extrapolation() == "Hold"
    # First data point is 1957-10-01 (MJD 36112)
    assert sw.mjd_min() == 36112.0
    # Should have data through recent dates
    assert sw.mjd_max() >= 60000.0
    assert sw.mjd_last_observed() >= 60000.0


def test_from_file(sw_test_filepath):
    """Test loading from a specific space weather file with exact assertions"""
    sw = brahe.FileSpaceWeatherProvider.from_file(sw_test_filepath, "Hold")

    assert sw.is_initialized() is True
    assert sw.sw_type() == "CssiSpaceWeather"
    assert sw.extrapolation() == "Hold"
    # First data point is 1957-10-01 (MJD 36112)
    assert sw.mjd_min() == 36112.0


def test_from_file_known_values(sw_test_filepath):
    """Test that known values from 1957-10-01 are read correctly"""
    sw = brahe.FileSpaceWeatherProvider.from_file(sw_test_filepath, "Hold")

    # MJD 36112.0 = 1957-10-01
    mjd = 36112.0

    # Known Kp values from first data line: 43 40 30 20 37 23 43 37
    # These are stored as (integer * 10 + fraction), where fraction 3=+1/3, 7=+2/3
    kp_all = sw.get_kp_all(mjd)
    assert len(kp_all) == 8
    assert kp_all[0] == pytest.approx(4.0 + 1.0 / 3.0, abs=1e-10)  # 43 -> 4+1/3
    assert kp_all[1] == pytest.approx(4.0, abs=1e-10)  # 40 -> 4.0
    assert kp_all[2] == pytest.approx(3.0, abs=1e-10)  # 30 -> 3.0
    assert kp_all[3] == pytest.approx(2.0, abs=1e-10)  # 20 -> 2.0
    assert kp_all[4] == pytest.approx(3.0 + 2.0 / 3.0, abs=1e-10)  # 37 -> 3+2/3
    assert kp_all[5] == pytest.approx(2.0 + 1.0 / 3.0, abs=1e-10)  # 23 -> 2+1/3
    assert kp_all[6] == pytest.approx(4.0 + 1.0 / 3.0, abs=1e-10)  # 43 -> 4+1/3
    assert kp_all[7] == pytest.approx(3.0 + 2.0 / 3.0, abs=1e-10)  # 37 -> 3+2/3

    # Known Ap values: 32 27 15 7 22 9 32 22
    ap_all = sw.get_ap_all(mjd)
    assert len(ap_all) == 8
    assert ap_all[0] == pytest.approx(32.0, abs=1e-10)
    assert ap_all[1] == pytest.approx(27.0, abs=1e-10)
    assert ap_all[2] == pytest.approx(15.0, abs=1e-10)
    assert ap_all[3] == pytest.approx(7.0, abs=1e-10)
    assert ap_all[4] == pytest.approx(22.0, abs=1e-10)
    assert ap_all[5] == pytest.approx(9.0, abs=1e-10)
    assert ap_all[6] == pytest.approx(32.0, abs=1e-10)
    assert ap_all[7] == pytest.approx(22.0, abs=1e-10)

    # Known Ap daily average: 21
    assert sw.get_ap_daily(mjd) == pytest.approx(21.0, abs=1e-10)

    # Known F10.7 adjusted: 269.8 (get_f107_observed returns adjusted value)
    assert sw.get_f107_observed(mjd) == pytest.approx(269.8, abs=1e-10)

    # Known International Sunspot Number: 334
    assert sw.get_sunspot_number(mjd) == 334


def test_from_file_kp_daily(sw_test_filepath):
    """Test daily Kp average calculation from known values"""
    sw = brahe.FileSpaceWeatherProvider.from_file(sw_test_filepath, "Hold")

    # MJD 36112.0 = 1957-10-01
    kp_daily = sw.get_kp_daily(36112.0)
    # The daily average uses the Kp sum from the file (273)
    # 273 / 10 / 8 = 27.3 / 8 = 3.4125
    expected = 273.0 / 10.0 / 8.0
    assert kp_daily == pytest.approx(expected, abs=1e-10)


def test_file_provider_get_kp():
    """Test getting Kp values from file provider"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()

    # Test MJD 60000.0 (2023-03-21)
    mjd = 60000.0
    kp = sw.get_kp(mjd)
    assert 0.0 <= kp <= 9.0  # Valid Kp range

    kp_all = sw.get_kp_all(mjd)
    assert len(kp_all) == 8
    for val in kp_all:
        assert 0.0 <= val <= 9.0

    kp_daily = sw.get_kp_daily(mjd)
    assert 0.0 <= kp_daily <= 9.0


def test_file_provider_get_ap():
    """Test getting Ap values from file provider"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()

    mjd = 60000.0
    ap = sw.get_ap(mjd)
    assert ap >= 0.0  # Ap is non-negative

    ap_all = sw.get_ap_all(mjd)
    assert len(ap_all) == 8
    for val in ap_all:
        assert val >= 0.0

    ap_daily = sw.get_ap_daily(mjd)
    assert ap_daily >= 0.0


def test_file_provider_get_f107():
    """Test getting F10.7 values from file provider"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()

    mjd = 60000.0
    f107_obs = sw.get_f107_observed(mjd)
    assert f107_obs > 0.0  # F10.7 is always positive

    f107_adj = sw.get_f107_adjusted(mjd)
    assert f107_adj >= 0.0

    f107_avg = sw.get_f107_obs_avg81(mjd)
    assert f107_avg > 0.0


def test_file_provider_get_sunspot():
    """Test getting sunspot number from file provider"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()

    mjd = 60000.0
    isn = sw.get_sunspot_number(mjd)
    assert isn >= 0  # Sunspot number is non-negative


def test_file_provider_historical_data():
    """Test accessing historical data from 1957"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()

    # First data point: 1957-10-01 (MJD 36112)
    mjd = 36112.0
    kp = sw.get_kp(mjd)
    assert 0.0 <= kp <= 9.0

    ap = sw.get_ap_daily(mjd)
    assert ap >= 0.0

    f107 = sw.get_f107_observed(mjd)
    assert f107 > 0.0


def test_file_provider_extrapolation():
    """Test extrapolation behavior"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()

    # Request data far in the future - should extrapolate (Hold)
    future_mjd = 80000.0
    kp = sw.get_kp(future_mjd)
    assert 0.0 <= kp <= 9.0  # Should hold last value


def test_file_provider_uninitialized():
    """Test uninitialized file provider"""
    sw = brahe.FileSpaceWeatherProvider()

    assert sw.is_initialized() is False


def test_file_provider_get_last_kp():
    """Test getting last N 3-hourly Kp values"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()
    base_mjd = 60000.0

    # Get last 5 values
    kp_values = sw.get_last_kp(base_mjd, 5)
    assert len(kp_values) == 5

    # All values should be valid Kp
    for kp in kp_values:
        assert 0.0 <= kp <= 9.0

    # Verify values are quantized to 1/3 increments
    for kp in kp_values:
        kp_times_3 = kp * 3.0
        assert pytest.approx(kp_times_3, abs=1e-10) == round(kp_times_3)


def test_file_provider_get_last_ap():
    """Test getting last N 3-hourly Ap values"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()
    base_mjd = 60000.0

    # Get last 5 values
    ap_values = sw.get_last_ap(base_mjd, 5)
    assert len(ap_values) == 5

    # All values should be non-negative
    for ap in ap_values:
        assert ap >= 0.0


def test_file_provider_get_last_kp_boundary():
    """Test that get_last_kp returns correct values at 3-hour boundaries"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()
    base_mjd = 60000.0

    # Get all 8 Kp values for the day
    kp_all = sw.get_kp_all(base_mjd)

    # At 00:00 (MJD.0), get_last_kp(1) should return first interval
    kp_00 = sw.get_last_kp(base_mjd, 1)
    assert pytest.approx(kp_00[0], abs=1e-10) == kp_all[0]

    # At 03:00 (MJD + 0.125), should return second interval
    kp_03 = sw.get_last_kp(base_mjd + 0.125, 1)
    assert pytest.approx(kp_03[0], abs=1e-10) == kp_all[1]

    # At 12:00 (MJD + 0.5), should return fifth interval
    kp_12 = sw.get_last_kp(base_mjd + 0.5, 1)
    assert pytest.approx(kp_12[0], abs=1e-10) == kp_all[4]


def test_file_provider_get_last_daily_kp():
    """Test getting last N daily average Kp values"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()
    base_mjd = 60000.0

    # Get last 3 daily averages
    daily_kp = sw.get_last_daily_kp(base_mjd, 3)
    assert len(daily_kp) == 3

    # Verify each matches get_kp_daily for that day
    assert pytest.approx(daily_kp[0], abs=1e-10) == sw.get_kp_daily(base_mjd - 2.0)
    assert pytest.approx(daily_kp[1], abs=1e-10) == sw.get_kp_daily(base_mjd - 1.0)
    assert pytest.approx(daily_kp[2], abs=1e-10) == sw.get_kp_daily(base_mjd)


def test_file_provider_get_last_daily_ap():
    """Test getting last N daily average Ap values"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()
    base_mjd = 60000.0

    # Get last 3 daily averages
    daily_ap = sw.get_last_daily_ap(base_mjd, 3)
    assert len(daily_ap) == 3

    # Verify each matches get_ap_daily for that day
    assert pytest.approx(daily_ap[0], abs=1e-10) == sw.get_ap_daily(base_mjd - 2.0)
    assert pytest.approx(daily_ap[1], abs=1e-10) == sw.get_ap_daily(base_mjd - 1.0)
    assert pytest.approx(daily_ap[2], abs=1e-10) == sw.get_ap_daily(base_mjd)


def test_file_provider_get_last_f107():
    """Test getting last N daily F10.7 values"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()
    base_mjd = 60000.0

    # Get last 3 daily F10.7 values
    f107 = sw.get_last_f107(base_mjd, 3)
    assert len(f107) == 3

    # Verify each matches get_f107_observed for that day
    assert pytest.approx(f107[0], abs=1e-10) == sw.get_f107_observed(base_mjd - 2.0)
    assert pytest.approx(f107[1], abs=1e-10) == sw.get_f107_observed(base_mjd - 1.0)
    assert pytest.approx(f107[2], abs=1e-10) == sw.get_f107_observed(base_mjd)


def test_file_provider_get_last_kp_crosses_day_boundary():
    """Test that get_last_kp correctly crosses day boundaries"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()
    base_mjd = 60000.0

    # Get last 10 values starting from 03:00 (second interval)
    # Should get 2 from today and 8 from yesterday
    values = sw.get_last_kp(base_mjd + 0.125, 10)
    assert len(values) == 10

    # Verify last two match today's first two intervals
    kp_all_today = sw.get_kp_all(base_mjd)
    assert pytest.approx(values[8], abs=1e-10) == kp_all_today[0]
    assert pytest.approx(values[9], abs=1e-10) == kp_all_today[1]

    # Verify first values match yesterday's intervals
    kp_all_yesterday = sw.get_kp_all(base_mjd - 1.0)
    assert pytest.approx(values[0], abs=1e-10) == kp_all_yesterday[0]
    assert pytest.approx(values[7], abs=1e-10) == kp_all_yesterday[7]


def test_file_provider_get_last_kpap_epochs():
    """Test getting epochs for last N 3-hourly Kp/Ap intervals"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()
    base_mjd = 60000.0

    # Get last 5 epochs
    epochs = sw.get_last_kpap_epochs(base_mjd, 5)
    assert len(epochs) == 5

    # Verify they are Epoch objects
    for epoch in epochs:
        assert hasattr(epoch, "mjd")

    # Verify epochs are in ascending order (oldest first)
    for i in range(len(epochs) - 1):
        assert epochs[i].mjd() < epochs[i + 1].mjd()

    # Last epoch should be for 00:00 on base_mjd day (first interval)
    assert pytest.approx(epochs[4].mjd(), abs=1e-10) == base_mjd

    # Epochs should be 3 hours apart (0.125 days)
    for i in range(len(epochs) - 1):
        diff = epochs[i + 1].mjd() - epochs[i].mjd()
        assert pytest.approx(diff, abs=1e-10) == 0.125


def test_file_provider_get_last_daily_epochs():
    """Test getting epochs for last N daily values"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()
    base_mjd = 60000.0

    # Get last 3 daily epochs
    epochs = sw.get_last_daily_epochs(base_mjd, 3)
    assert len(epochs) == 3

    # Verify they are Epoch objects
    for epoch in epochs:
        assert hasattr(epoch, "mjd")

    # Verify epochs are in ascending order (oldest first)
    for i in range(len(epochs) - 1):
        assert epochs[i].mjd() < epochs[i + 1].mjd()

    # Epochs should be at 00:00 UT for each day
    assert pytest.approx(epochs[0].mjd(), abs=1e-10) == base_mjd - 2.0
    assert pytest.approx(epochs[1].mjd(), abs=1e-10) == base_mjd - 1.0
    assert pytest.approx(epochs[2].mjd(), abs=1e-10) == base_mjd


def test_file_provider_get_last_kpap_epochs_crosses_day():
    """Test that get_last_kpap_epochs correctly crosses day boundaries"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()
    base_mjd = 60000.0

    # At 03:00 (second interval), get last 10 epochs
    # Should get 2 from today and 8 from yesterday
    epochs = sw.get_last_kpap_epochs(base_mjd + 0.125, 10)
    assert len(epochs) == 10

    # First epoch should be at 00:00 yesterday
    assert pytest.approx(epochs[0].mjd(), abs=1e-10) == base_mjd - 1.0

    # Last two should be at 00:00 and 03:00 today
    assert pytest.approx(epochs[8].mjd(), abs=1e-10) == base_mjd
    assert pytest.approx(epochs[9].mjd(), abs=1e-10) == base_mjd + 0.125


def test_file_provider_epoch_kp_alignment():
    """Test that epochs align with Kp values"""
    sw = brahe.FileSpaceWeatherProvider.from_default_file()
    base_mjd = 60000.0

    # Get both epochs and Kp values
    epochs = sw.get_last_kpap_epochs(base_mjd, 5)
    kp_values = sw.get_last_kp(base_mjd, 5)

    # Verify we can use the epoch to retrieve the same Kp value
    for epoch, expected_kp in zip(epochs, kp_values):
        retrieved_kp = sw.get_kp(epoch.mjd())
        assert pytest.approx(retrieved_kp, abs=1e-10) == expected_kp
