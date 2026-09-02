import pytest

import brahe


def test_datetime_to_jd():
    assert brahe.datetime_to_jd(2000, 1, 1, 12, 0, 0.0, 0.0) == 2451545.0


def test_datetime_to_mjd():
    assert brahe.datetime_to_mjd(2000, 1, 1, 12, 0, 0.0, 0.0) == 51544.5


def test_jd_to_datetime():
    assert brahe.jd_to_datetime(2451545.0) == (2000, 1, 1, 12, 0, 0.0, 0.0)


def test_mjd_to_datetime():
    assert brahe.mjd_to_datetime(51544.5) == (2000, 1, 1, 12, 0, 0.0, 0.0)


def test_time_system_offset_tai_utc_before_1972(eop):
    """Mirror of test_time_system_offset_tai_utc_before_1972 in Rust."""
    # Before 1972 UTC ran at a rate offset from TAI rather than a whole number
    # of seconds. The IERS entry in force from 1968 February 1 is
    # TAI - UTC = 4.2131700 + (MJD - 39126) * 0.0025920 seconds, so at
    # 1971-12-31T23:59:59 UTC (MJD 41316.99998843) it is 9.892242 s.
    fd = 86399.0 / 86400.0
    expected = 4.2131700 + (41316.0 + fd - 39126.0) * 0.0025920
    assert expected == pytest.approx(9.892242, abs=1e-6)

    jd = 2441316.5 + fd
    utc_to_tai = brahe.time_system_offset_for_jd(jd, brahe.UTC, brahe.TAI)
    assert utc_to_tai == pytest.approx(expected, abs=1e-6)

    tai_to_utc = brahe.time_system_offset_for_jd(
        jd + utc_to_tai / 86400.0, brahe.TAI, brahe.UTC
    )
    assert tai_to_utc == pytest.approx(-expected, abs=1e-6)


def test_time_system_offset_tai_utc_within_a_leap_second(eop):
    """Mirror of test_time_system_offset_tai_utc_within_a_leap_second in Rust."""
    # 2016-12-31T23:59:60 UTC is TAI 2017-01-01T00:00:36, so TAI - UTC is still
    # 36 s throughout the leap second and steps to 37 s after it.
    for tai_seconds_into_day, expected in [
        (86435.0, -36.0),
        (86436.0, -36.0),
        (86436.5, -36.0),
        (86437.0, -37.0),
    ]:
        jd = 2457753.5 + tai_seconds_into_day / 86400.0
        offset = brahe.time_system_offset_for_jd(jd, brahe.TAI, brahe.UTC)
        assert offset == expected, f"TAI second {tai_seconds_into_day}"


def test_time_system_offset_for_jd(eop):  # Test date
    jd = brahe.datetime_to_jd(2018, 6, 1, 0, 0, 0.0, 0.0)

    # UTC - TAI offset
    dutc = -37.0
    dut1 = 0.0769966

    # GPS
    assert brahe.time_system_offset_for_jd(jd, brahe.GPS, brahe.GPS) == 0.0
    assert brahe.time_system_offset_for_jd(jd, brahe.GPS, brahe.TT) == brahe.TT_GPS
    assert (
        brahe.time_system_offset_for_jd(jd, brahe.GPS, brahe.UTC)
        == dutc + brahe.TAI_GPS
    )
    assert brahe.time_system_offset_for_jd(jd, brahe.GPS, brahe.UT1) == pytest.approx(
        dutc + brahe.TAI_GPS + dut1, abs=1e-6
    )
    assert brahe.time_system_offset_for_jd(jd, brahe.GPS, brahe.TAI) == brahe.TAI_GPS

    # TT
    assert brahe.time_system_offset_for_jd(jd, brahe.TT, brahe.GPS) == brahe.GPS_TT
    assert brahe.time_system_offset_for_jd(jd, brahe.TT, brahe.TT) == 0.0
    assert (
        brahe.time_system_offset_for_jd(jd, brahe.TT, brahe.UTC) == dutc + brahe.TAI_TT
    )
    assert brahe.time_system_offset_for_jd(jd, brahe.TT, brahe.UT1) == pytest.approx(
        dutc + brahe.TAI_TT + dut1, abs=1e-6
    )
    assert brahe.time_system_offset_for_jd(jd, brahe.TT, brahe.TAI) == brahe.TAI_TT

    # UTC
    assert (
        brahe.time_system_offset_for_jd(jd, brahe.UTC, brahe.GPS)
        == -dutc + brahe.GPS_TAI
    )
    assert (
        brahe.time_system_offset_for_jd(jd, brahe.UTC, brahe.TT) == -dutc + brahe.TT_TAI
    )
    assert brahe.time_system_offset_for_jd(jd, brahe.UTC, brahe.UTC) == 0.0
    assert brahe.time_system_offset_for_jd(jd, brahe.UTC, brahe.UT1) == pytest.approx(
        dut1, abs=1e-6
    )
    assert brahe.time_system_offset_for_jd(jd, brahe.UTC, brahe.TAI) == -dutc

    # UT1
    assert brahe.time_system_offset_for_jd(jd, brahe.UT1, brahe.GPS) == pytest.approx(
        -dutc + brahe.GPS_TAI - dut1, abs=1e-6
    )
    assert brahe.time_system_offset_for_jd(jd, brahe.UT1, brahe.TT) == pytest.approx(
        -dutc + brahe.TT_TAI - dut1, abs=1e-6
    )
    assert brahe.time_system_offset_for_jd(jd, brahe.UT1, brahe.UTC) == pytest.approx(
        -dut1, abs=1e-6
    )
    assert brahe.time_system_offset_for_jd(jd, brahe.UT1, brahe.UT1) == pytest.approx(
        0.0, abs=1e-6
    )
    assert brahe.time_system_offset_for_jd(jd, brahe.UT1, brahe.TAI) == pytest.approx(
        -dutc - dut1, abs=1e-6
    )

    # TAI
    assert brahe.time_system_offset_for_jd(jd, brahe.TAI, brahe.GPS) == brahe.GPS_TAI
    assert brahe.time_system_offset_for_jd(jd, brahe.TAI, brahe.TT) == brahe.TT_TAI
    assert brahe.time_system_offset_for_jd(jd, brahe.TAI, brahe.UTC) == dutc
    assert brahe.time_system_offset_for_jd(jd, brahe.TAI, brahe.UT1) == pytest.approx(
        dutc + dut1, abs=1e-6
    )
    assert brahe.time_system_offset_for_jd(jd, brahe.TAI, brahe.TAI) == 0.0


def test_time_system_offset_for_mjd(eop):  # Test date
    mjd = brahe.datetime_to_mjd(2018, 6, 1, 0, 0, 0.0, 0.0)

    # UTC - TAI offset
    dutc = -37.0
    dut1 = 0.0769966

    # GPS
    assert brahe.time_system_offset_for_mjd(mjd, brahe.GPS, brahe.GPS) == 0.0
    assert brahe.time_system_offset_for_mjd(mjd, brahe.GPS, brahe.TT) == brahe.TT_GPS
    assert (
        brahe.time_system_offset_for_mjd(mjd, brahe.GPS, brahe.UTC)
        == dutc + brahe.TAI_GPS
    )
    assert brahe.time_system_offset_for_mjd(mjd, brahe.GPS, brahe.UT1) == pytest.approx(
        dutc + brahe.TAI_GPS + dut1, abs=1e-6
    )
    assert brahe.time_system_offset_for_mjd(mjd, brahe.GPS, brahe.TAI) == brahe.TAI_GPS

    # TT
    assert brahe.time_system_offset_for_mjd(mjd, brahe.TT, brahe.GPS) == brahe.GPS_TT
    assert brahe.time_system_offset_for_mjd(mjd, brahe.TT, brahe.TT) == 0.0
    assert (
        brahe.time_system_offset_for_mjd(mjd, brahe.TT, brahe.UTC)
        == dutc + brahe.TAI_TT
    )
    assert brahe.time_system_offset_for_mjd(mjd, brahe.TT, brahe.UT1) == pytest.approx(
        dutc + brahe.TAI_TT + dut1, abs=1e-6
    )
    assert brahe.time_system_offset_for_mjd(mjd, brahe.TT, brahe.TAI) == brahe.TAI_TT

    # UTC
    assert (
        brahe.time_system_offset_for_mjd(mjd, brahe.UTC, brahe.GPS)
        == -dutc + brahe.GPS_TAI
    )
    assert (
        brahe.time_system_offset_for_mjd(mjd, brahe.UTC, brahe.TT)
        == -dutc + brahe.TT_TAI
    )
    assert brahe.time_system_offset_for_mjd(mjd, brahe.UTC, brahe.UTC) == 0.0
    assert brahe.time_system_offset_for_mjd(mjd, brahe.UTC, brahe.UT1) == pytest.approx(
        dut1, abs=1e-6
    )
    assert brahe.time_system_offset_for_mjd(mjd, brahe.UTC, brahe.TAI) == -dutc

    # UT1
    assert brahe.time_system_offset_for_mjd(mjd, brahe.UT1, brahe.GPS) == pytest.approx(
        -dutc + brahe.GPS_TAI - dut1, abs=1e-6
    )
    assert brahe.time_system_offset_for_mjd(mjd, brahe.UT1, brahe.TT) == pytest.approx(
        -dutc + brahe.TT_TAI - dut1, abs=1e-6
    )
    assert brahe.time_system_offset_for_mjd(mjd, brahe.UT1, brahe.UTC) == pytest.approx(
        -dut1, abs=1e-6
    )
    assert brahe.time_system_offset_for_mjd(mjd, brahe.UT1, brahe.UT1) == pytest.approx(
        0.0, abs=1e-6
    )
    assert brahe.time_system_offset_for_mjd(mjd, brahe.UT1, brahe.TAI) == pytest.approx(
        -dutc - dut1, abs=1e-6
    )

    # TAI
    assert brahe.time_system_offset_for_mjd(mjd, brahe.TAI, brahe.GPS) == brahe.GPS_TAI
    assert brahe.time_system_offset_for_mjd(mjd, brahe.TAI, brahe.TT) == brahe.TT_TAI
    assert brahe.time_system_offset_for_mjd(mjd, brahe.TAI, brahe.UTC) == dutc
    assert brahe.time_system_offset_for_mjd(mjd, brahe.TAI, brahe.UT1) == pytest.approx(
        dutc + dut1, abs=1e-6
    )
    assert brahe.time_system_offset_for_mjd(mjd, brahe.TAI, brahe.TAI) == 0.0


def test_time_system_offset_for_datetime(eop):  # Test date
    # UTC - TAI offset
    dutc = -37.0
    dut1 = 0.0769966

    # GPS
    assert (
        brahe.time_system_offset_for_datetime(
            2018, 6, 1, 0, 0, 0.0, 0.0, brahe.GPS, brahe.GPS
        )
        == 0.0
    )
    assert (
        brahe.time_system_offset_for_datetime(
            2018, 6, 1, 0, 0, 0.0, 0.0, brahe.GPS, brahe.TT
        )
        == brahe.TT_GPS
    )
    assert (
        brahe.time_system_offset_for_datetime(
            2018, 6, 1, 0, 0, 0.0, 0.0, brahe.GPS, brahe.UTC
        )
        == dutc + brahe.TAI_GPS
    )
    assert brahe.time_system_offset_for_datetime(
        2018, 6, 1, 0, 0, 0.0, 0.0, brahe.GPS, brahe.UT1
    ) == pytest.approx(dutc + brahe.TAI_GPS + dut1, abs=1e-6)
    assert (
        brahe.time_system_offset_for_datetime(
            2018, 6, 1, 0, 0, 0.0, 0.0, brahe.GPS, brahe.TAI
        )
        == brahe.TAI_GPS
    )

    # TT
    assert (
        brahe.time_system_offset_for_datetime(
            2018, 6, 1, 0, 0, 0.0, 0.0, brahe.TT, brahe.GPS
        )
        == brahe.GPS_TT
    )
    assert (
        brahe.time_system_offset_for_datetime(
            2018, 6, 1, 0, 0, 0.0, 0.0, brahe.TT, brahe.TT
        )
        == 0.0
    )
    assert (
        brahe.time_system_offset_for_datetime(
            2018, 6, 1, 0, 0, 0.0, 0.0, brahe.TT, brahe.UTC
        )
        == dutc + brahe.TAI_TT
    )
    assert brahe.time_system_offset_for_datetime(
        2018, 6, 1, 0, 0, 0.0, 0.0, brahe.TT, brahe.UT1
    ) == pytest.approx(dutc + brahe.TAI_TT + dut1, abs=1e-6)
    assert (
        brahe.time_system_offset_for_datetime(
            2018, 6, 1, 0, 0, 0.0, 0.0, brahe.TT, brahe.TAI
        )
        == brahe.TAI_TT
    )

    # UTC
    assert (
        brahe.time_system_offset_for_datetime(
            2018, 6, 1, 0, 0, 0.0, 0.0, brahe.UTC, brahe.GPS
        )
        == -dutc + brahe.GPS_TAI
    )
    assert (
        brahe.time_system_offset_for_datetime(
            2018, 6, 1, 0, 0, 0.0, 0.0, brahe.UTC, brahe.TT
        )
        == -dutc + brahe.TT_TAI
    )
    assert (
        brahe.time_system_offset_for_datetime(
            2018, 6, 1, 0, 0, 0.0, 0.0, brahe.UTC, brahe.UTC
        )
        == 0.0
    )
    assert brahe.time_system_offset_for_datetime(
        2018, 6, 1, 0, 0, 0.0, 0.0, brahe.UTC, brahe.UT1
    ) == pytest.approx(dut1, abs=1e-6)
    assert (
        brahe.time_system_offset_for_datetime(
            2018, 6, 1, 0, 0, 0.0, 0.0, brahe.UTC, brahe.TAI
        )
        == -dutc
    )

    # UT1
    assert brahe.time_system_offset_for_datetime(
        2018, 6, 1, 0, 0, 0.0, 0.0, brahe.UT1, brahe.GPS
    ) == pytest.approx(-dutc + brahe.GPS_TAI - dut1, abs=1e-6)
    assert brahe.time_system_offset_for_datetime(
        2018, 6, 1, 0, 0, 0.0, 0.0, brahe.UT1, brahe.TT
    ) == pytest.approx(-dutc + brahe.TT_TAI - dut1, abs=1e-6)
    assert brahe.time_system_offset_for_datetime(
        2018, 6, 1, 0, 0, 0.0, 0.0, brahe.UT1, brahe.UTC
    ) == pytest.approx(-dut1, abs=1e-6)
    assert brahe.time_system_offset_for_datetime(
        2018, 6, 1, 0, 0, 0.0, 0.0, brahe.UT1, brahe.UT1
    ) == pytest.approx(0.0, abs=1e-6)
    assert brahe.time_system_offset_for_datetime(
        2018, 6, 1, 0, 0, 0.0, 0.0, brahe.UT1, brahe.TAI
    ) == pytest.approx(-dutc - dut1, abs=1e-6)

    # TAI
    assert (
        brahe.time_system_offset_for_datetime(
            2018, 6, 1, 0, 0, 0.0, 0.0, brahe.TAI, brahe.GPS
        )
        == brahe.GPS_TAI
    )
    assert (
        brahe.time_system_offset_for_datetime(
            2018, 6, 1, 0, 0, 0.0, 0.0, brahe.TAI, brahe.TT
        )
        == brahe.TT_TAI
    )
    assert (
        brahe.time_system_offset_for_datetime(
            2018, 6, 1, 0, 0, 0.0, 0.0, brahe.TAI, brahe.UTC
        )
        == dutc
    )
    assert brahe.time_system_offset_for_datetime(
        2018, 6, 1, 0, 0, 0.0, 0.0, brahe.TAI, brahe.UT1
    ) == pytest.approx(dutc + dut1, abs=1e-6)
    assert (
        brahe.time_system_offset_for_datetime(
            2018, 6, 1, 0, 0, 0.0, 0.0, brahe.TAI, brahe.TAI
        )
        == 0.0
    )
