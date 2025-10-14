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

def test_time_system_offset_for_jd(eop):  # Test date
    jd = brahe.datetime_to_jd(2018, 6, 1, 0, 0, 0.0, 0.0)

    # UTC - TAI offset
    dutc = -37.0
    dut1 = 0.0769966

    # GPS
    assert brahe.time_system_offset_for_jd(jd, brahe.GPS, brahe.GPS) == 0.0
    assert brahe.time_system_offset_for_jd(jd, brahe.GPS, brahe.TT)  == brahe.TT_GPS
    assert brahe.time_system_offset_for_jd(jd, brahe.GPS, brahe.UTC) == dutc + brahe.TAI_GPS
    assert brahe.time_system_offset_for_jd(jd, brahe.GPS, brahe.UT1) == pytest.approx(dutc + brahe.TAI_GPS + dut1, abs=1e-6)
    assert brahe.time_system_offset_for_jd(jd, brahe.GPS, brahe.TAI) == brahe.TAI_GPS

    # TT
    assert brahe.time_system_offset_for_jd(jd, brahe.TT, brahe.GPS) == brahe.GPS_TT
    assert brahe.time_system_offset_for_jd(jd, brahe.TT, brahe.TT)  == 0.0
    assert brahe.time_system_offset_for_jd(jd, brahe.TT, brahe.UTC) == dutc + brahe.TAI_TT
    assert brahe.time_system_offset_for_jd(jd, brahe.TT, brahe.UT1) == pytest.approx(dutc + brahe.TAI_TT + dut1, abs=1e-6)
    assert brahe.time_system_offset_for_jd(jd, brahe.TT, brahe.TAI) == brahe.TAI_TT

    # UTC
    assert brahe.time_system_offset_for_jd(jd, brahe.UTC, brahe.GPS) == -dutc + brahe.GPS_TAI
    assert brahe.time_system_offset_for_jd(jd, brahe.UTC, brahe.TT)  == -dutc + brahe.TT_TAI
    assert brahe.time_system_offset_for_jd(jd, brahe.UTC, brahe.UTC) == 0.0
    assert brahe.time_system_offset_for_jd(jd, brahe.UTC, brahe.UT1) == pytest.approx(dut1, abs=1e-6)
    assert brahe.time_system_offset_for_jd(jd, brahe.UTC, brahe.TAI) == -dutc

    # UT1
    assert brahe.time_system_offset_for_jd(jd, brahe.UT1, brahe.GPS) == pytest.approx(-dutc + brahe.GPS_TAI - dut1, abs=1e-6)
    assert brahe.time_system_offset_for_jd(jd, brahe.UT1, brahe.TT)  == pytest.approx(-dutc + brahe.TT_TAI - dut1, abs=1e-6)
    assert brahe.time_system_offset_for_jd(jd, brahe.UT1, brahe.UTC) == pytest.approx(-dut1, abs=1e-6)
    assert brahe.time_system_offset_for_jd(jd, brahe.UT1, brahe.UT1) == pytest.approx(0.0, abs=1e-6)
    assert brahe.time_system_offset_for_jd(jd, brahe.UT1, brahe.TAI) == pytest.approx(-dutc - dut1, abs=1e-6)

    # TAI
    assert brahe.time_system_offset_for_jd(jd, brahe.TAI, brahe.GPS) == brahe.GPS_TAI
    assert brahe.time_system_offset_for_jd(jd, brahe.TAI, brahe.TT)  == brahe.TT_TAI
    assert brahe.time_system_offset_for_jd(jd, brahe.TAI, brahe.UTC) == dutc
    assert brahe.time_system_offset_for_jd(jd, brahe.TAI, brahe.UT1) == pytest.approx(dutc + dut1, abs=1e-6)
    assert brahe.time_system_offset_for_jd(jd, brahe.TAI, brahe.TAI) == 0.0

def test_time_system_offset_for_mjd(eop):  # Test date
    mjd = brahe.datetime_to_mjd(2018, 6, 1, 0, 0, 0.0, 0.0)

    # UTC - TAI offset
    dutc = -37.0
    dut1 = 0.0769966

    # GPS
    assert brahe.time_system_offset_for_mjd(mjd, brahe.GPS, brahe.GPS) == 0.0
    assert brahe.time_system_offset_for_mjd(mjd, brahe.GPS, brahe.TT)  == brahe.TT_GPS
    assert brahe.time_system_offset_for_mjd(mjd, brahe.GPS, brahe.UTC) == dutc + brahe.TAI_GPS
    assert brahe.time_system_offset_for_mjd(mjd, brahe.GPS, brahe.UT1) == pytest.approx(dutc + brahe.TAI_GPS + dut1, abs=1e-6)
    assert brahe.time_system_offset_for_mjd(mjd, brahe.GPS, brahe.TAI) == brahe.TAI_GPS

    # TT
    assert brahe.time_system_offset_for_mjd(mjd, brahe.TT, brahe.GPS) == brahe.GPS_TT
    assert brahe.time_system_offset_for_mjd(mjd, brahe.TT, brahe.TT)  == 0.0
    assert brahe.time_system_offset_for_mjd(mjd, brahe.TT, brahe.UTC) == dutc + brahe.TAI_TT
    assert brahe.time_system_offset_for_mjd(mjd, brahe.TT, brahe.UT1) == pytest.approx(dutc + brahe.TAI_TT + dut1, abs=1e-6)
    assert brahe.time_system_offset_for_mjd(mjd, brahe.TT, brahe.TAI) == brahe.TAI_TT

    # UTC
    assert brahe.time_system_offset_for_mjd(mjd, brahe.UTC, brahe.GPS) == -dutc + brahe.GPS_TAI
    assert brahe.time_system_offset_for_mjd(mjd, brahe.UTC, brahe.TT)  == -dutc + brahe.TT_TAI
    assert brahe.time_system_offset_for_mjd(mjd, brahe.UTC, brahe.UTC) == 0.0
    assert brahe.time_system_offset_for_mjd(mjd, brahe.UTC, brahe.UT1) == pytest.approx(dut1, abs=1e-6)
    assert brahe.time_system_offset_for_mjd(mjd, brahe.UTC, brahe.TAI) == -dutc

    # UT1
    assert brahe.time_system_offset_for_mjd(mjd, brahe.UT1, brahe.GPS) == pytest.approx(-dutc + brahe.GPS_TAI - dut1, abs=1e-6)
    assert brahe.time_system_offset_for_mjd(mjd, brahe.UT1, brahe.TT)  == pytest.approx(-dutc + brahe.TT_TAI - dut1, abs=1e-6)
    assert brahe.time_system_offset_for_mjd(mjd, brahe.UT1, brahe.UTC) == pytest.approx(-dut1, abs=1e-6)
    assert brahe.time_system_offset_for_mjd(mjd, brahe.UT1, brahe.UT1) == pytest.approx(0.0, abs=1e-6)
    assert brahe.time_system_offset_for_mjd(mjd, brahe.UT1, brahe.TAI) == pytest.approx(-dutc - dut1, abs=1e-6)

    # TAI
    assert brahe.time_system_offset_for_mjd(mjd, brahe.TAI, brahe.GPS) == brahe.GPS_TAI
    assert brahe.time_system_offset_for_mjd(mjd, brahe.TAI, brahe.TT)  == brahe.TT_TAI
    assert brahe.time_system_offset_for_mjd(mjd, brahe.TAI, brahe.UTC) == dutc
    assert brahe.time_system_offset_for_mjd(mjd, brahe.TAI, brahe.UT1) == pytest.approx(dutc + dut1, abs=1e-6)
    assert brahe.time_system_offset_for_mjd(mjd, brahe.TAI, brahe.TAI) == 0.0

def test_time_system_offset_for_datetime(eop):  # Test date
    # UTC - TAI offset
    dutc = -37.0
    dut1 = 0.0769966

    # GPS
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, brahe.GPS, brahe.GPS) == 0.0
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, brahe.GPS, brahe.TT)  == brahe.TT_GPS
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, brahe.GPS, brahe.UTC) == dutc + brahe.TAI_GPS
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, brahe.GPS, brahe.UT1) == pytest.approx(dutc + brahe.TAI_GPS + dut1, abs=1e-6)
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, brahe.GPS, brahe.TAI) == brahe.TAI_GPS

    # TT
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, brahe.TT, brahe.GPS) == brahe.GPS_TT
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, brahe.TT, brahe.TT)  == 0.0
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, brahe.TT, brahe.UTC) == dutc + brahe.TAI_TT
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, brahe.TT, brahe.UT1) == pytest.approx(dutc + brahe.TAI_TT + dut1, abs=1e-6)
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, brahe.TT, brahe.TAI) == brahe.TAI_TT

    # UTC
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, brahe.UTC, brahe.GPS) == -dutc + brahe.GPS_TAI
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, brahe.UTC, brahe.TT)  == -dutc + brahe.TT_TAI
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, brahe.UTC, brahe.UTC) == 0.0
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, brahe.UTC, brahe.UT1) == pytest.approx(dut1, abs=1e-6)
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, brahe.UTC, brahe.TAI) == -dutc

    # UT1
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, brahe.UT1, brahe.GPS) == pytest.approx(-dutc + brahe.GPS_TAI - dut1, abs=1e-6)
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, brahe.UT1, brahe.TT)  == pytest.approx(-dutc + brahe.TT_TAI - dut1, abs=1e-6)
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, brahe.UT1, brahe.UTC) == pytest.approx(-dut1, abs=1e-6)
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, brahe.UT1, brahe.UT1) == pytest.approx(0.0, abs=1e-6)
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, brahe.UT1, brahe.TAI) == pytest.approx(-dutc - dut1, abs=1e-6)

    # TAI
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, brahe.TAI, brahe.GPS) == brahe.GPS_TAI
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, brahe.TAI, brahe.TT)  == brahe.TT_TAI
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, brahe.TAI, brahe.UTC) == dutc
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, brahe.TAI, brahe.UT1) == pytest.approx(dutc + dut1, abs=1e-6)
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, brahe.TAI, brahe.TAI) == 0.0
