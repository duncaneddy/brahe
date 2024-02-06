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
    assert brahe.time_system_offset_for_jd(jd, "GPS", "GPS") == 0.0
    assert brahe.time_system_offset_for_jd(jd, "GPS", "TT")  == brahe.TT_GPS
    assert brahe.time_system_offset_for_jd(jd, "GPS", "UTC") == dutc + brahe.TAI_GPS
    assert brahe.time_system_offset_for_jd(jd, "GPS", "UT1") == pytest.approx(dutc + brahe.TAI_GPS + dut1, abs=1e-6)
    assert brahe.time_system_offset_for_jd(jd, "GPS", "TAI") == brahe.TAI_GPS

    # TT
    assert brahe.time_system_offset_for_jd(jd, "TT", "GPS") == brahe.GPS_TT
    assert brahe.time_system_offset_for_jd(jd, "TT", "TT")  == 0.0
    assert brahe.time_system_offset_for_jd(jd, "TT", "UTC") == dutc + brahe.TAI_TT
    assert brahe.time_system_offset_for_jd(jd, "TT", "UT1") == pytest.approx(dutc + brahe.TAI_TT + dut1, abs=1e-6)
    assert brahe.time_system_offset_for_jd(jd, "TT", "TAI") == brahe.TAI_TT

    # UTC
    assert brahe.time_system_offset_for_jd(jd, "UTC", "GPS") == -dutc + brahe.GPS_TAI
    assert brahe.time_system_offset_for_jd(jd, "UTC", "TT")  == -dutc + brahe.TT_TAI
    assert brahe.time_system_offset_for_jd(jd, "UTC", "UTC") == 0.0
    assert brahe.time_system_offset_for_jd(jd, "UTC", "UT1") == pytest.approx(dut1, abs=1e-6)
    assert brahe.time_system_offset_for_jd(jd, "UTC", "TAI") == -dutc

    # UT1
    assert brahe.time_system_offset_for_jd(jd, "UT1", "GPS") == pytest.approx(-dutc + brahe.GPS_TAI - dut1, abs=1e-6)
    assert brahe.time_system_offset_for_jd(jd, "UT1", "TT")  == pytest.approx(-dutc + brahe.TT_TAI - dut1, abs=1e-6)
    assert brahe.time_system_offset_for_jd(jd, "UT1", "UTC") == pytest.approx(-dut1, abs=1e-6)
    assert brahe.time_system_offset_for_jd(jd, "UT1", "UT1") == pytest.approx(0.0, abs=1e-6)
    assert brahe.time_system_offset_for_jd(jd, "UT1", "TAI") == pytest.approx(-dutc - dut1, abs=1e-6)

    # TAI
    assert brahe.time_system_offset_for_jd(jd, "TAI", "GPS") == brahe.GPS_TAI
    assert brahe.time_system_offset_for_jd(jd, "TAI", "TT")  == brahe.TT_TAI
    assert brahe.time_system_offset_for_jd(jd, "TAI", "UTC") == dutc
    assert brahe.time_system_offset_for_jd(jd, "TAI", "UT1") == pytest.approx(dutc + dut1, abs=1e-6)
    assert brahe.time_system_offset_for_jd(jd, "TAI", "TAI") == 0.0

def test_time_system_offset_for_mjd(eop):  # Test date
    mjd = brahe.datetime_to_mjd(2018, 6, 1, 0, 0, 0.0, 0.0)

    # UTC - TAI offset
    dutc = -37.0
    dut1 = 0.0769966

    # GPS
    assert brahe.time_system_offset_for_mjd(mjd, "GPS", "GPS") == 0.0
    assert brahe.time_system_offset_for_mjd(mjd, "GPS", "TT")  == brahe.TT_GPS
    assert brahe.time_system_offset_for_mjd(mjd, "GPS", "UTC") == dutc + brahe.TAI_GPS
    assert brahe.time_system_offset_for_mjd(mjd, "GPS", "UT1") == pytest.approx(dutc + brahe.TAI_GPS + dut1, abs=1e-6)
    assert brahe.time_system_offset_for_mjd(mjd, "GPS", "TAI") == brahe.TAI_GPS

    # TT
    assert brahe.time_system_offset_for_mjd(mjd, "TT", "GPS") == brahe.GPS_TT
    assert brahe.time_system_offset_for_mjd(mjd, "TT", "TT")  == 0.0
    assert brahe.time_system_offset_for_mjd(mjd, "TT", "UTC") == dutc + brahe.TAI_TT
    assert brahe.time_system_offset_for_mjd(mjd, "TT", "UT1") == pytest.approx(dutc + brahe.TAI_TT + dut1, abs=1e-6)
    assert brahe.time_system_offset_for_mjd(mjd, "TT", "TAI") == brahe.TAI_TT

    # UTC
    assert brahe.time_system_offset_for_mjd(mjd, "UTC", "GPS") == -dutc + brahe.GPS_TAI
    assert brahe.time_system_offset_for_mjd(mjd, "UTC", "TT")  == -dutc + brahe.TT_TAI
    assert brahe.time_system_offset_for_mjd(mjd, "UTC", "UTC") == 0.0
    assert brahe.time_system_offset_for_mjd(mjd, "UTC", "UT1") == pytest.approx(dut1, abs=1e-6)
    assert brahe.time_system_offset_for_mjd(mjd, "UTC", "TAI") == -dutc

    # UT1
    assert brahe.time_system_offset_for_mjd(mjd, "UT1", "GPS") == pytest.approx(-dutc + brahe.GPS_TAI - dut1, abs=1e-6)
    assert brahe.time_system_offset_for_mjd(mjd, "UT1", "TT")  == pytest.approx(-dutc + brahe.TT_TAI - dut1, abs=1e-6)
    assert brahe.time_system_offset_for_mjd(mjd, "UT1", "UTC") == pytest.approx(-dut1, abs=1e-6)
    assert brahe.time_system_offset_for_mjd(mjd, "UT1", "UT1") == pytest.approx(0.0, abs=1e-6)
    assert brahe.time_system_offset_for_mjd(mjd, "UT1", "TAI") == pytest.approx(-dutc - dut1, abs=1e-6)

    # TAI
    assert brahe.time_system_offset_for_mjd(mjd, "TAI", "GPS") == brahe.GPS_TAI
    assert brahe.time_system_offset_for_mjd(mjd, "TAI", "TT")  == brahe.TT_TAI
    assert brahe.time_system_offset_for_mjd(mjd, "TAI", "UTC") == dutc
    assert brahe.time_system_offset_for_mjd(mjd, "TAI", "UT1") == pytest.approx(dutc + dut1, abs=1e-6)
    assert brahe.time_system_offset_for_mjd(mjd, "TAI", "TAI") == 0.0

def test_time_system_offset_for_datetime(eop):  # Test date
    # UTC - TAI offset
    dutc = -37.0
    dut1 = 0.0769966

    # GPS
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, "GPS", "GPS") == 0.0
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, "GPS", "TT")  == brahe.TT_GPS
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, "GPS", "UTC") == dutc + brahe.TAI_GPS
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, "GPS", "UT1") == pytest.approx(dutc + brahe.TAI_GPS + dut1, abs=1e-6)
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, "GPS", "TAI") == brahe.TAI_GPS

    # TT
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, "TT", "GPS") == brahe.GPS_TT
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, "TT", "TT")  == 0.0
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, "TT", "UTC") == dutc + brahe.TAI_TT
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, "TT", "UT1") == pytest.approx(dutc + brahe.TAI_TT + dut1, abs=1e-6)
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, "TT", "TAI") == brahe.TAI_TT

    # UTC
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, "UTC", "GPS") == -dutc + brahe.GPS_TAI
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, "UTC", "TT")  == -dutc + brahe.TT_TAI
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, "UTC", "UTC") == 0.0
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, "UTC", "UT1") == pytest.approx(dut1, abs=1e-6)
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, "UTC", "TAI") == -dutc

    # UT1
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, "UT1", "GPS") == pytest.approx(-dutc + brahe.GPS_TAI - dut1, abs=1e-6)
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, "UT1", "TT")  == pytest.approx(-dutc + brahe.TT_TAI - dut1, abs=1e-6)
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, "UT1", "UTC") == pytest.approx(-dut1, abs=1e-6)
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, "UT1", "UT1") == pytest.approx(0.0, abs=1e-6)
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, "UT1", "TAI") == pytest.approx(-dutc - dut1, abs=1e-6)

    # TAI
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, "TAI", "GPS") == brahe.GPS_TAI
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, "TAI", "TT")  == brahe.TT_TAI
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, "TAI", "UTC") == dutc
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, "TAI", "UT1") == pytest.approx(dutc + dut1, abs=1e-6)
    assert brahe.time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, "TAI", "TAI") == 0.0
