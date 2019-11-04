# Test Imports
from pytest import approx

# Modules Under Test
import brahe.constants as _constants
from brahe.time import *

def test_caldate_to_mjd():
    assert caldate_to_mjd(2000, 1, 1, 12, 0, 0) == 51544.5

def test_mjd_to_caldate():
    year, month, day, hour, minute, second, nanosecond = mjd_to_caldate(51544.5)
    
    assert year       == 2000
    assert month      == 1
    assert day        == 1
    assert hour       == 12
    assert minute     == 0
    assert second     == 0
    assert nanosecond == 0

def test_caldate_to_jd():
    assert caldate_to_jd(2000, 1, 1, 12, 0, 0) == 2451545.0

def test_jd_to_caldate():
    year, month, day, hour, minute, second, nanosecond = jd_to_caldate(2451545.0)
    
    assert year       == 2000
    assert month      == 1
    assert day        == 1
    assert hour       == 12
    assert minute     == 0
    assert second     == 0
    assert nanosecond == 0

def test_time_system_offset():
    jd = caldate_to_jd(2018, 6, 1)

    dutc = -37.0

    # GPS
    assert time_system_offset(jd, 0, "GPS", "GPS") == 0
    assert time_system_offset(jd, 0, "GPS", "TT")  == _constants.TT_GPS
    assert time_system_offset(jd, 0, "GPS", "UTC") == -18
    assert approx(time_system_offset(jd, 0, "GPS", "UT1"), -17.92267, abs=1e-4)
    assert time_system_offset(jd, 0, "GPS", "TAI") == _constants.TAI_GPS

    # TT
    assert time_system_offset(jd, 0, "TT", "GPS") == _constants.GPS_TT
    assert time_system_offset(jd, 0, "TT", "TT")  == 0
    assert time_system_offset(jd, 0, "TT", "UTC") == dutc + _constants.TAI_TT
    assert approx(time_system_offset(jd, 0, "TT", "UT1"), -69.10667, abs=1e-4)
    assert time_system_offset(jd, 0, "TT", "TAI") == _constants.TAI_TT

    # UTC
    assert time_system_offset(jd, 0, "UTC", "GPS") == 18
    assert time_system_offset(jd, 0, "UTC", "TT")  == -dutc + _constants.TT_TAI
    assert time_system_offset(jd, 0, "UTC", "UTC") == 0.0
    assert approx(time_system_offset(jd, 0, "UTC", "UT1"), 0.0769968, abs=1e-4)
    assert time_system_offset(jd, 0, "UTC", "TAI") == -dutc

    # UT1
    assert approx(time_system_offset(jd, 0, "UT1", "GPS"), 17.9230032, abs=1e-4)
    assert approx(time_system_offset(jd, 0, "UT1", "TT"), 69.1070032, abs=1e-4)
    assert approx(time_system_offset(jd, 0, "UT1", "UTC"), -0.0769968, abs=1e-4)
    assert time_system_offset(jd, 0, "UT1", "UT1") == 0
    assert approx(time_system_offset(jd, 0, "UT1", "TAI"), 36.9230032, abs=1e-4)

    # TAI
    assert time_system_offset(jd, 0, "TAI", "GPS") == _constants.GPS_TAI
    assert time_system_offset(jd, 0, "TAI", "TT")  == _constants.TT_TAI
    assert time_system_offset(jd, 0, "TAI", "UTC") == dutc
    assert approx(time_system_offset(jd, 0, "TAI", "UT1"), -36.92267, abs=1e-4)
    assert time_system_offset(jd, 0, "TAI", "TAI") == 0