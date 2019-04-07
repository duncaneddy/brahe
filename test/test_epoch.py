#!/usr/local/bin/python3

# Test Modules
import sys
import pytest
import logging
from   pytest import approx
from   os import path

# Import module under test
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import brahe.constants as _constants
from brahe.epoch import *

# Set Log level
LOG_FORMAT = '%(levelname)s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)

# Other imports
import math

def test_constructor_date():
    epc = Epoch(2018, 12, 20, 0, 0, 0, 0.0, tsys="TAI")
    assert epc.days        == 2458472
    assert epc.seconds     == 43200
    assert epc.nanoseconds == 0.0
    assert epc.tsys        == "TAI"

    epc = Epoch(2018, 12, 20, 0, 0, .5, 1.0001, tsys="TAI")
    assert epc.days        == 2458472
    assert epc.seconds     == 43200
    assert epc.nanoseconds == 500000001.0001
    assert epc.tsys        == "TAI"

# String constructor
def test_constructor_string():
    epc = Epoch("2018-12-20")
    year, month, day, hour, minute, seconds, nanoseconds = epc.caldate()
    assert year        == 2018
    assert month       == 12
    assert day         == 20
    assert hour        == 0
    assert minute      == 0
    assert seconds     == 0
    assert nanoseconds == 0.0
    assert epc.tsys    == "UTC"

    epc = Epoch("2018-12-20T16:22:19.0Z")
    year, month, day, hour, minute, seconds, nanoseconds = epc.caldate()
    assert year        == 2018
    assert month       == 12
    assert day         == 20
    assert hour        == 16
    assert minute      == 22
    assert seconds     == 19
    assert nanoseconds == 0.0
    assert epc.tsys    == "UTC"

    epc = Epoch("2018-12-20T16:22:19.123Z")
    year, month, day, hour, minute, seconds, nanoseconds = epc.caldate()
    assert year        == 2018
    assert month       == 12
    assert day         == 20
    assert hour        == 16
    assert minute      == 22
    assert seconds     == 19
    assert nanoseconds == 123000000
    assert epc.tsys    == "UTC"

    epc = Epoch("2018-12-20T16:22:19.123456789Z")
    year, month, day, hour, minute, seconds, nanoseconds = epc.caldate()
    assert year        == 2018
    assert month       == 12
    assert day         == 20
    assert hour        == 16
    assert minute      == 22
    assert seconds     == 19
    assert nanoseconds == 123456789
    assert epc.tsys    == "UTC"

    epc = Epoch("2018-12-20T16:22:19Z")
    year, month, day, hour, minute, seconds, nanoseconds = epc.caldate()
    assert year        == 2018
    assert month       == 12
    assert day         == 20
    assert hour        == 16
    assert minute      == 22
    assert seconds     == 19
    assert nanoseconds == 0.0
    assert epc.tsys    == "UTC"

    epc = Epoch("20181220T162219Z")
    year, month, day, hour, minute, seconds, nanoseconds = epc.caldate()
    assert year        == 2018
    assert month       == 12
    assert day         == 20
    assert hour        == 16
    assert minute      == 22
    assert seconds     == 19
    assert nanoseconds == 0.0
    assert epc.tsys    == "UTC"

    epc = Epoch("2018-12-01 16:22:19 GPS")
    year, month, day, hour, minute, seconds, nanoseconds = epc.caldate(tsys="GPS")
    assert year        == 2018
    assert month       == 12
    assert day         == 1
    assert hour        == 16
    assert minute      == 22
    assert seconds     == 19
    assert nanoseconds == 0.0
    assert epc.tsys    == "GPS"

    epc = Epoch("2018-12-01 16:22:19.0 GPS")
    year, month, day, hour, minute, seconds, nanoseconds = epc.caldate(tsys="GPS")
    assert year        == 2018
    assert month       == 12
    assert day         == 1
    assert hour        == 16
    assert minute      == 22
    assert seconds     == 19
    assert nanoseconds == 0.0
    assert epc.tsys    == "GPS"

    epc = Epoch("2018-12-01 16:22:19.123 GPS")
    year, month, day, hour, minute, seconds, nanoseconds = epc.caldate(tsys="GPS")
    assert year        == 2018
    assert month       == 12
    assert day         == 1
    assert hour        == 16
    assert minute      == 22
    assert seconds     == 19
    assert nanoseconds == 123000000
    assert epc.tsys    == "GPS"

    epc = Epoch("2018-12-01 16:22:19.123456789 GPS")
    year, month, day, hour, minute, seconds, nanoseconds = epc.caldate(tsys="GPS")
    assert year        == 2018
    assert month       == 12
    assert day         == 1
    assert hour        == 16
    assert minute      == 22
    assert seconds     == 19
    assert nanoseconds == 123456789
    assert epc.tsys    == "GPS"

def test_operators():
    epc = Epoch("2019-01-01 12:00:00 TAI")
    assert epc.days        == 2458485
    assert epc.seconds     == 0
    assert epc.nanoseconds == 0

    epc += 1.0e-9
    assert epc.days        == 2458485
    assert epc.seconds     == 0
    assert epc.nanoseconds == 1

    epc -= 2.0e-9
    assert epc.days        == 2458484
    assert epc.seconds     == 86399
    assert epc.nanoseconds == 999999999

    for _ in range(0, 86400):
        epc += 1

    assert epc.days        == 2458485
    assert epc.seconds     == 86399
    assert epc.nanoseconds == 999999999

    year, month, day, hour, minute, second, nanoseconds = epc.caldate("TAI")
    assert year        == 2019
    assert month       == 1
    assert day         == 2
    assert hour        == 11
    assert minute      == 59
    assert second      == 59
    assert nanoseconds == 999999999

    # This test takes longer to run
    # for _ in range(0, 365*86400):
    #     epc += 1

    # assert epc.days        == 2458849
    # assert epc.seconds     == 86399
    # assert epc.nanoseconds == 999999999

    # year, month, day, hour, minute, second, nanoseconds = epc.caldate("TAI")
    # assert year        == 2020
    # assert month       == 1
    # assert day         == 1
    # assert hour        == 11
    # assert minute      == 59
    # assert second      == 59
    # assert nanoseconds == 999999999

def test_conversion():
    epc = Epoch("2018-12-20T16:22:19.123456789Z", tsys="TAI")
    year, month, day, hour, minute, second, nanoseconds = epc.caldate(tsys="TAI")
    assert year        == 2018
    assert month       == 12
    assert day         == 20
    assert hour        == 16
    assert minute      == 22
    assert second      == 19 + 37
    assert nanoseconds == 123456789.0

    epc = Epoch("2018-12-20T16:22:19.123456789Z")
    year, month, day, hour, minute, second, nanoseconds = epc.caldate(tsys="GPS")
    assert year        == 2018
    assert month       == 12
    assert day         == 20
    assert hour        == 16
    assert minute      == 22
    assert second      == 19 + 37 - 19
    assert nanoseconds == 123456789.0

def test_jd():
    epc = Epoch(2000, 1, 1, tsys="GPS")
    assert epc.jd() == 2451544.5
    assert epc.jd(tsys="UTC") < 2451544.5

def test_mjd():
    epc = Epoch(2000, 1, 1, 0, tsys="GPS")
    assert epc.mjd() == 51544
    assert epc.mjd(tsys="UTC") < 51544

def test_day_of_year():
    epc = Epoch(2000, 1, 1)
    assert epc.day_of_year() == 1

    epc = Epoch(2000, 1, 1, 12)
    assert epc.day_of_year() == 1.5

    epc = Epoch(2000, 12, 31)
    assert epc.day_of_year() == 366

    epc = Epoch(2001, 1, 1)
    assert epc.day_of_year() == 1

    epc = Epoch(2001, 12, 31)
    assert epc.day_of_year() == 365

def test_gmst():
    epc = Epoch(2000, 1, 1)
    assert approx(epc.gmst(use_degrees=True), 99.835, abs=1e-3)

    epc = Epoch(2000, 1, 1)
    assert approx(epc.gmst(use_degrees=False), 1.742, abs=1e-3)

def test_gast():
    epc = Epoch(2000, 1, 1)
    assert approx(epc.gast(use_degrees=True), 99.832, abs=1e-3)

    epc = Epoch(2000, 1, 1)
    assert approx(epc.gast(use_degrees=False), 1.742, abs=1e-3)

# Arithmetic Comparisons
def test_logic():
    # Equal, Not Equal
    assert Epoch(2000, 1, 1, 12, 23, 59, 123456789) == Epoch(2000, 1, 1, 12, 23, 59, 123456789)
    assert Epoch(2000, 1, 1, 12, 23, 59, 123456789) != Epoch(2000, 1, 1, 12, 23, 59, 123456788)

    # Less-than, Less-than-equal
    assert not (Epoch(2000, 1, 1, 12, 23, 59, 123456789) < Epoch(2000, 1, 1, 12, 23, 59, 123456789))
    assert not (Epoch(2000, 1, 1, 12, 23, 59, 123456789) <= Epoch(2000, 1, 1, 12, 23, 59, 123456788))
    assert Epoch(2000, 1, 1, 12, 23, 59, 123456788) < Epoch(2000, 1, 1, 12, 23, 59, 123456789)
    assert Epoch(2000, 1, 1, 12, 23, 59, 123456788) <= Epoch(2000, 1, 1, 12, 23, 59, 123456789)

    # Greater-than, Greater-than-equal
    assert not (Epoch(2000, 1, 1, 12, 23, 59, 123456789) > Epoch(2000, 1, 1, 12, 23, 59, 123456789))
    assert not (Epoch(2000, 1, 1, 12, 23, 59, 123456788) >= Epoch(2000, 1, 1, 12, 23, 59, 123456789))
    assert Epoch(2000, 1, 1, 12, 23, 59, 123456789) > Epoch(2000, 1, 1, 12, 23, 59, 123456788)
    assert Epoch(2000, 1, 1, 12, 23, 59, 123456789) >= Epoch(2000, 1, 1, 12, 23, 59, 123456788)

if __name__ == '__main__':
    test_constructor_date()
    test_constructor_string()
    test_operators()
    test_conversion()
    test_jd()
    test_mjd()
    test_day_of_year()
    test_gmst()
    test_gast()
    test_logic()