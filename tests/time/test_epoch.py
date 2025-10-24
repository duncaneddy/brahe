import math
from datetime import datetime

import pytest

import brahe as bh


def test_epoch_string(eop):
    epc = bh.Epoch.from_date(2022, 1, 1, bh.GPS)

    assert str(epc) == "2022-01-01 00:00:00.000 GPS"


def test_epoch_repr(eop):
    epc = bh.Epoch.from_date(2022, 1, 1, bh.GPS)

    assert epc.__repr__() == "Epoch<2459580, 43219, 0, 0, GPS>"


def test_epoch_time_system(eop):
    epc = bh.Epoch.from_date(2022, 1, 1, bh.GPS)
    assert epc.time_system == bh.GPS

    epc = bh.Epoch.from_date(2022, 1, 1, bh.TAI)
    assert epc.time_system == bh.TAI

    epc = bh.Epoch.from_date(2022, 1, 1, bh.TT)
    assert epc.time_system == bh.TT

    epc = bh.Epoch.from_date(2022, 1, 1, bh.UTC)
    assert epc.time_system == bh.UTC

    epc = bh.Epoch.from_date(2022, 1, 1, bh.UT1)
    assert epc.time_system == bh.UT1


def test_epoch_from_date(eop):
    epc = bh.Epoch.from_date(2020, 1, 2, bh.GPS)

    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()

    assert year == 2020
    assert month == 1
    assert day == 2
    assert hour == 0
    assert minute == 0
    assert second == 0.0
    assert nanosecond == 0.0


def test_epoch_from_datetime(eop):
    epc = bh.Epoch.from_datetime(2020, 1, 2, 3, 4, 5.0, 6.0, bh.GPS)

    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()

    assert year == 2020
    assert month == 1
    assert day == 2
    assert hour == 3
    assert minute == 4
    assert second == 5.0
    assert nanosecond == 6.0


def test_epoch_now(eop):
    # Get current time using Epoch.now()
    epoch_now = bh.Epoch.now()

    # Verify it's in UTC time system
    assert epoch_now.time_system == bh.UTC

    # Get current time from datetime for comparison
    from datetime import datetime, timezone

    dt_now = datetime.now(timezone.utc)

    # Create an Epoch from the datetime
    dt_epoch = bh.Epoch(dt_now)

    # The two epochs should be very close (within 1 second to account for execution time)
    diff = abs(epoch_now - dt_epoch)
    assert diff < 1.0, f"Epoch.now() differs from datetime by {diff} seconds"


def test_epoch_from_string(eop):
    epc = bh.Epoch.from_string("2018-12-20")
    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()
    assert year == 2018
    assert month == 12
    assert day == 20
    assert hour == 0
    assert minute == 0
    assert second == 0.0
    assert nanosecond == 0.0
    assert epc.time_system == bh.UTC

    epc = bh.Epoch.from_string("2018-12-20T16:22:19.0Z")
    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()
    assert year == 2018
    assert month == 12
    assert day == 20
    assert hour == 16
    assert minute == 22
    assert second == 19.0
    assert nanosecond == 0.0
    assert epc.time_system == bh.UTC

    epc = bh.Epoch.from_string("2018-12-20T16:22:19.123Z")
    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()
    assert year == 2018
    assert month == 12
    assert day == 20
    assert hour == 16
    assert minute == 22
    assert second == 19.0
    assert nanosecond == 123000000.0
    assert epc.time_system == bh.UTC

    epc = bh.Epoch.from_string("2018-12-20T16:22:19.123456789Z")
    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()
    assert year == 2018
    assert month == 12
    assert day == 20
    assert hour == 16
    assert minute == 22
    assert second == 19.0
    assert nanosecond == 123456789.0
    assert epc.time_system == bh.UTC

    epc = bh.Epoch.from_string("2018-12-20T16:22:19Z")
    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()
    assert year == 2018
    assert month == 12
    assert day == 20
    assert hour == 16
    assert minute == 22
    assert second == 19.0
    assert nanosecond == 0.0
    assert epc.time_system == bh.UTC

    epc = bh.Epoch.from_string("20181220T162219Z")
    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()
    assert year == 2018
    assert month == 12
    assert day == 20
    assert hour == 16
    assert minute == 22
    assert second == 19.0
    assert nanosecond == 0.0
    assert epc.time_system == bh.UTC

    epc = bh.Epoch.from_string("2018-12-01 16:22:19 GPS")
    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()
    assert year == 2018
    assert month == 12
    assert day == 1
    assert hour == 16
    assert minute == 22
    assert second == 19.0
    assert nanosecond == 0.0
    assert epc.time_system == bh.GPS

    epc = bh.Epoch.from_string("2018-12-01 16:22:19.0 GPS")
    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()
    assert year == 2018
    assert month == 12
    assert day == 1
    assert hour == 16
    assert minute == 22
    assert second == 19.0
    assert nanosecond == 0.0
    assert epc.time_system == bh.GPS

    epc = bh.Epoch.from_string("2018-12-01 16:22:19.123 GPS")
    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()
    assert year == 2018
    assert month == 12
    assert day == 1
    assert hour == 16
    assert minute == 22
    assert second == 19.0
    assert nanosecond == 123000000.0
    assert epc.time_system == bh.GPS

    epc = bh.Epoch.from_string("2018-12-01 16:22:19.123456789 GPS")
    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()
    assert year == 2018
    assert month == 12
    assert day == 1
    assert hour == 16
    assert minute == 22
    assert second == 19.0
    assert nanosecond == 123456789.0
    assert epc.time_system == bh.GPS


def test_epoch_from_jd(eop):
    epc = bh.Epoch.from_jd(bh.MJD_ZERO + bh.MJD2000, bh.TAI)
    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()
    assert year == 2000
    assert month == 1
    assert day == 1
    assert hour == 12
    assert minute == 0
    assert second == 0.0
    assert nanosecond == 0.0
    assert epc.time_system == bh.TAI

    epc = bh.Epoch.from_jd(bh.MJD_ZERO + bh.MJD2000, bh.GPS)
    (year, month, day, hour, minute, second, nanosecond) = (
        epc.to_datetime_as_time_system(bh.TAI)
    )
    assert year == 2000
    assert month == 1
    assert day == 1
    assert hour == 12
    assert minute == 0
    assert second == 19.0
    assert (
        nanosecond == 17643.974853515625
    )  # Rounding error from floating point conversion
    assert epc.time_system == bh.GPS


def test_epoch_from_mjd(eop):
    epc = bh.Epoch.from_mjd(bh.MJD2000, bh.TAI)
    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()
    assert year == 2000
    assert month == 1
    assert day == 1
    assert hour == 12
    assert minute == 0
    assert second == 0.0
    assert nanosecond == 0.0
    assert epc.time_system == bh.TAI

    epc = bh.Epoch.from_mjd(bh.MJD2000, bh.GPS)
    (year, month, day, hour, minute, second, nanosecond) = (
        epc.to_datetime_as_time_system(bh.TAI)
    )
    assert year == 2000
    assert month == 1
    assert day == 1
    assert hour == 12
    assert minute == 0
    assert second == 19.0
    assert (
        nanosecond == 17643.974853515625
    )  # Rounding error from floating point conversion
    assert epc.time_system == bh.GPS


def test_epoch_from_gps_date():
    epc = bh.Epoch.from_gps_date(0, 0.0)
    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()
    assert year == 1980
    assert month == 1
    assert day == 6
    assert hour == 0
    assert minute == 0
    assert second == 0.0
    assert nanosecond == 0.0
    assert epc.time_system == bh.GPS

    epc = bh.Epoch.from_gps_date(2194, 435781.5)
    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()
    assert year == 2022
    assert month == 1
    assert day == 28
    assert hour == 1
    assert minute == 3
    assert second == 1.0
    assert nanosecond == 500000000.0
    assert epc.time_system == bh.GPS


def test_epoch_from_gps_seconds():
    epc = bh.Epoch.from_gps_seconds(0.0)
    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()
    assert year == 1980
    assert month == 1
    assert day == 6
    assert hour == 0
    assert minute == 0
    assert second == 0.0
    assert nanosecond == 0.0
    assert epc.time_system == bh.GPS

    epc = bh.Epoch.from_gps_seconds(2194.0 * 7.0 * 86400.0 + 3.0 * 3600.0 + 61.5)
    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()
    assert year == 2022
    assert month == 1
    assert day == 23
    assert hour == 3
    assert minute == 1
    assert second == 1.0
    assert nanosecond == 500000000.0
    assert epc.time_system == bh.GPS


def test_epoch_from_gps_nanoseconds():
    epc = bh.Epoch.from_gps_nanoseconds(0)
    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()
    assert year == 1980
    assert month == 1
    assert day == 6
    assert hour == 0
    assert minute == 0
    assert second == 0.0
    assert nanosecond == 0.0
    assert epc.time_system == bh.GPS

    gpsns = (2194 * 7 * 86400 + 3 * 3600 + 61) * 1000000000 + 1
    epc = bh.Epoch.from_gps_nanoseconds(gpsns)
    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()
    assert year == 2022
    assert month == 1
    assert day == 23
    assert hour == 3
    assert minute == 1
    assert second == 1.0
    assert nanosecond == 1.0
    assert epc.time_system == bh.GPS


def test_epoch_to_jd(eop):
    epc = bh.Epoch.from_datetime(2000, 1, 1, 12, 0, 0.0, 0.0, bh.TAI)

    assert epc.jd() == bh.MJD_ZERO + bh.MJD2000

    epc = bh.Epoch.from_datetime(2000, 1, 1, 12, 0, 0.0, 0.0, bh.TAI)
    assert epc.jd_as_time_system(bh.UTC) == bh.MJD_ZERO + bh.MJD2000 - 32.0 / 86400.0


def test_epoch_to_mjd(eop):
    epc = bh.Epoch.from_datetime(2000, 1, 1, 12, 0, 0.0, 0.0, bh.TAI)

    assert epc.mjd() == bh.MJD2000

    epc = bh.Epoch.from_datetime(2000, 1, 1, 12, 0, 0.0, 0.0, bh.TAI)
    assert epc.mjd_as_time_system(bh.UTC) == bh.MJD2000 - 32.0 / 86400.0


def test_gps_date(eop):
    epc = bh.Epoch.from_date(2018, 3, 1, bh.GPS)
    (gps_week, gps_seconds) = epc.gps_date()
    assert gps_week == 1990
    assert gps_seconds == 4.0 * 86400.0

    epc = bh.Epoch.from_date(2018, 3, 8, bh.GPS)
    (gps_week, gps_seconds) = epc.gps_date()
    assert gps_week == 1991
    assert gps_seconds == 4.0 * 86400.0

    epc = bh.Epoch.from_date(2018, 3, 11, bh.GPS)
    (gps_week, gps_seconds) = epc.gps_date()
    assert gps_week == 1992
    assert gps_seconds == 0.0 * 86400.0

    epc = bh.Epoch.from_date(2018, 3, 24, bh.GPS)
    (gps_week, gps_seconds) = epc.gps_date()
    assert gps_week == 1993
    assert gps_seconds == 6.0 * 86400.0


def test_gps_seconds(eop):
    epc = bh.Epoch.from_date(1980, 1, 6, bh.GPS)
    assert epc.gps_seconds() == 0.0

    epc = bh.Epoch.from_datetime(1980, 1, 7, 0, 0, 1.0, 0.0, bh.GPS)
    assert epc.gps_seconds() == 86401.0


def test_gps_nanoseconds(eop):
    epc = bh.Epoch.from_date(1980, 1, 6, bh.GPS)
    assert epc.gps_nanoseconds() == 0.0

    epc = bh.Epoch.from_datetime(1980, 1, 7, 0, 0, 1.0, 0.0, bh.GPS)
    assert epc.gps_nanoseconds() == 86401.0 * 1.0e9


def test_isostring(eop):
    # Confirm Before the leap second
    epc = bh.Epoch.from_datetime(2016, 12, 31, 23, 59, 59.0, 0.0, bh.UTC)
    assert epc.isostring() == "2016-12-31T23:59:59Z"

    # The leap second
    epc = bh.Epoch.from_datetime(2016, 12, 31, 23, 59, 60.0, 0.0, bh.UTC)
    assert epc.isostring() == "2016-12-31T23:59:60Z"

    # After the leap second
    epc = bh.Epoch.from_datetime(2017, 1, 1, 0, 0, 0.0, 0.0, bh.UTC)
    assert epc.isostring() == "2017-01-01T00:00:00Z"


def test_isostring_with_decimals(eop):
    # Confirm Before the leap second
    epc = bh.Epoch.from_datetime(2000, 1, 1, 12, 0, 1.23456, 0.0, bh.UTC)
    assert epc.isostring_with_decimals(0) == "2000-01-01T12:00:01Z"
    assert epc.isostring_with_decimals(1) == "2000-01-01T12:00:01.2Z"
    assert epc.isostring_with_decimals(2) == "2000-01-01T12:00:01.23Z"
    assert epc.isostring_with_decimals(3) == "2000-01-01T12:00:01.234Z"


def test_to_string_as_time_system(eop):
    # Confirm Before the leap second
    epc = bh.Epoch.from_datetime(2020, 1, 1, 0, 0, 0.0, 0.0, bh.UTC)
    epc.to_string_as_time_system(bh.UTC) == "2020-01-01 00:00:00.000 UTC"
    epc.to_string_as_time_system(bh.GPS) == "2020-01-01 00:00:18.000 GPS"


def test_gmst(eop):
    epc = bh.Epoch.from_date(2000, 1, 1, bh.UTC)
    assert epc.gmst(bh.AngleFormat.DEGREES) == pytest.approx(99.969, abs=1e-3)

    epc = bh.Epoch.from_date(2000, 1, 1, bh.UTC)
    assert epc.gmst(bh.AngleFormat.RADIANS) == pytest.approx(
        99.969 * math.pi / 180.0, abs=1.0e-3
    )


def test_gast(eop):
    epc = bh.Epoch.from_date(2000, 1, 1, bh.UTC)
    assert epc.gast(bh.AngleFormat.DEGREES) == pytest.approx(99.965, abs=1.0e-3)

    epc = bh.Epoch.from_date(2000, 1, 1, bh.UTC)
    assert epc.gast(bh.AngleFormat.RADIANS) == pytest.approx(
        99.965 * math.pi / 180.0, abs=1.0e-3
    )


def test_ops_add_assign():
    # Test Positive additions of different size
    epc = bh.Epoch.from_date(2022, 1, 31, bh.TAI)
    epc += 1.0
    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()
    assert year == 2022
    assert month == 1
    assert day == 31
    assert hour == 0
    assert minute == 0
    assert second == 1.0
    assert nanosecond == 0.0
    assert epc.time_system == bh.TAI

    epc = bh.Epoch.from_date(2022, 1, 31, bh.TAI)
    epc += 86400.5
    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()
    assert year == 2022
    assert month == 2
    assert day == 1
    assert hour == 0
    assert minute == 0
    assert second == 0.0
    assert nanosecond == 500_000_000.0
    assert epc.time_system == bh.TAI

    epc = bh.Epoch.from_date(2022, 1, 31, bh.TAI)
    epc += 1.23456789e-9
    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()
    assert year == 2022
    assert month == 1
    assert day == 31
    assert hour == 0
    assert minute == 0
    assert second == 0.0
    assert nanosecond == 1.23456789
    assert epc.time_system == bh.TAI

    # Test subtractions of different size
    epc = bh.Epoch.from_date(2022, 1, 31, bh.TAI)
    epc += -1.0
    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()
    assert year == 2022
    assert month == 1
    assert day == 30
    assert hour == 23
    assert minute == 59
    assert second == 59.0
    assert nanosecond == 0.0
    assert epc.time_system == bh.TAI

    epc = bh.Epoch.from_date(2022, 1, 31, bh.TAI)
    epc += -86400.5
    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()
    assert year == 2022
    assert month == 1
    assert day == 29
    assert hour == 23
    assert minute == 59
    assert second == 59.0
    assert nanosecond == 500_000_000.0
    assert epc.time_system == bh.TAI

    # Test types
    epc = bh.Epoch.from_date(2022, 1, 31, bh.TAI)
    epc += 1
    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()
    assert year == 2022
    assert month == 1
    assert day == 31
    assert hour == 0
    assert minute == 0
    assert second == 1.0
    assert nanosecond == 0.0
    assert epc.time_system == bh.TAI

    epc = bh.Epoch.from_date(2022, 1, 31, bh.TAI)
    epc += -1
    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()
    assert year == 2022
    assert month == 1
    assert day == 30
    assert hour == 23
    assert minute == 59
    assert second == 59.0
    assert nanosecond == 0.0
    assert epc.time_system == bh.TAI


def test_ops_sub_assign():
    epc = bh.Epoch.from_date(2022, 1, 31, bh.TAI)
    epc -= 1.23456789e-9
    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()
    assert year == 2022
    assert month == 1
    assert day == 30
    assert hour == 23
    assert minute == 59
    assert second == 59.0
    assert nanosecond == 999_999_999.7654321
    assert epc.time_system == bh.TAI

    # Test subtractions of different size
    epc = bh.Epoch.from_date(2022, 1, 31, bh.TAI)
    epc -= 1.0
    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()
    assert year == 2022
    assert month == 1
    assert day == 30
    assert hour == 23
    assert minute == 59
    assert second == 59.0
    assert nanosecond == 0.0
    assert epc.time_system == bh.TAI

    epc = bh.Epoch.from_date(2022, 1, 31, bh.TAI)
    epc -= 86400.5
    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()
    assert year == 2022
    assert month == 1
    assert day == 29
    assert hour == 23
    assert minute == 59
    assert second == 59.0
    assert nanosecond == 500_000_000.0
    assert epc.time_system == bh.TAI

    # Test types
    epc = bh.Epoch.from_date(2022, 1, 31, bh.TAI)
    epc -= 1
    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()
    assert year == 2022
    assert month == 1
    assert day == 30
    assert hour == 23
    assert minute == 59
    assert second == 59.0
    assert nanosecond == 0.0
    assert epc.time_system == bh.TAI


def test_ops_add():
    # Base epoch
    epc = bh.Epoch.from_date(2022, 1, 31, bh.TAI)

    # Test Positive additions of different size
    epc_2 = epc + 1.0
    (year, month, day, hour, minute, second, nanosecond) = epc_2.to_datetime()
    assert year == 2022
    assert month == 1
    assert day == 31
    assert hour == 0
    assert minute == 0
    assert second == 1.0
    assert nanosecond == 0.0
    assert epc.time_system == bh.TAI

    epc_2 = epc + 86400.5
    (year, month, day, hour, minute, second, nanosecond) = epc_2.to_datetime()
    assert year == 2022
    assert month == 2
    assert day == 1
    assert hour == 0
    assert minute == 0
    assert second == 0.0
    assert nanosecond == 500_000_000.0
    assert epc.time_system == bh.TAI

    epc_2 = epc + 1.23456789e-9
    (year, month, day, hour, minute, second, nanosecond) = epc_2.to_datetime()
    assert year == 2022
    assert month == 1
    assert day == 31
    assert hour == 0
    assert minute == 0
    assert second == 0.0
    assert nanosecond == 1.23456789
    assert epc.time_system == bh.TAI

    # Test subtractions of different size
    epc_2 = epc + -1.0
    (year, month, day, hour, minute, second, nanosecond) = epc_2.to_datetime()
    assert year == 2022
    assert month == 1
    assert day == 30
    assert hour == 23
    assert minute == 59
    assert second == 59.0
    assert nanosecond == 0.0
    assert epc.time_system == bh.TAI

    epc_2 = epc + -86400.5
    (year, month, day, hour, minute, second, nanosecond) = epc_2.to_datetime()
    assert year == 2022
    assert month == 1
    assert day == 29
    assert hour == 23
    assert minute == 59
    assert second == 59.0
    assert nanosecond == 500_000_000.0
    assert epc.time_system == bh.TAI

    # Test types
    epc_2 = epc + 1
    (year, month, day, hour, minute, second, nanosecond) = epc_2.to_datetime()
    assert year == 2022
    assert month == 1
    assert day == 31
    assert hour == 0
    assert minute == 0
    assert second == 1.0
    assert nanosecond == 0.0
    assert epc.time_system == bh.TAI

    epc_2 = epc + -1
    (year, month, day, hour, minute, second, nanosecond) = epc_2.to_datetime()
    assert year == 2022
    assert month == 1
    assert day == 30
    assert hour == 23
    assert minute == 59
    assert second == 59.0
    assert nanosecond == 0.0
    assert epc.time_system == bh.TAI


# def test_ops_sub(eop):
#     # Base epoch
#     epc = bh.Epoch.from_date(2022, 1, 31, bh.TAI)
#
#     # Test subtractions of different size
#     epc_2 = epc - 1.0
#     (year, month, day, hour, minute, second, nanosecond) = epc_2.to_datetime()
#     assert year == 2022
#     assert month == 1
#     assert day == 30
#     assert hour == 23
#     assert minute == 59
#     assert second == 59.0
#     assert nanosecond == 0.0
#     assert epc.time_system == bh.TAI
#
#     epc_2 = epc - 86400.5
#     (year, month, day, hour, minute, second, nanosecond) = epc_2.to_datetime()
#     assert year == 2022
#     assert month == 1
#     assert day == 29
#     assert hour == 23
#     assert minute == 59
#     assert second == 59.0
#     assert nanosecond == 500_000_000.0
#     assert epc.time_system == bh.TAI
#
#     # Test types
#     epc_2 = epc - 1
#     (year, month, day, hour, minute, second, nanosecond) = epc_2.to_datetime()
#     assert year == 2022
#     assert month == 1
#     assert day == 30
#     assert hour == 23
#     assert minute == 59
#     assert second == 59.0
#     assert nanosecond == 0.0
#     assert epc.time_system == bh.TAI


def test_ops_sub_epoch():
    epc_1 = bh.Epoch.from_date(2022, 1, 31, bh.TAI)
    epc_2 = bh.Epoch.from_date(2022, 2, 1, bh.TAI)
    assert epc_2 - epc_1 == 86400.0

    epc_1 = bh.Epoch.from_date(2021, 1, 1, bh.TAI)
    epc_2 = bh.Epoch.from_date(2022, 1, 1, bh.TAI)
    assert epc_2 - epc_1 == 86400.0 * 365.0

    epc_1 = bh.Epoch.from_datetime(2022, 1, 1, 0, 0, 0.0, 0.0, bh.TAI)
    epc_2 = bh.Epoch.from_datetime(2022, 1, 1, 0, 0, 0.0, 1.0, bh.TAI)
    assert epc_2 - epc_1 == 1.0e-9

    epc_1 = bh.Epoch.from_datetime(2022, 1, 1, 0, 0, 0.0, 0.0, bh.TAI)
    epc_2 = bh.Epoch.from_datetime(2022, 1, 2, 1, 1, 1.0, 1.0, bh.TAI)
    assert epc_2 - epc_1 == 86400.0 + 3600.0 + 60.0 + 1.0 + 1.0e-9

    epc_1 = bh.Epoch.from_datetime(2022, 1, 1, 0, 0, 0.0, 0.0, bh.TAI)
    epc_2 = bh.Epoch.from_datetime(2022, 1, 1, 0, 0, 19.0, 0.0, bh.TAI)
    assert epc_2 - epc_1 == 19.0
    assert epc_1 - epc_2 == -19.0
    assert epc_1 - epc_1 == 0.0


def test_eq_epoch(eop):
    epc_1 = bh.Epoch.from_datetime(2022, 1, 1, 12, 23, 59.9, 1.23456789, bh.TAI)
    epc_2 = bh.Epoch.from_datetime(2022, 1, 1, 12, 23, 59.9, 1.23456789, bh.TAI)
    assert epc_1 == epc_2

    epc_1 = bh.Epoch.from_datetime(2022, 1, 1, 12, 23, 59.9, 1.234, bh.TAI)
    epc_2 = bh.Epoch.from_datetime(2022, 1, 1, 12, 23, 59.9, 1.235, bh.TAI)
    assert epc_1 != epc_2

    # Check instant comparison against time systems works
    epc_1 = bh.Epoch.from_datetime(1980, 1, 6, 0, 0, 0.0, 0.0, bh.GPS)
    epc_2 = bh.Epoch.from_datetime(1980, 1, 6, 0, 0, 19.0, 0.0, bh.TAI)
    assert epc_1 == epc_2


def test_cmp_epoch(eop):
    epc_1 = bh.Epoch.from_datetime(2022, 1, 1, 12, 23, 59.9, 1.23456, bh.TAI)
    epc_2 = bh.Epoch.from_datetime(2022, 1, 1, 12, 23, 59.9, 1.23455, bh.TAI)
    assert (epc_1 > epc_2) is True
    assert (epc_1 >= epc_2) is True
    assert (epc_1 < epc_2) is False
    assert (epc_1 <= epc_2) is False

    epc_1 = bh.Epoch.from_datetime(2022, 1, 1, 12, 23, 59.9, 1.23456, bh.TAI)
    epc_2 = bh.Epoch.from_datetime(2022, 1, 1, 12, 23, 59.9, 1.23456, bh.TAI)
    assert (epc_1 > epc_2) is False
    assert (epc_1 >= epc_2) is True
    assert (epc_1 < epc_2) is False
    assert (epc_1 <= epc_2) is True


# def test_nanosecond_addition_stability():
#     pass
#
def test_addition_stability():
    epc = bh.Epoch.from_datetime(2022, 1, 1, 0, 0, 0.0, 0.0, bh.TAI)

    # Advance a year 1 second at a time
    for _ in range(0, 86400 * 365):
        epc += 1.0

    (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime()
    assert year == 2023
    assert month == 1
    assert day == 1
    assert hour == 0
    assert minute == 0
    assert second == 0.0
    assert nanosecond == 0.0


def test_epoch_datetime_accessors(eop):
    """Test individual datetime accessor functions."""
    # Test with a specific date and time - use whole seconds for exact comparison
    epoch = bh.Epoch.from_datetime(2023, 12, 25, 14, 30, 45.0, 123.0, bh.UTC)

    assert epoch.year() == 2023
    assert epoch.month() == 12
    assert epoch.day() == 25
    assert epoch.hour() == 14
    assert epoch.minute() == 30
    assert epoch.second() == 45.0
    assert (
        abs(epoch.nanosecond() - 123.0) < 1.0
    )  # Allow for small precision differences


def test_epoch_datetime_accessors_edge_cases(eop):
    """Test datetime accessors with edge cases."""
    # January 1st, start of day
    epoch_start = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TAI)
    assert epoch_start.year() == 2024
    assert epoch_start.month() == 1
    assert epoch_start.day() == 1
    assert epoch_start.hour() == 0
    assert epoch_start.minute() == 0
    assert epoch_start.second() == 0.0
    assert epoch_start.nanosecond() == 0.0

    # December 31st, end of day
    epoch_end = bh.Epoch.from_datetime(2023, 12, 31, 23, 59, 59.0, 999999999.0, bh.GPS)
    assert epoch_end.year() == 2023
    assert epoch_end.month() == 12
    assert epoch_end.day() == 31
    assert epoch_end.hour() == 23
    assert epoch_end.minute() == 59
    assert epoch_end.second() == 59.0
    assert abs(epoch_end.nanosecond() - 999999999.0) < 1.0


def test_epoch_datetime_accessors_different_time_systems(eop):
    """Test that accessors work correctly for different time systems."""
    test_cases = [bh.UTC, bh.TAI, bh.GPS, bh.TT]

    for time_system in test_cases:
        epoch = bh.Epoch.from_datetime(2020, 6, 15, 12, 0, 0.0, 0.0, time_system)

        # The accessors should return the components in the epoch's time system
        assert epoch.year() == 2020
        assert epoch.month() == 6
        assert epoch.day() == 15
        assert epoch.hour() == 12
        assert epoch.minute() == 0
        assert epoch.second() == 0.0
        assert abs(epoch.nanosecond() - 0.0) < 1.0


def test_epoch_datetime_accessors_leap_year(eop):
    """Test datetime accessors with leap year date."""
    # Test leap year date (February 29, 2020)
    leap_epoch = bh.Epoch.from_datetime(2020, 2, 29, 8, 45, 30.0, 456.0, bh.UTC)

    assert leap_epoch.year() == 2020
    assert leap_epoch.month() == 2
    assert leap_epoch.day() == 29
    assert leap_epoch.hour() == 8
    assert leap_epoch.minute() == 45
    assert leap_epoch.second() == 30.0
    assert abs(leap_epoch.nanosecond() - 456.0) < 1.0


def test_epoch_datetime_accessors_vs_to_datetime(eop):
    """Test that individual accessors match to_datetime() results."""
    epochs = [
        bh.Epoch.from_datetime(2023, 1, 1, 0, 0, 0.0, 0.0, bh.UTC),
        bh.Epoch.from_datetime(2024, 6, 15, 12, 30, 45.123, 456789.0, bh.TAI),
        bh.Epoch.from_datetime(2000, 12, 31, 23, 59, 59.999, 999999999.0, bh.GPS),
    ]

    for epoch in epochs:
        (year, month, day, hour, minute, second, nanosecond) = epoch.to_datetime()

        assert epoch.year() == year
        assert epoch.month() == month
        assert epoch.day() == day
        assert epoch.hour() == hour
        assert epoch.minute() == minute
        assert epoch.second() == second
        assert epoch.nanosecond() == nanosecond


def test_epoch_from_day_of_year(eop):
    """Test creating Epoch from day-of-year."""
    # Test January 1st (day 1.0)
    epoch_jan1 = bh.Epoch.from_day_of_year(2023, 1.0, bh.UTC)
    assert epoch_jan1.year() == 2023
    assert epoch_jan1.month() == 1
    assert epoch_jan1.day() == 1
    assert epoch_jan1.hour() == 0
    assert epoch_jan1.minute() == 0
    assert epoch_jan1.second() == 0.0

    # Test January 2nd (day 2.0)
    epoch_jan2 = bh.Epoch.from_day_of_year(2023, 2.0, bh.UTC)
    assert epoch_jan2.year() == 2023
    assert epoch_jan2.month() == 1
    assert epoch_jan2.day() == 2
    assert epoch_jan2.hour() == 0
    assert epoch_jan2.minute() == 0
    assert epoch_jan2.second() == 0.0

    # Test day 100.5 (should be around April 10th, noon)
    epoch_100_5 = bh.Epoch.from_day_of_year(2023, 100.5, bh.UTC)
    assert epoch_100_5.year() == 2023
    assert epoch_100_5.month() == 4
    assert epoch_100_5.day() == 10
    assert epoch_100_5.hour() == 12
    assert epoch_100_5.minute() == 0
    assert epoch_100_5.second() == 0.0


def test_epoch_from_day_of_year_leap_year(eop):
    """Test day-of-year in leap years."""
    # Test February 29th in a leap year (day 60.0)
    epoch_feb29 = bh.Epoch.from_day_of_year(2020, 60.0, bh.UTC)
    assert epoch_feb29.year() == 2020
    assert epoch_feb29.month() == 2
    assert epoch_feb29.day() == 29
    assert epoch_feb29.hour() == 0

    # Test March 1st in a leap year (day 61.0)
    epoch_mar1 = bh.Epoch.from_day_of_year(2020, 61.0, bh.UTC)
    assert epoch_mar1.year() == 2020
    assert epoch_mar1.month() == 3
    assert epoch_mar1.day() == 1


def test_epoch_from_day_of_year_fractional(eop):
    """Test fractional day-of-year values."""
    # Test fractional day (day 1.25 = January 1st, 6:00 AM)
    epoch_frac = bh.Epoch.from_day_of_year(2023, 1.25, bh.UTC)
    assert epoch_frac.year() == 2023
    assert epoch_frac.month() == 1
    assert epoch_frac.day() == 1
    assert epoch_frac.hour() == 6
    assert epoch_frac.minute() == 0
    assert epoch_frac.second() == 0.0

    # Test fractional day (day 1.75 = January 1st, 6:00 PM)
    epoch_frac2 = bh.Epoch.from_day_of_year(2023, 1.75, bh.UTC)
    assert epoch_frac2.year() == 2023
    assert epoch_frac2.month() == 1
    assert epoch_frac2.day() == 1
    assert epoch_frac2.hour() == 18
    assert epoch_frac2.minute() == 0
    assert epoch_frac2.second() == 0.0


def test_epoch_from_day_of_year_time_systems(eop):
    """Test day-of-year with different time systems."""
    test_cases = [bh.UTC, bh.TAI, bh.GPS, bh.TT]

    for time_system in test_cases:
        epoch = bh.Epoch.from_day_of_year(2023, 100.0, time_system)
        assert epoch.year() == 2023
        assert epoch.month() == 4
        assert epoch.day() == 10
        assert epoch.time_system == time_system


# Test __new__ constructor with *args and **kwargs
def test_epoch_new_from_date_components(eop):
    """Test Epoch() constructor with year, month, day."""
    epc = bh.Epoch(2024, 1, 1)

    year, month, day, hour, minute, second, nanosecond = epc.to_datetime()
    assert year == 2024
    assert month == 1
    assert day == 1
    assert hour == 0
    assert minute == 0
    assert second == 0.0
    assert nanosecond == 0.0
    assert epc.time_system == bh.UTC


def test_epoch_new_from_full_datetime(eop):
    """Test Epoch() constructor with full datetime components."""
    epc = bh.Epoch(2024, 1, 1, 12, 30, 45.5, 0.0)

    year, month, day, hour, minute, second, nanosecond = epc.to_datetime()
    assert year == 2024
    assert month == 1
    assert day == 1
    assert hour == 12
    assert minute == 30
    # Fractional seconds get split into second + nanosecond
    assert second == 45.0
    assert nanosecond == pytest.approx(500000000.0, rel=1e-6)
    assert epc.time_system == bh.UTC


def test_epoch_new_with_time_system(eop):
    """Test Epoch() constructor with time_system kwarg."""
    epc = bh.Epoch(2024, 1, 1, 12, 0, 0.0, 0.0, time_system=bh.GPS)

    assert epc.time_system == bh.GPS
    year, month, day, hour, minute, second, nanosecond = epc.to_datetime()
    assert year == 2024
    assert month == 1
    assert day == 1
    assert hour == 12


def test_epoch_new_from_string(eop):
    """Test Epoch() constructor from string."""
    epc = bh.Epoch("2024-01-01 12:00:00.000 UTC")

    year, month, day, hour, minute, second, nanosecond = epc.to_datetime()
    assert year == 2024
    assert month == 1
    assert day == 1
    assert hour == 12
    assert minute == 0
    assert second == 0.0
    assert epc.time_system == bh.UTC


def test_epoch_new_from_string_iso_z(eop):
    """Test Epoch() constructor from ISO 8601 Z format."""
    epc = bh.Epoch("2024-01-01T12:00:00Z")

    year, month, day, hour, minute, second, nanosecond = epc.to_datetime()
    assert year == 2024
    assert month == 1
    assert day == 1
    assert hour == 12
    assert minute == 0
    assert second == 0.0


def test_epoch_new_from_datetime(eop):
    """Test Epoch() constructor from Python datetime."""
    dt = datetime(2024, 1, 1, 12, 0, 0)
    epc = bh.Epoch(dt)

    year, month, day, hour, minute, second, nanosecond = epc.to_datetime()
    assert year == 2024
    assert month == 1
    assert day == 1
    assert hour == 12
    assert minute == 0
    assert second == 0.0
    assert epc.time_system == bh.UTC


def test_epoch_new_from_datetime_with_time_system(eop):
    """Test Epoch() constructor from Python datetime with custom time system."""
    dt = datetime(2024, 6, 15, 14, 30, 45)
    epc = bh.Epoch(dt, time_system=bh.TAI)

    assert epc.time_system == bh.TAI
    year, month, day, hour, minute, second, nanosecond = epc.to_datetime()
    assert year == 2024
    assert month == 6
    assert day == 15


def test_epoch_new_copy_constructor(eop):
    """Test Epoch() copy constructor."""
    epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.GPS)
    epc2 = bh.Epoch(epc1)

    assert epc2.time_system == bh.GPS
    assert epc2 == epc1


def test_epoch_new_partial_datetime_4_args(eop):
    """Test Epoch() with year, month, day, hour."""
    epc = bh.Epoch(2024, 1, 1, 12)

    year, month, day, hour, minute, second, nanosecond = epc.to_datetime()
    assert year == 2024
    assert month == 1
    assert day == 1
    assert hour == 12
    assert minute == 0
    assert second == 0.0


def test_epoch_new_partial_datetime_5_args(eop):
    """Test Epoch() with year, month, day, hour, minute."""
    epc = bh.Epoch(2024, 1, 1, 12, 30)

    year, month, day, hour, minute, second, nanosecond = epc.to_datetime()
    assert year == 2024
    assert month == 1
    assert day == 1
    assert hour == 12
    assert minute == 30
    assert second == 0.0


def test_epoch_new_partial_datetime_6_args(eop):
    """Test Epoch() with year, month, day, hour, minute, second."""
    epc = bh.Epoch(2024, 1, 1, 12, 30, 45.5)

    year, month, day, hour, minute, second, nanosecond = epc.to_datetime()
    assert year == 2024
    assert month == 1
    assert day == 1
    assert hour == 12
    assert minute == 30
    # Fractional seconds get split into second + nanosecond
    assert second == 45.0
    assert nanosecond == pytest.approx(500000000.0, rel=1e-6)


def test_epoch_new_error_no_args(eop):
    """Test that Epoch() raises error with no arguments."""
    with pytest.raises(ValueError, match="No arguments provided"):
        bh.Epoch()


def test_epoch_new_error_invalid_arg_count(eop):
    """Test that Epoch() raises error with invalid number of args."""
    with pytest.raises(TypeError, match="Invalid number of arguments"):
        bh.Epoch(2024, 1)  # 2 args is invalid


def test_epoch_new_error_invalid_single_arg_type(eop):
    """Test that Epoch() raises error with invalid single argument type."""
    with pytest.raises(
        TypeError,
        match="Single argument must be str, float/int \\(MJD/JD\\), datetime, or Epoch",
    ):
        bh.Epoch([123.456])  # list is not valid


class TestEpochFloatMJD:
    """Test Epoch initialization from MJD values (float < 1000000)."""

    def test_mjd_float(self):
        """Test initialization from float MJD value."""
        mjd = 60310.5
        epc1 = bh.Epoch(mjd)
        epc2 = bh.Epoch.from_mjd(mjd, bh.TimeSystem.UTC)

        assert epc1 == epc2
        assert epc1.mjd() == pytest.approx(mjd)

    def test_mjd_int(self):
        """Test initialization from integer MJD value."""
        mjd = 60310
        epc1 = bh.Epoch(mjd)
        epc2 = bh.Epoch.from_mjd(float(mjd), bh.TimeSystem.UTC)

        assert epc1 == epc2
        assert epc1.mjd() == pytest.approx(float(mjd))

    def test_mjd_with_time_system(self):
        """Test MJD initialization with explicit time system."""
        mjd = 60310.5

        # Test with different time systems
        epc_utc = bh.Epoch(mjd, time_system=bh.TimeSystem.UTC)
        epc_tai = bh.Epoch(mjd, time_system=bh.TimeSystem.TAI)
        epc_gps = bh.Epoch(mjd, time_system=bh.TimeSystem.GPS)

        assert epc_utc == bh.Epoch.from_mjd(mjd, bh.TimeSystem.UTC)
        assert epc_tai == bh.Epoch.from_mjd(mjd, bh.TimeSystem.TAI)
        assert epc_gps == bh.Epoch.from_mjd(mjd, bh.TimeSystem.GPS)

        # Different time systems should have different internal representations
        assert epc_utc != epc_tai
        assert epc_utc != epc_gps

    def test_mjd_boundary_low(self):
        """Test MJD at low boundary values."""
        # Test small positive MJD
        mjd = 1.0
        epc = bh.Epoch(mjd)
        assert epc.mjd() == pytest.approx(mjd)

        # Test zero MJD (1858-11-17T00:00:00 UTC)
        mjd = 0.0
        epc = bh.Epoch(mjd)
        assert epc.mjd() == pytest.approx(mjd)

    def test_mjd_boundary_high(self):
        """Test MJD at high boundary near threshold."""
        # Test MJD just below threshold (999999.9)
        mjd = 999999.9
        epc1 = bh.Epoch(mjd)
        epc2 = bh.Epoch.from_mjd(mjd, bh.TimeSystem.UTC)

        assert epc1 == epc2
        assert epc1.mjd() == pytest.approx(mjd)


class TestEpochFloatJD:
    """Test Epoch initialization from JD values (float >= 1000000)."""

    def test_jd_float(self):
        """Test initialization from float JD value."""
        jd = 2460310.5
        epc1 = bh.Epoch(jd)
        epc2 = bh.Epoch.from_jd(jd, bh.TimeSystem.UTC)

        assert epc1 == epc2
        assert epc1.jd() == pytest.approx(jd)

    def test_jd_int(self):
        """Test initialization from integer JD value."""
        jd = 2460310
        epc1 = bh.Epoch(jd)
        epc2 = bh.Epoch.from_jd(float(jd), bh.TimeSystem.UTC)

        assert epc1 == epc2
        assert epc1.jd() == pytest.approx(float(jd))

    def test_jd_with_time_system(self):
        """Test JD initialization with explicit time system."""
        jd = 2460310.5

        # Test with different time systems
        epc_utc = bh.Epoch(jd, time_system=bh.TimeSystem.UTC)
        epc_tai = bh.Epoch(jd, time_system=bh.TimeSystem.TAI)
        epc_gps = bh.Epoch(jd, time_system=bh.TimeSystem.GPS)

        assert epc_utc == bh.Epoch.from_jd(jd, bh.TimeSystem.UTC)
        assert epc_tai == bh.Epoch.from_jd(jd, bh.TimeSystem.TAI)
        assert epc_gps == bh.Epoch.from_jd(jd, bh.TimeSystem.GPS)

        # Different time systems should have different internal representations
        assert epc_utc != epc_tai
        assert epc_utc != epc_gps

    def test_jd_boundary_threshold(self):
        """Test JD exactly at threshold value."""
        # Test JD exactly at threshold (1000000.0)
        jd = 1000000.0
        epc1 = bh.Epoch(jd)
        epc2 = bh.Epoch.from_jd(jd, bh.TimeSystem.UTC)

        assert epc1 == epc2
        assert epc1.jd() == pytest.approx(jd)

    def test_jd_typical_values(self):
        """Test JD with typical astronomical values."""
        # Modern epoch (2024-01-01)
        jd = 2460310.5
        epc = bh.Epoch(jd)
        assert epc.jd() == pytest.approx(jd)

        # J2000 epoch (2000-01-01T12:00:00)
        jd_j2000 = 2451545.0
        epc = bh.Epoch(jd_j2000)
        assert epc.jd() == pytest.approx(jd_j2000)


def test_epoch_float_mjd_jd_equivalence():
    """Test that same epoch can be created via MJD or JD."""
    # MJD = JD - 2400000.5
    mjd = 60310.5
    jd = mjd + 2400000.5

    epc_mjd = bh.Epoch(mjd)
    epc_jd = bh.Epoch(jd)

    # Should represent the same instant in time
    assert epc_mjd == epc_jd
    assert epc_mjd.mjd() == pytest.approx(mjd)
    assert epc_jd.jd() == pytest.approx(jd)


def test_epoch_float_roundtrip_mjd():
    """Test roundtrip: Epoch(mjd) -> jd() -> Epoch(jd) -> mjd()."""
    original_mjd = 60310.5

    epc1 = bh.Epoch(original_mjd)
    jd = epc1.jd()
    epc2 = bh.Epoch(jd)
    final_mjd = epc2.mjd()

    assert original_mjd == pytest.approx(final_mjd)
    assert epc1 == epc2


def test_epoch_float_defaults_to_utc():
    """Test that float initialization defaults to UTC time system."""
    mjd = 60310.5
    epc = bh.Epoch(mjd)

    assert epc.time_system == bh.TimeSystem.UTC


def test_epoch_float_negative_mjd():
    """Test that negative values are treated as MJD (< 1000000)."""
    mjd = -100.0  # Valid MJD before MJD epoch
    epc1 = bh.Epoch(mjd)
    epc2 = bh.Epoch.from_mjd(mjd, bh.TimeSystem.UTC)

    assert epc1 == epc2
    assert epc1.mjd() == pytest.approx(mjd)


def test_epoch_float_zero():
    """Test that 0.0 is treated as MJD."""
    epc1 = bh.Epoch(0.0)
    epc2 = bh.Epoch.from_mjd(0.0, bh.TimeSystem.UTC)

    assert epc1 == epc2
    assert epc1.mjd() == pytest.approx(0.0)


def test_epoch_float_very_large_jd():
    """Test very large JD values."""
    jd = 3000000.0  # Far future
    epc1 = bh.Epoch(jd)
    epc2 = bh.Epoch.from_jd(jd, bh.TimeSystem.UTC)

    assert epc1 == epc2
    assert epc1.jd() == pytest.approx(jd)


def test_epoch_compatibility_float_vs_string():
    """Test that float and string initialization give same result."""
    # 2024-01-01 12:00:00 UTC
    mjd = 60310.5
    string_repr = "2024-01-01T12:00:00Z"

    epc_float = bh.Epoch(mjd)
    epc_string = bh.Epoch(string_repr)

    # Should be very close (string parsing has some limitations)
    assert abs(epc_float.mjd() - epc_string.mjd()) < 1e-6


def test_epoch_compatibility_float_vs_datetime_components():
    """Test that float and datetime component initialization match."""
    # 2024-01-01 00:00:00 UTC = MJD 60310.0
    mjd = 60310.0

    epc_float = bh.Epoch(mjd)
    epc_datetime = bh.Epoch(2024, 1, 1, 0, 0, 0.0, 0.0)

    # Should be very close (may have small numerical differences)
    assert epc_float.mjd() == pytest.approx(epc_datetime.mjd(), abs=1e-6)
    assert epc_float.mjd() == pytest.approx(60310.0)
