"""GMAT benchmark implementations for time-system conversions.

GMAT exposes a TimeSystemConverter singleton via
`gmat.TimeSystemConverter.Instance()`. The Instance() call returns the same
singleton on every call, so it is hoisted out of the timed region.

Time-system enum codes (tcv.UTC, .TAI, .TT, .UT1) are integer values hoisted
before the timed loop.  GPS is not a native GMAT time system; it is derived
as GPS = TAI − 19 s (the constant offset since 1980-01-06).

GMAT uses a non-standard "A1 Modified Julian Date" epoch rooted at
1941-01-05 rather than the standard MJD epoch (1858-11-17).  The offset to
standard Julian Date is:

    JD = A1MJD + 2430000.0

All result values are returned as Julian Dates so they are directly comparable
to the brahe Python/Rust/Java implementations which also return JD.

Input parameters carry ``datetimes`` dicts (year, month, day, hour, minute,
second, nanosecond).  These are converted to GMAT Gregorian strings
("DD Mon YYYY HH:MM:SS.sss") via :func:`_dt_to_gmat_greg`.  GMAT's Gregorian
parser accepts millisecond precision only, so nanosecond values below 0.5 ms
are silently truncated — resulting in a maximum accuracy loss of ~0.5 ms
(~5.8e-9 days) relative to implementations with full nanosecond precision.
"""

from benchmarks.comparative.implementations.gmat.base import (
    build_task_result,
    time_iterations,
)

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Offset from GMAT A1MJD to standard Julian Date:  JD = A1MJD + _A1MJD_TO_JD
_A1MJD_TO_JD = 2430000.0

# GPS = TAI - 19 seconds (fixed constant since the GPS epoch 1980-01-06)
_GPS_TAI_OFFSET_DAYS = -19.0 / 86400.0


def _dt_to_gmat_greg(dt: dict) -> str:
    """Convert a datetime dict to a GMAT Gregorian string.

    Format: ``DD Mon YYYY HH:MM:SS.sss`` (millisecond precision — GMAT limit).
    """
    total_sec = dt["second"] + dt["nanosecond"] * 1e-9
    return (
        f"{dt['day']:02d} {_MONTHS[dt['month'] - 1]} {dt['year']:04d} "
        f"{dt['hour']:02d}:{dt['minute']:02d}:{total_sec:06.3f}"
    )


def _tcv():
    import gmatpy as gmat
    return gmat.TimeSystemConverter.Instance()


def epoch_creation(params: dict, iterations: int):
    """Construct epoch objects from datetime components; return UTC Julian Dates."""
    import gmatpy as gmat

    datetimes = params["datetimes"]
    greg_strings = [_dt_to_gmat_greg(dt) for dt in datetimes]
    tcv = gmat.TimeSystemConverter.Instance()

    def run():
        return [
            tcv.ConvertGregorianToMjd(g) + _A1MJD_TO_JD
            for g in greg_strings
        ]

    times, results = time_iterations(run, iterations)
    return build_task_result("time.epoch_creation", iterations, times, results)


def utc_to_tai(params: dict, iterations: int):
    """Convert UTC datetimes to TAI Julian Dates via TimeSystemConverter."""
    tcv = _tcv()
    src = tcv.UTC
    dst = tcv.TAI

    datetimes = params["datetimes"]
    utc_a1mjds = [
        tcv.ConvertGregorianToMjd(_dt_to_gmat_greg(dt)) for dt in datetimes
    ]

    def run():
        return [
            tcv.Convert(mjd, src, dst) + _A1MJD_TO_JD
            for mjd in utc_a1mjds
        ]

    times, results = time_iterations(run, iterations)
    return build_task_result("time.utc_to_tai", iterations, times, results)


def utc_to_tt(params: dict, iterations: int):
    """Convert UTC datetimes to TT Julian Dates via TimeSystemConverter."""
    tcv = _tcv()
    src = tcv.UTC
    dst = tcv.TT

    datetimes = params["datetimes"]
    utc_a1mjds = [
        tcv.ConvertGregorianToMjd(_dt_to_gmat_greg(dt)) for dt in datetimes
    ]

    def run():
        return [
            tcv.Convert(mjd, src, dst) + _A1MJD_TO_JD
            for mjd in utc_a1mjds
        ]

    times, results = time_iterations(run, iterations)
    return build_task_result("time.utc_to_tt", iterations, times, results)


def utc_to_gps(params: dict, iterations: int):
    """Convert UTC datetimes to GPS Julian Dates.

    GMAT does not expose GPS as a native time system.  GPS is derived from TAI
    by subtracting the constant 19-second offset (GPS = TAI − 19 s), which has
    been fixed since the GPS epoch 1980-01-06.
    """
    tcv = _tcv()
    src = tcv.UTC
    dst = tcv.TAI

    datetimes = params["datetimes"]
    utc_a1mjds = [
        tcv.ConvertGregorianToMjd(_dt_to_gmat_greg(dt)) for dt in datetimes
    ]

    def run():
        return [
            tcv.Convert(mjd, src, dst) + _GPS_TAI_OFFSET_DAYS + _A1MJD_TO_JD
            for mjd in utc_a1mjds
        ]

    times, results = time_iterations(run, iterations)
    return build_task_result("time.utc_to_gps", iterations, times, results)


def utc_to_ut1(params: dict, iterations: int):
    """Convert UTC datetimes to UT1 Julian Dates via TimeSystemConverter."""
    tcv = _tcv()
    src = tcv.UTC
    dst = tcv.UT1

    datetimes = params["datetimes"]
    utc_a1mjds = [
        tcv.ConvertGregorianToMjd(_dt_to_gmat_greg(dt)) for dt in datetimes
    ]

    def run():
        return [
            tcv.Convert(mjd, src, dst) + _A1MJD_TO_JD
            for mjd in utc_a1mjds
        ]

    times, results = time_iterations(run, iterations)
    return build_task_result("time.utc_to_ut1", iterations, times, results)
