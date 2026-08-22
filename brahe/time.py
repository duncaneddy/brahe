"""
Time Module

Time systems, epochs, and time conversions.

This module provides:
- Epoch: Core time representation supporting multiple time systems (UTC, TAI, GPS, TT, UT1)
- TimeSystem: Enumeration of supported time systems
- TimeRange: Time range and iteration utilities
- Time conversion functions between different representations (MJD, JD, datetime)
- Time system offset calculations
"""

from brahe._brahe import (
    BDT,
    # Time system constants
    GPS,
    GST,
    TAI,
    TCB,
    TCG,
    TDB,
    TT,
    UT1,
    UTC,
    # Core classes
    Epoch,
    TimeRange,
    TimeSystem,
    datetime_to_jd,
    datetime_to_mjd,
    jd_to_datetime,
    # Conversion functions
    mjd_to_datetime,
    time_system_offset_for_datetime,
    time_system_offset_for_jd,
    time_system_offset_for_mjd,
)

__all__ = [
    "BDT",
    # Time system constants
    "GPS",
    "GST",
    "TAI",
    "TCB",
    "TCG",
    "TDB",
    "TT",
    "UT1",
    "UTC",
    # Core classes
    "Epoch",
    "TimeRange",
    "TimeSystem",
    "datetime_to_jd",
    "datetime_to_mjd",
    "jd_to_datetime",
    # Conversion functions
    "mjd_to_datetime",
    "time_system_offset_for_datetime",
    "time_system_offset_for_jd",
    "time_system_offset_for_mjd",
]
