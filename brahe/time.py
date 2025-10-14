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
    # Core classes
    Epoch,
    TimeSystem,
    TimeRange,
    # Time system constants
    GPS,
    TAI,
    TT,
    UTC,
    UT1,
    # Conversion functions
    mjd_to_datetime,
    datetime_to_mjd,
    jd_to_datetime,
    datetime_to_jd,
    time_system_offset_for_mjd,
    time_system_offset_for_jd,
    time_system_offset_for_datetime,
)

__all__ = [
    # Core classes
    "Epoch",
    "TimeSystem",
    "TimeRange",
    # Time system constants
    "GPS",
    "TAI",
    "TT",
    "UTC",
    "UT1",
    # Conversion functions
    "mjd_to_datetime",
    "datetime_to_mjd",
    "jd_to_datetime",
    "datetime_to_jd",
    "time_system_offset_for_mjd",
    "time_system_offset_for_jd",
    "time_system_offset_for_datetime",
]
