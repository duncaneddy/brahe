"""
CCSDS Orbit Data Message (ODM) support.

Provides parsing and writing of CCSDS standard orbit data messages:

- OEM: Orbit Ephemeris Message (time-series state vectors)
- OMM: Orbit Mean-elements Message (SGP4/TLE data)
- OPM: Orbit Parameter Message (single state vector)

Supported formats: KVN (text), XML, JSON.
"""

from brahe._brahe import (
    CDM,
    CDMObject,
    CDMRTNCovariance,
    CDMStateVector,
    OEM,
    OEMSegment,
    OEMSegments,
    OEMStateVector,
    OEMStates,
    OMM,
    OPM,
    OPMManeuver,
    OPMManeuvers,
)

__all__ = [
    "CDM",
    "CDMObject",
    "CDMRTNCovariance",
    "CDMStateVector",
    "OEM",
    "OEMSegment",
    "OEMSegments",
    "OEMStateVector",
    "OEMStates",
    "OMM",
    "OPM",
    "OPMManeuver",
    "OPMManeuvers",
]
