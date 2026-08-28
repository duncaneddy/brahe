"""
CCSDS Orbit Data Message (ODM), Conjunction Data Message (CDM), and
Attitude Data Message (ADM) support.

Provides parsing and writing of CCSDS standard messages:

- OEM: Orbit Ephemeris Message (time-series state vectors)
- OMM: Orbit Mean-elements Message (SGP4/TLE data)
- OPM: Orbit Parameter Message (single state vector)
- CDM: Conjunction Data Message (conjunction assessment)
- APM: Attitude Parameter Message (single-epoch attitude state)

Supported formats: KVN (text), XML, JSON.
"""

from brahe._brahe import (
    APM,
    CDM,
    OEM,
    OMM,
    OPM,
    APMAngularVelocity,
    APMEulerState,
    APMInertia,
    APMManeuver,
    APMQuaternionState,
    APMSpin,
    CDMObject,
    CDMRTNCovariance,
    CDMStateVector,
    OEMSegment,
    OEMSegments,
    OEMStates,
    OEMStateVector,
    OPMManeuver,
    OPMManeuvers,
)

__all__ = [
    "APM",
    "CDM",
    "OEM",
    "OMM",
    "OPM",
    "APMAngularVelocity",
    "APMEulerState",
    "APMInertia",
    "APMManeuver",
    "APMQuaternionState",
    "APMSpin",
    "CDMObject",
    "CDMRTNCovariance",
    "CDMStateVector",
    "OEMSegment",
    "OEMSegments",
    "OEMStateVector",
    "OEMStates",
    "OPMManeuver",
    "OPMManeuvers",
]
