"""
CCSDS Orbit Data Message (ODM), Conjunction Data Message (CDM), and
Attitude Data Message (ADM) support.

Provides parsing and writing of CCSDS standard messages:

- OEM: Orbit Ephemeris Message (time-series state vectors)
- OMM: Orbit Mean-elements Message (SGP4/TLE data)
- OPM: Orbit Parameter Message (single state vector)
- CDM: Conjunction Data Message (conjunction assessment)
- APM: Attitude Parameter Message (single-epoch attitude state)
- AEM: Attitude Ephemeris Message (time-series attitude data)

Supported formats: KVN (text), XML, JSON.
"""

from brahe._brahe import (
    AEM,
    APM,
    CDM,
    OEM,
    OMM,
    OPM,
    AEMAttitudeState,
    AEMSegment,
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
    "AEM",
    "APM",
    "CDM",
    "OEM",
    "OMM",
    "OPM",
    "AEMAttitudeState",
    "AEMSegment",
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
