"""
Relative Motion Module

Satellite relative motion and orbital reference frames.

This module provides transformations between inertial frames and orbital
reference frames such as RTN (Radial-Tangential-Normal).

The RTN frame is an orbital reference frame defined as:
- R (Radial): Points from Earth's center to satellite position
- T (Tangential): Along-track direction in orbital plane
- N (Normal): Perpendicular to orbital plane (angular momentum direction)

Functions are provided for:
- Rotation matrices between ECI and RTN frames
- (Future) Relative motion dynamics (Clohessy-Wiltshire equations, etc.)
"""

from brahe._brahe import (
    rotation_rtn_to_eci,
    rotation_eci_to_rtn,
    state_rtn_to_eci,
    state_eci_to_rtn,
    state_oe_to_roe,
    state_roe_to_oe,
)

__all__ = [
    "rotation_rtn_to_eci",
    "rotation_eci_to_rtn",
    "state_rtn_to_eci",
    "state_eci_to_rtn",
    "state_oe_to_roe",
    "state_roe_to_oe",
]
