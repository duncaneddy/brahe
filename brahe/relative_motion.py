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

The LVLH frame (CCSDS/SANA definition) has Z toward nadir, Y opposite the orbit normal, and
X = Y × Z.
"""

from brahe._brahe import (
    covariance_eci_to_lvlh,
    covariance_eci_to_rtn,
    covariance_lvlh_to_eci,
    covariance_rtn_to_eci,
    jacobian_eci_to_lvlh,
    jacobian_eci_to_rtn,
    jacobian_lvlh_to_eci,
    jacobian_rtn_to_eci,
    omega_lvlh,
    omega_rtn,
    rotation_eci_to_lvlh,
    rotation_eci_to_rtn,
    rotation_lvlh_to_eci,
    rotation_rtn_to_eci,
    state_eci_to_lvlh,
    state_eci_to_roe,
    state_eci_to_rtn,
    state_lvlh_to_eci,
    state_oe_to_roe,
    state_roe_to_eci,
    state_roe_to_oe,
    state_rtn_to_eci,
)

__all__ = [
    "covariance_eci_to_lvlh",
    "covariance_eci_to_rtn",
    "covariance_lvlh_to_eci",
    "covariance_rtn_to_eci",
    "jacobian_eci_to_lvlh",
    "jacobian_eci_to_rtn",
    "jacobian_lvlh_to_eci",
    "jacobian_rtn_to_eci",
    "omega_lvlh",
    "omega_rtn",
    "rotation_eci_to_lvlh",
    "rotation_eci_to_rtn",
    "rotation_lvlh_to_eci",
    "rotation_rtn_to_eci",
    "state_eci_to_lvlh",
    "state_eci_to_roe",
    "state_eci_to_rtn",
    "state_lvlh_to_eci",
    "state_oe_to_roe",
    "state_roe_to_eci",
    "state_roe_to_oe",
    "state_rtn_to_eci",
]
