"""
Relative Motion Module

Satellite relative motion and orbital reference frames.

This module provides transformations between inertial frames and orbital
reference frames such as RTN, LVLH, and NTW.

The RTN frame is an orbital reference frame defined as:
- R (Radial): Points from Earth's center to satellite position
- T (Tangential): Along-track direction in orbital plane
- N (Normal): Perpendicular to orbital plane (angular momentum direction)

The LVLH frame (CCSDS/SANA definition) has Z toward nadir, Y opposite the
orbit normal, and X = Y × Z.

The NTW frame has Y along velocity, Z along the orbit normal, and
X = Y × Z; it coincides with RTN on a circular orbit.

Functions are provided for:
- Rotation matrices between ECI and RTN frames
- Rotation matrices between ECI and LVLH frames
- Rotation matrices between ECI and NTW frames
- (Future) Relative motion dynamics (Clohessy-Wiltshire equations, etc.)
"""

from brahe._brahe import (
    covariance_eci_to_lvlh,
    covariance_eci_to_ntw,
    covariance_eci_to_rtn,
    covariance_inertial_to_ntw_for_body,
    covariance_lvlh_to_eci,
    covariance_ntw_to_eci,
    covariance_ntw_to_inertial_for_body,
    covariance_rtn_to_eci,
    jacobian_eci_to_lvlh,
    jacobian_eci_to_ntw,
    jacobian_eci_to_rtn,
    jacobian_inertial_to_ntw_for_body,
    jacobian_lvlh_to_eci,
    jacobian_ntw_to_eci,
    jacobian_ntw_to_inertial_for_body,
    jacobian_rtn_to_eci,
    omega_lvlh,
    omega_ntw,
    omega_ntw_for_body,
    omega_rtn,
    rotation_eci_to_lvlh,
    rotation_eci_to_ntw,
    rotation_eci_to_rtn,
    rotation_lvlh_to_eci,
    rotation_ntw_to_eci,
    rotation_rtn_to_eci,
    state_eci_to_lvlh,
    state_eci_to_ntw,
    state_eci_to_roe,
    state_eci_to_rtn,
    state_inertial_to_ntw_for_body,
    state_lvlh_to_eci,
    state_ntw_to_eci,
    state_ntw_to_inertial_for_body,
    state_oe_to_roe,
    state_roe_to_eci,
    state_roe_to_oe,
    state_rtn_to_eci,
)

__all__ = [
    "covariance_eci_to_lvlh",
    "covariance_eci_to_ntw",
    "covariance_eci_to_rtn",
    "covariance_inertial_to_ntw_for_body",
    "covariance_lvlh_to_eci",
    "covariance_ntw_to_eci",
    "covariance_ntw_to_inertial_for_body",
    "covariance_rtn_to_eci",
    "jacobian_eci_to_lvlh",
    "jacobian_eci_to_ntw",
    "jacobian_eci_to_rtn",
    "jacobian_inertial_to_ntw_for_body",
    "jacobian_lvlh_to_eci",
    "jacobian_ntw_to_eci",
    "jacobian_ntw_to_inertial_for_body",
    "jacobian_rtn_to_eci",
    "omega_lvlh",
    "omega_ntw",
    "omega_ntw_for_body",
    "omega_rtn",
    "rotation_eci_to_lvlh",
    "rotation_eci_to_ntw",
    "rotation_eci_to_rtn",
    "rotation_lvlh_to_eci",
    "rotation_ntw_to_eci",
    "rotation_rtn_to_eci",
    "state_eci_to_lvlh",
    "state_eci_to_ntw",
    "state_eci_to_roe",
    "state_eci_to_rtn",
    "state_inertial_to_ntw_for_body",
    "state_lvlh_to_eci",
    "state_ntw_to_eci",
    "state_ntw_to_inertial_for_body",
    "state_oe_to_roe",
    "state_roe_to_eci",
    "state_roe_to_oe",
    "state_rtn_to_eci",
]
