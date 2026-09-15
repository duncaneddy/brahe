"""
Relative Motion Module

Satellite relative motion and orbital reference frames.

This module provides transformations between inertial frames and orbital
reference frames such as RTN, LVLH, NTW, TNW, and VNC.

The RTN frame is an orbital reference frame defined as:
- R (Radial): Points from Earth's center to satellite position
- T (Tangential): Along-track direction in orbital plane
- N (Normal): Perpendicular to orbital plane (angular momentum direction)

The LVLH frame (CCSDS/SANA definition) has Z toward nadir, Y opposite the
orbit normal, and X = Y × Z.

The NTW frame has Y along velocity, Z along the orbit normal, and
X = Y × Z; it coincides with RTN on a circular orbit.

The TNW frame has X along velocity, Z along the orbit normal, and Y = Z × X
pointing inward.

The VNC frame has X along velocity, Y along the orbit normal, and Z = X × Y;
STK calls it VNB.

Functions are provided for:
- Rotation matrices between ECI and RTN frames
- Rotation matrices between ECI and LVLH frames
- Rotation matrices between ECI and NTW frames
- Rotation matrices between ECI and TNW frames
- Rotation matrices between ECI and VNC frames
- (Future) Relative motion dynamics (Clohessy-Wiltshire equations, etc.)
"""

from brahe._brahe import (
    covariance_eci_to_lvlh,
    covariance_eci_to_ntw,
    covariance_eci_to_rtn,
    covariance_eci_to_tnw,
    covariance_eci_to_vnc,
    covariance_inertial_to_ntw_for_body,
    covariance_inertial_to_tnw_for_body,
    covariance_inertial_to_vnc_for_body,
    covariance_lvlh_to_eci,
    covariance_ntw_to_eci,
    covariance_ntw_to_inertial_for_body,
    covariance_rtn_to_eci,
    covariance_tnw_to_eci,
    covariance_tnw_to_inertial_for_body,
    covariance_vnc_to_eci,
    covariance_vnc_to_inertial_for_body,
    jacobian_eci_to_lvlh,
    jacobian_eci_to_ntw,
    jacobian_eci_to_rtn,
    jacobian_eci_to_tnw,
    jacobian_eci_to_vnc,
    jacobian_inertial_to_ntw_for_body,
    jacobian_inertial_to_tnw_for_body,
    jacobian_inertial_to_vnc_for_body,
    jacobian_lvlh_to_eci,
    jacobian_ntw_to_eci,
    jacobian_ntw_to_inertial_for_body,
    jacobian_rtn_to_eci,
    jacobian_tnw_to_eci,
    jacobian_tnw_to_inertial_for_body,
    jacobian_vnc_to_eci,
    jacobian_vnc_to_inertial_for_body,
    omega_lvlh,
    omega_ntw,
    omega_ntw_for_body,
    omega_rtn,
    omega_tnw,
    omega_tnw_for_body,
    omega_vnc,
    omega_vnc_for_body,
    rotation_eci_to_lvlh,
    rotation_eci_to_ntw,
    rotation_eci_to_rtn,
    rotation_eci_to_tnw,
    rotation_eci_to_vnc,
    rotation_lvlh_to_eci,
    rotation_ntw_to_eci,
    rotation_rtn_to_eci,
    rotation_tnw_to_eci,
    rotation_vnc_to_eci,
    state_eci_to_lvlh,
    state_eci_to_ntw,
    state_eci_to_roe,
    state_eci_to_rtn,
    state_eci_to_tnw,
    state_eci_to_vnc,
    state_inertial_to_ntw_for_body,
    state_inertial_to_tnw_for_body,
    state_inertial_to_vnc_for_body,
    state_lvlh_to_eci,
    state_ntw_to_eci,
    state_ntw_to_inertial_for_body,
    state_oe_to_roe,
    state_roe_to_eci,
    state_roe_to_oe,
    state_rtn_to_eci,
    state_tnw_to_eci,
    state_tnw_to_inertial_for_body,
    state_vnc_to_eci,
    state_vnc_to_inertial_for_body,
)

__all__ = [
    "covariance_eci_to_lvlh",
    "covariance_eci_to_ntw",
    "covariance_eci_to_rtn",
    "covariance_eci_to_tnw",
    "covariance_eci_to_vnc",
    "covariance_inertial_to_ntw_for_body",
    "covariance_inertial_to_tnw_for_body",
    "covariance_inertial_to_vnc_for_body",
    "covariance_lvlh_to_eci",
    "covariance_ntw_to_eci",
    "covariance_ntw_to_inertial_for_body",
    "covariance_rtn_to_eci",
    "covariance_tnw_to_eci",
    "covariance_tnw_to_inertial_for_body",
    "covariance_vnc_to_eci",
    "covariance_vnc_to_inertial_for_body",
    "jacobian_eci_to_lvlh",
    "jacobian_eci_to_ntw",
    "jacobian_eci_to_rtn",
    "jacobian_eci_to_tnw",
    "jacobian_eci_to_vnc",
    "jacobian_inertial_to_ntw_for_body",
    "jacobian_inertial_to_tnw_for_body",
    "jacobian_inertial_to_vnc_for_body",
    "jacobian_lvlh_to_eci",
    "jacobian_ntw_to_eci",
    "jacobian_ntw_to_inertial_for_body",
    "jacobian_rtn_to_eci",
    "jacobian_tnw_to_eci",
    "jacobian_tnw_to_inertial_for_body",
    "jacobian_vnc_to_eci",
    "jacobian_vnc_to_inertial_for_body",
    "omega_lvlh",
    "omega_ntw",
    "omega_ntw_for_body",
    "omega_rtn",
    "omega_tnw",
    "omega_tnw_for_body",
    "omega_vnc",
    "omega_vnc_for_body",
    "rotation_eci_to_lvlh",
    "rotation_eci_to_ntw",
    "rotation_eci_to_rtn",
    "rotation_eci_to_tnw",
    "rotation_eci_to_vnc",
    "rotation_lvlh_to_eci",
    "rotation_ntw_to_eci",
    "rotation_rtn_to_eci",
    "rotation_tnw_to_eci",
    "rotation_vnc_to_eci",
    "state_eci_to_lvlh",
    "state_eci_to_ntw",
    "state_eci_to_roe",
    "state_eci_to_rtn",
    "state_eci_to_tnw",
    "state_eci_to_vnc",
    "state_inertial_to_ntw_for_body",
    "state_inertial_to_tnw_for_body",
    "state_inertial_to_vnc_for_body",
    "state_lvlh_to_eci",
    "state_ntw_to_eci",
    "state_ntw_to_inertial_for_body",
    "state_oe_to_roe",
    "state_roe_to_eci",
    "state_roe_to_oe",
    "state_rtn_to_eci",
    "state_tnw_to_eci",
    "state_tnw_to_inertial_for_body",
    "state_vnc_to_eci",
    "state_vnc_to_inertial_for_body",
]
