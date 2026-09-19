"""
Relative Motion Module

Satellite relative motion and orbital reference frames.

This module provides transformations between inertial frames and orbital
reference frames such as RTN, LVLH, NTW, TNW, VNC, PQW, EQW, and NSW.

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

The PQW perifocal frame has P toward periapsis, W along the orbit normal, and
Q = W × P; it is an inertial snapshot with no rate.

The EQW equinoctial frame has E along the ascending node, W along the orbit
normal, and Q = W × E; it is an inertial snapshot with no rate.

The NSW frame has X toward nadir, Y as close to the Sun as possible while
normal to X, and Z = X × Y; its functions take the Sun state as an argument.

Functions are provided for:
- Rotation matrices between ECI and RTN frames
- Rotation matrices between ECI and LVLH frames
- Rotation matrices between ECI and NTW frames
- Rotation matrices between ECI and TNW frames
- Rotation matrices between ECI and VNC frames
- Rotation matrices between ECI and PQW frames
- Rotation matrices between ECI and EQW frames
- Rotation matrices between ECI and NSW frames
- (Future) Relative motion dynamics (Clohessy-Wiltshire equations, etc.)
"""

from brahe._brahe import (
    covariance_eci_to_eqw,
    covariance_eci_to_lvlh,
    covariance_eci_to_nsw,
    covariance_eci_to_ntw,
    covariance_eci_to_pqw,
    covariance_eci_to_rtn,
    covariance_eci_to_tnw,
    covariance_eci_to_vnc,
    covariance_eqw_to_eci,
    covariance_inertial_to_ntw_for_body,
    covariance_inertial_to_pqw_for_body,
    covariance_inertial_to_tnw_for_body,
    covariance_inertial_to_vnc_for_body,
    covariance_lvlh_to_eci,
    covariance_nsw_to_eci,
    covariance_ntw_to_eci,
    covariance_ntw_to_inertial_for_body,
    covariance_pqw_to_eci,
    covariance_pqw_to_inertial_for_body,
    covariance_rtn_to_eci,
    covariance_tnw_to_eci,
    covariance_tnw_to_inertial_for_body,
    covariance_vnc_to_eci,
    covariance_vnc_to_inertial_for_body,
    jacobian_eci_to_eqw,
    jacobian_eci_to_lvlh,
    jacobian_eci_to_nsw,
    jacobian_eci_to_ntw,
    jacobian_eci_to_pqw,
    jacobian_eci_to_rtn,
    jacobian_eci_to_tnw,
    jacobian_eci_to_vnc,
    jacobian_eqw_to_eci,
    jacobian_inertial_to_ntw_for_body,
    jacobian_inertial_to_pqw_for_body,
    jacobian_inertial_to_tnw_for_body,
    jacobian_inertial_to_vnc_for_body,
    jacobian_lvlh_to_eci,
    jacobian_nsw_to_eci,
    jacobian_ntw_to_eci,
    jacobian_ntw_to_inertial_for_body,
    jacobian_pqw_to_eci,
    jacobian_pqw_to_inertial_for_body,
    jacobian_rtn_to_eci,
    jacobian_tnw_to_eci,
    jacobian_tnw_to_inertial_for_body,
    jacobian_vnc_to_eci,
    jacobian_vnc_to_inertial_for_body,
    omega_lvlh,
    omega_nsw,
    omega_ntw,
    omega_ntw_for_body,
    omega_rtn,
    omega_tnw,
    omega_tnw_for_body,
    omega_vnc,
    omega_vnc_for_body,
    rotation_eci_to_eqw,
    rotation_eci_to_lvlh,
    rotation_eci_to_nsw,
    rotation_eci_to_ntw,
    rotation_eci_to_pqw,
    rotation_eci_to_rtn,
    rotation_eci_to_tnw,
    rotation_eci_to_vnc,
    rotation_eqw_to_eci,
    rotation_inertial_to_pqw_for_body,
    rotation_lvlh_to_eci,
    rotation_nsw_to_eci,
    rotation_ntw_to_eci,
    rotation_pqw_to_eci,
    rotation_pqw_to_inertial_for_body,
    rotation_rtn_to_eci,
    rotation_tnw_to_eci,
    rotation_vnc_to_eci,
    state_eci_to_eqw,
    state_eci_to_lvlh,
    state_eci_to_nsw,
    state_eci_to_ntw,
    state_eci_to_pqw,
    state_eci_to_roe,
    state_eci_to_rtn,
    state_eci_to_tnw,
    state_eci_to_vnc,
    state_eqw_to_eci,
    state_inertial_to_ntw_for_body,
    state_inertial_to_pqw_for_body,
    state_inertial_to_tnw_for_body,
    state_inertial_to_vnc_for_body,
    state_lvlh_to_eci,
    state_nsw_to_eci,
    state_ntw_to_eci,
    state_ntw_to_inertial_for_body,
    state_oe_to_roe,
    state_pqw_to_eci,
    state_pqw_to_inertial_for_body,
    state_roe_to_eci,
    state_roe_to_oe,
    state_rtn_to_eci,
    state_tnw_to_eci,
    state_tnw_to_inertial_for_body,
    state_vnc_to_eci,
    state_vnc_to_inertial_for_body,
)

__all__ = [
    "covariance_eci_to_eqw",
    "covariance_eci_to_lvlh",
    "covariance_eci_to_nsw",
    "covariance_eci_to_ntw",
    "covariance_eci_to_pqw",
    "covariance_eci_to_rtn",
    "covariance_eci_to_tnw",
    "covariance_eci_to_vnc",
    "covariance_eqw_to_eci",
    "covariance_inertial_to_ntw_for_body",
    "covariance_inertial_to_pqw_for_body",
    "covariance_inertial_to_tnw_for_body",
    "covariance_inertial_to_vnc_for_body",
    "covariance_lvlh_to_eci",
    "covariance_nsw_to_eci",
    "covariance_ntw_to_eci",
    "covariance_ntw_to_inertial_for_body",
    "covariance_pqw_to_eci",
    "covariance_pqw_to_inertial_for_body",
    "covariance_rtn_to_eci",
    "covariance_tnw_to_eci",
    "covariance_tnw_to_inertial_for_body",
    "covariance_vnc_to_eci",
    "covariance_vnc_to_inertial_for_body",
    "jacobian_eci_to_eqw",
    "jacobian_eci_to_lvlh",
    "jacobian_eci_to_nsw",
    "jacobian_eci_to_ntw",
    "jacobian_eci_to_pqw",
    "jacobian_eci_to_rtn",
    "jacobian_eci_to_tnw",
    "jacobian_eci_to_vnc",
    "jacobian_eqw_to_eci",
    "jacobian_inertial_to_ntw_for_body",
    "jacobian_inertial_to_pqw_for_body",
    "jacobian_inertial_to_tnw_for_body",
    "jacobian_inertial_to_vnc_for_body",
    "jacobian_lvlh_to_eci",
    "jacobian_nsw_to_eci",
    "jacobian_ntw_to_eci",
    "jacobian_ntw_to_inertial_for_body",
    "jacobian_pqw_to_eci",
    "jacobian_pqw_to_inertial_for_body",
    "jacobian_rtn_to_eci",
    "jacobian_tnw_to_eci",
    "jacobian_tnw_to_inertial_for_body",
    "jacobian_vnc_to_eci",
    "jacobian_vnc_to_inertial_for_body",
    "omega_lvlh",
    "omega_nsw",
    "omega_ntw",
    "omega_ntw_for_body",
    "omega_rtn",
    "omega_tnw",
    "omega_tnw_for_body",
    "omega_vnc",
    "omega_vnc_for_body",
    "rotation_eci_to_eqw",
    "rotation_eci_to_lvlh",
    "rotation_eci_to_nsw",
    "rotation_eci_to_ntw",
    "rotation_eci_to_pqw",
    "rotation_eci_to_rtn",
    "rotation_eci_to_tnw",
    "rotation_eci_to_vnc",
    "rotation_eqw_to_eci",
    "rotation_inertial_to_pqw_for_body",
    "rotation_lvlh_to_eci",
    "rotation_nsw_to_eci",
    "rotation_ntw_to_eci",
    "rotation_pqw_to_eci",
    "rotation_pqw_to_inertial_for_body",
    "rotation_rtn_to_eci",
    "rotation_tnw_to_eci",
    "rotation_vnc_to_eci",
    "state_eci_to_eqw",
    "state_eci_to_lvlh",
    "state_eci_to_nsw",
    "state_eci_to_ntw",
    "state_eci_to_pqw",
    "state_eci_to_roe",
    "state_eci_to_rtn",
    "state_eci_to_tnw",
    "state_eci_to_vnc",
    "state_eqw_to_eci",
    "state_inertial_to_ntw_for_body",
    "state_inertial_to_pqw_for_body",
    "state_inertial_to_tnw_for_body",
    "state_inertial_to_vnc_for_body",
    "state_lvlh_to_eci",
    "state_nsw_to_eci",
    "state_ntw_to_eci",
    "state_ntw_to_inertial_for_body",
    "state_oe_to_roe",
    "state_pqw_to_eci",
    "state_pqw_to_inertial_for_body",
    "state_roe_to_eci",
    "state_roe_to_oe",
    "state_rtn_to_eci",
    "state_tnw_to_eci",
    "state_tnw_to_inertial_for_body",
    "state_vnc_to_eci",
    "state_vnc_to_inertial_for_body",
]
