/*!
 * Shared relative-state transport for the local orbital frames in this
 * module.
 */

use nalgebra::Vector3;

use crate::frames::{state_inertial_to_rotating, state_rotating_to_inertial};
use crate::math::{SMatrix3, SVector6};

/// Relative state of a deputy with respect to a chief in a rotating local
/// orbital frame.
///
/// The relative position is `ρ = R (r_d − r_c)` and the relative velocity
/// applies the transport term of the rotating axes,
/// `ρ̇ = R (v_d − v_c) − ω × ρ`.
///
/// # Arguments
/// - `r_eci_to_frame`: Rotation from the inertial axes into the local frame (dimensionless)
/// - `omega`: Angular velocity of the local frame relative to the inertial axes, expressed in the local frame (rad/s)
/// - `x_chief`: Chief Cartesian state in the inertial frame (m, m/s)
/// - `x_deputy`: Deputy Cartesian state in the inertial frame (m, m/s)
///
/// # Returns
/// - Deputy state relative to the chief, in the local frame (m, m/s)
pub(crate) fn relative_state_to_frame(
    r_eci_to_frame: &SMatrix3,
    omega: &Vector3<f64>,
    x_chief: SVector6,
    x_deputy: SVector6,
) -> SVector6 {
    state_inertial_to_rotating(r_eci_to_frame, omega, &(x_deputy - x_chief))
}

/// Inverse of [`relative_state_to_frame`]: the deputy's inertial state from
/// its relative state in a rotating local orbital frame.
///
/// # Arguments
/// - `r_eci_to_frame`: Rotation from the inertial axes into the local frame (dimensionless)
/// - `omega`: Angular velocity of the local frame relative to the inertial axes, expressed in the local frame (rad/s)
/// - `x_chief`: Chief Cartesian state in the inertial frame (m, m/s)
/// - `x_rel`: Deputy state relative to the chief, in the local frame (m, m/s)
///
/// # Returns
/// - Deputy Cartesian state in the inertial frame (m, m/s)
pub(crate) fn relative_state_from_frame(
    r_eci_to_frame: &SMatrix3,
    omega: &Vector3<f64>,
    x_chief: SVector6,
    x_rel: SVector6,
) -> SVector6 {
    x_chief + state_rotating_to_inertial(r_eci_to_frame, omega, &x_rel)
}

/// True-anomaly rate of an orbit, `ḟ = |r × v| / r²`, the rate at which the
/// radial direction rotates under two-body motion.
///
/// # Arguments
/// - `x_inertial`: Cartesian state in an inertial frame (position, velocity) (m, m/s)
///
/// # Returns
/// - True-anomaly rate (rad/s)
///
/// # References:
/// - K. T. Alfriend, S. R. Vadali, P. Gurfil, J. P. How, L. S. Breger, *Spacecraft Formation Flying*, Elsevier, 2010, eq. 2.16
pub(crate) fn true_anomaly_rate(x_inertial: SVector6) -> f64 {
    let r = x_inertial.fixed_rows::<3>(0);
    let v = x_inertial.fixed_rows::<3>(3);
    (r.cross(&v)).norm() / (r.norm().powi(2))
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::AngleFormat;
    use crate::R_EARTH;
    use crate::coordinates::state_koe_to_eci;
    use crate::relative_motion::{
        omega_rtn, rotation_eci_to_rtn, state_eci_to_rtn, state_rtn_to_eci,
    };
    use serial_test::parallel;

    fn chief_and_deputy() -> (SVector6, SVector6) {
        let x_chief = state_koe_to_eci(
            SVector6::new(R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0),
            AngleFormat::Degrees,
        );
        let x_deputy = state_koe_to_eci(
            SVector6::new(R_EARTH + 701e3, 0.0115, 97.85, 15.05, 30.05, 45.05),
            AngleFormat::Degrees,
        );
        (x_chief, x_deputy)
    }

    #[test]
    #[parallel]
    fn test_relative_state_helpers_match_rtn_bitwise() {
        let (x_chief, x_deputy) = chief_and_deputy();
        let r = rotation_eci_to_rtn(x_chief);
        let omega = omega_rtn(x_chief);

        let rel = relative_state_to_frame(&r, &omega, x_chief, x_deputy);
        assert_eq!(rel, state_eci_to_rtn(x_chief, x_deputy));

        let back = relative_state_from_frame(&r, &omega, x_chief, rel);
        assert_eq!(back, state_rtn_to_eci(x_chief, rel));
    }

    #[test]
    #[parallel]
    fn test_true_anomaly_rate_matches_omega_rtn_bitwise() {
        let x = state_koe_to_eci(
            SVector6::new(R_EARTH + 700e3, 0.1, 97.8, 15.0, 30.0, 45.0),
            AngleFormat::Degrees,
        );
        assert_eq!(true_anomaly_rate(x), omega_rtn(x)[2]);
    }
}
