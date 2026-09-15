/*!
 * Shared relative-state transport and Jacobian assembly for the local
 * orbital frames in this module.
 */

use nalgebra::Vector3;

use crate::frames::{
    OrbitRelativeFrameVariant, state_inertial_to_rotating, state_rotating_to_inertial,
};
use crate::math::{SMatrix3, SMatrix6, SVector6, block_diagonal, skew_symmetric};

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

/// 6x6 Jacobian of the state map from a local orbital frame into inertial
/// axes: `[[R, 0], [R [ω]×, R]]` for the rotating variant and the block
/// diagonal `[[R, 0], [0, R]]` for the inertial snapshot.
///
/// # Arguments
/// - `r_frame_to_eci`: Rotation from the local frame into the inertial axes (dimensionless)
/// - `omega`: Angular velocity of the local frame relative to the inertial axes, expressed in the local frame (rad/s)
/// - `variant`: Rotating or inertial snapshot
///
/// # Returns
/// - Jacobian `J` such that `P_eci = J P_frame Jᵀ`
pub(crate) fn jacobian_to_inertial(
    r_frame_to_eci: &SMatrix3,
    omega: &Vector3<f64>,
    variant: OrbitRelativeFrameVariant,
) -> SMatrix6 {
    let mut j = block_diagonal(r_frame_to_eci, r_frame_to_eci);
    if variant == OrbitRelativeFrameVariant::Rotating {
        let coupling = r_frame_to_eci * skew_symmetric(omega);
        j.fixed_view_mut::<3, 3>(3, 0).copy_from(&coupling);
    }
    j
}

/// 6x6 Jacobian of the state map from inertial axes into a local orbital
/// frame: `[[Rᵀ, 0], [−[ω]× Rᵀ, Rᵀ]]` for the rotating variant and the
/// block diagonal for the inertial snapshot. Exact inverse of
/// [`jacobian_to_inertial`].
///
/// # Arguments
/// - `r_eci_to_frame`: Rotation from the inertial axes into the local frame (dimensionless)
/// - `omega`: Angular velocity of the local frame relative to the inertial axes, expressed in the local frame (rad/s)
/// - `variant`: Rotating or inertial snapshot
///
/// # Returns
/// - Jacobian `J` such that `P_frame = J P_eci Jᵀ`
pub(crate) fn jacobian_from_inertial(
    r_eci_to_frame: &SMatrix3,
    omega: &Vector3<f64>,
    variant: OrbitRelativeFrameVariant,
) -> SMatrix6 {
    let mut j = block_diagonal(r_eci_to_frame, r_eci_to_frame);
    if variant == OrbitRelativeFrameVariant::Rotating {
        let coupling = -skew_symmetric(omega) * r_eci_to_frame;
        j.fixed_view_mut::<3, 3>(3, 0).copy_from(&coupling);
    }
    j
}

/// Rate at which the velocity direction turns under two-body motion.
///
/// With `a = −μ r / r³`, the component of the acceleration perpendicular to
/// the velocity has magnitude `μ |r × v̂| / r³ = μ |h| / (r³ |v|)`, and the
/// unit velocity turns at that magnitude divided by `|v|`, about the orbit
/// normal `ĥ`, in the direction of motion:
///
/// `ω_v = μ |h| / (r³ v²)`
///
/// This reduces to `v / r` on a circular orbit. It is the rotation rate of
/// every local orbital frame whose axes are built from `v̂` and `ĥ` (NTW, TNW,
/// VNC), derived from the basis-vector kinematic identity
/// `ω = [ė_y·e_z, ė_z·e_x, ė_x·e_y]`.
///
/// # Arguments
/// - `x_inertial`: Cartesian state in an inertial frame centered on the attracting body (m, m/s)
/// - `gm`: Gravitational parameter of the attracting body (m³/s²)
///
/// # Returns
/// - Angular rate of the velocity direction about the orbit normal (rad/s)
///
/// # References
/// - H. Schaub and J. L. Junkins, *Analytical Mechanics of Space Systems*, 4th ed., AIAA, 2018, Section 3.3
#[allow(dead_code)]
pub(crate) fn velocity_direction_rate(x_inertial: SVector6, gm: f64) -> f64 {
    let r = x_inertial.fixed_rows::<3>(0);
    let v = x_inertial.fixed_rows::<3>(3);
    let h = r.cross(&v).norm();
    gm * h / (r.norm().powi(3) * v.norm_squared())
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::AngleFormat;
    use crate::GM_EARTH;
    use crate::R_EARTH;
    use crate::coordinates::state_koe_to_eci;
    use crate::orbits::mean_motion;
    use crate::relative_motion::{
        jacobian_eci_to_rtn, jacobian_rtn_to_eci, omega_rtn, rotation_eci_to_rtn,
        rotation_rtn_to_eci, state_eci_to_rtn, state_rtn_to_eci,
    };
    use approx::assert_abs_diff_eq;
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

    #[test]
    #[parallel]
    fn test_jacobian_helpers_match_rtn_bitwise() {
        let (x_chief, _) = chief_and_deputy();
        let omega = omega_rtn(x_chief);
        for variant in [
            OrbitRelativeFrameVariant::Inertial,
            OrbitRelativeFrameVariant::Rotating,
        ] {
            assert_eq!(
                jacobian_to_inertial(&rotation_rtn_to_eci(x_chief), &omega, variant),
                jacobian_rtn_to_eci(x_chief, variant)
            );
            assert_eq!(
                jacobian_from_inertial(&rotation_eci_to_rtn(x_chief), &omega, variant),
                jacobian_eci_to_rtn(x_chief, variant)
            );
        }
    }

    #[test]
    #[parallel]
    fn test_velocity_direction_rate_circular_equals_orbit_rate() {
        let sma = R_EARTH + 700e3;
        let x = state_koe_to_eci(
            SVector6::new(sma, 0.0, 45.0, 10.0, 0.0, 20.0),
            AngleFormat::Degrees,
        );
        let expected = mean_motion(sma, AngleFormat::Radians);
        assert_abs_diff_eq!(
            velocity_direction_rate(x, GM_EARTH),
            expected,
            epsilon = 1e-13
        );
        // Velocity-direction rate exceeds the position-direction rate at periapsis
        let x_peri = state_koe_to_eci(
            SVector6::new(sma, 0.2, 45.0, 10.0, 0.0, 0.0),
            AngleFormat::Degrees,
        );
        assert!(velocity_direction_rate(x_peri, GM_EARTH) < omega_rtn(x_peri)[2]);
    }
}
