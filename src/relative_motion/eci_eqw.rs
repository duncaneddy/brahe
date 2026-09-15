/*!
 * Earth-Centered Inertial (ECI) to Equinoctial (EQW) Frame Transformations
 */

use nalgebra::Vector3;

use crate::frames::{OrbitRelativeFrameVariant, rotate_covariance_6};
use crate::math::{SMatrix3, SMatrix6, SVector6};
use crate::relative_motion::common::{
    jacobian_from_inertial, jacobian_to_inertial, relative_state_from_frame,
    relative_state_to_frame,
};

/// Below this node-vector norm the ascending node is undefined and E is
/// taken along the inertial x axis projected into the orbit plane.
const DEGENERATE_TOLERANCE: f64 = 1e-9;

/// Computes the rotation matrix transforming a vector in the equinoctial (EQW) frame to the
/// Earth-Centered Inertial (ECI) frame.
///
/// The ECI frame can be any inertial frame centered on the orbited body, such as GCRF or EME2000.
///
/// The EQW frame follows the SANA definition:
/// - E: Unit vector along the ascending node, `ẑ × ĥ` normalized, where `ẑ` is the inertial
///   pole and `ĥ` the orbit normal.
/// - W: Unit vector along the orbital angular momentum `r × v`.
/// - Q: `W × E`, completing the right-handed set.
///
/// SANA registers EQW only as a quasi-inertial snapshot: the axes are taken from the state at
/// the evaluation epoch and treated as fixed, so there is no `omega_eqw`. On an equatorial orbit
/// the node is undefined (node vector norm below 1e-9) and E is taken along the inertial x axis
/// projected into the orbit plane, matching the zero right-ascension convention while keeping
/// the matrix orthonormal.
///
/// # Arguments:
/// - `x_eci`: 6D state vector in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming from EQW to ECI frame
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `EQW_INERTIAL`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - Orekit `LOFType.EQW`, <https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/frames/LOFType.java>
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::*;
///
/// let x_eci = state_koe_to_eci(SVector6::new(R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0), AngleFormat::Degrees);
///
/// let rotation_matrix = rotation_eqw_to_eci(x_eci);
/// ```
pub fn rotation_eqw_to_eci(x_eci: SVector6) -> SMatrix3 {
    let r = x_eci.fixed_rows::<3>(0);
    let v = x_eci.fixed_rows::<3>(3);

    let h = r.cross(&v);
    let w_hat = h / h.norm();

    let node = Vector3::new(-w_hat[1], w_hat[0], 0.0);
    let e_hat = if node.norm() > DEGENERATE_TOLERANCE {
        node / node.norm()
    } else {
        let x_in_plane = Vector3::x() - w_hat[0] * w_hat;
        x_in_plane / x_in_plane.norm()
    };
    let q_hat = w_hat.cross(&e_hat);

    SMatrix3::from_columns(&[e_hat, q_hat, w_hat])
}

/// Computes the rotation matrix transforming a vector in the Earth-Centered Inertial (ECI)
/// frame to the equinoctial (EQW) frame.
///
/// # Arguments:
/// - `x_eci`: 6D state vector in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming from the ECI frame to EQW
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `EQW_INERTIAL`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - Orekit `LOFType.EQW`, <https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/frames/LOFType.java>
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::*;
///
/// let x_eci = state_koe_to_eci(SVector6::new(R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0), AngleFormat::Degrees);
///
/// let rotation_matrix = rotation_eci_to_eqw(x_eci);
/// ```
pub fn rotation_eci_to_eqw(x_eci: SVector6) -> SMatrix3 {
    rotation_eqw_to_eci(x_eci).transpose()
}

/// 6x6 Jacobian taking an EQW state covariance into ECI axes.
///
/// EQW is a quasi-inertial snapshot with no angular rate, so the Jacobian is the block diagonal
/// `[[R, 0], [0, R]]` with `R` the EQW-to-ECI rotation.
///
/// # Arguments:
/// - `x_eci`: 6D state vector of the frame's origin in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Returns:
/// - `j`: 6x6 Jacobian such that `P_eci = J P_eqw Jᵀ`
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::*;
///
/// let x_eci = state_koe_to_eci(SVector6::new(R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0), AngleFormat::Degrees);
///
/// let j = jacobian_eqw_to_eci(x_eci);
/// ```
///
/// # References:
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
pub fn jacobian_eqw_to_eci(x_eci: SVector6) -> SMatrix6 {
    jacobian_to_inertial(
        &rotation_eqw_to_eci(x_eci),
        &Vector3::zeros(),
        OrbitRelativeFrameVariant::Inertial,
    )
}

/// 6x6 Jacobian taking an ECI state covariance into EQW axes. Exact inverse of
/// [`jacobian_eqw_to_eci`].
///
/// EQW is a quasi-inertial snapshot with no angular rate, so the Jacobian is the block diagonal
/// `[[Rᵀ, 0], [0, Rᵀ]]` with `R` the EQW-to-ECI rotation.
///
/// # Arguments:
/// - `x_eci`: 6D state vector of the frame's origin in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Returns:
/// - `j`: 6x6 Jacobian such that `P_eqw = J P_eci Jᵀ`
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::*;
///
/// let x_eci = state_koe_to_eci(SVector6::new(R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0), AngleFormat::Degrees);
///
/// let j = jacobian_eci_to_eqw(x_eci);
/// ```
///
/// # References:
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
pub fn jacobian_eci_to_eqw(x_eci: SVector6) -> SMatrix6 {
    jacobian_from_inertial(
        &rotation_eci_to_eqw(x_eci),
        &Vector3::zeros(),
        OrbitRelativeFrameVariant::Inertial,
    )
}

/// Transforms a 6x6 state covariance from EQW axes into ECI axes.
///
/// Applies the congruence `P_eci = J P_eqw Jᵀ` with `J` from [`jacobian_eqw_to_eci`], and
/// symmetrizes the result.
///
/// # Arguments:
/// - `x_eci`: 6D state vector of the frame's origin in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `covariance`: 6x6 state covariance in EQW axes (m², m²/s, m²/s²)
///
/// # Returns:
/// - `p_eci`: 6x6 state covariance in ECI axes (m², m²/s, m²/s²)
///
/// # Examples:
/// ```
/// use brahe::{SVector6, SMatrix6};
/// use brahe::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::*;
///
/// let x_eci = state_koe_to_eci(SVector6::new(R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0), AngleFormat::Degrees);
/// let p_eqw = SMatrix6::identity();
///
/// let p_eci = covariance_eqw_to_eci(x_eci, &p_eqw);
/// ```
///
/// # References:
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
pub fn covariance_eqw_to_eci(x_eci: SVector6, covariance: &SMatrix6) -> SMatrix6 {
    rotate_covariance_6(covariance, &jacobian_eqw_to_eci(x_eci))
}

/// Transforms a 6x6 state covariance from ECI axes into EQW axes.
///
/// Applies the congruence `P_eqw = J P_eci Jᵀ` with `J` from [`jacobian_eci_to_eqw`], and
/// symmetrizes the result.
///
/// # Arguments:
/// - `x_eci`: 6D state vector of the frame's origin in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `covariance`: 6x6 state covariance in ECI axes (m², m²/s, m²/s²)
///
/// # Returns:
/// - `p_eqw`: 6x6 state covariance in EQW axes (m², m²/s, m²/s²)
///
/// # Examples:
/// ```
/// use brahe::{SVector6, SMatrix6};
/// use brahe::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::*;
///
/// let x_eci = state_koe_to_eci(SVector6::new(R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0), AngleFormat::Degrees);
/// let p_eci = SMatrix6::identity();
///
/// let p_eqw = covariance_eci_to_eqw(x_eci, &p_eci);
/// ```
///
/// # References:
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
pub fn covariance_eci_to_eqw(x_eci: SVector6, covariance: &SMatrix6) -> SMatrix6 {
    rotate_covariance_6(covariance, &jacobian_eci_to_eqw(x_eci))
}

/// Transforms the absolute states of a chief and deputy satellite from the Earth-Centered
/// Inertial (ECI) frame to the relative state of the deputy with respect to the chief in the
/// equinoctial (EQW) frame.
///
/// EQW is a quasi-inertial snapshot, so the relative velocity is a pure rotation of the inertial
/// relative velocity with no transport term.
///
/// # Arguments:
/// - `x_chief`: 6D state vector of the chief satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_deputy`: 6D state vector of the deputy satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Returns:
/// - `x_rel_eqw`: 6D relative state of the deputy with respect to the chief in the EQW frame [ρ_E, ρ_Q, ρ_W, ρ̇_E, ρ̇_Q, ρ̇_W] (m, m/s)
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `EQW_INERTIAL`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - Orekit `LOFType.EQW`, <https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/frames/LOFType.java>
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::*;
///
/// let oe_chief = SVector6::new(R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0);
/// let oe_deputy = SVector6::new(R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05);
///
/// let x_chief = state_koe_to_eci(oe_chief, AngleFormat::Degrees);
/// let x_deputy = state_koe_to_eci(oe_deputy, AngleFormat::Degrees);
///
/// let x_rel_eqw = state_eci_to_eqw(x_chief, x_deputy);
/// ```
pub fn state_eci_to_eqw(x_chief: SVector6, x_deputy: SVector6) -> SVector6 {
    relative_state_to_frame(
        &rotation_eci_to_eqw(x_chief),
        &Vector3::zeros(),
        x_chief,
        x_deputy,
    )
}

/// Transforms the relative state of a deputy satellite with respect to a chief satellite from
/// the equinoctial (EQW) frame to the absolute state of the deputy in the Earth-Centered
/// Inertial (ECI) frame.
///
/// EQW is a quasi-inertial snapshot, so the deputy's inertial relative velocity is a pure
/// rotation of the EQW relative velocity with no transport term.
///
/// # Arguments:
/// - `x_chief`: 6D state vector of the chief satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_rel_eqw`: 6D relative state of the deputy with respect to the chief in the EQW frame [ρ_E, ρ_Q, ρ_W, ρ̇_E, ρ̇_Q, ρ̇_W] (m, m/s)
///
/// # Returns:
/// - `x_deputy`: 6D state vector of the deputy satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `EQW_INERTIAL`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - Orekit `LOFType.EQW`, <https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/frames/LOFType.java>
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::*;
///
/// let oe_chief = SVector6::new(R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0);
/// let x_chief = state_koe_to_eci(oe_chief, AngleFormat::Degrees);
/// let x_rel_eqw = SVector6::new(1000.0, 500.0, -300.0, 0.0, 0.0, 0.0);
///
/// let x_deputy = state_eqw_to_eci(x_chief, x_rel_eqw);
/// ```
pub fn state_eqw_to_eci(x_chief: SVector6, x_rel_eqw: SVector6) -> SVector6 {
    relative_state_from_frame(
        &rotation_eci_to_eqw(x_chief),
        &Vector3::zeros(),
        x_chief,
        x_rel_eqw,
    )
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::AngleFormat;
    use crate::constants::{DEG2RAD, R_EARTH};
    use crate::coordinates::state_koe_to_eci;
    use crate::math::block_diagonal;
    use approx::assert_abs_diff_eq;
    use serial_test::parallel;

    fn state(e: f64, i: f64, raan: f64, argp: f64, m: f64) -> SVector6 {
        state_koe_to_eci(
            SVector6::new(R_EARTH + 700e3, e, i, raan, argp, m),
            AngleFormat::Degrees,
        )
    }

    #[test]
    #[parallel]
    fn test_rotation_eqw_e_axis_is_ascending_node() {
        let x = state(0.1, 97.8, 15.0, 30.0, 45.0);
        let h_hat = {
            let h = x.fixed_rows::<3>(0).cross(&x.fixed_rows::<3>(3));
            h / h.norm()
        };
        let m = rotation_eqw_to_eci(x);
        let e: Vector3<f64> = m.column(0).into();
        let q: Vector3<f64> = m.column(1).into();
        let w: Vector3<f64> = m.column(2).into();
        let node = Vector3::new((15.0 * DEG2RAD).cos(), (15.0 * DEG2RAD).sin(), 0.0);
        assert_abs_diff_eq!(e, node, epsilon = 1e-12);
        assert_abs_diff_eq!(e[2], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(w, h_hat, epsilon = 1e-15);
        assert_abs_diff_eq!(q, w.cross(&e), epsilon = 1e-15);
        assert_abs_diff_eq!(m.determinant(), 1.0, epsilon = 1e-14);
    }

    #[test]
    #[parallel]
    fn test_rotation_eqw_retrograde_orbit_node_follows_momentum() {
        // For i > 90 deg the node computed from ẑ × ĥ is still the ascending node
        let x = state(0.05, 120.0, 200.0, 10.0, 80.0);
        let e: Vector3<f64> = rotation_eqw_to_eci(x).column(0).into();
        let node = Vector3::new((200.0 * DEG2RAD).cos(), (200.0 * DEG2RAD).sin(), 0.0);
        assert_abs_diff_eq!(e, node, epsilon = 1e-12);
    }

    #[test]
    #[parallel]
    fn test_rotation_eqw_equatorial_orbit_uses_x_axis() {
        let x = state(0.1, 0.0, 15.0, 30.0, 45.0);
        let m = rotation_eqw_to_eci(x);
        let e: Vector3<f64> = m.column(0).into();
        assert_abs_diff_eq!(e, Vector3::x(), epsilon = 1e-12);
        assert_abs_diff_eq!(m.determinant(), 1.0, epsilon = 1e-14);
    }

    #[test]
    #[parallel]
    fn test_rotation_eqw_near_equatorial_fallback_is_orthonormal() {
        let x = state(0.1, 3e-8, 15.0, 30.0, 45.0);
        let m = rotation_eqw_to_eci(x);
        assert_abs_diff_eq!(m.transpose() * m, SMatrix3::identity(), epsilon = 1e-15);
        let e: Vector3<f64> = m.column(0).into();
        assert_abs_diff_eq!(e, Vector3::x(), epsilon = 1e-9);
        assert_abs_diff_eq!(m.determinant(), 1.0, epsilon = 1e-14);
    }

    #[test]
    #[parallel]
    fn test_rotation_eci_to_eqw_is_transpose() {
        let x = state(0.1, 97.8, 15.0, 30.0, 45.0);
        assert_eq!(rotation_eci_to_eqw(x), rotation_eqw_to_eci(x).transpose());
    }

    #[test]
    #[parallel]
    fn test_jacobian_eqw_is_block_diagonal_and_invertible() {
        let x = state(0.1, 97.8, 15.0, 30.0, 45.0);
        let j = jacobian_eqw_to_eci(x);
        let r = rotation_eqw_to_eci(x);
        assert_abs_diff_eq!((j - block_diagonal(&r, &r)).norm(), 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(
            (jacobian_eci_to_eqw(x) * j - SMatrix6::identity()).norm(),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    #[parallel]
    fn test_covariance_eqw_eci_round_trip() {
        let x = state(0.1, 97.8, 15.0, 30.0, 45.0);
        let mut p = SMatrix6::zeros();
        for i in 0..3 {
            p[(i, i)] = 100.0;
            p[(3 + i, 3 + i)] = 0.01;
        }
        p[(0, 1)] = 25.0;
        p[(1, 0)] = 25.0;
        let p_eci = covariance_eqw_to_eci(x, &p);
        let p_back = covariance_eci_to_eqw(x, &p_eci);
        assert_abs_diff_eq!((p_back - p).norm() / p.norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    #[parallel]
    fn test_state_eci_to_eqw_is_pure_rotation_of_relative_state() {
        let x_chief = state(0.1, 97.8, 15.0, 30.0, 45.0);
        let x_deputy = state(0.1015, 97.85, 15.05, 30.05, 45.05);
        let r = rotation_eci_to_eqw(x_chief);
        let diff = x_deputy - x_chief;
        let rel = state_eci_to_eqw(x_chief, x_deputy);
        assert_abs_diff_eq!(
            rel.fixed_rows::<3>(0).into_owned(),
            r * diff.fixed_rows::<3>(0),
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(
            rel.fixed_rows::<3>(3).into_owned(),
            r * diff.fixed_rows::<3>(3),
            epsilon = 1e-12
        );
    }

    #[test]
    #[parallel]
    fn test_state_eqw_position_of_chief_is_in_plane() {
        // The chief's own position in EQW is r [cos u, sin u, 0] with u the argument of latitude
        let x = state(0.1, 97.8, 15.0, 30.0, 45.0);
        let r_eqw = rotation_eci_to_eqw(x) * x.fixed_rows::<3>(0);
        assert_abs_diff_eq!(r_eqw[2], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(r_eqw.norm(), x.fixed_rows::<3>(0).norm(), epsilon = 1e-9);
    }

    #[test]
    #[parallel]
    fn test_state_eqw_to_eci_round_trip() {
        let x_chief = state(0.1, 97.8, 15.0, 30.0, 45.0);
        let x_rel = SVector6::new(1000.0, 500.0, -300.0, 0.1, -0.05, 0.02);
        let x_deputy = state_eqw_to_eci(x_chief, x_rel);
        assert_abs_diff_eq!(state_eci_to_eqw(x_chief, x_deputy), x_rel, epsilon = 1e-8);
    }
}
