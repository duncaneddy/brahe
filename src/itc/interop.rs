/*!
 * Conversions between Modified ITC messages and orbit trajectories.
 */

use crate::frames::OrbitRelativeFrameVariant;
use crate::math::linalg::{SMatrix3, SMatrix6, SVector6};
use crate::relative_motion::{omega_rtn, rotation_rtn_to_eci};

/// Places `r` on both diagonal blocks of a 6x6 matrix.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn block_diagonal(r: SMatrix3) -> SMatrix6 {
    let mut m = SMatrix6::zeros();
    m.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
    m.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);
    m
}

/// Averages a matrix with its transpose to remove floating-point asymmetry.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn symmetrize(m: SMatrix6) -> SMatrix6 {
    (m + m.transpose()) * 0.5
}

#[cfg_attr(not(test), allow(dead_code))]
fn skew(w: nalgebra::Vector3<f64>) -> SMatrix3 {
    SMatrix3::new(0.0, -w[2], w[1], w[2], 0.0, -w[0], -w[1], w[0], 0.0)
}

/// Jacobian taking a covariance from the RTN frame of `x` into the inertial
/// frame `x` is expressed in.
///
/// `Inertial` gives the block-diagonal `[[R, 0], [0, R]]` form used by the
/// NASA CA Handbook (Appendix N eq. N-13) and CARA's `RIC2ECI`. `Rotating`
/// gives `[[R, 0], [R·[ω×], R]]`, the transform of a truly rotating frame
/// with ω the RTN frame rate in RTN components.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn rtn_to_frame_jacobian(x: SVector6, variant: OrbitRelativeFrameVariant) -> SMatrix6 {
    let r = rotation_rtn_to_eci(x);
    let mut j = block_diagonal(r);
    if variant == OrbitRelativeFrameVariant::Rotating {
        let coupling = r * skew(omega_rtn(x));
        j.fixed_view_mut::<3, 3>(3, 0).copy_from(&coupling);
    }
    j
}

/// Inverse of [`rtn_to_frame_jacobian`]: `[[Rᵀ, 0], [0, Rᵀ]]` or
/// `[[Rᵀ, 0], [-[ω×]Rᵀ, Rᵀ]]`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn frame_to_rtn_jacobian(x: SVector6, variant: OrbitRelativeFrameVariant) -> SMatrix6 {
    let rt = rotation_rtn_to_eci(x).transpose();
    let mut j = block_diagonal(rt);
    if variant == OrbitRelativeFrameVariant::Rotating {
        let coupling = -skew(omega_rtn(x)) * rt;
        j.fixed_view_mut::<3, 3>(3, 0).copy_from(&coupling);
    }
    j
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::relative_motion::{omega_rtn, rotation_rtn_to_eci};
    use approx::assert_abs_diff_eq;
    use serial_test::parallel;

    fn sample_state() -> SVector6 {
        SVector6::new(
            4244346.5367594,
            1264325.4891872,
            5043982.6441325,
            3595.1547629,
            5258.7956583,
            -4335.0352914,
        )
    }

    #[test]
    #[parallel]
    fn test_rtn_jacobian_inertial_is_block_diagonal() {
        let x = sample_state();
        let r = rotation_rtn_to_eci(x);
        let j = rtn_to_frame_jacobian(x, OrbitRelativeFrameVariant::Inertial);
        for i in 0..3 {
            for k in 0..3 {
                assert_abs_diff_eq!(j[(i, k)], r[(i, k)], epsilon = 1e-15);
                assert_abs_diff_eq!(j[(i + 3, k + 3)], r[(i, k)], epsilon = 1e-15);
                assert_eq!(j[(i, k + 3)], 0.0);
                assert_eq!(j[(i + 3, k)], 0.0);
            }
        }
    }

    #[test]
    #[parallel]
    fn test_rtn_jacobian_rotating_couples_position_into_velocity() {
        let x = sample_state();
        let r = rotation_rtn_to_eci(x);
        let w = omega_rtn(x);
        let j = rtn_to_frame_jacobian(x, OrbitRelativeFrameVariant::Rotating);
        let skew = SMatrix3::new(0.0, -w[2], w[1], w[2], 0.0, -w[0], -w[1], w[0], 0.0);
        let lower_left = r * skew;
        for i in 0..3 {
            for k in 0..3 {
                assert_abs_diff_eq!(j[(i + 3, k)], lower_left[(i, k)], epsilon = 1e-15);
                assert_abs_diff_eq!(j[(i, k)], r[(i, k)], epsilon = 1e-15);
                assert_abs_diff_eq!(j[(i + 3, k + 3)], r[(i, k)], epsilon = 1e-15);
                assert_eq!(j[(i, k + 3)], 0.0);
            }
        }
        assert!(lower_left.norm() > 0.0);
    }

    #[test]
    #[parallel]
    fn test_rtn_jacobians_are_inverses() {
        let x = sample_state();
        for variant in [
            OrbitRelativeFrameVariant::Inertial,
            OrbitRelativeFrameVariant::Rotating,
        ] {
            let j = rtn_to_frame_jacobian(x, variant);
            let jinv = frame_to_rtn_jacobian(x, variant);
            let identity = j * jinv;
            for i in 0..6 {
                for k in 0..6 {
                    let expected = if i == k { 1.0 } else { 0.0 };
                    assert_abs_diff_eq!(identity[(i, k)], expected, epsilon = 1e-12);
                }
            }
        }
    }

    #[test]
    #[parallel]
    fn test_symmetrize_and_block_diagonal() {
        let mut m = SMatrix6::zeros();
        m[(0, 1)] = 2.0;
        let s = symmetrize(m);
        assert_eq!(s[(0, 1)], 1.0);
        assert_eq!(s[(1, 0)], 1.0);
        let b = block_diagonal(SMatrix3::identity() * 3.0);
        assert_eq!(b[(4, 4)], 3.0);
        assert_eq!(b[(0, 4)], 0.0);
    }
}
