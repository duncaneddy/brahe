/*!
 * Covariance transformation between reference frames.
 *
 * Every transform the frame router performs on a Cartesian state is affine:
 * a rotation of the axes, the angular-velocity coupling that carries
 * position into velocity for rotating frames, and a translation between
 * centers. A covariance therefore transforms with the constant 6x6
 * Jacobian of that map at the epoch, `P' = J P Jᵀ`, with an identity block
 * for any state elements beyond the orbital six.
 */

use nalgebra::DMatrix;

use crate::frames::ReferenceFrame;
use crate::frames::transform::state_frame_to_frame;
use crate::math::covariance::{symmetrize, symmetrize_6};
use crate::math::linalg::{SMatrix6, SVector6};
use crate::time::Epoch;
use crate::utils::errors::BraheError;

/// Jacobian of the state transform from `from` to `to` at `epoch`.
///
/// The router's state transform is affine, so the Jacobian is recovered
/// exactly (to rounding) by transforming the origin and six scaled basis
/// states and differencing: position probes of `1e7` m and velocity probes
/// of `1e4` m/s keep the differences well above the rounding of any
/// heliocentric translation.
///
/// # Arguments
/// * `from` - Frame the covariance is expressed in
/// * `to` - Frame to rotate it into
/// * `epoch` - Epoch of the transform
///
/// # Returns
/// * `Ok(SMatrix6)`: `J` such that `P_to = J P_from Jᵀ`
/// * `Err(BraheError)`: If the router cannot transform states between the frames at this epoch
///
/// # Examples
///
/// ```
/// use brahe::frames::{CelestialFrame, state_transform_jacobian, rotation_gcrf_to_eme2000};
/// use brahe::time::{Epoch, TimeSystem};
/// # brahe::eop::set_global_eop_provider(
/// #     brahe::eop::StaticEOPProvider::from_values((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
/// # );
///
/// let epc = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let j = state_transform_jacobian(CelestialFrame::GCRF, CelestialFrame::EME2000, epc).unwrap();
/// let r = rotation_gcrf_to_eme2000();
/// assert!((j.fixed_view::<3, 3>(0, 0) - r).norm() < 1e-12);
/// assert!(j.fixed_view::<3, 3>(3, 0).norm() < 1e-12);
/// ```
pub fn state_transform_jacobian(
    from: impl Into<ReferenceFrame>,
    to: impl Into<ReferenceFrame>,
    epoch: Epoch,
) -> Result<SMatrix6, BraheError> {
    let from = from.into();
    let to = to.into();
    let origin = state_frame_to_frame(from.clone(), to.clone(), epoch, SVector6::zeros())?;
    let mut j = SMatrix6::zeros();
    for i in 0..6 {
        let scale = if i < 3 { 1.0e7 } else { 1.0e4 };
        let mut probe = SVector6::zeros();
        probe[i] = scale;
        let image = state_frame_to_frame(from.clone(), to.clone(), epoch, probe)?;
        j.set_column(i, &((image - origin) / scale));
    }
    Ok(j)
}

/// Rotates an `n x n` covariance (`n >= 6`) with a 6x6 state Jacobian,
/// leaving elements beyond the orbital six unchanged, and symmetrizes the
/// result.
///
/// The full transform is `blockdiag(jacobian, I)`, so a covariance that
/// carries extra parameters (drag coefficient, clock bias) keeps their
/// variances and picks up the rotation in the cross-covariances with the
/// orbital block.
///
/// # Arguments
/// * `covariance` - Square `n x n` covariance, `n >= 6`, whose leading six elements are the Cartesian state
/// * `jacobian` - 6x6 state-transform Jacobian, e.g. from [`state_transform_jacobian`]
///
/// # Returns
/// * `Ok(DMatrix<f64>)`: The `n x n` rotated covariance
/// * `Err(BraheError)`: If `covariance` is not square, or is smaller than 6x6
///
/// # Examples
///
/// ```
/// use brahe::frames::rotate_covariance;
/// use brahe::math::linalg::SMatrix6;
/// use nalgebra::DMatrix;
///
/// let p = DMatrix::<f64>::identity(7, 7);
/// let j = SMatrix6::identity();
/// let rotated = rotate_covariance(&p, &j).unwrap();
///
/// assert_eq!(rotated.nrows(), 7);
/// assert_eq!(rotated[(6, 6)], 1.0);
/// ```
pub fn rotate_covariance(
    covariance: &DMatrix<f64>,
    jacobian: &SMatrix6,
) -> Result<DMatrix<f64>, BraheError> {
    let (rows, cols) = (covariance.nrows(), covariance.ncols());
    if rows != cols {
        return Err(BraheError::Error(format!(
            "covariance must be square, got {}x{}",
            rows, cols
        )));
    }
    if rows < 6 {
        return Err(BraheError::Error(format!(
            "covariance must be at least 6x6, got {}x{}",
            rows, cols
        )));
    }

    let mut j = DMatrix::<f64>::identity(rows, rows);
    j.view_mut((0, 0), (6, 6)).copy_from(jacobian);

    Ok(symmetrize(&(&j * covariance * j.transpose())))
}

/// Fixed-size form of [`rotate_covariance`].
///
/// # Arguments
/// * `covariance` - 6x6 Cartesian state covariance
/// * `jacobian` - 6x6 state-transform Jacobian, e.g. from [`state_transform_jacobian`]
///
/// # Returns
/// * `SMatrix6`: The rotated, symmetrized covariance
///
/// # Examples
///
/// ```
/// use brahe::frames::rotate_covariance_6;
/// use brahe::math::linalg::SMatrix6;
///
/// let p = SMatrix6::identity() * 100.0;
/// let rotated = rotate_covariance_6(&p, &SMatrix6::identity());
///
/// assert_eq!(rotated[(0, 0)], 100.0);
/// ```
pub fn rotate_covariance_6(covariance: &SMatrix6, jacobian: &SMatrix6) -> SMatrix6 {
    symmetrize_6(&(jacobian * covariance * jacobian.transpose()))
}

/// Covariance transformed from `from` to `to` at `epoch`.
///
/// Composes [`state_transform_jacobian`] with [`rotate_covariance`], so the
/// covariance may be expressed in, and rotated into, any frame the router
/// knows.
///
/// # Arguments
/// * `from` - Frame the covariance is expressed in
/// * `to` - Frame to rotate it into
/// * `epoch` - Epoch of the transform
/// * `covariance` - Square `n x n` covariance, `n >= 6`, whose leading six elements are the Cartesian state
///
/// # Returns
/// * `Ok(DMatrix<f64>)`: The `n x n` covariance in `to`
/// * `Err(BraheError)`: If the router cannot transform states between the frames at this epoch, or the covariance has an unusable shape
///
/// # Examples
///
/// ```
/// use brahe::frames::{CelestialFrame, covariance_frame_to_frame};
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::DMatrix;
/// # brahe::eop::set_global_eop_provider(
/// #     brahe::eop::StaticEOPProvider::from_values((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
/// # );
///
/// let epc = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let p = DMatrix::<f64>::identity(6, 6) * 100.0;
/// let p_itrf = covariance_frame_to_frame(
///     CelestialFrame::GCRF,
///     CelestialFrame::ITRF,
///     epc,
///     &p,
/// ).unwrap();
///
/// assert_eq!(p_itrf.nrows(), 6);
/// ```
pub fn covariance_frame_to_frame(
    from: impl Into<ReferenceFrame>,
    to: impl Into<ReferenceFrame>,
    epoch: Epoch,
    covariance: &DMatrix<f64>,
) -> Result<DMatrix<f64>, BraheError> {
    let jacobian = state_transform_jacobian(from, to, epoch)?;
    rotate_covariance(covariance, &jacobian)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::constants::OMEGA_EARTH;
    use crate::frames::{CelestialFrame, rotation_frame_to_frame, rotation_gcrf_to_eme2000};
    use crate::math::linalg::{SMatrix3, block_diagonal, skew_symmetric};
    use crate::time::TimeSystem;
    use crate::utils::testing::setup_global_test_eop;
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector3;
    use serial_test::parallel;

    fn test_epoch() -> Epoch {
        Epoch::from_datetime(2024, 3, 15, 6, 30, 0.0, 0.0, TimeSystem::UTC)
    }

    fn leo_state() -> SVector6 {
        SVector6::new(6878.0e3, 1200.0e3, -900.0e3, -1.1e3, 6.2e3, 3.4e3)
    }

    #[test]
    #[parallel]
    fn test_state_transform_jacobian_gcrf_to_eme2000_is_frame_bias() {
        setup_global_test_eop();

        let j =
            state_transform_jacobian(CelestialFrame::GCRF, CelestialFrame::EME2000, test_epoch())
                .unwrap();

        let r = rotation_gcrf_to_eme2000();
        let expected = block_diagonal(&r, &r);
        assert_abs_diff_eq!((j - expected).norm(), 0.0, epsilon = 1e-12);

        // A constant bias couples no position into velocity.
        assert_abs_diff_eq!(j.fixed_view::<3, 3>(3, 0).norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    #[parallel]
    fn test_state_transform_jacobian_gcrf_to_itrf_matches_router() {
        setup_global_test_eop();

        let epc = test_epoch();
        let j = state_transform_jacobian(CelestialFrame::GCRF, CelestialFrame::ITRF, epc).unwrap();

        // The map is affine, so J x reproduces the router's increment exactly.
        let x = leo_state();
        let origin = state_frame_to_frame(
            CelestialFrame::GCRF,
            CelestialFrame::ITRF,
            epc,
            SVector6::zeros(),
        )
        .unwrap();
        let image =
            state_frame_to_frame(CelestialFrame::GCRF, CelestialFrame::ITRF, epc, x).unwrap();
        let increment = image - origin;
        let predicted = j * x;
        for i in 0..3 {
            assert_abs_diff_eq!(predicted[i], increment[i], epsilon = 1e-6);
            assert_abs_diff_eq!(predicted[3 + i], increment[3 + i], epsilon = 1e-9);
        }

        // The position block is the plain axis rotation.
        let r = rotation_frame_to_frame(CelestialFrame::GCRF, CelestialFrame::ITRF, epc).unwrap();
        assert_abs_diff_eq!(
            (j.fixed_view::<3, 3>(0, 0) - r).norm(),
            0.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            (j.fixed_view::<3, 3>(3, 3) - r).norm(),
            0.0,
            epsilon = 1e-12
        );

        // The velocity-position coupling is -[omega]x R with omega the Earth
        // rotation rate in ITRF axes. Polar motion tilts that vector off the
        // ITRF z axis by a few tenths of an arcsecond, so the nominal
        // [0, 0, OMEGA_EARTH] matches only to a relative 1e-5; the sign
        // convention is what this comparison fixes.
        let omega_nominal = Vector3::new(0.0, 0.0, OMEGA_EARTH);
        let expected: SMatrix3 = -skew_symmetric(&omega_nominal) * r;
        let coupling: SMatrix3 = j.fixed_view::<3, 3>(3, 0).into();
        assert_abs_diff_eq!(
            (coupling - expected).norm() / expected.norm(),
            0.0,
            epsilon = 1e-5
        );

        // Recovering omega from the coupling is exact: -[omega]x = coupling Rᵀ
        // is skew-symmetric, its axial vector has magnitude OMEGA_EARTH, and
        // it points along ITRF z to within the polar-motion tilt.
        let skew: SMatrix3 = -coupling * r.transpose();
        assert_abs_diff_eq!((skew + skew.transpose()).norm(), 0.0, epsilon = 1e-18);
        let omega = Vector3::new(skew[(2, 1)], skew[(0, 2)], skew[(1, 0)]);
        assert_abs_diff_eq!(omega.norm(), OMEGA_EARTH, epsilon = 1e-16);
        assert_abs_diff_eq!(omega[2] / omega.norm(), 1.0, epsilon = 1e-10);
    }

    #[test]
    #[parallel]
    fn test_state_transform_jacobian_round_trip_is_identity() {
        setup_global_test_eop();

        let epc = test_epoch();
        let forward =
            state_transform_jacobian(CelestialFrame::GCRF, CelestialFrame::ITRF, epc).unwrap();
        let backward =
            state_transform_jacobian(CelestialFrame::ITRF, CelestialFrame::GCRF, epc).unwrap();

        assert_abs_diff_eq!(
            (backward * forward - SMatrix6::identity()).norm(),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    #[parallel]
    fn test_rotate_covariance_preserves_extra_dimensions() {
        // A 90-degree rotation about z maps x into y, so the leading block
        // moves while element (6, 6) is untouched and the (0, 6)
        // cross-covariance only picks up the rotation.
        let r = SMatrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let j = block_diagonal(&r, &r);

        let mut p = DMatrix::<f64>::identity(7, 7);
        p[(0, 0)] = 4.0;
        p[(6, 6)] = 9.0;
        p[(0, 6)] = 2.0;
        p[(6, 0)] = 2.0;

        let rotated = rotate_covariance(&p, &j).unwrap();

        assert_eq!(rotated.nrows(), 7);
        assert_abs_diff_eq!(rotated[(6, 6)], 9.0, epsilon = 1e-12);
        // Row 0 of R is (0, -1, 0), so the x-parameter cross-covariance takes
        // the value that sat on the y axis, which is zero here.
        assert_abs_diff_eq!(rotated[(0, 6)], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(rotated[(1, 6)], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(rotated[(1, 1)], 4.0, epsilon = 1e-12);
    }

    #[test]
    #[parallel]
    fn test_rotate_covariance_6() {
        // A 90-degree rotation about z maps x into y, so the leading variances
        // swap and the position-velocity cross term follows the same axes.
        let r = SMatrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let j = block_diagonal(&r, &r);

        let mut p = SMatrix6::identity();
        p[(0, 0)] = 4.0;
        p[(1, 1)] = 9.0;
        p[(0, 3)] = 2.0;
        p[(3, 0)] = 2.0;

        let rotated = rotate_covariance_6(&p, &j);

        assert_abs_diff_eq!(rotated[(0, 0)], 9.0, epsilon = 1e-12);
        assert_abs_diff_eq!(rotated[(1, 1)], 4.0, epsilon = 1e-12);
        // Row 0 of R is (0, -1, 0) and so is row 3 of blockdiag(R, R) shifted,
        // so the (0, 3) cross term moves to (1, 4) and keeps its magnitude.
        assert_abs_diff_eq!(rotated[(1, 4)], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(rotated[(0, 3)], 0.0, epsilon = 1e-12);

        // The result is symmetric and agrees with the dynamic form.
        assert_abs_diff_eq!((rotated - rotated.transpose()).norm(), 0.0, epsilon = 1e-18);
        let dynamic =
            rotate_covariance(&DMatrix::from_iterator(6, 6, p.iter().copied()), &j).unwrap();
        for i in 0..6 {
            for k in 0..6 {
                assert_abs_diff_eq!(rotated[(i, k)], dynamic[(i, k)], epsilon = 1e-15);
            }
        }
    }

    #[test]
    #[parallel]
    fn test_rotate_covariance_shape_errors() {
        let j = SMatrix6::identity();

        let err = rotate_covariance(&DMatrix::<f64>::identity(5, 5), &j).unwrap_err();
        assert!(err.to_string().contains("at least 6x6"));
        assert!(err.to_string().contains("5x5"));

        let err = rotate_covariance(&DMatrix::<f64>::zeros(6, 5), &j).unwrap_err();
        assert!(err.to_string().contains("must be square"));
        assert!(err.to_string().contains("6x5"));
    }

    #[test]
    #[parallel]
    fn test_covariance_frame_to_frame_round_trip() {
        setup_global_test_eop();

        let epc = test_epoch();
        let mut p = DMatrix::<f64>::zeros(6, 6);
        for i in 0..3 {
            p[(i, i)] = 100.0;
            p[(3 + i, 3 + i)] = 0.01;
        }
        p[(0, 1)] = 25.0;
        p[(1, 0)] = 25.0;
        p[(0, 3)] = 0.5;
        p[(3, 0)] = 0.5;

        let p_itrf =
            covariance_frame_to_frame(CelestialFrame::GCRF, CelestialFrame::ITRF, epc, &p).unwrap();
        let p_back =
            covariance_frame_to_frame(CelestialFrame::ITRF, CelestialFrame::GCRF, epc, &p_itrf)
                .unwrap();

        assert_abs_diff_eq!((p_back - &p).norm() / p.norm(), 0.0, epsilon = 1e-9);
    }
}
