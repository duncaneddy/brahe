/*!
 * Covariance transformation between reference frames.
 *
 * Every transform the frame router performs on a Cartesian state is affine:
 * a rotation of the axes, the angular-velocity coupling that carries
 * position into velocity for rotating frames, and a translation between
 * centers. A covariance therefore transforms with the constant 6x6
 * Jacobian of that map at the epoch, `P' = J P Jᵀ`, with an identity block
 * for any state elements beyond the orbital six.
 *
 * The Jacobian is not assembled analytically from each frame pair's
 * rotation and angular rate. It is obtained numerically, by probing
 * [`crate::frames::state_frame_to_frame`] itself: the transform of the zero
 * state gives the constant term, and the transform of each scaled basis
 * state, differenced against it, gives the corresponding column. Because
 * the map is affine, that difference quotient is not an approximation —
 * `f(x) = J x + b` gives `(f(s eᵢ) - f(0)) / s = J eᵢ` for every nonzero
 * `s`, so a single one-sided difference recovers the exact Jacobian up to
 * floating-point rounding. Probing the router rather than restating its
 * algebra also means the covariance path picks up every frame the router
 * supports, and cannot drift out of step with it.
 */

use nalgebra::DMatrix;

use crate::frames::ReferenceFrame;
use crate::frames::transform::{CelestialFrame, state_frame_to_frame};
use crate::math::linalg::{SMatrix6, SVector6};
use crate::math::{symmetrize, symmetrize_6};
use crate::time::Epoch;
use crate::utils::batch::{try_batch_map, try_batch_map_epochs, try_batch_zip};
use crate::utils::errors::BraheError;

/// Jacobian of the state transform from `from` to `to` at `epoch`.
///
/// Computed numerically, by probing [`state_frame_to_frame`]: the transform
/// of the zero state gives the constant term, and the transform of each of
/// six scaled basis states, differenced against it, gives one column. The
/// router's state transform is affine, so this one-sided difference is the
/// exact Jacobian rather than a finite-difference estimate of it.
///
/// # Probe scale
///
/// Because the map is affine, `(f(s eᵢ) - f(0)) / s` equals `J eᵢ` for every
/// nonzero probe scale `s` in exact arithmetic; the scale is therefore not a
/// step size to be tuned, and no truncation error rides on it. It matters
/// only through floating-point cancellation, and only when the probe crosses
/// a translation, which is the generic [`ReferenceFrame::Body`] and
/// [`ReferenceFrame::OrbitRelative`] path: there both `f(s eᵢ)` and `f(0)`
/// carry the same large center offset, and differencing them discards the
/// leading digits the two share. The relative error of a column is then
/// about `eps * |offset| / s`. Position probes of `1e7` m and velocity
/// probes of `1e4` m/s hold that to a few parts in `1e12` even against a
/// heliocentric offset of `1.5e11` m (`2.2e-16 * 1.5e11 / 1e7 = 3e-12` on
/// position, and a matching bound on velocity against the `~3e4` m/s of
/// Earth's orbital motion). Nearer offsets do proportionally better: an
/// Earth-orbit offset of `1e7` m leaves the columns at full double
/// precision.
///
/// The celestial pairs handled below never see that cancellation at all:
/// the probe is taken about a single common center, so the translation is
/// factored out before differencing and the columns come back to full
/// double precision.
///
/// Only the two orientations matter. A change of center is a translation,
/// whose Jacobian is the identity whatever the offset, and the router
/// evaluates a given orientation identically at every center. Between two
/// celestial frames the probe therefore never crosses centers: identical axes
/// return the identity outright, and otherwise the target axes are probed
/// about the source's own center. A covariance query between, say, `LCI` and
/// `GCRF` consults no ephemeris and needs no SPK coverage. Body and
/// orbit-relative frames are probed as given, since their orientation chains
/// are resolved through the frame registry rather than by axes alone.
///
/// # Arguments
/// * `from` - Frame the covariance is expressed in
/// * `to` - Frame to rotate it into
/// * `epoch` - Epoch of the transform
///
/// # Returns
/// * `Ok(SMatrix6)`: `J` such that `P_to = J P_from Jᵀ`
/// * `Err(BraheError)`: If the router cannot evaluate either frame's orientation at this epoch
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
///
/// // Frames that share their axes differ only by a translation.
/// let j = state_transform_jacobian(CelestialFrame::LCI, CelestialFrame::GCRF, epc).unwrap();
/// assert_eq!(j, brahe::math::linalg::SMatrix6::identity());
/// ```
pub fn state_transform_jacobian(
    from: impl Into<ReferenceFrame>,
    to: impl Into<ReferenceFrame>,
    epoch: Epoch,
) -> Result<SMatrix6, BraheError> {
    let from = from.into();
    let to = to.into();

    let target = match (&from, &to) {
        (ReferenceFrame::Celestial(a), ReferenceFrame::Celestial(b)) => {
            if a.axes() == b.axes() {
                return Ok(SMatrix6::identity());
            }
            ReferenceFrame::Celestial(CelestialFrame::centered(a.center(), b.axes()))
        }
        _ => to.clone(),
    };

    let origin = state_frame_to_frame(from.clone(), target.clone(), epoch, SVector6::zeros())?;
    let mut j = SMatrix6::zeros();
    for i in 0..6 {
        let scale = if i < 3 { 1.0e7 } else { 1.0e4 };
        let mut probe = SVector6::zeros();
        probe[i] = scale;
        let image = state_frame_to_frame(from.clone(), target.clone(), epoch, probe)?;
        j.set_column(i, &((image - origin) / scale));
    }
    Ok(j)
}

/// Jacobians of the state transform from `from` to `to`, one per epoch.
///
/// Batch form of [`state_transform_jacobian`], mirroring
/// [`crate::frames::rotations_frame_to_frame`]. The same `from`/`to` values
/// are reused for every epoch, but a non-celestial frame is resolved through
/// the frame/object registries fresh for each one. Evaluation runs on the
/// global thread pool for large inputs.
///
/// # Arguments
/// * `from` - Frame the covariance is expressed in
/// * `to` - Frame to rotate it into
/// * `epochs` - Epochs of the transforms
///
/// # Returns
/// * `Ok(Vec<SMatrix6>)`: One Jacobian per epoch, in input order
/// * `Err(BraheError)`: If the router cannot evaluate either frame's orientation at any epoch
///
/// # Examples
///
/// ```
/// use brahe::frames::{CelestialFrame, state_transform_jacobians};
/// use brahe::time::{Epoch, TimeSystem};
/// # brahe::eop::set_global_eop_provider(
/// #     brahe::eop::StaticEOPProvider::from_values((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
/// # );
///
/// let epc = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0];
/// let j = state_transform_jacobians(CelestialFrame::GCRF, CelestialFrame::ITRF, &epochs).unwrap();
/// assert_eq!(j.len(), 2);
/// ```
pub fn state_transform_jacobians(
    from: impl Into<ReferenceFrame>,
    to: impl Into<ReferenceFrame>,
    epochs: &[Epoch],
) -> Result<Vec<SMatrix6>, BraheError> {
    let from = from.into();
    let to = to.into();
    try_batch_map(
        |epc| state_transform_jacobian(from.clone(), to.clone(), *epc),
        epochs,
    )
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

/// Rotates a batch of covariances with a batch of 6x6 state Jacobians.
///
/// Batch form of [`rotate_covariance`]. `covariances` and `jacobians` follow
/// the broadcast rule: each has length 1 or the common batch length, so a
/// single Jacobian rotates a whole batch of covariances and a single
/// covariance may be rotated by a batch of Jacobians. Evaluation runs on the
/// global thread pool for large inputs.
///
/// # Arguments
/// * `covariances` - Square `n x n` covariances, `n >= 6`, length 1 or the batch length
/// * `jacobians` - 6x6 state-transform Jacobians, e.g. from [`state_transform_jacobians`], length 1 or the batch length
///
/// # Returns
/// * `Ok(Vec<DMatrix<f64>>)`: The rotated covariances, in input order
/// * `Err(BraheError)`: If the lengths do not satisfy the broadcast rule, or a covariance is not square or is smaller than 6x6
///
/// # Examples
///
/// ```
/// use brahe::frames::rotate_covariances;
/// use brahe::math::linalg::SMatrix6;
/// use nalgebra::DMatrix;
///
/// let covariances = vec![DMatrix::<f64>::identity(6, 6) * 100.0; 3];
/// let rotated = rotate_covariances(&covariances, &[SMatrix6::identity()]).unwrap();
///
/// assert_eq!(rotated.len(), 3);
/// ```
pub fn rotate_covariances(
    covariances: &[DMatrix<f64>],
    jacobians: &[SMatrix6],
) -> Result<Vec<DMatrix<f64>>, BraheError> {
    try_batch_zip(rotate_covariance, covariances, jacobians)
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

/// Transforms a batch of covariances from `from` to `to`.
///
/// Batch form of [`covariance_frame_to_frame`], mirroring
/// [`crate::frames::states_frame_to_frame`]. `epochs` and `covariances`
/// follow the broadcast rule: each has length 1 or the common batch length.
/// A single epoch computes the Jacobian once and applies it to every
/// covariance; per-element epochs compute one Jacobian per covariance.
/// Evaluation runs on the global thread pool for large inputs.
///
/// # Arguments
/// * `from` - Frame the covariances are expressed in
/// * `to` - Frame to rotate them into
/// * `epochs` - Epochs of the transforms, length 1 or the batch length
/// * `covariances` - Square `n x n` covariances, `n >= 6`, whose leading six elements are the Cartesian state, length 1 or the batch length
///
/// # Returns
/// * `Ok(Vec<DMatrix<f64>>)`: The covariances in `to`, in input order
/// * `Err(BraheError)`: If the lengths do not satisfy the broadcast rule, the router cannot transform states between the frames at an epoch, or a covariance has an unusable shape
///
/// # Examples
///
/// ```
/// use brahe::frames::{CelestialFrame, covariances_frame_to_frame};
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::DMatrix;
/// # brahe::eop::set_global_eop_provider(
/// #     brahe::eop::StaticEOPProvider::from_values((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
/// # );
///
/// let epc = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0];
/// let covariances = vec![DMatrix::<f64>::identity(6, 6) * 100.0];
/// let rotated = covariances_frame_to_frame(
///     CelestialFrame::GCRF,
///     CelestialFrame::ITRF,
///     &epochs,
///     &covariances,
/// ).unwrap();
///
/// assert_eq!(rotated.len(), 2);
/// ```
pub fn covariances_frame_to_frame(
    from: impl Into<ReferenceFrame>,
    to: impl Into<ReferenceFrame>,
    epochs: &[Epoch],
    covariances: &[DMatrix<f64>],
) -> Result<Vec<DMatrix<f64>>, BraheError> {
    let from = from.into();
    let to = to.into();
    try_batch_map_epochs(
        |epc| state_transform_jacobian(from.clone(), to.clone(), epc),
        |jacobian, covariance| rotate_covariance(covariance, jacobian),
        epochs,
        covariances,
    )
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::attitude::RotationMatrix;
    use crate::constants::OMEGA_EARTH;
    use crate::frames::object_registry::FnProvider;
    use crate::frames::{
        CelestialFrame, clear_frame_registry, clear_object_registry, register_frame,
        register_object, rotation_frame_to_frame, rotation_gcrf_to_eme2000,
    };
    use crate::math::linalg::{SMatrix3, block_diagonal, skew_symmetric};
    use crate::time::TimeSystem;
    use crate::utils::testing::{
        CacheRedirect, NetworkModeGuard, setup_global_test_eop, without_spice_kernels,
    };
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector3;
    use serial_test::{parallel, serial};

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

    /// Runs `f` with no SPICE kernel resident, an empty cache, and the network
    /// off, so any ephemeris lookup fails instead of silently reloading
    /// `de440s`. Clearing the registry alone is not enough: the translation
    /// leg calls `ensure_bodies_loadable`, which reloads the default DE kernel
    /// from the cache on demand.
    fn without_any_ephemeris<T>(f: impl FnOnce() -> T) -> T {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("offline"));
        without_spice_kernels(f)
    }

    #[test]
    #[serial] // clears the SPICE registry and redirects the cache
    fn test_state_transform_jacobian_shared_axes_is_identity_without_ephemeris() {
        setup_global_test_eop();

        // LCI and GCRF share ICRF axes and differ only by the Earth-Moon
        // offset. A translation has an identity Jacobian, so the answer comes
        // back with no ephemeris at all -- which is what routing the probe
        // through the source's own center buys.
        without_any_ephemeris(|| {
            for (from, to) in [
                (CelestialFrame::LCI, CelestialFrame::GCRF),
                (CelestialFrame::GCRF, CelestialFrame::LCI),
                (CelestialFrame::SSBI, CelestialFrame::MCI),
            ] {
                // The router itself cannot cross these centers here.
                assert!(
                    state_frame_to_frame(from, to, test_epoch(), SVector6::zeros()).is_err(),
                    "{from} -> {to} unexpectedly resolved an ephemeris"
                );

                let j = state_transform_jacobian(from, to, test_epoch()).unwrap();
                assert_eq!(j, SMatrix6::identity(), "{from} -> {to}");
            }
        });
    }

    #[test]
    #[serial] // clears the SPICE registry and redirects the cache
    fn test_state_transform_jacobian_is_center_independent() {
        setup_global_test_eop();

        // The orientation change ICRF -> ITRF is evaluated identically at
        // every center, so a Moon- or Mars-centered source gives the
        // Earth-centered Jacobian, and says so with no ephemeris loaded.
        let epc = test_epoch();
        let expected =
            state_transform_jacobian(CelestialFrame::GCRF, CelestialFrame::ITRF, epc).unwrap();

        without_any_ephemeris(|| {
            for from in [CelestialFrame::LCI, CelestialFrame::MCI] {
                let j = state_transform_jacobian(from, CelestialFrame::ITRF, epc).unwrap();
                assert_abs_diff_eq!((j - expected).norm(), 0.0, epsilon = 1e-12);
            }
        });
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

    /// Re-implements the probe loop of [`state_transform_jacobian`] with
    /// caller-chosen probe scales, so a test can vary only the scale.
    ///
    /// It deliberately skips the same-axes and common-center routing
    /// shortcuts and probes `from` -> `to` as given, so what a comparison
    /// measures is the raw difference quotient at that scale.
    fn jacobian_with_probe_scales(
        from: ReferenceFrame,
        to: ReferenceFrame,
        epoch: Epoch,
        position_scale: f64,
        velocity_scale: f64,
    ) -> SMatrix6 {
        let origin =
            state_frame_to_frame(from.clone(), to.clone(), epoch, SVector6::zeros()).unwrap();
        let mut j = SMatrix6::zeros();
        for i in 0..6 {
            let scale = if i < 3 {
                position_scale
            } else {
                velocity_scale
            };
            let mut probe = SVector6::zeros();
            probe[i] = scale;
            let image = state_frame_to_frame(from.clone(), to.clone(), epoch, probe).unwrap();
            j.set_column(i, &((image - origin) / scale));
        }
        j
    }

    #[test]
    #[parallel]
    fn test_state_transform_jacobian_is_probe_scale_independent() {
        setup_global_test_eop();

        // GCRF -> ITRF is a same-center pair, so the probe crosses no
        // translation and there is nothing for the difference to cancel
        // against: every scale returns the same matrix to full double
        // precision. Unit probes are 1e7 (position) and 1e4 (velocity) times
        // smaller than the production ones, and the large probes 1e2 times
        // larger; over that 1e9 range of scales the Jacobian moves by
        // 1.9e-16 in Frobenius norm, i.e. one ulp.
        let epc = test_epoch();
        let production =
            state_transform_jacobian(CelestialFrame::GCRF, CelestialFrame::ITRF, epc).unwrap();

        let unit = jacobian_with_probe_scales(
            CelestialFrame::GCRF.into(),
            CelestialFrame::ITRF.into(),
            epc,
            1.0,
            1.0,
        );
        assert_abs_diff_eq!((production - unit).norm(), 0.0, epsilon = 1e-12);

        let large = jacobian_with_probe_scales(
            CelestialFrame::GCRF.into(),
            CelestialFrame::ITRF.into(),
            epc,
            1.0e9,
            1.0e6,
        );
        assert_abs_diff_eq!((production - large).norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    #[serial] // registers a body frame and an object in the global registries
    fn test_state_transform_jacobian_probe_scale_across_a_translation() {
        setup_global_test_eop();
        clear_frame_registry();
        clear_object_registry();

        // A body frame held at a constant 30-degree yaw about a spacecraft in
        // LEO. The transform is a translation by the spacecraft state
        // followed by that fixed rotation, so the exact Jacobian is
        // blockdiag(R, R) and any departure from it is pure floating-point
        // cancellation across the ~7e6 m offset.
        let (s, c) = 30.0_f64.to_radians().sin_cos();
        let r = SMatrix3::new(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0);
        register_frame(
            ReferenceFrame::SC_BODY("SC"),
            CelestialFrame::GCRF.into(),
            RotationMatrix::from_matrix(r).unwrap(),
        )
        .unwrap();
        let x_sc = leo_state();
        register_object("SC", FnProvider(move |_| Ok(x_sc)), CelestialFrame::GCRF).unwrap();

        let epc = test_epoch();
        let body = ReferenceFrame::SC_BODY("SC");
        let expected = block_diagonal(&r, &r);

        let production = state_transform_jacobian(CelestialFrame::GCRF, body.clone(), epc).unwrap();
        let unit =
            jacobian_with_probe_scales(CelestialFrame::GCRF.into(), body.clone(), epc, 1.0, 1.0);

        clear_frame_registry();
        clear_object_registry();

        // The production scales keep the full accuracy of the exact answer
        // (measured 2.7e-16); unit probes lose about eps * |offset| / scale,
        // which for the ~7.0e6 m position offset of this spacecraft comes out
        // at a measured 3.4e-10.
        assert_abs_diff_eq!((production - expected).norm(), 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!((unit - expected).norm(), 0.0, epsilon = 1e-8);
        assert!(
            (production - expected).norm() < (unit - expected).norm(),
            "production probes ({}) should beat unit probes ({}) across a translation",
            (production - expected).norm(),
            (unit - expected).norm()
        );
    }

    #[test]
    #[parallel]
    fn test_state_transform_jacobians_matches_per_epoch_loop() {
        setup_global_test_eop();

        let epc = test_epoch();
        let epochs: Vec<Epoch> = (0..5).map(|i| epc + (i as f64) * 600.0).collect();
        let batch =
            state_transform_jacobians(CelestialFrame::GCRF, CelestialFrame::ITRF, &epochs).unwrap();

        assert_eq!(batch.len(), epochs.len());
        for (i, e) in epochs.iter().enumerate() {
            assert_eq!(
                batch[i],
                state_transform_jacobian(CelestialFrame::GCRF, CelestialFrame::ITRF, *e).unwrap()
            );
        }

        assert!(
            state_transform_jacobians(CelestialFrame::GCRF, CelestialFrame::ITRF, &[])
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    #[parallel]
    fn test_rotate_covariances_matches_loop_and_broadcasts() {
        let r = SMatrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let jacobians = vec![SMatrix6::identity(), block_diagonal(&r, &r)];
        let covariances: Vec<DMatrix<f64>> = (0..2)
            .map(|i| DMatrix::<f64>::identity(7, 7) * (1.0 + i as f64))
            .collect();

        let batch = rotate_covariances(&covariances, &jacobians).unwrap();
        assert_eq!(batch.len(), 2);
        for i in 0..2 {
            assert_eq!(
                batch[i],
                rotate_covariance(&covariances[i], &jacobians[i]).unwrap()
            );
        }

        // A single Jacobian broadcasts across the covariance batch, and a
        // single covariance across the Jacobian batch.
        let broadcast_jacobian = rotate_covariances(&covariances, &jacobians[1..]).unwrap();
        for i in 0..2 {
            assert_eq!(
                broadcast_jacobian[i],
                rotate_covariance(&covariances[i], &jacobians[1]).unwrap()
            );
        }
        let broadcast_covariance = rotate_covariances(&covariances[..1], &jacobians).unwrap();
        for i in 0..2 {
            assert_eq!(
                broadcast_covariance[i],
                rotate_covariance(&covariances[0], &jacobians[i]).unwrap()
            );
        }

        // Mismatched lengths and unusable shapes both surface as errors.
        let three = vec![DMatrix::<f64>::identity(6, 6); 3];
        assert!(rotate_covariances(&three, &jacobians).is_err());
        let err =
            rotate_covariances(&[DMatrix::<f64>::identity(5, 5)], &jacobians[..1]).unwrap_err();
        assert!(err.to_string().contains("at least 6x6"));
    }

    #[test]
    #[parallel]
    fn test_covariances_frame_to_frame_matches_per_epoch_loop() {
        setup_global_test_eop();

        let epc = test_epoch();
        let epochs: Vec<Epoch> = (0..4).map(|i| epc + (i as f64) * 900.0).collect();
        let covariances: Vec<DMatrix<f64>> = (0..4)
            .map(|i| DMatrix::<f64>::identity(6, 6) * (100.0 + i as f64))
            .collect();

        let batch = covariances_frame_to_frame(
            CelestialFrame::GCRF,
            CelestialFrame::ITRF,
            &epochs,
            &covariances,
        )
        .unwrap();

        assert_eq!(batch.len(), epochs.len());
        for (i, e) in epochs.iter().enumerate() {
            let expected = covariance_frame_to_frame(
                CelestialFrame::GCRF,
                CelestialFrame::ITRF,
                *e,
                &covariances[i],
            )
            .unwrap();
            assert_abs_diff_eq!((&batch[i] - &expected).norm(), 0.0, epsilon = 1e-18);
        }

        // A single epoch hoists one Jacobian across the whole batch, and a
        // single covariance broadcasts across the epochs.
        let one_epoch = covariances_frame_to_frame(
            CelestialFrame::GCRF,
            CelestialFrame::ITRF,
            &epochs[..1],
            &covariances,
        )
        .unwrap();
        for (i, cov) in covariances.iter().enumerate() {
            let expected = covariance_frame_to_frame(
                CelestialFrame::GCRF,
                CelestialFrame::ITRF,
                epochs[0],
                cov,
            )
            .unwrap();
            assert_abs_diff_eq!((&one_epoch[i] - &expected).norm(), 0.0, epsilon = 1e-18);
        }

        let one_covariance = covariances_frame_to_frame(
            CelestialFrame::GCRF,
            CelestialFrame::ITRF,
            &epochs,
            &covariances[..1],
        )
        .unwrap();
        assert_eq!(one_covariance.len(), epochs.len());
        for (i, e) in epochs.iter().enumerate() {
            let expected = covariance_frame_to_frame(
                CelestialFrame::GCRF,
                CelestialFrame::ITRF,
                *e,
                &covariances[0],
            )
            .unwrap();
            assert_abs_diff_eq!(
                (&one_covariance[i] - &expected).norm(),
                0.0,
                epsilon = 1e-18
            );
        }

        // Lengths that do not broadcast are rejected.
        assert!(
            covariances_frame_to_frame(
                CelestialFrame::GCRF,
                CelestialFrame::ITRF,
                &epochs[..2],
                &covariances,
            )
            .is_err()
        );
    }
}
