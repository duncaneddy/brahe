/*!
 * Conversions between Modified ITC messages and orbit trajectories.
 */

use nalgebra::{DMatrix, DVector};

use crate::frames::{
    CelestialFrame, OrbitRelativeFrameVariant, ReferenceFrame, state_frame_to_frame,
};
use crate::math::linalg::{SMatrix3, SMatrix6, SVector6};
use crate::relative_motion::{omega_rtn, rotation_rtn_to_eci};
use crate::time::Epoch;
use crate::trajectories::dorbit_trajectory::DOrbitTrajectory;
use crate::trajectories::traits::{
    OrbitRepresentation, covariance_frame_allowed, inertial_covariance_rotation,
};
use crate::utils::BraheError;

use super::types::{ITC, ITCCovarianceFrame, ITCHeader, ITCStateVector};

/// Places `r` on both diagonal blocks of a 6x6 matrix.
///
/// # Arguments
/// * `r` - 3x3 matrix to duplicate onto the position and velocity blocks
///
/// # Returns
/// * `SMatrix6`: Block-diagonal matrix `[[r, 0], [0, r]]`
///
/// # Examples
/// ```text
/// block_diagonal(3 * I3) is the 6x6 matrix with 3 on the diagonal in
/// rows/columns 0-2 and 3-5, and zero everywhere else.
/// ```
fn block_diagonal(r: SMatrix3) -> SMatrix6 {
    let mut m = SMatrix6::zeros();
    m.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
    m.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);
    m
}

/// Averages a matrix with its transpose to remove floating-point asymmetry.
///
/// # Arguments
/// * `m` - 6x6 matrix expected to be symmetric up to floating-point error
///
/// # Returns
/// * `SMatrix6`: `(m + mᵀ) / 2`
///
/// # Examples
/// ```text
/// symmetrize applied to a matrix with a 2.0 in position (0, 1) and zero
/// in position (1, 0) produces 1.0 in both positions.
/// ```
fn symmetrize(m: SMatrix6) -> SMatrix6 {
    (m + m.transpose()) * 0.5
}

/// Skew-symmetric (cross-product) matrix of a 3-vector, satisfying
/// `skew(w) * v == w.cross(&v)`.
///
/// # Arguments
/// * `w` - 3-element angular velocity, rad/s
///
/// # Returns
/// * `SMatrix3`: Skew-symmetric matrix of `w`
///
/// # Examples
/// ```text
/// skew([0, 0, 2]) produces [[0, -2, 0], [2, 0, 0], [0, 0, 0]]
/// ```
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
///
/// # Arguments
/// * `x` - 6-element Cartesian state (position, m; velocity, m/s) in the target inertial frame
/// * `variant` - RTN rotation convention
///
/// # Returns
/// * `SMatrix6`: Jacobian `J` such that `P_inertial = J * P_rtn * Jᵀ`
///
/// # Examples
/// ```text
/// For OrbitRelativeFrameVariant::Inertial, the returned 6x6 matrix has
/// R = rotation_rtn_to_eci(x) on both diagonal blocks and zero elsewhere.
/// For OrbitRelativeFrameVariant::Rotating, the lower-left block also
/// carries R * skew(omega_rtn(x)).
/// ```
fn rtn_to_frame_jacobian(x: SVector6, variant: OrbitRelativeFrameVariant) -> SMatrix6 {
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
///
/// # Arguments
/// * `x` - 6-element Cartesian state (position, m; velocity, m/s) in the inertial frame
/// * `variant` - RTN rotation convention
///
/// # Returns
/// * `SMatrix6`: Jacobian `J` such that `P_rtn = J * P_inertial * Jᵀ`
///
/// # Examples
/// ```text
/// For OrbitRelativeFrameVariant::Inertial, the returned 6x6 matrix has
/// Rᵀ = rotation_rtn_to_eci(x).transpose() on both diagonal blocks and
/// zero elsewhere. For OrbitRelativeFrameVariant::Rotating, the
/// lower-left block also carries -skew(omega_rtn(x)) * Rᵀ.
/// ```
fn frame_to_rtn_jacobian(x: SVector6, variant: OrbitRelativeFrameVariant) -> SMatrix6 {
    let rt = rotation_rtn_to_eci(x).transpose();
    let mut j = block_diagonal(rt);
    if variant == OrbitRelativeFrameVariant::Rotating {
        let coupling = -skew(omega_rtn(x)) * rt;
        j.fixed_view_mut::<3, 3>(3, 0).copy_from(&coupling);
    }
    j
}

/// Converts an ITC record's position and velocity into a single
/// 6-element Cartesian state.
///
/// # Arguments
/// * `sv` - ITC state vector with `position` (m) and `velocity` (m/s)
///
/// # Returns
/// * `SVector6`: `[x, y, z, vx, vy, vz]`
///
/// # Examples
/// ```text
/// state_vector for position [7e6, 0, 0] m and velocity [0, 7.5e3, 0] m/s
/// produces [7e6, 0, 0, 0, 7.5e3, 0]
/// ```
fn state_vector(sv: &ITCStateVector) -> SVector6 {
    SVector6::new(
        sv.position[0],
        sv.position[1],
        sv.position[2],
        sv.velocity[0],
        sv.velocity[1],
        sv.velocity[2],
    )
}

/// Copies a fixed-size 6x6 matrix into a dynamically-sized `DMatrix`.
///
/// # Arguments
/// * `m` - 6x6 matrix to copy
///
/// # Returns
/// * `DMatrix<f64>`: Dynamically-sized 6x6 matrix with the same entries
///
/// # Examples
/// ```text
/// dmatrix_from(&SMatrix6::identity()) produces a 6x6 DMatrix equal to
/// the identity matrix.
/// ```
fn dmatrix_from(m: &SMatrix6) -> DMatrix<f64> {
    DMatrix::from_iterator(6, 6, m.iter().cloned())
}

impl ITC {
    /// Converts the message to a trajectory in `header.state_frame`.
    ///
    /// Equivalent to [`ITC::to_trajectory_with_covariance_variant`] with
    /// `OrbitRelativeFrameVariant::Inertial`, the block-diagonal RTN rotation
    /// used by conjunction assessment tooling.
    ///
    /// # Returns
    /// * `Ok(DOrbitTrajectory)`: Six-dimensional Cartesian trajectory with covariance attached when the message carries it
    /// * `Err(BraheError)`: If the message is empty, or covariance is present but the state frame or covariance frame combination is unsupported
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::itc::ITC;
    ///
    /// let itc = ITC::from_file(
    ///     "test_assets/starlink/MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt",
    /// ).unwrap();
    /// let traj = itc.to_trajectory().unwrap();
    /// assert_eq!(traj.states.len(), 50);
    /// assert_eq!(traj.name.as_deref(), Some("STARLINK-37711"));
    /// ```
    pub fn to_trajectory(&self) -> Result<DOrbitTrajectory, BraheError> {
        self.to_trajectory_with_covariance_variant(OrbitRelativeFrameVariant::Inertial)
    }

    /// Converts the message to a trajectory, choosing how RTN covariance is
    /// rotated into the state frame.
    ///
    /// Covariance requires a state frame that may carry it (GCRF or EME2000);
    /// other frames are an error. RTN covariance is rotated with the record's own state:
    /// `Inertial` uses `[[R, 0], [0, R]]` (NASA CA Handbook Appendix N eq.
    /// N-13; CARA `RIC2ECI`), `Rotating` uses `[[R, 0], [R·[ω×], R]]` with ω
    /// the RTN frame rate. A covariance frame of `EME2000` is accepted only
    /// with an `EME2000` state frame; `ITRF` covariance is not supported.
    ///
    /// # Arguments
    /// * `variant` - RTN rotation convention for the covariance
    ///
    /// # Returns
    /// * `Ok(DOrbitTrajectory)`: The trajectory
    /// * `Err(BraheError)`: If the message is empty or the frame combination is unsupported
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::frames::OrbitRelativeFrameVariant;
    /// use brahe::itc::ITC;
    ///
    /// let itc = ITC::from_file(
    ///     "test_assets/starlink/MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt",
    /// ).unwrap();
    /// let traj = itc.to_trajectory_with_covariance_variant(OrbitRelativeFrameVariant::Rotating).unwrap();
    /// assert!(traj.covariance_at(itc.states[0].epoch).unwrap().is_some());
    /// ```
    pub fn to_trajectory_with_covariance_variant(
        &self,
        variant: OrbitRelativeFrameVariant,
    ) -> Result<DOrbitTrajectory, BraheError> {
        if self.states.is_empty() {
            return Err(BraheError::Error(
                "cannot convert an empty Modified ITC message to a trajectory".to_string(),
            ));
        }
        let frame: ReferenceFrame = self.header.state_frame.into();
        let epochs: Vec<Epoch> = self.states.iter().map(|s| s.epoch).collect();
        let states: Vec<SVector6> = self.states.iter().map(state_vector).collect();

        let covariances = if self.has_covariance() {
            if !covariance_frame_allowed(&frame) {
                return Err(BraheError::Error(format!(
                    "Modified ITC covariance cannot be attached to a trajectory in {}; covariance is supported only for EME2000 and GCRF state frames",
                    frame
                )));
            }
            let rotated: Vec<DMatrix<f64>> = match self.header.covariance_frame {
                ITCCovarianceFrame::RTN => states
                    .iter()
                    .zip(&self.covariances)
                    .map(|(x, p)| {
                        let j = rtn_to_frame_jacobian(*x, variant);
                        dmatrix_from(&symmetrize(j * p * j.transpose()))
                    })
                    .collect(),
                ITCCovarianceFrame::EME2000 => {
                    if self.header.state_frame != CelestialFrame::EME2000 {
                        return Err(BraheError::Error(format!(
                            "Modified ITC covariance frame EME2000 requires an EME2000 state frame, found {}",
                            frame
                        )));
                    }
                    self.covariances.iter().map(dmatrix_from).collect()
                }
                ITCCovarianceFrame::ITRF => {
                    return Err(BraheError::Error(format!(
                        "Modified ITC covariance frame ITRF cannot be attached to a trajectory in {}; trajectory covariance is supported only in EME2000 or GCRF axes, so ITRF covariance must be rotated before conversion",
                        frame
                    )));
                }
            };
            Some(rotated)
        } else {
            None
        };

        let states: Vec<DVector<f64>> = states
            .iter()
            .map(|x| DVector::from_column_slice(x.as_slice()))
            .collect();
        let mut traj = DOrbitTrajectory::from_orbital_data(
            epochs,
            states,
            frame,
            OrbitRepresentation::Cartesian,
            None,
            covariances,
        )?;
        traj.name = self.source_name.as_ref().map(|n| n.object_name.clone());
        Ok(traj)
    }

    /// Builds a message from a trajectory's stored samples.
    ///
    /// Equivalent to [`ITC::from_trajectory_with_covariance_variant`] with
    /// `OrbitRelativeFrameVariant::Inertial`.
    ///
    /// # Arguments
    /// * `trajectory` - Six-dimensional Cartesian trajectory
    /// * `header` - Header whose `state_frame`, `covariance_frame`, `created` and `ephemeris_source` are kept; start, stop and step are filled from the samples
    ///
    /// # Returns
    /// * `Ok(ITC)`: The message
    /// * `Err(BraheError)`: If the trajectory is empty, not six-dimensional Cartesian, or its covariance cannot be expressed in the requested frames
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::itc::{ITC, ITCHeader};
    ///
    /// let itc = ITC::from_file(
    ///     "test_assets/starlink/MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt",
    /// ).unwrap();
    /// let traj = itc.to_trajectory().unwrap();
    /// let back = ITC::from_trajectory(&traj, ITCHeader::new().with_ephemeris_source("brahe")).unwrap();
    /// assert_eq!(back.len(), 50);
    /// assert!(back.has_covariance());
    /// ```
    pub fn from_trajectory(
        trajectory: &DOrbitTrajectory,
        header: ITCHeader,
    ) -> Result<Self, BraheError> {
        Self::from_trajectory_with_covariance_variant(
            trajectory,
            header,
            OrbitRelativeFrameVariant::Inertial,
        )
    }

    /// Builds a message from a trajectory, choosing how covariance is rotated
    /// into the RTN frame.
    ///
    /// Each stored sample is converted from the trajectory frame to
    /// `header.state_frame`. When the trajectory carries covariance, its frame
    /// and the state frame must both be GCRF or EME2000; the covariance is
    /// rotated by the constant frame bias where needed and then, for an RTN
    /// covariance frame, into the RTN frame of the converted state. An
    /// `EME2000` covariance frame requires an `EME2000` state frame; `ITRF`
    /// is not supported.
    ///
    /// # Arguments
    /// * `trajectory` - Six-dimensional Cartesian trajectory
    /// * `header` - Header template; `ephemeris_start` and `ephemeris_stop` come from the trajectory's first and last samples, and `step_size` from the interval between the first two samples (`None` for a single sample)
    /// * `variant` - RTN rotation convention for the covariance
    ///
    /// # Returns
    /// * `Ok(ITC)`: The message
    /// * `Err(BraheError)`: If the trajectory is empty, not six-dimensional Cartesian, or the frame combination is unsupported
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::frames::OrbitRelativeFrameVariant;
    /// use brahe::itc::{ITC, ITCHeader};
    ///
    /// let itc = ITC::from_file(
    ///     "test_assets/starlink/MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt",
    /// ).unwrap();
    /// let traj = itc.to_trajectory_with_covariance_variant(OrbitRelativeFrameVariant::Rotating).unwrap();
    /// let back = ITC::from_trajectory_with_covariance_variant(&traj, ITCHeader::new(), OrbitRelativeFrameVariant::Rotating).unwrap();
    /// assert_eq!(back.len(), itc.len());
    /// ```
    pub fn from_trajectory_with_covariance_variant(
        trajectory: &DOrbitTrajectory,
        header: ITCHeader,
        variant: OrbitRelativeFrameVariant,
    ) -> Result<Self, BraheError> {
        if trajectory.states.is_empty() {
            return Err(BraheError::Error(
                "cannot build a Modified ITC message from an empty trajectory".to_string(),
            ));
        }
        if trajectory.representation != OrbitRepresentation::Cartesian
            || trajectory.dimension() != 6
        {
            return Err(BraheError::Error(format!(
                "Modified ITC requires a six-dimensional Cartesian trajectory; found dimension {} in {} representation (convert with to_frame or to_eci first)",
                trajectory.dimension(),
                trajectory.representation
            )));
        }
        let target: ReferenceFrame = header.state_frame.into();
        let covariance_frame = header.covariance_frame;
        let state_frame = header.state_frame;

        let axes_rotation = match &trajectory.covariances {
            Some(_) => {
                if covariance_frame == ITCCovarianceFrame::ITRF {
                    return Err(BraheError::Error(format!(
                        "Modified ITC covariance frame ITRF cannot be produced from a trajectory in {}; trajectory covariance is supported only in EME2000 or GCRF axes",
                        target
                    )));
                }
                if covariance_frame == ITCCovarianceFrame::EME2000
                    && state_frame != CelestialFrame::EME2000
                {
                    return Err(BraheError::Error(format!(
                        "Modified ITC covariance frame EME2000 requires an EME2000 state frame, found {}",
                        target
                    )));
                }
                Some(block_diagonal(inertial_covariance_rotation(
                    &trajectory.frame,
                    &target,
                )?))
            }
            None => None,
        };

        let mut itc = ITC::new(header);
        for (index, (epoch, x)) in trajectory.into_iter().enumerate() {
            let x6 = SVector6::from_column_slice(&x.as_slice()[..6]);
            let x_t = if trajectory.frame == target {
                x6
            } else {
                state_frame_to_frame(trajectory.frame.clone(), target.clone(), epoch, x6)?
            };
            let sv = ITCStateVector::new(epoch, [x_t[0], x_t[1], x_t[2]], [x_t[3], x_t[4], x_t[5]]);
            match (&trajectory.covariances, axes_rotation) {
                (Some(covs), Some(r6)) => {
                    let p = covs.get(index).ok_or_else(|| {
                        BraheError::Error(format!(
                            "trajectory has no covariance for sample {}",
                            index
                        ))
                    })?;
                    if p.nrows() < 6 || p.ncols() < 6 {
                        return Err(BraheError::Error(format!(
                            "trajectory covariance for sample {} is {}x{}; expected at least 6x6",
                            index,
                            p.nrows(),
                            p.ncols()
                        )));
                    }
                    let p6 = SMatrix6::from_fn(|i, k| p[(i, k)]);
                    let p_target = r6 * p6 * r6.transpose();
                    let p_out = match covariance_frame {
                        ITCCovarianceFrame::RTN => {
                            let jinv = frame_to_rtn_jacobian(x_t, variant);
                            jinv * p_target * jinv.transpose()
                        }
                        _ => p_target,
                    };
                    itc.push_state_with_covariance(sv, symmetrize(p_out))?;
                }
                _ => itc.push_state(sv)?,
            }
        }

        itc.header.ephemeris_start = itc.start_epoch();
        itc.header.ephemeris_stop = itc.end_epoch();
        itc.header.step_size =
            (itc.states.len() >= 2).then(|| itc.states[1].epoch - itc.states[0].epoch);
        Ok(itc)
    }
}

impl TryFrom<&ITC> for DOrbitTrajectory {
    type Error = BraheError;

    fn try_from(itc: &ITC) -> Result<Self, Self::Error> {
        itc.to_trajectory()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::rotation_eme2000_to_gcrf;
    use crate::trajectories::traits::{InterpolatableTrajectory, Trajectory};
    use crate::utils::testing::setup_global_test_eop;
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

    const FULL: &str = "test_assets/starlink/MEME_100001_STARLINK-38128_2540142_Operational_1473385380_UNCLASSIFIED.txt";
    const TRUNCATED: &str = "test_assets/starlink/MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt";

    fn assert_matrix_close(a: &DMatrix<f64>, b: &SMatrix6, rel: f64) {
        for i in 0..6 {
            for k in 0..6 {
                let scale = b[(i, k)].abs().max(1e-20);
                assert_abs_diff_eq!(a[(i, k)], b[(i, k)], epsilon = rel * scale);
            }
        }
    }

    fn strip_covariance(text: &str) -> String {
        text.lines()
            .enumerate()
            .filter(|(i, l)| {
                *i < 4
                    || l.split_whitespace().next().is_some_and(|t| {
                        t.len() >= 13 && t[..13].chars().all(|c| c.is_ascii_digit())
                    })
            })
            .map(|(_, l)| l)
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    #[parallel]
    fn test_to_trajectory_full_asset() {
        setup_global_test_eop();
        let itc = ITC::from_file(FULL).unwrap();
        let traj = itc.to_trajectory().unwrap();
        assert_eq!(traj.frame, ReferenceFrame::from(CelestialFrame::EME2000));
        assert_eq!(traj.states.len(), 4321);
        assert_eq!(traj.name.as_deref(), Some("STARLINK-38128"));
        assert_eq!(traj.representation, OrbitRepresentation::Cartesian);

        let x0 = SVector6::from_column_slice(traj.states[0].as_slice());
        let expected = symmetrize(
            block_diagonal(rotation_rtn_to_eci(x0))
                * itc.covariances[0]
                * block_diagonal(rotation_rtn_to_eci(x0)).transpose(),
        );
        let cov0 = traj.covariance_at(itc.states[0].epoch).unwrap().unwrap();
        assert_matrix_close(&cov0, &expected, 1e-9);
        let trace_rtn: f64 = (0..3).map(|i| itc.covariances[0][(i, i)]).sum();
        let trace_out: f64 = (0..3).map(|i| cov0[(i, i)]).sum();
        assert_abs_diff_eq!(trace_rtn, trace_out, epsilon = 1e-9 * trace_rtn);

        let mid = itc.states[0].epoch + 30.0;
        let x_mid = traj.interpolate(&mid).unwrap();
        let r_mid = (x_mid[0].powi(2) + x_mid[1].powi(2) + x_mid[2].powi(2)).sqrt();
        assert!(r_mid > 6.5e6 && r_mid < 7.5e6);
        let d0 =
            ((x_mid[0] - x0[0]).powi(2) + (x_mid[1] - x0[1]).powi(2) + (x_mid[2] - x0[2]).powi(2))
                .sqrt();
        assert!(d0 > 100.0e3 && d0 < 300.0e3);
    }

    #[test]
    #[parallel]
    fn test_to_trajectory_to_eci_applies_frame_bias() {
        setup_global_test_eop();
        let traj = ITC::from_file(TRUNCATED).unwrap().to_trajectory().unwrap();
        let eci = traj.to_eci().unwrap();
        let a = &traj.states[0];
        let b = &eci.states[0];
        let d = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
        assert!(d > 0.1 && d < 2.0, "bias displacement {} m", d);

        let r_eme = nalgebra::Vector3::new(a[0], a[1], a[2]);
        let expected = rotation_eme2000_to_gcrf() * r_eme;
        for i in 0..3 {
            assert_abs_diff_eq!(b[i], expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    #[parallel]
    fn test_to_trajectory_rotating_variant_preserves_position_block() {
        setup_global_test_eop();
        let itc = ITC::from_file(TRUNCATED).unwrap();
        let inertial = itc.to_trajectory().unwrap();
        let rotating = itc
            .to_trajectory_with_covariance_variant(OrbitRelativeFrameVariant::Rotating)
            .unwrap();
        let e = itc.states[0].epoch;
        let a = inertial.covariance_at(e).unwrap().unwrap();
        let b = rotating.covariance_at(e).unwrap().unwrap();
        for i in 0..3 {
            for k in 0..3 {
                assert_abs_diff_eq!(
                    a[(i, k)],
                    b[(i, k)],
                    epsilon = 1e-9 * a[(i, k)].abs().max(1e-20)
                );
            }
        }
        let mut differs = false;
        for i in 3..6 {
            for k in 0..6 {
                if (a[(i, k)] - b[(i, k)]).abs() > 1e-12 * a[(i, k)].abs().max(1e-20) {
                    differs = true;
                }
            }
        }
        assert!(differs);
    }

    #[test]
    #[parallel]
    fn test_to_trajectory_without_covariance() {
        setup_global_test_eop();
        let content = std::fs::read_to_string(TRUNCATED).unwrap();
        let no_cov = strip_covariance(&content);
        let itc = ITC::from_str(&no_cov).unwrap();
        assert!(!itc.has_covariance());
        let traj = itc.to_trajectory().unwrap();
        assert_eq!(traj.states.len(), 50);
        assert!(traj.covariance_at(itc.states[0].epoch).unwrap().is_none());
        assert!(traj.name.is_none());
    }

    #[test]
    #[parallel]
    fn test_to_trajectory_frame_errors() {
        setup_global_test_eop();
        let mut itc = ITC::from_file(TRUNCATED).unwrap();
        itc.header.state_frame = CelestialFrame::TEME;
        assert!(itc.to_trajectory().is_err());
        itc.header.state_frame = CelestialFrame::ITRF;
        assert!(itc.to_trajectory().is_err());
        itc.header.state_frame = CelestialFrame::EME2000;
        itc.header.covariance_frame = ITCCovarianceFrame::ITRF;
        assert!(itc.to_trajectory().is_err());
        itc.header.covariance_frame = ITCCovarianceFrame::EME2000;
        assert!(itc.to_trajectory().is_ok());
        itc.header.state_frame = CelestialFrame::GCRF;
        assert!(itc.to_trajectory().is_err());
        let empty = ITC::new(ITCHeader::new());
        assert!(empty.to_trajectory().is_err());
        let via_try: Result<DOrbitTrajectory, _> = (&ITC::from_file(TRUNCATED).unwrap()).try_into();
        assert!(via_try.is_ok());
    }

    #[test]
    #[parallel]
    fn test_to_trajectory_teme_without_covariance_is_allowed() {
        setup_global_test_eop();
        let content = std::fs::read_to_string(TRUNCATED).unwrap();
        let no_cov = strip_covariance(&content);
        let mut itc = ITC::from_str(&no_cov).unwrap();
        itc.header.state_frame = CelestialFrame::TEME;
        let traj = itc.to_trajectory().unwrap();
        assert_eq!(traj.frame, ReferenceFrame::from(CelestialFrame::TEME));
    }

    #[test]
    #[parallel]
    fn test_to_trajectory_feeds_location_accesses() {
        setup_global_test_eop();
        let itc = ITC::from_file(FULL).unwrap();
        let traj = itc.to_trajectory().unwrap();
        let location = crate::access::PointLocation::new(-122.4194, 37.7749, 0.0).unwrap();
        let constraint = crate::access::ElevationConstraint::new(Some(10.0), None).unwrap();
        let start = itc.start_epoch().unwrap();
        let end = start + 12.0 * 3600.0;
        let windows =
            crate::access::location_accesses(&location, &traj, start, end, &constraint, None, None)
                .unwrap();
        assert!(!windows.is_empty());
        for w in &windows {
            assert!(w.window_open >= start && w.window_close <= end);
        }
    }

    #[test]
    #[parallel]
    fn test_from_trajectory_round_trip() {
        setup_global_test_eop();
        let itc = ITC::from_file(TRUNCATED).unwrap();
        let traj = itc.to_trajectory().unwrap();
        let back =
            ITC::from_trajectory(&traj, ITCHeader::new().with_ephemeris_source("round-trip"))
                .unwrap();
        assert_eq!(back.len(), 50);
        assert_eq!(back.header.ephemeris_source.as_deref(), Some("round-trip"));
        assert_eq!(back.header.ephemeris_start, Some(itc.states[0].epoch));
        assert_eq!(back.header.ephemeris_stop, Some(itc.states[49].epoch));
        assert_abs_diff_eq!(back.header.step_size.unwrap(), 60.0, epsilon = 1e-9);
        for (a, b) in itc.states.iter().zip(&back.states) {
            assert_eq!(a.epoch, b.epoch);
            for i in 0..3 {
                assert_abs_diff_eq!(a.position[i], b.position[i], epsilon = 1e-6);
                assert_abs_diff_eq!(a.velocity[i], b.velocity[i], epsilon = 1e-9);
            }
        }
        assert_eq!(back.covariances.len(), itc.covariances.len());
        assert_eq!(back.covariances.len(), 50);
        for (a, b) in itc.covariances.iter().zip(&back.covariances) {
            for i in 0..6 {
                for k in 0..6 {
                    assert_abs_diff_eq!(
                        a[(i, k)],
                        b[(i, k)],
                        epsilon = 1e-9 * a[(i, k)].abs().max(1e-20)
                    );
                }
            }
        }
        for variant in [
            OrbitRelativeFrameVariant::Inertial,
            OrbitRelativeFrameVariant::Rotating,
        ] {
            let t = itc.to_trajectory_with_covariance_variant(variant).unwrap();
            let b = ITC::from_trajectory_with_covariance_variant(&t, ITCHeader::new(), variant)
                .unwrap();
            assert_abs_diff_eq!(
                b.covariances[0][(3, 1)],
                itc.covariances[0][(3, 1)],
                epsilon = 1e-9 * itc.covariances[0][(3, 1)].abs()
            );
        }
    }

    #[test]
    #[parallel]
    fn test_from_trajectory_eme2000_covariance_frame_keeps_inertial_covariance() {
        setup_global_test_eop();
        let itc = ITC::from_file(TRUNCATED).unwrap();
        let traj = itc.to_trajectory().unwrap();
        let back = ITC::from_trajectory(
            &traj,
            ITCHeader::new().with_covariance_frame(ITCCovarianceFrame::EME2000),
        )
        .unwrap();
        let cov0 = traj.covariance_at(itc.states[0].epoch).unwrap().unwrap();
        for i in 0..6 {
            for k in 0..6 {
                assert_abs_diff_eq!(
                    back.covariances[0][(i, k)],
                    cov0[(i, k)],
                    epsilon = 1e-9 * cov0[(i, k)].abs().max(1e-20)
                );
            }
        }
    }

    #[test]
    #[parallel]
    fn test_from_trajectory_gcrf_covariance_rotates_through_bias() {
        setup_global_test_eop();
        let itc = ITC::from_file(TRUNCATED).unwrap();
        let eme = itc.to_trajectory().unwrap();
        let bias = crate::frames::rotation_eme2000_to_gcrf();
        let b6 = block_diagonal(bias);
        let epochs = eme.epochs.clone();
        let states: Vec<DVector<f64>> = eme
            .states
            .iter()
            .map(|s| {
                let x = SVector6::from_column_slice(s.as_slice());
                DVector::from_column_slice((b6 * x).as_slice())
            })
            .collect();
        let covs: Vec<DMatrix<f64>> = eme
            .covariances
            .as_ref()
            .unwrap()
            .iter()
            .map(|p| {
                let p6 = SMatrix6::from_iterator(p.iter().cloned());
                dmatrix_from(&(b6 * p6 * b6.transpose()))
            })
            .collect();
        let gcrf = DOrbitTrajectory::from_orbital_data(
            epochs,
            states,
            CelestialFrame::GCRF,
            OrbitRepresentation::Cartesian,
            None,
            Some(covs),
        )
        .unwrap();
        let back = ITC::from_trajectory(&gcrf, ITCHeader::new()).unwrap();
        for (a, b) in itc.states.iter().zip(&back.states) {
            for i in 0..3 {
                assert_abs_diff_eq!(a.position[i], b.position[i], epsilon = 1e-5);
                assert_abs_diff_eq!(a.velocity[i], b.velocity[i], epsilon = 1e-8);
            }
        }
        assert_eq!(back.covariances.len(), itc.covariances.len());
        assert_eq!(back.covariances.len(), 50);
        for (a, b) in itc.covariances.iter().zip(&back.covariances) {
            for i in 0..6 {
                for k in 0..6 {
                    assert_abs_diff_eq!(
                        a[(i, k)],
                        b[(i, k)],
                        epsilon = 1e-8 * a[(i, k)].abs().max(1e-20)
                    );
                }
            }
        }
    }

    #[test]
    #[parallel]
    fn test_from_trajectory_without_covariance_and_frame_conversion() {
        setup_global_test_eop();
        let content = std::fs::read_to_string(TRUNCATED).unwrap();
        let no_cov = strip_covariance(&content);
        let itc = ITC::from_str(&no_cov).unwrap();
        let eci = itc.to_trajectory().unwrap().to_eci().unwrap();
        assert!(eci.covariances.is_none());
        let back = ITC::from_trajectory(&eci, ITCHeader::new()).unwrap();
        assert!(!back.has_covariance());
        for (a, b) in itc.states.iter().zip(&back.states) {
            for i in 0..3 {
                assert_abs_diff_eq!(a.position[i], b.position[i], epsilon = 1e-5);
            }
        }
        let teme = ITC::from_trajectory(
            &eci,
            ITCHeader::new().with_state_frame(CelestialFrame::TEME),
        )
        .unwrap();
        assert_eq!(teme.header.state_frame, CelestialFrame::TEME);
        let d = (teme.states[0].position[0] - itc.states[0].position[0]).abs();
        assert!(d > 1.0e3);
    }

    #[test]
    #[parallel]
    fn test_from_trajectory_errors() {
        setup_global_test_eop();
        let itc = ITC::from_file(TRUNCATED).unwrap();
        let traj = itc.to_trajectory().unwrap();
        assert!(
            ITC::from_trajectory(
                &traj,
                ITCHeader::new().with_covariance_frame(ITCCovarianceFrame::ITRF)
            )
            .is_err()
        );
        assert!(
            ITC::from_trajectory(
                &traj,
                ITCHeader::new().with_state_frame(CelestialFrame::TEME)
            )
            .is_err()
        );
        assert!(
            ITC::from_trajectory(
                &traj,
                ITCHeader::new()
                    .with_state_frame(CelestialFrame::GCRF)
                    .with_covariance_frame(ITCCovarianceFrame::EME2000)
            )
            .is_err()
        );
        let gcrf_rtn = ITC::from_trajectory(
            &traj,
            ITCHeader::new().with_state_frame(CelestialFrame::GCRF),
        )
        .unwrap();
        assert_eq!(gcrf_rtn.header.state_frame, CelestialFrame::GCRF);

        let mut kep = traj.clone();
        kep.representation = OrbitRepresentation::Keplerian;
        assert!(ITC::from_trajectory(&kep, ITCHeader::new()).is_err());

        let empty = DOrbitTrajectory::new(
            6,
            CelestialFrame::EME2000,
            OrbitRepresentation::Cartesian,
            None,
        )
        .unwrap();
        assert!(ITC::from_trajectory(&empty, ITCHeader::new()).is_err());

        let mut seven = DOrbitTrajectory::new(
            7,
            CelestialFrame::EME2000,
            OrbitRepresentation::Cartesian,
            None,
        )
        .unwrap();
        seven
            .add(
                itc.states[0].epoch,
                DVector::from_column_slice(&[7.0e6, 0.0, 0.0, 0.0, 7.5e3, 0.0, 1.0]),
            )
            .unwrap();
        assert!(ITC::from_trajectory(&seven, ITCHeader::new()).is_err());
    }
}
