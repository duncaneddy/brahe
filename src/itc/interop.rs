/*!
 * Conversions between Modified ITC messages and orbit trajectories.
 */

use nalgebra::{DMatrix, DVector};

use crate::frames::{
    CelestialFrame, OrbitRelativeFrameVariant, ReferenceFrame, rotate_covariance_6,
    state_frame_to_frame, state_transform_jacobian,
};
use crate::math::linalg::{SMatrix6, SVector6};
use crate::relative_motion::{covariance_eci_to_rtn, covariance_rtn_to_eci};
use crate::time::Epoch;
use crate::trajectories::dorbit_trajectory::DOrbitTrajectory;
use crate::trajectories::traits::{OrbitRepresentation, is_icrf_axes_frame};
use crate::utils::BraheError;

use super::types::{ITC, ITCCovarianceFrame, ITCHeader, ITCStateVector};

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

/// Celestial frame a non-RTN Modified ITC covariance frame names.
///
/// # Arguments
/// * `frame` - Covariance frame from the message header
///
/// # Returns
/// * `Some(CelestialFrame)`: `EME2000` or `ITRF`
/// * `None`: For [`ITCCovarianceFrame::RTN`], which is orbit-relative and has no fixed axes
///
/// # Examples
/// ```text
/// celestial_covariance_frame(ITCCovarianceFrame::ITRF) is Some(CelestialFrame::ITRF)
/// celestial_covariance_frame(ITCCovarianceFrame::RTN) is None
/// ```
fn celestial_covariance_frame(frame: ITCCovarianceFrame) -> Option<CelestialFrame> {
    match frame {
        ITCCovarianceFrame::RTN => None,
        ITCCovarianceFrame::EME2000 => Some(CelestialFrame::EME2000),
        ITCCovarianceFrame::ITRF => Some(CelestialFrame::ITRF),
    }
}

/// State expressed in ICRF axes, for building the RTN frame it defines.
///
/// A frame that already carries ICRF axes is returned untouched, so the RTN
/// frame of a GCRF state costs no router call; every other frame is routed to
/// GCRF.
///
/// # Arguments
/// * `frame` - Frame `x` is expressed in
/// * `epoch` - Epoch of the state
/// * `x` - 6-element Cartesian state (position, m; velocity, m/s)
///
/// # Returns
/// * `Ok(SVector6)`: The state in ICRF axes
/// * `Err(BraheError)`: If the router cannot reach GCRF from `frame` at this epoch
///
/// # Examples
/// ```text
/// state_in_icrf_axes(GCRF, epoch, x) returns x unchanged;
/// state_in_icrf_axes(ITRF, epoch, x) returns the GCRF state.
/// ```
fn state_in_icrf_axes(
    frame: &ReferenceFrame,
    epoch: Epoch,
    x: SVector6,
) -> Result<SVector6, BraheError> {
    if is_icrf_axes_frame(frame) {
        Ok(x)
    } else {
        state_frame_to_frame(frame.clone(), CelestialFrame::GCRF, epoch, x)
    }
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
    /// * `Err(BraheError)`: If the message is empty, or the router cannot reach the state frame at a record's epoch
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::itc::ITC;
    /// # brahe::eop::set_global_eop_provider(
    /// #     brahe::eop::StaticEOPProvider::from_values((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    /// # );
    ///
    /// let itc = ITC::from_file(
    ///     "test_assets/starlink/MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt",
    /// ).unwrap();
    /// let traj = itc.to_trajectory().unwrap();
    /// assert_eq!(traj.states.len(), 50);
    /// assert_eq!(traj.name.as_deref(), Some("STARLINK-37711"));
    /// ```
    ///
    /// # References
    /// 1. *Spaceflight Safety Handbook for Satellite Operators*, Version 1.7, 18th Space Defense
    ///    Squadron, Space-Track.org,
    ///    <https://www.space-track.org/documents/Spaceflight_Safety_Handbook_for_Operators.pdf>
    /// 2. NASA Conjunction Assessment Risk Analysis (CARA), *Conjunction Assessment Handbook*,
    ///    NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13),
    ///    <https://ntrs.nasa.gov/citations/20205011318>
    /// 3. NASA CARA Analysis Tools, `RIC2ECI.m`, <https://github.com/nasa/CARA_Analysis_Tools>
    /// 4. D. A. Vallado and S. Alfano, "Covariance Transformations for Satellite Flight Dynamics
    ///    Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003,
    ///    <https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf>
    pub fn to_trajectory(&self) -> Result<DOrbitTrajectory, BraheError> {
        self.to_trajectory_with_covariance_variant(OrbitRelativeFrameVariant::Inertial)
    }

    /// Converts the message to a trajectory, choosing how RTN covariance is
    /// rotated into the state frame.
    ///
    /// The covariance is attached in the state frame, whatever frame the
    /// header names for it. An `RTN` covariance is rotated with the record's
    /// own state, taken in ICRF axes about the state frame's center (the
    /// format's frames are all Earth-centered, so the RTN basis is the
    /// geocentric one): `Inertial` uses `[[R, 0], [0, R]]`,
    /// `Rotating` uses `[[R, 0], [R·[ω×], R]]` with ω the RTN frame rate. An
    /// `EME2000` or `ITRF` covariance is rotated by the state-transform
    /// Jacobian from that frame to the state frame, which is the identity when
    /// the two agree and carries the Earth-rotation coupling when one of them
    /// is Earth-fixed.
    ///
    /// # Arguments
    /// * `variant` - RTN rotation convention for the covariance
    ///
    /// # Returns
    /// * `Ok(DOrbitTrajectory)`: The trajectory
    /// * `Err(BraheError)`: If the message is empty, or the router cannot relate the covariance frame and the state frame at a record's epoch
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::frames::OrbitRelativeFrameVariant;
    /// use brahe::itc::ITC;
    /// # brahe::eop::set_global_eop_provider(
    /// #     brahe::eop::StaticEOPProvider::from_values((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    /// # );
    ///
    /// let itc = ITC::from_file(
    ///     "test_assets/starlink/MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt",
    /// ).unwrap();
    /// let traj = itc.to_trajectory_with_covariance_variant(OrbitRelativeFrameVariant::Rotating).unwrap();
    /// assert!(traj.covariance_at(itc.states[0].epoch).unwrap().is_some());
    /// ```
    ///
    /// # References
    /// 1. *Spaceflight Safety Handbook for Satellite Operators*, Version 1.7, 18th Space Defense
    ///    Squadron, Space-Track.org,
    ///    <https://www.space-track.org/documents/Spaceflight_Safety_Handbook_for_Operators.pdf>
    /// 2. NASA Conjunction Assessment Risk Analysis (CARA), *Conjunction Assessment Handbook*,
    ///    NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13),
    ///    <https://ntrs.nasa.gov/citations/20205011318>
    /// 3. NASA CARA Analysis Tools, `RIC2ECI.m`, <https://github.com/nasa/CARA_Analysis_Tools>
    /// 4. D. A. Vallado and S. Alfano, "Covariance Transformations for Satellite Flight Dynamics
    ///    Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003,
    ///    <https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf>
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
        let states: Vec<SVector6> = self.states.iter().map(ITCStateVector::to_vector).collect();

        let covariances = if self.has_covariance() {
            let source = celestial_covariance_frame(self.header.covariance_frame);
            let mut rotated = Vec::with_capacity(self.covariances.len());
            for ((epoch, x), p) in epochs.iter().zip(&states).zip(&self.covariances) {
                let p_state = match source {
                    Some(cov_frame) => rotate_covariance_6(
                        p,
                        &state_transform_jacobian(cov_frame, frame.clone(), *epoch)?,
                    ),
                    None => {
                        let x_icrf = state_in_icrf_axes(&frame, *epoch, *x)?;
                        let p_icrf = covariance_rtn_to_eci(x_icrf, p, variant);
                        rotate_covariance_6(
                            &p_icrf,
                            &state_transform_jacobian(CelestialFrame::GCRF, frame.clone(), *epoch)?,
                        )
                    }
                };
                rotated.push(dmatrix_from(&p_state));
            }
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
    /// * `Err(BraheError)`: If the trajectory is empty, not six-dimensional Cartesian, or the router cannot reach the header's frames
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::itc::{ITC, ITCHeader};
    /// # brahe::eop::set_global_eop_provider(
    /// #     brahe::eop::StaticEOPProvider::from_values((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    /// # );
    ///
    /// let itc = ITC::from_file(
    ///     "test_assets/starlink/MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt",
    /// ).unwrap();
    /// let traj = itc.to_trajectory().unwrap();
    /// let back = ITC::from_trajectory(&traj, ITCHeader::new().with_ephemeris_source("brahe")).unwrap();
    /// assert_eq!(back.len(), 50);
    /// assert!(back.has_covariance());
    /// ```
    ///
    /// # References
    /// 1. *Spaceflight Safety Handbook for Satellite Operators*, Version 1.7, 18th Space Defense
    ///    Squadron, Space-Track.org,
    ///    <https://www.space-track.org/documents/Spaceflight_Safety_Handbook_for_Operators.pdf>
    /// 2. NASA Conjunction Assessment Risk Analysis (CARA), *Conjunction Assessment Handbook*,
    ///    NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13),
    ///    <https://ntrs.nasa.gov/citations/20205011318>
    /// 3. NASA CARA Analysis Tools, `RIC2ECI.m`, <https://github.com/nasa/CARA_Analysis_Tools>
    /// 4. D. A. Vallado and S. Alfano, "Covariance Transformations for Satellite Flight Dynamics
    ///    Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003,
    ///    <https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf>
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
    /// `header.state_frame`. A covariance is rotated from the trajectory frame
    /// into `header.covariance_frame`: `EME2000` and `ITRF` through the
    /// state-transform Jacobian between the two frames, `RTN` through ICRF axes
    /// and the RTN frame of the sample's own state. The trajectory must hold a
    /// six-dimensional Cartesian state.
    ///
    /// # Arguments
    /// * `trajectory` - Six-dimensional Cartesian trajectory
    /// * `header` - Header template; `ephemeris_start` and `ephemeris_stop` come from the trajectory's first and last samples, and `step_size` from the interval between the first two samples (`None` for a single sample)
    /// * `variant` - RTN rotation convention for the covariance
    ///
    /// # Returns
    /// * `Ok(ITC)`: The message
    /// * `Err(BraheError)`: If the trajectory is empty, not six-dimensional Cartesian, a covariance is smaller than 6x6, or the router cannot reach the header's frames
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::frames::OrbitRelativeFrameVariant;
    /// use brahe::itc::{ITC, ITCHeader};
    /// # brahe::eop::set_global_eop_provider(
    /// #     brahe::eop::StaticEOPProvider::from_values((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    /// # );
    ///
    /// let itc = ITC::from_file(
    ///     "test_assets/starlink/MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt",
    /// ).unwrap();
    /// let traj = itc.to_trajectory_with_covariance_variant(OrbitRelativeFrameVariant::Rotating).unwrap();
    /// let back = ITC::from_trajectory_with_covariance_variant(&traj, ITCHeader::new(), OrbitRelativeFrameVariant::Rotating).unwrap();
    /// assert_eq!(back.len(), itc.len());
    /// ```
    ///
    /// # References
    /// 1. *Spaceflight Safety Handbook for Satellite Operators*, Version 1.7, 18th Space Defense
    ///    Squadron, Space-Track.org,
    ///    <https://www.space-track.org/documents/Spaceflight_Safety_Handbook_for_Operators.pdf>
    /// 2. NASA Conjunction Assessment Risk Analysis (CARA), *Conjunction Assessment Handbook*,
    ///    NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13),
    ///    <https://ntrs.nasa.gov/citations/20205011318>
    /// 3. NASA CARA Analysis Tools, `RIC2ECI.m`, <https://github.com/nasa/CARA_Analysis_Tools>
    /// 4. D. A. Vallado and S. Alfano, "Covariance Transformations for Satellite Flight Dynamics
    ///    Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003,
    ///    <https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf>
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
        let covariance_target = celestial_covariance_frame(header.covariance_frame);

        let mut itc = ITC::new(header);
        for (index, (epoch, x)) in trajectory.into_iter().enumerate() {
            let x6 = SVector6::from_column_slice(&x.as_slice()[..6]);
            let x_t = if trajectory.frame == target {
                x6
            } else {
                state_frame_to_frame(trajectory.frame.clone(), target.clone(), epoch, x6)?
            };
            let sv = ITCStateVector::new(epoch, [x_t[0], x_t[1], x_t[2]], [x_t[3], x_t[4], x_t[5]]);
            match &trajectory.covariances {
                Some(covs) => {
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
                    let p_out = match covariance_target {
                        Some(cov_frame) => rotate_covariance_6(
                            &p6,
                            &state_transform_jacobian(trajectory.frame.clone(), cov_frame, epoch)?,
                        ),
                        None => {
                            let x_icrf = state_in_icrf_axes(&trajectory.frame, epoch, x6)?;
                            let p_icrf = rotate_covariance_6(
                                &p6,
                                &state_transform_jacobian(
                                    trajectory.frame.clone(),
                                    CelestialFrame::GCRF,
                                    epoch,
                                )?,
                            );
                            covariance_eci_to_rtn(x_icrf, &p_icrf, variant)
                        }
                    };
                    itc.push_state_with_covariance(sv, p_out)?;
                }
                None => itc.push_state(sv)?,
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
    use crate::math::{block_diagonal, symmetrize_6};
    use crate::relative_motion::rotation_rtn_to_eci;
    use crate::trajectories::traits::{InterpolatableTrajectory, Trajectory};
    use crate::utils::testing::setup_global_test_eop;
    use approx::assert_abs_diff_eq;
    use serial_test::parallel;

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

    /// Compares two covariances element by element, each tolerance scaled by
    /// the Cauchy-Schwarz bound `sqrt(b_ii b_kk)` on the element rather than by
    /// the element itself. Off-diagonal covariance entries are differences of
    /// much larger products, so an element-relative tolerance measures the
    /// cancellation in the input data rather than the accuracy of the
    /// transformation.
    fn assert_covariance_close(a: &SMatrix6, b: &SMatrix6, rel: f64) {
        for i in 0..6 {
            for k in 0..6 {
                let scale = (b[(i, i)] * b[(k, k)]).sqrt();
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
        let r = rotation_rtn_to_eci(x0);
        let j = block_diagonal(&r, &r);
        let expected = symmetrize_6(&(j * itc.covariances[0] * j.transpose()));
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
    fn test_to_trajectory_accepts_every_frame_combination() {
        setup_global_test_eop();
        let mut itc = ITC::from_file(TRUNCATED).unwrap();
        for state_frame in [
            CelestialFrame::EME2000,
            CelestialFrame::GCRF,
            CelestialFrame::TEME,
            CelestialFrame::ITRF,
        ] {
            for covariance_frame in [
                ITCCovarianceFrame::RTN,
                ITCCovarianceFrame::EME2000,
                ITCCovarianceFrame::ITRF,
            ] {
                itc.header.state_frame = state_frame;
                itc.header.covariance_frame = covariance_frame;
                let traj = itc.to_trajectory().unwrap();
                assert_eq!(traj.frame, ReferenceFrame::from(state_frame));
                let cov = traj
                    .covariance_at(itc.states[0].epoch)
                    .unwrap()
                    .expect("covariance attached");
                assert!(cov[(0, 0)] > 0.0, "{} / {}", state_frame, covariance_frame);
                let identity_pair = (state_frame == CelestialFrame::ITRF
                    && covariance_frame == ITCCovarianceFrame::ITRF)
                    || (state_frame == CelestialFrame::EME2000
                        && covariance_frame == ITCCovarianceFrame::EME2000);
                if identity_pair {
                    let original = itc.covariances[0];
                    for i in 0..6 {
                        for k in 0..6 {
                            assert_eq!(
                                cov[(i, k)],
                                original[(i, k)],
                                "{} / {}",
                                state_frame,
                                covariance_frame
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[parallel]
    fn test_to_trajectory_empty_message_is_error() {
        setup_global_test_eop();
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
            assert_covariance_close(b, a, 1e-11);
        }
    }

    #[test]
    #[parallel]
    fn test_from_trajectory_rtn_round_trips_for_both_variants() {
        setup_global_test_eop();
        let itc = ITC::from_file(TRUNCATED).unwrap();
        for variant in [
            OrbitRelativeFrameVariant::Inertial,
            OrbitRelativeFrameVariant::Rotating,
        ] {
            let traj = itc.to_trajectory_with_covariance_variant(variant).unwrap();
            let back =
                ITC::from_trajectory_with_covariance_variant(&traj, ITCHeader::new(), variant)
                    .unwrap();
            assert_eq!(back.header.covariance_frame, ITCCovarianceFrame::RTN);
            for (a, b) in itc.covariances.iter().zip(&back.covariances) {
                assert_covariance_close(b, a, 1e-11);
            }
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
    fn test_itrf_covariance_round_trips_through_an_eme2000_trajectory() {
        setup_global_test_eop();
        // Relabel the fixture's covariance as ITRF: the numbers no longer
        // describe the same physical uncertainty, but the pair of rotations
        // must still invert one another exactly.
        let mut itc = ITC::from_file(TRUNCATED).unwrap();
        itc.header.covariance_frame = ITCCovarianceFrame::ITRF;
        let traj = itc.to_trajectory().unwrap();
        assert_eq!(traj.frame, ReferenceFrame::from(CelestialFrame::EME2000));

        // The trajectory carries the covariance in EME2000, so it differs from
        // the ITRF input by the full Earth-rotation Jacobian.
        let cov0 = traj.covariance_at(itc.states[0].epoch).unwrap().unwrap();
        assert!((cov0[(0, 0)] - itc.covariances[0][(0, 0)]).abs() > 1e-6);

        let back = ITC::from_trajectory(
            &traj,
            ITCHeader::new().with_covariance_frame(ITCCovarianceFrame::ITRF),
        )
        .unwrap();
        assert_eq!(back.header.covariance_frame, ITCCovarianceFrame::ITRF);
        for (a, b) in itc.covariances.iter().zip(&back.covariances) {
            assert_covariance_close(b, a, 1e-11);
        }
    }

    #[test]
    #[parallel]
    fn test_eme2000_covariance_round_trips_through_a_teme_trajectory() {
        setup_global_test_eop();
        let mut itc = ITC::from_file(TRUNCATED).unwrap();
        itc.header.state_frame = CelestialFrame::TEME;
        itc.header.covariance_frame = ITCCovarianceFrame::EME2000;
        let traj = itc.to_trajectory().unwrap();
        assert_eq!(traj.frame, ReferenceFrame::from(CelestialFrame::TEME));

        let back = ITC::from_trajectory(
            &traj,
            ITCHeader::new()
                .with_state_frame(CelestialFrame::TEME)
                .with_covariance_frame(ITCCovarianceFrame::EME2000),
        )
        .unwrap();
        for (a, b) in itc.covariances.iter().zip(&back.covariances) {
            assert_covariance_close(b, a, 1e-11);
        }
    }

    #[test]
    #[parallel]
    fn test_from_trajectory_gcrf_covariance_rotates_through_bias() {
        setup_global_test_eop();
        let itc = ITC::from_file(TRUNCATED).unwrap();
        let eme = itc.to_trajectory().unwrap();
        let bias = rotation_eme2000_to_gcrf();
        let b6 = block_diagonal(&bias, &bias);
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
            assert_covariance_close(b, a, 1e-10);
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
    fn test_from_trajectory_teme_state_frame_carries_covariance() {
        setup_global_test_eop();
        let itc = ITC::from_file(TRUNCATED).unwrap();
        let traj = itc.to_trajectory().unwrap();
        let teme = ITC::from_trajectory(
            &traj,
            ITCHeader::new().with_state_frame(CelestialFrame::TEME),
        )
        .unwrap();
        assert_eq!(teme.header.state_frame, CelestialFrame::TEME);
        assert!(teme.has_covariance());
        // The covariance frame stays RTN, which is tied to the orbit rather
        // than to the state frame, so the numbers survive the frame change.
        for (a, b) in itc.covariances.iter().zip(&teme.covariances) {
            assert_covariance_close(b, a, 1e-11);
        }
    }

    #[test]
    #[parallel]
    fn test_from_trajectory_errors() {
        setup_global_test_eop();
        let itc = ITC::from_file(TRUNCATED).unwrap();
        let traj = itc.to_trajectory().unwrap();

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
