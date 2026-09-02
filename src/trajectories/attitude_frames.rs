/*!
 * Frame composition for [`AttitudeTrajectory`].
 *
 * [`AttitudeTrajectory`] stores an attitude relating its own two frame
 * endpoints. This module adds the composition that re-expresses that
 * attitude against an arbitrary frame, which needs the frames router.
 */

use crate::attitude::{FromAttitude, Quaternion, RotationMatrix};
use crate::frames::{OrientationProvider, ReferenceFrame, rotation_frame_to_frame};
use crate::time::Epoch;
use crate::utils::errors::BraheError;

use super::attitude_trajectory::AttitudeTrajectory;

impl AttitudeTrajectory {
    /// Re-expresses this trajectory's attitude relative to an arbitrary
    /// reference frame `from`, given that `frame_a` is itself a
    /// [`ReferenceFrame::Celestial`] frame.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the attitude
    /// * `from` - Reference frame to express the attitude relative to
    ///
    /// # Returns
    /// * `Ok(Quaternion)` - Attitude quaternion from `from` to `frame_b` at `epoch`
    /// * `Err(BraheError)` - If `frame_a` is not `ReferenceFrame::Celestial`, the frame
    ///   transformation from `from` to `frame_a`'s reference frame fails, or the
    ///   attitude at `epoch` cannot be computed
    ///
    /// # Examples
    /// ```
    /// use brahe::attitude::Quaternion;
    /// use brahe::frames::{BodyFrame, CelestialFrame, ReferenceFrame};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::traits::Trajectory;
    /// use brahe::trajectories::{AttitudeState, AttitudeTrajectory};
    ///
    /// // GCRF <-> EME2000 is a fixed frame-bias rotation and needs no EOP data.
    /// let mut traj = AttitudeTrajectory::new(
    ///     ReferenceFrame::from(CelestialFrame::GCRF),
    ///     ReferenceFrame::from(BodyFrame::SCBody(None)),
    /// );
    /// let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// traj.add(epoch, AttitudeState::new(Quaternion::new(1.0, 0.0, 0.0, 0.0))).unwrap();
    ///
    /// let q = traj.quaternion_from_frame(epoch, CelestialFrame::EME2000).unwrap();
    /// ```
    pub fn quaternion_from_frame(
        &self,
        epoch: Epoch,
        from: impl Into<ReferenceFrame>,
    ) -> Result<Quaternion, BraheError> {
        let a = match &self.frame_a {
            ReferenceFrame::Celestial(celestial) => *celestial,
            ReferenceFrame::OrbitRelative { .. } => {
                return Err(BraheError::Error(
                    "quaternion_from_frame requires frame_a to be ReferenceFrame::Celestial, but \
                     this trajectory's frame_a is ReferenceFrame::OrbitRelative"
                        .to_string(),
                ));
            }
            ReferenceFrame::Body { .. } => {
                return Err(BraheError::Error(
                    "quaternion_from_frame requires frame_a to be ReferenceFrame::Celestial, but \
                     this trajectory's frame_a is ReferenceFrame::Body"
                        .to_string(),
                ));
            }
        };

        let r_from_to_a = rotation_frame_to_frame(from, a, epoch)?;
        let q_from_to_a =
            Quaternion::from_rotation_matrix(RotationMatrix::from_matrix(r_from_to_a)?);

        // The stored quaternion rotates `a` into frame_b, so the result is
        // the composition `from -> a` then `a -> b`. brahe's Hamilton
        // product applies its left operand first (R(x * y) = R(y) * R(x)),
        // so that composition is written left-to-right rather than reversed.
        Ok(q_from_to_a * self.quaternion(epoch)?)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::attitude::{EulerAngle, ToAttitude};
    use crate::frames::CelestialFrame;
    use crate::time::{Epoch, TimeSystem};
    use nalgebra::Vector3;

    use crate::attitude::Quaternion;
    use crate::frames::BodyFrame;
    use crate::traits::Trajectory;
    use crate::trajectories::AttitudeState;

    fn spacecraft_frames() -> (ReferenceFrame, ReferenceFrame) {
        (
            ReferenceFrame::from(BodyFrame::SCBody(None)),
            ReferenceFrame::from(BodyFrame::SCBody(None)),
        )
    }

    /// Quaternion for a rotation of `theta` radians about the z-axis.
    fn z_axis_quaternion(theta: f64) -> Quaternion {
        Quaternion::new((theta / 2.0).cos(), 0.0, 0.0, (theta / 2.0).sin())
    }

    fn small_attitude_trajectory() -> AttitudeTrajectory {
        let (a, b) = spacecraft_frames();
        let mut traj = AttitudeTrajectory::new(a, b);
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        traj.add(t0, AttitudeState::new(z_axis_quaternion(0.0)))
            .unwrap();
        traj.add(t0 + 60.0, AttitudeState::new(z_axis_quaternion(0.2)))
            .unwrap();
        traj
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_provider_quaternion_and_defaults_consistent() {
        use crate::attitude::EulerAngleOrder;

        let traj = small_attitude_trajectory();
        let epoch = traj.start_epoch().unwrap() + 30.0;

        let q = traj.quaternion(epoch).unwrap();

        // euler_angle default: EulerAngle::from_quaternion(quaternion, order)
        let euler = traj.euler_angle(epoch, EulerAngleOrder::ZYX).unwrap();
        let expected_euler = EulerAngle::from_quaternion(q, EulerAngleOrder::ZYX);
        assert_eq!(euler.phi, expected_euler.phi);
        assert_eq!(euler.theta, expected_euler.theta);
        assert_eq!(euler.psi, expected_euler.psi);

        // euler_axis default: ToAttitude::to_euler_axis on the same quaternion
        let axis = traj.euler_axis(epoch).unwrap();
        let expected_axis = q.to_euler_axis();
        assert_eq!(axis.angle, expected_axis.angle);

        // rotation_matrix default: ToAttitude::to_rotation_matrix on the same quaternion
        let r = traj.rotation_matrix(epoch).unwrap();
        let expected_r = q.to_rotation_matrix();
        assert_eq!(r.to_matrix(), expected_r.to_matrix());
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_provider_angular_velocity_none_without_rates() {
        let traj = small_attitude_trajectory();
        let epoch = traj.start_epoch().unwrap();

        // The merged `OrientationProvider` contract reports a provider that
        // carries no rate data as `Ok(None)`; `Err` is reserved for real
        // evaluation failures such as an out-of-coverage epoch.
        assert_eq!(traj.angular_velocity(epoch).unwrap(), None);
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_provider_angular_velocity_with_rates() {
        let (a, b) = spacecraft_frames();
        let mut traj = AttitudeTrajectory::new(a, b);
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let omega = Vector3::new(0.0, 0.0, 0.01);
        traj.add(
            t0,
            AttitudeState::new(z_axis_quaternion(0.0)).with_angular_velocity(omega),
        )
        .unwrap();
        traj.add(
            t0 + 60.0,
            AttitudeState::new(z_axis_quaternion(0.6)).with_angular_velocity(omega),
        )
        .unwrap();

        let result = traj.angular_velocity(t0 + 30.0).unwrap();
        assert_eq!(result, Some(omega));
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_provider_plural_batch_methods() {
        let (a, b) = spacecraft_frames();
        let mut traj = AttitudeTrajectory::new(a, b);
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let omega = Vector3::new(0.0, 0.0, 0.01);
        traj.add(
            t0,
            AttitudeState::new(z_axis_quaternion(0.0)).with_angular_velocity(omega),
        )
        .unwrap();
        traj.add(
            t0 + 60.0,
            AttitudeState::new(z_axis_quaternion(0.6)).with_angular_velocity(omega),
        )
        .unwrap();

        let epochs = vec![t0, t0 + 15.0, t0 + 30.0, t0 + 60.0];

        let quaternions = traj.quaternions(&epochs).unwrap();
        assert_eq!(quaternions.len(), epochs.len());
        for (i, &epoch) in epochs.iter().enumerate() {
            assert_eq!(quaternions[i], traj.quaternion(epoch).unwrap());
        }

        let omegas = traj.angular_velocities(&epochs).unwrap().unwrap();
        assert_eq!(omegas.len(), epochs.len());
        for (i, &epoch) in epochs.iter().enumerate() {
            assert_eq!(Some(omegas[i]), traj.angular_velocity(epoch).unwrap());
        }
    }

    #[test]
    #[serial_test::parallel]
    fn test_quaternion_from_frame_matches_manual_composition() {
        use crate::utils::testing::setup_global_test_eop;
        setup_global_test_eop();

        let mut traj = AttitudeTrajectory::new(
            ReferenceFrame::from(CelestialFrame::GCRF),
            ReferenceFrame::from(BodyFrame::SCBody(None)),
        );
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        traj.add(t0, AttitudeState::new(z_axis_quaternion(0.3)))
            .unwrap();
        traj.add(t0 + 60.0, AttitudeState::new(z_axis_quaternion(0.5)))
            .unwrap();

        let epoch = t0 + 30.0;
        let q = traj
            .quaternion_from_frame(epoch, CelestialFrame::EME2000)
            .unwrap();

        // Manual composition: q_from_to_a * q_a_to_b, with q_from_to_a built
        // from the frame-router rotation EME2000 -> GCRF.
        let r_from_to_a =
            rotation_frame_to_frame(CelestialFrame::EME2000, CelestialFrame::GCRF, epoch).unwrap();
        let q_from_to_a =
            Quaternion::from_rotation_matrix(RotationMatrix::from_matrix(r_from_to_a).unwrap());
        let expected = q_from_to_a * traj.quaternion(epoch).unwrap();

        assert_eq!(q, expected);
    }

    #[test]
    #[serial_test::parallel]
    fn test_quaternion_from_frame_errors_for_body_frame_a() {
        let (a, b) = spacecraft_frames();
        let mut traj = AttitudeTrajectory::new(a, b);
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        traj.add(t0, AttitudeState::new(z_axis_quaternion(0.0)))
            .unwrap();

        let result = traj.quaternion_from_frame(t0, CelestialFrame::EME2000);
        assert!(result.is_err());
        let message = format!("{}", result.unwrap_err());
        assert!(message.contains("Body"));
    }

    #[test]
    #[serial_test::parallel]
    fn test_quaternion_from_frame_errors_for_orbit_relative_frame_a() {
        use crate::frames::{
            OrbitRelativeFrame, OrbitRelativeFrameKind, OrbitRelativeFrameVariant,
        };

        let a = ReferenceFrame::from(
            OrbitRelativeFrame::new(
                OrbitRelativeFrameKind::RTN,
                OrbitRelativeFrameVariant::Rotating,
            )
            .unwrap(),
        );
        let (_, b) = spacecraft_frames();
        let mut traj = AttitudeTrajectory::new(a, b);
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        traj.add(t0, AttitudeState::new(z_axis_quaternion(0.0)))
            .unwrap();

        let result = traj.quaternion_from_frame(t0, CelestialFrame::EME2000);
        assert!(result.is_err());
        let message = format!("{}", result.unwrap_err());
        assert!(message.contains("OrbitRelative"));
    }
}
