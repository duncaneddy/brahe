/*!
 * `OrientationProvider` trait and the orientation providers built on it.
 *
 * [`OrientationProvider`] is the common interface every rotation source in
 * the frame graph implements: a constant attitude, a user callback, or (in
 * later tasks) an interpolated attitude trajectory. The frame registry
 * stores providers behind this trait so the graph layer never needs to know
 * how a given link's rotation is produced.
 *
 * The base attitude types ([`Quaternion`], [`RotationMatrix`],
 * [`EulerAngle`], [`EulerAxis`]) implement [`OrientationProvider`] directly,
 * so a constant attitude can be registered with no wrapper struct.
 * [`CallbackOrientation`] adapts a pair of closures (rotation, and
 * optionally angular velocity) into a provider. [`NumericalRates`] (built
 * via [`OrientationProvider::with_numerical_rates`]) wraps any provider
 * that carries no angular-velocity data and derives it by central
 * differencing the DCM, by explicit opt-in only, matching the crate's
 * existing precedent in `register_custom_frame`.
 */

use nalgebra::Vector3;

use crate::attitude::{
    EulerAngle, EulerAngleOrder, EulerAxis, Quaternion, RotationMatrix, ToAttitude,
};
use crate::math::SMatrix3;
use crate::time::Epoch;
use crate::utils::BraheError;

/// Source of a frame's orientation relative to its parent frame.
///
/// A provider supplies the rotation (as a unit quaternion, A→B passive:
/// `q.to_rotation_matrix() * v_parent = v_frame`) and, optionally, the
/// angular velocity of the frame relative to its parent, expressed in the
/// frame itself.
///
/// # Examples
///
/// ```rust
/// use brahe::attitude::Quaternion;
/// use brahe::frames::OrientationProvider;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
/// let epc = Epoch::from_date(2024, 1, 1, TimeSystem::TAI);
/// assert_eq!(OrientationProvider::quaternion(&q, epc).unwrap(), q);
/// assert!(OrientationProvider::coverage(&q).is_none());
/// ```
pub trait OrientationProvider: Send + Sync {
    /// Unit quaternion rotating parent-frame vectors into this frame at
    /// `epoch`.
    ///
    /// # Arguments
    /// * `epoch` - The epoch to evaluate the orientation at
    ///
    /// # Returns
    /// `Result<Quaternion, BraheError>`: The A→B passive attitude
    /// quaternion, or an error if `epoch` cannot be evaluated (e.g. out of
    /// coverage)
    fn quaternion(&self, epoch: Epoch) -> Result<Quaternion, BraheError>;

    /// Angular velocity of this frame relative to its parent, expressed in
    /// this frame, at `epoch`. Units: (rad/s)
    ///
    /// `Ok(None)` means the provider fundamentally carries no rate data
    /// (e.g. a rotation-only callback); it is not a failure. `Err` is
    /// reserved for real failures, such as `epoch` falling outside the
    /// provider's coverage.
    ///
    /// # Arguments
    /// * `epoch` - The epoch to evaluate the angular velocity at
    ///
    /// # Returns
    /// `Result<Option<Vector3<f64>>, BraheError>`: The angular velocity
    /// (rad/s) if the provider carries rate data, `None` if it does not, or
    /// an error on evaluation failure
    fn angular_velocity(&self, epoch: Epoch) -> Result<Option<Vector3<f64>>, BraheError>;

    /// Time coverage of the provider. `None` means valid for all time (e.g.
    /// a constant rotation).
    ///
    /// # Returns
    /// `Option<(Epoch, Epoch)>`: The `(start, end)` coverage bounds, or
    /// `None` if unbounded
    fn coverage(&self) -> Option<(Epoch, Epoch)> {
        None
    }

    /// Rotation matrix (DCM) rotating parent-frame vectors into this frame
    /// at `epoch`. Default implementation converts [`Self::quaternion`].
    ///
    /// # Arguments
    /// * `epoch` - The epoch to evaluate the orientation at
    ///
    /// # Returns
    /// `Result<RotationMatrix, BraheError>`: The A→B passive rotation
    /// matrix, or an error if `epoch` cannot be evaluated
    fn rotation_matrix(&self, epoch: Epoch) -> Result<RotationMatrix, BraheError> {
        Ok(self.quaternion(epoch)?.to_rotation_matrix())
    }

    /// Euler angles of the parent→frame rotation at `epoch`, in `order`.
    /// Default implementation converts [`Self::quaternion`].
    ///
    /// # Arguments
    /// * `epoch` - The epoch to evaluate the orientation at
    /// * `order` - The Euler angle rotation sequence
    ///
    /// # Returns
    /// `Result<EulerAngle, BraheError>`: The A→B passive attitude as Euler
    /// angles, or an error if `epoch` cannot be evaluated
    fn euler_angle(&self, epoch: Epoch, order: EulerAngleOrder) -> Result<EulerAngle, BraheError> {
        Ok(self.quaternion(epoch)?.to_euler_angle(order))
    }

    /// Euler axis and angle of the parent→frame rotation at `epoch`.
    /// Default implementation converts [`Self::quaternion`].
    ///
    /// # Arguments
    /// * `epoch` - The epoch to evaluate the orientation at
    ///
    /// # Returns
    /// `Result<EulerAxis, BraheError>`: The A→B passive attitude as an Euler
    /// axis and angle, or an error if `epoch` cannot be evaluated
    fn euler_axis(&self, epoch: Epoch) -> Result<EulerAxis, BraheError> {
        Ok(self.quaternion(epoch)?.to_euler_axis())
    }

    /// Wraps this provider so that a missing angular velocity (`Ok(None)`)
    /// is derived numerically by central differencing the rotation matrix,
    /// evaluated over `±step/2`.
    ///
    /// A provider that already reports rates keeps them: the wrapper only
    /// fills the gap left by a rotation-only provider.
    ///
    /// # Arguments
    /// * `step` - Central-difference step, which must be positive and
    ///   finite. Units: (s)
    ///
    /// # Returns
    /// * `Ok(NumericalRates<Self>)`: The wrapped provider
    /// * `Err(BraheError)`: If `step` is not positive and finite
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::{CallbackOrientation, OrientationProvider};
    /// use brahe::math::SMatrix3;
    /// use brahe::time::Epoch;
    ///
    /// let provider = CallbackOrientation::new(|_epc: Epoch| Ok(SMatrix3::identity()), None)
    ///     .with_numerical_rates(1.0)
    ///     .unwrap();
    /// ```
    fn with_numerical_rates(self, step: f64) -> Result<NumericalRates<Self>, BraheError>
    where
        Self: Sized,
    {
        NumericalRates::new(self, step)
    }
}

/// Implements [`OrientationProvider`] for a constant [`ToAttitude`] type:
/// `quaternion` ignores `epoch` and returns `self.to_quaternion()`,
/// `angular_velocity` is always `Ok(Some(Vector3::zeros()))` (a constant
/// rotation has zero rate relative to its parent), and `coverage` defaults
/// to `None` (valid for all time).
macro_rules! impl_orientation_provider_for_attitude {
    ($t:ty) => {
        impl OrientationProvider for $t {
            fn quaternion(&self, _epoch: Epoch) -> Result<Quaternion, BraheError> {
                Ok(self.to_quaternion())
            }

            fn angular_velocity(&self, _epoch: Epoch) -> Result<Option<Vector3<f64>>, BraheError> {
                Ok(Some(Vector3::zeros()))
            }
        }
    };
}

impl_orientation_provider_for_attitude!(Quaternion);
impl_orientation_provider_for_attitude!(EulerAngle);
impl_orientation_provider_for_attitude!(EulerAxis);

impl OrientationProvider for RotationMatrix {
    fn quaternion(&self, _epoch: Epoch) -> Result<Quaternion, BraheError> {
        Ok(self.to_quaternion())
    }

    fn angular_velocity(&self, _epoch: Epoch) -> Result<Option<Vector3<f64>>, BraheError> {
        Ok(Some(Vector3::zeros()))
    }

    // Already a rotation matrix, so this skips the round trip through
    // `Self::quaternion` that the default implementation would otherwise
    // take.
    fn rotation_matrix(&self, _epoch: Epoch) -> Result<RotationMatrix, BraheError> {
        Ok(*self)
    }
}

/// Angular-velocity callback for [`CallbackOrientation`]. Units: (rad/s)
type OmegaCallback = dyn Fn(Epoch) -> Result<Vector3<f64>, BraheError> + Send + Sync;

/// [`OrientationProvider`] built from a rotation callback and an optional
/// angular-velocity callback.
///
/// The rotation callback returns the parent→frame DCM directly; a missing
/// angular-velocity callback makes [`OrientationProvider::angular_velocity`]
/// return `Ok(None)` rather than an error, per [`OrientationProvider`]'s
/// contract. Pair with [`OrientationProvider::with_numerical_rates`] to
/// derive rates numerically when no callback is available.
///
/// # Examples
///
/// ```rust
/// use brahe::frames::CallbackOrientation;
/// use brahe::math::SMatrix3;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let t0 = Epoch::from_date(2024, 1, 1, TimeSystem::TAI);
/// let provider = CallbackOrientation::new(
///     move |epc: Epoch| {
///         let theta = 0.001 * (epc - t0);
///         let (s, c) = theta.sin_cos();
///         Ok(SMatrix3::new(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0))
///     },
///     None,
/// );
/// ```
pub struct CallbackOrientation {
    rotation: Box<dyn Fn(Epoch) -> Result<SMatrix3, BraheError> + Send + Sync>,
    omega: Option<Box<OmegaCallback>>,
}

impl CallbackOrientation {
    /// Constructs a provider from a rotation callback and an optional
    /// angular-velocity callback.
    ///
    /// # Arguments
    /// * `rotation` - Callback returning the parent→frame DCM at an epoch
    /// * `omega` - Optional callback returning the frame's angular velocity
    ///   relative to its parent, expressed in the frame. Units: (rad/s)
    ///
    /// # Returns
    /// `CallbackOrientation`: The constructed provider
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::CallbackOrientation;
    /// use nalgebra::Vector3;
    /// use brahe::math::SMatrix3;
    /// use brahe::time::Epoch;
    ///
    /// let provider = CallbackOrientation::new(
    ///     |_epc: Epoch| Ok(SMatrix3::identity()),
    ///     Some(Box::new(|_epc: Epoch| Ok(Vector3::zeros()))),
    /// );
    /// ```
    pub fn new(
        rotation: impl Fn(Epoch) -> Result<SMatrix3, BraheError> + Send + Sync + 'static,
        omega: Option<Box<OmegaCallback>>,
    ) -> Self {
        Self {
            rotation: Box::new(rotation),
            omega,
        }
    }
}

impl OrientationProvider for CallbackOrientation {
    fn quaternion(&self, epoch: Epoch) -> Result<Quaternion, BraheError> {
        Ok(self.rotation_matrix(epoch)?.to_quaternion())
    }

    fn angular_velocity(&self, epoch: Epoch) -> Result<Option<Vector3<f64>>, BraheError> {
        match &self.omega {
            Some(omega) => Ok(Some(omega(epoch)?)),
            None => Ok(None),
        }
    }

    fn rotation_matrix(&self, epoch: Epoch) -> Result<RotationMatrix, BraheError> {
        RotationMatrix::from_matrix((self.rotation)(epoch)?)
    }
}

/// Wraps an [`OrientationProvider`] that carries no angular-velocity data
/// and derives it by central differencing the provider's rotation matrix.
///
/// Constructed via [`OrientationProvider::with_numerical_rates`]. When
/// the inner provider's [`OrientationProvider::angular_velocity`] returns
/// `Ok(Some(_))`, that value passes through unchanged; otherwise the rate is
/// derived from `[ω]× = -Ṙ Rᵀ`, with `Ṙ ≈ (R(t+h/2) − R(t−h/2))/h` for step
/// `h`, re-expressed in the frame by evaluating `R` at `t`. This is the same
/// derivation `register_custom_frame` uses for its rotation-only fallback.
///
/// # Examples
///
/// ```rust
/// use brahe::frames::{CallbackOrientation, OrientationProvider};
/// use brahe::math::SMatrix3;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let t0 = Epoch::from_date(2024, 1, 1, TimeSystem::TAI);
/// let provider = CallbackOrientation::new(
///     move |epc: Epoch| {
///         let theta = 0.001 * (epc - t0);
///         let (s, c) = theta.sin_cos();
///         Ok(SMatrix3::new(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0))
///     },
///     None,
/// )
/// .with_numerical_rates(1.0)
/// .unwrap();
///
/// let w = provider.angular_velocity(t0 + 3600.0).unwrap().unwrap();
/// assert!((w.z - 0.001).abs() < 1e-9);
/// ```
pub struct NumericalRates<P: OrientationProvider> {
    inner: P,
    step: f64,
}

impl<P: OrientationProvider> NumericalRates<P> {
    /// Wraps `inner`, validating the central-difference step.
    ///
    /// The single validation point behind
    /// [`OrientationProvider::with_numerical_rates`]: a non-positive or
    /// non-finite step would make the difference quotient meaningless, so it
    /// is rejected at construction rather than producing a `NaN` rate at
    /// query time.
    ///
    /// # Arguments
    /// * `inner` - The provider whose missing rates are to be derived
    /// * `step` - Central-difference step, which must be positive and
    ///   finite. Units: (s)
    ///
    /// # Returns
    /// * `Ok(NumericalRates<P>)`: The wrapped provider
    /// * `Err(BraheError)`: If `step` is zero, negative, `NaN`, or infinite
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::attitude::Quaternion;
    /// use brahe::frames::NumericalRates;
    ///
    /// let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
    /// assert!(NumericalRates::new(q, 1.0).is_ok());
    /// assert!(NumericalRates::new(q, 0.0).is_err());
    /// ```
    pub fn new(inner: P, step: f64) -> Result<Self, BraheError> {
        if !(step.is_finite() && step > 0.0) {
            return Err(BraheError::Error(format!(
                "numerical rate step must be a positive, finite number of seconds, got {step}"
            )));
        }
        Ok(Self { inner, step })
    }
}

impl<P: OrientationProvider> OrientationProvider for NumericalRates<P> {
    fn quaternion(&self, epoch: Epoch) -> Result<Quaternion, BraheError> {
        self.inner.quaternion(epoch)
    }

    fn angular_velocity(&self, epoch: Epoch) -> Result<Option<Vector3<f64>>, BraheError> {
        if let Some(w) = self.inner.angular_velocity(epoch)? {
            return Ok(Some(w));
        }

        let half_step = self.step / 2.0;
        let r = self.inner.rotation_matrix(epoch)?.to_matrix();
        let r_plus = self.inner.rotation_matrix(epoch + half_step)?.to_matrix();
        let r_minus = self.inner.rotation_matrix(epoch - half_step)?.to_matrix();
        let r_dot = (r_plus - r_minus) / self.step;

        // For the passive parent->frame DCM `r_frame = R r_parent`, Poisson's
        // kinematic equation reads Ṙ = -[omega]× R, so [omega]× = -Ṙ Rᵀ.
        // Differentiating R Rᵀ = I confirms that -Ṙ Rᵀ is skew-symmetric, so
        // the angular velocity is recovered from its off-diagonal entries.
        // See Markley, F. L. and Crassidis, J. L., "Fundamentals of Spacecraft
        // Attitude Determination and Control", Springer, 2014, attitude
        // kinematics; and Schaub, H. and Junkins, J. L., "Analytical Mechanics
        // of Space Systems", 4th ed., AIAA, 2018, rigid body kinematics.
        let s = -r_dot * r.transpose();
        Ok(Some(Vector3::new(
            0.5 * (s[(2, 1)] - s[(1, 2)]),
            0.5 * (s[(0, 2)] - s[(2, 0)]),
            0.5 * (s[(1, 0)] - s[(0, 1)]),
        )))
    }

    fn coverage(&self) -> Option<(Epoch, Epoch)> {
        self.inner.coverage().map(|(start, end)| {
            let half_step = self.step / 2.0;
            (start + half_step, end - half_step)
        })
    }

    fn rotation_matrix(&self, epoch: Epoch) -> Result<RotationMatrix, BraheError> {
        self.inner.rotation_matrix(epoch)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector3;
    use serial_test::parallel;

    use super::*;
    use crate::attitude::EulerAxis;
    use crate::constants::AngleFormat;
    use crate::time::TimeSystem;

    #[test]
    #[parallel]
    fn test_attitude_types_are_orientation_providers() {
        let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let epc = Epoch::from_date(2024, 1, 1, TimeSystem::TAI);
        assert_eq!(OrientationProvider::quaternion(&q, epc).unwrap(), q);
        assert_eq!(q.angular_velocity(epc).unwrap(), Some(Vector3::zeros()));
        assert!(OrientationProvider::coverage(&q).is_none());
        // RotationMatrix / EulerAngle / EulerAxis round-trip through the trait too
        let r = q.to_rotation_matrix();
        assert_eq!(OrientationProvider::quaternion(&r, epc).unwrap(), q);
    }

    #[test]
    #[parallel]
    fn test_rotation_matrix_provider_returns_self_directly() {
        // RotationMatrix's OrientationProvider::rotation_matrix must return
        // the input matrix directly, bit-identical, rather than round
        // tripping it through quaternion() and back.
        let axis = Vector3::new(1.0, 2.0, 3.0).normalize();
        let ea = EulerAxis::new(axis, 37.0, AngleFormat::Degrees);
        let r = ea.to_rotation_matrix();
        let epc = Epoch::from_date(2024, 1, 1, TimeSystem::TAI);
        let out = OrientationProvider::rotation_matrix(&r, epc).unwrap();
        assert_eq!(out.to_matrix(), r.to_matrix());
    }

    #[test]
    #[parallel]
    fn test_numerical_rates_uniform_spin() {
        // Body spinning about z at 0.001 rad/s: analytic omega recovered to 1e-9
        let rate = 0.001;
        let t0 = Epoch::from_date(2024, 1, 1, TimeSystem::TAI);
        let p = CallbackOrientation::new(
            move |epc: Epoch| {
                let th = rate * (epc - t0);
                Ok(SMatrix3::new(
                    th.cos(),
                    th.sin(),
                    0.0,
                    -th.sin(),
                    th.cos(),
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ))
            },
            None,
        );
        let epc = t0 + 3600.0;
        assert_eq!(p.angular_velocity(epc).unwrap(), None);
        let wrapped = p.with_numerical_rates(1.0).unwrap();
        let w = wrapped.angular_velocity(epc).unwrap().unwrap();
        assert_abs_diff_eq!(w, Vector3::new(0.0, 0.0, rate), epsilon = 1e-9);
    }

    #[test]
    #[parallel]
    fn test_numerical_rates_tilted_axis() {
        // Spin about the fixed axis (1,1,1)/sqrt(3): full 3-axis omega recovered.
        let rate = 0.002;
        let axis = Vector3::new(1.0, 1.0, 1.0).normalize();
        let t0 = Epoch::from_date(2024, 1, 1, TimeSystem::TAI);
        let p = CallbackOrientation::new(
            move |epc: Epoch| {
                let ea = EulerAxis::new(axis, rate * (epc - t0), AngleFormat::Radians);
                Ok(ea.to_rotation_matrix().to_matrix())
            },
            None,
        )
        .with_numerical_rates(1.0)
        .unwrap();
        let w = p.angular_velocity(t0 + 1800.0).unwrap().unwrap();
        assert_abs_diff_eq!(w, axis * rate, epsilon = 1e-9);
    }

    #[test]
    #[parallel]
    fn test_numerical_rates_time_varying_axis() {
        // R(t) = Rz(omega1 t) Rx(theta0) Rz(omega2 t): a 3-1-3 coning
        // sequence whose instantaneous axis, expressed in the frame,
        // precesses at omega1 about z rather than staying fixed, so the
        // frame-expressed omega genuinely differs from its parent-frame
        // expression Rᵀw (checked below).
        let omega1 = 1.0e-3;
        let omega2 = 2.0e-3;
        let theta0 = 0.5;
        let t0 = Epoch::from_date(2024, 1, 1, TimeSystem::TAI);
        let p = CallbackOrientation::new(
            move |epc: Epoch| {
                let t = epc - t0;
                let rz1 = RotationMatrix::Rz(omega1 * t, AngleFormat::Radians).to_matrix();
                let rx = RotationMatrix::Rx(theta0, AngleFormat::Radians).to_matrix();
                let rz2 = RotationMatrix::Rz(omega2 * t, AngleFormat::Radians).to_matrix();
                Ok(rz1 * rx * rz2)
            },
            None,
        )
        .with_numerical_rates(1.0)
        .unwrap();

        let epc = t0 + 1800.0;
        let alpha: f64 = omega1 * (epc - t0);
        let w_expected = Vector3::new(
            omega2 * theta0.sin() * alpha.sin(),
            omega2 * theta0.sin() * alpha.cos(),
            omega1 + omega2 * theta0.cos(),
        );
        let w = p.angular_velocity(epc).unwrap().unwrap();
        assert_abs_diff_eq!(w, w_expected, epsilon = 1e-8);

        // The frame and parent representations of omega genuinely differ
        // here, unlike in the fixed-axis tests above.
        let r = p.rotation_matrix(epc).unwrap().to_matrix();
        assert!((w - r.transpose() * w).norm() > 1e-4);
    }

    #[test]
    #[parallel]
    fn test_callback_orientation_quaternion_matches_rotation_matrix() {
        let t0 = Epoch::from_date(2024, 1, 1, TimeSystem::TAI);
        let rate = 0.001;
        let p = CallbackOrientation::new(
            move |epc: Epoch| {
                let theta = rate * (epc - t0);
                let (s, c) = theta.sin_cos();
                Ok(SMatrix3::new(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0))
            },
            None,
        );
        let epc = t0 + 3600.0;
        let q = p.quaternion(epc).unwrap();
        let r_from_q = q.to_rotation_matrix().to_matrix();
        let r_direct = p.rotation_matrix(epc).unwrap().to_matrix();
        assert_abs_diff_eq!(r_from_q, r_direct, epsilon = 1e-12);
    }

    #[test]
    #[parallel]
    fn test_numerical_rates_quaternion_delegates_to_inner() {
        let t0 = Epoch::from_date(2024, 1, 1, TimeSystem::TAI);
        let p = CallbackOrientation::new(
            move |epc: Epoch| {
                let theta = 0.001 * (epc - t0);
                let (s, c) = theta.sin_cos();
                Ok(SMatrix3::new(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0))
            },
            None,
        )
        .with_numerical_rates(1.0)
        .unwrap();

        let epc = t0 + 1800.0;
        assert_eq!(
            p.quaternion(epc).unwrap(),
            p.rotation_matrix(epc).unwrap().to_quaternion()
        );
    }

    /// Rotation-only provider with a fixed, non-`None` coverage bound, used
    /// to exercise `NumericalRates::coverage`'s half-step contraction.
    struct BoundedProvider {
        start: Epoch,
        end: Epoch,
    }

    impl OrientationProvider for BoundedProvider {
        fn quaternion(&self, _epoch: Epoch) -> Result<Quaternion, BraheError> {
            Ok(Quaternion::new(1.0, 0.0, 0.0, 0.0))
        }

        fn angular_velocity(&self, _epoch: Epoch) -> Result<Option<Vector3<f64>>, BraheError> {
            Ok(None)
        }

        fn coverage(&self) -> Option<(Epoch, Epoch)> {
            Some((self.start, self.end))
        }
    }

    #[test]
    #[parallel]
    fn test_numerical_rates_coverage_contracts_by_half_step() {
        let start = Epoch::from_date(2024, 1, 1, TimeSystem::TAI);
        let end = start + 86400.0;
        let p = BoundedProvider { start, end }
            .with_numerical_rates(2.0)
            .unwrap();
        let (c_start, c_end) = p.coverage().unwrap();
        assert_abs_diff_eq!(c_start - start, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c_end - end, -1.0, epsilon = 1e-12);
    }

    #[test]
    #[parallel]
    fn test_numerical_rates_passes_through_inner_angular_velocity() {
        // When the inner provider already carries rate data, NumericalRates
        // returns it unchanged instead of deriving it numerically.
        let rate = Vector3::new(0.0, 0.0, 5.0e-4);
        let p = CallbackOrientation::new(
            |_epc: Epoch| Ok(SMatrix3::identity()),
            Some(Box::new(move |_epc: Epoch| Ok(rate))),
        )
        .with_numerical_rates(1.0)
        .unwrap();
        let epc = Epoch::from_date(2024, 1, 1, TimeSystem::TAI);
        assert_eq!(p.angular_velocity(epc).unwrap(), Some(rate));
    }

    #[test]
    #[parallel]
    fn test_with_numerical_rates_rejects_invalid_step() {
        // A step that is not positive and finite makes the central-difference
        // quotient meaningless, so it is rejected at construction.
        for step in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let p = CallbackOrientation::new(|_epc: Epoch| Ok(SMatrix3::identity()), None);
            let err = p.with_numerical_rates(step).err().unwrap().to_string();
            assert!(err.contains("positive, finite"));
        }
        // The same validation backs NumericalRates::new directly.
        let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        assert!(NumericalRates::new(q, 0.0).is_err());
        assert!(NumericalRates::new(q, 1.0).is_ok());
    }

    #[test]
    #[parallel]
    fn test_orientation_provider_euler_defaults_match_quaternion() {
        // euler_angle and euler_axis default to converting the provider's
        // quaternion, so they agree with the direct attitude conversions.
        let t0 = Epoch::from_date(2024, 1, 1, TimeSystem::TAI);
        let rate = 0.001;
        let p = CallbackOrientation::new(
            move |epc: Epoch| {
                let theta = rate * (epc - t0);
                let (s, c) = theta.sin_cos();
                Ok(SMatrix3::new(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0))
            },
            None,
        );
        let epc = t0 + 3600.0;
        let q = p.quaternion(epc).unwrap();

        let ea = p.euler_angle(epc, EulerAngleOrder::XYZ).unwrap();
        let expected_ea = q.to_euler_angle(EulerAngleOrder::XYZ);
        assert_abs_diff_eq!(ea.phi, expected_ea.phi, epsilon = 1e-15);
        assert_abs_diff_eq!(ea.theta, expected_ea.theta, epsilon = 1e-15);
        assert_abs_diff_eq!(ea.psi, expected_ea.psi, epsilon = 1e-15);

        let ex = p.euler_axis(epc).unwrap();
        let expected_ex = q.to_euler_axis();
        assert_abs_diff_eq!(ex.angle, expected_ex.angle, epsilon = 1e-15);
        assert_abs_diff_eq!(ex.axis, expected_ex.axis, epsilon = 1e-15);

        // The constant-attitude implementations share the same defaults.
        let axis = Vector3::new(1.0, 2.0, 3.0).normalize();
        let constant = EulerAxis::new(axis, 37.0, AngleFormat::Degrees);
        let from_provider = constant.euler_axis(epc).unwrap();
        assert_abs_diff_eq!(from_provider.axis, axis, epsilon = 1e-14);
        assert_abs_diff_eq!(
            from_provider.angle,
            constant.to_euler_axis().angle,
            epsilon = 1e-14
        );
    }
}
