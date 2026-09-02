/*!
 * Attitude trajectory storage and interpolation.
 *
 * This module provides [`AttitudeState`] (a quaternion plus optional body
 * rate) and [`AttitudeTrajectory`] (a chronologically sorted collection of
 * `AttitudeState` samples relating two [`ReferenceFrame`] endpoints).
 *
 * `AttitudeTrajectory` implements [`Trajectory`] but not
 * `InterpolatableTrajectory`: that trait's default interpolation methods
 * require `StateVector: Mul<f64> + Add`, and `AttitudeState` cannot satisfy
 * those bounds because unit quaternions are not closed under scalar
 * multiplication or addition. `AttitudeTrajectory::interpolate` is instead
 * an inherent method implementing quaternion-aware interpolation (slerp,
 * hemisphere-aligned linear, or hemisphere-aligned Lagrange).
 */

use std::collections::HashMap;

use nalgebra::{Vector3, Vector4};
use serde_json::Value;

use crate::attitude::Quaternion;
use crate::frames::{OrientationProvider, ReferenceFrame};
use crate::math::interpolate_lagrange_svector;
use crate::time::Epoch;
use crate::utils::BraheError;

use super::traits::{Trajectory, TrajectoryEvictionPolicy, compute_lagrange_window};

/// A single attitude sample: a unit quaternion and an optional body rate.
///
/// The quaternion represents the attitude of frame B relative to frame A
/// (see [`AttitudeTrajectory`]). When present, `angular_velocity` is the
/// angular velocity of frame B relative to frame A, expressed in frame B,
/// in rad/s.
#[derive(Debug, Clone, PartialEq)]
pub struct AttitudeState {
    /// Unit quaternion attitude, frame A to frame B.
    pub quaternion: Quaternion,
    /// Angular velocity of frame B relative to frame A, expressed in frame
    /// B. Units: rad/s. `None` if the state does not carry rate data.
    pub angular_velocity: Option<Vector3<f64>>,
}

impl AttitudeState {
    /// Create a new attitude state with no angular velocity.
    ///
    /// # Arguments
    /// * `quaternion` - Unit quaternion attitude, frame A to frame B
    ///
    /// # Returns
    /// A new `AttitudeState` with `angular_velocity` set to `None`
    ///
    /// # Examples
    /// ```rust
    /// use brahe::attitude::Quaternion;
    /// use brahe::trajectories::AttitudeState;
    ///
    /// let state = AttitudeState::new(Quaternion::new(1.0, 0.0, 0.0, 0.0));
    /// assert!(state.angular_velocity.is_none());
    /// ```
    pub fn new(quaternion: Quaternion) -> Self {
        Self {
            quaternion,
            angular_velocity: None,
        }
    }

    /// Attach an angular velocity to the state using builder pattern.
    ///
    /// # Arguments
    /// * `omega` - Angular velocity of frame B relative to frame A, expressed in frame B. Units: rad/s
    ///
    /// # Returns
    /// Self with `angular_velocity` set to `Some(omega)`
    ///
    /// # Examples
    /// ```rust
    /// use brahe::attitude::Quaternion;
    /// use brahe::trajectories::AttitudeState;
    /// use nalgebra::Vector3;
    ///
    /// let state = AttitudeState::new(Quaternion::new(1.0, 0.0, 0.0, 0.0))
    ///     .with_angular_velocity(Vector3::new(0.0, 0.0, 0.01));
    /// assert!(state.angular_velocity.is_some());
    /// ```
    pub fn with_angular_velocity(mut self, omega: Vector3<f64>) -> Self {
        self.angular_velocity = Some(omega);
        self
    }
}

/// Interpolation method for retrieving an [`AttitudeState`] at an arbitrary epoch.
///
/// [`Slerp`](Self::Slerp) is the default because it is the only method here
/// that depends on nothing beyond the two bracketing quaternions: it needs no
/// angular-velocity data, stays on the unit sphere without renormalization,
/// and resolves the double cover of rotations to the shorter arc, so it
/// behaves the same for a trajectory that carries rates and one that does
/// not. Methods that consume the stored rates for a higher-order fit
/// (attitude analogues of Hermite interpolation) would be exact for more
/// motions, but only for a trajectory that carries rates, so they are
/// additive rather than a better default.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum AttitudeInterpolationMethod {
    /// Spherical linear interpolation (slerp) of the bracketing quaternions.
    /// Exact for constant-angular-rate motion and always produces a unit
    /// quaternion. Angular velocity, if present, interpolates linearly.
    #[default]
    Slerp,
    /// Componentwise linear interpolation of the bracketing quaternions
    /// (scalar-first `Vector4`), hemisphere-aligned before lerping and
    /// renormalized afterward. Angular velocity, if present, interpolates
    /// linearly.
    Linear,
    /// Lagrange polynomial interpolation of degree `degree` over a window
    /// of `degree + 1` samples centered on the query epoch, hemisphere-aligned
    /// sequentially (each sample against the previously aligned one) and
    /// renormalized afterward. Angular velocity, if present, uses the same
    /// polynomial degree.
    Lagrange {
        /// Polynomial degree for Lagrange interpolation.
        degree: usize,
    },
}

impl AttitudeInterpolationMethod {
    /// Returns the minimum number of data points required for this interpolation method.
    pub fn min_points_required(&self) -> usize {
        match self {
            Self::Slerp | Self::Linear => 2,
            Self::Lagrange { degree } => degree + 1,
        }
    }
}

/// A chronologically sorted collection of [`AttitudeState`] samples relating
/// two [`ReferenceFrame`] endpoints.
///
/// Every stored quaternion represents the attitude of `frame_b` relative to
/// `frame_a`. All states in a trajectory must uniformly carry angular
/// velocity or uniformly omit it; [`Trajectory::add`] rejects a state that
/// would mix the two.
///
/// # Memory Management
/// The trajectory supports the same eviction policies as other `Trajectory`
/// implementations: a maximum state count or a maximum age relative to the
/// most recent state.
///
/// # Examples
/// ```rust
/// use brahe::attitude::Quaternion;
/// use brahe::frames::{BodyFrame, ReferenceFrame};
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::traits::Trajectory;
/// use brahe::trajectories::{AttitudeState, AttitudeTrajectory};
///
/// let mut traj = AttitudeTrajectory::new(
///     ReferenceFrame::from(BodyFrame::SCBody(None)),
///     ReferenceFrame::from(BodyFrame::SCBody(None)),
/// );
///
/// let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// traj.add(epoch, AttitudeState::new(Quaternion::new(1.0, 0.0, 0.0, 0.0))).unwrap();
/// assert_eq!(traj.len(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct AttitudeTrajectory {
    /// Time epochs for each state, maintained in chronological order.
    pub epochs: Vec<Epoch>,

    /// Attitude states corresponding to epochs.
    pub states: Vec<AttitudeState>,

    /// Frame endpoint A. Stored quaternions represent the attitude of
    /// `frame_b` relative to `frame_a`.
    pub frame_a: ReferenceFrame,

    /// Frame endpoint B. Stored quaternions represent the attitude of
    /// `frame_b` relative to `frame_a`.
    pub frame_b: ReferenceFrame,

    /// Interpolation method for state retrieval at arbitrary epochs.
    /// Default is spherical linear interpolation (slerp).
    pub interpolation_method: AttitudeInterpolationMethod,

    /// Memory management policy for automatic state eviction.
    pub eviction_policy: TrajectoryEvictionPolicy,

    /// Generic metadata storage supporting arbitrary key-value pairs.
    pub metadata: HashMap<String, Value>,

    /// Optional human-readable name for the trajectory (e.g., an object name
    /// carried over from CCSDS interop).
    pub name: Option<String>,

    /// Optional numeric identifier.
    pub id: Option<u64>,

    /// Optional UUID identifier.
    pub uuid: Option<uuid::Uuid>,

    /// Maximum number of states to retain (for `KeepCount` policy).
    max_size: Option<usize>,

    /// Maximum age of states to retain in seconds (for `KeepWithinDuration` policy).
    max_age: Option<f64>,
}

impl AttitudeTrajectory {
    /// Creates a new empty attitude trajectory with default slerp interpolation.
    ///
    /// # Arguments
    /// * `frame_a` - Frame endpoint A
    /// * `frame_b` - Frame endpoint B (stored quaternions rotate from `frame_a` to `frame_b`)
    ///
    /// # Returns
    /// A new empty `AttitudeTrajectory`
    ///
    /// # Examples
    /// ```rust
    /// use brahe::frames::{BodyFrame, ReferenceFrame};
    /// use brahe::traits::Trajectory;
    /// use brahe::trajectories::AttitudeTrajectory;
    ///
    /// let traj = AttitudeTrajectory::new(
    ///     ReferenceFrame::from(BodyFrame::SCBody(None)),
    ///     ReferenceFrame::from(BodyFrame::SCBody(None)),
    /// );
    /// assert_eq!(traj.len(), 0);
    /// ```
    pub fn new(frame_a: ReferenceFrame, frame_b: ReferenceFrame) -> Self {
        Self {
            epochs: Vec::new(),
            states: Vec::new(),
            frame_a,
            frame_b,
            interpolation_method: AttitudeInterpolationMethod::default(),
            eviction_policy: TrajectoryEvictionPolicy::None,
            metadata: HashMap::new(),
            name: None,
            id: None,
            uuid: None,
            max_size: None,
            max_age: None,
        }
    }

    /// Creates an attitude trajectory from vectors of epochs and states.
    ///
    /// Unlike [`Trajectory::from_data`], which cannot express frame identity
    /// because the trait signature carries no frame parameters, this
    /// constructor takes the frame endpoints explicitly and should be
    /// preferred whenever the frames are known.
    ///
    /// # Arguments
    /// * `epochs` - Vector of epochs (must be non-empty and same length as states)
    /// * `states` - Vector of attitude states; all must uniformly carry angular velocity or uniformly omit it
    /// * `frame_a` - Frame endpoint A
    /// * `frame_b` - Frame endpoint B (stored quaternions rotate from `frame_a` to `frame_b`)
    ///
    /// # Returns
    /// * `Ok(Self)` - Trajectory successfully created with sorted data
    /// * `Err(BraheError)` - If validation fails (length mismatch, empty vectors, or mixed angular velocity presence)
    ///
    /// # Examples
    /// ```rust
    /// use brahe::attitude::Quaternion;
    /// use brahe::frames::{BodyFrame, ReferenceFrame};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::trajectories::{AttitudeState, AttitudeTrajectory};
    ///
    /// let epochs = vec![
    ///     Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC),
    ///     Epoch::from_datetime(2023, 1, 1, 12, 1, 0.0, 0.0, TimeSystem::UTC),
    /// ];
    /// let states = vec![
    ///     AttitudeState::new(Quaternion::new(1.0, 0.0, 0.0, 0.0)),
    ///     AttitudeState::new(Quaternion::new(0.9997, 0.0, 0.0, 0.0245)),
    /// ];
    ///
    /// let traj = AttitudeTrajectory::from_data(
    ///     epochs,
    ///     states,
    ///     ReferenceFrame::from(BodyFrame::SCBody(None)),
    ///     ReferenceFrame::from(BodyFrame::SCBody(None)),
    /// ).unwrap();
    /// ```
    pub fn from_data(
        epochs: Vec<Epoch>,
        states: Vec<AttitudeState>,
        frame_a: ReferenceFrame,
        frame_b: ReferenceFrame,
    ) -> Result<Self, BraheError> {
        if epochs.len() != states.len() {
            return Err(BraheError::Error(
                "Epochs and states vectors must have the same length".to_string(),
            ));
        }

        if epochs.is_empty() {
            return Err(BraheError::Error(
                "Cannot create trajectory from empty data".to_string(),
            ));
        }

        validate_rate_uniformity(&states)?;

        // Ensure epochs are sorted. `sort_by` is stable, so states sharing an
        // epoch keep their input order and the last one given remains the
        // most recently added state at that discontinuity.
        let mut indices: Vec<usize> = (0..epochs.len()).collect();
        indices.sort_by(|&i, &j| epochs[i].partial_cmp(&epochs[j]).unwrap());

        let sorted_epochs: Vec<Epoch> = indices.iter().map(|&i| epochs[i]).collect();
        let sorted_states: Vec<AttitudeState> =
            indices.iter().map(|&i| states[i].clone()).collect();

        Ok(Self {
            epochs: sorted_epochs,
            states: sorted_states,
            frame_a,
            frame_b,
            interpolation_method: AttitudeInterpolationMethod::default(),
            eviction_policy: TrajectoryEvictionPolicy::None,
            metadata: HashMap::new(),
            name: None,
            id: None,
            uuid: None,
            max_size: None,
            max_age: None,
        })
    }

    /// Sets the interpolation method using builder pattern.
    ///
    /// # Arguments
    /// * `method` - Method to use for state interpolation between epochs
    ///
    /// # Returns
    /// Self with updated interpolation method
    ///
    /// # Examples
    /// ```rust
    /// use brahe::frames::{BodyFrame, ReferenceFrame};
    /// use brahe::trajectories::{AttitudeInterpolationMethod, AttitudeTrajectory};
    ///
    /// let traj = AttitudeTrajectory::new(
    ///     ReferenceFrame::from(BodyFrame::SCBody(None)),
    ///     ReferenceFrame::from(BodyFrame::SCBody(None)),
    /// )
    /// .with_interpolation_method(AttitudeInterpolationMethod::Linear);
    /// ```
    pub fn with_interpolation_method(mut self, method: AttitudeInterpolationMethod) -> Self {
        self.interpolation_method = method;
        self
    }

    /// Sets the interpolation method in place.
    ///
    /// # Arguments
    /// * `method` - Method to use for state interpolation between epochs
    pub fn set_interpolation_method(&mut self, method: AttitudeInterpolationMethod) {
        self.interpolation_method = method;
    }

    /// Sets the eviction policy to keep at most `max_size` states, using the
    /// builder pattern.
    ///
    /// # Arguments
    /// * `max_size` - Maximum number of states to retain (must be > 0)
    ///
    /// # Returns
    /// * `Ok(Self)` - Trajectory with updated eviction policy
    /// * `Err(BraheError)` - If `max_size` is zero
    ///
    /// # Examples
    /// ```rust
    /// use brahe::frames::{BodyFrame, ReferenceFrame};
    /// use brahe::trajectories::AttitudeTrajectory;
    ///
    /// let traj = AttitudeTrajectory::new(
    ///     ReferenceFrame::from(BodyFrame::SCBody(None)),
    ///     ReferenceFrame::from(BodyFrame::SCBody(None)),
    /// )
    /// .with_eviction_policy_max_size(100)
    /// .unwrap();
    /// ```
    pub fn with_eviction_policy_max_size(mut self, max_size: usize) -> Result<Self, BraheError> {
        self.set_eviction_policy_max_size(max_size)?;
        Ok(self)
    }

    /// Sets the eviction policy to keep states within a maximum age, using the
    /// builder pattern.
    ///
    /// # Arguments
    /// * `max_age` - Maximum age of states to retain. Units: (s)
    ///
    /// # Returns
    /// * `Ok(Self)` - Trajectory with updated eviction policy
    /// * `Err(BraheError)` - If `max_age` is not positive
    ///
    /// # Examples
    /// ```rust
    /// use brahe::frames::{BodyFrame, ReferenceFrame};
    /// use brahe::trajectories::AttitudeTrajectory;
    ///
    /// let traj = AttitudeTrajectory::new(
    ///     ReferenceFrame::from(BodyFrame::SCBody(None)),
    ///     ReferenceFrame::from(BodyFrame::SCBody(None)),
    /// )
    /// .with_eviction_policy_max_age(3600.0)
    /// .unwrap();
    /// ```
    pub fn with_eviction_policy_max_age(mut self, max_age: f64) -> Result<Self, BraheError> {
        self.set_eviction_policy_max_age(max_age)?;
        Ok(self)
    }

    /// Returns true if the trajectory is non-empty and its states carry angular velocity.
    ///
    /// # Returns
    /// `true` if the trajectory has at least one state and that state (and,
    /// by the rate-uniformity invariant enforced by `add`, every state) has
    /// `angular_velocity` set. `false` for an empty trajectory or one whose
    /// states omit angular velocity.
    ///
    /// # Examples
    /// ```rust
    /// use brahe::attitude::Quaternion;
    /// use brahe::frames::{BodyFrame, ReferenceFrame};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::traits::Trajectory;
    /// use brahe::trajectories::{AttitudeState, AttitudeTrajectory};
    ///
    /// let mut traj = AttitudeTrajectory::new(
    ///     ReferenceFrame::from(BodyFrame::SCBody(None)),
    ///     ReferenceFrame::from(BodyFrame::SCBody(None)),
    /// );
    /// assert!(!traj.has_rates());
    ///
    /// let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// traj.add(epoch, AttitudeState::new(Quaternion::new(1.0, 0.0, 0.0, 0.0))).unwrap();
    /// assert!(!traj.has_rates());
    /// ```
    pub fn has_rates(&self) -> bool {
        self.states
            .first()
            .map(|s| s.angular_velocity.is_some())
            .unwrap_or(false)
    }

    /// Apply the eviction policy to manage trajectory memory.
    fn apply_eviction_policy(&mut self) {
        match self.eviction_policy {
            TrajectoryEvictionPolicy::None => {
                // No eviction
            }
            TrajectoryEvictionPolicy::KeepCount => {
                if let Some(max_size) = self.max_size
                    && self.epochs.len() > max_size
                {
                    let to_remove = self.epochs.len() - max_size;
                    self.epochs.drain(0..to_remove);
                    self.states.drain(0..to_remove);
                }
            }
            TrajectoryEvictionPolicy::KeepWithinDuration => {
                if let Some(max_age) = self.max_age
                    && let Some(&last_epoch) = self.epochs.last()
                {
                    let mut indices_to_keep = Vec::new();
                    for (i, &epoch) in self.epochs.iter().enumerate() {
                        if (last_epoch - epoch).abs() <= max_age {
                            indices_to_keep.push(i);
                        }
                    }

                    let new_epochs: Vec<Epoch> =
                        indices_to_keep.iter().map(|&i| self.epochs[i]).collect();
                    let new_states: Vec<AttitudeState> = indices_to_keep
                        .iter()
                        .map(|&i| self.states[i].clone())
                        .collect();

                    self.epochs = new_epochs;
                    self.states = new_states;
                }
            }
        }
    }

    /// Interpolate the attitude state at a given epoch using the configured interpolation method.
    ///
    /// An epoch matching a stored node exactly returns that node's state
    /// directly (no interpolation error even if the trajectory has fewer
    /// points than the configured method requires). Otherwise the epoch must
    /// lie within `[start_epoch, end_epoch]`.
    ///
    /// # Arguments
    /// * `epoch` - Target epoch for interpolation
    ///
    /// # Returns
    /// * `Ok(state)` - Interpolated attitude state
    /// * `Err(BraheError)` - If the epoch is out of range, the trajectory has too few points for
    ///   the configured method, or the configured method is `Lagrange { degree: 0 }` (Lagrange
    ///   interpolation requires a degree of at least 1)
    ///
    /// # Examples
    /// ```rust
    /// use brahe::attitude::Quaternion;
    /// use brahe::frames::{BodyFrame, ReferenceFrame};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::traits::Trajectory;
    /// use brahe::trajectories::{AttitudeState, AttitudeTrajectory};
    ///
    /// let mut traj = AttitudeTrajectory::new(
    ///     ReferenceFrame::from(BodyFrame::SCBody(None)),
    ///     ReferenceFrame::from(BodyFrame::SCBody(None)),
    /// );
    /// let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// traj.add(t0, AttitudeState::new(Quaternion::new(1.0, 0.0, 0.0, 0.0))).unwrap();
    /// traj.add(t0 + 60.0, AttitudeState::new(Quaternion::new(0.0, 1.0, 0.0, 0.0))).unwrap();
    ///
    /// let state = traj.interpolate(&(t0 + 30.0)).unwrap();
    /// ```
    pub fn interpolate(&self, epoch: &Epoch) -> Result<AttitudeState, BraheError> {
        // Explicit bounds checking, mirroring STrajectory's boundary semantics
        if let Some(start) = self.start_epoch()
            && *epoch < start
        {
            return Err(BraheError::OutOfBoundsError(format!(
                "Cannot interpolate: epoch {} is before trajectory start {}",
                epoch, start
            )));
        }

        if let Some(end) = self.end_epoch()
            && *epoch > end
        {
            return Err(BraheError::OutOfBoundsError(format!(
                "Cannot interpolate: epoch {} is after trajectory end {}",
                epoch, end
            )));
        }

        let idx1 = self.index_before_epoch(epoch)?;
        let idx2 = self.index_after_epoch(epoch)?;

        // Exact match against a stored node. `add` inserts a state after
        // any it already holds at that epoch, and for a repeated epoch
        // `index_before_epoch` lands on the last of the run while
        // `index_after_epoch` lands on the first, so the higher index is the
        // most recently added state. Returning it makes a query at a
        // discontinuity right-continuous with the states that follow.
        if idx1 == idx2 || self.epochs[idx1] == self.epochs[idx2] {
            return self.state_at_idx(idx1.max(idx2));
        }

        let method = self.interpolation_method;
        if let AttitudeInterpolationMethod::Lagrange { degree } = method
            && degree == 0
        {
            return Err(BraheError::Error(
                "Lagrange interpolation requires a degree >= 1, got 0".to_string(),
            ));
        }

        // Validate minimum point count for the interpolation method
        let required = method.min_points_required();
        if self.len() < required {
            return Err(BraheError::Error(format!(
                "{:?} requires {} points, trajectory has {}",
                method,
                required,
                self.len()
            )));
        }

        match method {
            AttitudeInterpolationMethod::Slerp => {
                let (epoch1, state1) = self.get(idx1)?;
                let (_epoch2, state2) = self.get(idx2)?;
                let t = (*epoch - epoch1) / (self.epochs[idx2] - epoch1);

                let quaternion = state1.quaternion.slerp(state2.quaternion, t);
                let angular_velocity =
                    lerp_angular_velocity(state1.angular_velocity, state2.angular_velocity, t);

                Ok(AttitudeState {
                    quaternion,
                    angular_velocity,
                })
            }

            AttitudeInterpolationMethod::Linear => {
                let (epoch1, state1) = self.get(idx1)?;
                let (_epoch2, state2) = self.get(idx2)?;
                let t = (*epoch - epoch1) / (self.epochs[idx2] - epoch1);

                // Unit quaternions are a double cover of rotations: q and -q
                // represent the same attitude. Lerping between two stored
                // representatives that happen to sit in opposite hemispheres
                // of that cover would interpolate through the short way
                // around the wrong great circle (or through the origin), so
                // the later sample is sign-aligned to the earlier one first.
                let v1 = state1.quaternion.to_vector(true);
                let mut v2 = state2.quaternion.to_vector(true);
                if v1.dot(&v2) < 0.0 {
                    v2 = -v2;
                }
                let v = v1 * (1.0 - t) + v2 * t;
                let quaternion = Quaternion::new(v[0], v[1], v[2], v[3]);

                let angular_velocity =
                    lerp_angular_velocity(state1.angular_velocity, state2.angular_velocity, t);

                Ok(AttitudeState {
                    quaternion,
                    angular_velocity,
                })
            }

            AttitudeInterpolationMethod::Lagrange { degree } => {
                let n_points = degree + 1;
                let (start_idx, end_idx) =
                    compute_lagrange_window(&self.epochs, idx1, idx2, n_points)?;

                let ref_epoch = self.start_epoch().unwrap();
                let times: Vec<f64> = (start_idx..=end_idx)
                    .map(|i| self.epochs[i] - ref_epoch)
                    .collect();

                // Hemisphere-align each sample against the *previously
                // aligned* sample rather than a single fixed reference: over
                // a wide window the total rotation can exceed pi, so a
                // sample far from the window's first element may have a
                // negative dot product with it even though it is on the
                // same side as its immediate neighbor. Sequential alignment
                // only relies on adjacent samples staying within 90 degrees
                // of each other in 4-space, which holds for any window fine
                // enough to interpolate meaningfully.
                let mut quaternion_values: Vec<Vector4<f64>> =
                    Vec::with_capacity(end_idx - start_idx + 1);
                let mut previous = self.states[start_idx].quaternion.to_vector(true);
                quaternion_values.push(previous);
                for i in (start_idx + 1)..=end_idx {
                    let mut v = self.states[i].quaternion.to_vector(true);
                    if v.dot(&previous) < 0.0 {
                        v = -v;
                    }
                    quaternion_values.push(v);
                    previous = v;
                }

                let t = *epoch - ref_epoch;
                let q_vec = interpolate_lagrange_svector(&times, &quaternion_values, t);
                let quaternion = Quaternion::new(q_vec[0], q_vec[1], q_vec[2], q_vec[3]);

                let angular_velocity = if self.has_rates() {
                    let mut omega_values: Vec<Vector3<f64>> =
                        Vec::with_capacity(end_idx - start_idx + 1);
                    for i in start_idx..=end_idx {
                        omega_values.push(self.states[i].angular_velocity.ok_or_else(|| {
                            BraheError::Error(
                                "AttitudeTrajectory::has_rates() reported true but a state in \
                                 the interpolation window has no angular_velocity; this \
                                 indicates a rate-uniformity invariant violation"
                                    .to_string(),
                            )
                        })?);
                    }
                    Some(interpolate_lagrange_svector(&times, &omega_values, t))
                } else {
                    None
                };

                Ok(AttitudeState {
                    quaternion,
                    angular_velocity,
                })
            }
        }
    }
}

/// Linearly interpolate an optional angular velocity pair.
///
/// Returns `None` unless both endpoints carry a rate (the rate-uniformity
/// invariant enforced by `add`/`from_data` means both will always agree).
fn lerp_angular_velocity(
    omega1: Option<Vector3<f64>>,
    omega2: Option<Vector3<f64>>,
    t: f64,
) -> Option<Vector3<f64>> {
    match (omega1, omega2) {
        (Some(w1), Some(w2)) => Some(w1 * (1.0 - t) + w2 * t),
        _ => None,
    }
}

/// Validate that every state in `states` uniformly carries angular velocity or uniformly omits it.
fn validate_rate_uniformity(states: &[AttitudeState]) -> Result<(), BraheError> {
    if let Some(first) = states.first() {
        let expects_rates = first.angular_velocity.is_some();
        for state in states.iter().skip(1) {
            if state.angular_velocity.is_some() != expects_rates {
                return Err(BraheError::Error(format!(
                    "AttitudeTrajectory states must uniformly {} angular velocity; found a state \
                     that does {}",
                    if expects_rates { "carry" } else { "omit" },
                    if expects_rates {
                        "not carry it"
                    } else {
                        "carry it"
                    }
                )));
            }
        }
    }
    Ok(())
}

impl Trajectory for AttitudeTrajectory {
    type StateVector = AttitudeState;

    /// Always returns an error: the `Trajectory` trait signature carries
    /// no frame information, and `AttitudeTrajectory` has no meaningful
    /// default for `frame_a`/`frame_b` (unlike, e.g., `SOrbitTrajectory`'s
    /// ECI default, there is no frame convention CCSDS or brahe assumes for
    /// an attitude). Fabricating a placeholder frame here would silently
    /// produce trajectories with wrong frame identity (e.g. writing out as
    /// `REF_FRAME_A/B = SC_BODY`). Use
    /// [`AttitudeTrajectory::from_data`] directly, which takes the frame
    /// endpoints explicitly.
    fn from_data(_epochs: Vec<Epoch>, _states: Vec<Self::StateVector>) -> Result<Self, BraheError> {
        Err(BraheError::Error(
            "AttitudeTrajectory cannot be constructed via Trajectory::from_data because it \
             carries no frame information; use \
             AttitudeTrajectory::from_data(epochs, states, frame_a, frame_b) instead"
                .to_string(),
        ))
    }

    /// Inserts `state` at `epoch` in chronological order.
    ///
    /// A repeated epoch is allowed, and represents a discontinuity: an
    /// impulsive attitude maneuver holds its pre- and post-maneuver states
    /// at the same instant. The new state is inserted after any already
    /// stored at `epoch`, so the most recently added is the one
    /// [`AttitudeTrajectory::interpolate`] returns for a query there.
    fn add(&mut self, epoch: Epoch, state: Self::StateVector) -> Result<(), BraheError> {
        if let Some(existing) = self.states.first() {
            let existing_has_rate = existing.angular_velocity.is_some();
            let new_has_rate = state.angular_velocity.is_some();
            if existing_has_rate != new_has_rate {
                return Err(BraheError::Error(format!(
                    "Cannot add state at epoch {}: trajectory states {} angular velocity, but \
                     the new state {}",
                    epoch,
                    if existing_has_rate {
                        "carry"
                    } else {
                        "do not carry"
                    },
                    if new_has_rate {
                        "carries angular velocity"
                    } else {
                        "does not carry angular velocity"
                    },
                )));
            }
        }

        // Find the correct position to insert based on epoch. A strict `<`
        // comparison places the new state after any already stored at the
        // same epoch, keeping the most recently added last in a repeated run.
        let mut insert_idx = self.epochs.len();
        for (i, existing_epoch) in self.epochs.iter().enumerate() {
            if epoch < *existing_epoch {
                insert_idx = i;
                break;
            }
        }

        self.epochs.insert(insert_idx, epoch);
        self.states.insert(insert_idx, state);

        self.apply_eviction_policy();

        Ok(())
    }

    fn epoch_at_idx(&self, index: usize) -> Result<Epoch, BraheError> {
        if index >= self.epochs.len() {
            return Err(BraheError::Error(format!(
                "Index {} out of bounds for trajectory with {} epochs",
                index,
                self.epochs.len()
            )));
        }

        Ok(self.epochs[index])
    }

    fn state_at_idx(&self, index: usize) -> Result<Self::StateVector, BraheError> {
        if index >= self.states.len() {
            return Err(BraheError::Error(format!(
                "Index {} out of bounds for trajectory with {} states",
                index,
                self.states.len()
            )));
        }

        Ok(self.states[index].clone())
    }

    fn nearest_state(&self, epoch: &Epoch) -> Result<(Epoch, Self::StateVector), BraheError> {
        if self.epochs.is_empty() {
            return Err(BraheError::Error(
                "Cannot find nearest state in empty trajectory".to_string(),
            ));
        }

        let mut nearest_idx = 0;
        let mut min_diff = f64::MAX;

        for (i, existing_epoch) in self.epochs.iter().enumerate() {
            let diff = (*epoch - *existing_epoch).abs();
            if diff < min_diff {
                min_diff = diff;
                nearest_idx = i;
            }

            if i > 0 && existing_epoch > epoch && diff > min_diff {
                break;
            }
        }

        Ok((self.epochs[nearest_idx], self.states[nearest_idx].clone()))
    }

    fn len(&self) -> usize {
        self.states.len()
    }

    fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    fn start_epoch(&self) -> Option<Epoch> {
        self.epochs.first().copied()
    }

    fn end_epoch(&self) -> Option<Epoch> {
        self.epochs.last().copied()
    }

    fn timespan(&self) -> Option<f64> {
        if self.epochs.len() < 2 {
            None
        } else {
            Some(*self.epochs.last().unwrap() - *self.epochs.first().unwrap())
        }
    }

    fn first(&self) -> Option<(Epoch, Self::StateVector)> {
        if self.epochs.is_empty() {
            None
        } else {
            Some((self.epochs[0], self.states[0].clone()))
        }
    }

    fn last(&self) -> Option<(Epoch, Self::StateVector)> {
        if self.epochs.is_empty() {
            None
        } else {
            let last_index = self.epochs.len() - 1;
            Some((self.epochs[last_index], self.states[last_index].clone()))
        }
    }

    fn clear(&mut self) {
        self.epochs.clear();
        self.states.clear();
    }

    fn remove_epoch(&mut self, epoch: &Epoch) -> Result<Self::StateVector, BraheError> {
        if let Some(index) = self.epochs.iter().position(|e| e == epoch) {
            let removed_state = self.states.remove(index);
            self.epochs.remove(index);
            Ok(removed_state)
        } else {
            Err(BraheError::Error(
                "Epoch not found in trajectory".to_string(),
            ))
        }
    }

    fn remove(&mut self, index: usize) -> Result<(Epoch, Self::StateVector), BraheError> {
        if index >= self.states.len() {
            return Err(BraheError::Error(format!(
                "Index {} out of bounds for trajectory with {} states",
                index,
                self.states.len()
            )));
        }

        let removed_epoch = self.epochs.remove(index);
        let removed_state = self.states.remove(index);
        Ok((removed_epoch, removed_state))
    }

    fn get(&self, index: usize) -> Result<(Epoch, Self::StateVector), BraheError> {
        if index >= self.states.len() {
            return Err(BraheError::Error(format!(
                "Index {} out of bounds for trajectory with {} states",
                index,
                self.states.len()
            )));
        }

        Ok((self.epochs[index], self.states[index].clone()))
    }

    fn index_before_epoch(&self, epoch: &Epoch) -> Result<usize, BraheError> {
        if self.epochs.is_empty() {
            return Err(BraheError::Error(
                "Cannot get index from empty trajectory".to_string(),
            ));
        }

        if epoch < &self.epochs[0] {
            return Err(BraheError::Error(
                "Epoch is before all states in trajectory".to_string(),
            ));
        }

        for i in (0..self.epochs.len()).rev() {
            if &self.epochs[i] <= epoch {
                return Ok(i);
            }
        }

        Err(BraheError::Error(
            "Failed to find index before epoch".to_string(),
        ))
    }

    fn index_after_epoch(&self, epoch: &Epoch) -> Result<usize, BraheError> {
        if self.epochs.is_empty() {
            return Err(BraheError::Error(
                "Cannot get index from empty trajectory".to_string(),
            ));
        }

        if epoch > self.epochs.last().unwrap() {
            return Err(BraheError::Error(
                "Epoch is after all states in trajectory".to_string(),
            ));
        }

        for i in 0..self.epochs.len() {
            if &self.epochs[i] >= epoch {
                return Ok(i);
            }
        }

        Err(BraheError::Error(
            "Failed to find index after epoch".to_string(),
        ))
    }

    fn set_eviction_policy_max_size(&mut self, max_size: usize) -> Result<(), BraheError> {
        if max_size < 1 {
            return Err(BraheError::Error("Maximum size must be >= 1".to_string()));
        }
        self.eviction_policy = TrajectoryEvictionPolicy::KeepCount;
        self.max_size = Some(max_size);
        self.max_age = None;
        self.apply_eviction_policy();
        Ok(())
    }

    fn set_eviction_policy_max_age(&mut self, max_age: f64) -> Result<(), BraheError> {
        if max_age <= 0.0 {
            return Err(BraheError::Error("Maximum age must be > 0.0".to_string()));
        }
        self.eviction_policy = TrajectoryEvictionPolicy::KeepWithinDuration;
        self.max_age = Some(max_age);
        self.max_size = None;
        self.apply_eviction_policy();
        Ok(())
    }

    fn get_eviction_policy(&self) -> TrajectoryEvictionPolicy {
        self.eviction_policy
    }
}

impl AttitudeTrajectory {
    /// Interpolated attitude quaternions at each epoch in `epochs`.
    ///
    /// # Arguments
    ///
    /// * `epochs` - Epochs to evaluate the attitude at
    ///
    /// # Returns
    ///
    /// * `Result<Vec<Quaternion>, BraheError>` - The interpolated quaternion
    ///   at each epoch, or an error if any epoch lies outside the trajectory
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::attitude::Quaternion;
    /// use brahe::frames::{BodyFrame, ReferenceFrame};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::trajectories::{AttitudeState, AttitudeTrajectory};
    /// use brahe::traits::Trajectory;
    ///
    /// let mut traj = AttitudeTrajectory::new(
    ///     ReferenceFrame::from(BodyFrame::SCBody(None)),
    ///     ReferenceFrame::from(BodyFrame::SCBody(None)),
    /// );
    /// let epc = Epoch::from_date(2024, 1, 1, TimeSystem::UTC);
    /// traj.add(epc, AttitudeState::new(Quaternion::new(1.0, 0.0, 0.0, 0.0))).unwrap();
    ///
    /// let quaternions = traj.quaternions(&[epc]).unwrap();
    /// assert_eq!(quaternions.len(), 1);
    /// ```
    pub fn quaternions(&self, epochs: &[Epoch]) -> Result<Vec<Quaternion>, BraheError> {
        epochs
            .iter()
            .map(|&epoch| OrientationProvider::quaternion(self, epoch))
            .collect()
    }

    /// Interpolated body angular velocities at each epoch in `epochs`.
    ///
    /// Rate data is uniform across a trajectory (see
    /// [`AttitudeTrajectory::has_rates`]), so this returns `Ok(None)` for a
    /// trajectory that carries no rates rather than a per-epoch mixture.
    ///
    /// # Arguments
    ///
    /// * `epochs` - Epochs to evaluate the angular velocity at
    ///
    /// # Returns
    ///
    /// * `Result<Option<Vec<Vector3<f64>>>, BraheError>` - The interpolated
    ///   angular velocity (rad/s) at each epoch, `None` if the trajectory
    ///   carries no rate data, or an error if any epoch lies outside the
    ///   trajectory
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::attitude::Quaternion;
    /// use brahe::frames::{BodyFrame, ReferenceFrame};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::trajectories::{AttitudeState, AttitudeTrajectory};
    /// use brahe::traits::Trajectory;
    ///
    /// let mut traj = AttitudeTrajectory::new(
    ///     ReferenceFrame::from(BodyFrame::SCBody(None)),
    ///     ReferenceFrame::from(BodyFrame::SCBody(None)),
    /// );
    /// let epc = Epoch::from_date(2024, 1, 1, TimeSystem::UTC);
    /// traj.add(epc, AttitudeState::new(Quaternion::new(1.0, 0.0, 0.0, 0.0))).unwrap();
    ///
    /// assert!(traj.angular_velocities(&[epc]).unwrap().is_none());
    /// ```
    pub fn angular_velocities(
        &self,
        epochs: &[Epoch],
    ) -> Result<Option<Vec<Vector3<f64>>>, BraheError> {
        if !self.has_rates() {
            // `Ok(None)` reports the absence of rate data, not the absence
            // of a failure; every epoch must still be validated against
            // coverage (and an empty trajectory must still error) so the
            // `OrientationProvider` contract's `Err`-for-out-of-coverage
            // guarantee holds regardless of whether rates are present.
            for &epoch in epochs {
                self.interpolate(&epoch)?;
            }
            return Ok(None);
        }
        let mut rates = Vec::with_capacity(epochs.len());
        for &epoch in epochs {
            match OrientationProvider::angular_velocity(self, epoch)? {
                Some(omega) => rates.push(omega),
                None => return Ok(None),
            }
        }
        Ok(Some(rates))
    }
}

impl OrientationProvider for AttitudeTrajectory {
    /// Interpolated attitude quaternion at `epoch`; see
    /// [`AttitudeTrajectory::interpolate`] for the interpolation-method
    /// semantics.
    fn quaternion(&self, epoch: Epoch) -> Result<Quaternion, BraheError> {
        Ok(self.interpolate(&epoch)?.quaternion)
    }

    /// Interpolated body angular velocity at `epoch`, or `Ok(None)` when
    /// [`AttitudeTrajectory::has_rates`] is `false`. `epoch` is still
    /// validated against the trajectory's coverage in the rate-less case,
    /// so an out-of-coverage query returns `Err` rather than `Ok(None)`.
    ///
    /// brahe never silently finite-differences a quaternion history to
    /// approximate a rate. A trajectory without rate data reports `None`
    /// here; callers that want a derived rate opt in explicitly through
    /// [`OrientationProvider::with_numerical_rates`].
    fn angular_velocity(&self, epoch: Epoch) -> Result<Option<Vector3<f64>>, BraheError> {
        if !self.has_rates() {
            // `Ok(None)` reports the absence of rate data, not the absence
            // of a failure; validate `epoch` against coverage (an empty
            // trajectory included) before reporting the trajectory
            // rate-less, so the `OrientationProvider` contract's
            // `Err`-for-out-of-coverage guarantee holds regardless of
            // whether rates are present.
            self.interpolate(&epoch)?;
            return Ok(None);
        }
        match self.interpolate(&epoch)?.angular_velocity {
            Some(omega) => Ok(Some(omega)),
            None => Err(BraheError::Error(
                "AttitudeTrajectory::has_rates() reported true but the interpolated state has \
                 no angular_velocity; this indicates a rate-uniformity invariant violation"
                    .to_string(),
            )),
        }
    }

    /// Coverage bounds of the trajectory, or `None` when it holds no states.
    fn coverage(&self) -> Option<(Epoch, Epoch)> {
        match (self.start_epoch(), self.end_epoch()) {
            (Some(start), Some(end)) => Some((start, end)),
            _ => None,
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::frames::BodyFrame;
    use crate::time::TimeSystem;
    use approx::assert_abs_diff_eq;

    fn body_frames() -> (ReferenceFrame, ReferenceFrame) {
        (
            ReferenceFrame::from(BodyFrame::SCBody(None)),
            ReferenceFrame::from(BodyFrame::SCBody(None)),
        )
    }

    /// Quaternion for a rotation of `theta` radians about the z-axis.
    fn z_axis_quaternion(theta: f64) -> Quaternion {
        Quaternion::new((theta / 2.0).cos(), 0.0, 0.0, (theta / 2.0).sin())
    }

    // =========================================================================
    // AttitudeState tests
    // =========================================================================

    #[test]
    #[serial_test::parallel]
    fn test_attitude_state_new() {
        let state = AttitudeState::new(Quaternion::new(1.0, 0.0, 0.0, 0.0));
        assert_eq!(state.quaternion, Quaternion::new(1.0, 0.0, 0.0, 0.0));
        assert!(state.angular_velocity.is_none());
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_state_with_angular_velocity() {
        let state = AttitudeState::new(Quaternion::new(1.0, 0.0, 0.0, 0.0))
            .with_angular_velocity(Vector3::new(0.1, 0.2, 0.3));
        assert_eq!(state.angular_velocity, Some(Vector3::new(0.1, 0.2, 0.3)));
    }

    // =========================================================================
    // AttitudeInterpolationMethod tests
    // =========================================================================

    #[test]
    #[serial_test::parallel]
    fn test_attitude_interpolation_method_default() {
        assert_eq!(
            AttitudeInterpolationMethod::default(),
            AttitudeInterpolationMethod::Slerp
        );
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_interpolation_method_min_points_required() {
        assert_eq!(AttitudeInterpolationMethod::Slerp.min_points_required(), 2);
        assert_eq!(AttitudeInterpolationMethod::Linear.min_points_required(), 2);
        assert_eq!(
            AttitudeInterpolationMethod::Lagrange { degree: 3 }.min_points_required(),
            4
        );
    }

    // =========================================================================
    // AttitudeTrajectory construction / Trajectory trait tests
    // =========================================================================

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_new() {
        let (a, b) = body_frames();
        let traj = AttitudeTrajectory::new(a, b);
        assert_eq!(traj.len(), 0);
        assert!(traj.is_empty());
        assert_eq!(
            traj.interpolation_method,
            AttitudeInterpolationMethod::Slerp
        );
        assert!(!traj.has_rates());
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_add_sorts_out_of_order_epochs() {
        let (a, b) = body_frames();
        let mut traj = AttitudeTrajectory::new(a, b);

        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        traj.add(t0 + 60.0, AttitudeState::new(z_axis_quaternion(0.1)))
            .unwrap();
        traj.add(t0, AttitudeState::new(z_axis_quaternion(0.0)))
            .unwrap();
        traj.add(t0 + 30.0, AttitudeState::new(z_axis_quaternion(0.05)))
            .unwrap();

        assert_eq!(traj.len(), 3);
        assert_eq!(traj.epoch_at_idx(0).unwrap(), t0);
        assert_eq!(traj.epoch_at_idx(1).unwrap(), t0 + 30.0);
        assert_eq!(traj.epoch_at_idx(2).unwrap(), t0 + 60.0);
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_add_repeated_epoch_is_discontinuity() {
        let (a, b) = body_frames();
        let mut traj = AttitudeTrajectory::new(a, b);

        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        traj.add(t0, AttitudeState::new(z_axis_quaternion(0.0)))
            .unwrap();
        traj.add(t0, AttitudeState::new(z_axis_quaternion(0.1)))
            .unwrap();

        // Both states are kept, in insertion order, so the pre- and
        // post-maneuver attitudes of an impulsive slew both survive.
        assert_eq!(traj.len(), 2);
        assert_eq!(
            traj.state_at_idx(0).unwrap().quaternion,
            z_axis_quaternion(0.0)
        );
        assert_eq!(
            traj.state_at_idx(1).unwrap().quaternion,
            z_axis_quaternion(0.1)
        );

        // A query at the discontinuity is right-continuous: it returns the
        // most recently added state rather than dividing by a zero-length
        // interval and producing NaN.
        assert_eq!(
            traj.interpolate(&t0).unwrap().quaternion,
            z_axis_quaternion(0.1)
        );
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_add_rate_mixing_error() {
        let (a, b) = body_frames();
        let mut traj = AttitudeTrajectory::new(a, b);

        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        traj.add(t0, AttitudeState::new(z_axis_quaternion(0.0)))
            .unwrap();

        let result = traj.add(
            t0 + 60.0,
            AttitudeState::new(z_axis_quaternion(0.1))
                .with_angular_velocity(Vector3::new(0.0, 0.0, 0.01)),
        );

        assert!(result.is_err());
        let message = format!("{}", result.unwrap_err());
        assert!(message.contains("angular velocity"));
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_from_data_rate_mixing_error() {
        let (a, b) = body_frames();
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

        let epochs = vec![t0, t0 + 60.0];
        let states = vec![
            AttitudeState::new(z_axis_quaternion(0.0)),
            AttitudeState::new(z_axis_quaternion(0.1))
                .with_angular_velocity(Vector3::new(0.0, 0.0, 0.01)),
        ];

        let result = AttitudeTrajectory::from_data(epochs, states, a, b);
        assert!(result.is_err());
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_from_data_rate_mixing_error_reverse_direction() {
        let (a, b) = body_frames();
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

        let epochs = vec![t0, t0 + 60.0];
        let states = vec![
            AttitudeState::new(z_axis_quaternion(0.0))
                .with_angular_velocity(Vector3::new(0.0, 0.0, 0.01)),
            AttitudeState::new(z_axis_quaternion(0.1)),
        ];

        let result = AttitudeTrajectory::from_data(epochs, states, a, b);
        assert!(result.is_err());
        let message = format!("{}", result.unwrap_err());
        assert!(message.contains("not carry it"));
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_from_data_length_mismatch_errors() {
        let (a, b) = body_frames();
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

        let epochs = vec![t0, t0 + 60.0];
        let states = vec![AttitudeState::new(z_axis_quaternion(0.0))];

        let result = AttitudeTrajectory::from_data(epochs, states, a, b);
        assert!(result.is_err());
        let message = format!("{}", result.unwrap_err());
        assert!(message.contains("same length"));
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_from_data_empty_errors() {
        let (a, b) = body_frames();
        let result = AttitudeTrajectory::from_data(Vec::new(), Vec::new(), a, b);
        assert!(result.is_err());
        let message = format!("{}", result.unwrap_err());
        assert!(message.contains("empty"));
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_from_data_repeated_epoch_is_discontinuity() {
        let (a, b) = body_frames();
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

        // Repeated epoch present even though the input is not pre-sorted.
        let epochs = vec![t0 + 60.0, t0, t0];
        let states = vec![
            AttitudeState::new(z_axis_quaternion(0.1)),
            AttitudeState::new(z_axis_quaternion(0.0)),
            AttitudeState::new(z_axis_quaternion(0.2)),
        ];

        let traj = AttitudeTrajectory::from_data(epochs, states, a, b).unwrap();

        // The sort is stable, so the two states at t0 keep their input order.
        assert_eq!(traj.len(), 3);
        assert_eq!(
            traj.state_at_idx(0).unwrap().quaternion,
            z_axis_quaternion(0.0)
        );
        assert_eq!(
            traj.state_at_idx(1).unwrap().quaternion,
            z_axis_quaternion(0.2)
        );

        // Right-continuous at the discontinuity, and finite rather than NaN.
        assert_eq!(
            traj.interpolate(&t0).unwrap().quaternion,
            z_axis_quaternion(0.2)
        );
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_has_rates() {
        let (a, b) = body_frames();
        let mut traj = AttitudeTrajectory::new(a, b);
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

        assert!(!traj.has_rates());

        traj.add(
            t0,
            AttitudeState::new(z_axis_quaternion(0.0))
                .with_angular_velocity(Vector3::new(0.0, 0.0, 0.01)),
        )
        .unwrap();

        assert!(traj.has_rates());
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_trajectory_from_data_errors_without_frames() {
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let epochs = vec![t0, t0 + 60.0];
        let states = vec![
            AttitudeState::new(z_axis_quaternion(0.0)),
            AttitudeState::new(z_axis_quaternion(0.1)),
        ];

        let result = <AttitudeTrajectory as Trajectory>::from_data(epochs, states);
        assert!(result.is_err());
        let message = format!("{}", result.unwrap_err());
        assert!(message.contains("AttitudeTrajectory::from_data"));
    }

    // =========================================================================
    // interpolate: exact node / out-of-range
    // =========================================================================

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_interpolate_exact_node_returns_stored_state() {
        let (a, b) = body_frames();
        let mut traj = AttitudeTrajectory::new(a, b);
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        traj.add(t0, AttitudeState::new(z_axis_quaternion(0.0)))
            .unwrap();
        traj.add(t0 + 60.0, AttitudeState::new(z_axis_quaternion(0.2)))
            .unwrap();

        let state = traj.interpolate(&t0).unwrap();
        assert_eq!(state.quaternion, z_axis_quaternion(0.0));

        let state = traj.interpolate(&(t0 + 60.0)).unwrap();
        assert_eq!(state.quaternion, z_axis_quaternion(0.2));
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_interpolate_out_of_range() {
        let (a, b) = body_frames();
        let mut traj = AttitudeTrajectory::new(a, b);
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        traj.add(t0, AttitudeState::new(z_axis_quaternion(0.0)))
            .unwrap();
        traj.add(t0 + 60.0, AttitudeState::new(z_axis_quaternion(0.2)))
            .unwrap();

        assert!(matches!(
            traj.interpolate(&(t0 - 10.0)),
            Err(BraheError::OutOfBoundsError(_))
        ));
        assert!(matches!(
            traj.interpolate(&(t0 + 70.0)),
            Err(BraheError::OutOfBoundsError(_))
        ));
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_interpolate_lagrange_min_points_error() {
        let (a, b) = body_frames();
        let mut traj = AttitudeTrajectory::new(a, b)
            .with_interpolation_method(AttitudeInterpolationMethod::Lagrange { degree: 3 });
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        traj.add(t0, AttitudeState::new(z_axis_quaternion(0.0)))
            .unwrap();
        traj.add(t0 + 60.0, AttitudeState::new(z_axis_quaternion(0.2)))
            .unwrap();

        // Only 2 points but degree 3 requires 4
        let result = traj.interpolate(&(t0 + 30.0));
        assert!(result.is_err());
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_interpolate_lagrange_degree_zero_errors() {
        let (a, b) = body_frames();
        let mut traj = AttitudeTrajectory::new(a, b)
            .with_interpolation_method(AttitudeInterpolationMethod::Lagrange { degree: 0 });
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        traj.add(t0, AttitudeState::new(z_axis_quaternion(0.0)))
            .unwrap();
        traj.add(t0 + 60.0, AttitudeState::new(z_axis_quaternion(0.2)))
            .unwrap();

        let err = traj.interpolate(&(t0 + 30.0)).unwrap_err();
        assert!(err.to_string().contains("degree"));
    }

    // =========================================================================
    // interpolate: Slerp exactness on constant-rate single-axis history
    // =========================================================================

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_interpolate_slerp_constant_rate_exact() {
        let (a, b) = body_frames();
        let mut traj = AttitudeTrajectory::new(a, b);

        let omega = 0.01; // rad/s
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let dt = 100.0; // seconds

        traj.add(
            t0,
            AttitudeState::new(z_axis_quaternion(0.0))
                .with_angular_velocity(Vector3::new(0.0, 0.0, omega)),
        )
        .unwrap();
        traj.add(
            t0 + dt,
            AttitudeState::new(z_axis_quaternion(omega * dt))
                .with_angular_velocity(Vector3::new(0.0, 0.0, omega)),
        )
        .unwrap();

        let f = 0.37;
        let query = t0 + f * dt;
        let state = traj.interpolate(&query).unwrap();

        let analytic = z_axis_quaternion(omega * f * dt);
        assert_abs_diff_eq!(
            state.quaternion.to_vector(true)[0],
            analytic.to_vector(true)[0],
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            state.quaternion.to_vector(true)[1],
            analytic.to_vector(true)[1],
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            state.quaternion.to_vector(true)[2],
            analytic.to_vector(true)[2],
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            state.quaternion.to_vector(true)[3],
            analytic.to_vector(true)[3],
            epsilon = 1e-12
        );

        let omega_interp = state.angular_velocity.unwrap();
        assert_abs_diff_eq!(omega_interp[2], omega, epsilon = 1e-12);
    }

    // =========================================================================
    // interpolate: Linear hemisphere-crossing continuity
    // =========================================================================

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_interpolate_linear_hemisphere_continuity() {
        let (a, b) = body_frames();
        let mut traj = AttitudeTrajectory::new(a, b)
            .with_interpolation_method(AttitudeInterpolationMethod::Linear);

        let omega = 0.05; // rad/s
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

        // Store 5 samples 1 second apart, with alternating sign to simulate
        // an arbitrary double-cover representative choice at each sample.
        for i in 0..5 {
            let t = t0 + i as f64;
            let theta = omega * i as f64;
            let mut q = z_axis_quaternion(theta);
            if i % 2 == 1 {
                let v = -q.to_vector(true);
                q = Quaternion::from_vector(v, true);
            }
            traj.add(t, AttitudeState::new(q)).unwrap();
        }

        // Query at the midpoint between index 2 and 3 (opposite stored signs)
        let query = t0 + 2.5;
        let state = traj.interpolate(&query).unwrap();

        let analytic = z_axis_quaternion(omega * 2.5);
        let dot = state
            .quaternion
            .to_vector(true)
            .dot(&analytic.to_vector(true));

        // A correctly hemisphere-aligned interpolation stays close to the
        // analytic attitude (dot near +1); a sign-flip bug would instead
        // land near the negative analytic quaternion or the degenerate
        // near-zero vector between opposite hemispheres.
        assert!(dot > 0.999, "dot = {}", dot);
    }

    // =========================================================================
    // interpolate: Lagrange tolerance on a smooth history
    // =========================================================================

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_interpolate_lagrange_tolerance() {
        let (a, b) = body_frames();
        let mut traj = AttitudeTrajectory::new(a, b)
            .with_interpolation_method(AttitudeInterpolationMethod::Lagrange { degree: 3 });

        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

        // Smooth, non-constant-rate rotation profile about the z-axis.
        let theta = |t: f64| 0.3 * t + 0.05 * t.sin();

        for i in 0..6 {
            let t = i as f64;
            traj.add(t0 + t, AttitudeState::new(z_axis_quaternion(theta(t))))
                .unwrap();
        }

        let query_t = 2.5;
        let state = traj.interpolate(&(t0 + query_t)).unwrap();
        let analytic = z_axis_quaternion(theta(query_t));

        let dot = state
            .quaternion
            .to_vector(true)
            .dot(&analytic.to_vector(true));
        let angular_error = 2.0 * dot.clamp(-1.0, 1.0).acos();

        assert!(angular_error < 5e-3, "angular_error = {}", angular_error);
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_interpolate_lagrange_centered_window_tight_tolerance() {
        let (a, b) = body_frames();
        let mut traj = AttitudeTrajectory::new(a, b)
            .with_interpolation_method(AttitudeInterpolationMethod::Lagrange { degree: 3 });

        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

        // Same smooth, non-constant-rate rotation profile as the edge-window
        // case above, but sampled ten times finer (0.1s spacing instead of
        // 1s). Degree-3 Lagrange interpolation error at a symmetric window
        // center scales roughly with the fourth power of the sample
        // spacing, so shrinking the spacing by 10x should shrink the error
        // by roughly 10^4x relative to the coarser edge-window case.
        let theta = |t: f64| 0.3 * t + 0.05 * t.sin();

        for i in 0..6 {
            let t = 0.1 * i as f64;
            traj.add(t0 + t, AttitudeState::new(z_axis_quaternion(theta(t))))
                .unwrap();
        }

        // At query_t = 0.15, compute_interpolation_window selects the window
        // [0, 1, 2, 3] (t = 0.0..0.3), which is exactly centered on the
        // query -- unlike the t = 2.5 case above, whose selected window is
        // not centered on the query and only meets a looser tolerance.
        let query_t = 0.15;
        let state = traj.interpolate(&(t0 + query_t)).unwrap();
        let analytic = z_axis_quaternion(theta(query_t));

        let dot = state
            .quaternion
            .to_vector(true)
            .dot(&analytic.to_vector(true));
        let angular_error = 2.0 * dot.clamp(-1.0, 1.0).acos();

        assert!(angular_error < 1e-6, "angular_error = {}", angular_error);
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_interpolate_lagrange_wide_span_sequential_alignment() {
        // Regression test: a window whose first and last samples are more
        // than pi apart (z-rotations of 0, 1.5, 3.0, 4.5 rad at 1s spacing,
        // i.e. a constant z-tumble at 1.5 rad/s) must not be hemisphere
        // flipped when aligned against a single fixed reference, since
        // dot(q(0), q(4.5 rad)) = cos((4.5 - 0) / 2) < 0 even though every
        // adjacent pair in the window is well within 90 degrees of each
        // other in 4-space. Sequential alignment (each sample against the
        // previously aligned one) must keep the fit continuous.
        let (a, b) = body_frames();
        let mut traj = AttitudeTrajectory::new(a, b)
            .with_interpolation_method(AttitudeInterpolationMethod::Lagrange { degree: 3 });

        let omega = 1.5; // rad/s
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

        for i in 0..4 {
            let t = i as f64;
            traj.add(t0 + t, AttitudeState::new(z_axis_quaternion(omega * t)))
                .unwrap();
        }

        // Query at the midpoint of the (only, entire-trajectory) window.
        let query_t = 1.5;
        let state = traj.interpolate(&(t0 + query_t)).unwrap();
        let analytic = z_axis_quaternion(omega * query_t);

        let dot = state
            .quaternion
            .to_vector(true)
            .dot(&analytic.to_vector(true));

        // A corrupted fit (fixed-reference alignment flipping only the
        // out-of-hemisphere sample) would land far from the analytic
        // attitude; sequential alignment should keep it close.
        assert!(dot > 0.99, "dot = {}", dot);
    }

    // =========================================================================
    // Eviction smoke test
    // =========================================================================

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_eviction_max_size() {
        let (a, b) = body_frames();
        let mut traj = AttitudeTrajectory::new(a, b);
        traj.set_eviction_policy_max_size(3).unwrap();

        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        for i in 0..5 {
            traj.add(
                t0 + i as f64,
                AttitudeState::new(z_axis_quaternion(0.01 * i as f64)),
            )
            .unwrap();
        }

        assert_eq!(traj.len(), 3);
        assert_eq!(traj.epoch_at_idx(0).unwrap(), t0 + 2.0);
        assert_eq!(traj.epoch_at_idx(2).unwrap(), t0 + 4.0);
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_eviction_max_age() {
        let (a, b) = body_frames();
        let mut traj = AttitudeTrajectory::new(a, b);
        traj.set_eviction_policy_max_age(2.5).unwrap();
        assert_eq!(
            traj.get_eviction_policy(),
            TrajectoryEvictionPolicy::KeepWithinDuration
        );

        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        for i in 0..5 {
            traj.add(
                t0 + i as f64,
                AttitudeState::new(z_axis_quaternion(0.01 * i as f64)),
            )
            .unwrap();
        }

        // Last epoch is t0 + 4.0; only states within 2.5s of it survive.
        assert_eq!(traj.len(), 3);
        assert_eq!(traj.epoch_at_idx(0).unwrap(), t0 + 2.0);
        assert_eq!(traj.epoch_at_idx(2).unwrap(), t0 + 4.0);
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_eviction_policy_setter_errors() {
        let (a, b) = body_frames();
        let mut traj = AttitudeTrajectory::new(a, b);
        assert!(traj.set_eviction_policy_max_size(0).is_err());
        assert!(traj.set_eviction_policy_max_age(0.0).is_err());
        assert!(traj.set_eviction_policy_max_age(-1.0).is_err());
        assert_eq!(traj.get_eviction_policy(), TrajectoryEvictionPolicy::None);
    }

    // =========================================================================
    // add: rate-mixing error, opposite direction
    // =========================================================================

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_add_rate_mixing_error_reverse_direction() {
        let (a, b) = body_frames();
        let mut traj = AttitudeTrajectory::new(a, b);
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

        traj.add(
            t0,
            AttitudeState::new(z_axis_quaternion(0.0))
                .with_angular_velocity(Vector3::new(0.0, 0.0, 0.01)),
        )
        .unwrap();

        let result = traj.add(t0 + 60.0, AttitudeState::new(z_axis_quaternion(0.1)));
        assert!(result.is_err());
        let message = format!("{}", result.unwrap_err());
        assert!(message.contains("carry"));
        assert!(message.contains("does not carry angular velocity"));
    }

    // =========================================================================
    // Trajectory trait: index/state accessors and bounds errors
    // =========================================================================

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_epoch_state_at_idx_out_of_bounds_errors() {
        let (a, b) = body_frames();
        let mut traj = AttitudeTrajectory::new(a, b);
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        traj.add(t0, AttitudeState::new(z_axis_quaternion(0.0)))
            .unwrap();

        assert!(traj.epoch_at_idx(1).is_err());
        assert!(traj.state_at_idx(1).is_err());
        assert!(traj.get(1).is_err());
        assert!(traj.remove(1).is_err());
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_nearest_state() {
        let (a, b) = body_frames();
        let mut traj = AttitudeTrajectory::new(a, b);
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        traj.add(t0, AttitudeState::new(z_axis_quaternion(0.0)))
            .unwrap();
        traj.add(t0 + 10.0, AttitudeState::new(z_axis_quaternion(0.1)))
            .unwrap();
        traj.add(t0 + 20.0, AttitudeState::new(z_axis_quaternion(0.2)))
            .unwrap();

        let (epoch, state) = traj.nearest_state(&(t0 + 3.0)).unwrap();
        assert_eq!(epoch, t0);
        assert_eq!(state.quaternion, z_axis_quaternion(0.0));

        let (epoch, _) = traj.nearest_state(&(t0 + 17.0)).unwrap();
        assert_eq!(epoch, t0 + 20.0);
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_nearest_state_empty_errors() {
        let (a, b) = body_frames();
        let traj = AttitudeTrajectory::new(a, b);
        assert!(traj.nearest_state(&Epoch::now()).is_err());
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_index_before_after_epoch_errors_on_empty() {
        let (a, b) = body_frames();
        let traj = AttitudeTrajectory::new(a, b);
        assert!(traj.index_before_epoch(&Epoch::now()).is_err());
        assert!(traj.index_after_epoch(&Epoch::now()).is_err());
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_index_before_after_epoch_errors_outside_range() {
        let (a, b) = body_frames();
        let mut traj = AttitudeTrajectory::new(a, b);
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        traj.add(t0, AttitudeState::new(z_axis_quaternion(0.0)))
            .unwrap();
        traj.add(t0 + 10.0, AttitudeState::new(z_axis_quaternion(0.1)))
            .unwrap();

        assert!(traj.index_before_epoch(&(t0 - 1.0)).is_err());
        assert!(traj.index_after_epoch(&(t0 + 11.0)).is_err());
        assert_eq!(traj.index_before_epoch(&t0).unwrap(), 0);
        assert_eq!(traj.index_after_epoch(&(t0 + 10.0)).unwrap(), 1);
    }

    // =========================================================================
    // Trajectory trait: timespan / first / last / clear / remove(_epoch)
    // =========================================================================

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_timespan_first_last_empty() {
        let (a, b) = body_frames();
        let traj = AttitudeTrajectory::new(a, b);
        assert!(traj.timespan().is_none());
        assert!(traj.first().is_none());
        assert!(traj.last().is_none());
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_timespan_first_last_populated() {
        let (a, b) = body_frames();
        let mut traj = AttitudeTrajectory::new(a, b);
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        traj.add(t0, AttitudeState::new(z_axis_quaternion(0.0)))
            .unwrap();
        // A single-state trajectory has no timespan.
        assert!(traj.timespan().is_none());

        traj.add(t0 + 30.0, AttitudeState::new(z_axis_quaternion(0.1)))
            .unwrap();

        assert_eq!(traj.timespan(), Some(30.0));
        assert_eq!(traj.first().unwrap().0, t0);
        assert_eq!(traj.last().unwrap().0, t0 + 30.0);
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_clear() {
        let (a, b) = body_frames();
        let mut traj = AttitudeTrajectory::new(a, b);
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        traj.add(t0, AttitudeState::new(z_axis_quaternion(0.0)))
            .unwrap();
        assert!(!traj.is_empty());

        traj.clear();
        assert!(traj.is_empty());
        assert_eq!(traj.len(), 0);
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_remove_epoch() {
        let (a, b) = body_frames();
        let mut traj = AttitudeTrajectory::new(a, b);
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        traj.add(t0, AttitudeState::new(z_axis_quaternion(0.0)))
            .unwrap();
        traj.add(t0 + 10.0, AttitudeState::new(z_axis_quaternion(0.1)))
            .unwrap();

        let removed = traj.remove_epoch(&t0).unwrap();
        assert_eq!(removed.quaternion, z_axis_quaternion(0.0));
        assert_eq!(traj.len(), 1);

        assert!(traj.remove_epoch(&t0).is_err());
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_remove_by_index() {
        let (a, b) = body_frames();
        let mut traj = AttitudeTrajectory::new(a, b);
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        traj.add(t0, AttitudeState::new(z_axis_quaternion(0.0)))
            .unwrap();
        traj.add(t0 + 10.0, AttitudeState::new(z_axis_quaternion(0.1)))
            .unwrap();

        let (epoch, state) = traj.remove(0).unwrap();
        assert_eq!(epoch, t0);
        assert_eq!(state.quaternion, z_axis_quaternion(0.0));
        assert_eq!(traj.len(), 1);
        assert_eq!(traj.epoch_at_idx(0).unwrap(), t0 + 10.0);
    }

    // =========================================================================
    // Lagrange windowing around a discontinuity
    // =========================================================================

    #[test]
    #[serial_test::parallel]
    fn test_lagrange_window_does_not_span_a_discontinuity() {
        let (a, b) = body_frames();
        let mut traj = AttitudeTrajectory::new(a, b);
        traj.set_interpolation_method(AttitudeInterpolationMethod::Lagrange { degree: 3 });

        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        // Continuous run, then an impulsive slew at t0 + 20 whose post-state
        // jumps, then a second continuous run.
        for (offset, angle) in [(0.0, 0.0), (10.0, 0.2), (20.0, 0.4)] {
            traj.add(t0 + offset, AttitudeState::new(z_axis_quaternion(angle)))
                .unwrap();
        }
        for (offset, angle) in [(20.0, 2.0), (30.0, 2.2), (40.0, 2.4)] {
            traj.add(t0 + offset, AttitudeState::new(z_axis_quaternion(angle)))
                .unwrap();
        }

        // A query inside the first run fits only that run's samples. Fitting
        // across the repeated epoch would divide by a zero-length abscissa
        // difference and return NaN.
        let q = traj.interpolate(&(t0 + 5.0)).unwrap().quaternion;
        let v = q.to_vector(true);
        assert!(v.iter().all(|c| c.is_finite()), "{:?}", v);
        assert_abs_diff_eq!(v.norm(), 1.0, epsilon = 1e-12);

        // The same holds for a query inside the second run.
        let q = traj.interpolate(&(t0 + 35.0)).unwrap().quaternion;
        let v = q.to_vector(true);
        assert!(v.iter().all(|c| c.is_finite()), "{:?}", v);
        assert_abs_diff_eq!(v.norm(), 1.0, epsilon = 1e-12);
    }

    // =========================================================================
    // interpolate: Lagrange with angular velocity / in-window hemisphere flip
    // =========================================================================

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_interpolate_lagrange_with_angular_velocity() {
        let (a, b) = body_frames();
        let mut traj = AttitudeTrajectory::new(a, b)
            .with_interpolation_method(AttitudeInterpolationMethod::Lagrange { degree: 3 });

        let omega = Vector3::new(0.0, 0.0, 0.02);
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

        for i in 0..4 {
            let t = i as f64;
            traj.add(
                t0 + t,
                AttitudeState::new(z_axis_quaternion(omega[2] * t)).with_angular_velocity(omega),
            )
            .unwrap();
        }

        let state = traj.interpolate(&(t0 + 1.5)).unwrap();
        let interpolated_omega = state.angular_velocity.unwrap();
        assert_abs_diff_eq!(interpolated_omega[2], omega[2], epsilon = 1e-12);
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_trajectory_interpolate_lagrange_in_window_hemisphere_flip() {
        // Regression test for the per-step hemisphere alignment inside the
        // Lagrange window itself: with omega = 4 rad/s at 1s spacing, the
        // rotation between *every* adjacent sample pair exceeds pi radians
        // in the quaternion double cover (half-angle > pi/2, so
        // dot(q_i, q_{i+1}) < 0), forcing the in-window sign flip on every
        // step rather than only when comparing the window's first and last
        // samples. A window this coarse relative to the rotation rate is
        // aliased (more than one full double-cover cycle spans the window),
        // so this only checks that the per-step alignment branch runs to
        // completion and still produces a unit quaternion, not that the
        // fit is numerically close to the continuous analytic attitude.
        let (a, b) = body_frames();
        let mut traj = AttitudeTrajectory::new(a, b)
            .with_interpolation_method(AttitudeInterpolationMethod::Lagrange { degree: 3 });

        let omega = 4.0; // rad/s
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

        for i in 0..4 {
            let t = i as f64;
            traj.add(t0 + t, AttitudeState::new(z_axis_quaternion(omega * t)))
                .unwrap();
        }

        let state = traj.interpolate(&(t0 + 1.5)).unwrap();
        let norm = state.quaternion.to_vector(true).norm();
        assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-9);
    }
}
