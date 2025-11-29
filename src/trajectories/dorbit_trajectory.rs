/*!
 * Dynamic orbital trajectory implementation for orbital state vectors with extended states.
 *
 * This module provides a runtime-sized, specialized trajectory container for orbital
 * mechanics applications, using dynamic `DVector<f64>` and `DMatrix<f64>` types
 * for flexibility. Supports state vectors of dimension 6 + N where the first 6 elements
 * are the orbital state (position + velocity) and additional elements are passed through
 * conversions unchanged. For a static (compile-time sized) alternative with better
 * performance, see `SOrbitTrajectory`.
 *
 * # Key Features
 * - **Dynamic dimensions**: Support 6D orbital states or 6+N extended states
 * - **Selective conversions**: Frame/representation conversions apply to first 6 elements only
 * - **Reference frame conversions** (ECI ↔ ECEF)
 * - **State representation conversions** (Cartesian ↔ Keplerian)
 * - **Angle format conversions** (Radians ↔ Degrees)
 * - Position and velocity extraction from Cartesian states
 * - Combined conversions for efficiency
 * - Runtime-sized vectors for integration with dynamic propagators
 *
 * # Examples
 *
 * ## Standard 6D orbital trajectory
 * ```rust
 * use brahe::trajectories::DOrbitTrajectory;
 * use brahe::traits::{Trajectory, OrbitFrame, OrbitRepresentation};
 * use brahe::AngleFormat;
 * use brahe::time::{Epoch, TimeSystem};
 * use nalgebra::DVector;
 *
 * // Create 6D orbital trajectory in ECI Cartesian coordinates
 * let mut traj = DOrbitTrajectory::new(
 *     6,  // dimension
 *     OrbitFrame::ECI,
 *     OrbitRepresentation::Cartesian,
 *     None,
 * );
 *
 * // Add state
 * let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
 * let state = DVector::from_vec(vec![6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0]);
 * traj.add(epoch, state);
 *
 * // Convert to Keplerian in degrees (only first 6 elements converted)
 * let kep_traj = traj.to_keplerian(AngleFormat::Degrees);
 * ```
 *
 * ## Extended state trajectory (6D + additional states)
 * ```rust
 * use brahe::trajectories::DOrbitTrajectory;
 * use brahe::traits::{Trajectory, OrbitFrame, OrbitRepresentation};
 * use brahe::time::{Epoch, TimeSystem};
 * use nalgebra::DVector;
 *
 * // Initialize EOP for frame conversions
 * brahe::eop::set_global_eop_provider(
 *     brahe::eop::StaticEOPProvider::from_values((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
 * );
 *
 * // Create 9D trajectory (6D orbit + 3 additional states)
 * let mut traj = DOrbitTrajectory::new(
 *     9,  // dimension
 *     OrbitFrame::ECI,
 *     OrbitRepresentation::Cartesian,
 *     None,
 * );
 *
 * // Add extended state
 * let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
 * let state = DVector::from_vec(vec![
 *     6.678e6, 0.0, 0.0,  // position
 *     0.0, 7.726e3, 0.0,  // velocity
 *     1.0, 2.0, 3.0,      // additional states (passed through conversions)
 * ]);
 * traj.add(epoch, state);
 *
 * // Convert to ECEF - first 6 elements converted, last 3 unchanged
 * let ecef_traj = traj.to_ecef();
 * ```
 */

use nalgebra::{DMatrix, DVector, SMatrix, Vector3, Vector6};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use uuid::Uuid;

use crate::constants::AngleFormat;
use crate::constants::{DEG2RAD, RAD2DEG};
use crate::coordinates::{state_eci_to_koe, state_koe_to_eci};
use crate::frames::{
    rotation_eme2000_to_gcrf, state_ecef_to_eci, state_eci_to_ecef, state_eme2000_to_gcrf,
    state_gcrf_to_eme2000, state_gcrf_to_itrf, state_itrf_to_gcrf,
};
use crate::math::{
    CovarianceInterpolationConfig, interpolate_covariance_sqrt_dmatrix,
    interpolate_covariance_two_wasserstein_dmatrix, interpolate_hermite_cubic_dvector6,
    interpolate_hermite_quintic_dvector6, interpolate_hermite_quintic_fd_dvector6,
    interpolate_lagrange_dvector,
};
use crate::relative_motion::rotation_eci_to_rtn;
use crate::time::Epoch;
use crate::utils::state_providers::{
    DCovarianceProvider, DOrbitCovarianceProvider, DOrbitStateProvider, DStateProvider,
};
use crate::utils::{BraheError, Identifiable};

/// Convert a DVector to a static Vector6.
/// Panics if the DVector doesn't have exactly 6 elements.
#[inline]
fn dvec_to_svec6(dv: DVector<f64>) -> Vector6<f64> {
    assert_eq!(dv.len(), 6, "DVector must have exactly 6 elements");
    Vector6::from_iterator(dv.iter().copied())
}

/// Convert a static Vector6 to a DVector.
#[inline]
fn svec6_to_dvec(sv: Vector6<f64>) -> DVector<f64> {
    DVector::from_iterator(6, sv.iter().copied())
}

/// Convert a DMatrix to a static SMatrix<6, 6>.
/// Panics if the DMatrix isn't 6x6.
#[inline]
#[allow(dead_code)]
fn dmat_to_smat66(dm: DMatrix<f64>) -> SMatrix<f64, 6, 6> {
    assert_eq!(dm.nrows(), 6, "DMatrix must have 6 rows");
    assert_eq!(dm.ncols(), 6, "DMatrix must have 6 columns");
    SMatrix::<f64, 6, 6>::from_iterator(dm.iter().copied())
}

/// Convert a static SMatrix<6, 6> to a DMatrix.
#[inline]
#[allow(dead_code)]
fn smat66_to_dmat(sm: SMatrix<f64, 6, 6>) -> DMatrix<f64> {
    DMatrix::from_iterator(6, 6, sm.iter().copied())
}

use super::traits::{
    CovarianceInterpolationMethod, InterpolatableTrajectory, InterpolationConfig,
    InterpolationMethod, OrbitFrame, OrbitRepresentation, STMStorage, SensitivityStorage,
    Trajectory, TrajectoryEvictionPolicy,
};

/// Dynamic (runtime-sized) orbital trajectory container.
///
/// This struct uses dynamic `DVector<f64>` and `DMatrix<f64>` types for flexibility
/// and integration with dynamic propagators. It provides orbital-specific functionality
/// including conversions between reference frames (ECI/ECEF), state representations
/// (Cartesian/Keplerian), and angle formats (radians/degrees).
///
/// For a static (compile-time sized) alternative with better performance, see `SOrbitTrajectory`.
#[derive(Debug, Clone, PartialEq)]
pub struct DOrbitTrajectory {
    /// Time epochs for each state, maintained in chronological order.
    /// All epochs should use consistent time systems for meaningful interpolation.
    pub epochs: Vec<Epoch>,

    /// R-dimensional state vectors corresponding to epochs.
    /// Units and interpretation depend on the specific use case:
    /// - Cartesian: [m, m, m, m/s, m/s, m/s]
    /// - Keplerian: [m, dimensionless, rad or deg, rad or deg, rad or deg, rad or deg]
    pub states: Vec<DVector<f64>>,

    /// Optional covariance matrices corresponding to states.
    /// Each covariance is a 6×6 symmetric matrix representing **orbital** state uncertainty only.
    /// Additional state elements (6+) are not included in covariance tracking.
    /// Units: [m², m·m/s, (m/s)²] for Cartesian states.
    /// If present, must have same length as states vector.
    pub covariances: Option<Vec<DMatrix<f64>>>,

    /// Optional state transition matrices (STM) corresponding to each state.
    /// Each STM is 6×6 relating orbital state changes: Φ(t,t₀) = ∂x_orbital(t)/∂x_orbital(t₀).
    /// Additional state elements (6+) are not included in STM computation.
    /// If present, must have the same length as `states` and each matrix must be 6×6.
    pub stms: Option<Vec<DMatrix<f64>>>,

    /// Optional sensitivity matrices corresponding to each state.
    /// Each matrix is 6×param_dim: ∂x_orbital/∂p where x_orbital is the orbital state only.
    /// Additional state elements (6+) are not included in sensitivity computation.
    /// If present, must have the same length as `states`.
    pub sensitivities: Option<Vec<DMatrix<f64>>>,

    /// Dimension of sensitivity matrices as (rows, cols) = (6, param_dim).
    /// Set when sensitivity storage is enabled.
    sensitivity_dimension: Option<(usize, usize)>,

    /// State vector dimension (must be >= 6).
    /// - Elements 0-5: orbital state (position + velocity)
    /// - Elements 6+: additional states (passed through conversions unchanged)
    dimension: usize,

    /// Interpolation method for state retrieval at arbitrary epochs.
    /// Default is linear interpolation for optimal performance/accuracy balance.
    pub interpolation_method: InterpolationMethod,

    /// Interpolation method for covariance retrieval at arbitrary epochs.
    /// Default is linear interpolation for element-wise interpolation.
    pub covariance_interpolation_method: CovarianceInterpolationMethod,

    /// Memory management policy for automatic state eviction.
    /// Controls how states are removed when limits are exceeded.
    pub eviction_policy: TrajectoryEvictionPolicy,

    /// Maximum number of states to retain (for KeepCount policy).
    max_size: Option<usize>,

    /// Maximum age of states to retain in seconds (for KeepWithinDuration policy).
    max_age: Option<f64>,

    /// Reference frame of the orbital states.
    pub frame: OrbitFrame,

    /// State representation (Cartesian or Keplerian).
    /// Keplerian elements are always in ECI frame.
    /// Cartesian can be in ECI or ECEF.
    pub representation: OrbitRepresentation,

    /// Angle format for angular elements
    /// None is used for Cartesian representation.
    /// Some(Radians) or Some(Degrees) can be used for Keplerian elements.
    pub angle_format: Option<AngleFormat>,

    /// Optional user-defined name for identification
    pub name: Option<String>,

    /// Optional user-defined numeric ID for identification
    pub id: Option<u64>,

    /// Optional UUID for unique identification
    pub uuid: Option<uuid::Uuid>,

    /// Generic metadata storage supporting arbitrary key-value pairs.
    /// Can store any JSON-serializable data including strings, numbers, booleans,
    /// arrays, and nested objects. For orbital trajectories, use ORBITAL_*_KEY constants.
    pub metadata: HashMap<String, Value>,

    /// Optional acceleration vectors corresponding to each state.
    /// If present, must have the same length as `states`.
    /// Used for quintic Hermite interpolation. Accelerations are dynamic vectors
    /// with dimension specified by `acceleration_dimension`.
    pub accelerations: Option<Vec<DVector<f64>>>,

    /// Dimension of acceleration vectors.
    /// Set when acceleration storage is enabled.
    acceleration_dimension: Option<usize>,
}

impl fmt::Display for DOrbitTrajectory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DOrbitTrajectory(frame={}, representation={}, states={})",
            self.frame,
            self.representation,
            self.len()
        )
    }
}

impl DOrbitTrajectory {
    /// Creates a new orbital trajectory with specified dimension, frame, representation, and angle format.
    ///
    /// # Arguments
    /// * `dimension` - State vector dimension (must be >= 6). First 6 elements are orbital state,
    ///   elements 6+ are additional states passed through conversions unchanged.
    /// * `frame` - Reference frame (ECI or ECEF)
    /// * `representation` - State representation (Cartesian or Keplerian)
    /// * `angle_format` - Angle format (None for Cartesian, Radians/Degrees for Keplerian)
    ///
    /// # Returns
    /// New empty orbital trajectory
    ///
    /// # Panics
    /// * If `dimension < 6`
    /// * If Keplerian representation without angle_format
    /// * If Cartesian representation with angle_format
    /// * If ECEF frame with Keplerian representation
    ///
    /// # Examples
    /// ```rust
    /// use brahe::trajectories::DOrbitTrajectory;
    /// use brahe::traits::{OrbitFrame, OrbitRepresentation};
    /// use brahe::AngleFormat;
    ///
    /// // Standard 6D orbital trajectory
    /// let traj = DOrbitTrajectory::new(
    ///     6,
    ///     OrbitFrame::ECI,
    ///     OrbitRepresentation::Cartesian,
    ///     None,
    /// );
    ///
    /// // Extended 9D trajectory (6D orbit + 3 additional states)
    /// let traj_extended = DOrbitTrajectory::new(
    ///     9,
    ///     OrbitFrame::ECI,
    ///     OrbitRepresentation::Cartesian,
    ///     None,
    /// );
    /// ```
    pub fn new(
        dimension: usize,
        frame: OrbitFrame,
        representation: OrbitRepresentation,
        angle_format: Option<AngleFormat>,
    ) -> Self {
        // Validate dimension
        if dimension < 6 {
            panic!(
                "State dimension must be at least 6 (position + velocity), got {}",
                dimension
            );
        }
        // Validate angle_format for representation (check this first)
        if representation == OrbitRepresentation::Keplerian && angle_format.is_none() {
            panic!("Angle format must be specified for Keplerian elements");
        }

        if representation == OrbitRepresentation::Cartesian && angle_format.is_some() {
            panic!("Angle format should be None for Cartesian representation");
        }

        // Validate frame for representation
        if frame == OrbitFrame::ECEF && representation == OrbitRepresentation::Keplerian {
            panic!("Keplerian elements should be in ECI frame");
        }

        Self {
            epochs: Vec::new(),
            states: Vec::new(),
            covariances: None,
            stms: None,
            sensitivities: None,
            sensitivity_dimension: None,
            dimension,
            interpolation_method: InterpolationMethod::Linear,
            covariance_interpolation_method: CovarianceInterpolationMethod::TwoWasserstein,
            eviction_policy: TrajectoryEvictionPolicy::None,
            max_size: None,
            max_age: None,
            frame,
            representation,
            angle_format,
            name: None,
            id: None,
            uuid: None,
            metadata: HashMap::new(),
            accelerations: None,
            acceleration_dimension: None,
        }
    }

    /// Sets the interpolation method using builder pattern.
    ///
    /// This method consumes self and returns a new trajectory with the specified
    /// interpolation method, allowing for method chaining.
    ///
    /// # Arguments
    /// * `interpolation_method` - Method to use for state interpolation between epochs
    ///
    /// # Returns
    /// Self with updated interpolation method
    ///
    /// # Examples
    /// ```rust
    /// use brahe::trajectories::DOrbitTrajectory;
    /// use brahe::traits::{OrbitFrame, OrbitRepresentation, InterpolationMethod};
    /// let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
    ///     .with_interpolation_method(InterpolationMethod::Linear);
    /// ```
    pub fn with_interpolation_method(mut self, interpolation_method: InterpolationMethod) -> Self {
        self.interpolation_method = interpolation_method;
        self
    }

    /// Sets the eviction policy to keep a maximum number of states using builder pattern.
    ///
    /// This method consumes self and returns a new trajectory with the specified
    /// eviction policy, allowing for method chaining.
    ///
    /// # Arguments
    /// * `max_size` - Maximum number of states to retain (must be >= 1)
    ///
    /// # Returns
    /// Self with updated eviction policy
    ///
    /// # Panics
    /// Panics if max_size is less than 1
    ///
    /// # Examples
    /// ```rust
    /// use brahe::trajectories::DOrbitTrajectory;
    /// use brahe::traits::{OrbitFrame, OrbitRepresentation};
    /// let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
    ///     .with_eviction_policy_max_size(100);
    /// ```
    pub fn with_eviction_policy_max_size(mut self, max_size: usize) -> Self {
        if max_size < 1 {
            panic!("Maximum size must be >= 1");
        }
        self.eviction_policy = TrajectoryEvictionPolicy::KeepCount;
        self.max_size = Some(max_size);
        self.max_age = None;
        self
    }

    /// Sets the eviction policy to keep states within a maximum age using builder pattern.
    ///
    /// This method consumes self and returns a new trajectory with the specified
    /// eviction policy, allowing for method chaining.
    ///
    /// # Arguments
    /// * `max_age` - Maximum age of states to retain in seconds (must be > 0.0)
    ///
    /// # Returns
    /// Self with updated eviction policy
    ///
    /// # Panics
    /// Panics if max_age is not positive
    ///
    /// # Examples
    /// ```rust
    /// use brahe::trajectories::DOrbitTrajectory;
    /// use brahe::traits::{OrbitFrame, OrbitRepresentation};
    /// let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
    ///     .with_eviction_policy_max_age(3600.0);
    /// ```
    pub fn with_eviction_policy_max_age(mut self, max_age: f64) -> Self {
        if max_age <= 0.0 {
            panic!("Maximum age must be > 0.0");
        }
        self.eviction_policy = TrajectoryEvictionPolicy::KeepWithinDuration;
        self.max_age = Some(max_age);
        self.max_size = None;
        self
    }

    /// Returns the total dimension of state vectors in this trajectory.
    ///
    /// # Returns
    /// Total state dimension (6 + N where N is the number of additional states)
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the dimension of the orbital part of state vectors (always 6).
    ///
    /// The orbital state consists of position (3) and velocity (3) components.
    ///
    /// # Returns
    /// Always returns 6
    pub fn orbital_dimension(&self) -> usize {
        6
    }

    /// Returns the number of additional state elements beyond the orbital state.
    ///
    /// # Returns
    /// Number of additional states (dimension - 6)
    pub fn additional_dimension(&self) -> usize {
        self.dimension.saturating_sub(6)
    }

    /// Apply a 6D conversion function to the orbital part of a state,
    /// preserving any additional states unchanged.
    ///
    /// This is the core helper for all frame and representation conversions.
    /// The conversion function is applied only to the first 6 elements (orbital state),
    /// while elements 6+ (additional states) are copied unchanged.
    ///
    /// # Arguments
    /// * `state` - Full state vector (6 + N dimensions)
    /// * `converter` - Function that converts 6D state vectors
    ///
    /// # Returns
    /// Converted state with same dimension as input
    fn convert_orbital_preserving_additional<F>(
        &self,
        state: &DVector<f64>,
        converter: F,
    ) -> DVector<f64>
    where
        F: Fn(Vector6<f64>) -> Vector6<f64>,
    {
        // Extract orbital part (first 6 elements)
        let orbital = dvec_to_svec6(state.rows(0, 6).into_owned());

        // Apply conversion to orbital part
        let converted_orbital = converter(orbital);

        // Reassemble full state
        let mut result = DVector::zeros(state.len());
        result
            .rows_mut(0, 6)
            .copy_from(&svec6_to_dvec(converted_orbital));

        // Copy additional states unchanged
        if state.len() > 6 {
            result
                .rows_mut(6, state.len() - 6)
                .copy_from(&state.rows(6, state.len() - 6));
        }

        result
    }

    /// Add a state with its corresponding covariance matrix to the trajectory.
    ///
    /// This method adds both the state and its covariance at the specified epoch,
    /// maintaining chronological order and parallel structure between states and covariances.
    ///
    /// # Arguments
    /// * `epoch` - The epoch for this state/covariance pair
    /// * `state` - The 6-element state vector
    /// * `covariance` - The 6x6 covariance matrix
    ///
    /// # Panics
    /// * If the trajectory doesn't have covariances initialized (is None)
    ///
    /// # Examples
    /// ```
    /// use brahe::trajectories::DOrbitTrajectory;
    /// use brahe::traits::{OrbitFrame, OrbitRepresentation};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use nalgebra::{DMatrix, DVector};
    ///
    /// let mut traj = DOrbitTrajectory::new(
    ///     6,
    ///     OrbitFrame::ECI,
    ///     OrbitRepresentation::Cartesian,
    ///     None,
    /// );
    ///
    /// // Initialize covariances
    /// traj.covariances = Some(Vec::new());
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = DVector::<f64>::zeros(6);
    /// let cov = DMatrix::<f64>::identity(6, 6);
    ///
    /// traj.add_state_and_covariance(epoch, state, cov);
    /// ```
    pub fn add_state_and_covariance(
        &mut self,
        epoch: Epoch,
        state: DVector<f64>,
        covariance: DMatrix<f64>,
    ) {
        if self.covariances.is_none() {
            panic!(
                "Cannot add state with covariance to trajectory without covariances initialized. Initialize trajectory with covariances or use from_orbital_data with covariances parameter."
            );
        }

        // Find the correct position to insert based on epoch
        // Insert after any existing states at the same epoch to support
        // impulsive maneuvers where we want both pre- and post-maneuver states
        let mut insert_idx = self.epochs.len();
        for (i, existing_epoch) in self.epochs.iter().enumerate() {
            if epoch < *existing_epoch {
                insert_idx = i;
                break;
            }
            // If epochs are equal, continue to find the position after all equal epochs
        }

        // Insert at the correct position
        self.epochs.insert(insert_idx, epoch);
        self.states.insert(insert_idx, state);
        if let Some(ref mut covs) = self.covariances {
            covs.insert(insert_idx, covariance);
        }

        // Apply eviction policy after adding state
        self.apply_eviction_policy();
    }

    /// Convert the trajectory to a matrix representation
    /// Returns a matrix where rows are time points (epochs) and columns are state elements
    /// The matrix has shape (n_epochs, 6) for a 6-element state vector
    pub fn to_matrix(&self) -> Result<nalgebra::DMatrix<f64>, BraheError> {
        if self.states.is_empty() {
            return Err(BraheError::Error(
                "Cannot convert empty trajectory to matrix".to_string(),
            ));
        }

        let n_epochs = self.states.len();
        let n_elements = 6;

        let mut matrix = nalgebra::DMatrix::<f64>::zeros(n_epochs, n_elements);

        for (row_idx, state) in self.states.iter().enumerate() {
            for col_idx in 0..n_elements {
                matrix[(row_idx, col_idx)] = state[col_idx];
            }
        }

        Ok(matrix)
    }

    /// Enable STM storage
    /// Get covariance matrix at a specific epoch (with interpolation)
    ///
    /// Returns None if covariance storage is not enabled or epoch is out of range.
    ///
    /// # Arguments
    /// * `epoch` - Time epoch to query
    ///
    /// # Returns
    /// Covariance matrix at the requested epoch (interpolated if necessary)
    pub fn covariance_at(&self, epoch: Epoch) -> Option<DMatrix<f64>> {
        let covs = self.covariances.as_ref()?;

        if self.epochs.is_empty() {
            return None;
        }

        // Handle exact match at endpoint
        if let Some((idx, _)) = self.epochs.iter().enumerate().find(|(_, e)| **e == epoch) {
            return Some(covs[idx].clone());
        }

        // Find surrounding indices for interpolation
        let (idx_before, idx_after) = self.find_surrounding_indices(epoch)?;

        // Handle exact matches
        if self.epochs[idx_before] == epoch {
            return Some(covs[idx_before].clone());
        }
        if self.epochs[idx_after] == epoch {
            return Some(covs[idx_after].clone());
        }

        // Interpolation parameter
        let t0 = self.epochs[idx_before] - self.epoch_initial()?;
        let t1 = self.epochs[idx_after] - self.epoch_initial()?;
        let t = epoch - self.epoch_initial()?;
        let alpha = (t - t0) / (t1 - t0);

        let cov0 = &covs[idx_before];
        let cov1 = &covs[idx_after];

        // Dispatch based on covariance interpolation method
        let cov = match self.covariance_interpolation_method {
            CovarianceInterpolationMethod::MatrixSquareRoot => {
                interpolate_covariance_sqrt_dmatrix(cov0, cov1, alpha)
            }
            CovarianceInterpolationMethod::TwoWasserstein => {
                interpolate_covariance_two_wasserstein_dmatrix(cov0, cov1, alpha)
            }
        };

        Some(cov)
    }

    /// Add a complete state record with all optional data
    ///
    /// This is the most flexible method, allowing any combination of
    /// covariance, STM, sensitivity, and acceleration to be provided or omitted.
    /// Automatically enables storage for any provided data.
    ///
    /// # Arguments
    /// * `epoch` - Time epoch
    /// * `state` - State vector (must match trajectory dimension)
    /// * `covariance` - Optional covariance matrix (6x6)
    /// * `stm` - Optional state transition matrix (6x6)
    /// * `sensitivity` - Optional sensitivity matrix (6 x param_dim)
    /// * `acceleration` - Optional acceleration vector (must match acceleration_dimension if set)
    ///
    /// # Panics
    /// Panics if dimensions don't match
    pub fn add_full(
        &mut self,
        epoch: Epoch,
        state: DVector<f64>,
        covariance: Option<DMatrix<f64>>,
        stm: Option<DMatrix<f64>>,
        sensitivity: Option<DMatrix<f64>>,
        acceleration: Option<DVector<f64>>,
    ) {
        // Validate state dimension
        if state.len() != self.dimension {
            panic!(
                "State vector dimension {} does not match trajectory dimension {}",
                state.len(),
                self.dimension
            );
        }

        // Validate and enable storage as needed
        if let Some(ref cov) = covariance {
            if cov.nrows() != 6 || cov.ncols() != 6 {
                panic!("Covariance dimension mismatch");
            }
            if self.covariances.is_none() {
                self.covariances = Some(vec![DMatrix::zeros(6, 6); self.states.len()]);
            }
        }

        if let Some(ref s) = stm {
            if s.nrows() != 6 || s.ncols() != 6 {
                panic!("STM dimension mismatch");
            }
            if self.stms.is_none() {
                STMStorage::enable_stm_storage(self);
            }
        }

        if let Some(ref sens) = sensitivity {
            if sens.nrows() != 6 {
                panic!("Sensitivity row dimension mismatch");
            }
            if let Some((_, cols)) = self.sensitivity_dimension
                && sens.ncols() != cols
            {
                panic!("Sensitivity column dimension mismatch");
            }
            if self.sensitivities.is_none() {
                SensitivityStorage::enable_sensitivity_storage(self, sens.ncols());
            }
        }

        if let Some(ref acc) = acceleration {
            if let Some(acc_dim) = self.acceleration_dimension
                && acc.len() != acc_dim
            {
                panic!(
                    "Acceleration dimension {} does not match trajectory acceleration dimension {}",
                    acc.len(),
                    acc_dim
                );
            }
            if self.accelerations.is_none() {
                let acc_dim = acc.len();
                self.acceleration_dimension = Some(acc_dim);
                self.accelerations = Some(vec![DVector::zeros(acc_dim); self.states.len()]);
            }
        }

        // Find insert position
        let mut insert_idx = self.epochs.len();
        for (i, existing_epoch) in self.epochs.iter().enumerate() {
            if epoch < *existing_epoch {
                insert_idx = i;
                break;
            }
        }

        // Insert core data
        self.epochs.insert(insert_idx, epoch);
        self.states.insert(insert_idx, state);

        // Insert optional data or placeholders
        if let Some(ref mut covs) = self.covariances {
            let cov_val = covariance.unwrap_or_else(|| DMatrix::zeros(6, 6));
            covs.insert(insert_idx, cov_val);
        }

        if let Some(ref mut stms) = self.stms {
            let stm_val = stm.unwrap_or_else(|| DMatrix::identity(6, 6));
            stms.insert(insert_idx, stm_val);
        }

        if let Some(ref mut sens) = self.sensitivities {
            let (_, param_dim) = self.sensitivity_dimension.unwrap();
            let sens_val = sensitivity.unwrap_or_else(|| DMatrix::zeros(6, param_dim));
            sens.insert(insert_idx, sens_val);
        }

        if let Some(ref mut accs) = self.accelerations {
            let acc_dim = self.acceleration_dimension.unwrap();
            let acc_val = acceleration.unwrap_or_else(|| DVector::zeros(acc_dim));
            accs.insert(insert_idx, acc_val);
        }

        self.apply_eviction_policy();
    }

    /// Helper to find initial epoch for interpolation
    fn epoch_initial(&self) -> Option<Epoch> {
        self.epochs.first().copied()
    }

    /// Helper to find surrounding indices for interpolation
    fn find_surrounding_indices(&self, epoch: Epoch) -> Option<(usize, usize)> {
        if self.epochs.is_empty() {
            return None;
        }

        // Check bounds
        if epoch < self.epochs[0] || epoch > *self.epochs.last()? {
            return None;
        }

        // Binary search to find the interval
        for i in 0..self.epochs.len() - 1 {
            if self.epochs[i] <= epoch && epoch <= self.epochs[i + 1] {
                return Some((i, i + 1));
            }
        }

        None
    }

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
                    if let Some(ref mut covs) = self.covariances {
                        covs.drain(0..to_remove);
                    }
                    if let Some(ref mut stms) = self.stms {
                        stms.drain(0..to_remove);
                    }
                    if let Some(ref mut sens) = self.sensitivities {
                        sens.drain(0..to_remove);
                    }
                    if let Some(ref mut accs) = self.accelerations {
                        accs.drain(0..to_remove);
                    }
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
                    let new_states: Vec<DVector<f64>> = indices_to_keep
                        .iter()
                        .map(|&i| self.states[i].clone())
                        .collect();
                    let new_covariances = self
                        .covariances
                        .as_ref()
                        .map(|covs| indices_to_keep.iter().map(|&i| covs[i].clone()).collect());
                    let new_stms = self
                        .stms
                        .as_ref()
                        .map(|stms| indices_to_keep.iter().map(|&i| stms[i].clone()).collect());
                    let new_sensitivities = self
                        .sensitivities
                        .as_ref()
                        .map(|sens| indices_to_keep.iter().map(|&i| sens[i].clone()).collect());
                    let new_accelerations = self
                        .accelerations
                        .as_ref()
                        .map(|accs| indices_to_keep.iter().map(|&i| accs[i].clone()).collect());

                    self.epochs = new_epochs;
                    self.states = new_states;
                    self.covariances = new_covariances;
                    self.stms = new_stms;
                    self.sensitivities = new_sensitivities;
                    self.accelerations = new_accelerations;
                }
            }
        }
    }

    // ==================== Acceleration Helper Methods ====================

    /// Enables acceleration storage with the specified dimension.
    ///
    /// If acceleration storage is not yet enabled, this initializes the storage
    /// with zero vectors for all existing states. If already enabled, validates
    /// the dimension matches.
    ///
    /// # Arguments
    /// * `dimension` - Dimension of acceleration vectors (typically 3 for [ax, ay, az])
    ///
    /// # Panics
    /// Panics if already enabled with a different dimension.
    pub fn enable_acceleration_storage(&mut self, dimension: usize) {
        if let Some(existing_dim) = self.acceleration_dimension {
            if existing_dim != dimension {
                panic!(
                    "Cannot change acceleration dimension from {} to {}",
                    existing_dim, dimension
                );
            }
            return;
        }
        self.acceleration_dimension = Some(dimension);
        self.accelerations = Some(vec![DVector::zeros(dimension); self.states.len()]);
    }

    /// Returns whether acceleration storage is enabled.
    pub fn has_accelerations(&self) -> bool {
        self.accelerations.is_some()
    }

    /// Returns the dimension of acceleration vectors if storage is enabled.
    pub fn acceleration_dim(&self) -> Option<usize> {
        self.acceleration_dimension
    }

    /// Returns a reference to the acceleration at the given index.
    ///
    /// # Arguments
    /// * `index` - Index into the trajectory
    ///
    /// # Returns
    /// Reference to the acceleration vector, or None if not stored
    pub fn acceleration_at_idx(&self, index: usize) -> Option<&DVector<f64>> {
        self.accelerations.as_ref()?.get(index)
    }

    /// Sets the acceleration at the given index.
    ///
    /// # Arguments
    /// * `index` - Index into the trajectory
    /// * `acceleration` - The acceleration vector to set
    ///
    /// # Panics
    /// * If acceleration storage is not enabled
    /// * If index is out of bounds
    /// * If acceleration dimension doesn't match
    pub fn set_acceleration_at(&mut self, index: usize, acceleration: DVector<f64>) {
        let accs = self
            .accelerations
            .as_mut()
            .expect("Acceleration storage not enabled");
        let acc_dim = self.acceleration_dimension.unwrap();
        if acceleration.len() != acc_dim {
            panic!(
                "Acceleration dimension {} does not match trajectory acceleration dimension {}",
                acceleration.len(),
                acc_dim
            );
        }
        accs[index] = acceleration;
    }

    /// Adds a state with acceleration to the trajectory.
    ///
    /// If acceleration storage is not yet enabled, enables it with the dimension
    /// of the provided acceleration vector.
    ///
    /// # Arguments
    /// * `epoch` - Time epoch
    /// * `state` - State vector
    /// * `acceleration` - Acceleration vector
    pub fn add_with_acceleration(
        &mut self,
        epoch: Epoch,
        state: DVector<f64>,
        acceleration: DVector<f64>,
    ) {
        self.add_full(epoch, state, None, None, None, Some(acceleration));
    }
}

impl Default for DOrbitTrajectory {
    /// Creates a default orbital trajectory in ECI Cartesian with 6D states.
    fn default() -> Self {
        Self::new(
            6, // dimension: standard 6D orbital states
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None, // angle_format is None for Cartesian
        )
    }
}

/// Index implementation returns state vector at given index
///
/// # Panics
/// Panics if index is out of bounds
impl std::ops::Index<usize> for DOrbitTrajectory {
    type Output = DVector<f64>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.states[index]
    }
}

/// Iterator over trajectory (epoch, state) pairs
pub struct DOrbitTrajectoryIterator<'a> {
    trajectory: &'a DOrbitTrajectory,
    index: usize,
}

impl<'a> Iterator for DOrbitTrajectoryIterator<'a> {
    type Item = (Epoch, DVector<f64>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.trajectory.len() {
            let result = self.trajectory.get(self.index).ok();
            self.index += 1;
            result
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.trajectory.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for DOrbitTrajectoryIterator<'a> {
    fn len(&self) -> usize {
        self.trajectory.len() - self.index
    }
}

/// IntoIterator implementation for iterating over (epoch, state) pairs
impl<'a> IntoIterator for &'a DOrbitTrajectory {
    type Item = (Epoch, DVector<f64>);
    type IntoIter = DOrbitTrajectoryIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        DOrbitTrajectoryIterator {
            trajectory: self,
            index: 0,
        }
    }
}

// Passthrough implementations for Trajectory trait
impl Trajectory for DOrbitTrajectory {
    type StateVector = DVector<f64>;

    /// Create trajectory from data. Assumes all data is in ECI Cartesian format.
    fn from_data(epochs: Vec<Epoch>, states: Vec<Self::StateVector>) -> Result<Self, BraheError> {
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

        // Infer dimension from first state
        let dimension = states[0].len();
        if dimension < 6 {
            return Err(BraheError::Error(format!(
                "State dimension must be at least 6 (position + velocity), got {}",
                dimension
            )));
        }

        // Validate all states have the same dimension
        for (i, state) in states.iter().enumerate() {
            if state.len() != dimension {
                return Err(BraheError::Error(format!(
                    "State {} has dimension {} but expected {} (inferred from first state)",
                    i,
                    state.len(),
                    dimension
                )));
            }
        }

        // Ensure epochs are sorted
        let mut indices: Vec<usize> = (0..epochs.len()).collect();
        indices.sort_by(|&i, &j| epochs[i].partial_cmp(&epochs[j]).unwrap());

        let sorted_epochs: Vec<Epoch> = indices.iter().map(|&i| epochs[i]).collect();
        let sorted_states: Vec<DVector<f64>> = indices.iter().map(|&i| states[i].clone()).collect();

        Ok(Self {
            epochs: sorted_epochs,
            states: sorted_states,
            covariances: None,
            stms: None,
            sensitivities: None,
            sensitivity_dimension: None,
            dimension,
            interpolation_method: InterpolationMethod::Linear, // Default to Linear
            covariance_interpolation_method: CovarianceInterpolationMethod::TwoWasserstein,
            eviction_policy: TrajectoryEvictionPolicy::None,
            max_size: None,
            max_age: None,
            frame: OrbitFrame::ECI, // Default to ECI Cartesian
            representation: OrbitRepresentation::Cartesian,
            angle_format: None, // angle_format is not meaningful for Cartesian
            name: None,
            id: None,
            uuid: None,
            metadata: HashMap::new(),
            accelerations: None,
            acceleration_dimension: None,
        })
    }

    fn add(&mut self, epoch: Epoch, state: Self::StateVector) {
        // Validate state dimension
        if state.len() != self.dimension {
            panic!(
                "State dimension {} does not match trajectory dimension {}",
                state.len(),
                self.dimension
            );
        }

        // Find the correct position to insert based on epoch
        // Insert after any existing states at the same epoch to support
        // impulsive maneuvers where we want both pre- and post-maneuver states
        let mut insert_idx = self.epochs.len();
        for (i, existing_epoch) in self.epochs.iter().enumerate() {
            if epoch < *existing_epoch {
                insert_idx = i;
                break;
            }
            // If epochs are equal, continue to find the position after all equal epochs
        }

        // Insert at the correct position
        self.epochs.insert(insert_idx, epoch);
        self.states.insert(insert_idx, state);

        // Insert placeholder covariance if storage is enabled
        if let Some(ref mut covs) = self.covariances {
            covs.insert(insert_idx, DMatrix::zeros(6, 6));
        }

        // Insert placeholder STM if storage is enabled
        if let Some(ref mut stms) = self.stms {
            stms.insert(insert_idx, DMatrix::identity(6, 6));
        }

        // Insert placeholder sensitivity if storage is enabled
        if let Some(ref mut sens) = self.sensitivities {
            let (_, param_dim) = self.sensitivity_dimension.unwrap();
            sens.insert(insert_idx, DMatrix::zeros(6, param_dim));
        }

        // Insert placeholder acceleration if storage is enabled
        if let Some(ref mut accs) = self.accelerations {
            let acc_dim = self.acceleration_dimension.unwrap();
            accs.insert(insert_idx, DVector::zeros(acc_dim));
        }

        // Apply eviction policy after adding state
        self.apply_eviction_policy();
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

            // Optimization: if we're past the epoch and moving away, we can stop
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
        if let Some(ref mut covs) = self.covariances {
            covs.clear();
        }
        if let Some(ref mut stms) = self.stms {
            stms.clear();
        }
        if let Some(ref mut sens) = self.sensitivities {
            sens.clear();
        }
        if let Some(ref mut accs) = self.accelerations {
            accs.clear();
        }
    }

    fn remove_epoch(&mut self, epoch: &Epoch) -> Result<Self::StateVector, BraheError> {
        // This could be improved with binary search since epochs are sorted
        if let Some(index) = self.epochs.iter().position(|e| e == epoch) {
            let removed_state = self.states.remove(index);
            self.epochs.remove(index);
            if let Some(ref mut covs) = self.covariances {
                covs.remove(index);
            }
            if let Some(ref mut stms) = self.stms {
                stms.remove(index);
            }
            if let Some(ref mut sens) = self.sensitivities {
                sens.remove(index);
            }
            if let Some(ref mut accs) = self.accelerations {
                accs.remove(index);
            }
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
        if let Some(ref mut covs) = self.covariances {
            covs.remove(index);
        }
        if let Some(ref mut stms) = self.stms {
            stms.remove(index);
        }
        if let Some(ref mut sens) = self.sensitivities {
            sens.remove(index);
        }
        if let Some(ref mut accs) = self.accelerations {
            accs.remove(index);
        }
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

        // If epoch is before the first state, error
        if epoch < &self.epochs[0] {
            return Err(BraheError::Error(
                "Epoch is before all states in trajectory".to_string(),
            ));
        }

        // Find the index at or before the epoch
        for i in (0..self.epochs.len()).rev() {
            if &self.epochs[i] <= epoch {
                return Ok(i);
            }
        }

        // Should never reach here given the checks above
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

        // If epoch is after the last state, error
        if epoch > self.epochs.last().unwrap() {
            return Err(BraheError::Error(
                "Epoch is after all states in trajectory".to_string(),
            ));
        }

        // Find the index at or after the epoch
        for i in 0..self.epochs.len() {
            if &self.epochs[i] >= epoch {
                return Ok(i);
            }
        }

        // Should never reach here given the checks above
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

// Implementation of InterpolationConfig trait
impl InterpolationConfig for DOrbitTrajectory {
    fn with_interpolation_method(mut self, method: InterpolationMethod) -> Self {
        self.interpolation_method = method;
        self
    }

    fn set_interpolation_method(&mut self, method: InterpolationMethod) {
        self.interpolation_method = method;
    }

    fn get_interpolation_method(&self) -> InterpolationMethod {
        self.interpolation_method
    }
}

impl InterpolatableTrajectory for DOrbitTrajectory {
    /// Interpolate state at a given epoch using the configured interpolation method.
    ///
    /// Overrides the default trait implementation to provide proper support for
    /// Lagrange and Hermite interpolation methods.
    ///
    /// # Arguments
    /// * `epoch` - Target epoch for interpolation
    ///
    /// # Returns
    /// * `Ok(state)` - Interpolated state vector
    /// * `Err(BraheError)` - If interpolation fails or epoch is out of range
    ///
    /// # Panics
    /// - HermiteCubic/HermiteQuintic panic if state dimension is not 6
    /// - HermiteQuintic panics if no accelerations stored AND fewer than 3 points
    fn interpolate(&self, epoch: &Epoch) -> Result<DVector<f64>, BraheError> {
        // Bounds checking
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

        // Get indices before and after the target epoch
        let idx1 = self.index_before_epoch(epoch)?;
        let idx2 = self.index_after_epoch(epoch)?;

        // If indices are the same, we have an exact match
        if idx1 == idx2 {
            return self.state_at_idx(idx1);
        }

        // Validate minimum point count
        let method = self.get_interpolation_method();
        let required = method.min_points_required();
        if self.len() < required {
            return Err(BraheError::Error(format!(
                "{:?} requires {} points, trajectory has {}",
                method,
                required,
                self.len()
            )));
        }

        // Get reference epoch for time calculations
        let ref_epoch = self.start_epoch().unwrap();

        match method {
            InterpolationMethod::Linear => self.interpolate_linear(epoch),

            InterpolationMethod::Lagrange { degree } => {
                // Collect degree+1 points centered around query epoch
                let n_points = degree + 1;
                let (start_idx, end_idx) =
                    self.compute_interpolation_window(idx1, idx2, n_points)?;

                // Build time and value arrays
                let times: Vec<f64> = (start_idx..=end_idx)
                    .map(|i| self.epochs[i] - ref_epoch)
                    .collect();
                let values: Vec<DVector<f64>> = (start_idx..=end_idx)
                    .map(|i| self.states[i].clone())
                    .collect();

                let t = *epoch - ref_epoch;
                Ok(interpolate_lagrange_dvector(&times, &values, t))
            }

            InterpolationMethod::HermiteCubic => {
                // Validate 6D state
                if self.dimension != 6 {
                    return Err(BraheError::Error(format!(
                        "HermiteCubic interpolation requires 6D states, trajectory has {}D. \
                         Use Linear/Lagrange interpolation for non-6D states.",
                        self.dimension
                    )));
                }

                // Get bracketing states
                let t0 = self.epochs[idx1] - ref_epoch;
                let t1 = self.epochs[idx2] - ref_epoch;
                let t = *epoch - ref_epoch;

                Ok(interpolate_hermite_cubic_dvector6(
                    t0,
                    t1,
                    &self.states[idx1],
                    &self.states[idx2],
                    t,
                ))
            }

            InterpolationMethod::HermiteQuintic => {
                // Validate 6D state
                if self.dimension != 6 {
                    return Err(BraheError::Error(format!(
                        "HermiteQuintic interpolation requires 6D states, trajectory has {}D. \
                         Use Linear/Lagrange interpolation for non-6D states.",
                        self.dimension
                    )));
                }

                // Check if we have stored accelerations
                if let Some(ref accs) = self.accelerations {
                    // Use explicit accelerations
                    let t0 = self.epochs[idx1] - ref_epoch;
                    let t1 = self.epochs[idx2] - ref_epoch;
                    let t = *epoch - ref_epoch;

                    Ok(interpolate_hermite_quintic_dvector6(
                        t0,
                        t1,
                        &self.states[idx1],
                        &self.states[idx2],
                        &accs[idx1],
                        &accs[idx2],
                        t,
                    ))
                } else if self.len() >= 3 {
                    // Use finite difference approximation
                    let (i0, i1, i2) = self.compute_fd_window(idx1, idx2)?;

                    let times = [
                        self.epochs[i0] - ref_epoch,
                        self.epochs[i1] - ref_epoch,
                        self.epochs[i2] - ref_epoch,
                    ];
                    let states = [
                        self.states[i0].clone(),
                        self.states[i1].clone(),
                        self.states[i2].clone(),
                    ];
                    let t = *epoch - ref_epoch;

                    Ok(interpolate_hermite_quintic_fd_dvector6(&times, &states, t))
                } else {
                    Err(BraheError::Error(format!(
                        "HermiteQuintic interpolation requires either stored accelerations \
                         or at least 3 trajectory points. This trajectory has {} points \
                         and no stored accelerations.",
                        self.len()
                    )))
                }
            }
        }
    }
}

impl DOrbitTrajectory {
    /// Compute the window of indices to use for Lagrange interpolation.
    fn compute_interpolation_window(
        &self,
        idx1: usize,
        idx2: usize,
        n_points: usize,
    ) -> Result<(usize, usize), BraheError> {
        if self.len() < n_points {
            return Err(BraheError::Error(format!(
                "Need {} points for interpolation, trajectory has {}",
                n_points,
                self.len()
            )));
        }

        let center = (idx1 + idx2) / 2;
        let half_window = n_points / 2;
        let mut start_idx = center.saturating_sub(half_window);
        let mut end_idx = start_idx + n_points - 1;

        if end_idx >= self.len() {
            end_idx = self.len() - 1;
            start_idx = end_idx.saturating_sub(n_points - 1);
        }

        Ok((start_idx, end_idx))
    }

    /// Compute the 3-point window for finite difference acceleration estimation.
    fn compute_fd_window(
        &self,
        idx1: usize,
        idx2: usize,
    ) -> Result<(usize, usize, usize), BraheError> {
        if idx1 > 0 {
            Ok((idx1 - 1, idx1, idx2))
        } else if idx2 + 1 < self.len() {
            Ok((idx1, idx2, idx2 + 1))
        } else {
            Err(BraheError::Error(
                "Cannot compute finite difference window with available points".to_string(),
            ))
        }
    }
}

impl STMStorage for DOrbitTrajectory {
    fn enable_stm_storage(&mut self) {
        if self.stms.is_none() {
            // Initialize with identity matrices for all existing states
            let identity = DMatrix::identity(6, 6);
            self.stms = Some(vec![identity; self.states.len()]);
        }
    }

    fn stm_at_idx(&self, index: usize) -> Option<&DMatrix<f64>> {
        self.stms.as_ref()?.get(index)
    }

    fn set_stm_at(&mut self, index: usize, stm: DMatrix<f64>) {
        if index >= self.states.len() {
            panic!(
                "Index {} out of bounds for trajectory with {} states",
                index,
                self.states.len()
            );
        }
        if stm.nrows() != 6 || stm.ncols() != 6 {
            panic!(
                "STM dimensions {}x{} do not match expected 6x6",
                stm.nrows(),
                stm.ncols()
            );
        }

        // Enable STM storage if not already enabled
        if self.stms.is_none() {
            self.enable_stm_storage();
        }

        if let Some(ref mut stms) = self.stms {
            stms[index] = stm;
        }
    }

    fn stm_dimensions(&self) -> (usize, usize) {
        (6, 6)
    }

    fn stm_storage(&self) -> Option<&Vec<DMatrix<f64>>> {
        self.stms.as_ref()
    }

    fn stm_storage_mut(&mut self) -> Option<&mut Vec<DMatrix<f64>>> {
        self.stms.as_mut()
    }

    // stm_at() uses default trait implementation
}

impl SensitivityStorage for DOrbitTrajectory {
    fn enable_sensitivity_storage(&mut self, param_dim: usize) {
        if param_dim == 0 {
            panic!("Parameter dimension must be > 0");
        }
        if self.sensitivities.is_none() {
            let zero_sens = DMatrix::zeros(6, param_dim);
            self.sensitivities = Some(vec![zero_sens; self.states.len()]);
            self.sensitivity_dimension = Some((6, param_dim));
        }
    }

    fn sensitivity_at_idx(&self, index: usize) -> Option<&DMatrix<f64>> {
        self.sensitivities.as_ref()?.get(index)
    }

    fn set_sensitivity_at(&mut self, index: usize, sensitivity: DMatrix<f64>) {
        if index >= self.states.len() {
            panic!(
                "Index {} out of bounds for trajectory with {} states",
                index,
                self.states.len()
            );
        }
        if sensitivity.nrows() != 6 {
            panic!(
                "Sensitivity row count {} does not match state dimension 6",
                sensitivity.nrows()
            );
        }

        // Check consistency with existing sensitivity dimension
        if let Some((_, existing_cols)) = self.sensitivity_dimension
            && sensitivity.ncols() != existing_cols
        {
            panic!(
                "Sensitivity column count {} does not match existing {}",
                sensitivity.ncols(),
                existing_cols
            );
        }

        // Enable sensitivity storage if not already enabled
        if self.sensitivities.is_none() {
            self.enable_sensitivity_storage(sensitivity.ncols());
        }

        if let Some(ref mut sens) = self.sensitivities {
            sens[index] = sensitivity;
        }
    }

    fn sensitivity_dimensions(&self) -> Option<(usize, usize)> {
        self.sensitivity_dimension
    }

    fn sensitivity_storage(&self) -> Option<&Vec<DMatrix<f64>>> {
        self.sensitivities.as_ref()
    }

    fn sensitivity_storage_mut(&mut self) -> Option<&mut Vec<DMatrix<f64>>> {
        self.sensitivities.as_mut()
    }

    // sensitivity_at() uses default trait implementation
}

// Implementation of CovarianceInterpolationConfig trait
impl CovarianceInterpolationConfig for DOrbitTrajectory {
    fn with_covariance_interpolation_method(
        mut self,
        method: CovarianceInterpolationMethod,
    ) -> Self {
        self.covariance_interpolation_method = method;
        self
    }

    fn set_covariance_interpolation_method(&mut self, method: CovarianceInterpolationMethod) {
        self.covariance_interpolation_method = method;
    }

    fn get_covariance_interpolation_method(&self) -> CovarianceInterpolationMethod {
        self.covariance_interpolation_method
    }
}

// Orbital-specific methods (matching OrbitalTrajectory trait methods)
// These are provided as inherent methods rather than trait implementations
// because the OrbitalTrajectory trait uses static types (SMatrix).
impl DOrbitTrajectory {
    /// Create orbital trajectory from data with specified orbital properties.
    ///
    /// # Arguments
    /// * `epochs` - Vector of epochs
    /// * `states` - Vector of state vectors
    /// * `frame` - Reference frame
    /// * `representation` - State representation (Cartesian or Keplerian)
    /// * `angle_format` - Angle format (None for Cartesian, Radians/Degrees for Keplerian)
    /// * `covariances` - Optional vector of 6x6 covariance matrices corresponding to states
    ///
    /// # Returns
    /// * `Ok(DOrbitTrajectory)` - New orbital trajectory with data
    /// * `Err(BraheError)` - If parameters are invalid or data validation fails
    ///
    /// # Panics
    /// * If covariances are provided but frame is not ECI or GCRF
    /// * If covariances length does not match states length
    pub fn from_orbital_data(
        epochs: Vec<Epoch>,
        states: Vec<DVector<f64>>,
        frame: OrbitFrame,
        representation: OrbitRepresentation,
        angle_format: Option<AngleFormat>,
        covariances: Option<Vec<DMatrix<f64>>>,
    ) -> Self {
        // Validate inputs
        if states.is_empty() {
            panic!("Cannot create trajectory from empty states");
        }

        // Infer dimension from first state
        let dimension = states[0].len();
        if dimension < 6 {
            panic!(
                "State dimension must be at least 6 (position + velocity), got {}",
                dimension
            );
        }

        // Validate all states have the same dimension
        for (i, state) in states.iter().enumerate() {
            if state.len() != dimension {
                panic!(
                    "State {} has dimension {} but expected {} (inferred from first state)",
                    i,
                    state.len(),
                    dimension
                );
            }
        }

        if frame == OrbitFrame::ECEF && representation == OrbitRepresentation::Keplerian {
            panic!("Keplerian elements should be in ECI frame");
        }

        // Validate covariances if provided
        if let Some(ref covs) = covariances {
            // Check that covariances length matches states length
            if covs.len() != states.len() {
                panic!(
                    "Covariances length ({}) must match states length ({})",
                    covs.len(),
                    states.len()
                );
            }

            // Check that frame is ECI, GCRF, or EME2000
            if frame != OrbitFrame::ECI && frame != OrbitFrame::GCRF && frame != OrbitFrame::EME2000
            {
                panic!(
                    "Covariances are only supported for ECI, GCRF, and EME2000 frames. Got: {}",
                    frame
                );
            }
        }

        // Note: angle_format is only meaningful for Keplerian representation
        // For Cartesian representation, the angle_format field should be None

        Self {
            epochs,
            states,
            covariances,
            stms: None,
            sensitivities: None,
            sensitivity_dimension: None,
            dimension,
            interpolation_method: InterpolationMethod::Linear,
            covariance_interpolation_method: CovarianceInterpolationMethod::TwoWasserstein,
            eviction_policy: TrajectoryEvictionPolicy::None,
            max_size: None,
            max_age: None,
            frame,
            representation,
            angle_format,
            name: None,
            id: None,
            uuid: None,
            metadata: HashMap::new(),
            accelerations: None,
            acceleration_dimension: None,
        }
    }

    /// Convert trajectory to ECI (Earth-Centered Inertial) frame with Cartesian representation.
    ///
    /// Converts all states to ECI frame and Cartesian representation.
    /// For Keplerian inputs, converts to Cartesian first.
    /// For ECEF/ITRF frames, uses epoch-dependent transformation.
    ///
    /// For extended states (dimension > 6), only the first 6 elements (orbital state)
    /// are converted. Additional elements (6+) are preserved unchanged.
    ///
    /// # Returns
    /// New trajectory in ECI frame with Cartesian states, preserving dimension.
    pub fn to_eci(&self) -> Self {
        let states_converted: Vec<DVector<f64>> = match self.representation {
            OrbitRepresentation::Keplerian => {
                let mut states_converted = Vec::with_capacity(self.states.len());
                let angle_fmt = self
                    .angle_format
                    .expect("Keplerian representation must have angle_format");
                // Convert Keplerian to Cartesian (first 6 elements only)
                for (_e, s) in self.into_iter() {
                    let converted = self.convert_orbital_preserving_additional(&s, |orbital| {
                        state_koe_to_eci(orbital, angle_fmt)
                    });
                    states_converted.push(converted);
                }
                states_converted
            }
            OrbitRepresentation::Cartesian => {
                match self.frame {
                    OrbitFrame::EME2000 => {
                        let mut states_converted = Vec::with_capacity(self.states.len());
                        // EME2000 Cartesian to GCRF Cartesian (no epoch needed)
                        for (_e, s) in self.into_iter() {
                            let converted = self
                                .convert_orbital_preserving_additional(&s, |orbital| {
                                    state_eme2000_to_gcrf(orbital)
                                });
                            states_converted.push(converted);
                        }
                        states_converted
                    }
                    OrbitFrame::ITRF | OrbitFrame::ECEF => {
                        let mut states_converted = Vec::with_capacity(self.states.len());
                        // ITRF/ECEF Cartesian to GCRF Cartesian (requires epoch)
                        for (e, s) in self.into_iter() {
                            let converted = self
                                .convert_orbital_preserving_additional(&s, |orbital| {
                                    state_itrf_to_gcrf(e, orbital)
                                });
                            states_converted.push(converted);
                        }
                        states_converted
                    }
                    OrbitFrame::ECI | OrbitFrame::GCRF => {
                        // No conversion needed
                        self.states.clone()
                    }
                }
            }
        };

        Self {
            epochs: self.epochs.clone(),
            states: states_converted,
            covariances: None,   // Covariances are dropped during frame conversions
            stms: None,          // STMs are dropped during frame conversions
            sensitivities: None, // Sensitivities are dropped during frame conversions
            sensitivity_dimension: None,
            dimension: self.dimension, // Preserve dimension
            interpolation_method: self.interpolation_method,
            covariance_interpolation_method: self.covariance_interpolation_method,
            eviction_policy: self.eviction_policy,
            max_size: self.max_size,
            max_age: self.max_age,
            frame: OrbitFrame::ECI,
            representation: OrbitRepresentation::Cartesian,
            angle_format: None,
            name: self.name.clone(),
            id: self.id,
            uuid: self.uuid,
            metadata: self.metadata.clone(),
            accelerations: None, // Accelerations are dropped during frame conversions
            acceleration_dimension: None,
        }
    }

    /// Convert trajectory to GCRF (Geocentric Celestial Reference Frame) with Cartesian representation.
    ///
    /// Converts all states to GCRF frame and Cartesian representation.
    /// For Keplerian inputs, converts to Cartesian first.
    /// For ECEF/ITRF frames, uses epoch-dependent transformation.
    ///
    /// For extended states (dimension > 6), only the first 6 elements (orbital state)
    /// are converted. Additional elements (6+) are preserved unchanged.
    ///
    /// # Returns
    /// New trajectory in GCRF frame with Cartesian states, preserving dimension.
    pub fn to_gcrf(&self) -> Self {
        let states_converted: Vec<DVector<f64>> = match self.representation {
            OrbitRepresentation::Keplerian => {
                let mut states_converted = Vec::with_capacity(self.states.len());
                let angle_fmt = self
                    .angle_format
                    .expect("Keplerian representation must have angle_format");
                // Convert Keplerian to Cartesian (first 6 elements only)
                for (_e, s) in self.into_iter() {
                    let converted = self.convert_orbital_preserving_additional(&s, |orbital| {
                        state_koe_to_eci(orbital, angle_fmt)
                    });
                    states_converted.push(converted);
                }
                states_converted
            }
            OrbitRepresentation::Cartesian => {
                match self.frame {
                    OrbitFrame::EME2000 => {
                        let mut states_converted = Vec::with_capacity(self.states.len());
                        // EME2000 Cartesian to GCRF Cartesian (no epoch needed)
                        for (_e, s) in self.into_iter() {
                            let converted = self
                                .convert_orbital_preserving_additional(&s, |orbital| {
                                    state_eme2000_to_gcrf(orbital)
                                });
                            states_converted.push(converted);
                        }
                        states_converted
                    }
                    OrbitFrame::ITRF | OrbitFrame::ECEF => {
                        let mut states_converted = Vec::with_capacity(self.states.len());
                        // ITRF/ECEF Cartesian to GCRF Cartesian (requires epoch)
                        for (e, s) in self.into_iter() {
                            let converted = self
                                .convert_orbital_preserving_additional(&s, |orbital| {
                                    state_itrf_to_gcrf(e, orbital)
                                });
                            states_converted.push(converted);
                        }
                        states_converted
                    }
                    OrbitFrame::ECI | OrbitFrame::GCRF => {
                        // No conversion needed
                        self.states.clone()
                    }
                }
            }
        };

        Self {
            epochs: self.epochs.clone(),
            states: states_converted,
            covariances: None,   // Covariances are dropped during frame conversions
            stms: None,          // STMs are dropped during frame conversions
            sensitivities: None, // Sensitivities are dropped during frame conversions
            sensitivity_dimension: None,
            dimension: self.dimension, // Preserve dimension
            interpolation_method: self.interpolation_method,
            covariance_interpolation_method: self.covariance_interpolation_method,
            eviction_policy: self.eviction_policy,
            max_size: self.max_size,
            max_age: self.max_age,
            frame: OrbitFrame::GCRF,
            representation: OrbitRepresentation::Cartesian,
            angle_format: None,
            name: self.name.clone(),
            id: self.id,
            uuid: self.uuid,
            metadata: self.metadata.clone(),
            accelerations: None, // Accelerations are dropped during frame conversions
            acceleration_dimension: None,
        }
    }

    /// Convert trajectory to ECEF (Earth-Centered Earth-Fixed) frame with Cartesian representation.
    ///
    /// Converts all states to ECEF frame and Cartesian representation.
    /// For Keplerian inputs, converts to Cartesian first.
    /// For ECI/GCRF/EME2000 frames, uses epoch-dependent transformation.
    ///
    /// For extended states (dimension > 6), only the first 6 elements (orbital state)
    /// are converted. Additional elements (6+) are preserved unchanged.
    ///
    /// # Returns
    /// New trajectory in ECEF frame with Cartesian states, preserving dimension.
    pub fn to_ecef(&self) -> Self {
        let states_converted: Vec<DVector<f64>> = match self.representation {
            OrbitRepresentation::Keplerian => {
                let mut states_converted = Vec::with_capacity(self.states.len());
                let angle_fmt = self
                    .angle_format
                    .expect("Keplerian representation must have angle_format");
                // Convert Keplerian to Cartesian ECI, then to ECEF
                for (e, s) in self.into_iter() {
                    let converted = self.convert_orbital_preserving_additional(&s, |orbital| {
                        let state_eci = state_koe_to_eci(orbital, angle_fmt);
                        state_eci_to_ecef(e, state_eci)
                    });
                    states_converted.push(converted);
                }
                states_converted
            }
            OrbitRepresentation::Cartesian => {
                match self.frame {
                    OrbitFrame::EME2000 => {
                        let mut states_converted = Vec::with_capacity(self.states.len());
                        // EME2000 -> GCRF -> ITRF
                        for (e, s) in self.into_iter() {
                            let converted =
                                self.convert_orbital_preserving_additional(&s, |orbital| {
                                    let state_gcrf = state_eme2000_to_gcrf(orbital);
                                    state_gcrf_to_itrf(e, state_gcrf)
                                });
                            states_converted.push(converted);
                        }
                        states_converted
                    }
                    OrbitFrame::ITRF | OrbitFrame::ECEF => {
                        // Already in ITRF frame
                        self.states.clone()
                    }
                    OrbitFrame::ECI | OrbitFrame::GCRF => {
                        let mut states_converted = Vec::with_capacity(self.states.len());
                        // GCRF/ECI to ITRF
                        for (e, s) in self.into_iter() {
                            let converted = self
                                .convert_orbital_preserving_additional(&s, |orbital| {
                                    state_gcrf_to_itrf(e, orbital)
                                });
                            states_converted.push(converted);
                        }
                        states_converted
                    }
                }
            }
        };

        Self {
            epochs: self.epochs.clone(),
            states: states_converted,
            covariances: None,   // Covariances are dropped during frame conversions
            stms: None,          // STMs are dropped during frame conversions
            sensitivities: None, // Sensitivities are dropped during frame conversions
            sensitivity_dimension: None,
            dimension: self.dimension, // Preserve dimension
            interpolation_method: self.interpolation_method,
            covariance_interpolation_method: self.covariance_interpolation_method,
            eviction_policy: self.eviction_policy,
            max_size: self.max_size,
            max_age: self.max_age,
            frame: OrbitFrame::ECEF,
            representation: OrbitRepresentation::Cartesian,
            angle_format: None,
            name: self.name.clone(),
            id: self.id,
            uuid: self.uuid,
            metadata: self.metadata.clone(),
            accelerations: None, // Accelerations are dropped during frame conversions
            acceleration_dimension: None,
        }
    }

    /// Convert trajectory to ITRF (International Terrestrial Reference Frame) with Cartesian representation.
    ///
    /// Converts all states to ITRF frame and Cartesian representation.
    /// For Keplerian inputs, converts to Cartesian first.
    /// For ECI/GCRF/EME2000 frames, uses epoch-dependent transformation.
    ///
    /// For extended states (dimension > 6), only the first 6 elements (orbital state)
    /// are converted. Additional elements (6+) are preserved unchanged.
    ///
    /// # Returns
    /// New trajectory in ITRF frame with Cartesian states, preserving dimension.
    pub fn to_itrf(&self) -> Self {
        let states_converted: Vec<DVector<f64>> = match self.representation {
            OrbitRepresentation::Keplerian => {
                let mut states_converted = Vec::with_capacity(self.states.len());
                let angle_fmt = self
                    .angle_format
                    .expect("Keplerian representation must have angle_format");
                // Keplerian to Cartesian (in GCRF/ECI), then GCRF to ITRF
                for (e, s) in self.into_iter() {
                    let converted = self.convert_orbital_preserving_additional(&s, |orbital| {
                        let state_cartesian = state_koe_to_eci(orbital, angle_fmt);
                        state_gcrf_to_itrf(e, state_cartesian)
                    });
                    states_converted.push(converted);
                }
                states_converted
            }
            OrbitRepresentation::Cartesian => {
                match self.frame {
                    OrbitFrame::EME2000 => {
                        let mut states_converted = Vec::with_capacity(self.states.len());
                        // EME2000 -> GCRF -> ITRF
                        for (e, s) in self.into_iter() {
                            let converted =
                                self.convert_orbital_preserving_additional(&s, |orbital| {
                                    let state_gcrf = state_eme2000_to_gcrf(orbital);
                                    state_gcrf_to_itrf(e, state_gcrf)
                                });
                            states_converted.push(converted);
                        }
                        states_converted
                    }
                    OrbitFrame::ITRF | OrbitFrame::ECEF => {
                        // Already in ITRF frame
                        self.states.clone()
                    }
                    OrbitFrame::ECI | OrbitFrame::GCRF => {
                        let mut states_converted = Vec::with_capacity(self.states.len());
                        // GCRF/ECI to ITRF
                        for (e, s) in self.into_iter() {
                            let converted = self
                                .convert_orbital_preserving_additional(&s, |orbital| {
                                    state_gcrf_to_itrf(e, orbital)
                                });
                            states_converted.push(converted);
                        }
                        states_converted
                    }
                }
            }
        };

        Self {
            epochs: self.epochs.clone(),
            states: states_converted,
            covariances: None,   // Covariances are dropped during frame conversions
            stms: None,          // STMs are dropped during frame conversions
            sensitivities: None, // Sensitivities are dropped during frame conversions
            sensitivity_dimension: None,
            dimension: self.dimension, // Preserve dimension
            interpolation_method: self.interpolation_method,
            covariance_interpolation_method: self.covariance_interpolation_method,
            eviction_policy: self.eviction_policy,
            max_size: self.max_size,
            max_age: self.max_age,
            frame: OrbitFrame::ITRF,
            representation: OrbitRepresentation::Cartesian,
            angle_format: None,
            name: self.name.clone(),
            id: self.id,
            uuid: self.uuid,
            metadata: self.metadata.clone(),
            accelerations: None, // Accelerations are dropped during frame conversions
            acceleration_dimension: None,
        }
    }

    /// Convert trajectory to EME2000 (Earth Mean Equator and Equinox of J2000) frame with Cartesian representation.
    ///
    /// Converts all states to EME2000 frame and Cartesian representation.
    /// For Keplerian inputs, converts to Cartesian first.
    /// For ECEF/ITRF frames, uses epoch-dependent transformation to GCRF first.
    ///
    /// For extended states (dimension > 6), only the first 6 elements (orbital state)
    /// are converted. Additional elements (6+) are preserved unchanged.
    ///
    /// # Returns
    /// New trajectory in EME2000 frame with Cartesian states, preserving dimension.
    pub fn to_eme2000(&self) -> Self {
        let states_converted: Vec<DVector<f64>> = match self.representation {
            OrbitRepresentation::Keplerian => {
                let mut states_converted = Vec::with_capacity(self.states.len());
                let angle_fmt = self
                    .angle_format
                    .expect("Keplerian representation must have angle_format");
                // Keplerian to Cartesian GCRF, then to EME2000
                for (_e, s) in self.into_iter() {
                    let converted = self.convert_orbital_preserving_additional(&s, |orbital| {
                        let state_cartesian = state_koe_to_eci(orbital, angle_fmt);
                        state_gcrf_to_eme2000(state_cartesian)
                    });
                    states_converted.push(converted);
                }
                states_converted
            }
            OrbitRepresentation::Cartesian => {
                match self.frame {
                    OrbitFrame::EME2000 => {
                        // Already in EME2000 frame
                        self.states.clone()
                    }
                    OrbitFrame::ITRF | OrbitFrame::ECEF => {
                        let mut states_converted = Vec::with_capacity(self.states.len());
                        // ITRF/ECEF -> GCRF -> EME2000
                        for (e, s) in self.into_iter() {
                            let converted =
                                self.convert_orbital_preserving_additional(&s, |orbital| {
                                    let state_gcrf = state_itrf_to_gcrf(e, orbital);
                                    state_gcrf_to_eme2000(state_gcrf)
                                });
                            states_converted.push(converted);
                        }
                        states_converted
                    }
                    OrbitFrame::ECI | OrbitFrame::GCRF => {
                        let mut states_converted = Vec::with_capacity(self.states.len());
                        // ECI/GCRF to EME2000
                        for (_e, s) in self.into_iter() {
                            let converted = self
                                .convert_orbital_preserving_additional(&s, |orbital| {
                                    state_gcrf_to_eme2000(orbital)
                                });
                            states_converted.push(converted);
                        }
                        states_converted
                    }
                }
            }
        };

        Self {
            epochs: self.epochs.clone(),
            states: states_converted,
            covariances: None,   // Covariances are dropped during frame conversions
            stms: None,          // STMs are dropped during frame conversions
            sensitivities: None, // Sensitivities are dropped during frame conversions
            sensitivity_dimension: None,
            dimension: self.dimension, // Preserve dimension
            interpolation_method: self.interpolation_method,
            covariance_interpolation_method: self.covariance_interpolation_method,
            eviction_policy: self.eviction_policy,
            max_size: self.max_size,
            max_age: self.max_age,
            frame: OrbitFrame::EME2000,
            representation: OrbitRepresentation::Cartesian,
            angle_format: None,
            name: self.name.clone(),
            id: self.id,
            uuid: self.uuid,
            metadata: self.metadata.clone(),
            accelerations: None, // Accelerations are dropped during frame conversions
            acceleration_dimension: None,
        }
    }

    /// Convert trajectory to Keplerian orbital elements representation.
    ///
    /// Converts all states to Keplerian elements [a, e, i, raan, argp, anomaly]
    /// in the current frame. For Cartesian inputs, uses two-body conversion.
    /// For Keplerian inputs with different angle format, converts angles.
    ///
    /// For extended states (dimension > 6), only the first 6 elements (orbital state)
    /// are converted. Additional elements (6+) are preserved unchanged.
    ///
    /// # Arguments
    /// * `angle_format` - Desired angle format (Radians or Degrees) for output elements
    ///
    /// # Returns
    /// New trajectory with Keplerian representation in specified angle format, preserving dimension.
    pub fn to_keplerian(&self, angle_format: AngleFormat) -> Self {
        let states_converted: Vec<DVector<f64>> = match self.representation {
            OrbitRepresentation::Keplerian => {
                match self.angle_format {
                    Some(current_format) if current_format == angle_format => {
                        // Already in desired format
                        self.states.clone()
                    }
                    Some(current_format) => {
                        let mut states_converted = Vec::with_capacity(self.states.len());
                        let conversion_factor = if current_format == AngleFormat::Degrees
                            && angle_format == AngleFormat::Radians
                        {
                            DEG2RAD
                        } else if current_format == AngleFormat::Radians
                            && angle_format == AngleFormat::Degrees
                        {
                            RAD2DEG
                        } else {
                            1.0
                        };

                        // Convert angles (elements 2-5 only, preserve additional states)
                        for (_e, s) in self.into_iter() {
                            let mut state_converted = s.clone();
                            state_converted[2] *= conversion_factor;
                            state_converted[3] *= conversion_factor;
                            state_converted[4] *= conversion_factor;
                            state_converted[5] *= conversion_factor;
                            // Additional states (6+) already preserved in clone
                            states_converted.push(state_converted);
                        }
                        states_converted
                    }
                    None => {
                        panic!(
                            "Current Keplerian representation missing required field angle_format"
                        );
                    }
                }
            }
            OrbitRepresentation::Cartesian => {
                match self.frame {
                    OrbitFrame::EME2000 => {
                        let mut states_converted = Vec::with_capacity(self.states.len());
                        // EME2000 -> GCRF -> Keplerian
                        for (_e, s) in self.into_iter() {
                            let converted =
                                self.convert_orbital_preserving_additional(&s, |orbital| {
                                    let state_gcrf = state_eme2000_to_gcrf(orbital);
                                    state_eci_to_koe(state_gcrf, angle_format)
                                });
                            states_converted.push(converted);
                        }
                        states_converted
                    }
                    OrbitFrame::ITRF | OrbitFrame::ECEF => {
                        let mut states_converted = Vec::with_capacity(self.states.len());
                        // ITRF/ECEF -> ECI -> Keplerian
                        for (e, s) in self.into_iter() {
                            let converted =
                                self.convert_orbital_preserving_additional(&s, |orbital| {
                                    let state_eci = state_ecef_to_eci(e, orbital);
                                    state_eci_to_koe(state_eci, angle_format)
                                });
                            states_converted.push(converted);
                        }
                        states_converted
                    }
                    OrbitFrame::ECI | OrbitFrame::GCRF => {
                        let mut states_converted = Vec::with_capacity(self.states.len());
                        // ECI/GCRF Cartesian -> Keplerian
                        for (_e, s) in self.into_iter() {
                            let converted = self
                                .convert_orbital_preserving_additional(&s, |orbital| {
                                    state_eci_to_koe(orbital, angle_format)
                                });
                            states_converted.push(converted);
                        }
                        states_converted
                    }
                }
            }
        };

        Self {
            epochs: self.epochs.clone(),
            states: states_converted,
            covariances: None, // Covariances are dropped during representation conversions
            stms: None,        // STMs are dropped during representation conversions
            sensitivities: None, // Sensitivities are dropped during representation conversions
            sensitivity_dimension: None,
            dimension: self.dimension, // Preserve dimension
            interpolation_method: self.interpolation_method,
            covariance_interpolation_method: self.covariance_interpolation_method,
            eviction_policy: self.eviction_policy,
            max_size: self.max_size,
            max_age: self.max_age,
            frame: OrbitFrame::ECI,
            representation: OrbitRepresentation::Keplerian,
            angle_format: Some(angle_format),
            name: self.name.clone(),
            id: self.id,
            uuid: self.uuid,
            metadata: self.metadata.clone(),
            accelerations: None, // Accelerations are dropped during representation conversions
            acceleration_dimension: None,
        }
    }
}

impl Identifiable for DOrbitTrajectory {
    fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    fn with_uuid(mut self, uuid: Uuid) -> Self {
        self.uuid = Some(uuid);
        self
    }

    fn with_new_uuid(mut self) -> Self {
        self.uuid = Some(Uuid::new_v4());
        self
    }

    fn with_id(mut self, id: u64) -> Self {
        self.id = Some(id);
        self
    }

    fn with_identity(mut self, name: Option<&str>, uuid: Option<Uuid>, id: Option<u64>) -> Self {
        self.name = name.map(|s| s.to_string());
        self.uuid = uuid;
        self.id = id;
        self
    }

    fn set_identity(&mut self, name: Option<&str>, uuid: Option<Uuid>, id: Option<u64>) {
        self.name = name.map(|s| s.to_string());
        self.uuid = uuid;
        self.id = id;
    }

    fn set_id(&mut self, id: Option<u64>) {
        self.id = id;
    }

    fn set_name(&mut self, name: Option<&str>) {
        self.name = name.map(|s| s.to_string());
    }

    fn generate_uuid(&mut self) {
        self.uuid = Some(Uuid::new_v4());
    }

    fn get_id(&self) -> Option<u64> {
        self.id
    }

    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn get_uuid(&self) -> Option<Uuid> {
        self.uuid
    }
}

// =============================================================================
// DStateProvider Trait
// =============================================================================

impl DStateProvider for DOrbitTrajectory {
    fn state(&self, epoch: Epoch) -> Result<DVector<f64>, BraheError> {
        // Delegate to existing interpolate method - returns FULL state (all dimensions)
        self.interpolate(&epoch)
    }

    fn state_dim(&self) -> usize {
        // Return actual dimension from first state (can be >6 for extended states)
        self.states.first().map(|s| s.len()).unwrap_or(6)
    }
}

// =============================================================================
// DCovarianceProvider Trait
// =============================================================================

impl DCovarianceProvider for DOrbitTrajectory {
    fn covariance(&self, epoch: Epoch) -> Result<DMatrix<f64>, BraheError> {
        // Check if covariance tracking is enabled
        if self.covariances.is_none() {
            return Err(BraheError::InitializationError(
                "Covariance not available: covariance tracking was not enabled for this trajectory"
                    .to_string(),
            ));
        }

        // Validate bounds
        if let Some(start) = self.start_epoch()
            && epoch < start
        {
            return Err(BraheError::OutOfBoundsError(format!(
                "Cannot get covariance at epoch {}: before trajectory start {}",
                epoch, start
            )));
        }
        if let Some(end) = self.end_epoch()
            && epoch > end
        {
            return Err(BraheError::OutOfBoundsError(format!(
                "Cannot get covariance at epoch {}: after trajectory end {}",
                epoch, end
            )));
        }

        // Delegate to existing method - returns FULL covariance matrix (all dimensions)
        self.covariance_at(epoch).ok_or_else(|| {
            BraheError::OutOfBoundsError(format!(
                "Cannot get covariance at epoch {}: no covariance data available",
                epoch
            ))
        })
    }

    fn covariance_dim(&self) -> usize {
        // Return actual dimension from first covariance (can be >6 for extended states)
        self.covariances
            .as_ref()
            .and_then(|covs| covs.first())
            .map(|c| c.nrows())
            .unwrap_or(6)
    }
}

// =============================================================================
// DOrbitStateProvider Trait
// =============================================================================

impl DOrbitStateProvider for DOrbitTrajectory {
    fn state_eci(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        // Get state in native format (full dimensions)
        let state_dvec = self.interpolate(&epoch)?;

        // Extract first 6 elements (orbital state only - ignore extended dimensions)
        let state = Vector6::from_iterator(state_dvec.iter().take(6).copied());

        Ok(match (self.frame, self.representation) {
            (OrbitFrame::ECI, OrbitRepresentation::Cartesian) => state,
            (OrbitFrame::GCRF, OrbitRepresentation::Cartesian) => state, // GCRF treated as ECI
            (OrbitFrame::ECI, OrbitRepresentation::Keplerian) => state_koe_to_eci(
                state,
                self.angle_format
                    .expect("Keplerian representation must have angle_format"),
            ),
            (OrbitFrame::GCRF, OrbitRepresentation::Keplerian) => state_koe_to_eci(
                state,
                self.angle_format
                    .expect("Keplerian representation must have angle_format"),
            ),
            (OrbitFrame::EME2000, OrbitRepresentation::Keplerian) => {
                state_eme2000_to_gcrf(state_koe_to_eci(
                    state,
                    self.angle_format
                        .expect("Keplerian representation must have angle_format"),
                ))
            }
            (OrbitFrame::ECEF, OrbitRepresentation::Cartesian) => state_ecef_to_eci(epoch, state),
            (OrbitFrame::ITRF, OrbitRepresentation::Cartesian) => state_itrf_to_gcrf(epoch, state),
            (OrbitFrame::EME2000, OrbitRepresentation::Cartesian) => state_eme2000_to_gcrf(state),
            (OrbitFrame::ECEF, OrbitRepresentation::Keplerian) => {
                return Err(BraheError::Error(
                    "Keplerian element trajectories should be in an inertial frame".to_string(),
                ));
            }
            (OrbitFrame::ITRF, OrbitRepresentation::Keplerian) => {
                return Err(BraheError::Error(
                    "Keplerian element trajectories should be in an inertial frame".to_string(),
                ));
            }
        })
    }

    fn state_gcrf(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        // Get state in native format (full dimensions)
        let state_dvec = self.interpolate(&epoch)?;

        // Extract first 6 elements (orbital state only - ignore extended dimensions)
        let state = Vector6::from_iterator(state_dvec.iter().take(6).copied());

        Ok(match (self.frame, self.representation) {
            (OrbitFrame::GCRF, OrbitRepresentation::Cartesian) => state,
            (OrbitFrame::ECI, OrbitRepresentation::Cartesian) => state, // ECI treated as GCRF
            (OrbitFrame::GCRF, OrbitRepresentation::Keplerian) => state_koe_to_eci(
                state,
                self.angle_format
                    .expect("Keplerian representation must have angle_format"),
            ),
            (OrbitFrame::ECI, OrbitRepresentation::Keplerian) => state_koe_to_eci(
                state,
                self.angle_format
                    .expect("Keplerian representation must have angle_format"),
            ),
            (OrbitFrame::EME2000, OrbitRepresentation::Keplerian) => {
                state_eme2000_to_gcrf(state_koe_to_eci(
                    state,
                    self.angle_format
                        .expect("Keplerian representation must have angle_format"),
                ))
            }
            (OrbitFrame::EME2000, OrbitRepresentation::Cartesian) => state_eme2000_to_gcrf(state),
            (OrbitFrame::ITRF, OrbitRepresentation::Cartesian) => state_itrf_to_gcrf(epoch, state),
            (OrbitFrame::ECEF, OrbitRepresentation::Cartesian) => state_itrf_to_gcrf(epoch, state),
            (OrbitFrame::ECEF, OrbitRepresentation::Keplerian) => {
                return Err(BraheError::Error(
                    "Keplerian element trajectories should be in an inertial frame".to_string(),
                ));
            }
            (OrbitFrame::ITRF, OrbitRepresentation::Keplerian) => {
                return Err(BraheError::Error(
                    "Keplerian element trajectories should be in an inertial frame".to_string(),
                ));
            }
        })
    }

    fn state_ecef(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        // Get state in native format (full dimensions)
        let state_dvec = self.interpolate(&epoch)?;

        // Extract first 6 elements (orbital state only - ignore extended dimensions)
        let state = Vector6::from_iterator(state_dvec.iter().take(6).copied());

        Ok(match (self.frame, self.representation) {
            (OrbitFrame::ECEF, OrbitRepresentation::Cartesian) => state,
            (OrbitFrame::ITRF, OrbitRepresentation::Cartesian) => state,
            (OrbitFrame::ECI, OrbitRepresentation::Cartesian) => state_eci_to_ecef(epoch, state),
            (OrbitFrame::GCRF, OrbitRepresentation::Cartesian) => state_gcrf_to_itrf(epoch, state),
            (OrbitFrame::EME2000, OrbitRepresentation::Cartesian) => {
                let state_gcrf = state_eme2000_to_gcrf(state);
                state_gcrf_to_itrf(epoch, state_gcrf)
            }
            (OrbitFrame::ECI, OrbitRepresentation::Keplerian) => {
                let state_eci_cart = state_koe_to_eci(
                    state,
                    self.angle_format
                        .expect("Keplerian representation must have angle_format"),
                );
                state_eci_to_ecef(epoch, state_eci_cart)
            }
            (OrbitFrame::EME2000, OrbitRepresentation::Keplerian) => {
                let state_eme2000_cart = state_koe_to_eci(
                    state,
                    self.angle_format
                        .expect("Keplerian representation must have angle_format"),
                );
                let state_gcrf = state_eme2000_to_gcrf(state_eme2000_cart);
                state_gcrf_to_itrf(epoch, state_gcrf)
            }
            (OrbitFrame::GCRF, OrbitRepresentation::Keplerian) => {
                let state_gcrf_cart = state_koe_to_eci(
                    state,
                    self.angle_format
                        .expect("Keplerian representation must have angle_format"),
                );
                state_gcrf_to_itrf(epoch, state_gcrf_cart)
            }
            (OrbitFrame::ECEF, OrbitRepresentation::Keplerian) => {
                return Err(BraheError::Error(
                    "Keplerian element trajectories should be in an inertial frame".to_string(),
                ));
            }
            (OrbitFrame::ITRF, OrbitRepresentation::Keplerian) => {
                return Err(BraheError::Error(
                    "Keplerian element trajectories should be in an inertial frame".to_string(),
                ));
            }
        })
    }

    fn state_itrf(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        // Get state in native format (full dimensions)
        let state_dvec = self.interpolate(&epoch)?;

        // Extract first 6 elements (orbital state only - ignore extended dimensions)
        let state = Vector6::from_iterator(state_dvec.iter().take(6).copied());

        Ok(match (self.frame, self.representation) {
            (OrbitFrame::ECEF, OrbitRepresentation::Cartesian) => state,
            (OrbitFrame::ITRF, OrbitRepresentation::Cartesian) => state,
            (OrbitFrame::ECI, OrbitRepresentation::Cartesian) => state_eci_to_ecef(epoch, state),
            (OrbitFrame::GCRF, OrbitRepresentation::Cartesian) => state_gcrf_to_itrf(epoch, state),
            (OrbitFrame::EME2000, OrbitRepresentation::Cartesian) => {
                let state_gcrf = state_eme2000_to_gcrf(state);
                state_gcrf_to_itrf(epoch, state_gcrf)
            }
            (OrbitFrame::ECI, OrbitRepresentation::Keplerian) => {
                let state_eci_cart = state_koe_to_eci(
                    state,
                    self.angle_format
                        .expect("Keplerian representation must have angle_format"),
                );
                state_eci_to_ecef(epoch, state_eci_cart)
            }
            (OrbitFrame::GCRF, OrbitRepresentation::Keplerian) => {
                let state_gcrf_cart = state_koe_to_eci(
                    state,
                    self.angle_format
                        .expect("Keplerian representation must have angle_format"),
                );
                state_gcrf_to_itrf(epoch, state_gcrf_cart)
            }
            (OrbitFrame::EME2000, OrbitRepresentation::Keplerian) => {
                let state_eme2000_cart = state_koe_to_eci(
                    state,
                    self.angle_format
                        .expect("Keplerian representation must have angle_format"),
                );
                let state_gcrf = state_eme2000_to_gcrf(state_eme2000_cart);
                state_gcrf_to_itrf(epoch, state_gcrf)
            }
            (OrbitFrame::ECEF, OrbitRepresentation::Keplerian) => {
                return Err(BraheError::Error(
                    "Keplerian element trajectories should be in an inertial frame".to_string(),
                ));
            }
            (OrbitFrame::ITRF, OrbitRepresentation::Keplerian) => {
                return Err(BraheError::Error(
                    "Keplerian element trajectories should be in an inertial frame".to_string(),
                ));
            }
        })
    }

    fn state_eme2000(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        // Get state in native format (full dimensions)
        let state_dvec = self.interpolate(&epoch)?;

        // Extract first 6 elements (orbital state only - ignore extended dimensions)
        let state = Vector6::from_iterator(state_dvec.iter().take(6).copied());

        Ok(match (self.frame, self.representation) {
            (OrbitFrame::EME2000, OrbitRepresentation::Cartesian) => state,
            (OrbitFrame::GCRF, OrbitRepresentation::Cartesian) => state_gcrf_to_eme2000(state),
            (OrbitFrame::ECI, OrbitRepresentation::Cartesian) => state_gcrf_to_eme2000(state), // ECI treated as GCRF
            (OrbitFrame::GCRF, OrbitRepresentation::Keplerian) => {
                let state_gcrf_cart = state_koe_to_eci(
                    state,
                    self.angle_format
                        .expect("Keplerian representation must have angle_format"),
                );
                state_gcrf_to_eme2000(state_gcrf_cart)
            }
            (OrbitFrame::ECI, OrbitRepresentation::Keplerian) => {
                let state_eci_cart = state_koe_to_eci(
                    state,
                    self.angle_format
                        .expect("Keplerian representation must have angle_format"),
                );
                state_gcrf_to_eme2000(state_eci_cart)
            }
            (OrbitFrame::EME2000, OrbitRepresentation::Keplerian) => state_koe_to_eci(
                state,
                self.angle_format
                    .expect("Keplerian representation must have angle_format"),
            ),
            (OrbitFrame::ITRF, OrbitRepresentation::Cartesian) => {
                let state_gcrf = state_itrf_to_gcrf(epoch, state);
                state_gcrf_to_eme2000(state_gcrf)
            }
            (OrbitFrame::ECEF, OrbitRepresentation::Cartesian) => {
                let state_gcrf = state_itrf_to_gcrf(epoch, state);
                state_gcrf_to_eme2000(state_gcrf)
            }
            (OrbitFrame::ECEF, OrbitRepresentation::Keplerian) => {
                return Err(BraheError::Error(
                    "Keplerian element trajectories should be in an inertial frame".to_string(),
                ));
            }
            (OrbitFrame::ITRF, OrbitRepresentation::Keplerian) => {
                return Err(BraheError::Error(
                    "Keplerian element trajectories should be in an inertial frame".to_string(),
                ));
            }
        })
    }

    fn state_koe_osc(
        &self,
        epoch: Epoch,
        angle_format: AngleFormat,
    ) -> Result<Vector6<f64>, BraheError> {
        // Get state in native format (full dimensions)
        let state_dvec = self.interpolate(&epoch)?;

        // Extract first 6 elements (orbital state only - ignore extended dimensions)
        let state = Vector6::from_iterator(state_dvec.iter().take(6).copied());

        Ok(match (self.frame, self.representation) {
            (OrbitFrame::ECI, OrbitRepresentation::Keplerian) => {
                // Already in Keplerian, just convert angle format if needed
                let native_format = self.angle_format.unwrap_or(AngleFormat::Radians);
                if native_format == angle_format {
                    state
                } else {
                    // Convert angles
                    let mut converted = state;
                    let factor = if angle_format == AngleFormat::Degrees {
                        RAD2DEG
                    } else {
                        DEG2RAD
                    };
                    converted[2] *= factor; // inclination
                    converted[3] *= factor; // RAAN
                    converted[4] *= factor; // arg periapsis
                    converted[5] *= factor; // mean anomaly
                    converted
                }
            }
            (OrbitFrame::GCRF, OrbitRepresentation::Keplerian) => {
                // Already in Keplerian, just convert angle format if needed
                let native_format = self.angle_format.unwrap_or(AngleFormat::Radians);
                if native_format == angle_format {
                    state
                } else {
                    // Convert angles
                    let mut converted = state;
                    let factor = if angle_format == AngleFormat::Degrees {
                        RAD2DEG
                    } else {
                        DEG2RAD
                    };
                    converted[2] *= factor; // inclination
                    converted[3] *= factor; // RAAN
                    converted[4] *= factor; // arg periapsis
                    converted[5] *= factor; // mean anomaly
                    converted
                }
            }
            (OrbitFrame::EME2000, OrbitRepresentation::Keplerian) => {
                // Convert back to Cartesian, to GCRF, then to osculating elements
                let state_eme2000_cart = state_koe_to_eci(
                    state,
                    self.angle_format
                        .expect("Keplerian representation must have angle_format"),
                );
                let state_gcrf = state_eme2000_to_gcrf(state_eme2000_cart);
                state_eci_to_koe(state_gcrf, angle_format)
            }
            (OrbitFrame::ECI, OrbitRepresentation::Cartesian) => {
                state_eci_to_koe(state, angle_format)
            }
            (OrbitFrame::GCRF, OrbitRepresentation::Cartesian) => {
                state_eci_to_koe(state, angle_format)
            }
            (OrbitFrame::EME2000, OrbitRepresentation::Cartesian) => {
                let state_gcrf = state_eme2000_to_gcrf(state);
                state_eci_to_koe(state_gcrf, angle_format)
            }
            (OrbitFrame::ECEF, OrbitRepresentation::Cartesian) => {
                let state_eci = state_ecef_to_eci(epoch, state);
                state_eci_to_koe(state_eci, angle_format)
            }
            (OrbitFrame::ITRF, OrbitRepresentation::Cartesian) => {
                let state_gcrf = state_itrf_to_gcrf(epoch, state);
                state_eci_to_koe(state_gcrf, angle_format)
            }
            (OrbitFrame::ECEF, OrbitRepresentation::Keplerian) => {
                return Err(BraheError::Error(
                    "Keplerian element trajectories should be in an inertial frame".to_string(),
                ));
            }
            (OrbitFrame::ITRF, OrbitRepresentation::Keplerian) => {
                return Err(BraheError::Error(
                    "Keplerian element trajectories should be in an inertial frame".to_string(),
                ));
            }
        })
    }
}

// =============================================================================
// DOrbitCovarianceProvider Trait
// =============================================================================

impl DOrbitCovarianceProvider for DOrbitTrajectory {
    fn covariance_eci(&self, epoch: Epoch) -> Result<DMatrix<f64>, BraheError> {
        let cov_native = self.covariance(epoch)?;
        let dim = cov_native.nrows();

        match self.frame {
            OrbitFrame::ECI | OrbitFrame::GCRF => Ok(cov_native),
            OrbitFrame::EME2000 => {
                // Apply frame bias rotation to first 6x6 block only
                let rot = rotation_eme2000_to_gcrf();

                // Build full-dimensional Jacobian
                let mut jacobian = DMatrix::<f64>::zeros(dim, dim);

                // Position and velocity blocks (top-left 3x3, and indices 3-5)
                for i in 0..3 {
                    for j in 0..3 {
                        jacobian[(i, j)] = rot[(i, j)];
                        jacobian[(3 + i, 3 + j)] = rot[(i, j)];
                    }
                }

                // Extended dimensions use identity (pass through unchanged)
                for i in 6..dim {
                    jacobian[(i, i)] = 1.0;
                }

                // Transform: C_ECI = J * C_EME2000 * J^T
                Ok(&jacobian * &cov_native * jacobian.transpose())
            }
            OrbitFrame::ECEF | OrbitFrame::ITRF => Err(BraheError::Error(
                "ECEF/ITRF covariance transformation not supported (requires time-dependent rotation derivatives)"
                    .to_string(),
            )),
        }
    }

    fn covariance_gcrf(&self, epoch: Epoch) -> Result<DMatrix<f64>, BraheError> {
        // GCRF ≈ ECI for practical purposes
        self.covariance_eci(epoch)
    }

    fn covariance_rtn(&self, epoch: Epoch) -> Result<DMatrix<f64>, BraheError> {
        let cov_eci = self.covariance_eci(epoch)?;
        let state_eci = self.state_eci(epoch)?;
        let dim = cov_eci.nrows();

        // Compute RTN rotation (from first 6 elements of orbital state)
        let rot_eci_to_rtn = rotation_eci_to_rtn(state_eci);

        // Compute angular velocity (Alfriend equation 2.16)
        let r = state_eci.fixed_rows::<3>(0);
        let v = state_eci.fixed_rows::<3>(3);
        let f_dot = (r.cross(&v)).norm() / r.norm().powi(2);
        let omega = Vector3::new(0.0, 0.0, f_dot);

        // Build skew-symmetric matrix [ω]×
        let omega_skew = SMatrix::<f64, 3, 3>::new(
            0.0, -omega[2], omega[1], omega[2], 0.0, -omega[0], -omega[1], omega[0], 0.0,
        );

        // Compute J21 = -[ω]× * R
        let j21 = -omega_skew * rot_eci_to_rtn;

        // Build full-dimensional Jacobian: J = [R, 0; J21, R; 0, 0, I]
        let mut jacobian = DMatrix::<f64>::zeros(dim, dim);

        // Position rotation block (top-left 3x3)
        for i in 0..3 {
            for j in 0..3 {
                jacobian[(i, j)] = rot_eci_to_rtn[(i, j)];
            }
        }

        // Velocity coupling block (indices 3-5, columns 0-2)
        for i in 0..3 {
            for j in 0..3 {
                jacobian[(3 + i, j)] = j21[(i, j)];
            }
        }

        // Velocity rotation block (indices 3-5, columns 3-5)
        for i in 0..3 {
            for j in 0..3 {
                jacobian[(3 + i, 3 + j)] = rot_eci_to_rtn[(i, j)];
            }
        }

        // Extended dimensions use identity (pass through unchanged)
        for i in 6..dim {
            jacobian[(i, i)] = 1.0;
        }

        // Transform: C_RTN = J * C_ECI * J^T
        Ok(&jacobian * &cov_eci * jacobian.transpose())
    }
}

// TODO: Add tests for DOrbitTrajectory
// Tests need to be adapted from SOrbitTrajectory for dynamic vector/matrix types.
// Provider trait tests were removed as DOrbitTrajectory uses inherent methods
// rather than trait implementations for frame conversions.
#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::time::{Epoch, TimeSystem};
    use crate::utils::testing::setup_global_test_eop;
    use approx::assert_abs_diff_eq;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_dorbittrajectory_new() {
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        assert_eq!(traj.frame, OrbitFrame::ECI);
        assert_eq!(traj.representation, OrbitRepresentation::Cartesian);
        assert_eq!(traj.dimension(), 6);
        assert_eq!(traj.orbital_dimension(), 6);
        assert_eq!(traj.additional_dimension(), 0);
        assert!(traj.is_empty());
    }

    #[test]
    fn test_dorbittrajectory_default() {
        let traj = DOrbitTrajectory::default();
        assert_eq!(traj.frame, OrbitFrame::ECI);
        assert_eq!(traj.representation, OrbitRepresentation::Cartesian);
        assert_eq!(traj.dimension(), 6);
        assert!(traj.is_empty());
    }

    #[test]
    fn test_dorbittrajectory_add_state() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);

        traj.add(epoch, state.clone());

        assert_eq!(traj.len(), 1);
        let (ep, st) = traj.get(0).unwrap();
        assert_eq!(ep, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(st[i], state[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_dorbittrajectory_display() {
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let display = format!("{}", traj);
        assert!(display.contains("DOrbitTrajectory"));
        assert!(display.contains("ECI"));
        assert!(display.contains("Cartesian"));
    }

    // ========== Extended State Tests ==========

    #[test]
    fn test_dorbittrajectory_extended_state_7d() {
        let mut traj =
            DOrbitTrajectory::new(7, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        assert_eq!(traj.dimension(), 7);
        assert_eq!(traj.orbital_dimension(), 6);
        assert_eq!(traj.additional_dimension(), 1);

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0, 42.0]);

        traj.add(epoch, state.clone());

        let (_, retrieved) = traj.get(0).unwrap();
        assert_eq!(retrieved.len(), 7);
        for i in 0..7 {
            assert_abs_diff_eq!(retrieved[i], state[i], epsilon = 1e-10);
        }
        // Verify additional state preserved
        assert_abs_diff_eq!(retrieved[6], 42.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dorbittrajectory_extended_state_9d() {
        let mut traj =
            DOrbitTrajectory::new(9, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        assert_eq!(traj.dimension(), 9);
        assert_eq!(traj.additional_dimension(), 3);

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![
            7000e3, 0.0, 0.0, // position
            0.0, 7.5e3, 0.0, // velocity
            1.0, 2.0, 3.0, // additional states
        ]);

        traj.add(epoch, state.clone());

        let (_, retrieved) = traj.get(0).unwrap();
        assert_eq!(retrieved.len(), 9);
        // Verify additional states preserved
        assert_abs_diff_eq!(retrieved[6], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(retrieved[7], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(retrieved[8], 3.0, epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "State dimension must be at least 6")]
    fn test_dorbittrajectory_invalid_dimension() {
        DOrbitTrajectory::new(5, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
    }

    #[test]
    #[should_panic(expected = "State dimension 6 does not match trajectory dimension 7")]
    fn test_dorbittrajectory_dimension_mismatch() {
        let mut traj =
            DOrbitTrajectory::new(7, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]); // 6D state for 7D trajectory
        traj.add(epoch, state);
    }

    #[test]
    fn test_dorbittrajectory_from_data_infers_dimension() {
        let epochs = vec![
            Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC),
            Epoch::from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, TimeSystem::UTC),
        ];
        let states = vec![
            DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0, 10.0, 20.0]),
            DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0, 11.0, 21.0]),
        ];

        let traj = DOrbitTrajectory::from_data(epochs, states).unwrap();

        assert_eq!(traj.dimension(), 8);
        assert_eq!(traj.additional_dimension(), 2);
        assert_eq!(traj.len(), 2);
    }

    #[test]
    fn test_dorbittrajectory_from_data_inconsistent_dimensions() {
        let epochs = vec![
            Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC),
            Epoch::from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, TimeSystem::UTC),
        ];
        let states = vec![
            DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0, 10.0]), // 7D
            DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]),       // 6D
        ];

        let result = DOrbitTrajectory::from_data(epochs, states);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("has dimension 6 but expected 7")
        );
    }

    // ========== Conversion Preservation Tests ==========

    #[test]
    fn test_dorbittrajectory_frame_conversion_preserves_additional() {
        setup_global_test_eop();

        let mut traj =
            DOrbitTrajectory::new(9, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![
            7000e3, 0.0, 0.0, // position
            0.0, 7.5e3, 0.0, // velocity
            10.0, 20.0, 30.0, // additional states
        ]);

        traj.add(epoch, state.clone());

        // Convert ECI -> ECEF
        let traj_ecef = traj.to_ecef();
        assert_eq!(traj_ecef.dimension(), 9);
        assert_eq!(traj_ecef.frame, OrbitFrame::ECEF);

        let (_, state_ecef) = traj_ecef.get(0).unwrap();
        // Orbital part should be different (transformed)
        assert!(state_ecef[0] != state[0] || state_ecef[1] != state[1]);
        // Additional part should be unchanged
        assert_abs_diff_eq!(state_ecef[6], 10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_ecef[7], 20.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_ecef[8], 30.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dorbittrajectory_to_gcrf_preserves_additional() {
        let mut traj =
            DOrbitTrajectory::new(9, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0, 100.0, 200.0, 300.0]);

        traj.add(epoch, state.clone());

        let traj_gcrf = traj.to_gcrf();
        assert_eq!(traj_gcrf.dimension(), 9);

        let (_, state_gcrf) = traj_gcrf.get(0).unwrap();
        // Additional states preserved
        assert_abs_diff_eq!(state_gcrf[6], 100.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_gcrf[7], 200.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_gcrf[8], 300.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dorbittrajectory_to_itrf_preserves_additional() {
        setup_global_test_eop();

        let mut traj =
            DOrbitTrajectory::new(8, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0, 50.0, 60.0]);

        traj.add(epoch, state.clone());

        let traj_itrf = traj.to_itrf();
        assert_eq!(traj_itrf.dimension(), 8);

        let (_, state_itrf) = traj_itrf.get(0).unwrap();
        assert_abs_diff_eq!(state_itrf[6], 50.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_itrf[7], 60.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dorbittrajectory_to_eme2000_preserves_additional() {
        let mut traj =
            DOrbitTrajectory::new(7, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0, 999.0]);

        traj.add(epoch, state.clone());

        let traj_eme2000 = traj.to_eme2000();
        assert_eq!(traj_eme2000.dimension(), 7);

        let (_, state_eme2000) = traj_eme2000.get(0).unwrap();
        assert_abs_diff_eq!(state_eme2000[6], 999.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dorbittrajectory_representation_conversion_preserves_additional() {
        use crate::constants::{GM_EARTH, R_EARTH};

        let mut traj =
            DOrbitTrajectory::new(9, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

        // Create a simple circular orbit state
        let a = R_EARTH + 500e3;
        let v = (GM_EARTH / a).sqrt();
        let state = DVector::from_vec(vec![
            a, 0.0, 0.0, // position
            0.0, v, 0.0, // velocity
            11.0, 22.0, 33.0, // additional
        ]);

        traj.add(epoch, state.clone());

        // Convert Cartesian -> Keplerian
        let traj_kep = traj.to_keplerian(AngleFormat::Degrees);
        assert_eq!(traj_kep.dimension(), 9);
        assert_eq!(traj_kep.representation, OrbitRepresentation::Keplerian);

        let (_, state_kep) = traj_kep.get(0).unwrap();
        // Additional states should be preserved
        assert_abs_diff_eq!(state_kep[6], 11.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_kep[7], 22.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_kep[8], 33.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dorbittrajectory_covariance_stays_6x6() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0, 1.0, 2.0, 3.0]);
        let cov = DMatrix::identity(6, 6);

        let traj = DOrbitTrajectory::from_orbital_data(
            vec![epoch],
            vec![state],
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            Some(vec![cov.clone()]),
        );

        let retrieved_cov = traj.covariance_at(epoch).unwrap();
        assert_eq!(retrieved_cov.nrows(), 6);
        assert_eq!(retrieved_cov.ncols(), 6);
    }

    // ========== Helper Function Tests ==========

    #[test]
    fn test_smat66_to_dmat_basic() {
        use nalgebra::SMatrix;

        let sm = SMatrix::<f64, 6, 6>::identity();
        let dm = smat66_to_dmat(sm);
        assert_eq!(dm.nrows(), 6);
        assert_eq!(dm.ncols(), 6);
        for i in 0..6 {
            for j in 0..6 {
                if i == j {
                    assert_abs_diff_eq!(dm[(i, j)], 1.0, epsilon = 1e-10);
                } else {
                    assert_abs_diff_eq!(dm[(i, j)], 0.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_dmat_to_smat66_basic() {
        let dm = DMatrix::identity(6, 6);
        let sm = dmat_to_smat66(dm);
        assert_eq!(sm.nrows(), 6);
        assert_eq!(sm.ncols(), 6);
        for i in 0..6 {
            for j in 0..6 {
                if i == j {
                    assert_abs_diff_eq!(sm[(i, j)], 1.0, epsilon = 1e-10);
                } else {
                    assert_abs_diff_eq!(sm[(i, j)], 0.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "DMatrix must have 6 rows")]
    fn test_dmat_to_smat66_panic_wrong_rows() {
        let dm = DMatrix::identity(5, 6);
        let _ = dmat_to_smat66(dm);
    }

    #[test]
    #[should_panic(expected = "DMatrix must have 6 columns")]
    fn test_dmat_to_smat66_panic_wrong_cols() {
        let dm = DMatrix::identity(6, 5);
        let _ = dmat_to_smat66(dm);
    }

    // ========== Constructor Panic Tests ==========

    #[test]
    #[should_panic(expected = "Angle format must be specified for Keplerian")]
    fn test_dorbittrajectory_new_panic_keplerian_no_angle_format() {
        DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Keplerian, None);
    }

    #[test]
    #[should_panic(expected = "Angle format should be None for Cartesian")]
    fn test_dorbittrajectory_new_panic_cartesian_with_angle_format() {
        DOrbitTrajectory::new(
            6,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            Some(AngleFormat::Degrees),
        );
    }

    #[test]
    #[should_panic(expected = "Keplerian elements should be in ECI frame")]
    fn test_dorbittrajectory_new_panic_ecef_keplerian() {
        DOrbitTrajectory::new(
            6,
            OrbitFrame::ECEF,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Degrees),
        );
    }

    // ========== Builder Method Tests ==========

    #[test]
    fn test_dorbittrajectory_with_interpolation_method_linear() {
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
            .with_interpolation_method(InterpolationMethod::Linear);
        assert_eq!(traj.interpolation_method, InterpolationMethod::Linear);
    }

    #[test]
    fn test_dorbittrajectory_with_interpolation_method_lagrange() {
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
            .with_interpolation_method(InterpolationMethod::Lagrange { degree: 5 });
        assert!(matches!(
            traj.interpolation_method,
            InterpolationMethod::Lagrange { degree: 5 }
        ));
    }

    #[test]
    fn test_dorbittrajectory_with_interpolation_method_hermite() {
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
            .with_interpolation_method(InterpolationMethod::HermiteCubic);
        assert_eq!(traj.interpolation_method, InterpolationMethod::HermiteCubic);
    }

    #[test]
    fn test_dorbittrajectory_with_eviction_policy_max_size() {
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
            .with_eviction_policy_max_size(100);
        assert_eq!(traj.eviction_policy, TrajectoryEvictionPolicy::KeepCount);
        assert_eq!(traj.max_size, Some(100));
        assert_eq!(traj.max_age, None);
    }

    #[test]
    #[should_panic(expected = "Maximum size must be >= 1")]
    fn test_dorbittrajectory_with_eviction_policy_max_size_panic_zero() {
        let _ = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
            .with_eviction_policy_max_size(0);
    }

    #[test]
    fn test_dorbittrajectory_with_eviction_policy_max_age() {
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
            .with_eviction_policy_max_age(3600.0);
        assert_eq!(
            traj.eviction_policy,
            TrajectoryEvictionPolicy::KeepWithinDuration
        );
        assert_eq!(traj.max_age, Some(3600.0));
        assert_eq!(traj.max_size, None);
    }

    #[test]
    #[should_panic(expected = "Maximum age must be > 0.0")]
    fn test_dorbittrajectory_with_eviction_policy_max_age_panic_zero() {
        let _ = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
            .with_eviction_policy_max_age(0.0);
    }

    #[test]
    #[should_panic(expected = "Maximum age must be > 0.0")]
    fn test_dorbittrajectory_with_eviction_policy_max_age_panic_negative() {
        let _ = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
            .with_eviction_policy_max_age(-1.0);
    }

    // ========== add_state_and_covariance Tests ==========

    #[test]
    fn test_dorbittrajectory_add_state_and_covariance_basic() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        traj.covariances = Some(Vec::new());

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let cov = DMatrix::identity(6, 6) * 100.0;

        traj.add_state_and_covariance(epoch, state.clone(), cov.clone());

        assert_eq!(traj.len(), 1);
        let retrieved_cov = traj.covariances.as_ref().unwrap()[0].clone();
        assert_abs_diff_eq!(retrieved_cov[(0, 0)], 100.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dorbittrajectory_add_state_and_covariance_ordering() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        traj.covariances = Some(Vec::new());

        let epoch1 = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, TimeSystem::UTC);
        let epoch3 = Epoch::from_datetime(2024, 1, 1, 11, 59, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let cov = DMatrix::identity(6, 6);

        traj.add_state_and_covariance(epoch1, state.clone(), cov.clone());
        traj.add_state_and_covariance(epoch2, state.clone(), cov.clone());
        traj.add_state_and_covariance(epoch3, state.clone(), cov.clone());

        assert_eq!(traj.len(), 3);
        // Should be sorted: epoch3 < epoch1 < epoch2
        assert_eq!(traj.epochs[0], epoch3);
        assert_eq!(traj.epochs[1], epoch1);
        assert_eq!(traj.epochs[2], epoch2);
    }

    #[test]
    #[should_panic(
        expected = "Cannot add state with covariance to trajectory without covariances initialized"
    )]
    fn test_dorbittrajectory_add_state_and_covariance_panic_no_init() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let cov = DMatrix::identity(6, 6);

        traj.add_state_and_covariance(epoch, state, cov);
    }

    // ========== to_matrix Tests ==========

    #[test]
    fn test_dorbittrajectory_to_matrix_basic() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 1000.0, 2000.0, 100.0, 7.5e3, 50.0]);
        traj.add(epoch, state.clone());

        let matrix = traj.to_matrix().unwrap();
        assert_eq!(matrix.nrows(), 1);
        assert_eq!(matrix.ncols(), 6);
        for i in 0..6 {
            assert_abs_diff_eq!(matrix[(0, i)], state[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_dorbittrajectory_to_matrix_multiple_states() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch1 = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, TimeSystem::UTC);
        let state1 = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let state2 = DVector::from_vec(vec![7100e3, 100.0, 50.0, 10.0, 7.4e3, 5.0]);
        traj.add(epoch1, state1.clone());
        traj.add(epoch2, state2.clone());

        let matrix = traj.to_matrix().unwrap();
        assert_eq!(matrix.nrows(), 2);
        assert_eq!(matrix.ncols(), 6);
        for i in 0..6 {
            assert_abs_diff_eq!(matrix[(0, i)], state1[i], epsilon = 1e-10);
            assert_abs_diff_eq!(matrix[(1, i)], state2[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_dorbittrajectory_to_matrix_error_empty() {
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let result = traj.to_matrix();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Cannot convert empty trajectory")
        );
    }

    // ========== covariance_at Tests ==========

    #[test]
    fn test_dorbittrajectory_covariance_at_exact_match() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let cov = DMatrix::identity(6, 6) * 42.0;

        let traj = DOrbitTrajectory::from_orbital_data(
            vec![epoch],
            vec![state],
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            Some(vec![cov.clone()]),
        );

        let retrieved = traj.covariance_at(epoch).unwrap();
        assert_abs_diff_eq!(retrieved[(0, 0)], 42.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dorbittrajectory_covariance_at_interpolation() {
        let epoch1 = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_datetime(2024, 1, 1, 12, 2, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let cov1 = DMatrix::identity(6, 6) * 100.0;
        let cov2 = DMatrix::identity(6, 6) * 200.0;

        let traj = DOrbitTrajectory::from_orbital_data(
            vec![epoch1, epoch2],
            vec![state.clone(), state],
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            Some(vec![cov1, cov2]),
        );

        // Query at midpoint
        let mid_epoch = Epoch::from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, TimeSystem::UTC);
        let retrieved = traj.covariance_at(mid_epoch).unwrap();
        // TwoWasserstein interpolation should give approximately midpoint for symmetric matrices
        assert!(retrieved[(0, 0)] > 100.0 && retrieved[(0, 0)] < 200.0);
    }

    #[test]
    fn test_dorbittrajectory_covariance_at_none_disabled() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        let result = traj.covariance_at(epoch);
        assert!(result.is_none());
    }

    #[test]
    fn test_dorbittrajectory_covariance_at_none_empty() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        traj.covariances = Some(Vec::new());

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let result = traj.covariance_at(epoch);
        assert!(result.is_none());
    }

    #[test]
    fn test_dorbittrajectory_covariance_at_none_out_of_range() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let cov = DMatrix::identity(6, 6);

        let traj = DOrbitTrajectory::from_orbital_data(
            vec![epoch],
            vec![state],
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            Some(vec![cov]),
        );

        let before = Epoch::from_datetime(2024, 1, 1, 11, 0, 0.0, 0.0, TimeSystem::UTC);
        let after = Epoch::from_datetime(2024, 1, 1, 13, 0, 0.0, 0.0, TimeSystem::UTC);
        assert!(traj.covariance_at(before).is_none());
        assert!(traj.covariance_at(after).is_none());
    }

    // ========== add_full Tests ==========

    #[test]
    fn test_dorbittrajectory_add_full_all_optional_provided() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let cov = DMatrix::identity(6, 6) * 10.0;
        let stm = DMatrix::identity(6, 6) * 2.0;
        let sens = DMatrix::zeros(6, 3);
        let accel = DVector::from_vec(vec![0.0, 0.0, -9.8]);

        traj.add_full(
            epoch,
            state,
            Some(cov.clone()),
            Some(stm.clone()),
            Some(sens.clone()),
            Some(accel.clone()),
        );

        assert_eq!(traj.len(), 1);
        assert!(traj.covariances.is_some());
        assert!(traj.stms.is_some());
        assert!(traj.sensitivities.is_some());
        assert!(traj.accelerations.is_some());
    }

    #[test]
    fn test_dorbittrajectory_add_full_only_state() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);

        traj.add_full(epoch, state, None, None, None, None);

        assert_eq!(traj.len(), 1);
        assert!(traj.covariances.is_none());
        assert!(traj.stms.is_none());
        assert!(traj.sensitivities.is_none());
        assert!(traj.accelerations.is_none());
    }

    #[test]
    fn test_dorbittrajectory_add_full_with_covariance() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let cov = DMatrix::identity(6, 6) * 50.0;

        traj.add_full(epoch, state, Some(cov), None, None, None);

        assert!(traj.covariances.is_some());
        assert_abs_diff_eq!(
            traj.covariances.as_ref().unwrap()[0][(0, 0)],
            50.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_dorbittrajectory_add_full_with_stm() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let stm = DMatrix::identity(6, 6) * 3.0;

        traj.add_full(epoch, state, None, Some(stm), None, None);

        assert!(traj.stms.is_some());
        assert_abs_diff_eq!(traj.stms.as_ref().unwrap()[0][(0, 0)], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dorbittrajectory_add_full_with_sensitivity() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let mut sens = DMatrix::zeros(6, 4);
        sens[(0, 0)] = 1.5;

        traj.add_full(epoch, state, None, None, Some(sens), None);

        assert!(traj.sensitivities.is_some());
        assert_abs_diff_eq!(
            traj.sensitivities.as_ref().unwrap()[0][(0, 0)],
            1.5,
            epsilon = 1e-10
        );
        assert_eq!(traj.sensitivity_dimension, Some((6, 4)));
    }

    #[test]
    fn test_dorbittrajectory_add_full_with_acceleration() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let accel = DVector::from_vec(vec![1.0, 2.0, 3.0]);

        traj.add_full(epoch, state, None, None, None, Some(accel));

        assert!(traj.accelerations.is_some());
        assert_eq!(traj.acceleration_dimension, Some(3));
        assert_abs_diff_eq!(
            traj.accelerations.as_ref().unwrap()[0][2],
            3.0,
            epsilon = 1e-10
        );
    }

    #[test]
    #[should_panic(expected = "State vector dimension")]
    fn test_dorbittrajectory_add_full_panic_state_dimension() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3]); // 5D instead of 6D

        traj.add_full(epoch, state, None, None, None, None);
    }

    #[test]
    #[should_panic(expected = "Covariance dimension mismatch")]
    fn test_dorbittrajectory_add_full_panic_cov_dimension() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let cov = DMatrix::identity(5, 5); // Wrong dimensions

        traj.add_full(epoch, state, Some(cov), None, None, None);
    }

    #[test]
    #[should_panic(expected = "STM dimension mismatch")]
    fn test_dorbittrajectory_add_full_panic_stm_dimension() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let stm = DMatrix::identity(5, 6); // Wrong dimensions

        traj.add_full(epoch, state, None, Some(stm), None, None);
    }

    #[test]
    #[should_panic(expected = "Sensitivity row dimension mismatch")]
    fn test_dorbittrajectory_add_full_panic_sens_rows() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let sens = DMatrix::zeros(5, 3); // Wrong row count

        traj.add_full(epoch, state, None, None, Some(sens), None);
    }

    #[test]
    #[should_panic(expected = "Sensitivity column dimension mismatch")]
    fn test_dorbittrajectory_add_full_panic_sens_cols() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch1 = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let sens1 = DMatrix::zeros(6, 3);
        let sens2 = DMatrix::zeros(6, 4); // Different column count

        traj.add_full(epoch1, state.clone(), None, None, Some(sens1), None);
        traj.add_full(epoch2, state, None, None, Some(sens2), None);
    }

    #[test]
    #[should_panic(expected = "Acceleration dimension")]
    fn test_dorbittrajectory_add_full_panic_accel_dimension() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch1 = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let accel1 = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let accel2 = DVector::from_vec(vec![1.0, 2.0]); // Different dimension

        traj.add_full(epoch1, state.clone(), None, None, None, Some(accel1));
        traj.add_full(epoch2, state, None, None, None, Some(accel2));
    }

    // ========== Trajectory Trait Tests ==========

    #[test]
    fn test_dorbittrajectory_from_data_basic() {
        let epochs = vec![
            Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC),
            Epoch::from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, TimeSystem::UTC),
        ];
        let states = vec![
            DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]),
            DVector::from_vec(vec![7100e3, 0.0, 0.0, 0.0, 7.4e3, 0.0]),
        ];

        let traj = DOrbitTrajectory::from_data(epochs, states).unwrap();
        assert_eq!(traj.len(), 2);
        assert_eq!(traj.dimension(), 6);
    }

    #[test]
    fn test_dorbittrajectory_from_data_sorts_epochs() {
        let epoch1 = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, TimeSystem::UTC);
        let epoch3 = Epoch::from_datetime(2024, 1, 1, 11, 59, 0.0, 0.0, TimeSystem::UTC);
        let epochs = vec![epoch1, epoch2, epoch3]; // Out of order
        let states = vec![
            DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]),
            DVector::from_vec(vec![7100e3, 0.0, 0.0, 0.0, 7.4e3, 0.0]),
            DVector::from_vec(vec![6900e3, 0.0, 0.0, 0.0, 7.6e3, 0.0]),
        ];

        let traj = DOrbitTrajectory::from_data(epochs, states).unwrap();
        assert_eq!(traj.epochs[0], epoch3);
        assert_eq!(traj.epochs[1], epoch1);
        assert_eq!(traj.epochs[2], epoch2);
    }

    #[test]
    fn test_dorbittrajectory_from_data_error_empty() {
        let epochs: Vec<Epoch> = vec![];
        let states: Vec<DVector<f64>> = vec![];
        let result = DOrbitTrajectory::from_data(epochs, states);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Cannot create trajectory from empty data")
        );
    }

    #[test]
    fn test_dorbittrajectory_from_data_error_length_mismatch() {
        let epochs = vec![
            Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC),
            Epoch::from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, TimeSystem::UTC),
        ];
        let states = vec![DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])];

        let result = DOrbitTrajectory::from_data(epochs, states);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("same length"));
    }

    #[test]
    fn test_dorbittrajectory_from_data_error_dimension_too_small() {
        let epochs = vec![Epoch::from_datetime(
            2024,
            1,
            1,
            12,
            0,
            0.0,
            0.0,
            TimeSystem::UTC,
        )];
        let states = vec![DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3])]; // 5D

        let result = DOrbitTrajectory::from_data(epochs, states);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("must be at least 6")
        );
    }

    #[test]
    fn test_dorbittrajectory_epoch_at_idx_valid() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        let retrieved = traj.epoch_at_idx(0).unwrap();
        assert_eq!(retrieved, epoch);
    }

    #[test]
    fn test_dorbittrajectory_epoch_at_idx_error_out_of_bounds() {
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let result = traj.epoch_at_idx(0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("out of bounds"));
    }

    #[test]
    fn test_dorbittrajectory_state_at_idx_valid() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 100.0, 200.0, 10.0, 7.5e3, 5.0]);
        traj.add(epoch, state.clone());

        let retrieved = traj.state_at_idx(0).unwrap();
        for i in 0..6 {
            assert_abs_diff_eq!(retrieved[i], state[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_dorbittrajectory_state_at_idx_error_out_of_bounds() {
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let result = traj.state_at_idx(0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("out of bounds"));
    }

    #[test]
    fn test_dorbittrajectory_nearest_state_exact() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state.clone());

        let (found_epoch, found_state) = traj.nearest_state(&epoch).unwrap();
        assert_eq!(found_epoch, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(found_state[i], state[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_dorbittrajectory_nearest_state_closest() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch1 = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_datetime(2024, 1, 1, 12, 10, 0.0, 0.0, TimeSystem::UTC);
        let state1 = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let state2 = DVector::from_vec(vec![7100e3, 0.0, 0.0, 0.0, 7.4e3, 0.0]);
        traj.add(epoch1, state1.clone());
        traj.add(epoch2, state2);

        // Query closer to epoch1
        let query = Epoch::from_datetime(2024, 1, 1, 12, 2, 0.0, 0.0, TimeSystem::UTC);
        let (found_epoch, _) = traj.nearest_state(&query).unwrap();
        assert_eq!(found_epoch, epoch1);

        // Query closer to epoch2
        let query = Epoch::from_datetime(2024, 1, 1, 12, 8, 0.0, 0.0, TimeSystem::UTC);
        let (found_epoch, _) = traj.nearest_state(&query).unwrap();
        assert_eq!(found_epoch, epoch2);
    }

    #[test]
    fn test_dorbittrajectory_nearest_state_error_empty() {
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let result = traj.nearest_state(&epoch);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty trajectory"));
    }

    #[test]
    fn test_dorbittrajectory_len_and_is_empty() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        assert!(traj.is_empty());
        assert_eq!(traj.len(), 0);

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        assert!(!traj.is_empty());
        assert_eq!(traj.len(), 1);
    }

    #[test]
    fn test_dorbittrajectory_start_end_epoch() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch1 = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_datetime(2024, 1, 1, 12, 10, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch1, state.clone());
        traj.add(epoch2, state);

        assert_eq!(traj.start_epoch(), Some(epoch1));
        assert_eq!(traj.end_epoch(), Some(epoch2));
    }

    #[test]
    fn test_dorbittrajectory_start_end_epoch_empty() {
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        assert_eq!(traj.start_epoch(), None);
        assert_eq!(traj.end_epoch(), None);
    }

    #[test]
    fn test_dorbittrajectory_timespan_multiple() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch1 = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_datetime(2024, 1, 1, 12, 10, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch1, state.clone());
        traj.add(epoch2, state);

        let timespan = traj.timespan().unwrap();
        assert_abs_diff_eq!(timespan, 600.0, epsilon = 1e-10); // 10 minutes = 600 seconds
    }

    #[test]
    fn test_dorbittrajectory_timespan_single_none() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        assert!(traj.timespan().is_none());
    }

    #[test]
    fn test_dorbittrajectory_timespan_empty_none() {
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        assert!(traj.timespan().is_none());
    }

    #[test]
    fn test_dorbittrajectory_first_last() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch1 = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_datetime(2024, 1, 1, 12, 10, 0.0, 0.0, TimeSystem::UTC);
        let state1 = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let state2 = DVector::from_vec(vec![7100e3, 0.0, 0.0, 0.0, 7.4e3, 0.0]);
        traj.add(epoch1, state1.clone());
        traj.add(epoch2, state2.clone());

        let (first_e, first_s) = traj.first().unwrap();
        assert_eq!(first_e, epoch1);
        assert_abs_diff_eq!(first_s[0], state1[0], epsilon = 1e-10);

        let (last_e, last_s) = traj.last().unwrap();
        assert_eq!(last_e, epoch2);
        assert_abs_diff_eq!(last_s[0], state2[0], epsilon = 1e-10);
    }

    #[test]
    fn test_dorbittrajectory_first_last_empty() {
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        assert!(traj.first().is_none());
        assert!(traj.last().is_none());
    }

    #[test]
    fn test_dorbittrajectory_clear_all_storage_types() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let cov = DMatrix::identity(6, 6);
        let stm = DMatrix::identity(6, 6);
        let sens = DMatrix::zeros(6, 3);
        let accel = DVector::from_vec(vec![0.0, 0.0, -9.8]);

        traj.add_full(epoch, state, Some(cov), Some(stm), Some(sens), Some(accel));
        assert_eq!(traj.len(), 1);

        traj.clear();
        assert_eq!(traj.len(), 0);
        assert!(
            traj.covariances
                .as_ref()
                .map(|v| v.is_empty())
                .unwrap_or(true)
        );
        assert!(traj.stms.as_ref().map(|v| v.is_empty()).unwrap_or(true));
        assert!(
            traj.sensitivities
                .as_ref()
                .map(|v| v.is_empty())
                .unwrap_or(true)
        );
        assert!(
            traj.accelerations
                .as_ref()
                .map(|v| v.is_empty())
                .unwrap_or(true)
        );
    }

    #[test]
    fn test_dorbittrajectory_remove_epoch_found() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch1 = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_datetime(2024, 1, 1, 12, 10, 0.0, 0.0, TimeSystem::UTC);
        let state1 = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let state2 = DVector::from_vec(vec![7100e3, 0.0, 0.0, 0.0, 7.4e3, 0.0]);
        traj.add(epoch1, state1.clone());
        traj.add(epoch2, state2);

        let removed = traj.remove_epoch(&epoch1).unwrap();
        assert_abs_diff_eq!(removed[0], state1[0], epsilon = 1e-10);
        assert_eq!(traj.len(), 1);
        assert_eq!(traj.epochs[0], epoch2);
    }

    #[test]
    fn test_dorbittrajectory_remove_epoch_not_found() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch1 = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_datetime(2024, 1, 1, 12, 10, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch1, state);

        let result = traj.remove_epoch(&epoch2);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_dorbittrajectory_remove_by_index() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch1 = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_datetime(2024, 1, 1, 12, 10, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch1, state.clone());
        traj.add(epoch2, state);

        let (removed_epoch, _) = traj.remove(0).unwrap();
        assert_eq!(removed_epoch, epoch1);
        assert_eq!(traj.len(), 1);
    }

    #[test]
    fn test_dorbittrajectory_remove_by_index_error_out_of_bounds() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let result = traj.remove(0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("out of bounds"));
    }

    #[test]
    fn test_dorbittrajectory_get_valid() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state.clone());

        let (got_epoch, got_state) = traj.get(0).unwrap();
        assert_eq!(got_epoch, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(got_state[i], state[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_dorbittrajectory_get_error_out_of_bounds() {
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let result = traj.get(0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("out of bounds"));
    }

    #[test]
    fn test_dorbittrajectory_index_before_epoch_valid() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch1 = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_datetime(2024, 1, 1, 12, 10, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch1, state.clone());
        traj.add(epoch2, state);

        // Query between epochs
        let query = Epoch::from_datetime(2024, 1, 1, 12, 5, 0.0, 0.0, TimeSystem::UTC);
        let idx = traj.index_before_epoch(&query).unwrap();
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_dorbittrajectory_index_before_epoch_exact_match() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        let idx = traj.index_before_epoch(&epoch).unwrap();
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_dorbittrajectory_index_before_epoch_error_empty() {
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let result = traj.index_before_epoch(&epoch);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty trajectory"));
    }

    #[test]
    fn test_dorbittrajectory_index_before_epoch_error_before_all() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        let before = Epoch::from_datetime(2024, 1, 1, 11, 0, 0.0, 0.0, TimeSystem::UTC);
        let result = traj.index_before_epoch(&before);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("before all states")
        );
    }

    #[test]
    fn test_dorbittrajectory_index_after_epoch_valid() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch1 = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_datetime(2024, 1, 1, 12, 10, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch1, state.clone());
        traj.add(epoch2, state);

        // Query between epochs
        let query = Epoch::from_datetime(2024, 1, 1, 12, 5, 0.0, 0.0, TimeSystem::UTC);
        let idx = traj.index_after_epoch(&query).unwrap();
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_dorbittrajectory_index_after_epoch_exact_match() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        let idx = traj.index_after_epoch(&epoch).unwrap();
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_dorbittrajectory_index_after_epoch_error_empty() {
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let result = traj.index_after_epoch(&epoch);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty trajectory"));
    }

    #[test]
    fn test_dorbittrajectory_index_after_epoch_error_after_all() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        let after = Epoch::from_datetime(2024, 1, 1, 13, 0, 0.0, 0.0, TimeSystem::UTC);
        let result = traj.index_after_epoch(&after);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("after all states"));
    }

    #[test]
    fn test_dorbittrajectory_set_eviction_policy_max_size() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        traj.set_eviction_policy_max_size(50).unwrap();
        assert_eq!(traj.eviction_policy, TrajectoryEvictionPolicy::KeepCount);
        assert_eq!(traj.max_size, Some(50));
    }

    #[test]
    fn test_dorbittrajectory_set_eviction_policy_max_size_error_zero() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let result = traj.set_eviction_policy_max_size(0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must be >= 1"));
    }

    #[test]
    fn test_dorbittrajectory_set_eviction_policy_max_age() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        traj.set_eviction_policy_max_age(1800.0).unwrap();
        assert_eq!(
            traj.eviction_policy,
            TrajectoryEvictionPolicy::KeepWithinDuration
        );
        assert_eq!(traj.max_age, Some(1800.0));
    }

    #[test]
    fn test_dorbittrajectory_set_eviction_policy_max_age_error_negative() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let result = traj.set_eviction_policy_max_age(-100.0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must be > 0.0"));
    }

    #[test]
    fn test_dorbittrajectory_get_eviction_policy() {
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        assert_eq!(traj.get_eviction_policy(), TrajectoryEvictionPolicy::None);

        let traj2 = traj.with_eviction_policy_max_size(10);
        assert_eq!(
            traj2.get_eviction_policy(),
            TrajectoryEvictionPolicy::KeepCount
        );
    }

    // ========== InterpolationConfig Trait Tests ==========

    #[test]
    fn test_dorbittrajectory_interpolation_config_set_get() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);

        // Default is Linear
        assert_eq!(traj.get_interpolation_method(), InterpolationMethod::Linear);

        // Set to Lagrange
        traj.set_interpolation_method(InterpolationMethod::Lagrange { degree: 5 });
        assert_eq!(
            traj.get_interpolation_method(),
            InterpolationMethod::Lagrange { degree: 5 }
        );

        // Set to HermiteCubic
        traj.set_interpolation_method(InterpolationMethod::HermiteCubic);
        assert_eq!(
            traj.get_interpolation_method(),
            InterpolationMethod::HermiteCubic
        );

        // Set to HermiteQuintic
        traj.set_interpolation_method(InterpolationMethod::HermiteQuintic);
        assert_eq!(
            traj.get_interpolation_method(),
            InterpolationMethod::HermiteQuintic
        );
    }

    #[test]
    fn test_dorbittrajectory_interpolation_config_with_builder() {
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
            .with_interpolation_method(InterpolationMethod::Lagrange { degree: 7 });

        assert_eq!(
            traj.get_interpolation_method(),
            InterpolationMethod::Lagrange { degree: 7 }
        );
    }

    // ========== InterpolatableTrajectory Trait Tests ==========

    #[test]
    fn test_dorbittrajectory_interpolate_linear_basic() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch1 = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_datetime(2024, 1, 1, 12, 10, 0.0, 0.0, TimeSystem::UTC);
        let state1 = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let state2 = DVector::from_vec(vec![7100e3, 0.0, 0.0, 0.0, 7.4e3, 0.0]);
        traj.add(epoch1, state1);
        traj.add(epoch2, state2);

        // Interpolate at midpoint
        let mid = Epoch::from_datetime(2024, 1, 1, 12, 5, 0.0, 0.0, TimeSystem::UTC);
        let result = traj.interpolate(&mid).unwrap();

        // Should be average of the two states
        assert_abs_diff_eq!(result[0], 7050e3, epsilon = 1.0);
        assert_abs_diff_eq!(result[4], 7.45e3, epsilon = 0.1);
    }

    #[test]
    fn test_dorbittrajectory_interpolate_exact_match() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 100.0, 200.0, 10.0, 7.5e3, 5.0]);
        traj.add(epoch, state.clone());

        let result = traj.interpolate(&epoch).unwrap();
        for i in 0..6 {
            assert_abs_diff_eq!(result[i], state[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_dorbittrajectory_interpolate_error_before_start() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        let before = Epoch::from_datetime(2024, 1, 1, 11, 0, 0.0, 0.0, TimeSystem::UTC);
        let result = traj.interpolate(&before);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("before trajectory start")
        );
    }

    #[test]
    fn test_dorbittrajectory_interpolate_error_after_end() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        let after = Epoch::from_datetime(2024, 1, 1, 13, 0, 0.0, 0.0, TimeSystem::UTC);
        let result = traj.interpolate(&after);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("after trajectory end")
        );
    }

    #[test]
    fn test_dorbittrajectory_interpolate_lagrange() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
                .with_interpolation_method(InterpolationMethod::Lagrange { degree: 3 });

        // Add 4 points for degree 3 Lagrange
        for i in 0..4 {
            let epoch = Epoch::from_datetime(2024, 1, 1, 12, i * 10, 0.0, 0.0, TimeSystem::UTC);
            let state = DVector::from_vec(vec![
                7000e3 + (i as f64) * 100e3,
                0.0,
                0.0,
                0.0,
                7.5e3 - (i as f64) * 0.1e3,
                0.0,
            ]);
            traj.add(epoch, state);
        }

        // Interpolate at midpoint
        let mid = Epoch::from_datetime(2024, 1, 1, 12, 15, 0.0, 0.0, TimeSystem::UTC);
        let result = traj.interpolate(&mid);
        assert!(result.is_ok());
        // Result should be somewhere between the states
        assert!(result.as_ref().unwrap()[0] > 7000e3);
        assert!(result.as_ref().unwrap()[0] < 7300e3);
    }

    #[test]
    fn test_dorbittrajectory_interpolate_lagrange_error_insufficient_points() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
                .with_interpolation_method(InterpolationMethod::Lagrange { degree: 5 });

        // Add only 3 points, but degree 5 requires 6 points
        for i in 0..3 {
            let epoch = Epoch::from_datetime(2024, 1, 1, 12, i * 10, 0.0, 0.0, TimeSystem::UTC);
            let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
            traj.add(epoch, state);
        }

        let mid = Epoch::from_datetime(2024, 1, 1, 12, 5, 0.0, 0.0, TimeSystem::UTC);
        let result = traj.interpolate(&mid);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("requires"));
    }

    #[test]
    fn test_dorbittrajectory_interpolate_hermite_cubic() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
                .with_interpolation_method(InterpolationMethod::HermiteCubic);

        let epoch1 = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_datetime(2024, 1, 1, 12, 10, 0.0, 0.0, TimeSystem::UTC);
        let state1 = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let state2 = DVector::from_vec(vec![7100e3, 0.0, 0.0, 0.0, 7.4e3, 0.0]);
        traj.add(epoch1, state1);
        traj.add(epoch2, state2);

        let mid = Epoch::from_datetime(2024, 1, 1, 12, 5, 0.0, 0.0, TimeSystem::UTC);
        let result = traj.interpolate(&mid);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dorbittrajectory_interpolate_hermite_cubic_error_non_6d() {
        let mut traj =
            DOrbitTrajectory::new(9, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
                .with_interpolation_method(InterpolationMethod::HermiteCubic);

        let epoch1 = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_datetime(2024, 1, 1, 12, 10, 0.0, 0.0, TimeSystem::UTC);
        let state1 = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0, 1.0, 2.0, 3.0]);
        let state2 = DVector::from_vec(vec![7100e3, 0.0, 0.0, 0.0, 7.4e3, 0.0, 4.0, 5.0, 6.0]);
        traj.add(epoch1, state1);
        traj.add(epoch2, state2);

        let mid = Epoch::from_datetime(2024, 1, 1, 12, 5, 0.0, 0.0, TimeSystem::UTC);
        let result = traj.interpolate(&mid);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("HermiteCubic interpolation requires 6D states"));
    }

    #[test]
    fn test_dorbittrajectory_interpolate_hermite_quintic_with_accelerations() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
                .with_interpolation_method(InterpolationMethod::HermiteQuintic);
        traj.enable_acceleration_storage(3);

        let epoch1 = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_datetime(2024, 1, 1, 12, 10, 0.0, 0.0, TimeSystem::UTC);
        let state1 = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let state2 = DVector::from_vec(vec![7100e3, 0.0, 0.0, 0.0, 7.4e3, 0.0]);
        let accel1 = DVector::from_vec(vec![-9.8, 0.0, 0.0]);
        let accel2 = DVector::from_vec(vec![-9.7, 0.0, 0.0]);

        traj.add_with_acceleration(epoch1, state1, accel1);
        traj.add_with_acceleration(epoch2, state2, accel2);

        let mid = Epoch::from_datetime(2024, 1, 1, 12, 5, 0.0, 0.0, TimeSystem::UTC);
        let result = traj.interpolate(&mid);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dorbittrajectory_interpolate_hermite_quintic_fd_fallback() {
        // Without accelerations, uses finite difference with 3+ points
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
                .with_interpolation_method(InterpolationMethod::HermiteQuintic);

        // Add 3 points for FD fallback
        for i in 0..3 {
            let epoch = Epoch::from_datetime(2024, 1, 1, 12, i * 10, 0.0, 0.0, TimeSystem::UTC);
            let state =
                DVector::from_vec(vec![7000e3 + (i as f64) * 100e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
            traj.add(epoch, state);
        }

        let mid = Epoch::from_datetime(2024, 1, 1, 12, 5, 0.0, 0.0, TimeSystem::UTC);
        let result = traj.interpolate(&mid);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dorbittrajectory_interpolate_hermite_quintic_error_insufficient() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
                .with_interpolation_method(InterpolationMethod::HermiteQuintic);

        // Only 2 points and no accelerations
        let epoch1 = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_datetime(2024, 1, 1, 12, 10, 0.0, 0.0, TimeSystem::UTC);
        let state1 = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let state2 = DVector::from_vec(vec![7100e3, 0.0, 0.0, 0.0, 7.4e3, 0.0]);
        traj.add(epoch1, state1);
        traj.add(epoch2, state2);

        let mid = Epoch::from_datetime(2024, 1, 1, 12, 5, 0.0, 0.0, TimeSystem::UTC);
        let result = traj.interpolate(&mid);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("HermiteQuintic interpolation requires either stored accelerations")
        );
    }

    // ========== STMStorage Trait Tests ==========

    #[test]
    fn test_dorbittrajectory_enable_stm_storage() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        assert!(traj.stms.is_none());
        traj.enable_stm_storage();
        assert!(traj.stms.is_some());

        // Should have identity matrix for existing state
        let stm = traj.stm_at_idx(0).unwrap();
        for i in 0..6 {
            for j in 0..6 {
                if i == j {
                    assert_abs_diff_eq!(stm[(i, j)], 1.0, epsilon = 1e-10);
                } else {
                    assert_abs_diff_eq!(stm[(i, j)], 0.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_dorbittrajectory_stm_at_idx_valid() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);
        traj.enable_stm_storage();

        let stm = traj.stm_at_idx(0);
        assert!(stm.is_some());
        assert_eq!(stm.unwrap().nrows(), 6);
        assert_eq!(stm.unwrap().ncols(), 6);
    }

    #[test]
    fn test_dorbittrajectory_stm_at_idx_none_disabled() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        // Without enabling STM storage
        let stm = traj.stm_at_idx(0);
        assert!(stm.is_none());
    }

    #[test]
    fn test_dorbittrajectory_set_stm_at() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        let mut custom_stm = DMatrix::zeros(6, 6);
        custom_stm[(0, 0)] = 2.5;
        custom_stm[(1, 1)] = 3.0;

        traj.set_stm_at(0, custom_stm.clone());

        let retrieved = traj.stm_at_idx(0).unwrap();
        assert_abs_diff_eq!(retrieved[(0, 0)], 2.5, epsilon = 1e-10);
        assert_abs_diff_eq!(retrieved[(1, 1)], 3.0, epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_dorbittrajectory_set_stm_at_panic_out_of_bounds() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let stm = DMatrix::identity(6, 6);
        traj.set_stm_at(0, stm); // No states added yet
    }

    #[test]
    #[should_panic(expected = "do not match expected 6x6")]
    fn test_dorbittrajectory_set_stm_at_panic_wrong_dimensions() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        let wrong_stm = DMatrix::identity(5, 5);
        traj.set_stm_at(0, wrong_stm);
    }

    #[test]
    fn test_dorbittrajectory_stm_dimensions() {
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        assert_eq!(traj.stm_dimensions(), (6, 6));
    }

    #[test]
    fn test_dorbittrajectory_stm_storage_mut() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        // Before enabling, should be None
        assert!(traj.stm_storage_mut().is_none());

        traj.enable_stm_storage();

        // After enabling, should be Some
        let storage = traj.stm_storage_mut();
        assert!(storage.is_some());
        assert_eq!(storage.unwrap().len(), 1);
    }

    // ========== SensitivityStorage Trait Tests ==========

    #[test]
    fn test_dorbittrajectory_enable_sensitivity_storage() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        assert!(traj.sensitivities.is_none());
        traj.enable_sensitivity_storage(4);
        assert!(traj.sensitivities.is_some());
        assert_eq!(traj.sensitivity_dimension, Some((6, 4)));

        // Should have zero matrix for existing state
        let sens = traj.sensitivity_at_idx(0).unwrap();
        assert_eq!(sens.nrows(), 6);
        assert_eq!(sens.ncols(), 4);
    }

    #[test]
    #[should_panic(expected = "Parameter dimension must be > 0")]
    fn test_dorbittrajectory_enable_sensitivity_storage_panic_zero() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        traj.enable_sensitivity_storage(0);
    }

    #[test]
    fn test_dorbittrajectory_sensitivity_at_idx_valid() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);
        traj.enable_sensitivity_storage(3);

        let sens = traj.sensitivity_at_idx(0);
        assert!(sens.is_some());
        assert_eq!(sens.unwrap().nrows(), 6);
        assert_eq!(sens.unwrap().ncols(), 3);
    }

    #[test]
    fn test_dorbittrajectory_sensitivity_at_idx_none_disabled() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        // Without enabling sensitivity storage
        let sens = traj.sensitivity_at_idx(0);
        assert!(sens.is_none());
    }

    #[test]
    fn test_dorbittrajectory_set_sensitivity_at() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        let mut custom_sens = DMatrix::zeros(6, 3);
        custom_sens[(0, 0)] = 1.5;
        custom_sens[(2, 1)] = 2.5;

        traj.set_sensitivity_at(0, custom_sens.clone());

        let retrieved = traj.sensitivity_at_idx(0).unwrap();
        assert_abs_diff_eq!(retrieved[(0, 0)], 1.5, epsilon = 1e-10);
        assert_abs_diff_eq!(retrieved[(2, 1)], 2.5, epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_dorbittrajectory_set_sensitivity_at_panic_out_of_bounds() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let sens = DMatrix::zeros(6, 3);
        traj.set_sensitivity_at(0, sens); // No states added yet
    }

    #[test]
    #[should_panic(expected = "does not match state dimension 6")]
    fn test_dorbittrajectory_set_sensitivity_at_panic_wrong_rows() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        let wrong_sens = DMatrix::zeros(5, 3); // Wrong row count
        traj.set_sensitivity_at(0, wrong_sens);
    }

    #[test]
    #[should_panic(expected = "does not match existing")]
    fn test_dorbittrajectory_set_sensitivity_at_panic_wrong_cols() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch1 = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch1, state.clone());
        traj.add(epoch2, state);

        // Set first with 3 columns
        traj.set_sensitivity_at(0, DMatrix::zeros(6, 3));
        // Try to set second with 4 columns - should panic
        traj.set_sensitivity_at(1, DMatrix::zeros(6, 4));
    }

    #[test]
    fn test_dorbittrajectory_sensitivity_dimensions() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);

        // Before enabling, should be None
        assert_eq!(traj.sensitivity_dimensions(), None);

        traj.enable_sensitivity_storage(5);
        assert_eq!(traj.sensitivity_dimensions(), Some((6, 5)));
    }

    #[test]
    fn test_dorbittrajectory_sensitivity_storage_mut() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        // Before enabling, should be None
        assert!(traj.sensitivity_storage_mut().is_none());

        traj.enable_sensitivity_storage(2);

        // After enabling, should be Some
        let storage = traj.sensitivity_storage_mut();
        assert!(storage.is_some());
        assert_eq!(storage.unwrap().len(), 1);
    }

    // ========== CovarianceInterpolationConfig Tests ==========

    #[test]
    fn test_dorbittrajectory_covariance_interpolation_with_builder() {
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
            .with_covariance_interpolation_method(CovarianceInterpolationMethod::MatrixSquareRoot);

        assert_eq!(
            traj.get_covariance_interpolation_method(),
            CovarianceInterpolationMethod::MatrixSquareRoot
        );
    }

    #[test]
    fn test_dorbittrajectory_covariance_interpolation_set_get() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);

        // Default is TwoWasserstein
        assert_eq!(
            traj.get_covariance_interpolation_method(),
            CovarianceInterpolationMethod::TwoWasserstein
        );

        traj.set_covariance_interpolation_method(CovarianceInterpolationMethod::MatrixSquareRoot);
        assert_eq!(
            traj.get_covariance_interpolation_method(),
            CovarianceInterpolationMethod::MatrixSquareRoot
        );
    }

    #[test]
    fn test_dorbittrajectory_covariance_interpolation_sqrt_method() {
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
            .with_covariance_interpolation_method(CovarianceInterpolationMethod::MatrixSquareRoot);

        assert_eq!(
            traj.get_covariance_interpolation_method(),
            CovarianceInterpolationMethod::MatrixSquareRoot
        );
    }

    #[test]
    fn test_dorbittrajectory_covariance_interpolation_wasserstein_method() {
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
            .with_covariance_interpolation_method(CovarianceInterpolationMethod::TwoWasserstein);

        assert_eq!(
            traj.get_covariance_interpolation_method(),
            CovarianceInterpolationMethod::TwoWasserstein
        );
    }

    // ========== from_orbital_data Tests ==========

    #[test]
    fn test_dorbittrajectory_from_orbital_data_basic() {
        let epochs = vec![
            Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC),
            Epoch::from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, TimeSystem::UTC),
        ];
        let states = vec![
            DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]),
            DVector::from_vec(vec![7100e3, 0.0, 0.0, 0.0, 7.4e3, 0.0]),
        ];

        let traj = DOrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            None,
        );

        assert_eq!(traj.len(), 2);
        assert_eq!(traj.dimension(), 6);
        assert_eq!(traj.frame, OrbitFrame::ECI);
        assert_eq!(traj.representation, OrbitRepresentation::Cartesian);
    }

    #[test]
    fn test_dorbittrajectory_from_orbital_data_with_covariances() {
        let epochs = vec![
            Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC),
            Epoch::from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, TimeSystem::UTC),
        ];
        let states = vec![
            DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]),
            DVector::from_vec(vec![7100e3, 0.0, 0.0, 0.0, 7.4e3, 0.0]),
        ];
        let covariances = vec![
            DMatrix::identity(6, 6) * 100.0,
            DMatrix::identity(6, 6) * 200.0,
        ];

        let traj = DOrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            Some(covariances),
        );

        assert!(traj.covariances.is_some());
        assert_eq!(traj.covariances.as_ref().unwrap().len(), 2);
    }

    #[test]
    #[should_panic(expected = "Cannot create trajectory from empty states")]
    fn test_dorbittrajectory_from_orbital_data_panic_empty_states() {
        let epochs: Vec<Epoch> = vec![];
        let states: Vec<DVector<f64>> = vec![];

        DOrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            None,
        );
    }

    #[test]
    #[should_panic(expected = "State dimension must be at least 6")]
    fn test_dorbittrajectory_from_orbital_data_panic_dimension_too_small() {
        let epochs = vec![Epoch::from_datetime(
            2024,
            1,
            1,
            12,
            0,
            0.0,
            0.0,
            TimeSystem::UTC,
        )];
        let states = vec![DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3])]; // 5D

        DOrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            None,
        );
    }

    #[test]
    #[should_panic(expected = "has dimension")]
    fn test_dorbittrajectory_from_orbital_data_panic_inconsistent_dimension() {
        let epochs = vec![
            Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC),
            Epoch::from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, TimeSystem::UTC),
        ];
        let states = vec![
            DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]), // 6D
            DVector::from_vec(vec![7100e3, 0.0, 0.0, 0.0, 7.4e3, 0.0, 1.0]), // 7D
        ];

        DOrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            None,
        );
    }

    #[test]
    #[should_panic(expected = "Keplerian elements should be in ECI frame")]
    fn test_dorbittrajectory_from_orbital_data_panic_ecef_keplerian() {
        let epochs = vec![Epoch::from_datetime(
            2024,
            1,
            1,
            12,
            0,
            0.0,
            0.0,
            TimeSystem::UTC,
        )];
        let states = vec![DVector::from_vec(vec![7000e3, 0.01, 0.5, 0.0, 0.0, 0.0])];

        DOrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECEF,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Radians),
            None,
        );
    }

    #[test]
    #[should_panic(expected = "Covariances length")]
    fn test_dorbittrajectory_from_orbital_data_panic_cov_length_mismatch() {
        let epochs = vec![
            Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC),
            Epoch::from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, TimeSystem::UTC),
        ];
        let states = vec![
            DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]),
            DVector::from_vec(vec![7100e3, 0.0, 0.0, 0.0, 7.4e3, 0.0]),
        ];
        let covariances = vec![DMatrix::identity(6, 6)]; // Only 1 covariance for 2 states

        DOrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            Some(covariances),
        );
    }

    #[test]
    #[should_panic(expected = "Covariances are only supported for")]
    fn test_dorbittrajectory_from_orbital_data_panic_cov_invalid_frame() {
        let epochs = vec![Epoch::from_datetime(
            2024,
            1,
            1,
            12,
            0,
            0.0,
            0.0,
            TimeSystem::UTC,
        )];
        let states = vec![DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])];
        let covariances = vec![DMatrix::identity(6, 6)];

        DOrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECEF,
            OrbitRepresentation::Cartesian,
            None,
            Some(covariances),
        );
    }

    // ========== Frame Conversion Tests ==========

    #[test]
    fn test_dorbittrajectory_to_eci_already_eci() {
        setup_global_test_eop();

        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state.clone());

        let converted = traj.to_eci();

        assert_eq!(converted.frame, OrbitFrame::ECI);
        assert_eq!(converted.representation, OrbitRepresentation::Cartesian);
        for i in 0..6 {
            assert_abs_diff_eq!(converted.states[0][i], state[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_dorbittrajectory_to_eci_already_gcrf() {
        setup_global_test_eop();

        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::GCRF, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state.clone());

        let converted = traj.to_eci();

        // GCRF and ECI are equivalent, so states should be the same
        assert_eq!(converted.frame, OrbitFrame::ECI);
        for i in 0..6 {
            assert_abs_diff_eq!(converted.states[0][i], state[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_dorbittrajectory_to_eci_from_keplerian() {
        setup_global_test_eop();

        let mut traj = DOrbitTrajectory::new(
            6,
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Radians),
        );
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        // a, e, i, raan, argp, M
        let state = DVector::from_vec(vec![7000e3, 0.01, 0.5, 0.0, 0.0, 0.0]);
        traj.add(epoch, state);

        let converted = traj.to_eci();

        assert_eq!(converted.frame, OrbitFrame::ECI);
        assert_eq!(converted.representation, OrbitRepresentation::Cartesian);
        // Cartesian state should have reasonable orbital values
        assert!(converted.states[0][0].abs() > 1e6); // Position should be in meters
    }

    #[test]
    fn test_dorbittrajectory_to_eci_from_ecef() {
        setup_global_test_eop();

        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECEF, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        let converted = traj.to_eci();

        assert_eq!(converted.frame, OrbitFrame::ECI);
        assert_eq!(converted.representation, OrbitRepresentation::Cartesian);
    }

    #[test]
    fn test_dorbittrajectory_to_gcrf_already_gcrf() {
        setup_global_test_eop();

        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::GCRF, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state.clone());

        let converted = traj.to_gcrf();

        assert_eq!(converted.frame, OrbitFrame::GCRF);
        for i in 0..6 {
            assert_abs_diff_eq!(converted.states[0][i], state[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_dorbittrajectory_to_gcrf_from_eci() {
        setup_global_test_eop();

        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state.clone());

        let converted = traj.to_gcrf();

        assert_eq!(converted.frame, OrbitFrame::GCRF);
        // ECI and GCRF are equivalent
        for i in 0..6 {
            assert_abs_diff_eq!(converted.states[0][i], state[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_dorbittrajectory_to_ecef_already_ecef() {
        setup_global_test_eop();

        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECEF, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state.clone());

        let converted = traj.to_ecef();

        assert_eq!(converted.frame, OrbitFrame::ECEF);
        for i in 0..6 {
            assert_abs_diff_eq!(converted.states[0][i], state[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_dorbittrajectory_to_ecef_from_eci() {
        setup_global_test_eop();

        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        let converted = traj.to_ecef();

        assert_eq!(converted.frame, OrbitFrame::ECEF);
        assert_eq!(converted.representation, OrbitRepresentation::Cartesian);
    }

    #[test]
    fn test_dorbittrajectory_to_itrf_already_itrf() {
        setup_global_test_eop();

        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ITRF, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state.clone());

        let converted = traj.to_itrf();

        assert_eq!(converted.frame, OrbitFrame::ITRF);
        for i in 0..6 {
            assert_abs_diff_eq!(converted.states[0][i], state[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_dorbittrajectory_to_itrf_from_eci() {
        setup_global_test_eop();

        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        let converted = traj.to_itrf();

        assert_eq!(converted.frame, OrbitFrame::ITRF);
    }

    #[test]
    fn test_dorbittrajectory_to_eme2000_already_eme2000() {
        setup_global_test_eop();

        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::EME2000, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state.clone());

        let converted = traj.to_eme2000();

        assert_eq!(converted.frame, OrbitFrame::EME2000);
        for i in 0..6 {
            assert_abs_diff_eq!(converted.states[0][i], state[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_dorbittrajectory_to_eme2000_from_eci() {
        setup_global_test_eop();

        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        let converted = traj.to_eme2000();

        assert_eq!(converted.frame, OrbitFrame::EME2000);
    }

    #[test]
    fn test_dorbittrajectory_to_keplerian_from_cartesian_eci() {
        setup_global_test_eop();

        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        let converted = traj.to_keplerian(AngleFormat::Radians);

        assert_eq!(converted.representation, OrbitRepresentation::Keplerian);
        assert_eq!(converted.angle_format, Some(AngleFormat::Radians));
        // Semi-major axis should be positive
        assert!(converted.states[0][0] > 0.0);
    }

    #[test]
    fn test_dorbittrajectory_to_keplerian_already_keplerian_same_format() {
        setup_global_test_eop();

        let mut traj = DOrbitTrajectory::new(
            6,
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Radians),
        );
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.01, 0.5, 0.0, 0.0, 0.0]);
        traj.add(epoch, state.clone());

        let converted = traj.to_keplerian(AngleFormat::Radians);

        assert_eq!(converted.representation, OrbitRepresentation::Keplerian);
        for i in 0..6 {
            assert_abs_diff_eq!(converted.states[0][i], state[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_dorbittrajectory_to_keplerian_angle_format_conversion() {
        setup_global_test_eop();

        let mut traj = DOrbitTrajectory::new(
            6,
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Radians),
        );
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        // a, e, i, raan, argp, M (radians)
        let state = DVector::from_vec(vec![
            7000e3,
            0.01,
            std::f64::consts::PI / 4.0,
            0.0,
            0.0,
            0.0,
        ]);
        traj.add(epoch, state);

        let converted = traj.to_keplerian(AngleFormat::Degrees);

        assert_eq!(converted.angle_format, Some(AngleFormat::Degrees));
        // Inclination in degrees should be 45
        assert_abs_diff_eq!(converted.states[0][2], 45.0, epsilon = 1e-6);
    }

    // ========== Identifiable Trait Tests ==========

    #[test]
    fn test_dorbittrajectory_with_name() {
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
            .with_name("TestTrajectory");

        assert_eq!(traj.get_name(), Some("TestTrajectory"));
    }

    #[test]
    fn test_dorbittrajectory_with_uuid() {
        let test_uuid = uuid::Uuid::new_v4();
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
            .with_uuid(test_uuid);

        assert_eq!(traj.get_uuid(), Some(test_uuid));
    }

    #[test]
    fn test_dorbittrajectory_with_new_uuid() {
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
            .with_new_uuid();

        assert!(traj.get_uuid().is_some());
    }

    #[test]
    fn test_dorbittrajectory_with_id() {
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
            .with_id(42);

        assert_eq!(traj.get_id(), Some(42));
    }

    #[test]
    fn test_dorbittrajectory_with_identity() {
        let test_uuid = uuid::Uuid::new_v4();
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
            .with_identity(Some("Name"), Some(test_uuid), Some(1));

        assert_eq!(traj.get_id(), Some(1));
        assert_eq!(traj.get_name(), Some("Name"));
        assert_eq!(traj.get_uuid(), Some(test_uuid));
    }

    #[test]
    fn test_dorbittrajectory_set_identity() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let test_uuid = uuid::Uuid::new_v4();
        traj.set_identity(Some("NewName"), Some(test_uuid), Some(10));

        assert_eq!(traj.get_id(), Some(10));
        assert_eq!(traj.get_name(), Some("NewName"));
        assert_eq!(traj.get_uuid(), Some(test_uuid));
    }

    #[test]
    fn test_dorbittrajectory_set_id() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        traj.set_id(Some(99));
        assert_eq!(traj.get_id(), Some(99));
    }

    #[test]
    fn test_dorbittrajectory_set_name() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        traj.set_name(Some("SetName"));
        assert_eq!(traj.get_name(), Some("SetName"));
    }

    #[test]
    fn test_dorbittrajectory_generate_uuid() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        assert!(traj.get_uuid().is_none());
        traj.generate_uuid();
        assert!(traj.get_uuid().is_some());
    }

    #[test]
    fn test_dorbittrajectory_get_id_name_uuid() {
        let test_uuid = uuid::Uuid::new_v4();
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
            .with_id(5)
            .with_name("GetTest")
            .with_uuid(test_uuid);

        assert_eq!(traj.get_id(), Some(5));
        assert_eq!(traj.get_name(), Some("GetTest"));
        assert_eq!(traj.get_uuid(), Some(test_uuid));
    }

    // ========== DStateProvider Trait Tests ==========

    #[test]
    fn test_dorbittrajectory_state_provider_state() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 100.0, 200.0, 10.0, 7.5e3, 5.0]);
        traj.add(epoch, state.clone());

        let retrieved = traj.state(epoch).unwrap();
        for i in 0..6 {
            assert_abs_diff_eq!(retrieved[i], state[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_dorbittrajectory_state_provider_state_dim() {
        // state_dim returns dimension from first state, or 6 if empty
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        assert_eq!(traj.state_dim(), 6); // default for empty

        // Add a 9D state to verify dimension is correctly detected
        let mut traj9 =
            DOrbitTrajectory::new(9, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0, 0.1, 0.2, 0.3]);
        traj9.add(epoch, state);
        assert_eq!(traj9.state_dim(), 9);
    }

    #[test]
    fn test_dorbittrajectory_state_provider_state_error() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        // Query epoch before trajectory
        let before = Epoch::from_datetime(2024, 1, 1, 11, 0, 0.0, 0.0, TimeSystem::UTC);
        let result = traj.state(before);
        assert!(result.is_err());
    }

    // ========== DCovarianceProvider Trait Tests ==========

    #[test]
    fn test_dorbittrajectory_covariance_provider_basic() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        traj.covariances = Some(Vec::new());

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let cov = DMatrix::identity(6, 6) * 100.0;
        traj.add_state_and_covariance(epoch, state, cov);

        let retrieved = traj.covariance(epoch).unwrap();
        assert_abs_diff_eq!(retrieved[(0, 0)], 100.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dorbittrajectory_covariance_provider_error_not_enabled() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        // Covariance storage not enabled
        let result = traj.covariance(epoch);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not enabled"));
    }

    #[test]
    fn test_dorbittrajectory_covariance_provider_error_before_start() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        traj.covariances = Some(Vec::new());

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let cov = DMatrix::identity(6, 6);
        traj.add_state_and_covariance(epoch, state, cov);

        let before = Epoch::from_datetime(2024, 1, 1, 11, 0, 0.0, 0.0, TimeSystem::UTC);
        let result = traj.covariance(before);
        assert!(result.is_err());
    }

    #[test]
    fn test_dorbittrajectory_covariance_provider_error_after_end() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        traj.covariances = Some(Vec::new());

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let cov = DMatrix::identity(6, 6);
        traj.add_state_and_covariance(epoch, state, cov);

        let after = Epoch::from_datetime(2024, 1, 1, 13, 0, 0.0, 0.0, TimeSystem::UTC);
        let result = traj.covariance(after);
        assert!(result.is_err());
    }

    #[test]
    fn test_dorbittrajectory_covariance_provider_dim() {
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        assert_eq!(traj.covariance_dim(), 6);
    }

    // ========== DOrbitStateProvider Trait Tests ==========

    #[test]
    fn test_dorbittrajectory_state_eci_from_eci_cartesian() {
        setup_global_test_eop();

        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 100.0, 200.0, 10.0, 7.5e3, 5.0]);
        traj.add(epoch, state.clone());

        let retrieved = traj.state_eci(epoch).unwrap();
        for i in 0..6 {
            assert_abs_diff_eq!(retrieved[i], state[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_dorbittrajectory_state_eci_from_gcrf_cartesian() {
        setup_global_test_eop();

        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::GCRF, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state.clone());

        let retrieved = traj.state_eci(epoch).unwrap();
        // GCRF is equivalent to ECI
        for i in 0..6 {
            assert_abs_diff_eq!(retrieved[i], state[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_dorbittrajectory_state_eci_from_eci_keplerian() {
        setup_global_test_eop();

        let mut traj = DOrbitTrajectory::new(
            6,
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Radians),
        );
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.01, 0.5, 0.0, 0.0, 0.0]);
        traj.add(epoch, state);

        let retrieved = traj.state_eci(epoch);
        assert!(retrieved.is_ok());
        // Should be Cartesian
        assert!(retrieved.unwrap()[0].abs() > 1e6);
    }

    #[test]
    fn test_dorbittrajectory_state_eci_from_ecef_cartesian() {
        setup_global_test_eop();

        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECEF, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        let retrieved = traj.state_eci(epoch);
        assert!(retrieved.is_ok());
    }

    #[test]
    fn test_dorbittrajectory_state_gcrf_from_gcrf_cartesian() {
        setup_global_test_eop();

        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::GCRF, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state.clone());

        let retrieved = traj.state_gcrf(epoch).unwrap();
        for i in 0..6 {
            assert_abs_diff_eq!(retrieved[i], state[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_dorbittrajectory_state_gcrf_from_eci_cartesian() {
        setup_global_test_eop();

        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state.clone());

        let retrieved = traj.state_gcrf(epoch).unwrap();
        // ECI is equivalent to GCRF
        for i in 0..6 {
            assert_abs_diff_eq!(retrieved[i], state[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_dorbittrajectory_state_ecef_from_ecef_cartesian() {
        setup_global_test_eop();

        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECEF, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state.clone());

        let retrieved = traj.state_ecef(epoch).unwrap();
        for i in 0..6 {
            assert_abs_diff_eq!(retrieved[i], state[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_dorbittrajectory_state_ecef_from_eci_cartesian() {
        setup_global_test_eop();

        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        let retrieved = traj.state_ecef(epoch);
        assert!(retrieved.is_ok());
    }

    #[test]
    fn test_dorbittrajectory_state_itrf_from_itrf_cartesian() {
        setup_global_test_eop();

        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ITRF, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state.clone());

        let retrieved = traj.state_itrf(epoch).unwrap();
        for i in 0..6 {
            assert_abs_diff_eq!(retrieved[i], state[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_dorbittrajectory_state_eme2000_from_eme2000_cartesian() {
        setup_global_test_eop();

        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::EME2000, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state.clone());

        let retrieved = traj.state_eme2000(epoch).unwrap();
        for i in 0..6 {
            assert_abs_diff_eq!(retrieved[i], state[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_dorbittrajectory_state_koe_osc_from_eci_cartesian() {
        setup_global_test_eop();

        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        let koe = traj.state_koe_osc(epoch, AngleFormat::Radians).unwrap();
        // Semi-major axis should be positive
        assert!(koe[0] > 0.0);
    }

    #[test]
    fn test_dorbittrajectory_state_koe_osc_from_eci_keplerian_same_format() {
        setup_global_test_eop();

        let mut traj = DOrbitTrajectory::new(
            6,
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Radians),
        );
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.01, 0.5, 0.0, 0.0, 0.0]);
        traj.add(epoch, state.clone());

        let koe = traj.state_koe_osc(epoch, AngleFormat::Radians).unwrap();
        for i in 0..6 {
            assert_abs_diff_eq!(koe[i], state[i], epsilon = 1e-6);
        }
    }

    // ========== DOrbitCovarianceProvider Trait Tests ==========

    #[test]
    fn test_dorbittrajectory_covariance_eci_from_eci() {
        setup_global_test_eop();

        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        traj.covariances = Some(Vec::new());

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let cov = DMatrix::identity(6, 6) * 100.0;
        traj.add_state_and_covariance(epoch, state, cov);

        let retrieved = traj.covariance_eci(epoch).unwrap();
        assert_abs_diff_eq!(retrieved[(0, 0)], 100.0, epsilon = 1e-6);
    }

    #[test]
    fn test_dorbittrajectory_covariance_eci_from_gcrf() {
        setup_global_test_eop();

        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::GCRF, OrbitRepresentation::Cartesian, None);
        traj.covariances = Some(Vec::new());

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let cov = DMatrix::identity(6, 6) * 100.0;
        traj.add_state_and_covariance(epoch, state, cov);

        let retrieved = traj.covariance_eci(epoch).unwrap();
        // GCRF = ECI
        assert_abs_diff_eq!(retrieved[(0, 0)], 100.0, epsilon = 1e-6);
    }

    #[test]
    fn test_dorbittrajectory_covariance_eci_error_ecef() {
        setup_global_test_eop();

        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECEF, OrbitRepresentation::Cartesian, None);
        traj.covariances = Some(Vec::new());

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let cov = DMatrix::identity(6, 6);
        traj.add_state_and_covariance(epoch, state, cov);

        // ECEF covariances cannot be transformed to ECI
        let result = traj.covariance_eci(epoch);
        assert!(result.is_err());
    }

    #[test]
    fn test_dorbittrajectory_covariance_gcrf() {
        setup_global_test_eop();

        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::GCRF, OrbitRepresentation::Cartesian, None);
        traj.covariances = Some(Vec::new());

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let cov = DMatrix::identity(6, 6) * 100.0;
        traj.add_state_and_covariance(epoch, state, cov);

        let retrieved = traj.covariance_gcrf(epoch).unwrap();
        assert_abs_diff_eq!(retrieved[(0, 0)], 100.0, epsilon = 1e-6);
    }

    #[test]
    fn test_dorbittrajectory_covariance_rtn() {
        setup_global_test_eop();

        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        traj.covariances = Some(Vec::new());

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let cov = DMatrix::identity(6, 6) * 100.0;
        traj.add_state_and_covariance(epoch, state, cov);

        let retrieved = traj.covariance_rtn(epoch);
        assert!(retrieved.is_ok());
    }

    // ========== Iterator and Index Tests ==========

    #[test]
    fn test_dorbittrajectory_index_valid() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 100.0, 200.0, 10.0, 7.5e3, 5.0]);
        traj.add(epoch, state.clone());

        // Index returns &DVector<f64> (state only, not epoch)
        let idx_state = &traj[0];
        for i in 0..6 {
            assert_abs_diff_eq!(idx_state[i], state[i], epsilon = 1e-10);
        }
    }

    #[test]
    #[should_panic]
    fn test_dorbittrajectory_index_panic_out_of_bounds() {
        let traj = DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let _ = &traj[0]; // Empty trajectory
    }

    #[test]
    fn test_dorbittrajectory_into_iter() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch1 = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch1, state.clone());
        traj.add(epoch2, state);

        let mut count = 0;
        for (e, _s) in &traj {
            count += 1;
            assert!(e == epoch1 || e == epoch2);
        }
        assert_eq!(count, 2);
    }

    #[test]
    fn test_dorbittrajectory_iterator_size_hint() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        for i in 0..5 {
            let epoch = Epoch::from_datetime(2024, 1, 1, 12, i, 0.0, 0.0, TimeSystem::UTC);
            let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
            traj.add(epoch, state);
        }

        let iter = traj.into_iter();
        assert_eq!(iter.size_hint(), (5, Some(5)));
    }

    #[test]
    fn test_dorbittrajectory_iterator_len() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        for i in 0..3 {
            let epoch = Epoch::from_datetime(2024, 1, 1, 12, i, 0.0, 0.0, TimeSystem::UTC);
            let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
            traj.add(epoch, state);
        }

        let iter = traj.into_iter();
        assert_eq!(iter.len(), 3);
    }

    #[test]
    fn test_dorbittrajectory_iterator_next() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        let mut iter = traj.into_iter();
        let first = iter.next();
        assert!(first.is_some());

        let (e, _s) = first.unwrap();
        assert_eq!(e, epoch);

        let second = iter.next();
        assert!(second.is_none());
    }

    // ========== Acceleration Storage Tests ==========

    #[test]
    fn test_dorbittrajectory_enable_acceleration_storage() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);

        assert!(traj.accelerations.is_none());
        traj.enable_acceleration_storage(3);
        assert!(traj.accelerations.is_some());
        assert_eq!(traj.acceleration_dimension, Some(3));
    }

    #[test]
    #[should_panic(expected = "does not match")]
    fn test_dorbittrajectory_enable_acceleration_storage_panic_mismatch() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        traj.enable_acceleration_storage(3);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let accel = DVector::from_vec(vec![1.0, 2.0]); // 2D instead of 3D
        traj.add_with_acceleration(epoch, state, accel);
    }

    #[test]
    fn test_dorbittrajectory_has_accelerations() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        assert!(!traj.has_accelerations());

        traj.enable_acceleration_storage(3);
        assert!(traj.has_accelerations());
    }

    #[test]
    fn test_dorbittrajectory_acceleration_dim() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        assert_eq!(traj.acceleration_dim(), None);

        traj.enable_acceleration_storage(4);
        assert_eq!(traj.acceleration_dim(), Some(4));
    }

    #[test]
    fn test_dorbittrajectory_acceleration_at_idx() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        traj.enable_acceleration_storage(3);

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let accel = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        traj.add_with_acceleration(epoch, state, accel);

        let retrieved = traj.acceleration_at_idx(0);
        assert!(retrieved.is_some());
        assert_abs_diff_eq!(retrieved.unwrap()[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(retrieved.unwrap()[2], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dorbittrajectory_set_acceleration_at() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        // Enable and set acceleration
        traj.enable_acceleration_storage(3);
        let accel = DVector::from_vec(vec![5.0, 6.0, 7.0]);
        traj.set_acceleration_at(0, accel);

        let retrieved = traj.acceleration_at_idx(0).unwrap();
        assert_abs_diff_eq!(retrieved[0], 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(retrieved[2], 7.0, epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "Acceleration dimension")]
    fn test_dorbittrajectory_set_acceleration_at_panic_dimension() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        traj.add(epoch, state);

        traj.enable_acceleration_storage(3);
        let wrong_accel = DVector::from_vec(vec![1.0, 2.0]); // Wrong dimension
        traj.set_acceleration_at(0, wrong_accel);
    }

    #[test]
    fn test_dorbittrajectory_add_with_acceleration() {
        let mut traj =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        traj.enable_acceleration_storage(3);

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let accel = DVector::from_vec(vec![-9.8, 0.0, 0.0]);

        traj.add_with_acceleration(epoch, state.clone(), accel.clone());

        assert_eq!(traj.len(), 1);
        let retrieved_accel = traj.acceleration_at_idx(0).unwrap();
        assert_abs_diff_eq!(retrieved_accel[0], -9.8, epsilon = 1e-10);
    }
}
