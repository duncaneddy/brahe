/*!
 * Static orbital trajectory implementation for 6-dimensional orbital state vectors.
 *
 * This module provides a compile-time sized, specialized trajectory container for orbital
 * mechanics applications, using static `SVector<f64, 6>` and `SMatrix<f64, 6, 6>` types
 * for maximum performance. For a dynamic (runtime-sized) alternative, see `DOrbitTrajectory`.
 *
 * # Key Features
 * - Reference frame conversions (ECI ↔ ECEF)
 * - State representation conversions (Cartesian ↔ Keplerian)
 * - Angle format conversions (Radians ↔ Degrees)
 * - Position and velocity extraction from Cartesian states
 * - Combined conversions for efficiency
 *
 * # Examples
 * ```rust
 * use brahe::trajectories::SOrbitTrajectory;
 * use brahe::traits::{Trajectory, OrbitalTrajectory, OrbitFrame, OrbitRepresentation};
 * use brahe::AngleFormat;
 * use brahe::time::{Epoch, TimeSystem};
 * use nalgebra::Vector6;
 *
 * // Create orbital trajectory in ECI Cartesian coordinates
 * let mut traj = SOrbitTrajectory::new(
 *     OrbitFrame::ECI,
 *     OrbitRepresentation::Cartesian,
 *     None,
 * );
 *
 * // Add state
 * let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
 * let state = Vector6::new(6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0);
 * traj.add(epoch, state);
 *
 * // Convert to Keplerian in degrees
 * let kep_traj = traj.to_keplerian(AngleFormat::Degrees);
 * ```
 */

use nalgebra::{DMatrix, SMatrix, SVector, Vector6};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use uuid::Uuid;

use crate::constants::AngleFormat;
use crate::constants::{DEG2RAD, RAD2DEG};
use crate::coordinates::{state_cartesian_to_osculating, state_osculating_to_cartesian};
use crate::frames::{
    rotation_eme2000_to_gcrf, state_ecef_to_eci, state_eci_to_ecef, state_eme2000_to_gcrf,
    state_gcrf_to_eme2000, state_gcrf_to_itrf, state_itrf_to_gcrf,
};
use crate::math::{
    CovarianceInterpolationConfig, interpolate_covariance_sqrt_smatrix,
    interpolate_covariance_two_wasserstein_smatrix,
};
use crate::propagators::traits::{
    SCovarianceProvider, SOrbitCovarianceProvider, SOrbitStateProvider, SStateProvider,
};
use crate::relative_motion::rotation_eci_to_rtn;
use crate::time::Epoch;
use crate::utils::{BraheError, Identifiable};

use super::traits::{
    CovarianceInterpolationMethod, InterpolatableTrajectory, InterpolationConfig,
    InterpolationMethod, OrbitFrame, OrbitRepresentation, OrbitalTrajectory, Trajectory,
    TrajectoryEvictionPolicy,
};

/// Static (compile-time sized) orbital trajectory container.
///
/// This struct uses static `SVector<f64, 6>` and `SMatrix<f64, 6, 6>` types for maximum
/// performance. It provides orbital-specific functionality including conversions between
/// reference frames (ECI/ECEF), state representations (Cartesian/Keplerian), and angle
/// formats (radians/degrees).
///
/// For a dynamic (runtime-sized) alternative, see `DOrbitTrajectory`.
#[derive(Debug, Clone, PartialEq)]
pub struct SOrbitTrajectory {
    /// Time epochs for each state, maintained in chronological order.
    /// All epochs should use consistent time systems for meaningful interpolation.
    pub epochs: Vec<Epoch>,

    /// R-dimensional state vectors corresponding to epochs.
    /// Units and interpretation depend on the specific use case:
    /// - Cartesian: [m, m, m, m/s, m/s, m/s]
    /// - Keplerian: [m, dimensionless, rad or deg, rad or deg, rad or deg, rad or deg]
    pub states: Vec<SVector<f64, 6>>,

    /// Optional covariance matrices corresponding to states.
    /// Each covariance is a 6x6 symmetric matrix representing state uncertainty.
    /// Units: [m², m²/s, m²/s², etc.] for Cartesian states.
    /// If present, must have same length as states vector.
    pub covariances: Option<Vec<SMatrix<f64, 6, 6>>>,

    /// Optional state transition matrices (STM) corresponding to each state.
    /// If present, must have the same length as `states` and each matrix is 6x6.
    pub stms: Option<Vec<SMatrix<f64, 6, 6>>>,

    /// Optional sensitivity matrices corresponding to each state.
    /// If present, must have the same length as `states`.
    /// Each matrix has shape (6 x param_dim). Uses DMatrix because param_dim varies at runtime.
    pub sensitivities: Option<Vec<DMatrix<f64>>>,

    /// Parameter dimension for sensitivity matrices.
    /// Set when sensitivity storage is enabled.
    sensitivity_param_dim: Option<usize>,

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
}

impl fmt::Display for SOrbitTrajectory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SOrbitTrajectory(frame={}, representation={}, states={})",
            self.frame,
            self.representation,
            self.len()
        )
    }
}

impl SOrbitTrajectory {
    /// Creates a new orbital trajectory with specified frame, representation, and angle format.
    ///
    /// # Arguments
    /// * `frame` - Reference frame (ECI or ECEF)
    /// * `representation` - State representation (Cartesian or Keplerian)
    /// * `angle_format` - Angle format (None for Cartesian, Radians/Degrees for Keplerian)
    ///
    /// # Returns
    /// * `Ok(SOrbitTrajectory)` - New empty orbital trajectory
    /// * `Err(BraheError)` - If parameters are invalid
    ///
    /// # Examples
    /// ```rust
    /// use brahe::trajectories::SOrbitTrajectory;
    /// use brahe::traits::{OrbitFrame, OrbitRepresentation};
    /// use brahe::AngleFormat;
    ///
    /// let traj = SOrbitTrajectory::new(
    ///     OrbitFrame::ECI,
    ///     OrbitRepresentation::Cartesian,
    ///     None,
    /// );
    /// ```
    pub fn new(
        frame: OrbitFrame,
        representation: OrbitRepresentation,
        angle_format: Option<AngleFormat>,
    ) -> Self {
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
            sensitivity_param_dim: None,
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
    /// use brahe::trajectories::SOrbitTrajectory;
    /// use brahe::traits::{OrbitFrame, OrbitRepresentation, InterpolationMethod};
    /// let traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
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
    /// use brahe::trajectories::SOrbitTrajectory;
    /// use brahe::traits::{OrbitFrame, OrbitRepresentation};
    /// let traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
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
    /// use brahe::trajectories::SOrbitTrajectory;
    /// use brahe::traits::{OrbitFrame, OrbitRepresentation};
    /// let traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
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

    /// Returns the dimension of state vectors in this trajectory.
    /// Always returns 6 for orbital state vectors (position + velocity).
    pub fn dimension(&self) -> usize {
        6
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
    /// use brahe::trajectories::SOrbitTrajectory;
    /// use brahe::traits::{OrbitFrame, OrbitRepresentation};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use nalgebra::{SMatrix, SVector};
    ///
    /// let mut traj = SOrbitTrajectory::new(
    ///     OrbitFrame::ECI,
    ///     OrbitRepresentation::Cartesian,
    ///     None,
    /// );
    ///
    /// // Initialize covariances
    /// traj.covariances = Some(Vec::new());
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = SVector::<f64, 6>::zeros();
    /// let cov = SMatrix::<f64, 6, 6>::identity();
    ///
    /// traj.add_state_and_covariance(epoch, state, cov);
    /// ```
    pub fn add_state_and_covariance(
        &mut self,
        epoch: Epoch,
        state: SVector<f64, 6>,
        covariance: SMatrix<f64, 6, 6>,
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
    ///
    /// Initializes the STM vector with identity matrices for all existing states.
    /// After calling this, STMs can be added using `add_full()` or `set_stm_at()`.
    pub fn enable_stm_storage(&mut self) {
        if self.stms.is_none() {
            // Initialize with identity matrices for all existing states
            let identity = SMatrix::<f64, 6, 6>::identity();
            self.stms = Some(vec![identity; self.states.len()]);
        }
    }

    /// Enable sensitivity matrix storage with specified parameter dimension
    ///
    /// Initializes the sensitivity vector with zero matrices for all existing states.
    /// After calling this, sensitivity matrices can be added using `add_full()` or
    /// `set_sensitivity_at()`.
    ///
    /// # Arguments
    /// * `param_dim` - Number of parameters (number of columns in sensitivity matrices)
    ///
    /// # Panics
    /// Panics if param_dim is zero
    pub fn enable_sensitivity_storage(&mut self, param_dim: usize) {
        if param_dim == 0 {
            panic!("Parameter dimension must be > 0");
        }
        if self.sensitivities.is_none() {
            let zero_sens = DMatrix::zeros(6, param_dim);
            self.sensitivities = Some(vec![zero_sens; self.states.len()]);
            self.sensitivity_param_dim = Some(param_dim);
        }
    }

    /// Set STM at a specific index
    ///
    /// Enables STM storage if not already enabled.
    ///
    /// # Arguments
    /// * `index` - Index in the trajectory
    /// * `stm` - State transition matrix (6x6)
    ///
    /// # Panics
    /// Panics if index is out of bounds
    pub fn set_stm_at(&mut self, index: usize, stm: SMatrix<f64, 6, 6>) {
        if index >= self.states.len() {
            panic!(
                "Index {} out of bounds for trajectory with {} states",
                index,
                self.states.len()
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

    /// Set sensitivity matrix at a specific index
    ///
    /// Enables sensitivity storage if not already enabled.
    ///
    /// # Arguments
    /// * `index` - Index in the trajectory
    /// * `sensitivity` - Sensitivity matrix (6 x param_dim)
    ///
    /// # Panics
    /// Panics if index is out of bounds or sensitivity dimensions are incorrect
    pub fn set_sensitivity_at(&mut self, index: usize, sensitivity: DMatrix<f64>) {
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
        if let Some(existing_cols) = self.sensitivity_param_dim
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

    /// Get STM at a specific index
    ///
    /// Returns None if STM storage is not enabled.
    pub fn stm_at_idx(&self, index: usize) -> Option<&SMatrix<f64, 6, 6>> {
        self.stms.as_ref()?.get(index)
    }

    /// Get sensitivity matrix at a specific index
    ///
    /// Returns None if sensitivity storage is not enabled.
    pub fn sensitivity_at_idx(&self, index: usize) -> Option<&DMatrix<f64>> {
        self.sensitivities.as_ref()?.get(index)
    }

    /// Get STM at a specific epoch (with linear interpolation)
    ///
    /// Returns None if STM storage is not enabled or epoch is out of range.
    ///
    /// # Arguments
    /// * `epoch` - Time epoch to query
    ///
    /// # Returns
    /// STM at the requested epoch (interpolated if necessary)
    pub fn stm_at(&self, epoch: Epoch) -> Option<SMatrix<f64, 6, 6>> {
        let stms = self.stms.as_ref()?;

        if self.epochs.is_empty() {
            return None;
        }

        // Handle exact match
        if let Some((idx, _)) = self.epochs.iter().enumerate().find(|(_, e)| **e == epoch) {
            return Some(stms[idx]);
        }

        // Find surrounding indices for interpolation
        let (idx_before, idx_after) = self.find_surrounding_indices(epoch)?;

        // Handle exact matches
        if self.epochs[idx_before] == epoch {
            return Some(stms[idx_before]);
        }
        if self.epochs[idx_after] == epoch {
            return Some(stms[idx_after]);
        }

        // Linear interpolation parameter
        let t0 = self.epochs[idx_before] - self.epoch_initial()?;
        let t1 = self.epochs[idx_after] - self.epoch_initial()?;
        let t = epoch - self.epoch_initial()?;
        let alpha = (t - t0) / (t1 - t0);

        // Linear interpolation: Φ(t) = (1-α)*Φ₀ + α*Φ₁
        let stm = stms[idx_before] * (1.0 - alpha) + stms[idx_after] * alpha;
        Some(stm)
    }

    /// Get sensitivity matrix at a specific epoch (with linear interpolation)
    ///
    /// Returns None if sensitivity storage is not enabled or epoch is out of range.
    ///
    /// # Arguments
    /// * `epoch` - Time epoch to query
    ///
    /// # Returns
    /// Sensitivity matrix at the requested epoch (interpolated if necessary)
    pub fn sensitivity_at(&self, epoch: Epoch) -> Option<DMatrix<f64>> {
        let sens = self.sensitivities.as_ref()?;

        if self.epochs.is_empty() {
            return None;
        }

        // Handle exact match
        if let Some((idx, _)) = self.epochs.iter().enumerate().find(|(_, e)| **e == epoch) {
            return Some(sens[idx].clone());
        }

        // Find surrounding indices for interpolation
        let (idx_before, idx_after) = self.find_surrounding_indices(epoch)?;

        // Handle exact matches
        if self.epochs[idx_before] == epoch {
            return Some(sens[idx_before].clone());
        }
        if self.epochs[idx_after] == epoch {
            return Some(sens[idx_after].clone());
        }

        // Linear interpolation parameter
        let t0 = self.epochs[idx_before] - self.epoch_initial()?;
        let t1 = self.epochs[idx_after] - self.epoch_initial()?;
        let t = epoch - self.epoch_initial()?;
        let alpha = (t - t0) / (t1 - t0);

        // Linear interpolation: S(t) = (1-α)*S₀ + α*S₁
        let sensitivity = &sens[idx_before] * (1.0 - alpha) + &sens[idx_after] * alpha;
        Some(sensitivity)
    }

    /// Add a complete state record with all optional data
    ///
    /// This is the most flexible method, allowing any combination of
    /// covariance, STM, and sensitivity to be provided or omitted.
    /// Automatically enables storage for any provided data.
    ///
    /// # Arguments
    /// * `epoch` - Time epoch
    /// * `state` - State vector (6 elements)
    /// * `covariance` - Optional covariance matrix (6x6)
    /// * `stm` - Optional state transition matrix (6x6)
    /// * `sensitivity` - Optional sensitivity matrix (6 x param_dim)
    ///
    /// # Panics
    /// Panics if dimensions don't match
    pub fn add_full(
        &mut self,
        epoch: Epoch,
        state: SVector<f64, 6>,
        covariance: Option<SMatrix<f64, 6, 6>>,
        stm: Option<SMatrix<f64, 6, 6>>,
        sensitivity: Option<DMatrix<f64>>,
    ) {
        // Validate and enable storage as needed
        if covariance.is_some() && self.covariances.is_none() {
            self.covariances = Some(vec![SMatrix::<f64, 6, 6>::zeros(); self.states.len()]);
        }

        if stm.is_some() && self.stms.is_none() {
            self.enable_stm_storage();
        }

        if let Some(ref sens) = sensitivity {
            if sens.nrows() != 6 {
                panic!("Sensitivity row dimension mismatch");
            }
            if let Some(cols) = self.sensitivity_param_dim
                && sens.ncols() != cols
            {
                panic!("Sensitivity column dimension mismatch");
            }
            if self.sensitivities.is_none() {
                self.enable_sensitivity_storage(sens.ncols());
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
            let cov_val = covariance.unwrap_or_else(SMatrix::<f64, 6, 6>::zeros);
            covs.insert(insert_idx, cov_val);
        }

        if let Some(ref mut stms) = self.stms {
            let stm_val = stm.unwrap_or_else(SMatrix::<f64, 6, 6>::identity);
            stms.insert(insert_idx, stm_val);
        }

        if let Some(ref mut sens) = self.sensitivities {
            let param_dim = self.sensitivity_param_dim.unwrap();
            let sens_val = sensitivity.unwrap_or_else(|| DMatrix::zeros(6, param_dim));
            sens.insert(insert_idx, sens_val);
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
                    let new_states: Vec<SVector<f64, 6>> =
                        indices_to_keep.iter().map(|&i| self.states[i]).collect();
                    let new_covariances = self
                        .covariances
                        .as_ref()
                        .map(|covs| indices_to_keep.iter().map(|&i| covs[i]).collect());
                    let new_stms = self
                        .stms
                        .as_ref()
                        .map(|stms| indices_to_keep.iter().map(|&i| stms[i]).collect());
                    let new_sensitivities = self
                        .sensitivities
                        .as_ref()
                        .map(|sens| indices_to_keep.iter().map(|&i| sens[i].clone()).collect());

                    self.epochs = new_epochs;
                    self.states = new_states;
                    self.covariances = new_covariances;
                    self.stms = new_stms;
                    self.sensitivities = new_sensitivities;
                }
            }
        }
    }
}

impl Default for SOrbitTrajectory {
    /// Creates a default orbital trajectory in ECI Cartesian with no angle format.
    fn default() -> Self {
        Self::new(
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
impl std::ops::Index<usize> for SOrbitTrajectory {
    type Output = SVector<f64, 6>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.states[index]
    }
}

/// Iterator over trajectory (epoch, state) pairs
pub struct SOrbitTrajectoryIterator<'a> {
    trajectory: &'a SOrbitTrajectory,
    index: usize,
}

impl<'a> Iterator for SOrbitTrajectoryIterator<'a> {
    type Item = (Epoch, SVector<f64, 6>);

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

impl<'a> ExactSizeIterator for SOrbitTrajectoryIterator<'a> {
    fn len(&self) -> usize {
        self.trajectory.len() - self.index
    }
}

/// IntoIterator implementation for iterating over (epoch, state) pairs
impl<'a> IntoIterator for &'a SOrbitTrajectory {
    type Item = (Epoch, SVector<f64, 6>);
    type IntoIter = SOrbitTrajectoryIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        SOrbitTrajectoryIterator {
            trajectory: self,
            index: 0,
        }
    }
}

// Passthrough implementations for Trajectory trait
impl Trajectory for SOrbitTrajectory {
    type StateVector = Vector6<f64>;

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

        // Ensure epochs are sorted
        let mut indices: Vec<usize> = (0..epochs.len()).collect();
        indices.sort_by(|&i, &j| epochs[i].partial_cmp(&epochs[j]).unwrap());

        let sorted_epochs: Vec<Epoch> = indices.iter().map(|&i| epochs[i]).collect();
        let sorted_states: Vec<SVector<f64, 6>> = indices.iter().map(|&i| states[i]).collect();

        Ok(Self {
            epochs: sorted_epochs,
            states: sorted_states,
            covariances: None,
            stms: None,
            sensitivities: None,
            sensitivity_param_dim: None,
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
        })
    }

    fn add(&mut self, epoch: Epoch, state: Self::StateVector) {
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

        Ok(self.states[index])
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

        Ok((self.epochs[nearest_idx], self.states[nearest_idx]))
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
            Some((self.epochs[0], self.states[0]))
        }
    }

    fn last(&self) -> Option<(Epoch, Self::StateVector)> {
        if self.epochs.is_empty() {
            None
        } else {
            let last_index = self.epochs.len() - 1;
            Some((self.epochs[last_index], self.states[last_index]))
        }
    }

    fn clear(&mut self) {
        self.epochs.clear();
        self.states.clear();
    }

    fn remove_epoch(&mut self, epoch: &Epoch) -> Result<Self::StateVector, BraheError> {
        // This could be improved with binary search since epochs are sorted
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

        Ok((self.epochs[index], self.states[index]))
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
impl InterpolationConfig for SOrbitTrajectory {
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

// InterpolatableTrajectory uses default implementations for interpolate and interpolate_linear
impl InterpolatableTrajectory for SOrbitTrajectory {}

// Implementation of CovarianceInterpolationConfig trait
impl CovarianceInterpolationConfig for SOrbitTrajectory {
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

// Implementation of OrbitalTrajectory trait
impl OrbitalTrajectory for SOrbitTrajectory {
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
    /// * `Ok(SOrbitTrajectory)` - New orbital trajectory with data
    /// * `Err(BraheError)` - If parameters are invalid or data validation fails
    ///
    /// # Panics
    /// * If covariances are provided but frame is not ECI or GCRF
    /// * If covariances length does not match states length
    fn from_orbital_data(
        epochs: Vec<Epoch>,
        states: Vec<Vector6<f64>>,
        frame: OrbitFrame,
        representation: OrbitRepresentation,
        angle_format: Option<AngleFormat>,
        covariances: Option<Vec<SMatrix<f64, 6, 6>>>,
    ) -> Self {
        // Validate inputs
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
            sensitivity_param_dim: None,
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
        }
    }

    fn to_eci(&self) -> Self
    where
        Self: Sized,
    {
        let states_converted = match self.representation {
            OrbitRepresentation::Keplerian => {
                let mut states_converted = Vec::with_capacity(self.states.len());
                // Just need to convert to Cartesian below
                for (_e, s) in self.into_iter() {
                    let state_cartesian = state_osculating_to_cartesian(
                        s,
                        self.angle_format
                            .expect("Keplerian representation must have angle_format"),
                    );
                    states_converted.push(state_cartesian);
                }
                states_converted
            }
            OrbitRepresentation::Cartesian => {
                match self.frame {
                    OrbitFrame::EME2000 => {
                        let mut states_converted = Vec::with_capacity(self.states.len());
                        // EME2000 Cartesian to GCRF Cartesian (no epoch needed)
                        for (_e, s) in self.into_iter() {
                            let state_gcrf = state_eme2000_to_gcrf(s);
                            states_converted.push(state_gcrf);
                        }
                        states_converted
                    }
                    OrbitFrame::ITRF | OrbitFrame::ECEF => {
                        let mut states_converted = Vec::with_capacity(self.states.len());
                        // ITRF/ECEF Cartesian to GCRF Cartesian (requires epoch)
                        for (e, s) in self.into_iter() {
                            let state_gcrf = state_itrf_to_gcrf(e, s);
                            states_converted.push(state_gcrf);
                        }
                        states_converted
                    }
                    OrbitFrame::ECI => {
                        // No need to convert ECI to GCRF, they are equivalent for our purposes
                        self.states.clone()
                    }
                    OrbitFrame::GCRF => {
                        // Already in GCRF frame
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
            sensitivity_param_dim: None,
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
        }
    }

    fn to_gcrf(&self) -> Self
    where
        Self: Sized,
    {
        let states_converted = match self.representation {
            OrbitRepresentation::Keplerian => {
                let mut states_converted = Vec::with_capacity(self.states.len());
                // Just need to convert to Cartesian below
                for (_e, s) in self.into_iter() {
                    let state_cartesian = state_osculating_to_cartesian(
                        s,
                        self.angle_format
                            .expect("Keplerian representation must have angle_format"),
                    );
                    states_converted.push(state_cartesian);
                }
                states_converted
            }
            OrbitRepresentation::Cartesian => {
                match self.frame {
                    OrbitFrame::EME2000 => {
                        let mut states_converted = Vec::with_capacity(self.states.len());
                        // EME2000 Cartesian to GCRF Cartesian (no epoch needed)
                        for (_e, s) in self.into_iter() {
                            let state_gcrf = state_eme2000_to_gcrf(s);
                            states_converted.push(state_gcrf);
                        }
                        states_converted
                    }
                    OrbitFrame::ITRF | OrbitFrame::ECEF => {
                        let mut states_converted = Vec::with_capacity(self.states.len());
                        // ITRF/ECEF Cartesian to GCRF Cartesian (requires epoch)
                        for (e, s) in self.into_iter() {
                            let state_gcrf = state_itrf_to_gcrf(e, s);
                            states_converted.push(state_gcrf);
                        }
                        states_converted
                    }
                    OrbitFrame::ECI => {
                        // No need to convert ECI to GCRF, they are equivalent for our purposes
                        self.states.clone()
                    }
                    OrbitFrame::GCRF => {
                        // Already in GCRF frame
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
            sensitivity_param_dim: None,
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
        }
    }

    fn to_ecef(&self) -> Self
    where
        Self: Sized,
    {
        let states_converted = match self.representation {
            OrbitRepresentation::Keplerian => {
                let mut states_converted = Vec::with_capacity(self.states.len());
                // Just need to convert to Cartesian below
                for (e, s) in self.into_iter() {
                    let state_eci = state_osculating_to_cartesian(
                        s,
                        self.angle_format
                            .expect("Keplerian representation must have angle_format"),
                    );
                    states_converted.push(state_eci_to_ecef(e, state_eci));
                }
                states_converted
            }
            OrbitRepresentation::Cartesian => {
                match self.frame {
                    OrbitFrame::EME2000 => {
                        let mut states_converted = Vec::with_capacity(self.states.len());
                        // EME2000 Cartesian to GCRF Cartesian (no epoch needed)
                        for (e, s) in self.into_iter() {
                            let state_itrf = state_gcrf_to_itrf(e, state_eme2000_to_gcrf(s));
                            states_converted.push(state_itrf);
                        }
                        states_converted
                    }
                    OrbitFrame::ITRF | OrbitFrame::ECEF => {
                        // Already in ITRF frame
                        self.states.clone()
                    }
                    OrbitFrame::ECI | OrbitFrame::GCRF => {
                        let mut states_converted = Vec::with_capacity(self.states.len());
                        // ITRF/ECEF Cartesian to GCRF Cartesian (requires epoch)
                        for (e, s) in self.into_iter() {
                            let state_itrf = state_gcrf_to_itrf(e, s);
                            states_converted.push(state_itrf);
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
            sensitivity_param_dim: None,
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
        }
    }

    fn to_itrf(&self) -> Self
    where
        Self: Sized,
    {
        let states_converted = match self.representation {
            OrbitRepresentation::Keplerian => {
                let mut states_converted = Vec::with_capacity(self.states.len());
                // Keplerian to Cartesian (in GCRF/ECI), then GCRF to ITRF
                for (e, s) in self.into_iter() {
                    let state_cartesian = state_osculating_to_cartesian(
                        s,
                        self.angle_format
                            .expect("Keplerian representation must have angle_format"),
                    );
                    let state_itrf = state_gcrf_to_itrf(e, state_cartesian);
                    states_converted.push(state_itrf);
                }
                states_converted
            }
            OrbitRepresentation::Cartesian => {
                match self.frame {
                    OrbitFrame::EME2000 => {
                        let mut states_converted = Vec::with_capacity(self.states.len());
                        // EME2000 Cartesian to GCRF Cartesian (no epoch needed)
                        for (e, s) in self.into_iter() {
                            let state_itrf = state_gcrf_to_itrf(e, state_eme2000_to_gcrf(s));
                            states_converted.push(state_itrf);
                        }
                        states_converted
                    }
                    OrbitFrame::ITRF | OrbitFrame::ECEF => {
                        // Already in ITRF frame
                        self.states.clone()
                    }
                    OrbitFrame::ECI | OrbitFrame::GCRF => {
                        let mut states_converted = Vec::with_capacity(self.states.len());
                        // ITRF/ECEF Cartesian to GCRF Cartesian (requires epoch)
                        for (e, s) in self.into_iter() {
                            let state_itrf = state_gcrf_to_itrf(e, s);
                            states_converted.push(state_itrf);
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
            sensitivity_param_dim: None,
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
        }
    }

    fn to_eme2000(&self) -> Self
    where
        Self: Sized,
    {
        let states_converted = match self.representation {
            OrbitRepresentation::Keplerian => {
                let mut states_converted = Vec::with_capacity(self.states.len());
                // Just need to convert to Cartesian below
                for (_e, s) in self.into_iter() {
                    let state_cartesian = state_gcrf_to_eme2000(state_osculating_to_cartesian(
                        s,
                        self.angle_format
                            .expect("Keplerian representation must have angle_format"),
                    ));
                    states_converted.push(state_cartesian);
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
                        // ITRF/ECEF Cartesian to GCRF Cartesian (requires epoch)
                        for (e, s) in self.into_iter() {
                            let state_gcrf = state_gcrf_to_eme2000(state_itrf_to_gcrf(e, s));
                            states_converted.push(state_gcrf);
                        }
                        states_converted
                    }
                    OrbitFrame::ECI | OrbitFrame::GCRF => {
                        let mut states_converted = Vec::with_capacity(self.states.len());
                        // ECI/GCRF Cartesian to EME2000 Cartesian (no epoch needed)
                        for (_e, s) in self.into_iter() {
                            let state_eme2000 = state_gcrf_to_eme2000(s);
                            states_converted.push(state_eme2000);
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
            sensitivity_param_dim: None,
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
        }
    }

    fn to_keplerian(&self, angle_format: AngleFormat) -> Self
    where
        Self: Sized,
    {
        let states_converted = match self.representation {
            OrbitRepresentation::Keplerian => {
                match self.angle_format {
                    Some(current_format) if current_format == angle_format => {
                        // Already in desired format
                        self.states.clone()
                    }
                    Some(current_format) => {
                        let mut states_converted = Vec::with_capacity(self.states.len());
                        // Convert angles
                        for (_e, s) in self.into_iter() {
                            let mut state_converted = s;
                            if current_format == AngleFormat::Degrees
                                && angle_format == AngleFormat::Radians
                            {
                                // Degrees to Radians
                                state_converted[2] *= DEG2RAD;
                                state_converted[3] *= DEG2RAD;
                                state_converted[4] *= DEG2RAD;
                                state_converted[5] *= DEG2RAD;
                            } else if current_format == AngleFormat::Radians
                                && angle_format == AngleFormat::Degrees
                            {
                                // Radians to Degrees
                                state_converted[2] *= RAD2DEG;
                                state_converted[3] *= RAD2DEG;
                                state_converted[4] *= RAD2DEG;
                                state_converted[5] *= RAD2DEG;
                            }
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
                        // ITRF/ECEF Cartesian to GCRF Cartesian (requires epoch)
                        for (_e, s) in self.into_iter() {
                            let state = state_cartesian_to_osculating(
                                state_eme2000_to_gcrf(s),
                                angle_format,
                            );
                            states_converted.push(state);
                        }
                        states_converted
                    }
                    OrbitFrame::ITRF | OrbitFrame::ECEF => {
                        let mut states_converted = Vec::with_capacity(self.states.len());
                        // ITRF/ECEF Cartesian to GCRF Cartesian (requires epoch)
                        for (e, s) in self.into_iter() {
                            let state = state_cartesian_to_osculating(
                                state_ecef_to_eci(e, s),
                                angle_format,
                            );
                            states_converted.push(state);
                        }
                        states_converted
                    }
                    OrbitFrame::ECI | OrbitFrame::GCRF => {
                        let mut states_converted = Vec::with_capacity(self.states.len());
                        // ITRF/ECEF Cartesian to GCRF Cartesian (requires epoch)
                        for (_e, s) in self.into_iter() {
                            let state = state_cartesian_to_osculating(s, angle_format);
                            states_converted.push(state);
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
            sensitivity_param_dim: None,
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
        }
    }
}

impl SStateProvider for SOrbitTrajectory {
    fn state(&self, epoch: Epoch) -> Vector6<f64> {
        // Interpolate state in native frame/representation
        self.interpolate(&epoch)
            .unwrap_or_else(|_| Vector6::zeros())
    }
}

impl SOrbitStateProvider for SOrbitTrajectory {
    fn state_eci(&self, epoch: Epoch) -> Vector6<f64> {
        // Get state in native format then convert to ECI Cartesian
        let state = self
            .interpolate(&epoch)
            .unwrap_or_else(|_| Vector6::zeros());

        match (self.frame, self.representation) {
            (OrbitFrame::ECI, OrbitRepresentation::Cartesian) => state,
            (OrbitFrame::GCRF, OrbitRepresentation::Cartesian) => state,
            (OrbitFrame::ECI, OrbitRepresentation::Keplerian) => state_osculating_to_cartesian(
                state,
                self.angle_format
                    .expect("Keplerian representation must have angle_format"),
            ),
            (OrbitFrame::GCRF, OrbitRepresentation::Keplerian) => state_osculating_to_cartesian(
                state,
                self.angle_format
                    .expect("Keplerian representation must have angle_format"),
            ),
            (OrbitFrame::EME2000, OrbitRepresentation::Keplerian) => {
                state_eme2000_to_gcrf(state_osculating_to_cartesian(
                    state,
                    self.angle_format
                        .expect("Keplerian representation must have angle_format"),
                ))
            }
            (OrbitFrame::ECEF, OrbitRepresentation::Cartesian) => state_ecef_to_eci(epoch, state),
            (OrbitFrame::ITRF, OrbitRepresentation::Cartesian) => state_itrf_to_gcrf(epoch, state),
            (OrbitFrame::EME2000, OrbitRepresentation::Cartesian) => state_eme2000_to_gcrf(state),
            (OrbitFrame::ECEF, OrbitRepresentation::Keplerian) => {
                panic!("Keplerian element trajectories should be in an inertial frame")
            }
            (OrbitFrame::ITRF, OrbitRepresentation::Keplerian) => {
                panic!("Keplerian element trajectories should be in an inertial frame")
            }
        }
    }

    fn state_gcrf(&self, epoch: Epoch) -> Vector6<f64> {
        // Get state in native format then convert to GCRF Cartesian
        let state = self
            .interpolate(&epoch)
            .unwrap_or_else(|_| Vector6::zeros());

        match (self.frame, self.representation) {
            (OrbitFrame::GCRF, OrbitRepresentation::Cartesian) => state,
            (OrbitFrame::ECI, OrbitRepresentation::Cartesian) => state, // ECI treated as GCRF
            (OrbitFrame::GCRF, OrbitRepresentation::Keplerian) => state_osculating_to_cartesian(
                state,
                self.angle_format
                    .expect("Keplerian representation must have angle_format"),
            ),
            (OrbitFrame::ECI, OrbitRepresentation::Keplerian) => state_osculating_to_cartesian(
                state,
                self.angle_format
                    .expect("Keplerian representation must have angle_format"),
            ),
            (OrbitFrame::EME2000, OrbitRepresentation::Keplerian) => {
                state_eme2000_to_gcrf(state_osculating_to_cartesian(
                    state,
                    self.angle_format
                        .expect("Keplerian representation must have angle_format"),
                ))
            }
            (OrbitFrame::EME2000, OrbitRepresentation::Cartesian) => state_eme2000_to_gcrf(state),
            (OrbitFrame::ITRF, OrbitRepresentation::Cartesian) => state_itrf_to_gcrf(epoch, state),
            (OrbitFrame::ECEF, OrbitRepresentation::Cartesian) => state_itrf_to_gcrf(epoch, state),
            (OrbitFrame::ECEF, OrbitRepresentation::Keplerian) => {
                panic!("Keplerian element trajectories should be in an inertial frame")
            }
            (OrbitFrame::ITRF, OrbitRepresentation::Keplerian) => {
                panic!("Keplerian element trajectories should be in an inertial frame")
            }
        }
    }

    fn state_ecef(&self, epoch: Epoch) -> Vector6<f64> {
        // Get state in native format then convert to ECEF Cartesian
        let state = self
            .interpolate(&epoch)
            .unwrap_or_else(|_| Vector6::zeros());

        match (self.frame, self.representation) {
            (OrbitFrame::ECEF, OrbitRepresentation::Cartesian) => state,
            (OrbitFrame::ITRF, OrbitRepresentation::Cartesian) => state,
            (OrbitFrame::ECI, OrbitRepresentation::Cartesian) => state_eci_to_ecef(epoch, state),
            (OrbitFrame::GCRF, OrbitRepresentation::Cartesian) => state_gcrf_to_itrf(epoch, state),
            (OrbitFrame::EME2000, OrbitRepresentation::Cartesian) => {
                let state_gcrf = state_eme2000_to_gcrf(state);
                state_gcrf_to_itrf(epoch, state_gcrf)
            }
            (OrbitFrame::ECI, OrbitRepresentation::Keplerian) => {
                let state_eci_cart = state_osculating_to_cartesian(
                    state,
                    self.angle_format
                        .expect("Keplerian representation must have angle_format"),
                );
                state_eci_to_ecef(epoch, state_eci_cart)
            }
            (OrbitFrame::EME2000, OrbitRepresentation::Keplerian) => {
                let state_eme2000_cart = state_osculating_to_cartesian(
                    state,
                    self.angle_format
                        .expect("Keplerian representation must have angle_format"),
                );
                let state_gcrf = state_eme2000_to_gcrf(state_eme2000_cart);
                state_gcrf_to_itrf(epoch, state_gcrf)
            }
            (OrbitFrame::GCRF, OrbitRepresentation::Keplerian) => {
                let state_gcrf_cart = state_osculating_to_cartesian(
                    state,
                    self.angle_format
                        .expect("Keplerian representation must have angle_format"),
                );
                state_gcrf_to_itrf(epoch, state_gcrf_cart)
            }
            (OrbitFrame::ECEF, OrbitRepresentation::Keplerian) => {
                panic!("Keplerian element trajectories should be in an inertial frame")
            }
            (OrbitFrame::ITRF, OrbitRepresentation::Keplerian) => {
                panic!("Keplerian element trajectories should be in an inertial frame")
            }
        }
    }

    fn state_itrf(&self, epoch: Epoch) -> Vector6<f64> {
        // Get state in native format then convert to ECEF Cartesian
        let state = self
            .interpolate(&epoch)
            .unwrap_or_else(|_| Vector6::zeros());

        match (self.frame, self.representation) {
            (OrbitFrame::ECEF, OrbitRepresentation::Cartesian) => state,
            (OrbitFrame::ITRF, OrbitRepresentation::Cartesian) => state,
            (OrbitFrame::ECI, OrbitRepresentation::Cartesian) => state_eci_to_ecef(epoch, state),
            (OrbitFrame::GCRF, OrbitRepresentation::Cartesian) => state_gcrf_to_itrf(epoch, state),
            (OrbitFrame::EME2000, OrbitRepresentation::Cartesian) => {
                let state_gcrf = state_eme2000_to_gcrf(state);
                state_gcrf_to_itrf(epoch, state_gcrf)
            }
            (OrbitFrame::ECI, OrbitRepresentation::Keplerian) => {
                let state_eci_cart = state_osculating_to_cartesian(
                    state,
                    self.angle_format
                        .expect("Keplerian representation must have angle_format"),
                );
                state_eci_to_ecef(epoch, state_eci_cart)
            }
            (OrbitFrame::GCRF, OrbitRepresentation::Keplerian) => {
                let state_gcrf_cart = state_osculating_to_cartesian(
                    state,
                    self.angle_format
                        .expect("Keplerian representation must have angle_format"),
                );
                state_gcrf_to_itrf(epoch, state_gcrf_cart)
            }
            (OrbitFrame::EME2000, OrbitRepresentation::Keplerian) => {
                let state_eme2000_cart = state_osculating_to_cartesian(
                    state,
                    self.angle_format
                        .expect("Keplerian representation must have angle_format"),
                );
                let state_gcrf = state_eme2000_to_gcrf(state_eme2000_cart);
                state_gcrf_to_itrf(epoch, state_gcrf)
            }
            (OrbitFrame::ECEF, OrbitRepresentation::Keplerian) => {
                panic!("Keplerian element trajectories should be in an inertial frame")
            }
            (OrbitFrame::ITRF, OrbitRepresentation::Keplerian) => {
                panic!("Keplerian element trajectories should be in an inertial frame")
            }
        }
    }

    fn state_eme2000(&self, epoch: Epoch) -> Vector6<f64> {
        // Get state in native format then convert to EME2000 Cartesian
        let state = self
            .interpolate(&epoch)
            .unwrap_or_else(|_| Vector6::zeros());

        match (self.frame, self.representation) {
            (OrbitFrame::EME2000, OrbitRepresentation::Cartesian) => state,
            (OrbitFrame::GCRF, OrbitRepresentation::Cartesian) => state_gcrf_to_eme2000(state),
            (OrbitFrame::ECI, OrbitRepresentation::Cartesian) => state_gcrf_to_eme2000(state), // ECI treated as GCRF
            (OrbitFrame::GCRF, OrbitRepresentation::Keplerian) => {
                let state_gcrf_cart = state_osculating_to_cartesian(
                    state,
                    self.angle_format
                        .expect("Keplerian representation must have angle_format"),
                );
                state_gcrf_to_eme2000(state_gcrf_cart)
            }
            (OrbitFrame::ECI, OrbitRepresentation::Keplerian) => {
                let state_eci_cart = state_osculating_to_cartesian(
                    state,
                    self.angle_format
                        .expect("Keplerian representation must have angle_format"),
                );
                state_gcrf_to_eme2000(state_eci_cart)
            }
            (OrbitFrame::EME2000, OrbitRepresentation::Keplerian) => state_osculating_to_cartesian(
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
                panic!("Keplerian element trajectories should be in an inertial frame")
            }
            (OrbitFrame::ITRF, OrbitRepresentation::Keplerian) => {
                panic!("Keplerian element trajectories should be in an inertial frame")
            }
        }
    }

    fn state_as_osculating_elements(
        &self,
        epoch: Epoch,
        angle_format: AngleFormat,
    ) -> Vector6<f64> {
        // Get state in native format then convert to osculating elements
        let state = self
            .interpolate(&epoch)
            .unwrap_or_else(|_| Vector6::zeros());

        match (self.frame, self.representation) {
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
                let state_eme2000_cart = state_osculating_to_cartesian(
                    state,
                    self.angle_format
                        .expect("Keplerian representation must have angle_format"),
                );
                let state_gcrf = state_eme2000_to_gcrf(state_eme2000_cart);
                state_cartesian_to_osculating(state_gcrf, angle_format)
            }
            (OrbitFrame::ECI, OrbitRepresentation::Cartesian) => {
                state_cartesian_to_osculating(state, angle_format)
            }
            (OrbitFrame::GCRF, OrbitRepresentation::Cartesian) => {
                state_cartesian_to_osculating(state, angle_format)
            }
            (OrbitFrame::EME2000, OrbitRepresentation::Cartesian) => {
                let state_gcrf = state_eme2000_to_gcrf(state);
                state_cartesian_to_osculating(state_gcrf, angle_format)
            }
            (OrbitFrame::ECEF, OrbitRepresentation::Cartesian) => {
                let state_eci = state_ecef_to_eci(epoch, state);
                state_cartesian_to_osculating(state_eci, angle_format)
            }
            (OrbitFrame::ITRF, OrbitRepresentation::Cartesian) => {
                let state_gcrf = state_itrf_to_gcrf(epoch, state);
                state_cartesian_to_osculating(state_gcrf, angle_format)
            }
            (OrbitFrame::ECEF, OrbitRepresentation::Keplerian) => {
                panic!("Keplerian element trajectories should be in an inertial frame")
            }
            (OrbitFrame::ITRF, OrbitRepresentation::Keplerian) => {
                panic!("Keplerian element trajectories should be in an inertial frame")
            }
        }
    }
}

// Implementation of SCovarianceProvider trait (base trait)
impl SCovarianceProvider for SOrbitTrajectory {
    fn covariance(&self, epoch: Epoch) -> Option<SMatrix<f64, 6, 6>> {
        // Return None if no covariances are stored
        let covs = self.covariances.as_ref()?;

        if covs.is_empty() {
            return None;
        }

        // Check for exact epoch match
        for (i, &e) in self.epochs.iter().enumerate() {
            if (e - epoch).abs() < 1e-9 {
                return Some(covs[i]);
            }
        }

        // Find bracketing indices for interpolation
        let idx_after = match self.index_after_epoch(&epoch) {
            Ok(idx) => idx,
            Err(_) => return None,
        };

        // Handle boundary cases - return None for epochs outside data range
        if idx_after == 0 {
            return None; // Epoch is before first data point
        }
        if idx_after >= self.epochs.len() {
            return None; // Epoch is after last data point
        }

        // Interpolate using matrix square root method
        let idx_before = idx_after - 1;
        let t0 = self.epochs[idx_before];
        let t1 = self.epochs[idx_after];
        let dt = t1 - t0;

        if dt.abs() < 1e-12 {
            return Some(covs[idx_before]);
        }

        // Compute interpolation parameter alpha
        let t = (epoch - t0) / dt;

        let cov0 = covs[idx_before];
        let cov1 = covs[idx_after];

        // Dispatch based on interpolation method
        let cov_interp = match self.covariance_interpolation_method {
            CovarianceInterpolationMethod::MatrixSquareRoot => {
                interpolate_covariance_sqrt_smatrix(cov0, cov1, t)
            }
            CovarianceInterpolationMethod::TwoWasserstein => {
                interpolate_covariance_two_wasserstein_smatrix(cov0, cov1, t)
            }
        };

        Some(cov_interp)
    }
}

// Implementation of SOrbitCovarianceProvider trait (frame-specific methods)
impl SOrbitCovarianceProvider for SOrbitTrajectory {
    fn covariance_eci(&self, epoch: Epoch) -> Option<SMatrix<f64, 6, 6>> {
        // Get covariance in native frame
        let cov_native = self.covariance(epoch)?;

        // Transform to ECI if needed
        match self.frame {
            OrbitFrame::ECI | OrbitFrame::GCRF => Some(cov_native),
            OrbitFrame::ECEF | OrbitFrame::ITRF => {
                panic!("Covariance transformation from ECEF/ITRF to ECI not implemented")
            }
            OrbitFrame::EME2000 => {
                // We just construct a block diagonal rotation matrix using the
                // EME2000 to GCRF rotation matrix
                let rot_eme2000_to_gcrf = rotation_eme2000_to_gcrf();

                let mut rot = nalgebra::Matrix6::zeros();
                // Position part (3x3 upper-left block)
                for i in 0..3 {
                    for j in 0..3 {
                        rot[(i, j)] = rot_eme2000_to_gcrf[(i, j)];
                        rot[(3 + i, 3 + j)] = rot_eme2000_to_gcrf[(i, j)];
                    }
                }
                // Transform covariance: C_ECI = R * C_EME2000 * R^T
                let cov_eci = rot * cov_native * rot.transpose();
                Some(cov_eci)
            }
        }
    }

    fn covariance_gcrf(&self, epoch: Epoch) -> Option<SMatrix<f64, 6, 6>> {
        // GCRF is essentially the same as ECI for our purposes
        self.covariance_eci(epoch)
    }

    fn covariance_rtn(&self, epoch: Epoch) -> Option<SMatrix<f64, 6, 6>> {
        // Get covariance in ECI/GCRF frame first
        // Note: because we go through covariance_eci, this will also handle EME2000 frame conversion
        // as well as erroring out for ECEF/ITRF frames
        let cov_eci = self.covariance_eci(epoch)?;

        // Get state in ECI/GCRF frame
        let state_eci = self.state_eci(epoch);

        // Get rotation matrix from ECI to RTN
        let rot_eci_to_rtn = rotation_eci_to_rtn(state_eci);

        // Extract position and velocity
        let r = state_eci.fixed_rows::<3>(0);
        let v = state_eci.fixed_rows::<3>(3);

        // Get angular velocity of RTN frame with respect to ECI frame (Alfriend equation 2.16)
        let f_dot = (r.cross(&v)).norm() / (r.norm().powi(2));
        let omega = nalgebra::Vector3::new(0.0, 0.0, f_dot);

        // Build skew-symmetric matrix of omega
        let omega_skew = SMatrix::<f64, 3, 3>::new(
            0.0, -omega[2], omega[1], omega[2], 0.0, -omega[0], -omega[1], omega[0], 0.0,
        );

        let j21 = -omega_skew * rot_eci_to_rtn;

        // Build full 6x6 Jacobian for ECI to RTN transformation
        let mut jacobian = nalgebra::Matrix6::zeros();

        // Block diagonal rotation parts
        for i in 0..3 {
            for j in 0..3 {
                jacobian[(i, j)] = rot_eci_to_rtn[(i, j)];
                jacobian[(3 + i, 3 + j)] = rot_eci_to_rtn[(i, j)];
            }
        }

        // Off-diagonal parts due to angular velocity
        for i in 3..6 {
            for j in 0..3 {
                jacobian[(i, j)] = j21[(i - 3, j)];
            }
        }

        // Transform covariance: C_RTN = J * C_ECI * J^T
        let cov_rtn = jacobian * cov_eci * jacobian.transpose();
        Some(cov_rtn)
    }
}

impl Identifiable for SOrbitTrajectory {
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

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::constants::{DEGREES, R_EARTH, RADIANS};
    use crate::time::{Epoch, TimeSystem};
    use crate::utils::testing::setup_global_test_eop;
    use approx::assert_abs_diff_eq;

    fn create_test_trajectory() -> SOrbitTrajectory {
        let mut traj = SOrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Degrees),
        );

        let epoch1 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state1 = Vector6::new(R_EARTH + 500e3, 0.001, 98.0, 15.0, 30.0, 45.0);
        traj.add(epoch1, state1);

        let epoch2 = Epoch::from_datetime(2023, 1, 1, 12, 10, 0.0, 0.0, TimeSystem::UTC);
        let state2 = Vector6::new(R_EARTH + 500e3, 0.001, 98.0, 15.0, 30.0, 60.0);
        traj.add(epoch2, state2);

        let epoch3 = Epoch::from_datetime(2023, 1, 1, 12, 20, 0.0, 0.0, TimeSystem::UTC);
        let state3 = Vector6::new(R_EARTH + 500e3, 0.001, 98.0, 15.0, 30.0, 75.0);
        traj.add(epoch3, state3);

        traj
    }

    #[test]
    fn test_orbittrajectory_new() {
        let traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);

        assert_eq!(traj.len(), 0);
        assert_eq!(traj.frame, OrbitFrame::ECI);
        assert_eq!(traj.representation, OrbitRepresentation::Cartesian);
        assert_eq!(traj.angle_format, None);
    }

    #[test]
    #[should_panic(expected = "Angle format must be specified for Keplerian elements")]
    fn test_orbittrajectory_new_invalid_keplerian_none() {
        SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Keplerian, None);
    }

    #[test]
    #[should_panic(expected = "Angle format should be None for Cartesian representation")]
    fn test_orbittrajectory_new_invalid_cartesian_degrees() {
        SOrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            Some(AngleFormat::Degrees),
        );
    }

    #[test]
    #[should_panic(expected = "Angle format should be None for Cartesian representation")]
    fn test_orbittrajectory_new_invalid_cartesian_radians() {
        SOrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            Some(AngleFormat::Radians),
        );
    }

    #[test]
    #[should_panic(expected = "Keplerian elements should be in ECI frame")]
    fn test_orbittrajectory_new_invalid_keplerian_ecef_degrees() {
        SOrbitTrajectory::new(
            OrbitFrame::ECEF,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Degrees),
        );
    }

    #[test]
    #[should_panic(expected = "Keplerian elements should be in ECI frame")]
    fn test_orbittrajectory_new_invalid_keplerian_ecef_radians() {
        SOrbitTrajectory::new(
            OrbitFrame::ECEF,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Radians),
        );
    }

    #[test]
    #[should_panic(expected = "Angle format must be specified for Keplerian elements")]
    fn test_orbittrajectory_new_invalid_keplerian_ecef_none() {
        SOrbitTrajectory::new(OrbitFrame::ECEF, OrbitRepresentation::Keplerian, None);
    }

    #[test]
    fn test_orbittrajetory_dimension() {
        let traj = create_test_trajectory();
        assert_eq!(traj.dimension(), 6);
    }

    #[test]
    fn test_orbittrajectory_to_matrix() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
            Epoch::from_jd(2451545.2, TimeSystem::UTC),
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
            Vector6::new(7200e3, 2000e3, 1000e3, 200.0, 7.7e3, 100.0),
        ];
        let traj = SOrbitTrajectory::from_orbital_data(
            epochs,
            states.clone(),
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            None,
        );

        // Convert to matrix
        let matrix = traj.to_matrix().unwrap();

        // Verify dimensions: 3 rows (time points) x 6 columns (state elements)
        assert_eq!(matrix.nrows(), 3);
        assert_eq!(matrix.ncols(), 6);

        // Verify first row matches first state
        assert_eq!(matrix[(0, 0)], states[0][0]);
        assert_eq!(matrix[(0, 1)], states[0][1]);
        assert_eq!(matrix[(0, 2)], states[0][2]);
        assert_eq!(matrix[(0, 3)], states[0][3]);
        assert_eq!(matrix[(0, 4)], states[0][4]);
        assert_eq!(matrix[(0, 5)], states[0][5]);

        // Verify second row matches second state
        assert_eq!(matrix[(1, 0)], states[1][0]);
        assert_eq!(matrix[(1, 1)], states[1][1]);
        assert_eq!(matrix[(1, 2)], states[1][2]);
        assert_eq!(matrix[(1, 3)], states[1][3]);
        assert_eq!(matrix[(1, 4)], states[1][4]);
        assert_eq!(matrix[(1, 5)], states[1][5]);

        // Verify third row matches third state
        assert_eq!(matrix[(2, 0)], states[2][0]);
        assert_eq!(matrix[(2, 1)], states[2][1]);
        assert_eq!(matrix[(2, 2)], states[2][2]);
        assert_eq!(matrix[(2, 3)], states[2][3]);
        assert_eq!(matrix[(2, 4)], states[2][4]);
        assert_eq!(matrix[(2, 5)], states[2][5]);

        // Verify first column contains first element of each state over time
        assert_eq!(matrix[(0, 0)], states[0][0]);
        assert_eq!(matrix[(1, 0)], states[1][0]);
        assert_eq!(matrix[(2, 0)], states[2][0]);
    }

    // Additional Trajectory Trait Tests

    #[test]
    fn test_orbittrajectory_trajectory_add() {
        let mut traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);

        // Add states in order
        let epoch1 = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state1 = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        traj.add(epoch1, state1);

        let epoch3 = Epoch::from_jd(2451545.2, TimeSystem::UTC);
        let state3 = Vector6::new(7200e3, 0.0, 0.0, 0.0, 7.7e3, 0.0);
        traj.add(epoch3, state3);

        // Add a state in between
        let epoch2 = Epoch::from_jd(2451545.1, TimeSystem::UTC);
        let state2 = Vector6::new(7100e3, 0.0, 0.0, 0.0, 7.6e3, 0.0);
        traj.add(epoch2, state2);

        assert_eq!(traj.len(), 3);
        let epochs = &traj.epochs;
        assert_eq!(epochs[0].jd(), 2451545.0);
        assert_eq!(epochs[1].jd(), 2451545.1);
        assert_eq!(epochs[2].jd(), 2451545.2);
    }

    #[test]
    fn test_orbittrajectory_trajectory_state() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
            Epoch::from_jd(2451545.2, TimeSystem::UTC),
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
            Vector6::new(7200e3, 2000e3, 1000e3, 200.0, 7.7e3, 100.0),
        ];
        let traj = SOrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            None,
        );

        // Test valid indices (use Trajectory::state to disambiguate from SOrbitStateProvider::state)
        let state0 = Trajectory::state_at_idx(&traj, 0).unwrap();
        assert_eq!(state0[0], 7000e3);

        let state1 = Trajectory::state_at_idx(&traj, 1).unwrap();
        assert_eq!(state1[0], 7100e3);

        let state2 = Trajectory::state_at_idx(&traj, 2).unwrap();
        assert_eq!(state2[0], 7200e3);

        // Test invalid index
        assert!(Trajectory::state_at_idx(&traj, 10).is_err());
    }

    #[test]
    fn test_orbittrajectory_trajectory_epoch() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
            Epoch::from_jd(2451545.2, TimeSystem::UTC),
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
            Vector6::new(7200e3, 2000e3, 1000e3, 200.0, 7.7e3, 100.0),
        ];
        let traj = SOrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            None,
        );

        // Test valid indices
        let epoch0 = traj.epoch_at_idx(0).unwrap();
        assert_eq!(epoch0.jd(), 2451545.0);

        let epoch1 = traj.epoch_at_idx(1).unwrap();
        assert_eq!(epoch1.jd(), 2451545.1);

        let epoch2 = traj.epoch_at_idx(2).unwrap();
        assert_eq!(epoch2.jd(), 2451545.2);

        // Test invalid index
        assert!(traj.epoch_at_idx(10).is_err());
    }

    #[test]
    fn test_orbittrajectory_trajectory_nearest_state() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
            Epoch::from_jd(2451545.2, TimeSystem::UTC),
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
            Vector6::new(7200e3, 2000e3, 1000e3, 200.0, 7.7e3, 100.0),
        ];
        let traj = SOrbitTrajectory::from_orbital_data(
            epochs.clone(),
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            None,
        );

        // Test before first epoch
        let test_epoch = Epoch::from_jd(2451544.9, TimeSystem::UTC);
        let (nearest_epoch, nearest_state) = traj.nearest_state(&test_epoch).unwrap();
        assert_eq!(nearest_epoch, epochs[0]);
        assert_eq!(nearest_state[0], 7000e3);

        // Test after last epoch
        let test_epoch = Epoch::from_jd(2451545.3, TimeSystem::UTC);
        let (nearest_epoch, nearest_state) = traj.nearest_state(&test_epoch).unwrap();
        assert_eq!(nearest_epoch, epochs[2]);
        assert_eq!(nearest_state[0], 7200e3);

        // Test between epochs
        let test_epoch = Epoch::from_jd(2451545.15, TimeSystem::UTC);
        let (nearest_epoch, nearest_state) = traj.nearest_state(&test_epoch).unwrap();
        assert_eq!(nearest_epoch, epochs[1]);
        assert_eq!(nearest_state[0], 7100e3);

        // Test exact match
        let test_epoch = Epoch::from_jd(2451545.1, TimeSystem::UTC);
        let (nearest_epoch, nearest_state) = traj.nearest_state(&test_epoch).unwrap();
        assert_eq!(nearest_epoch, epochs[1]);
        assert_eq!(nearest_state[0], 7100e3);

        // Test just before second epoch
        let test_epoch = Epoch::from_jd(2451545.0999, TimeSystem::UTC);
        let (nearest_epoch, nearest_state) = traj.nearest_state(&test_epoch).unwrap();
        assert_eq!(nearest_epoch, epochs[1]);
        assert_eq!(nearest_state[0], 7100e3);
    }

    #[test]
    fn test_orbittrajectory_trajectory_len() {
        let mut traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);

        assert_eq!(traj.len(), 0);
        assert!(traj.is_empty());

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        traj.add(epoch, state);

        assert_eq!(traj.len(), 1);
        assert!(!traj.is_empty());
    }

    #[test]
    fn test_orbittrajectory_trajectory_is_empty() {
        let mut traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);

        assert!(traj.is_empty());

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        traj.add(epoch, state);

        assert!(!traj.is_empty());
    }

    #[test]
    fn test_orbittrajectory_trajectory_start_epoch() {
        let mut traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);

        assert!(traj.start_epoch().is_none());

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        traj.add(epoch, state);

        assert_eq!(traj.start_epoch().unwrap(), epoch);
    }

    #[test]
    fn test_orbittrajectory_trajectory_end_epoch() {
        let mut traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);

        assert!(traj.end_epoch().is_none());

        let epoch1 = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_jd(2451545.1, TimeSystem::UTC);
        let state = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        traj.add(epoch1, state);
        traj.add(epoch2, state);

        assert_eq!(traj.end_epoch().unwrap(), epoch2);
    }

    #[test]
    fn test_orbittrajectory_trajectory_timespan() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
        ];
        let traj = SOrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            None,
        );

        let timespan = traj.timespan().unwrap();
        assert_abs_diff_eq!(timespan, 0.1 * 86400.0, epsilon = 1e-5);
    }

    #[test]
    fn test_orbittrajectory_trajectory_first() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
        ];
        let traj = SOrbitTrajectory::from_orbital_data(
            epochs.clone(),
            states.clone(),
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            None,
        );

        let (first_epoch, first_state) = traj.first().unwrap();
        assert_eq!(first_epoch, epochs[0]);
        assert_eq!(first_state, states[0]);
    }

    #[test]
    fn test_orbittrajectory_trajectory_last() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
        ];
        let traj = SOrbitTrajectory::from_orbital_data(
            epochs.clone(),
            states.clone(),
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            None,
        );

        let (last_epoch, last_state) = traj.last().unwrap();
        assert_eq!(last_epoch, epochs[1]);
        assert_eq!(last_state, states[1]);
    }

    #[test]
    fn test_orbittrajectory_trajectory_clear() {
        let mut traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        traj.add(epoch, state);

        assert_eq!(traj.len(), 1);
        traj.clear();
        assert_eq!(traj.len(), 0);
    }

    #[test]
    fn test_orbittrajectory_trajectory_remove_epoch() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
        ];
        let mut traj = SOrbitTrajectory::from_orbital_data(
            epochs.clone(),
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            None,
        );

        let removed_state = traj.remove_epoch(&epochs[0]).unwrap();
        assert_eq!(removed_state[0], 7000e3);
        assert_eq!(traj.len(), 1);
    }

    #[test]
    fn test_orbittrajectory_trajectory_remove() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
        ];
        let mut traj = SOrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            None,
        );

        let (removed_epoch, removed_state) = traj.remove(0).unwrap();
        assert_eq!(removed_epoch.jd(), 2451545.0);
        assert_eq!(removed_state[0], 7000e3);
        assert_eq!(traj.len(), 1);
    }

    #[test]
    fn test_orbittrajectory_trajectory_get() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
        ];
        let traj = SOrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            None,
        );

        let (epoch, state) = traj.get(1).unwrap();
        assert_eq!(epoch.jd(), 2451545.1);
        assert_eq!(state[0], 7100e3);
    }

    #[test]
    fn test_orbittrajectory_trajectory_index_before_epoch() {
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;
        let t2 = t0 + 120.0;

        let epochs = vec![t0, t1, t2];
        let states = vec![
            Vector6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            Vector6::new(11.0, 12.0, 13.0, 14.0, 15.0, 16.0),
            Vector6::new(21.0, 22.0, 23.0, 24.0, 25.0, 26.0),
        ];

        let traj = SOrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            None,
        );

        // Test finding index before t0 (should error - before all states)
        let before_t0 = t0 - 10.0;
        assert!(traj.index_before_epoch(&before_t0).is_err());

        // Test finding index before t0+30s (should return index 0)
        let t0_plus_30 = t0 + 30.0;
        assert_eq!(traj.index_before_epoch(&t0_plus_30).unwrap(), 0);

        // Test finding index before t0+60s (should return index 1 - exact match)
        assert_eq!(traj.index_before_epoch(&t1).unwrap(), 1);

        // Test finding index before t0+90s (should return index 1)
        let t0_plus_90 = t0 + 90.0;
        assert_eq!(traj.index_before_epoch(&t0_plus_90).unwrap(), 1);

        // Test finding index before t0+120s (should return index 2 - exact match)
        assert_eq!(traj.index_before_epoch(&t2).unwrap(), 2);

        // Test finding index before t0+150s (should return index 2)
        let t0_plus_150 = t0 + 150.0;
        assert_eq!(traj.index_before_epoch(&t0_plus_150).unwrap(), 2);
    }

    #[test]
    fn test_orbittrajectory_trajectory_index_after_epoch() {
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;
        let t2 = t0 + 120.0;

        let epochs = vec![t0, t1, t2];
        let states = vec![
            Vector6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            Vector6::new(11.0, 12.0, 13.0, 14.0, 15.0, 16.0),
            Vector6::new(21.0, 22.0, 23.0, 24.0, 25.0, 26.0),
        ];

        let traj = SOrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            None,
        );

        // Test finding index after t0-30s (should return index 0)
        let t0_minus_30 = t0 - 30.0;
        assert_eq!(traj.index_after_epoch(&t0_minus_30).unwrap(), 0);

        // Test finding index after t0 (should return index 0 - exact match)
        assert_eq!(traj.index_after_epoch(&t0).unwrap(), 0);

        // Test finding index after t0+30s (should return index 1)
        let t0_plus_30 = t0 + 30.0;
        assert_eq!(traj.index_after_epoch(&t0_plus_30).unwrap(), 1);

        // Test finding index after t0+60s (should return index 1 - exact match)
        assert_eq!(traj.index_after_epoch(&t1).unwrap(), 1);

        // Test finding index after t0+90s (should return index 2)
        let t0_plus_90 = t0 + 90.0;
        assert_eq!(traj.index_after_epoch(&t0_plus_90).unwrap(), 2);

        // Test finding index after t0+120s (should return index 2 - exact match)
        assert_eq!(traj.index_after_epoch(&t2).unwrap(), 2);

        // Test finding index after t0+150s (should error - after all states)
        let t0_plus_150 = t0 + 150.0;
        assert!(traj.index_after_epoch(&t0_plus_150).is_err());
    }

    #[test]
    fn test_orbittrajectory_trajectory_state_before_epoch() {
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;
        let t2 = t0 + 120.0;

        let epochs = vec![t0, t1, t2];
        let states = vec![
            Vector6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            Vector6::new(11.0, 12.0, 13.0, 14.0, 15.0, 16.0),
            Vector6::new(21.0, 22.0, 23.0, 24.0, 25.0, 26.0),
        ];

        let traj = SOrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            None,
        );

        // Test that state_before_epoch returns correct (epoch, state) tuples
        let t0_plus_30 = t0 + 30.0;
        let (epoch, state) = traj.state_before_epoch(&t0_plus_30).unwrap();
        assert_eq!(epoch, t0);
        assert_abs_diff_eq!(state[0], 1.0, epsilon = 1e-10);

        let t0_plus_90 = t0 + 90.0;
        let (epoch, state) = traj.state_before_epoch(&t0_plus_90).unwrap();
        assert_eq!(epoch, t1);
        assert_abs_diff_eq!(state[0], 11.0, epsilon = 1e-10);

        // Test error case for epoch before all states
        let before_t0 = t0 - 10.0;
        assert!(traj.state_before_epoch(&before_t0).is_err());

        // Verify it uses the default trait implementation correctly
        let (epoch, state) = traj.state_before_epoch(&t2).unwrap();
        assert_eq!(epoch, t2);
        assert_abs_diff_eq!(state[0], 21.0, epsilon = 1e-10);
    }

    #[test]
    fn test_orbittrajectory_trajectory_state_after_epoch() {
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;
        let t2 = t0 + 120.0;

        let epochs = vec![t0, t1, t2];
        let states = vec![
            Vector6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            Vector6::new(11.0, 12.0, 13.0, 14.0, 15.0, 16.0),
            Vector6::new(21.0, 22.0, 23.0, 24.0, 25.0, 26.0),
        ];

        let traj = SOrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            None,
        );

        // Test that state_after_epoch returns correct (epoch, state) tuples
        let t0_plus_30 = t0 + 30.0;
        let (epoch, state) = traj.state_after_epoch(&t0_plus_30).unwrap();
        assert_eq!(epoch, t1);
        assert_abs_diff_eq!(state[0], 11.0, epsilon = 1e-10);

        let t0_plus_90 = t0 + 90.0;
        let (epoch, state) = traj.state_after_epoch(&t0_plus_90).unwrap();
        assert_eq!(epoch, t2);
        assert_abs_diff_eq!(state[0], 21.0, epsilon = 1e-10);

        // Test error case for epoch after all states
        let after_t2 = t2 + 10.0;
        assert!(traj.state_after_epoch(&after_t2).is_err());

        // Verify it uses the default trait implementation correctly
        let (epoch, state) = traj.state_after_epoch(&t0).unwrap();
        assert_eq!(epoch, t0);
        assert_abs_diff_eq!(state[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_orbittrajectory_trajectory_set_eviction_policy_max_size() {
        let mut traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);

        // Add 5 states
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        for i in 0..5 {
            let epoch = t0 + (i as f64 * 60.0);
            let state = Vector6::new(7000e3 + i as f64 * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0);
            traj.add(epoch, state);
        }

        assert_eq!(traj.len(), 5);

        // Set max size to 3
        traj.set_eviction_policy_max_size(3).unwrap();

        // Should only have 3 most recent states
        assert_eq!(traj.len(), 3);

        // First state should be the 3rd original state (oldest 2 evicted)
        let first_state = Trajectory::state_at_idx(&traj, 0).unwrap();
        assert_abs_diff_eq!(first_state[0], 7000e3 + 2000.0, epsilon = 1.0);

        // Add another state - should still maintain max size
        let new_epoch = t0 + 5.0 * 60.0;
        let new_state = Vector6::new(7000e3 + 5000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        traj.add(new_epoch, new_state);

        assert_eq!(traj.len(), 3);

        // Test error case
        assert!(traj.set_eviction_policy_max_size(0).is_err());
    }

    #[test]
    fn test_orbittrajectory_trajectory_set_eviction_policy_max_age() {
        let mut traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);

        // Add states spanning 5 minutes
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        for i in 0..6 {
            let epoch = t0 + (i as f64 * 60.0); // 0, 60, 120, 180, 240, 300 seconds
            let state = Vector6::new(7000e3 + i as f64 * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0);
            traj.add(epoch, state);
        }

        assert_eq!(traj.len(), 6);

        // Set max age to 240 seconds
        traj.set_eviction_policy_max_age(240.0).unwrap();
        assert_eq!(traj.len(), 5);

        let first_state = Trajectory::state_at_idx(&traj, 0).unwrap();
        assert_abs_diff_eq!(first_state[0], 7000e3 + 1000.0, epsilon = 1.0);

        // Set max age to 239 seconds
        traj.set_eviction_policy_max_age(239.0).unwrap();

        assert_eq!(traj.len(), 4);
        let first_state = Trajectory::state_at_idx(&traj, 0).unwrap();
        assert_abs_diff_eq!(first_state[0], 7000e3 + 2000.0, epsilon = 1.0);

        // Test error case
        assert!(traj.set_eviction_policy_max_age(0.0).is_err());
        assert!(traj.set_eviction_policy_max_age(-10.0).is_err());
    }

    // Default Trait Tests

    #[test]
    fn test_orbittrajectory_default() {
        let traj = SOrbitTrajectory::default();
        assert_eq!(traj.len(), 0);
        assert!(traj.is_empty());
        assert_eq!(traj.frame, OrbitFrame::ECI);
        assert_eq!(traj.representation, OrbitRepresentation::Cartesian);
        assert_eq!(traj.angle_format, None);
    }

    // Index Trait Tests

    #[test]
    fn test_orbittrajectory_index_index() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
            Epoch::from_jd(2451545.2, TimeSystem::UTC),
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
            Vector6::new(7200e3, 2000e3, 1000e3, 200.0, 7.7e3, 100.0),
        ];
        let traj = SOrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            None,
        );

        // Test indexing returns state vectors
        let state0 = &traj[0];
        assert_eq!(state0[0], 7000e3);

        let state1 = &traj[1];
        assert_eq!(state1[0], 7100e3);

        let state2 = &traj[2];
        assert_eq!(state2[0], 7200e3);
    }

    #[test]
    #[should_panic]
    fn test_orbittrajectory_index_index_out_of_bounds() {
        let epochs = vec![Epoch::from_jd(2451545.0, TimeSystem::UTC)];
        let states = vec![Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0)];
        let traj = SOrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            None,
        );

        let _ = &traj[10]; // Should panic
    }

    // IntoIterator Trait Tests

    #[test]
    fn test_orbittrajectory_intoiterator_into_iter() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
            Epoch::from_jd(2451545.2, TimeSystem::UTC),
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
            Vector6::new(7200e3, 2000e3, 1000e3, 200.0, 7.7e3, 100.0),
        ];
        let traj = SOrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            None,
        );

        let mut count = 0;
        for (epoch, state) in &traj {
            match count {
                0 => {
                    assert_eq!(epoch.jd(), 2451545.0);
                    assert_eq!(state[0], 7000e3);
                }
                1 => {
                    assert_eq!(epoch.jd(), 2451545.1);
                    assert_eq!(state[0], 7100e3);
                }
                2 => {
                    assert_eq!(epoch.jd(), 2451545.2);
                    assert_eq!(state[0], 7200e3);
                }
                _ => panic!("Too many iterations"),
            }
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn test_orbittrajectory_intoiterator_into_iter_empty() {
        let traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);

        let mut count = 0;
        for _ in &traj {
            count += 1;
        }
        assert_eq!(count, 0);
    }

    #[test]
    fn test_orbittrajectory_iterator_iterator_size_hint() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
            Epoch::from_jd(2451545.2, TimeSystem::UTC),
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
            Vector6::new(7200e3, 2000e3, 1000e3, 200.0, 7.7e3, 100.0),
        ];
        let traj = SOrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            None,
        );

        let iter = traj.into_iter();
        let (lower, upper) = iter.size_hint();
        assert_eq!(lower, 3);
        assert_eq!(upper, Some(3));
    }

    #[test]
    fn test_orbittrajectory_iterator_iterator_len() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
            Epoch::from_jd(2451545.2, TimeSystem::UTC),
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
            Vector6::new(7200e3, 2000e3, 1000e3, 200.0, 7.7e3, 100.0),
        ];
        let traj = SOrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            None,
        );

        let iter = traj.into_iter();
        assert_eq!(iter.len(), 3);
    }

    // Interpolatable Trait Tests

    #[test]
    fn test_orbittrajectory_interpolatable_set_interpolation_method() {
        let mut traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);

        assert_eq!(traj.get_interpolation_method(), InterpolationMethod::Linear);

        traj.set_interpolation_method(InterpolationMethod::Linear);
        assert_eq!(traj.get_interpolation_method(), InterpolationMethod::Linear);
    }

    #[test]
    fn test_orbittrajectory_interpolatable_get_interpolation_method() {
        let mut traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);

        // Test that get_interpolation_method returns Linear
        assert_eq!(traj.get_interpolation_method(), InterpolationMethod::Linear);

        // Set it to different methods and verify get_interpolation_method returns the correct value

        traj.set_interpolation_method(InterpolationMethod::Linear);
        assert_eq!(traj.get_interpolation_method(), InterpolationMethod::Linear);
    }

    #[test]
    fn test_orbittrajectory_interpolatable_interpolate_linear() {
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;
        let t2 = t0 + 120.0;

        let epochs = vec![t0, t1, t2];
        let states = vec![
            Vector6::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            Vector6::new(60.0, 120.0, 180.0, 240.0, 300.0, 360.0),
            Vector6::new(120.0, 240.0, 360.0, 480.0, 600.0, 720.0),
        ];

        let traj = SOrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            None,
        );

        // Test interpolate_linear at midpoints and exact epochs
        let state_at_t0 = traj.interpolate_linear(&t0).unwrap();
        assert_abs_diff_eq!(state_at_t0[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_at_t0[1], 0.0, epsilon = 1e-10);

        let state_at_t1 = traj.interpolate_linear(&t1).unwrap();
        assert_abs_diff_eq!(state_at_t1[0], 60.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_at_t1[1], 120.0, epsilon = 1e-10);

        let state_at_t2 = traj.interpolate_linear(&t2).unwrap();
        assert_abs_diff_eq!(state_at_t2[0], 120.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_at_t2[1], 240.0, epsilon = 1e-10);

        // Test interpolation at midpoint between t0 and t1
        let t0_plus_30 = t0 + 30.0;
        let state_at_midpoint = traj.interpolate_linear(&t0_plus_30).unwrap();
        assert_abs_diff_eq!(state_at_midpoint[0], 30.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_at_midpoint[1], 60.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_at_midpoint[2], 90.0, epsilon = 1e-10);

        // Test interpolation at midpoint between t1 and t2
        let t1_plus_30 = t1 + 30.0;
        let state_at_midpoint2 = traj.interpolate_linear(&t1_plus_30).unwrap();
        assert_abs_diff_eq!(state_at_midpoint2[0], 90.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_at_midpoint2[1], 180.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_at_midpoint2[2], 270.0, epsilon = 1e-10);

        // Test edge case: single state trajectory
        let single_epoch = vec![t0];
        let single_state = vec![Vector6::new(100.0, 200.0, 300.0, 400.0, 500.0, 600.0)];
        let single_traj = SOrbitTrajectory::from_orbital_data(
            single_epoch,
            single_state,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            None,
        );

        let state_single = single_traj.interpolate_linear(&t0).unwrap();
        assert_abs_diff_eq!(state_single[0], 100.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_single[1], 200.0, epsilon = 1e-10);
    }

    #[test]
    fn test_orbittrajectory_interpolatable_interpolate() {
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;
        let t2 = t0 + 120.0;

        let epochs = vec![t0, t1, t2];
        let states = vec![
            Vector6::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            Vector6::new(60.0, 120.0, 180.0, 240.0, 300.0, 360.0),
            Vector6::new(120.0, 240.0, 360.0, 480.0, 600.0, 720.0),
        ];

        let traj = SOrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            None,
        );

        // Test that interpolate() with Linear method returns same result as interpolate_linear()
        let t0_plus_30 = t0 + 30.0;
        let state_interpolate = traj.interpolate(&t0_plus_30).unwrap();
        let state_interpolate_linear = traj.interpolate_linear(&t0_plus_30).unwrap();

        for i in 0..6 {
            assert_abs_diff_eq!(
                state_interpolate[i],
                state_interpolate_linear[i],
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_orbittrajectory_interpolate_before_start() {
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;
        let t2 = t0 + 120.0;

        let epochs = vec![t0, t1, t2];
        let states = vec![
            Vector6::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            Vector6::new(60.0, 120.0, 180.0, 240.0, 300.0, 360.0),
            Vector6::new(120.0, 240.0, 360.0, 480.0, 600.0, 720.0),
        ];

        let traj = SOrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            None,
        );

        // Test interpolation before trajectory start
        let before_start = t0 - 10.0;
        let result = traj.interpolate_linear(&before_start);
        assert!(result.is_err());
        match result {
            Err(BraheError::OutOfBoundsError(_)) => {} // Expected error type
            _ => panic!("Expected OutOfBoundsError for interpolation before start"),
        }

        // Also test with interpolate() method
        let result = traj.interpolate(&before_start);
        assert!(result.is_err());
        match result {
            Err(BraheError::OutOfBoundsError(_)) => {} // Expected error type
            _ => panic!("Expected OutOfBoundsError for interpolation before start"),
        }
    }

    #[test]
    fn test_orbittrajectory_interpolate_after_end() {
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;
        let t2 = t0 + 120.0;

        let epochs = vec![t0, t1, t2];
        let states = vec![
            Vector6::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            Vector6::new(60.0, 120.0, 180.0, 240.0, 300.0, 360.0),
            Vector6::new(120.0, 240.0, 360.0, 480.0, 600.0, 720.0),
        ];

        let traj = SOrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            None,
        );

        // Test interpolation after trajectory end
        let after_end = t0 + 130.0;
        let result = traj.interpolate_linear(&after_end);
        assert!(result.is_err());
        match result {
            Err(BraheError::OutOfBoundsError(_)) => {} // Expected error type
            _ => panic!("Expected OutOfBoundsError for interpolation after end"),
        }

        // Also test with interpolate() method
        let result = traj.interpolate(&after_end);
        assert!(result.is_err());
        match result {
            Err(BraheError::OutOfBoundsError(_)) => {} // Expected error type
            _ => panic!("Expected OutOfBoundsError for interpolation after end"),
        }
    }

    // OrbitalTrajectory Trait Tests

    #[test]
    fn test_orbittrajectory_orbitaltrajectory_from_orbital_data() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
        ];

        let traj = SOrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            None,
        );

        assert_eq!(traj.len(), 2);
        assert_eq!(traj.frame, OrbitFrame::ECI);
        assert_eq!(traj.representation, OrbitRepresentation::Cartesian);
    }

    #[test]
    fn test_orbittrajectory_orbitaltrajectory_to_eci() {
        setup_global_test_eop();
        let tol = 1e-6;

        let state_base = state_osculating_to_cartesian(
            Vector6::new(R_EARTH + 500e3, 0.01, 97.0, 15.0, 30.0, 45.0),
            DEGREES,
        );

        println!("Base ECI State: {:?}", state_base);
        let x1 = state_cartesian_to_osculating(state_base, DEGREES);
        let x2 = state_cartesian_to_osculating(state_base, RADIANS);
        println!("Recoverted Degrees Keplerian: {:?}", x1);
        println!("Recoverted Radians Keplerian: {:?}", x2);

        // No transformation needed if already in ECI
        let mut traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);

        let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        traj.add(epoch, state_base);

        let eci_traj = traj.to_eci();
        assert_eq!(eci_traj.frame, OrbitFrame::ECI);
        assert_eq!(eci_traj.representation, OrbitRepresentation::Cartesian);
        assert_eq!(eci_traj.angle_format, None);
        assert_eq!(eci_traj.len(), 1);
        let (epoch_out, state_out) = eci_traj.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = tol);
        }

        // Convert Keplerian to ECI - Radians
        let mut kep_traj = SOrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Radians),
        );
        let kep_state_rad = state_cartesian_to_osculating(state_base, RADIANS);
        kep_traj.add(epoch, kep_state_rad);

        let eci_from_kep_rad = kep_traj.to_eci();
        assert_eq!(eci_from_kep_rad.frame, OrbitFrame::ECI);
        assert_eq!(
            eci_from_kep_rad.representation,
            OrbitRepresentation::Cartesian
        );
        assert_eq!(eci_from_kep_rad.angle_format, None);
        assert_eq!(eci_from_kep_rad.len(), 1);
        let (epoch_out, state_out) = eci_from_kep_rad.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = tol);
        }

        // Convert Keplerian to ECI - Degrees
        let mut kep_traj_deg = SOrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Degrees),
        );
        let kep_state_deg = state_cartesian_to_osculating(state_base, DEGREES);
        kep_traj_deg.add(epoch, kep_state_deg);
        let eci_from_kep_deg = kep_traj_deg.to_eci();
        assert_eq!(eci_from_kep_deg.frame, OrbitFrame::ECI);
        assert_eq!(
            eci_from_kep_deg.representation,
            OrbitRepresentation::Cartesian
        );
        assert_eq!(eci_from_kep_deg.angle_format, None);
        assert_eq!(eci_from_kep_deg.len(), 1);
        let (epoch_out, state_out) = eci_from_kep_deg.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = tol);
        }

        // Convert ECEF to ECI
        let mut ecef_traj =
            SOrbitTrajectory::new(OrbitFrame::ECEF, OrbitRepresentation::Cartesian, None);
        let ecef_state = state_eci_to_ecef(epoch, state_base);
        ecef_traj.add(epoch, ecef_state);
        let eci_from_ecef = ecef_traj.to_eci();
        assert_eq!(eci_from_ecef.frame, OrbitFrame::ECI);
        assert_eq!(eci_from_ecef.representation, OrbitRepresentation::Cartesian);
        assert_eq!(eci_from_ecef.angle_format, None);
        assert_eq!(eci_from_ecef.len(), 1);
        let (epoch_out, state_out) = eci_from_ecef.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = tol);
        }
    }

    #[test]
    fn test_orbittrajectory_orbitaltrajectory_to_ecef() {
        setup_global_test_eop();
        let tol = 1e-6;

        let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state_base = state_eci_to_ecef(
            epoch,
            state_osculating_to_cartesian(
                Vector6::new(R_EARTH + 500e3, 0.01, 97.0, 15.0, 30.0, 45.0),
                DEGREES,
            ),
        );

        // No transformation needed if already in ECEF
        let mut traj =
            SOrbitTrajectory::new(OrbitFrame::ECEF, OrbitRepresentation::Cartesian, None);

        traj.add(epoch, state_base);
        let ecef_traj = traj.to_ecef();
        assert_eq!(ecef_traj.frame, OrbitFrame::ECEF);
        assert_eq!(ecef_traj.representation, OrbitRepresentation::Cartesian);
        assert_eq!(ecef_traj.angle_format, None);
        assert_eq!(ecef_traj.len(), 1);
        let (epoch_out, state_out) = ecef_traj.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = tol);
        }

        // Convert ECI to ECEF
        let mut eci_traj =
            SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let eci_state = state_ecef_to_eci(epoch, state_base);
        eci_traj.add(epoch, eci_state);
        let ecef_from_eci = eci_traj.to_ecef();
        assert_eq!(ecef_from_eci.frame, OrbitFrame::ECEF);
        assert_eq!(ecef_from_eci.representation, OrbitRepresentation::Cartesian);
        assert_eq!(ecef_from_eci.angle_format, None);
        assert_eq!(ecef_from_eci.len(), 1);
        let (epoch_out, state_out) = ecef_from_eci.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = tol);
        }

        // Convert Keplerian to ECEF - Radians
        let mut kep_traj = SOrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Radians),
        );
        let kep_state_rad = state_cartesian_to_osculating(eci_state, RADIANS);
        kep_traj.add(epoch, kep_state_rad);
        let ecef_from_kep_rad = kep_traj.to_ecef();
        assert_eq!(ecef_from_kep_rad.frame, OrbitFrame::ECEF);
        assert_eq!(
            ecef_from_kep_rad.representation,
            OrbitRepresentation::Cartesian
        );
        assert_eq!(ecef_from_kep_rad.angle_format, None);
        assert_eq!(ecef_from_kep_rad.len(), 1);
        let (epoch_out, state_out) = ecef_from_kep_rad.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = tol);
        }

        // Convert Keplerian to ECEF - Degrees
        let mut kep_traj_deg = SOrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Degrees),
        );
        let kep_state_deg = state_cartesian_to_osculating(eci_state, DEGREES);
        kep_traj_deg.add(epoch, kep_state_deg);
        let ecef_from_kep_deg = kep_traj_deg.to_ecef();
        assert_eq!(ecef_from_kep_deg.frame, OrbitFrame::ECEF);
        assert_eq!(
            ecef_from_kep_deg.representation,
            OrbitRepresentation::Cartesian
        );
        assert_eq!(ecef_from_kep_deg.angle_format, None);
        assert_eq!(ecef_from_kep_deg.len(), 1);
        let (epoch_out, state_out) = ecef_from_kep_deg.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = tol);
        }
    }

    #[test]
    fn test_orbittrajectory_orbitaltrajectory_to_itrf() {
        setup_global_test_eop();
        let tol = 1e-6;

        let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state_base = state_eci_to_ecef(
            epoch,
            state_osculating_to_cartesian(
                Vector6::new(R_EARTH + 500e3, 0.01, 97.0, 15.0, 30.0, 45.0),
                DEGREES,
            ),
        );

        // No transformation needed if already in ITRF
        let mut traj =
            SOrbitTrajectory::new(OrbitFrame::ITRF, OrbitRepresentation::Cartesian, None);

        traj.add(epoch, state_base);
        let itrf_traj = traj.to_itrf();
        assert_eq!(itrf_traj.frame, OrbitFrame::ITRF);
        assert_eq!(itrf_traj.representation, OrbitRepresentation::Cartesian);
        assert_eq!(itrf_traj.angle_format, None);
        assert_eq!(itrf_traj.len(), 1);
        let (epoch_out, state_out) = itrf_traj.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = tol);
        }

        // Convert ECI to ITRF
        let mut eci_traj =
            SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let eci_state = state_ecef_to_eci(epoch, state_base);
        eci_traj.add(epoch, eci_state);
        let itrf_from_eci = eci_traj.to_itrf();
        assert_eq!(itrf_from_eci.frame, OrbitFrame::ITRF);
        assert_eq!(itrf_from_eci.representation, OrbitRepresentation::Cartesian);
        assert_eq!(itrf_from_eci.angle_format, None);
        assert_eq!(itrf_from_eci.len(), 1);
        let (epoch_out, state_out) = itrf_from_eci.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = tol);
        }

        // Convert GCRF to ITRF
        let mut gcrf_traj =
            SOrbitTrajectory::new(OrbitFrame::GCRF, OrbitRepresentation::Cartesian, None);
        let gcrf_state = state_itrf_to_gcrf(epoch, state_base);
        gcrf_traj.add(epoch, gcrf_state);
        let itrf_from_gcrf = gcrf_traj.to_itrf();
        assert_eq!(itrf_from_gcrf.frame, OrbitFrame::ITRF);
        assert_eq!(
            itrf_from_gcrf.representation,
            OrbitRepresentation::Cartesian
        );
        assert_eq!(itrf_from_gcrf.angle_format, None);
        assert_eq!(itrf_from_gcrf.len(), 1);
        let (epoch_out, state_out) = itrf_from_gcrf.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = tol);
        }

        // Convert Keplerian to ITRF - Radians
        let mut kep_traj = SOrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Radians),
        );
        let kep_state_rad = state_cartesian_to_osculating(eci_state, RADIANS);
        kep_traj.add(epoch, kep_state_rad);
        let itrf_from_kep_rad = kep_traj.to_itrf();
        assert_eq!(itrf_from_kep_rad.frame, OrbitFrame::ITRF);
        assert_eq!(
            itrf_from_kep_rad.representation,
            OrbitRepresentation::Cartesian
        );
        assert_eq!(itrf_from_kep_rad.angle_format, None);
        assert_eq!(itrf_from_kep_rad.len(), 1);
        let (epoch_out, state_out) = itrf_from_kep_rad.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = tol);
        }

        // Convert Keplerian to itrf - Degrees
        let mut kep_traj_deg = SOrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Degrees),
        );
        let kep_state_deg = state_cartesian_to_osculating(eci_state, DEGREES);
        kep_traj_deg.add(epoch, kep_state_deg);
        let itrf_from_kep_deg = kep_traj_deg.to_itrf();
        assert_eq!(itrf_from_kep_deg.frame, OrbitFrame::ITRF);
        assert_eq!(
            itrf_from_kep_deg.representation,
            OrbitRepresentation::Cartesian
        );
        assert_eq!(itrf_from_kep_deg.angle_format, None);
        assert_eq!(itrf_from_kep_deg.len(), 1);
        let (epoch_out, state_out) = itrf_from_kep_deg.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = tol);
        }
    }

    #[test]
    fn test_orbittrajectory_orbitaltrajectory_to_keplerian_deg() {
        setup_global_test_eop();
        let tol = 1e-6;

        let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state_kep_deg = Vector6::new(7000e3, 0.01, 97.0, 15.0, 30.0, 45.0);

        // No transformation needed if already in Keplerian Degrees
        let mut traj = SOrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Degrees),
        );
        traj.add(epoch, state_kep_deg);
        let kep_traj = traj.to_keplerian(AngleFormat::Degrees);
        assert_eq!(kep_traj.frame, OrbitFrame::ECI);
        assert_eq!(kep_traj.representation, OrbitRepresentation::Keplerian);
        assert_eq!(kep_traj.angle_format, Some(AngleFormat::Degrees));
        assert_eq!(kep_traj.len(), 1);
        let (epoch_out, state_out) = kep_traj.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_kep_deg[i], epsilon = tol);
        }

        // Convert Keplerian Radians to Keplerian Degrees
        let mut kep_rad_traj = SOrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Radians),
        );
        let mut state_kep_rad = state_kep_deg;
        for i in 2..6 {
            state_kep_rad[i] = state_kep_deg[i] * DEG2RAD;
        }
        kep_rad_traj.add(epoch, state_kep_rad);
        let kep_from_rad = kep_rad_traj.to_keplerian(AngleFormat::Degrees);
        assert_eq!(kep_from_rad.frame, OrbitFrame::ECI);
        assert_eq!(kep_from_rad.representation, OrbitRepresentation::Keplerian);
        assert_eq!(kep_from_rad.angle_format, Some(AngleFormat::Degrees));
        assert_eq!(kep_from_rad.len(), 1);
        let (epoch_out, state_out) = kep_from_rad.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_kep_deg[i], epsilon = tol);
        }

        // Convert ECI to Keplerian Degrees
        let mut cart_traj =
            SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let cart_state = state_osculating_to_cartesian(state_kep_deg, DEGREES);
        cart_traj.add(epoch, cart_state);
        let kep_from_cart = cart_traj.to_keplerian(AngleFormat::Degrees);
        assert_eq!(kep_from_cart.frame, OrbitFrame::ECI);
        assert_eq!(kep_from_cart.representation, OrbitRepresentation::Keplerian);
        assert_eq!(kep_from_cart.angle_format, Some(AngleFormat::Degrees));
        assert_eq!(kep_from_cart.len(), 1);
        let (epoch_out, state_out) = kep_from_cart.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_kep_deg[i], epsilon = tol);
        }

        // Convert ECEF to Keplerian Degrees
        let mut ecef_traj =
            SOrbitTrajectory::new(OrbitFrame::ECEF, OrbitRepresentation::Cartesian, None);
        let ecef_state = state_eci_to_ecef(epoch, cart_state);
        ecef_traj.add(epoch, ecef_state);
        let kep_from_ecef = ecef_traj.to_keplerian(AngleFormat::Degrees);
        assert_eq!(kep_from_ecef.frame, OrbitFrame::ECI);
        assert_eq!(kep_from_ecef.representation, OrbitRepresentation::Keplerian);
        assert_eq!(kep_from_ecef.angle_format, Some(AngleFormat::Degrees));
        assert_eq!(kep_from_ecef.len(), 1);
        let (epoch_out, state_out) = kep_from_ecef.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_kep_deg[i], epsilon = tol);
        }
    }

    #[test]
    fn test_orbittrajectory_orbitaltrajectory_to_keplerian_rad() {
        setup_global_test_eop();
        let tol = 1e-6;

        let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state_kep_deg = Vector6::new(7000e3, 0.01, 97.0, 15.0, 30.0, 45.0);
        let mut state_kep_rad = state_kep_deg;
        for i in 2..6 {
            state_kep_rad[i] = state_kep_deg[i] * DEG2RAD;
        }

        // No transformation needed if already in Keplerian Radians
        let mut traj = SOrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Radians),
        );
        traj.add(epoch, state_kep_rad);
        let kep_traj = traj.to_keplerian(AngleFormat::Radians);
        assert_eq!(kep_traj.frame, OrbitFrame::ECI);
        assert_eq!(kep_traj.representation, OrbitRepresentation::Keplerian);
        assert_eq!(kep_traj.angle_format, Some(AngleFormat::Radians));
        assert_eq!(kep_traj.len(), 1);
        let (epoch_out, state_out) = kep_traj.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_kep_rad[i], epsilon = tol);
        }

        // Convert Keplerian Degrees to Keplerian Radians
        let mut kep_deg_traj = SOrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Degrees),
        );
        kep_deg_traj.add(epoch, state_kep_deg);
        let kep_from_deg = kep_deg_traj.to_keplerian(AngleFormat::Radians);
        assert_eq!(kep_from_deg.frame, OrbitFrame::ECI);
        assert_eq!(kep_from_deg.representation, OrbitRepresentation::Keplerian);
        assert_eq!(kep_from_deg.angle_format, Some(AngleFormat::Radians));
        assert_eq!(kep_from_deg.len(), 1);
        let (epoch_out, state_out) = kep_from_deg.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_kep_rad[i], epsilon = tol);
        }

        // Convert ECI to Keplerian Radians
        let mut cart_traj =
            SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let cart_state = state_osculating_to_cartesian(state_kep_deg, DEGREES);
        cart_traj.add(epoch, cart_state);
        let kep_from_cart = cart_traj.to_keplerian(AngleFormat::Radians);
        assert_eq!(kep_from_cart.frame, OrbitFrame::ECI);
        assert_eq!(kep_from_cart.representation, OrbitRepresentation::Keplerian);
        assert_eq!(kep_from_cart.angle_format, Some(AngleFormat::Radians));
        assert_eq!(kep_from_cart.len(), 1);
        let (epoch_out, state_out) = kep_from_cart.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_kep_rad[i], epsilon = tol);
        }

        // Convert ECEF to Keplerian Radians
        let mut ecef_traj =
            SOrbitTrajectory::new(OrbitFrame::ECEF, OrbitRepresentation::Cartesian, None);
        let ecef_state = state_eci_to_ecef(epoch, cart_state);
        ecef_traj.add(epoch, ecef_state);
        let kep_from_ecef = ecef_traj.to_keplerian(AngleFormat::Radians);
        assert_eq!(kep_from_ecef.frame, OrbitFrame::ECI);
        assert_eq!(kep_from_ecef.representation, OrbitRepresentation::Keplerian);
        assert_eq!(kep_from_ecef.angle_format, Some(AngleFormat::Radians));
        assert_eq!(kep_from_ecef.len(), 1);
        let (epoch_out, state_out) = kep_from_ecef.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_kep_rad[i], epsilon = tol);
        }
    }

    #[test]
    fn test_orbittrajectory_orbitaltrajectory_to_gcrf() {
        setup_global_test_eop();
        let tol = 1e-6;

        let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state_base = state_osculating_to_cartesian(
            Vector6::new(R_EARTH + 500e3, 0.01, 97.0, 15.0, 30.0, 45.0),
            DEGREES,
        );

        // No transformation needed if already in GCRF
        let mut traj =
            SOrbitTrajectory::new(OrbitFrame::GCRF, OrbitRepresentation::Cartesian, None);
        traj.add(epoch, state_base);
        let gcrf_traj = traj.to_gcrf();
        assert_eq!(gcrf_traj.frame, OrbitFrame::GCRF);
        assert_eq!(gcrf_traj.representation, OrbitRepresentation::Cartesian);
        assert_eq!(gcrf_traj.angle_format, None);
        assert_eq!(gcrf_traj.len(), 1);
        let (epoch_out, state_out) = gcrf_traj.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = tol);
        }

        // Convert ECI to GCRF (should be same since ECI is treated as GCRF)
        let mut eci_traj =
            SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        eci_traj.add(epoch, state_base);
        let gcrf_from_eci = eci_traj.to_gcrf();
        assert_eq!(gcrf_from_eci.frame, OrbitFrame::GCRF);
        assert_eq!(gcrf_from_eci.representation, OrbitRepresentation::Cartesian);
        assert_eq!(gcrf_from_eci.angle_format, None);
        assert_eq!(gcrf_from_eci.len(), 1);
        let (epoch_out, state_out) = gcrf_from_eci.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = tol);
        }

        // Convert EME2000 to GCRF
        let mut eme2000_traj =
            SOrbitTrajectory::new(OrbitFrame::EME2000, OrbitRepresentation::Cartesian, None);
        let eme2000_state = state_gcrf_to_eme2000(state_base);
        eme2000_traj.add(epoch, eme2000_state);
        let gcrf_from_eme2000 = eme2000_traj.to_gcrf();
        assert_eq!(gcrf_from_eme2000.frame, OrbitFrame::GCRF);
        assert_eq!(
            gcrf_from_eme2000.representation,
            OrbitRepresentation::Cartesian
        );
        assert_eq!(gcrf_from_eme2000.angle_format, None);
        assert_eq!(gcrf_from_eme2000.len(), 1);
        let (epoch_out, state_out) = gcrf_from_eme2000.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = tol);
        }

        // Convert ITRF to GCRF
        let mut itrf_traj =
            SOrbitTrajectory::new(OrbitFrame::ITRF, OrbitRepresentation::Cartesian, None);
        let itrf_state = state_gcrf_to_itrf(epoch, state_base);
        itrf_traj.add(epoch, itrf_state);
        let gcrf_from_itrf = itrf_traj.to_gcrf();
        assert_eq!(gcrf_from_itrf.frame, OrbitFrame::GCRF);
        assert_eq!(
            gcrf_from_itrf.representation,
            OrbitRepresentation::Cartesian
        );
        assert_eq!(gcrf_from_itrf.angle_format, None);
        assert_eq!(gcrf_from_itrf.len(), 1);
        let (epoch_out, state_out) = gcrf_from_itrf.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = tol);
        }

        // Convert Keplerian to GCRF - Radians
        let mut kep_traj = SOrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Radians),
        );
        let kep_state_rad = state_cartesian_to_osculating(state_base, RADIANS);
        kep_traj.add(epoch, kep_state_rad);
        let gcrf_from_kep_rad = kep_traj.to_gcrf();
        assert_eq!(gcrf_from_kep_rad.frame, OrbitFrame::GCRF);
        assert_eq!(
            gcrf_from_kep_rad.representation,
            OrbitRepresentation::Cartesian
        );
        assert_eq!(gcrf_from_kep_rad.angle_format, None);
        assert_eq!(gcrf_from_kep_rad.len(), 1);
        let (epoch_out, state_out) = gcrf_from_kep_rad.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = tol);
        }

        // Convert Keplerian to GCRF - Degrees
        let mut kep_traj_deg = SOrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Degrees),
        );
        let kep_state_deg = state_cartesian_to_osculating(state_base, DEGREES);
        kep_traj_deg.add(epoch, kep_state_deg);
        let gcrf_from_kep_deg = kep_traj_deg.to_gcrf();
        assert_eq!(gcrf_from_kep_deg.frame, OrbitFrame::GCRF);
        assert_eq!(
            gcrf_from_kep_deg.representation,
            OrbitRepresentation::Cartesian
        );
        assert_eq!(gcrf_from_kep_deg.angle_format, None);
        assert_eq!(gcrf_from_kep_deg.len(), 1);
        let (epoch_out, state_out) = gcrf_from_kep_deg.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = tol);
        }
    }

    #[test]
    fn test_orbittrajectory_orbitaltrajectory_to_eme2000() {
        setup_global_test_eop();
        let tol = 1e-6;

        let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state_base = state_gcrf_to_eme2000(state_osculating_to_cartesian(
            Vector6::new(R_EARTH + 500e3, 0.01, 97.0, 15.0, 30.0, 45.0),
            DEGREES,
        ));

        // No transformation needed if already in EME2000
        let mut traj =
            SOrbitTrajectory::new(OrbitFrame::EME2000, OrbitRepresentation::Cartesian, None);
        traj.add(epoch, state_base);
        let eme2000_traj = traj.to_eme2000();
        assert_eq!(eme2000_traj.frame, OrbitFrame::EME2000);
        assert_eq!(eme2000_traj.representation, OrbitRepresentation::Cartesian);
        assert_eq!(eme2000_traj.angle_format, None);
        assert_eq!(eme2000_traj.len(), 1);
        let (epoch_out, state_out) = eme2000_traj.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = tol);
        }

        // Convert GCRF to EME2000
        let mut gcrf_traj =
            SOrbitTrajectory::new(OrbitFrame::GCRF, OrbitRepresentation::Cartesian, None);
        let gcrf_state = state_eme2000_to_gcrf(state_base);
        gcrf_traj.add(epoch, gcrf_state);
        let eme2000_from_gcrf = gcrf_traj.to_eme2000();
        assert_eq!(eme2000_from_gcrf.frame, OrbitFrame::EME2000);
        assert_eq!(
            eme2000_from_gcrf.representation,
            OrbitRepresentation::Cartesian
        );
        assert_eq!(eme2000_from_gcrf.angle_format, None);
        assert_eq!(eme2000_from_gcrf.len(), 1);
        let (epoch_out, state_out) = eme2000_from_gcrf.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = tol);
        }

        // Convert ECI to EME2000
        let mut eci_traj =
            SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        eci_traj.add(epoch, gcrf_state);
        let eme2000_from_eci = eci_traj.to_eme2000();
        assert_eq!(eme2000_from_eci.frame, OrbitFrame::EME2000);
        assert_eq!(
            eme2000_from_eci.representation,
            OrbitRepresentation::Cartesian
        );
        assert_eq!(eme2000_from_eci.angle_format, None);
        assert_eq!(eme2000_from_eci.len(), 1);
        let (epoch_out, state_out) = eme2000_from_eci.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = tol);
        }

        // Convert ITRF to EME2000
        let mut itrf_traj =
            SOrbitTrajectory::new(OrbitFrame::ITRF, OrbitRepresentation::Cartesian, None);
        let itrf_state = state_gcrf_to_itrf(epoch, gcrf_state);
        itrf_traj.add(epoch, itrf_state);
        let eme2000_from_itrf = itrf_traj.to_eme2000();
        assert_eq!(eme2000_from_itrf.frame, OrbitFrame::EME2000);
        assert_eq!(
            eme2000_from_itrf.representation,
            OrbitRepresentation::Cartesian
        );
        assert_eq!(eme2000_from_itrf.angle_format, None);
        assert_eq!(eme2000_from_itrf.len(), 1);
        let (epoch_out, state_out) = eme2000_from_itrf.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = tol);
        }

        // Convert Keplerian to EME2000 - Radians
        let mut kep_traj = SOrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Radians),
        );
        let kep_state_rad = state_cartesian_to_osculating(gcrf_state, RADIANS);
        kep_traj.add(epoch, kep_state_rad);
        let eme2000_from_kep_rad = kep_traj.to_eme2000();
        assert_eq!(eme2000_from_kep_rad.frame, OrbitFrame::EME2000);
        assert_eq!(
            eme2000_from_kep_rad.representation,
            OrbitRepresentation::Cartesian
        );
        assert_eq!(eme2000_from_kep_rad.angle_format, None);
        assert_eq!(eme2000_from_kep_rad.len(), 1);
        let (epoch_out, state_out) = eme2000_from_kep_rad.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = tol);
        }

        // Convert Keplerian to EME2000 - Degrees
        let mut kep_traj_deg = SOrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Degrees),
        );
        let kep_state_deg = state_cartesian_to_osculating(gcrf_state, DEGREES);
        kep_traj_deg.add(epoch, kep_state_deg);
        let eme2000_from_kep_deg = kep_traj_deg.to_eme2000();
        assert_eq!(eme2000_from_kep_deg.frame, OrbitFrame::EME2000);
        assert_eq!(
            eme2000_from_kep_deg.representation,
            OrbitRepresentation::Cartesian
        );
        assert_eq!(eme2000_from_kep_deg.angle_format, None);
        assert_eq!(eme2000_from_kep_deg.len(), 1);
        let (epoch_out, state_out) = eme2000_from_kep_deg.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = tol);
        }
    }

    // SOrbitStateProvider Trait Tests

    #[test]
    fn test_orbittrajectory_stateprovider_state_eci_cartesian() {
        // Test SOrbitStateProvider::state() for ECI Cartesian trajectory
        let mut traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);

        let epoch1 = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state1 = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        traj.add(epoch1, state1);

        let epoch2 = Epoch::from_jd(2451545.5, TimeSystem::UTC);
        let state2 = Vector6::new(7200e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0);
        traj.add(epoch2, state2);

        // Query at exact epoch
        let state_at_1 = SStateProvider::state(&traj, epoch1);
        for i in 0..6 {
            assert_abs_diff_eq!(state_at_1[i], state1[i], epsilon = 1e-6);
        }

        // Query at interpolated epoch
        let epoch_mid = Epoch::from_jd(2451545.25, TimeSystem::UTC);
        let state_mid = SStateProvider::state(&traj, epoch_mid);
        // Should be interpolated between state1 and state2
        assert!(state_mid[0] > state1[0] && state_mid[0] < state2[0]);
    }

    #[test]
    fn test_orbittrajectory_stateprovider_state_eci() {
        setup_global_test_eop();

        // Test SOrbitStateProvider::state_eci() for ECI Cartesian trajectory
        let mut traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state_eci = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        traj.add(epoch, state_eci);

        // Query ECI state
        let result = traj.state_eci(epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(result[i], state_eci[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_orbittrajectory_stateprovider_state_eci_from_keplerian() {
        // Test SOrbitStateProvider::state_eci() for Keplerian trajectory
        let mut traj = SOrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Degrees),
        );

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state_kep = Vector6::new(R_EARTH + 500e3, 0.001, 98.0, 15.0, 30.0, 45.0);
        traj.add(epoch, state_kep);

        // Query ECI Cartesian state
        let result = traj.state_eci(epoch);

        // Convert Keplerian to Cartesian manually for comparison
        let expected = state_osculating_to_cartesian(state_kep, AngleFormat::Degrees);

        for i in 0..6 {
            assert_abs_diff_eq!(result[i], expected[i], epsilon = 1e-3);
        }
    }

    #[test]
    fn test_orbittrajectory_stateprovider_state_eci_from_ecef() {
        setup_global_test_eop();

        // Test SOrbitStateProvider::state_eci() for ECEF Cartesian trajectory
        let mut traj =
            SOrbitTrajectory::new(OrbitFrame::ECEF, OrbitRepresentation::Cartesian, None);

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state_ecef = Vector6::new(7000e3, 0.0, 0.0, 0.0, 0.0, 7.5e3);
        traj.add(epoch, state_ecef);

        // Query ECI state
        let result = traj.state_eci(epoch);

        // Convert ECEF to ECI manually for comparison
        let expected = state_ecef_to_eci(epoch, state_ecef);

        for i in 0..6 {
            assert_abs_diff_eq!(result[i], expected[i], epsilon = 1e-3);
        }
    }

    #[test]
    fn test_orbittrajectory_stateprovider_state_gcrf_cartesian() {
        // Test SOrbitStateProvider::state() for ECI Cartesian trajectory
        let mut traj =
            SOrbitTrajectory::new(OrbitFrame::GCRF, OrbitRepresentation::Cartesian, None);

        let epoch1 = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state1 = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        traj.add(epoch1, state1);

        let epoch2 = Epoch::from_jd(2451545.5, TimeSystem::UTC);
        let state2 = Vector6::new(7200e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0);
        traj.add(epoch2, state2);

        // Query at exact epoch
        let state_at_1 = SStateProvider::state(&traj, epoch1);
        for i in 0..6 {
            assert_abs_diff_eq!(state_at_1[i], state1[i], epsilon = 1e-6);
        }

        // Query at interpolated epoch
        let epoch_mid = Epoch::from_jd(2451545.25, TimeSystem::UTC);
        let state_mid = SStateProvider::state(&traj, epoch_mid);
        // Should be interpolated between state1 and state2
        assert!(state_mid[0] > state1[0] && state_mid[0] < state2[0]);
    }

    #[test]
    fn test_orbittrajectory_stateprovider_state_gcrf() {
        setup_global_test_eop();

        // Test SOrbitStateProvider::state_gcrf() for ECI Cartesian trajectory
        let mut traj =
            SOrbitTrajectory::new(OrbitFrame::GCRF, OrbitRepresentation::Cartesian, None);

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state_gcrf = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        traj.add(epoch, state_gcrf);

        // Query GCRF state
        let result = traj.state_gcrf(epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(result[i], state_gcrf[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_orbittrajectory_stateprovider_state_gcrf_from_keplerian() {
        // Test SOrbitStateProvider::state_gcrf() for Keplerian trajectory
        let mut traj = SOrbitTrajectory::new(
            OrbitFrame::GCRF,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Degrees),
        );

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state_kep = Vector6::new(R_EARTH + 500e3, 0.001, 98.0, 15.0, 30.0, 45.0);
        traj.add(epoch, state_kep);

        // Query GCRF Cartesian state
        let result = traj.state_gcrf(epoch);

        // Convert Keplerian to Cartesian manually for comparison
        let expected = state_osculating_to_cartesian(state_kep, AngleFormat::Degrees);

        for i in 0..6 {
            assert_abs_diff_eq!(result[i], expected[i], epsilon = 1e-3);
        }
    }

    #[test]
    fn test_orbittrajectory_stateprovider_state_gcrf_from_itrf() {
        setup_global_test_eop();

        // Test SOrbitStateProvider::state_gcrf() for ITRF Cartesian trajectory
        let mut traj =
            SOrbitTrajectory::new(OrbitFrame::ITRF, OrbitRepresentation::Cartesian, None);

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state_itrf = Vector6::new(7000e3, 0.0, 0.0, 0.0, 0.0, 7.5e3);
        traj.add(epoch, state_itrf);

        // Query GCRF state
        let result = traj.state_gcrf(epoch);

        // Convert ITRF to GCRF manually for comparison
        let expected = state_itrf_to_gcrf(epoch, state_itrf);

        for i in 0..6 {
            assert_abs_diff_eq!(result[i], expected[i], epsilon = 1e-3);
        }
    }

    #[test]
    fn test_orbittrajectory_stateprovider_state_ecef() {
        setup_global_test_eop();

        // Test SOrbitStateProvider::state_ecef() for ECEF Cartesian trajectory
        let mut traj =
            SOrbitTrajectory::new(OrbitFrame::ECEF, OrbitRepresentation::Cartesian, None);

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state_ecef = Vector6::new(7000e3, 0.0, 0.0, 0.0, 0.0, 7.5e3);
        traj.add(epoch, state_ecef);

        // Query ECEF state
        let result = traj.state_ecef(epoch);

        for i in 0..6 {
            assert_abs_diff_eq!(result[i], state_ecef[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_orbittrajectory_stateprovider_state_ecef_from_eci() {
        setup_global_test_eop();

        // Test SOrbitStateProvider::state_ecef() for ECI Cartesian trajectory
        let mut traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state_eci = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        traj.add(epoch, state_eci);

        // Query ECEF state
        let result = traj.state_ecef(epoch);

        // Convert ECI to ECEF manually for comparison
        let expected = state_eci_to_ecef(epoch, state_eci);

        for i in 0..6 {
            assert_abs_diff_eq!(result[i], expected[i], epsilon = 1e-3);
        }
    }

    #[test]
    fn test_orbittrajectory_stateprovider_state_ecef_from_keplerian() {
        setup_global_test_eop();

        // Test SOrbitStateProvider::state_ecef() for Keplerian trajectory
        let mut traj = SOrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Degrees),
        );

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state_kep = Vector6::new(R_EARTH + 500e3, 0.001, 98.0, 15.0, 30.0, 45.0);
        traj.add(epoch, state_kep);

        // Query ECEF state
        let result = traj.state_ecef(epoch);

        // Convert Keplerian -> ECI Cartesian -> ECEF manually for comparison
        let state_eci_cart = state_osculating_to_cartesian(state_kep, AngleFormat::Degrees);
        let expected = state_eci_to_ecef(epoch, state_eci_cart);

        for i in 0..6 {
            assert_abs_diff_eq!(result[i], expected[i], epsilon = 1e-3);
        }
    }

    #[test]
    fn test_orbittrajectory_stateprovider_state_itrf() {
        setup_global_test_eop();

        // Test SOrbitStateProvider::state_itrf() for ECEF Cartesian trajectory
        let mut traj =
            SOrbitTrajectory::new(OrbitFrame::ITRF, OrbitRepresentation::Cartesian, None);

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state_itrf = Vector6::new(7000e3, 0.0, 0.0, 0.0, 0.0, 7.5e3);
        traj.add(epoch, state_itrf);

        // Query ECEF state
        let result = traj.state_itrf(epoch);

        for i in 0..6 {
            assert_abs_diff_eq!(result[i], state_itrf[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_orbittrajectory_stateprovider_state_itrf_from_gcrf() {
        setup_global_test_eop();

        // Test SOrbitStateProvider::state_itrf() for ECI Cartesian trajectory
        let mut traj =
            SOrbitTrajectory::new(OrbitFrame::GCRF, OrbitRepresentation::Cartesian, None);

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state_eci = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        traj.add(epoch, state_eci);

        // Query ECEF state
        let result = traj.state_itrf(epoch);

        // Convert ECI to ECEF manually for comparison
        let expected = state_gcrf_to_itrf(epoch, state_eci);

        for i in 0..6 {
            assert_abs_diff_eq!(result[i], expected[i], epsilon = 1e-3);
        }
    }

    #[test]
    fn test_orbittrajectory_stateprovider_state_itrf_from_keplerian() {
        setup_global_test_eop();

        // Test SOrbitStateProvider::state_itrf() for Keplerian trajectory
        let mut traj = SOrbitTrajectory::new(
            OrbitFrame::GCRF,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Degrees),
        );

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state_kep = Vector6::new(R_EARTH + 500e3, 0.001, 98.0, 15.0, 30.0, 45.0);
        traj.add(epoch, state_kep);

        // Query ECEF state
        let result = traj.state_itrf(epoch);

        // Convert Keplerian -> ECI Cartesian -> ECEF manually for comparison
        let state_gcrf_cart = state_osculating_to_cartesian(state_kep, AngleFormat::Degrees);
        let expected = state_gcrf_to_itrf(epoch, state_gcrf_cart);

        for i in 0..6 {
            assert_abs_diff_eq!(result[i], expected[i], epsilon = 1e-3);
        }
    }

    #[test]
    fn test_orbittrajectory_stateprovider_state_eme2000() {
        // Test SOrbitStateProvider::state_eme2000() for EME2000 Cartesian trajectory
        let mut traj =
            SOrbitTrajectory::new(OrbitFrame::EME2000, OrbitRepresentation::Cartesian, None);

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state_eme2000 = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        traj.add(epoch, state_eme2000);

        // Query state
        let result = traj.state_eme2000(epoch);

        for i in 0..6 {
            assert_abs_diff_eq!(result[i], state_eme2000[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_orbittrajectory_stateprovider_state_eme2000_from_keplerian() {
        // Test SOrbitStateProvider::state_eme2000() for Keplerian trajectory
        let mut traj = SOrbitTrajectory::new(
            OrbitFrame::GCRF,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Degrees),
        );

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state_kep = Vector6::new(R_EARTH + 500e3, 0.001, 98.0, 15.0, 30.0, 45.0);
        traj.add(epoch, state_kep);

        // Query EME2000 state
        let result = traj.state_eme2000(epoch);

        // Convert Keplerian -> GCRF Cartesian -> EME2000 manually for comparison
        let state_gcrf_cart = state_osculating_to_cartesian(state_kep, AngleFormat::Degrees);
        let expected = state_gcrf_to_eme2000(state_gcrf_cart);

        for i in 0..6 {
            assert_abs_diff_eq!(result[i], expected[i], epsilon = 1e-3);
        }
    }

    #[test]
    fn test_orbittrajectory_stateprovider_state_eme2000_from_gcrf() {
        // Test SOrbitStateProvider::state_eme2000() for GCRF Cartesian trajectory
        let mut traj =
            SOrbitTrajectory::new(OrbitFrame::GCRF, OrbitRepresentation::Cartesian, None);

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state_gcrf = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        traj.add(epoch, state_gcrf);

        // Query EME2000 state
        let result = traj.state_eme2000(epoch);

        // Convert GCRF to EME2000 manually for comparison
        let expected = state_gcrf_to_eme2000(state_gcrf);

        for i in 0..6 {
            assert_abs_diff_eq!(result[i], expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_orbittrajectory_stateprovider_state_eme2000_from_itrf() {
        setup_global_test_eop();

        // Test SOrbitStateProvider::state_eme2000() for ITRF Cartesian trajectory
        let mut traj =
            SOrbitTrajectory::new(OrbitFrame::ITRF, OrbitRepresentation::Cartesian, None);

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state_itrf = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        traj.add(epoch, state_itrf);

        // Query EME2000 state
        let result = traj.state_eme2000(epoch);

        // Convert ITRF -> GCRF -> EME2000 manually for comparison
        let state_gcrf = state_itrf_to_gcrf(epoch, state_itrf);
        let expected = state_gcrf_to_eme2000(state_gcrf);

        for i in 0..6 {
            assert_abs_diff_eq!(result[i], expected[i], epsilon = 1e-3);
        }
    }

    #[test]
    fn test_orbittrajectory_stateprovider_state_as_osculating_elements_from_cartesian() {
        // Test SOrbitStateProvider::state_as_osculating_elements() for ECI Cartesian trajectory
        let mut traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state_cart = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        traj.add(epoch, state_cart);

        // Query osculating elements in degrees
        let result_deg = traj.state_as_osculating_elements(epoch, AngleFormat::Degrees);

        // Convert Cartesian to Keplerian manually for comparison
        let expected_deg = state_cartesian_to_osculating(state_cart, AngleFormat::Degrees);

        for i in 0..6 {
            assert_abs_diff_eq!(result_deg[i], expected_deg[i], epsilon = 1e-3);
        }

        // Query osculating elements in radians
        let result_rad = traj.state_as_osculating_elements(epoch, AngleFormat::Radians);
        let expected_rad = state_cartesian_to_osculating(state_cart, AngleFormat::Radians);

        for i in 0..6 {
            assert_abs_diff_eq!(result_rad[i], expected_rad[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_orbittrajectory_stateprovider_state_as_osculating_elements_from_keplerian() {
        // Test SOrbitStateProvider::state_as_osculating_elements() for Keplerian trajectory
        let mut traj = SOrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Degrees),
        );

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state_kep_deg = Vector6::new(R_EARTH + 500e3, 0.001, 98.0, 15.0, 30.0, 45.0);
        traj.add(epoch, state_kep_deg);

        // Query osculating elements in degrees (same as native format)
        let result_deg = traj.state_as_osculating_elements(epoch, AngleFormat::Degrees);

        for i in 0..6 {
            assert_abs_diff_eq!(result_deg[i], state_kep_deg[i], epsilon = 1e-6);
        }

        // Query osculating elements in radians (requires conversion)
        let result_rad = traj.state_as_osculating_elements(epoch, AngleFormat::Radians);

        // First two elements unchanged (a, e)
        assert_abs_diff_eq!(result_rad[0], state_kep_deg[0], epsilon = 1e-6);
        assert_abs_diff_eq!(result_rad[1], state_kep_deg[1], epsilon = 1e-9);

        // Angle elements converted
        use crate::constants::math::DEG2RAD;
        let expected_i_rad = state_kep_deg[2] * DEG2RAD;
        let expected_raan_rad = state_kep_deg[3] * DEG2RAD;
        let expected_argp_rad = state_kep_deg[4] * DEG2RAD;
        let expected_m_rad = state_kep_deg[5] * DEG2RAD;

        assert_abs_diff_eq!(result_rad[2], expected_i_rad, epsilon = 1e-9);
        assert_abs_diff_eq!(result_rad[3], expected_raan_rad, epsilon = 1e-9);
        assert_abs_diff_eq!(result_rad[4], expected_argp_rad, epsilon = 1e-9);
        assert_abs_diff_eq!(result_rad[5], expected_m_rad, epsilon = 1e-9);
    }

    #[test]
    fn test_orbittrajectory_stateprovider_state_as_osculating_elements_from_ecef() {
        setup_global_test_eop();

        // Test SOrbitStateProvider::state_as_osculating_elements() for ECEF Cartesian trajectory
        let mut traj =
            SOrbitTrajectory::new(OrbitFrame::ECEF, OrbitRepresentation::Cartesian, None);

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state_ecef = Vector6::new(7000e3, 0.0, 0.0, 0.0, 0.0, 7.5e3);
        traj.add(epoch, state_ecef);

        // Query osculating elements
        let result = traj.state_as_osculating_elements(epoch, AngleFormat::Degrees);

        // Convert ECEF -> ECI -> Keplerian manually for comparison
        let state_eci = state_ecef_to_eci(epoch, state_ecef);
        let expected = state_cartesian_to_osculating(state_eci, AngleFormat::Degrees);

        for i in 0..6 {
            assert_abs_diff_eq!(result[i], expected[i], epsilon = 1e-3);
        }
    }

    #[test]
    fn test_orbittrajectory_identifiable_with_name() {
        let traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let traj = traj.with_name("Test Trajectory");

        assert_eq!(traj.get_name(), Some("Test Trajectory"));
    }

    #[test]
    fn test_orbittrajectory_identifiable_with_id() {
        let traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let traj = traj.with_id(12345);

        assert_eq!(traj.get_id(), Some(12345));
    }

    #[test]
    fn test_orbittrajectory_identifiable_with_uuid() {
        let traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let traj = traj.with_new_uuid();

        assert!(traj.get_uuid().is_some());
    }

    #[test]
    fn test_orbittrajectory_identifiable_with_identity() {
        let uuid = Uuid::new_v4();
        let traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let traj = traj.with_identity(Some("Test"), Some(uuid), Some(999));

        assert_eq!(traj.get_name(), Some("Test"));
        assert_eq!(traj.get_id(), Some(999));
        assert_eq!(traj.get_uuid(), Some(uuid));
    }

    #[test]
    fn test_orbittrajectory_identifiable_set_methods() {
        let mut traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);

        traj.set_name(Some("Updated Name"));
        assert_eq!(traj.get_name(), Some("Updated Name"));

        traj.set_id(Some(777));
        assert_eq!(traj.get_id(), Some(777));

        traj.generate_uuid();
        assert!(traj.get_uuid().is_some());
    }

    // Covariance functionality tests
    #[test]
    fn test_from_orbital_data_with_covariances() {
        setup_global_test_eop();

        let epoch1 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_datetime(2024, 1, 1, 0, 10, 0.0, 0.0, TimeSystem::UTC);

        let state1 = Vector6::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0);
        let state2 = Vector6::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0);

        let cov1 = SMatrix::<f64, 6, 6>::identity() * 100.0;
        let cov2 = SMatrix::<f64, 6, 6>::identity() * 200.0;

        let traj = SOrbitTrajectory::from_orbital_data(
            vec![epoch1, epoch2],
            vec![state1, state2],
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            Some(vec![cov1, cov2]),
        );

        assert_eq!(traj.covariances.as_ref().unwrap().len(), 2);
    }

    #[test]
    #[should_panic(expected = "Covariances length (1) must match states length (2)")]
    fn test_from_orbital_data_covariances_length_mismatch() {
        let epoch1 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_datetime(2024, 1, 1, 0, 10, 0.0, 0.0, TimeSystem::UTC);

        let state1 = Vector6::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0);
        let state2 = Vector6::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0);

        let cov1 = SMatrix::<f64, 6, 6>::identity();

        SOrbitTrajectory::from_orbital_data(
            vec![epoch1, epoch2],
            vec![state1, state2],
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            Some(vec![cov1]),
        );
    }

    #[test]
    #[should_panic(expected = "Covariances are only supported for ECI, GCRF, and EME2000 frames")]
    fn test_from_orbital_data_covariances_invalid_frame_itrf() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0);
        let cov = SMatrix::<f64, 6, 6>::identity();

        SOrbitTrajectory::from_orbital_data(
            vec![epoch],
            vec![state],
            OrbitFrame::ITRF,
            OrbitRepresentation::Cartesian,
            None,
            Some(vec![cov]),
        );
    }

    #[test]
    #[should_panic(expected = "Covariances are only supported for ECI, GCRF, and EME2000 frames")]
    fn test_from_orbital_data_covariances_invalid_frame_ecef() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0);
        let cov = SMatrix::<f64, 6, 6>::identity();

        SOrbitTrajectory::from_orbital_data(
            vec![epoch],
            vec![state],
            OrbitFrame::ECEF,
            OrbitRepresentation::Cartesian,
            None,
            Some(vec![cov]),
        );
    }

    #[test]
    fn test_add_state_and_covariance() {
        setup_global_test_eop();

        let mut traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        traj.covariances = Some(Vec::new());

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = SVector::<f64, 6>::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);
        let cov = SMatrix::<f64, 6, 6>::identity() * 100.0;

        traj.add_state_and_covariance(epoch, state, cov);

        assert_eq!(traj.len(), 1);
        assert_eq!(traj.covariances.as_ref().unwrap().len(), 1);
        assert_abs_diff_eq!(
            traj.covariances.as_ref().unwrap()[0][(0, 0)],
            100.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_covariance_provider_basic() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0);
        let cov = SMatrix::<f64, 6, 6>::identity() * 100.0;

        let traj = SOrbitTrajectory::from_orbital_data(
            vec![epoch],
            vec![state],
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            Some(vec![cov]),
        );

        let result = traj.covariance(epoch);
        assert!(result.is_some());
        assert_abs_diff_eq!(result.unwrap()[(0, 0)], 100.0, epsilon = 1e-6);
    }

    #[test]
    fn test_covariance_rtn() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0);
        let cov = SMatrix::<f64, 6, 6>::identity() * 100.0;

        let traj = SOrbitTrajectory::from_orbital_data(
            vec![epoch],
            vec![state],
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            Some(vec![cov]),
        );

        let result = traj.covariance_rtn(epoch);
        assert!(result.is_some());

        let result_cov = result.unwrap();
        assert!(result_cov[(0, 0)].abs() > 1e-6);
        assert!(result_cov[(1, 1)].abs() > 1e-6);
        assert!(result_cov[(2, 2)].abs() > 1e-6);
    }

    #[test]
    fn test_orbit_trajectory_covariance_eci() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0);
        let cov = SMatrix::<f64, 6, 6>::identity() * 100.0;

        let traj = SOrbitTrajectory::from_orbital_data(
            vec![epoch],
            vec![state],
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            Some(vec![cov]),
        );

        let result = traj.covariance_eci(epoch);
        assert!(result.is_some());

        let result_cov = result.unwrap();
        assert_abs_diff_eq!(result_cov[(0, 0)], 100.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result_cov[(1, 1)], 100.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result_cov[(2, 2)], 100.0, epsilon = 1e-6);
    }

    #[test]
    fn test_orbit_trajectory_covariance_gcrf() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0);
        let cov = SMatrix::<f64, 6, 6>::identity() * 100.0;

        let traj = SOrbitTrajectory::from_orbital_data(
            vec![epoch],
            vec![state],
            OrbitFrame::GCRF,
            OrbitRepresentation::Cartesian,
            None,
            Some(vec![cov]),
        );

        let result = traj.covariance_gcrf(epoch);
        assert!(result.is_some());

        let result_cov = result.unwrap();
        assert_abs_diff_eq!(result_cov[(0, 0)], 100.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result_cov[(1, 1)], 100.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result_cov[(2, 2)], 100.0, epsilon = 1e-6);
    }

    #[test]
    fn test_covariance_eci_from_eme2000_frame() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0);

        // Create a diagonal covariance matrix in EME2000 frame
        let cov_eme2000 = SMatrix::<f64, 6, 6>::identity() * 100.0;

        let traj = SOrbitTrajectory::from_orbital_data(
            vec![epoch],
            vec![state],
            OrbitFrame::EME2000,
            OrbitRepresentation::Cartesian,
            None,
            Some(vec![cov_eme2000]),
        );

        // Get covariance in ECI frame (should be transformed)
        let result = traj.covariance_eci(epoch);
        assert!(result.is_some());

        let cov_eci = result.unwrap();

        // Verify covariance is symmetric (should be preserved by transformation)
        for i in 0..6 {
            for j in 0..6 {
                assert_abs_diff_eq!(cov_eci[(i, j)], cov_eci[(j, i)], epsilon = 1e-10);
            }
        }

        // Verify diagonal elements are positive (positive-definiteness check)
        for i in 0..6 {
            assert!(cov_eci[(i, i)] > 0.0);
        }

        // Verify transformation occurred (ECI covariance should differ from EME2000 due to rotation)
        // The EME2000-GCRF bias is very small (~10^-8 rad), so transformation may produce
        // very small off-diagonal elements or nearly preserve the diagonal structure
        // We verify that the diagonal elements are preserved (within numerical precision)
        for i in 0..6 {
            assert_abs_diff_eq!(cov_eci[(i, i)], 100.0, epsilon = 1e-3);
        }
    }

    #[test]
    fn test_covariance_gcrf_from_eme2000_frame() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0);
        let cov_eme2000 = SMatrix::<f64, 6, 6>::identity() * 100.0;

        let traj = SOrbitTrajectory::from_orbital_data(
            vec![epoch],
            vec![state],
            OrbitFrame::EME2000,
            OrbitRepresentation::Cartesian,
            None,
            Some(vec![cov_eme2000]),
        );

        // covariance_gcrf should delegate to covariance_eci
        let result_gcrf = traj.covariance_gcrf(epoch);
        let result_eci = traj.covariance_eci(epoch);

        assert!(result_gcrf.is_some());
        assert!(result_eci.is_some());

        let cov_gcrf = result_gcrf.unwrap();
        let cov_eci = result_eci.unwrap();

        // GCRF and ECI should be identical for EME2000 transformation
        for i in 0..6 {
            for j in 0..6 {
                assert_abs_diff_eq!(cov_gcrf[(i, j)], cov_eci[(i, j)], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_covariance_rtn_from_eme2000_frame() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0);
        let cov_eme2000 = SMatrix::<f64, 6, 6>::identity() * 100.0;

        let traj = SOrbitTrajectory::from_orbital_data(
            vec![epoch],
            vec![state],
            OrbitFrame::EME2000,
            OrbitRepresentation::Cartesian,
            None,
            Some(vec![cov_eme2000]),
        );

        // Get covariance in RTN frame (should go EME2000 -> ECI -> RTN)
        let result = traj.covariance_rtn(epoch);
        assert!(result.is_some());

        let cov_rtn = result.unwrap();

        // Verify covariance is symmetric
        for i in 0..6 {
            for j in 0..6 {
                assert_abs_diff_eq!(cov_rtn[(i, j)], cov_rtn[(j, i)], epsilon = 1e-10);
            }
        }

        // Verify diagonal elements are positive
        for i in 0..6 {
            assert!(cov_rtn[(i, i)] > 0.0);
        }

        // RTN covariance should be non-trivial (not identity)
        let is_non_identity = (0..6).any(|i| (cov_rtn[(i, i)] - 100.0).abs() > 1e-6)
            || (0..6).any(|i| (0..6).any(|j| i != j && cov_rtn[(i, j)].abs() > 1e-6));
        assert!(
            is_non_identity,
            "RTN transformation should produce non-identity matrix"
        );
    }

    #[test]
    fn test_covariance_interpolatable_trait_methods() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0);
        let cov = SMatrix::<f64, 6, 6>::identity() * 100.0;

        let mut traj = SOrbitTrajectory::from_orbital_data(
            vec![epoch],
            vec![state],
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            Some(vec![cov]),
        );

        // Test getter - default should be TwoWasserstein
        assert_eq!(
            traj.get_covariance_interpolation_method(),
            CovarianceInterpolationMethod::TwoWasserstein
        );

        // Test setter
        traj.set_covariance_interpolation_method(CovarianceInterpolationMethod::MatrixSquareRoot);
        assert_eq!(
            traj.get_covariance_interpolation_method(),
            CovarianceInterpolationMethod::MatrixSquareRoot
        );

        // Test builder pattern
        let traj2 = traj
            .with_covariance_interpolation_method(CovarianceInterpolationMethod::TwoWasserstein);
        assert_eq!(
            traj2.get_covariance_interpolation_method(),
            CovarianceInterpolationMethod::TwoWasserstein
        );
    }

    #[test]
    fn test_covariance_interpolation_edge_cases_matrix_square_root() {
        setup_global_test_eop();

        let epoch1 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_datetime(2024, 1, 1, 0, 10, 0.0, 0.0, TimeSystem::UTC);
        let epoch3 = Epoch::from_datetime(2024, 1, 1, 0, 20, 0.0, 0.0, TimeSystem::UTC);

        let state1 = Vector6::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0);
        let state2 = Vector6::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0);
        let state3 = Vector6::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0);

        let cov1 = SMatrix::<f64, 6, 6>::identity() * 100.0;
        let cov2 = SMatrix::<f64, 6, 6>::identity() * 200.0;
        let cov3 = SMatrix::<f64, 6, 6>::identity() * 300.0;

        let traj = SOrbitTrajectory::from_orbital_data(
            vec![epoch1, epoch2, epoch3],
            vec![state1, state2, state3],
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            Some(vec![cov1, cov2, cov3]),
        )
        .with_covariance_interpolation_method(CovarianceInterpolationMethod::MatrixSquareRoot);

        // Test at exact epoch
        let result_exact = traj.covariance(epoch2);
        assert!(result_exact.is_some());
        assert_abs_diff_eq!(result_exact.unwrap()[(0, 0)], 200.0, epsilon = 1e-6);

        // Test halfway between epoch1 and epoch2 (should be interpolated)
        let epoch_halfway = Epoch::from_datetime(2024, 1, 1, 0, 5, 0.0, 0.0, TimeSystem::UTC);
        let result_halfway = traj.covariance(epoch_halfway);
        assert!(result_halfway.is_some());
        // Verify interpolation gives value between endpoints
        let halfway_val = result_halfway.unwrap()[(0, 0)];
        assert!(halfway_val > 100.0 && halfway_val < 200.0);

        // Test before data range (should return None)
        let epoch_before = Epoch::from_datetime(2023, 12, 31, 23, 50, 0.0, 0.0, TimeSystem::UTC);
        let result_before = traj.covariance(epoch_before);
        assert!(result_before.is_none());

        // Test after data range (should return None)
        let epoch_after = Epoch::from_datetime(2024, 1, 1, 0, 30, 0.0, 0.0, TimeSystem::UTC);
        let result_after = traj.covariance(epoch_after);
        assert!(result_after.is_none());
    }

    #[test]
    fn test_covariance_interpolation_edge_cases_two_wasserstein() {
        setup_global_test_eop();

        let epoch1 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_datetime(2024, 1, 1, 0, 10, 0.0, 0.0, TimeSystem::UTC);
        let epoch3 = Epoch::from_datetime(2024, 1, 1, 0, 20, 0.0, 0.0, TimeSystem::UTC);

        let state1 = Vector6::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0);
        let state2 = Vector6::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0);
        let state3 = Vector6::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0);

        let cov1 = SMatrix::<f64, 6, 6>::identity() * 100.0;
        let cov2 = SMatrix::<f64, 6, 6>::identity() * 200.0;
        let cov3 = SMatrix::<f64, 6, 6>::identity() * 300.0;

        let traj = SOrbitTrajectory::from_orbital_data(
            vec![epoch1, epoch2, epoch3],
            vec![state1, state2, state3],
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            Some(vec![cov1, cov2, cov3]),
        )
        .with_covariance_interpolation_method(CovarianceInterpolationMethod::TwoWasserstein);

        // Test at exact epoch
        let result_exact = traj.covariance(epoch2);
        assert!(result_exact.is_some());
        assert_abs_diff_eq!(result_exact.unwrap()[(0, 0)], 200.0, epsilon = 1e-6);

        // Test halfway between epoch1 and epoch2 (should be interpolated)
        let epoch_halfway = Epoch::from_datetime(2024, 1, 1, 0, 5, 0.0, 0.0, TimeSystem::UTC);
        let result_halfway = traj.covariance(epoch_halfway);
        assert!(result_halfway.is_some());
        // Verify interpolation gives value between endpoints
        let halfway_val = result_halfway.unwrap()[(0, 0)];
        assert!(halfway_val > 100.0 && halfway_val < 200.0);

        // Test before data range (should return None)
        let epoch_before = Epoch::from_datetime(2023, 12, 31, 23, 50, 0.0, 0.0, TimeSystem::UTC);
        let result_before = traj.covariance(epoch_before);
        assert!(result_before.is_none());

        // Test after data range (should return None)
        let epoch_after = Epoch::from_datetime(2024, 1, 1, 0, 30, 0.0, 0.0, TimeSystem::UTC);
        let result_after = traj.covariance(epoch_after);
        assert!(result_after.is_none());
    }

    #[test]
    fn test_covariance_interpolation_methods_comparison() {
        setup_global_test_eop();

        let epoch1 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_datetime(2024, 1, 1, 0, 10, 0.0, 0.0, TimeSystem::UTC);

        let state1 = Vector6::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0);
        let state2 = Vector6::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0);

        let cov1 = SMatrix::<f64, 6, 6>::identity() * 100.0;
        let cov2 = SMatrix::<f64, 6, 6>::identity() * 200.0;

        // Test both interpolation methods
        let traj_wasserstein = SOrbitTrajectory::from_orbital_data(
            vec![epoch1, epoch2],
            vec![state1, state2],
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            Some(vec![cov1, cov2]),
        )
        .with_covariance_interpolation_method(CovarianceInterpolationMethod::TwoWasserstein);

        let traj_matrix_sqrt = SOrbitTrajectory::from_orbital_data(
            vec![epoch1, epoch2],
            vec![state1, state2],
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            Some(vec![cov1, cov2]),
        )
        .with_covariance_interpolation_method(CovarianceInterpolationMethod::MatrixSquareRoot);

        // Query at midpoint
        let epoch_mid = Epoch::from_datetime(2024, 1, 1, 0, 5, 0.0, 0.0, TimeSystem::UTC);

        let cov_wasserstein = traj_wasserstein.covariance(epoch_mid).unwrap();
        let cov_matrix_sqrt = traj_matrix_sqrt.covariance(epoch_mid).unwrap();

        // Both methods should produce positive-definite symmetric matrices
        for i in 0..6 {
            assert!(cov_wasserstein[(i, i)] > 0.0);
            assert!(cov_matrix_sqrt[(i, i)] > 0.0);
            for j in 0..6 {
                assert_abs_diff_eq!(
                    cov_wasserstein[(i, j)],
                    cov_wasserstein[(j, i)],
                    epsilon = 1e-10
                );
                assert_abs_diff_eq!(
                    cov_matrix_sqrt[(i, j)],
                    cov_matrix_sqrt[(j, i)],
                    epsilon = 1e-10
                );
            }
        }

        // Both methods should produce values in reasonable range (between endpoints)
        assert!(cov_wasserstein[(0, 0)] > 100.0 && cov_wasserstein[(0, 0)] < 200.0);
        assert!(cov_matrix_sqrt[(0, 0)] > 100.0 && cov_matrix_sqrt[(0, 0)] < 200.0);

        // For diagonal matrices, both methods should give identical results
        assert_abs_diff_eq!(
            cov_wasserstein[(0, 0)],
            cov_matrix_sqrt[(0, 0)],
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_covariance_single_point_trajectory() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0);
        let cov = SMatrix::<f64, 6, 6>::identity() * 100.0;

        let traj = SOrbitTrajectory::from_orbital_data(
            vec![epoch],
            vec![state],
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            Some(vec![cov]),
        );

        // Exact epoch should return covariance
        let result_exact = traj.covariance(epoch);
        assert!(result_exact.is_some());
        assert_abs_diff_eq!(result_exact.unwrap()[(0, 0)], 100.0, epsilon = 1e-6);

        // Different epoch should return None (no interpolation possible with single point)
        let epoch_later = epoch + 60.0;
        let result_later = traj.covariance(epoch_later);
        assert!(result_later.is_none());
    }

    #[test]
    fn test_covariance_rtn_elliptical_orbit() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        // Create elliptical inclined orbit state
        // a = R_EARTH + 600km, e = 0.2, i = 63.4 deg
        let a = R_EARTH + 600e3;
        let e = 0.2;
        let i = 63.4_f64.to_radians();
        let raan = 45.0_f64.to_radians();
        let argp = 30.0_f64.to_radians();
        let nu = 0.0; // True anomaly

        use crate::coordinates::state_osculating_to_cartesian;

        let oe = Vector6::new(a, e, i, raan, argp, nu);
        let state = state_osculating_to_cartesian(oe, AngleFormat::Radians);

        let cov = SMatrix::<f64, 6, 6>::identity() * 100.0;

        let traj = SOrbitTrajectory::from_orbital_data(
            vec![epoch],
            vec![state],
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            Some(vec![cov]),
        );

        // Get covariance in RTN frame
        let result = traj.covariance_rtn(epoch);
        assert!(result.is_some());

        let cov_rtn = result.unwrap();

        // Verify RTN covariance is symmetric
        for i in 0..6 {
            for j in 0..6 {
                assert_abs_diff_eq!(cov_rtn[(i, j)], cov_rtn[(j, i)], epsilon = 1e-10);
            }
        }

        // Verify diagonal elements are positive (positive-definiteness check)
        for i in 0..6 {
            assert!(cov_rtn[(i, i)] > 0.0);
        }

        // RTN transformation should produce different values than identity
        let differs_from_identity = (0..6).any(|i| (cov_rtn[(i, i)] - 100.0).abs() > 1e-6)
            || (0..6).any(|i| (0..6).any(|j| i != j && cov_rtn[(i, j)].abs() > 1e-6));
        assert!(differs_from_identity);
    }

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_orbittrajectory_display() {
        let traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let display = format!("{}", traj);
        assert_eq!(
            display,
            "SOrbitTrajectory(frame=ECI, representation=Cartesian, states=0)"
        );

        // Add some states and check display changes
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let mut traj = traj;
        traj.add(t0, Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0));
        traj.add(t0 + 60.0, Vector6::new(7100e3, 0.0, 0.0, 0.0, 7.5e3, 0.0));

        let display = format!("{}", traj);
        assert_eq!(
            display,
            "SOrbitTrajectory(frame=ECI, representation=Cartesian, states=2)"
        );
    }

    #[test]
    fn test_orbittrajectory_with_interpolation_method() {
        let traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
            .with_interpolation_method(InterpolationMethod::Linear);

        assert_eq!(traj.get_interpolation_method(), InterpolationMethod::Linear);
    }

    #[test]
    fn test_orbittrajectory_with_eviction_policy_max_size() {
        let traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
            .with_eviction_policy_max_size(10);

        assert_eq!(
            traj.get_eviction_policy(),
            TrajectoryEvictionPolicy::KeepCount
        );
    }

    #[test]
    #[should_panic(expected = "Maximum size must be >= 1")]
    fn test_orbittrajectory_with_eviction_policy_max_size_zero_panics() {
        let _ = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
            .with_eviction_policy_max_size(0);
    }

    #[test]
    fn test_orbittrajectory_with_eviction_policy_max_age() {
        let traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
            .with_eviction_policy_max_age(3600.0);

        assert_eq!(
            traj.get_eviction_policy(),
            TrajectoryEvictionPolicy::KeepWithinDuration
        );
    }

    #[test]
    #[should_panic(expected = "Maximum age must be > 0.0")]
    fn test_orbittrajectory_with_eviction_policy_max_age_zero_panics() {
        let _ = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
            .with_eviction_policy_max_age(0.0);
    }

    #[test]
    #[should_panic(expected = "Maximum age must be > 0.0")]
    fn test_orbittrajectory_with_eviction_policy_max_age_negative_panics() {
        let _ = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
            .with_eviction_policy_max_age(-100.0);
    }

    #[test]
    fn test_orbittrajectory_add_state_and_covariance_insert_before() {
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 120.0; // Second epoch at +120s
        let t_middle = t0 + 60.0; // Middle epoch at +60s

        let state1 = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        let state2 = Vector6::new(7200e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        let state_middle = Vector6::new(7100e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);

        let cov1 = SMatrix::<f64, 6, 6>::identity() * 100.0;
        let cov2 = SMatrix::<f64, 6, 6>::identity() * 200.0;
        let cov_middle = SMatrix::<f64, 6, 6>::identity() * 150.0;

        let mut traj = SOrbitTrajectory::from_orbital_data(
            vec![t0, t1],
            vec![state1, state2],
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            Some(vec![cov1, cov2]),
        );

        // Insert state in the middle (before t1)
        traj.add_state_and_covariance(t_middle, state_middle, cov_middle);

        assert_eq!(traj.len(), 3);

        // Verify order is correct
        let (epoch0, _) = traj.get(0).unwrap();
        let (epoch1, state1_ret) = traj.get(1).unwrap();
        let (epoch2, _) = traj.get(2).unwrap();

        assert_eq!(epoch0, t0);
        assert_eq!(epoch1, t_middle);
        assert_eq!(epoch2, t1);

        // Verify middle state is correct
        assert_abs_diff_eq!(state1_ret[0], 7100e3, epsilon = 1.0);

        // Verify middle covariance is correct
        let cov_ret = traj.covariance(t_middle).unwrap();
        assert_abs_diff_eq!(cov_ret[(0, 0)], 150.0, epsilon = 1e-6);
    }

    #[test]
    fn test_orbittrajectory_add_state_and_covariance_append_equal_epoch() {
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;

        let state1 = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        let state2 = Vector6::new(7100e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);

        let cov1 = SMatrix::<f64, 6, 6>::identity() * 100.0;
        let cov2 = SMatrix::<f64, 6, 6>::identity() * 200.0;

        let mut traj = SOrbitTrajectory::from_orbital_data(
            vec![t0, t1],
            vec![state1, state2],
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            Some(vec![cov1, cov2]),
        );

        // Append new state at t0 (after existing state)
        let new_state = Vector6::new(8000e3, 0.0, 0.0, 0.0, 8.0e3, 0.0);
        let new_cov = SMatrix::<f64, 6, 6>::identity() * 500.0;
        traj.add_state_and_covariance(t0, new_state, new_cov);

        // Length should now be 3 (original 2 + 1 appended)
        assert_eq!(traj.len(), 3);

        // Verify original state at index 0 is unchanged
        let (_, state_ret) = traj.get(0).unwrap();
        assert_abs_diff_eq!(state_ret[0], 7000e3, epsilon = 1.0);

        // Verify new state was appended at index 1
        let (_, new_state_ret) = traj.get(1).unwrap();
        assert_abs_diff_eq!(new_state_ret[0], 8000e3, epsilon = 1.0);

        // Verify covariances are correct
        let covs = traj.covariances.as_ref().unwrap();
        assert_abs_diff_eq!(covs[0][(0, 0)], 100.0, epsilon = 1e-6);
        assert_abs_diff_eq!(covs[1][(0, 0)], 500.0, epsilon = 1e-6);
    }

    #[test]
    fn test_orbittrajectory_from_data_trait() {
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;

        let epochs = vec![t0, t1];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
        ];

        let traj = <SOrbitTrajectory as Trajectory>::from_data(epochs, states).unwrap();
        assert_eq!(traj.len(), 2);
    }

    #[test]
    fn test_orbittrajectory_from_data_length_mismatch() {
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let epochs = vec![t0];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
        ];

        let result = <SOrbitTrajectory as Trajectory>::from_data(epochs, states);
        assert!(result.is_err());
    }

    #[test]
    fn test_orbittrajectory_from_data_empty() {
        let epochs: Vec<Epoch> = vec![];
        let states: Vec<Vector6<f64>> = vec![];

        let result = <SOrbitTrajectory as Trajectory>::from_data(epochs, states);
        assert!(result.is_err());
    }

    #[test]
    fn test_orbittrajectory_nearest_state_empty_trajectory() {
        let traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

        let result = traj.nearest_state(&t0);
        assert!(result.is_err());
    }

    #[test]
    fn test_orbittrajectory_with_uuid() {
        let uuid = Uuid::new_v4();
        let traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None)
            .with_uuid(uuid);

        assert_eq!(traj.get_uuid(), Some(uuid));
    }

    #[test]
    fn test_orbittrajectory_get_eviction_policy_default() {
        let traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        assert_eq!(traj.get_eviction_policy(), TrajectoryEvictionPolicy::None);
    }

    #[test]
    fn test_orbittrajectory_timespan_empty() {
        let traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        assert!(traj.timespan().is_none());
    }

    #[test]
    fn test_orbittrajectory_timespan_single_state() {
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let mut traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        traj.add(t0, Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0));

        assert!(traj.timespan().is_none());
    }

    #[test]
    fn test_orbittrajectory_first_empty() {
        let traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        assert!(traj.first().is_none());
    }

    #[test]
    fn test_orbittrajectory_last_empty() {
        let traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        assert!(traj.last().is_none());
    }

    #[test]
    fn test_orbittrajectory_remove_epoch_not_found() {
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;
        let t_not_in = t0 + 30.0;

        let mut traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        traj.add(t0, Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0));
        traj.add(t1, Vector6::new(7100e3, 0.0, 0.0, 0.0, 7.5e3, 0.0));

        let result = traj.remove_epoch(&t_not_in);
        assert!(result.is_err());
    }

    #[test]
    fn test_orbittrajectory_remove_out_of_bounds() {
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let mut traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        traj.add(t0, Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0));

        let result = traj.remove(10);
        assert!(result.is_err());
    }

    #[test]
    fn test_orbittrajectory_get_out_of_bounds() {
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let mut traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        traj.add(t0, Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0));

        let result = traj.get(10);
        assert!(result.is_err());
    }

    #[test]
    fn test_orbittrajectory_index_before_epoch_empty() {
        let traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

        let result = traj.index_before_epoch(&t0);
        assert!(result.is_err());
    }

    #[test]
    fn test_orbittrajectory_index_before_epoch_before_all_states() {
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let t_before = t0 - 60.0;

        let mut traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        traj.add(t0, Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0));

        let result = traj.index_before_epoch(&t_before);
        assert!(result.is_err());
    }

    #[test]
    fn test_orbittrajectory_index_after_epoch_empty() {
        let traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

        let result = traj.index_after_epoch(&t0);
        assert!(result.is_err());
    }

    #[test]
    fn test_orbittrajectory_index_after_epoch_after_all_states() {
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let t_after = t0 + 60.0;

        let mut traj = SOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        traj.add(t0, Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0));

        let result = traj.index_after_epoch(&t_after);
        assert!(result.is_err());
    }
}
