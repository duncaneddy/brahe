/*!
Defines the `NumericalIntegrator` trait and its implementations, which provides a common interface
for numerical integration routines.
 */

use nalgebra::{DMatrix, DVector, SMatrix, SVector};

use crate::integrators::config::IntegratorConfig;
use crate::math::jacobian::{DJacobianProvider, SJacobianProvider};
use crate::math::sensitivity::{DSensitivityProvider, SSensitivityProvider};

// ============================================================================
// Type Aliases for Static-Sized Integrators
// ============================================================================

/// Dynamics function type for static-sized state vectors.
///
/// The third parameter is optional parameter vector that the dynamics may depend on.
/// This enables sensitivity matrix propagation for parameter estimation and matches
/// the signature of `DStateDynamics`.
///
/// # Type Parameters
/// - `S`: State vector dimension
/// - `P`: Parameter vector dimension
pub type SStateDynamics<const S: usize, const P: usize> =
    Box<dyn Fn(f64, SVector<f64, S>, Option<&SVector<f64, P>>) -> SVector<f64, S>>;

/// Jacobian provider type for static-sized variational matrix propagation.
///
/// # Type Parameters
/// - `S`: State vector dimension
/// - `P`: Parameter vector dimension
pub type SVariationalMatrix<const S: usize, const P: usize> =
    Option<Box<dyn SJacobianProvider<S, P>>>;

/// Control input function type for static-sized state vectors.
/// Returns perturbation to be added to dynamics output.
///
/// The third parameter is an optional parameter vector that the control law may depend on.
/// This enables control laws that vary based on physical parameters (e.g., mass, area).
pub type SControlInput<const S: usize, const P: usize> =
    Option<Box<dyn Fn(f64, SVector<f64, S>, Option<&SVector<f64, P>>) -> SVector<f64, S>>>;

// ============================================================================
// Type Aliases for Dynamic-Sized Integrators
// ============================================================================

/// Dynamics function type for dynamic-sized state vectors.
///
/// The third parameter is optional consider parameters that the dynamics may depend on.
/// This enables sensitivity matrix propagation for parameter estimation.
pub type DStateDynamics = Box<dyn Fn(f64, DVector<f64>, Option<&DVector<f64>>) -> DVector<f64>>;

/// Jacobian provider type for dynamic-sized variational matrix propagation.
pub type DVariationalMatrix = Option<Box<dyn DJacobianProvider>>;

/// Control input function type for dynamic-sized state vectors.
/// Returns perturbation to be added to dynamics output.
///
/// The third parameter is an optional parameter vector that the control law may depend on.
/// This enables control laws that vary based on physical parameters (e.g., mass, area).
pub type DControlInput =
    Option<Box<dyn Fn(f64, DVector<f64>, Option<&DVector<f64>>) -> DVector<f64>>>;

// ============================================================================
// Sensitivity Matrix Support
// ============================================================================

// Re-export sensitivity providers from math module
pub use crate::math::sensitivity::{
    DAnalyticSensitivity, DNumericalSensitivity, SAnalyticSensitivity, SNumericalSensitivity,
};

/// Sensitivity provider type for static-sized sensitivity matrix propagation.
pub type SSensitivity<const S: usize, const P: usize> = Option<Box<dyn SSensitivityProvider<S, P>>>;

/// Sensitivity provider type for dynamic-sized sensitivity matrix propagation.
pub type DSensitivity = Option<Box<dyn DSensitivityProvider>>;

// ============================================================================
// Base Integrator Traits
// ============================================================================

/// Base trait for dynamic-sized integrators.
///
/// Defines the common constructor and accessor interface that all dynamic-sized
/// integrators must implement.
pub trait DIntegrator: Sized {
    /// Create a new integrator with default configuration.
    ///
    /// # Arguments
    /// - `dimension`: State vector dimension
    /// - `f`: Dynamics function
    /// - `varmat`: Optional Jacobian provider for variational matrix propagation
    /// - `sensmat`: Optional sensitivity provider for parameter uncertainty propagation
    /// - `control`: Optional control input function
    fn new(
        dimension: usize,
        f: DStateDynamics,
        varmat: DVariationalMatrix,
        sensmat: DSensitivity,
        control: DControlInput,
    ) -> Self;

    /// Create a new integrator with custom configuration.
    ///
    /// # Arguments
    /// - `dimension`: State vector dimension
    /// - `f`: Dynamics function
    /// - `varmat`: Optional Jacobian provider for variational matrix propagation
    /// - `sensmat`: Optional sensitivity provider for parameter uncertainty propagation
    /// - `control`: Optional control input function
    /// - `config`: Integration configuration (tolerances, step sizes, etc.)
    fn with_config(
        dimension: usize,
        f: DStateDynamics,
        varmat: DVariationalMatrix,
        sensmat: DSensitivity,
        control: DControlInput,
        config: IntegratorConfig,
    ) -> Self;

    /// Get the state vector dimension.
    fn dimension(&self) -> usize;

    /// Get the integrator configuration.
    fn config(&self) -> &IntegratorConfig;

    /// Advance the state by one timestep using this integration method.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `dt`: Integration timestep (can be negative for backward integration).
    ///   For fixed-step integrators, can be None to use config's step size.
    ///   For adaptive-step integrators, this is the requested timestep.
    ///
    /// # Returns
    /// Integration result with state and optional adaptive step information
    fn step(&self, t: f64, state: DVector<f64>, dt: Option<f64>) -> DIntegratorStepResult;

    /// Advance both state and state transition matrix by one timestep.
    ///
    /// Integrates state and its variational equations simultaneously for uncertainty propagation.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `phi`: State transition matrix at time t
    /// - `dt`: Integration timestep (None for fixed-step to use config)
    ///
    /// # Returns
    /// Integration result with state, STM, and optional adaptive step information
    fn step_with_varmat(
        &self,
        t: f64,
        state: DVector<f64>,
        phi: DMatrix<f64>,
        dt: Option<f64>,
    ) -> DIntegratorStepResult;

    /// Advance state and sensitivity matrix by one timestep.
    ///
    /// Propagates the sensitivity matrix S that maps parameter uncertainties to state uncertainties.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `sens`: Sensitivity matrix at time t
    /// - `params`: Parameter vector
    /// - `dt`: Integration timestep (None for fixed-step to use config)
    ///
    /// # Returns
    /// Integration result with state, sensitivity matrix, and optional adaptive step information
    fn step_with_sensmat(
        &self,
        t: f64,
        state: DVector<f64>,
        sens: DMatrix<f64>,
        params: &DVector<f64>,
        dt: Option<f64>,
    ) -> DIntegratorStepResult;

    /// Advance state, variational matrix (STM), and sensitivity matrix by one timestep.
    ///
    /// Propagates both matrices simultaneously for complete uncertainty quantification.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `phi`: State transition matrix at time t
    /// - `sens`: Sensitivity matrix at time t
    /// - `params`: Parameter vector
    /// - `dt`: Integration timestep (None for fixed-step to use config)
    ///
    /// # Returns
    /// Integration result with state, STM, sensitivity, and optional adaptive step information
    fn step_with_varmat_sensmat(
        &self,
        t: f64,
        state: DVector<f64>,
        phi: DMatrix<f64>,
        sens: DMatrix<f64>,
        params: &DVector<f64>,
        dt: Option<f64>,
    ) -> DIntegratorStepResult;
}

/// Base trait for static-sized integrators.
///
/// Defines the common constructor and accessor interface that all static-sized
/// integrators must implement.
///
/// # Type Parameters
/// - `S`: State vector dimension
/// - `P`: Parameter vector dimension (for sensitivity matrix)
pub trait SIntegrator<const S: usize, const P: usize>: Sized {
    /// Create a new integrator with default configuration.
    ///
    /// # Arguments
    /// - `f`: Dynamics function
    /// - `varmat`: Optional Jacobian provider for variational matrix propagation
    /// - `sensmat`: Optional sensitivity provider for parameter uncertainty propagation
    /// - `control`: Optional control input function
    fn new(
        f: SStateDynamics<S, P>,
        varmat: SVariationalMatrix<S, P>,
        sensmat: SSensitivity<S, P>,
        control: SControlInput<S, P>,
    ) -> Self;

    /// Create a new integrator with custom configuration.
    ///
    /// # Arguments
    /// - `f`: Dynamics function
    /// - `varmat`: Optional Jacobian provider for variational matrix propagation
    /// - `sensmat`: Optional sensitivity provider for parameter uncertainty propagation
    /// - `control`: Optional control input function
    /// - `config`: Integration configuration (tolerances, step sizes, etc.)
    fn with_config(
        f: SStateDynamics<S, P>,
        varmat: SVariationalMatrix<S, P>,
        sensmat: SSensitivity<S, P>,
        control: SControlInput<S, P>,
        config: IntegratorConfig,
    ) -> Self;

    /// Get the integrator configuration.
    fn config(&self) -> &IntegratorConfig;

    /// Advance the state by one timestep using this integration method.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `dt`: Integration timestep (can be negative for backward integration).
    ///   For fixed-step integrators, can be None to use config's step size.
    ///   For adaptive-step integrators, this is the requested timestep.
    ///
    /// # Returns
    /// Integration result with state and optional adaptive step information
    fn step(&self, t: f64, state: SVector<f64, S>, dt: Option<f64>) -> SIntegratorStepResult<S, P>;

    /// Advance both state and state transition matrix by one timestep.
    ///
    /// Integrates state and its variational equations simultaneously for uncertainty propagation.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `phi`: State transition matrix at time t
    /// - `dt`: Integration timestep (None for fixed-step to use config)
    ///
    /// # Returns
    /// Integration result with state, STM, and optional adaptive step information
    fn step_with_varmat(
        &self,
        t: f64,
        state: SVector<f64, S>,
        phi: SMatrix<f64, S, S>,
        dt: Option<f64>,
    ) -> SIntegratorStepResult<S, P>;

    /// Advance state and sensitivity matrix by one timestep.
    ///
    /// Propagates the sensitivity matrix S that maps parameter uncertainties to state uncertainties.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `sens`: Sensitivity matrix at time t (S × P)
    /// - `params`: Parameter vector
    /// - `dt`: Integration timestep (None for fixed-step to use config)
    ///
    /// # Returns
    /// Integration result with state, sensitivity matrix, and optional adaptive step information
    fn step_with_sensmat(
        &self,
        t: f64,
        state: SVector<f64, S>,
        sens: SMatrix<f64, S, P>,
        params: &SVector<f64, P>,
        dt: Option<f64>,
    ) -> SIntegratorStepResult<S, P>;

    /// Advance state, variational matrix (STM), and sensitivity matrix by one timestep.
    ///
    /// Propagates both matrices simultaneously for complete uncertainty quantification.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `phi`: State transition matrix at time t (S × S)
    /// - `sens`: Sensitivity matrix at time t (S × P)
    /// - `params`: Parameter vector
    /// - `dt`: Integration timestep (None for fixed-step to use config)
    ///
    /// # Returns
    /// Integration result with state, STM, sensitivity, and optional adaptive step information
    fn step_with_varmat_sensmat(
        &self,
        t: f64,
        state: SVector<f64, S>,
        phi: SMatrix<f64, S, S>,
        sens: SMatrix<f64, S, P>,
        params: &SVector<f64, P>,
        dt: Option<f64>,
    ) -> SIntegratorStepResult<S, P>;
}

// ============================================================================
// Unified Result Types
// ============================================================================

/// Unified result type for numerical integration steps (static-sized).
///
/// This type supports both fixed-step and adaptive-step integrators.
/// For fixed-step integrators, dt_used and dt_next will equal the fixed step size,
/// and error_estimate will be None. For adaptive-step integrators, dt_used and dt_next
/// will reflect the actual used and recommended step sizes, and error_estimate will be Some.
///
/// # Type Parameters
/// - `S`: State vector dimension
/// - `P`: Parameter vector dimension (for sensitivity matrix)
#[derive(Debug, Clone)]
pub struct SIntegratorStepResult<const S: usize, const P: usize> {
    /// New state vector at time t + dt
    pub state: SVector<f64, S>,

    /// State transition matrix at time t + dt (if computed)
    pub phi: Option<SMatrix<f64, S, S>>,

    /// Sensitivity matrix at time t + dt (if computed)
    pub sens: Option<SMatrix<f64, S, P>>,

    /// Actual timestep used (equals requested dt for fixed-step integrators)
    pub dt_used: f64,

    /// Estimated truncation error (Some for adaptive integrators, None for fixed-step)
    pub error_estimate: Option<f64>,

    /// Suggested next timestep (equals dt for fixed-step integrators)
    pub dt_next: f64,
}

/// Unified result type for numerical integration steps (dynamic-sized).
///
/// This type supports both fixed-step and adaptive-step integrators.
/// For fixed-step integrators, dt_used and dt_next will equal the fixed step size,
/// and error_estimate will be None. For adaptive-step integrators, dt_used and dt_next
/// will reflect the actual used and recommended step sizes, and error_estimate will be Some.
#[derive(Debug, Clone)]
pub struct DIntegratorStepResult {
    /// New state vector at time t + dt
    pub state: DVector<f64>,

    /// State transition matrix at time t + dt (if computed)
    pub phi: Option<DMatrix<f64>>,

    /// Sensitivity matrix at time t + dt (if computed)
    pub sens: Option<DMatrix<f64>>,

    /// Actual timestep used (equals requested dt for fixed-step integrators)
    pub dt_used: f64,

    /// Estimated truncation error (Some for adaptive integrators, None for fixed-step)
    pub error_estimate: Option<f64>,

    /// Suggested next timestep (equals dt for fixed-step integrators)
    pub dt_next: f64,
}

/// Compute the optimal next step size based on error estimate.
///
/// Uses the standard error control formula for embedded RK methods.
///
/// # Arguments
/// - `error`: Normalized error estimate (should be <= 1.0 for acceptance)
/// - `h`: Current step size
/// - `order_accept`: Order for accepted step scaling (e.g., 0.2 for 5th order)
/// - `order_reject`: Order for rejected step scaling (e.g., 0.25 for 4th order)
/// - `config`: Integrator configuration
///
/// # Returns
/// Tuple of (is_accepted, next_step_size)
pub(crate) fn compute_next_step_size(
    error: f64,
    h: f64,
    order_accept: f64,
    config: &IntegratorConfig,
) -> f64 {
    if error > 0.0 {
        let raw_scale = (1.0 / error).powf(order_accept);
        let scale = config
            .step_safety_factor
            .map_or(raw_scale, |safety| safety * raw_scale);

        let h_sign = h.signum();
        let h_abs = h.abs();
        let mut h_next = h_abs * scale;

        // Apply min scale factor if configured
        if let Some(min_scale) = config.min_step_scale_factor {
            h_next = h_next.max(min_scale * h_abs);
        }

        // Apply max scale factor if configured
        if let Some(max_scale) = config.max_step_scale_factor {
            h_next = h_next.min(max_scale * h_abs);
        }

        // Apply absolute step limits if configured
        if let Some(max_step) = config.max_step {
            h_next = h_next.min(max_step);
        }
        if let Some(min_step) = config.min_step {
            h_next = h_next.max(min_step);
        }

        h_sign * h_next
    } else {
        // Error is zero - use maximum increase
        let h_sign = h.signum();
        let h_abs = h.abs();
        let h_next = if let Some(max_scale) = config.max_step_scale_factor {
            max_scale * h_abs
        } else {
            10.0 * h_abs // Default max growth if unconfigured
        };

        // Respect absolute max if configured
        let h_next = config.max_step.map_or(h_next, |max| h_next.min(max));
        h_sign * h_next
    }
}

/// Compute the reduced step size after a rejected step.
///
/// # Arguments
/// - `error`: Normalized error estimate
/// - `h`: Current step size
/// - `order_reject`: Order for rejected step scaling (e.g., 0.25 for 4th order)
/// - `config`: Integrator configuration
///
/// # Returns
/// Reduced step size to try
pub(crate) fn compute_reduced_step_size(
    error: f64,
    h: f64,
    order_reject: f64,
    config: &IntegratorConfig,
) -> f64 {
    let raw_scale = (1.0 / error).powf(order_reject);
    let scale = config
        .step_safety_factor
        .map_or(raw_scale, |safety| safety * raw_scale);

    let h_sign = h.signum();
    let h_abs = h.abs();
    let mut h_new = h_abs * scale;

    // Apply min scale factor if configured
    if let Some(min_scale) = config.min_step_scale_factor {
        h_new = h_new.max(min_scale * h_abs);
    }

    // Respect absolute minimum if configured
    if let Some(min_step) = config.min_step {
        h_new = h_new.max(min_step);
    }

    h_sign * h_new
}

/// Compute normalized error from error vector for adaptive step control (dynamic-sized).
///
/// # Arguments
/// - `error_vec`: Difference between high and low order solutions
/// - `state_high`: High-order solution
/// - `state_orig`: Original state
/// - `config`: Integrator configuration
///
/// # Returns
/// Maximum normalized error across all components
pub(crate) fn compute_normalized_error(
    error_vec: &DVector<f64>,
    state_high: &DVector<f64>,
    state_orig: &DVector<f64>,
    config: &IntegratorConfig,
) -> f64 {
    let mut error: f64 = 0.0;
    for i in 0..error_vec.len() {
        let tol = config.abs_tol + config.rel_tol * state_high[i].abs().max(state_orig[i].abs());
        error = error.max((error_vec[i] / tol).abs());
    }
    error
}

/// Compute normalized error from error vector for adaptive step control (static-sized).
///
/// # Arguments
/// - `error_vec`: Difference between high and low order solutions
/// - `state_high`: High-order solution
/// - `state_orig`: Original state
/// - `config`: Integrator configuration
///
/// # Returns
/// Maximum normalized error across all components
pub(crate) fn compute_normalized_error_s<const S: usize>(
    error_vec: &SVector<f64, S>,
    state_high: &SVector<f64, S>,
    state_orig: &SVector<f64, S>,
    config: &IntegratorConfig,
) -> f64 {
    let mut error: f64 = 0.0;
    for i in 0..S {
        let tol = config.abs_tol + config.rel_tol * state_high[i].abs().max(state_orig[i].abs());
        error = error.max((error_vec[i] / tol).abs());
    }
    error
}

/// Determines the step size to use for fixed-step integration.
///
/// # Arguments
/// - `dt`: Optional explicitly provided step size
/// - `config`: Integrator configuration that may contain a fixed step size
///
/// # Returns
/// The step size to use for integration
///
/// # Panics
/// Panics if both `dt` and `config.fixed_step_size` are `None`. Fixed-step integrators
/// require an explicit step size either through the `dt` parameter or via configuration.
///
/// # Examples
///
/// ```
/// use brahe::integrators::{IntegratorConfig, get_step_size};
///
/// let config = IntegratorConfig::fixed_step(1.0);
///
/// // Use config's step size
/// let dt = get_step_size(None, &config);
/// assert_eq!(dt, 1.0);
///
/// // Override with explicit dt
/// let dt = get_step_size(Some(0.5), &config);
/// assert_eq!(dt, 0.5);
/// ```
pub fn get_step_size(dt: Option<f64>, config: &IntegratorConfig) -> f64 {
    match dt {
        Some(step) => step,
        None => config.fixed_step_size.unwrap_or_else(|| {
            panic!(
                "Fixed-step integrator requires a step size. \
                Either provide dt to step() or set fixed_step_size in IntegratorConfig."
            )
        }),
    }
}
