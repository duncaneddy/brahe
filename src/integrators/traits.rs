/*!
Defines the `NumericalIntegrator` trait and its implementations, which provides a common interface
for numerical integration routines.
 */

use nalgebra::{DMatrix, DVector, SMatrix, SVector};

use crate::integrators::config::{AdaptiveStepSResult, IntegratorConfig};
use crate::math::jacobian::{DJacobianProvider, SJacobianProvider};
use crate::math::sensitivity::{DSensitivityProvider, SSensitivityProvider};

// ============================================================================
// Type Aliases for Static-Sized Integrators
// ============================================================================

/// Dynamics function type for static-sized state vectors.
pub type StateDynamics<const S: usize> = Box<dyn Fn(f64, SVector<f64, S>) -> SVector<f64, S>>;

/// Jacobian provider type for static-sized variational matrix propagation.
pub type VariationalMatrix<const S: usize> = Option<Box<dyn SJacobianProvider<S>>>;

/// Control input function type for static-sized state vectors.
/// Returns perturbation to be added to dynamics output.
///
/// The third parameter is an optional parameter vector that the control law may depend on.
/// This enables control laws that vary based on physical parameters (e.g., mass, area).
pub type ControlInput<const S: usize, const P: usize> =
    Option<Box<dyn Fn(f64, SVector<f64, S>, Option<&SVector<f64, P>>) -> SVector<f64, S>>>;

// ============================================================================
// Type Aliases for Dynamic-Sized Integrators
// ============================================================================

/// Dynamics function type for dynamic-sized state vectors.
///
/// The third parameter is optional consider parameters that the dynamics may depend on.
/// This enables sensitivity matrix propagation for parameter estimation.
pub type StateDynamicsD = Box<dyn Fn(f64, DVector<f64>, Option<&DVector<f64>>) -> DVector<f64>>;

/// Jacobian provider type for dynamic-sized variational matrix propagation.
pub type VariationalMatrixD = Option<Box<dyn DJacobianProvider>>;

/// Control input function type for dynamic-sized state vectors.
/// Returns perturbation to be added to dynamics output.
///
/// The third parameter is an optional parameter vector that the control law may depend on.
/// This enables control laws that vary based on physical parameters (e.g., mass, area).
pub type ControlInputD =
    Option<Box<dyn Fn(f64, DVector<f64>, Option<&DVector<f64>>) -> DVector<f64>>>;

// ============================================================================
// Sensitivity Matrix Support
// ============================================================================

// Re-export sensitivity providers from math module
pub use crate::math::sensitivity::{
    DAnalyticSensitivity, DNumericalSensitivity, SAnalyticSensitivity, SNumericalSensitivity,
};

/// Sensitivity provider type for static-sized sensitivity matrix propagation.
pub type SensitivityS<const S: usize, const P: usize> = Option<Box<dyn SSensitivityProvider<S, P>>>;

/// Sensitivity provider type for dynamic-sized sensitivity matrix propagation.
pub type SensitivityD = Option<Box<dyn DSensitivityProvider>>;

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
        f: StateDynamicsD,
        varmat: VariationalMatrixD,
        sensmat: SensitivityD,
        control: ControlInputD,
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
        f: StateDynamicsD,
        varmat: VariationalMatrixD,
        sensmat: SensitivityD,
        control: ControlInputD,
        config: IntegratorConfig,
    ) -> Self;

    /// Get the state vector dimension.
    fn dimension(&self) -> usize;

    /// Get the integrator configuration.
    fn config(&self) -> &IntegratorConfig;
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
        f: StateDynamics<S>,
        varmat: VariationalMatrix<S>,
        sensmat: SensitivityS<S, P>,
        control: ControlInput<S, P>,
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
        f: StateDynamics<S>,
        varmat: VariationalMatrix<S>,
        sensmat: SensitivityS<S, P>,
        control: ControlInput<S, P>,
        config: IntegratorConfig,
    ) -> Self;

    /// Get the integrator configuration.
    fn config(&self) -> &IntegratorConfig;
}

// ============================================================================
// Internal Step Result Types (for consolidation)
// ============================================================================

/// Internal result type for consolidated fixed-step integration (dynamic-sized).
///
/// This struct holds all possible outputs from a single integration step,
/// allowing a single internal implementation to handle all step variants.
#[derive(Debug, Clone)]
pub(crate) struct FixedStepInternalResultD {
    /// New state vector at time t + dt
    pub state: DVector<f64>,

    /// State transition matrix at time t + dt (if computed)
    pub phi: Option<DMatrix<f64>>,

    /// Sensitivity matrix at time t + dt (if computed)
    pub sens: Option<DMatrix<f64>>,
}

/// Internal result type for consolidated fixed-step integration (static-sized).
///
/// This struct holds all possible outputs from a single integration step,
/// allowing a single internal implementation to handle all step variants.
///
/// # Type Parameters
/// - `S`: State vector dimension
/// - `P`: Parameter vector dimension (for sensitivity matrix)
#[derive(Debug, Clone)]
pub(crate) struct FixedStepInternalResultS<const S: usize, const P: usize> {
    /// New state vector at time t + dt
    pub state: SVector<f64, S>,

    /// State transition matrix at time t + dt (if computed)
    pub phi: Option<SMatrix<f64, S, S>>,

    /// Sensitivity matrix at time t + dt (if computed)
    pub sens: Option<SMatrix<f64, S, P>>,
}

/// Internal result type for consolidated adaptive-step integration (dynamic-sized).
///
/// This struct holds all possible outputs from a single adaptive integration step,
/// including error estimation and step size recommendations.
#[derive(Debug, Clone)]
pub(crate) struct AdaptiveStepInternalResultD {
    /// New state vector at time t + dt_used
    pub state: DVector<f64>,

    /// State transition matrix at time t + dt_used (if computed)
    pub phi: Option<DMatrix<f64>>,

    /// Sensitivity matrix at time t + dt_used (if computed)
    pub sens: Option<DMatrix<f64>>,

    /// Actual timestep used (may be smaller than requested)
    pub dt_used: f64,

    /// Estimated truncation error
    pub error_estimate: f64,

    /// Suggested next timestep based on error control
    pub dt_next: f64,
}

/// Internal result type for consolidated adaptive-step integration (static-sized).
///
/// This struct holds all possible outputs from a single adaptive integration step,
/// including error estimation and step size recommendations.
///
/// # Type Parameters
/// - `S`: State vector dimension
/// - `P`: Parameter vector dimension (for sensitivity matrix)
#[derive(Debug, Clone)]
pub(crate) struct AdaptiveStepInternalResultS<const S: usize, const P: usize> {
    /// New state vector at time t + dt_used
    pub state: SVector<f64, S>,

    /// State transition matrix at time t + dt_used (if computed)
    pub phi: Option<SMatrix<f64, S, S>>,

    /// Sensitivity matrix at time t + dt_used (if computed)
    pub sens: Option<SMatrix<f64, S, P>>,

    /// Actual timestep used (may be smaller than requested)
    pub dt_used: f64,

    /// Estimated truncation error
    pub error_estimate: f64,

    /// Suggested next timestep based on error control
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

/// Trait defining interface for fixed-step numerical integration methods (static-sized).
///
/// Provides basic integration functionality with fixed timesteps for compile-time sized state vectors.
/// All static numerical integrators must implement this trait.
///
/// # Type Parameters
/// - `S`: State vector dimension
/// - `P`: Parameter vector dimension (for sensitivity matrix propagation)
pub trait FixedStepSIntegrator<const S: usize, const P: usize> {
    /// Advance the state by one timestep using this integration method.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `dt`: Optional integration timestep (can be negative for backward integration).
    ///   If None, uses the step size from the integrator's configuration.
    ///
    /// # Returns
    /// State vector at time t + dt
    ///
    /// # Panics
    /// Panics if `dt` is None and the integrator's configuration doesn't have a fixed_step_size set.
    fn step(&self, t: f64, state: SVector<f64, S>, dt: Option<f64>) -> SVector<f64, S>;

    /// Advance both state and state transition matrix by one timestep.
    ///
    /// Integrates state and its variational equations simultaneously for uncertainty propagation.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `phi`: State transition matrix at time t
    /// - `dt`: Optional integration timestep. If None, uses the step size from the integrator's configuration.
    ///
    /// # Returns
    /// Tuple of (state at t+dt, state transition matrix at t+dt)
    ///
    /// # Panics
    /// Panics if `dt` is None and the integrator's configuration doesn't have a fixed_step_size set.
    fn step_with_varmat(
        &self,
        t: f64,
        state: SVector<f64, S>,
        phi: SMatrix<f64, S, S>,
        dt: Option<f64>,
    ) -> (SVector<f64, S>, SMatrix<f64, S, S>);

    /// Advance state and sensitivity matrix by one timestep.
    ///
    /// Propagates the sensitivity matrix S that maps parameter uncertainties to state uncertainties.
    /// The sensitivity evolves according to dS/dt = A*S + B where A = ∂f/∂x and B = ∂f/∂p.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `sens`: Sensitivity matrix at time t (S × P)
    /// - `params`: Parameter vector
    /// - `dt`: Optional integration timestep. If None, uses the step size from configuration.
    ///
    /// # Returns
    /// Tuple of (state at t+dt, sensitivity matrix at t+dt)
    ///
    /// # Panics
    /// May panic if sensitivity provider is not configured.
    fn step_with_sensmat(
        &self,
        t: f64,
        state: SVector<f64, S>,
        sens: SMatrix<f64, S, P>,
        params: &SVector<f64, P>,
        dt: Option<f64>,
    ) -> (SVector<f64, S>, SMatrix<f64, S, P>);

    /// Advance state, variational matrix (STM), and sensitivity matrix by one timestep.
    ///
    /// Propagates both matrices simultaneously for complete uncertainty quantification.
    /// - STM (Φ): Maps initial state uncertainties to current state uncertainties
    /// - Sensitivity (S): Maps parameter uncertainties to state uncertainties
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `phi`: State transition matrix at time t (S × S)
    /// - `sens`: Sensitivity matrix at time t (S × P)
    /// - `params`: Parameter vector
    /// - `dt`: Optional integration timestep. If None, uses the step size from configuration.
    ///
    /// # Returns
    /// Tuple of (state at t+dt, STM at t+dt, sensitivity at t+dt)
    ///
    /// # Panics
    /// May panic if Jacobian/sensitivity providers are not configured.
    fn step_with_varmat_sensmat(
        &self,
        t: f64,
        state: SVector<f64, S>,
        phi: SMatrix<f64, S, S>,
        sens: SMatrix<f64, S, P>,
        params: &SVector<f64, P>,
        dt: Option<f64>,
    ) -> (SVector<f64, S>, SMatrix<f64, S, S>, SMatrix<f64, S, P>);
}

/// Trait defining interface for adaptive-step numerical integration methods (static-sized).
///
/// Provides automatic step size control based on embedded error estimation for compile-time sized state vectors.
/// Typically implemented by embedded Runge-Kutta methods (RKF45, DP54, etc.).
///
/// # Type Parameters
/// - `S`: State vector dimension
/// - `P`: Parameter vector dimension (for sensitivity matrix propagation)
pub trait AdaptiveStepSIntegrator<const S: usize, const P: usize> {
    /// Advance the state with adaptive step control.
    ///
    /// Automatically adjusts the timestep to meet specified tolerances using
    /// embedded error estimation. Tolerances are read from the integrator's
    /// configuration.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `dt`: Requested integration timestep
    ///
    /// # Returns
    /// AdaptiveStepSResult containing new state, actual dt used, error estimate, and suggested next dt
    fn step(&self, t: f64, state: SVector<f64, S>, dt: f64) -> AdaptiveStepSResult<S>;

    /// Advance state and STM with adaptive step control.
    ///
    /// Combines adaptive stepping with variational matrix propagation for uncertainty
    /// quantification with automatic step size control. Tolerances are read from the
    /// integrator's configuration.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `phi`: State transition matrix at time t
    /// - `dt`: Requested integration timestep
    ///
    /// # Returns
    /// Tuple of (new state, new STM, actual dt used, error estimate, suggested next dt)
    fn step_with_varmat(
        &self,
        t: f64,
        state: SVector<f64, S>,
        phi: SMatrix<f64, S, S>,
        dt: f64,
    ) -> (SVector<f64, S>, SMatrix<f64, S, S>, f64, f64, f64);

    /// Advance state and sensitivity matrix with adaptive step control.
    ///
    /// Propagates the sensitivity matrix S that maps parameter uncertainties to state uncertainties.
    /// The sensitivity evolves according to dS/dt = A*S + B where A = ∂f/∂x and B = ∂f/∂p.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `sens`: Sensitivity matrix at time t (S × P)
    /// - `params`: Parameter vector
    /// - `dt`: Requested integration timestep
    ///
    /// # Returns
    /// Tuple of (new state, new sensitivity, actual dt used, error estimate, suggested next dt)
    ///
    /// # Panics
    /// May panic if sensitivity provider is not configured.
    fn step_with_sensmat(
        &self,
        t: f64,
        state: SVector<f64, S>,
        sens: SMatrix<f64, S, P>,
        params: &SVector<f64, P>,
        dt: f64,
    ) -> (SVector<f64, S>, SMatrix<f64, S, P>, f64, f64, f64);

    /// Advance state, variational matrix (STM), and sensitivity matrix with adaptive step control.
    ///
    /// Propagates both matrices simultaneously for complete uncertainty quantification.
    /// - STM (Φ): Maps initial state uncertainties to current state uncertainties
    /// - Sensitivity (S): Maps parameter uncertainties to state uncertainties
    ///
    /// The matrices evolve according to:
    /// - dΦ/dt = A*Φ where A = ∂f/∂x
    /// - dS/dt = A*S + B where B = ∂f/∂p
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `phi`: State transition matrix at time t (S × S)
    /// - `sens`: Sensitivity matrix at time t (S × P)
    /// - `params`: Parameter vector
    /// - `dt`: Requested integration timestep
    ///
    /// # Returns
    /// Tuple of (new state, new STM, new sensitivity, actual dt used, error estimate, suggested next dt)
    ///
    /// # Panics
    /// May panic if Jacobian/sensitivity providers are not configured.
    fn step_with_varmat_sensmat(
        &self,
        t: f64,
        state: SVector<f64, S>,
        phi: SMatrix<f64, S, S>,
        sens: SMatrix<f64, S, P>,
        params: &SVector<f64, P>,
        dt: f64,
    ) -> (
        SVector<f64, S>,
        SMatrix<f64, S, S>,
        SMatrix<f64, S, P>,
        f64,
        f64,
        f64,
    );
}

// ============================================================================
// Dynamic Integrator Traits (Runtime-sized vectors)
// ============================================================================

/// Result type for adaptive-step integration with dynamic-sized state vectors.
///
/// Contains the new state vector, actual timestep used, error estimate, and suggested next timestep.
/// This is the dynamic counterpart to `AdaptiveStepResult<S>` for use with `DVector`.
#[derive(Debug, Clone)]
pub struct AdaptiveStepDResult {
    /// New state vector at time t + dt_used
    pub state: DVector<f64>,

    /// Actual timestep used (may be smaller than requested)
    pub dt_used: f64,

    /// Estimated truncation error
    pub error_estimate: f64,

    /// Suggested next timestep based on error control
    pub dt_next: f64,
}

/// Trait defining interface for fixed-step numerical integration with dynamic-sized state vectors.
///
/// This is the dynamic-sized counterpart to `FixedStepIntegrator<S>`. It uses `DVector` instead
/// of `SVector`, allowing state dimension to be determined at runtime rather than compile time.
/// This makes it ideal for Python bindings and scenarios where flexibility is more important
/// than compile-time optimization.
///
/// # Examples
///
/// ```rust
/// use brahe::integrators::{FixedStepDIntegrator, RK4DIntegrator};
/// use nalgebra::DVector;
///
/// let dynamics = |t: f64, state: DVector<f64>, _: Option<&DVector<f64>>| -> DVector<f64> {
///     // Simple exponential decay: dy/dt = -y
///     state.map(|y| -y)
/// };
///
/// let integrator = RK4DIntegrator::new(2, Box::new(dynamics), None, None, None);
/// let state = DVector::from_vec(vec![1.0, 2.0]);
/// let new_state = integrator.step(0.0, state, Some(0.1));
/// ```
pub trait FixedStepDIntegrator {
    /// Advance the state by one timestep using this integration method.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t (dimension must match integrator)
    /// - `dt`: Optional integration timestep (can be negative for backward integration).
    ///   If None, uses the step size from the integrator's configuration.
    ///
    /// # Returns
    /// State vector at time t + dt
    ///
    /// # Panics
    /// - May panic if `state` dimension doesn't match the integrator's expected dimension.
    /// - Panics if `dt` is None and the integrator's configuration doesn't have a fixed_step_size set.
    fn step(&self, t: f64, state: DVector<f64>, dt: Option<f64>) -> DVector<f64>;

    /// Advance both state and state transition matrix by one timestep.
    ///
    /// Integrates state and its variational equations simultaneously for uncertainty propagation.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `phi`: State transition matrix at time t (must be dimension × dimension)
    /// - `dt`: Optional integration timestep. If None, uses the step size from the integrator's configuration.
    ///
    /// # Returns
    /// Tuple of (state at t+dt, state transition matrix at t+dt)
    ///
    /// # Panics
    /// - May panic if dimensions don't match the integrator's expected dimension.
    /// - Panics if `dt` is None and the integrator's configuration doesn't have a fixed_step_size set.
    fn step_with_varmat(
        &self,
        t: f64,
        state: DVector<f64>,
        phi: DMatrix<f64>,
        dt: Option<f64>,
    ) -> (DVector<f64>, DMatrix<f64>);

    /// Advance state and sensitivity matrix by one timestep.
    ///
    /// Propagates the sensitivity matrix S that maps parameter uncertainties to state uncertainties.
    /// The sensitivity evolves according to dS/dt = A*S + B where A = ∂f/∂x and B = ∂f/∂p.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `sens`: Sensitivity matrix at time t (state_dim × param_dim)
    /// - `params`: Consider parameters
    /// - `dt`: Optional integration timestep. If None, uses the step size from configuration.
    ///
    /// # Returns
    /// Tuple of (state at t+dt, sensitivity matrix at t+dt)
    ///
    /// # Panics
    /// May panic if dimensions don't match or if sensitivity provider is not configured.
    fn step_with_sensmat(
        &self,
        t: f64,
        state: DVector<f64>,
        sens: DMatrix<f64>,
        params: &DVector<f64>,
        dt: Option<f64>,
    ) -> (DVector<f64>, DMatrix<f64>);

    /// Advance state, variational matrix (STM), and sensitivity matrix by one timestep.
    ///
    /// Propagates both matrices simultaneously for complete uncertainty quantification.
    /// - STM (Φ): Maps initial state uncertainties to current state uncertainties
    /// - Sensitivity (S): Maps parameter uncertainties to state uncertainties
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `phi`: State transition matrix at time t (state_dim × state_dim)
    /// - `sens`: Sensitivity matrix at time t (state_dim × param_dim)
    /// - `params`: Consider parameters
    /// - `dt`: Optional integration timestep. If None, uses the step size from configuration.
    ///
    /// # Returns
    /// Tuple of (state at t+dt, STM at t+dt, sensitivity at t+dt)
    ///
    /// # Panics
    /// May panic if dimensions don't match or if Jacobian/sensitivity providers are not configured.
    fn step_with_varmat_sensmat(
        &self,
        t: f64,
        state: DVector<f64>,
        phi: DMatrix<f64>,
        sens: DMatrix<f64>,
        params: &DVector<f64>,
        dt: Option<f64>,
    ) -> (DVector<f64>, DMatrix<f64>, DMatrix<f64>);
}

/// Trait defining interface for adaptive-step numerical integration with dynamic-sized state vectors.
///
/// This is the dynamic-sized counterpart to `AdaptiveStepIntegrator<S>`. It provides automatic
/// step size control with runtime-determined state dimensions, making it ideal for Python bindings
/// and applications requiring flexibility.
///
/// # Examples
///
/// ```rust
/// use brahe::integrators::{AdaptiveStepDIntegrator, RKF45DIntegrator, IntegratorConfig};
/// use nalgebra::DVector;
///
/// let dynamics = |t: f64, state: DVector<f64>, _: Option<&DVector<f64>>| -> DVector<f64> {
///     state.map(|y| -y)  // Exponential decay
/// };
///
/// let config = IntegratorConfig::adaptive(1e-9, 1e-6);
/// let integrator = RKF45DIntegrator::with_config(2, Box::new(dynamics), None, None, None, config);
/// let state = DVector::from_vec(vec![1.0, 2.0]);
/// let result = integrator.step(0.0, state, 0.1);
///
/// println!("New state: {:?}", result.state);
/// println!("Used dt: {}, Suggested next: {}", result.dt_used, result.dt_next);
/// ```
pub trait AdaptiveStepDIntegrator {
    /// Advance the state with adaptive step control.
    ///
    /// Automatically adjusts the timestep to meet specified tolerances using
    /// embedded error estimation. Tolerances are read from the integrator's
    /// configuration.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `dt`: Requested integration timestep
    ///
    /// # Returns
    /// `AdaptiveStepDResult` containing new state, actual dt used, error estimate, and suggested next dt
    ///
    /// # Panics
    /// May panic if `state` dimension doesn't match the integrator's expected dimension.
    fn step(&self, t: f64, state: DVector<f64>, dt: f64) -> AdaptiveStepDResult;

    /// Advance state and STM with adaptive step control.
    ///
    /// Combines adaptive stepping with variational matrix propagation for uncertainty
    /// quantification with automatic step size control. Tolerances are read from the
    /// integrator's configuration.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `phi`: State transition matrix at time t
    /// - `dt`: Requested integration timestep
    ///
    /// # Returns
    /// Tuple of (new state, new STM, actual dt used, error estimate, suggested next dt)
    ///
    /// # Panics
    /// May panic if dimensions don't match the integrator's expected dimension.
    fn step_with_varmat(
        &self,
        t: f64,
        state: DVector<f64>,
        phi: DMatrix<f64>,
        dt: f64,
    ) -> (DVector<f64>, DMatrix<f64>, f64, f64, f64);

    /// Advance state and sensitivity matrix with adaptive step control.
    ///
    /// Propagates the sensitivity matrix S that maps parameter uncertainties to state uncertainties.
    /// The sensitivity evolves according to dS/dt = A*S + B where A = ∂f/∂x and B = ∂f/∂p.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `sens`: Sensitivity matrix at time t (state_dim × param_dim)
    /// - `params`: Consider parameters
    /// - `dt`: Requested integration timestep
    ///
    /// # Returns
    /// Tuple of (new state, new sensitivity, actual dt used, error estimate, suggested next dt)
    ///
    /// # Panics
    /// May panic if dimensions don't match or if sensitivity provider is not configured.
    fn step_with_sensmat(
        &self,
        t: f64,
        state: DVector<f64>,
        sens: DMatrix<f64>,
        params: &DVector<f64>,
        dt: f64,
    ) -> (DVector<f64>, DMatrix<f64>, f64, f64, f64);

    /// Advance state, variational matrix (STM), and sensitivity matrix with adaptive step control.
    ///
    /// Propagates both matrices simultaneously for complete uncertainty quantification.
    /// - STM (Φ): Maps initial state uncertainties to current state uncertainties
    /// - Sensitivity (S): Maps parameter uncertainties to state uncertainties
    ///
    /// The matrices evolve according to:
    /// - dΦ/dt = A*Φ where A = ∂f/∂x
    /// - dS/dt = A*S + B where B = ∂f/∂p
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `phi`: State transition matrix at time t (state_dim × state_dim)
    /// - `sens`: Sensitivity matrix at time t (state_dim × param_dim)
    /// - `params`: Consider parameters
    /// - `dt`: Requested integration timestep
    ///
    /// # Returns
    /// Tuple of (new state, new STM, new sensitivity, actual dt used, error estimate, suggested next dt)
    ///
    /// # Panics
    /// May panic if dimensions don't match or if Jacobian/sensitivity providers are not configured.
    fn step_with_varmat_sensmat(
        &self,
        t: f64,
        state: DVector<f64>,
        phi: DMatrix<f64>,
        sens: DMatrix<f64>,
        params: &DVector<f64>,
        dt: f64,
    ) -> (DVector<f64>, DMatrix<f64>, DMatrix<f64>, f64, f64, f64);
}
