/*!
Defines the `NumericalIntegrator` trait and its implementations, which provides a common interface
for numerical integration routines.
 */

use nalgebra::{DMatrix, DVector, SMatrix, SVector};

use crate::integrators::config::AdaptiveStepSResult;

/// Trait defining interface for fixed-step numerical integration methods (static-sized).
///
/// Provides basic integration functionality with fixed timesteps for compile-time sized state vectors.
/// All static numerical integrators must implement this trait.
pub trait FixedStepSIntegrator<const S: usize> {
    /// Advance the state by one timestep using this integration method.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `dt`: Integration timestep (can be negative for backward integration)
    ///
    /// # Returns
    /// State vector at time t + dt
    fn step(&self, t: f64, state: SVector<f64, S>, dt: f64) -> SVector<f64, S>;

    /// Advance both state and state transition matrix by one timestep.
    ///
    /// Integrates state and its variational equations simultaneously for uncertainty propagation.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `phi`: State transition matrix at time t
    /// - `dt`: Integration timestep
    ///
    /// # Returns
    /// Tuple of (state at t+dt, state transition matrix at t+dt)
    fn step_with_varmat(
        &self,
        t: f64,
        state: SVector<f64, S>,
        phi: SMatrix<f64, S, S>,
        dt: f64,
    ) -> (SVector<f64, S>, SMatrix<f64, S, S>);
}

/// Trait defining interface for adaptive-step numerical integration methods (static-sized).
///
/// Provides automatic step size control based on embedded error estimation for compile-time sized state vectors.
/// Typically implemented by embedded Runge-Kutta methods (RKF45, DP54, etc.).
pub trait AdaptiveStepSIntegrator<const S: usize> {
    /// Advance the state with adaptive step control.
    ///
    /// Automatically adjusts the timestep to meet specified tolerances using
    /// embedded error estimation.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `dt`: Requested integration timestep
    /// - `abs_tol`: Absolute error tolerance
    /// - `rel_tol`: Relative error tolerance
    ///
    /// # Returns
    /// AdaptiveStepSResult containing new state, actual dt used, error estimate, and suggested next dt
    fn step(
        &self,
        t: f64,
        state: SVector<f64, S>,
        dt: f64,
        abs_tol: f64,
        rel_tol: f64,
    ) -> AdaptiveStepSResult<S>;

    /// Advance state and STM with adaptive step control.
    ///
    /// Combines adaptive stepping with variational matrix propagation for uncertainty
    /// quantification with automatic step size control.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `phi`: State transition matrix at time t
    /// - `dt`: Requested integration timestep
    /// - `abs_tol`: Absolute error tolerance
    /// - `rel_tol`: Relative error tolerance
    ///
    /// # Returns
    /// Tuple of (new state, new STM, actual dt used, error estimate, suggested next dt)
    fn step_with_varmat(
        &self,
        t: f64,
        state: SVector<f64, S>,
        phi: SMatrix<f64, S, S>,
        dt: f64,
        abs_tol: f64,
        rel_tol: f64,
    ) -> (SVector<f64, S>, SMatrix<f64, S, S>, f64, f64, f64);
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
/// let dynamics = |t: f64, state: DVector<f64>| -> DVector<f64> {
///     // Simple exponential decay: dy/dt = -y
///     state.map(|y| -y)
/// };
///
/// let integrator = RK4DIntegrator::new(2, Box::new(dynamics), None);
/// let state = DVector::from_vec(vec![1.0, 2.0]);
/// let new_state = integrator.step(0.0, state, 0.1);
/// ```
pub trait FixedStepDIntegrator {
    /// Advance the state by one timestep using this integration method.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t (dimension must match integrator)
    /// - `dt`: Integration timestep (can be negative for backward integration)
    ///
    /// # Returns
    /// State vector at time t + dt
    ///
    /// # Panics
    /// May panic if `state` dimension doesn't match the integrator's expected dimension.
    fn step(&self, t: f64, state: DVector<f64>, dt: f64) -> DVector<f64>;

    /// Advance both state and state transition matrix by one timestep.
    ///
    /// Integrates state and its variational equations simultaneously for uncertainty propagation.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `phi`: State transition matrix at time t (must be dimension × dimension)
    /// - `dt`: Integration timestep
    ///
    /// # Returns
    /// Tuple of (state at t+dt, state transition matrix at t+dt)
    ///
    /// # Panics
    /// May panic if dimensions don't match the integrator's expected dimension.
    fn step_with_varmat(
        &self,
        t: f64,
        state: DVector<f64>,
        phi: DMatrix<f64>,
        dt: f64,
    ) -> (DVector<f64>, DMatrix<f64>);
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
/// use brahe::integrators::{AdaptiveStepDIntegrator, RKF45DIntegrator};
/// use nalgebra::DVector;
///
/// let dynamics = |t: f64, state: DVector<f64>| -> DVector<f64> {
///     state.map(|y| -y)  // Exponential decay
/// };
///
/// let integrator = RKF45DIntegrator::new(2, Box::new(dynamics), None);
/// let state = DVector::from_vec(vec![1.0, 2.0]);
/// let result = integrator.step(0.0, state, 0.1, 1e-9, 1e-6);
///
/// println!("New state: {:?}", result.state);
/// println!("Used dt: {}, Suggested next: {}", result.dt_used, result.dt_next);
/// ```
pub trait AdaptiveStepDIntegrator {
    /// Advance the state with adaptive step control.
    ///
    /// Automatically adjusts the timestep to meet specified tolerances using
    /// embedded error estimation.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `dt`: Requested integration timestep
    /// - `abs_tol`: Absolute error tolerance
    /// - `rel_tol`: Relative error tolerance
    ///
    /// # Returns
    /// `AdaptiveStepDResult` containing new state, actual dt used, error estimate, and suggested next dt
    ///
    /// # Panics
    /// May panic if `state` dimension doesn't match the integrator's expected dimension.
    fn step(
        &self,
        t: f64,
        state: DVector<f64>,
        dt: f64,
        abs_tol: f64,
        rel_tol: f64,
    ) -> AdaptiveStepDResult;

    /// Advance state and STM with adaptive step control.
    ///
    /// Combines adaptive stepping with variational matrix propagation for uncertainty
    /// quantification with automatic step size control.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `phi`: State transition matrix at time t
    /// - `dt`: Requested integration timestep
    /// - `abs_tol`: Absolute error tolerance
    /// - `rel_tol`: Relative error tolerance
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
        abs_tol: f64,
        rel_tol: f64,
    ) -> (DVector<f64>, DMatrix<f64>, f64, f64, f64);
}
