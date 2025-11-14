/*!
Defines the `NumericalIntegrator` trait and its implementations, which provides a common interface
for numerical integration routines.
 */

use nalgebra::{SMatrix, SVector};

use crate::integrators::config::AdaptiveStepResult;

/// Trait defining interface for numerical integration methods.
pub trait NumericalIntegrator<const S: usize> {
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

    /// Advance the state with adaptive step control (if supported).
    ///
    /// For integrators with embedded error estimation, automatically adjusts the timestep
    /// to meet specified tolerances. Falls back to fixed-step for non-adaptive methods.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `dt`: Requested integration timestep
    /// - `abs_tol`: Absolute error tolerance
    /// - `rel_tol`: Relative error tolerance
    ///
    /// # Returns
    /// AdaptiveStepResult containing new state, actual dt used, and error estimate
    fn step_adaptive(
        &self,
        t: f64,
        state: SVector<f64, S>,
        dt: f64,
        _abs_tol: f64,
        _rel_tol: f64,
    ) -> AdaptiveStepResult<S> {
        // Default implementation: fixed-step with zero error estimate
        AdaptiveStepResult {
            state: self.step(t, state, dt),
            dt_used: dt,
            error_estimate: 0.0,
            dt_next: dt, // Suggest same step size for non-adaptive methods
        }
    }

    /// Advance state and STM with adaptive step control (if supported).
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
    /// Tuple of (new state, new STM, actual dt used, error estimate)
    fn step_adaptive_with_varmat(
        &self,
        t: f64,
        state: SVector<f64, S>,
        phi: SMatrix<f64, S, S>,
        dt: f64,
        _abs_tol: f64,
        _rel_tol: f64,
    ) -> (SVector<f64, S>, SMatrix<f64, S, S>, f64, f64) {
        // Default implementation: fixed-step with zero error estimate
        let (new_state, new_phi) = self.step_with_varmat(t, state, phi, dt);
        (new_state, new_phi, dt, 0.0)
    }
}

/// Compute state transition matrix (variational matrix) using percentage-based finite differences.
///
/// Perturbs each state component by a percentage of its value to compute partial derivatives
/// numerically. Suitable when state components have similar scales and non-zero values.
///
/// # Arguments
/// - `t`: Current time
/// - `state`: State vector at time t
/// - `f`: State derivative function (dynamics)
/// - `percentage`: Fractional perturbation size (e.g., 0.01 for 1%)
///
/// # Returns
/// State transition matrix (Jacobian) approximated via finite differences
pub fn varmat_from_percentage_offset<const S: usize>(
    t: f64,
    state: SVector<f64, S>,
    f: &dyn Fn(f64, SVector<f64, S>) -> SVector<f64, S>,
    percentage: f64,
) -> SMatrix<f64, S, S> {
    // Note: The variational matrix development along with the perturbation calculation is
    // defined as a seprate funciton in the RK4Integrator struct instead of being implemented
    // as part of the rk_step_varmat function. This is because the choice of how the variational
    // matrix is calculated is specific

    // Evaluate unperturbed state derivative function
    let fx = f(t, state);

    // Initialize the variational matrix
    let mut phi = SMatrix::<f64, S, S>::zeros();

    // Compute the variational matrix for each state component
    for i in 0..S {
        // Compute the perturbation for the element
        let mut px = state;
        let offset = state[i] * percentage;
        px[i] += offset;

        let pfx = f(t, px);
        phi.set_column(i, &((pfx - fx) / offset));
    }

    phi
}

/// Compute state transition matrix using fixed absolute offset finite differences.
///
/// Perturbs each state component by a fixed absolute amount to compute partial derivatives.
/// Suitable when all state components have similar units/scales or when percentage-based
/// perturbations would fail (e.g., near-zero values).
///
/// # Arguments
/// - `t`: Current time
/// - `state`: State vector at time t
/// - `f`: State derivative function (dynamics)
/// - `offset`: Absolute perturbation size for all components
///
/// # Returns
/// State transition matrix (Jacobian) approximated via finite differences
pub fn varmat_from_fixed_offset<const S: usize>(
    t: f64,
    state: SVector<f64, S>,
    f: &dyn Fn(f64, SVector<f64, S>) -> SVector<f64, S>,
    offset: f64,
) -> SMatrix<f64, S, S> {
    // Note: The variational matrix development along with the perturbation calculation is
    // defined as a seprate funciton in the RK4Integrator struct instead of being implemented
    // as part of the rk_step_varmat function. This is because the choice of how the variational
    // matrix is calculated is specific

    // Evaluate unperturbed state derivative function
    let fx = f(t, state);

    // Initialize the variational matrix
    let mut phi = SMatrix::<f64, S, S>::zeros();

    // Compute the variational matrix for each state component
    for i in 0..S {
        // Compute the perturbation for the element
        let mut px = state;
        px[i] += offset;

        let pfx = f(t, px);
        phi.set_column(i, &((pfx - fx) / offset));
    }

    phi
}

/// Compute state transition matrix using component-specific offset finite differences.
///
/// Perturbs each state component by its corresponding offset value to compute partial
/// derivatives. Most flexible option, allowing different perturbation sizes for different
/// state components (e.g., position vs velocity).
///
/// # Arguments
/// - `t`: Current time
/// - `state`: State vector at time t
/// - `f`: State derivative function (dynamics)
/// - `offset`: Vector of perturbation sizes, one for each state component
///
/// # Returns
/// State transition matrix (Jacobian) approximated via finite differences
pub fn varmat_from_offset_vector<const S: usize>(
    t: f64,
    state: SVector<f64, S>,
    f: &dyn Fn(f64, SVector<f64, S>) -> SVector<f64, S>,
    offset: SVector<f64, S>,
) -> SMatrix<f64, S, S> {
    // Note: The variational matrix development along with the perturbation calculation is
    // defined as a seprate funciton in the RK4Integrator struct instead of being implemented
    // as part of the rk_step_varmat function. This is because the choice of how the variational
    // matrix is calculated is specific

    // Evaluate unperturbed state derivative function
    let fx = f(t, state);

    // Initialize the variational matrix
    let mut phi = SMatrix::<f64, S, S>::zeros();

    // Compute the variational matrix for each state component
    for i in 0..S {
        // Compute the perturbed for each state component
        let mut px = state;
        px[i] += offset[i];

        let pfx = f(t, px);
        phi.set_column(i, &((pfx - fx) / offset[i]));
    }

    phi
}

#[cfg(test)]
mod tests {
    use nalgebra::SVector;

    #[test]
    fn test_varmat_from_percentage_offset() {
        let t = 0.0;
        let state = SVector::<f64, 2>::new(1.0, 2.0);
        let f = |_t: f64, state: SVector<f64, 2>| -> SVector<f64, 2> {
            SVector::<f64, 2>::new(state[0], state[1])
        };

        let phi = super::varmat_from_percentage_offset(t, state, &f, 0.01);
        assert!(phi[(0, 0)] >= 1.0);
        assert_eq!(phi[(0, 1)], 0.0);
        assert_eq!(phi[(1, 0)], 0.0);
        assert!(phi[(1, 1)] >= 1.0);
    }

    #[test]
    fn test_varmat_from_fixed_offset() {
        let t = 0.0;
        let state = SVector::<f64, 2>::new(1.0, 2.0);
        let f = |_t: f64, state: SVector<f64, 2>| -> SVector<f64, 2> {
            SVector::<f64, 2>::new(state[0], state[1])
        };

        let phi = super::varmat_from_fixed_offset(t, state, &f, 0.01);
        assert_ne!(phi[(0, 0)], 1.0);
        assert_eq!(phi[(0, 1)], 0.0);
        assert_eq!(phi[(1, 0)], 0.0);
        assert_ne!(phi[(1, 1)], 1.0);
    }

    #[test]
    fn test_varmat_from_offset_vector() {
        let t = 0.0;
        let state = SVector::<f64, 2>::new(1.0, 2.0);
        let f = |_t: f64, state: SVector<f64, 2>| -> SVector<f64, 2> {
            SVector::<f64, 2>::new(state[0], state[1])
        };

        let offset = SVector::<f64, 2>::new(0.01, 0.01);
        let phi = super::varmat_from_offset_vector(t, state, &f, offset);
        assert_ne!(phi[(0, 0)], 1.0);
        assert_eq!(phi[(0, 1)], 0.0);
        assert_eq!(phi[(1, 0)], 0.0);
        assert_ne!(phi[(1, 1)], 1.0);
    }
}
