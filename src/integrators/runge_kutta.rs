/*!
Implementation of Runge-Kutta integration methods.
 */

use nalgebra::{SMatrix, SVector};

#[cfg(test)]
use crate::constants::RADIANS;
use crate::integrators::butcher_tableau::{
    ButcherTableau, EmbeddedButcherTableau, RK4_TABLEAU, RKF45_TABLEAU,
};
use crate::integrators::config::{AdaptiveStepResult, IntegratorConfig};
use crate::integrators::numerical_integrator::NumericalIntegrator;

// Type aliases for complex function types
type StateDynamics<const S: usize> = Box<dyn Fn(f64, SVector<f64, S>) -> SVector<f64, S>>;
type VariationalMatrix<const S: usize> =
    Option<Box<dyn Fn(f64, SVector<f64, S>) -> SMatrix<f64, S, S>>>;

/// Implementation of the 4th order Runge-Kutta numerical integrator. This implementation is generic
/// over the size of the state vector.
///
/// # Example
///
/// ```
/// use nalgebra::{SVector, SMatrix};
/// use brahe::integrators::{RK4Integrator, NumericalIntegrator};
///
/// // Define a simple function for testing x' = 2x,
/// let f = |t: f64, state: SVector<f64, 1>| -> SVector<f64, 1> {
///    let mut state_new = SVector::<f64, 1>::zeros();
///     state_new[0] = 2.0*t;
///     state_new
/// };
///
///
/// // Create a new RK4 integrator
/// let rk4 = RK4Integrator::new(Box::new(f), None);
///
/// // Define the initial state and time step
/// let mut t = 0.0;
/// let mut state = SVector::<f64, 1>::new(0.0);
/// let dt = 0.01;
///
/// // Integrate the system forward in time to t = 1.0 (analytic solution is x = 1.0)
/// for i in 0..100{
///    state = rk4.step(t, state, dt);
///    t += dt;
/// }
///
/// assert!(state[0] - 1.0 < 1.0e-12);
///
/// // Now integrate the system forward in time to t = 10.0 (analytic solution is x = 100.0)
/// for i in 100..1000{
///     state = rk4.step(t, state, dt);
///     t += dt;
/// }
///
/// assert!(state[0] - 100.0 < 1.0e-12);
/// ```
pub struct RK4Integrator<const S: usize> {
    f: StateDynamics<S>,
    varmat: VariationalMatrix<S>,
    bt: ButcherTableau<4>,
    config: IntegratorConfig,
}

impl<const S: usize> RK4Integrator<S> {
    /// Create a new 4th-order Runge-Kutta integrator.
    ///
    /// Initializes RK4 integrator with classical Butcher tableau. Fourth-order accuracy
    /// provides good balance between accuracy and computational cost for most ODE systems.
    ///
    /// # Arguments
    /// - `f`: State derivative function defining the dynamics (closure or function pointer)
    /// - `varmat`: Variational matrix computation function for state transition matrix propagation
    ///
    /// # Returns
    /// RK4Integrator instance ready for numerical integration
    ///
    /// # Note
    /// This constructor provides backward compatibility. Uses default configuration.
    /// For custom configuration, use `with_config()`.
    pub fn new(f: StateDynamics<S>, varmat: VariationalMatrix<S>) -> Self {
        Self::with_config(f, varmat, IntegratorConfig::default())
    }

    /// Create a new 4th-order Runge-Kutta integrator with custom configuration.
    ///
    /// # Arguments
    /// - `f`: State derivative function defining the dynamics
    /// - `varmat`: Variational matrix computation function for STM propagation
    /// - `config`: Integration configuration (tolerances, step sizes, etc.)
    ///
    /// # Returns
    /// RK4Integrator instance with specified configuration
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::SVector;
    /// use brahe::integrators::{RK4Integrator, IntegratorConfig};
    ///
    /// let f = |t: f64, state: SVector<f64, 1>| -> SVector<f64, 1> {
    ///     SVector::<f64, 1>::new(2.0 * t)
    /// };
    ///
    /// let config = IntegratorConfig::fixed_step(0.01);
    /// let rk4 = RK4Integrator::with_config(Box::new(f), None, config);
    /// ```
    pub fn with_config(
        f: StateDynamics<S>,
        varmat: VariationalMatrix<S>,
        config: IntegratorConfig,
    ) -> Self {
        Self {
            f,
            varmat,
            bt: RK4_TABLEAU,
            config,
        }
    }

    /// Get a reference to the integrator configuration.
    pub fn config(&self) -> &IntegratorConfig {
        &self.config
    }
}

impl<const S: usize> NumericalIntegrator<S> for RK4Integrator<S> {
    fn step(&self, t: f64, state: SVector<f64, S>, dt: f64) -> SVector<f64, S> {
        let mut k = SMatrix::<f64, S, 4>::zeros();
        let mut state_update = SVector::<f64, S>::zeros();

        // Compute internal steps based on the Butcher tableau
        for i in 0..4 {
            let mut ksum = SVector::<f64, S>::zeros();
            for j in 0..i {
                ksum += self.bt.a[(i, j)] * k.column(j);
            }

            k.set_column(i, &(self.f)(t + self.bt.c[i] * dt, state + dt * ksum));
        }

        // Compute the state update from each internal step
        for i in 0..4 {
            state_update += dt * self.bt.b[i] * k.column(i);
        }

        // Combine the state and the state update to get the new state
        state + state_update
    }

    fn step_with_varmat(
        &self,
        t: f64,
        state: SVector<f64, S>,
        phi: SMatrix<f64, S, S>,
        dt: f64,
    ) -> (SVector<f64, S>, SMatrix<f64, S, S>) {
        // Define working variables to hold internal step state
        let mut k = SMatrix::<f64, S, 4>::zeros();
        let mut k_phi = [SMatrix::<f64, S, S>::zeros(); 4];

        // Define working variables to hold the state and variational matrix updates
        let mut state_update = SVector::<f64, S>::zeros();
        let mut phi_update = SMatrix::<f64, S, S>::zeros();

        // Compute internal steps based on the Butcher tableau
        for i in 0..4 {
            let mut ksum = SVector::<f64, S>::zeros();
            let mut k_phi_sum = SMatrix::<f64, S, S>::zeros();

            for (j, k_phi_j) in k_phi.iter().enumerate().take(i) {
                ksum += self.bt.a[(i, j)] * k.column(j);
                k_phi_sum += self.bt.a[(i, j)] * k_phi_j;
            }

            k.set_column(i, &(self.f)(t + self.bt.c[i] * dt, state + dt * ksum));
            k_phi[i] = self.varmat.as_ref().unwrap()(t + self.bt.c[i] * dt, state + dt * ksum)
                * (phi + dt * k_phi_sum);
        }

        // Compute the state update from each internal step
        for (i, k_phi_i) in k_phi.iter().enumerate().take(4) {
            state_update += dt * self.bt.b[i] * k.column(i);
            phi_update += dt * self.bt.b[i] * k_phi_i;
        }

        // Combine the state and the state update to get the new state
        (state + state_update, phi + phi_update)
    }
}

/// Runge-Kutta-Fehlberg 4(5) adaptive integrator.
///
/// Embedded 5th/4th order method with automatic step size control. Uses error
/// estimation from embedded solution to adapt timestep for efficiency and accuracy.
///
/// # Example
///
/// ```
/// use nalgebra::SVector;
/// use brahe::integrators::{RKF45Integrator, NumericalIntegrator, IntegratorConfig};
///
/// let f = |t: f64, state: SVector<f64, 1>| -> SVector<f64, 1> {
///     SVector::<f64, 1>::new(2.0 * t)
/// };
///
/// let config = IntegratorConfig::adaptive(1e-8, 1e-6);
/// let rkf45 = RKF45Integrator::with_config(Box::new(f), None, config);
///
/// let state = SVector::<f64, 1>::new(0.0);
/// let result = rkf45.step_adaptive(0.0, state, 0.5, 1e-8, 1e-6);
/// ```
pub struct RKF45Integrator<const S: usize> {
    f: StateDynamics<S>,
    varmat: VariationalMatrix<S>,
    bt: EmbeddedButcherTableau<6>,
    config: IntegratorConfig,
}

impl<const S: usize> RKF45Integrator<S> {
    /// Create a new RKF45 integrator with default configuration.
    pub fn new(f: StateDynamics<S>, varmat: VariationalMatrix<S>) -> Self {
        Self::with_config(f, varmat, IntegratorConfig::default())
    }

    /// Create a new RKF45 integrator with custom configuration.
    pub fn with_config(
        f: StateDynamics<S>,
        varmat: VariationalMatrix<S>,
        config: IntegratorConfig,
    ) -> Self {
        Self {
            f,
            varmat,
            bt: RKF45_TABLEAU,
            config,
        }
    }

    /// Get a reference to the integrator configuration.
    pub fn config(&self) -> &IntegratorConfig {
        &self.config
    }

    /// Compute single RK step with both high and low order solutions.
    ///
    /// Returns (high_order_state, low_order_state) for error estimation.
    fn step_embedded(
        &self,
        t: f64,
        state: SVector<f64, S>,
        dt: f64,
    ) -> (SVector<f64, S>, SVector<f64, S>) {
        let mut k = SMatrix::<f64, S, 6>::zeros();

        // Compute internal stages
        for i in 0..6 {
            let mut ksum = SVector::<f64, S>::zeros();
            for j in 0..i {
                ksum += self.bt.a[(i, j)] * k.column(j);
            }
            k.set_column(i, &(self.f)(t + self.bt.c[i] * dt, state + dt * ksum));
        }

        // Compute high-order solution (5th order)
        let mut state_high = SVector::<f64, S>::zeros();
        for i in 0..6 {
            state_high += dt * self.bt.b_high[i] * k.column(i);
        }

        // Compute low-order solution (4th order)
        let mut state_low = SVector::<f64, S>::zeros();
        for i in 0..6 {
            state_low += dt * self.bt.b_low[i] * k.column(i);
        }

        (state + state_high, state + state_low)
    }
}

impl<const S: usize> NumericalIntegrator<S> for RKF45Integrator<S> {
    fn step(&self, t: f64, state: SVector<f64, S>, dt: f64) -> SVector<f64, S> {
        // Use high-order solution for fixed-step integration
        let (state_high, _) = self.step_embedded(t, state, dt);
        state_high
    }

    fn step_with_varmat(
        &self,
        t: f64,
        state: SVector<f64, S>,
        phi: SMatrix<f64, S, S>,
        dt: f64,
    ) -> (SVector<f64, S>, SMatrix<f64, S, S>) {
        let mut k = SMatrix::<f64, S, 6>::zeros();
        let mut k_phi = [SMatrix::<f64, S, S>::zeros(); 6];

        // Compute internal stages for both state and STM
        for i in 0..6 {
            let mut ksum = SVector::<f64, S>::zeros();
            let mut k_phi_sum = SMatrix::<f64, S, S>::zeros();

            for (j, k_phi_j) in k_phi.iter().enumerate().take(i) {
                ksum += self.bt.a[(i, j)] * k.column(j);
                k_phi_sum += self.bt.a[(i, j)] * k_phi_j;
            }

            k.set_column(i, &(self.f)(t + self.bt.c[i] * dt, state + dt * ksum));
            k_phi[i] = self.varmat.as_ref().unwrap()(t + self.bt.c[i] * dt, state + dt * ksum)
                * (phi + dt * k_phi_sum);
        }

        // Use high-order solution
        let mut state_update = SVector::<f64, S>::zeros();
        let mut phi_update = SMatrix::<f64, S, S>::zeros();

        for (i, k_phi_i) in k_phi.iter().enumerate().take(6) {
            state_update += dt * self.bt.b_high[i] * k.column(i);
            phi_update += dt * self.bt.b_high[i] * k_phi_i;
        }

        (state + state_update, phi + phi_update)
    }

    fn step_adaptive(
        &self,
        t: f64,
        state: SVector<f64, S>,
        dt: f64,
        abs_tol: f64,
        rel_tol: f64,
    ) -> AdaptiveStepResult<S> {
        let mut h = dt;
        let mut attempts = 0;

        loop {
            attempts += 1;
            if attempts > self.config.max_step_attempts {
                // Fall back to minimum step if configured, otherwise use current
                if let Some(min_step) = self.config.min_step {
                    h = min_step;
                }
                break;
            }

            // Compute embedded solutions
            let (state_high, state_low) = self.step_embedded(t, state, h);

            // Estimate error as difference between solutions
            let error_vec = state_high - state_low;
            let mut error: f64 = 0.0;

            // Compute normalized error
            for i in 0..S {
                let tol = abs_tol + rel_tol * state_high[i].abs().max(state[i].abs());
                error = error.max((error_vec[i] / tol).abs());
            }

            // Check if step should be accepted
            let min_step_reached = self.config.min_step.is_some_and(|min| h <= min);

            if error <= 1.0 || min_step_reached {
                // Step accepted - calculate optimal next step size
                let dt_next = if error > 0.0 {
                    // Use 5th order accuracy for step size calculation (power = 1/5)
                    let raw_scale = (1.0 / error).powf(0.2);
                    let scale = self
                        .config
                        .step_safety_factor
                        .map_or(raw_scale, |safety| safety * raw_scale);

                    let mut h_next = h * scale;

                    // Apply min scale factor if configured
                    if let Some(min_scale) = self.config.min_step_scale_factor {
                        h_next = h_next.max(min_scale * h);
                    }

                    // Apply max scale factor if configured
                    if let Some(max_scale) = self.config.max_step_scale_factor {
                        h_next = h_next.min(max_scale * h);
                    }

                    // Apply absolute step limits if configured
                    if let Some(max_step) = self.config.max_step {
                        h_next = h_next.min(max_step);
                    }
                    if let Some(min_step) = self.config.min_step {
                        h_next = h_next.max(min_step);
                    }

                    h_next
                } else {
                    // Error is zero - use maximum increase
                    let h_next = if let Some(max_scale) = self.config.max_step_scale_factor {
                        max_scale * h
                    } else {
                        10.0 * h // Default max growth if unconfigured
                    };

                    // Respect absolute max if configured
                    self.config.max_step.map_or(h_next, |max| h_next.min(max))
                };

                return AdaptiveStepResult {
                    state: state_high,
                    dt_used: h,
                    error_estimate: error,
                    dt_next,
                };
            }

            // Step rejected - reduce step size using 4th order (power = 1/4)
            let raw_scale = (1.0 / error).powf(0.25);
            let scale = self
                .config
                .step_safety_factor
                .map_or(raw_scale, |safety| safety * raw_scale);

            let mut h_new = h * scale;

            // Apply min scale factor if configured
            if let Some(min_scale) = self.config.min_step_scale_factor {
                h_new = h_new.max(min_scale * h);
            }

            // Respect absolute minimum if configured
            if let Some(min_step) = self.config.min_step {
                h_new = h_new.max(min_step);
            }

            h = h_new;
        }

        // Fallback: use minimum step
        let (state_high, state_low) = self.step_embedded(t, state, h);
        let error_vec = state_high - state_low;
        let mut error: f64 = 0.0;
        for i in 0..S {
            let scale = abs_tol + rel_tol * state_high[i].abs().max(state[i].abs());
            error = error.max((error_vec[i] / scale).abs());
        }

        AdaptiveStepResult {
            state: state_high,
            dt_used: h,
            error_estimate: error,
            dt_next: h, // Keep same step for fallback
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use crate::{
        Epoch, GM_EARTH, R_EARTH, TimeSystem, orbital_period, state_osculating_to_cartesian,
        varmat_from_fixed_offset, varmat_from_offset_vector,
    };

    use super::*;

    #[test]
    fn test_rk4_integrator_cubic() {
        // Define a simple function for testing x' = 2x,
        let f = |t: f64, _: SVector<f64, 1>| -> SVector<f64, 1> {
            let mut state_new = SVector::<f64, 1>::zeros();
            state_new[0] = 3.0 * t * t;
            state_new
        };

        let rk4 = RK4Integrator::new(Box::new(f), None);

        let mut state = SVector::<f64, 1>::new(0.0);
        let dt = 1.0;

        for i in 0..10 {
            state = rk4.step(i as f64, state, dt);
        }

        assert_abs_diff_eq!(state[0], 1000.0, epsilon = 1.0e-12);
    }

    #[test]
    fn test_rk4_integrator_parabola() {
        // Define a simple function for testing x' = 2x,
        let f = |t: f64, _: SVector<f64, 1>| -> SVector<f64, 1> {
            let mut state_new = SVector::<f64, 1>::zeros();
            state_new[0] = 2.0 * t;
            state_new
        };

        let rk4 = RK4Integrator::new(Box::new(f), None);

        let mut t = 0.0;
        let mut state = SVector::<f64, 1>::new(0.0);
        let dt = 0.01;

        for _ in 0..100 {
            state = rk4.step(t, state, dt);
            t += dt;
        }

        assert_abs_diff_eq!(state[0], 1.0, epsilon = 1.0e-12);
    }

    fn point_earth(_: f64, x: SVector<f64, 6>) -> SVector<f64, 6> {
        let r = x.fixed_rows::<3>(0);
        let v = x.fixed_rows::<3>(3);

        // Calculate acceleration
        let a = -GM_EARTH / r.norm().powi(3);

        // Construct state derivative
        let r_dot = v;
        let v_dot = a * r;

        let mut x_dot = SVector::<f64, 6>::zeros();
        x_dot.fixed_rows_mut::<3>(0).copy_from(&r_dot);
        x_dot.fixed_rows_mut::<3>(3).copy_from(&v_dot);

        x_dot
    }

    #[test]
    fn test_rk4_integrator_orbit() {
        let rk4 = RK4Integrator::new(Box::new(point_earth), None);

        // Get start state
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 90.0, 0.0, 0.0, 0.0);
        let state0 = state_osculating_to_cartesian(oe0, RADIANS);
        let mut state = state0;

        // Get start and end times of propagation (1 orbit)
        let epc0 = Epoch::from_date(2024, 1, 1, TimeSystem::TAI);
        let epcf = epc0 + orbital_period(oe0[0]);
        let mut dt;
        let mut epc = epc0;

        while epc < epcf {
            dt = (epcf - epc).min(1.0);
            state = rk4.step(epc - epc0, state, dt);
            epc += dt;
        }

        assert_abs_diff_eq!(state.norm(), state0.norm(), epsilon = 1.0e-7);
        assert_abs_diff_eq!(state[0], state0[0], epsilon = 1.0e-5);
        assert_abs_diff_eq!(state[1], state0[1], epsilon = 1.0e-5);
        assert_abs_diff_eq!(state[2], state0[2], epsilon = 1.0e-5);
        assert_abs_diff_eq!(state[3], state0[3], epsilon = 1.0e-5);
        assert_abs_diff_eq!(state[4], state0[4], epsilon = 1.0e-5);
        assert_abs_diff_eq!(state[5], state0[5], epsilon = 1.0e-5);
    }

    #[test]
    fn test_rk4_integrator_varmat() {
        // Define how we want to calculate the variational matrix for the RK4 integrator
        let varmat = |t: f64, state: SVector<f64, 6>| -> SMatrix<f64, 6, 6> {
            varmat_from_fixed_offset(t, state, &point_earth, 1.0)
        };

        let rk4 = RK4Integrator::new(Box::new(point_earth), Some(Box::new(varmat)));

        // Get start state
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 90.0, 0.0, 0.0, 0.0);
        let state0 = state_osculating_to_cartesian(oe0, RADIANS);
        let phi0 = SMatrix::<f64, 6, 6>::identity();

        // Take no setp and confirm the variational matrix is the identity matrix
        let (_, phi1) = rk4.step_with_varmat(0.0, state0, phi0, 0.0);
        for i in 0..6 {
            for j in 0..6 {
                if i == j {
                    assert_abs_diff_eq!(phi1[(i, j)], 1.0, epsilon = 1.0e-12);
                } else {
                    assert_abs_diff_eq!(phi1[(i, j)], 0.0, epsilon = 1.0e-12);
                }
            }
        }

        // Propagate one step and indecently confirm the variational matrix update
        let (_, phi2) = rk4.step_with_varmat(0.0, state0, phi0, 1.0);
        for i in 0..6 {
            for j in 0..6 {
                if i == j {
                    assert_ne!(phi2[(i, i)], 1.0);
                    assert_ne!(phi2[(i, i)], 0.0);
                    assert_abs_diff_eq!(phi2[(i, i)], 1.0, epsilon = 1.0e-5);
                } else {
                    // Ensure there are off-diagonal elements are now populated
                    assert_ne!(phi2[(i, j)], 0.0);
                }
            }
        }

        // Compare updating the state with a perturbation and the result from using the variational matrix
        // Define a simple perturbation to simplify the tests
        let pert = SVector::<f64, 6>::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let varmat = |t: f64, state: SVector<f64, 6>| -> SMatrix<f64, 6, 6> {
            varmat_from_offset_vector(
                t,
                state,
                &point_earth,
                SVector::<f64, 6>::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            )
        };
        let rk4 = RK4Integrator::new(Box::new(point_earth), Some(Box::new(varmat)));

        // Get the state with a perturbation
        let (state_pert, _) = rk4.step_with_varmat(0.0, state0 + pert, phi0, 1.0);

        // Get the state with a perturbation by using the integrated variational matrix
        let state_stm = rk4.step(0.0, state0, 1.0) + phi2 * pert;

        // Compare the two states - they should be the same
        assert_abs_diff_eq!(state_pert[0], state_stm[0], epsilon = 1.0e-9);
        assert_abs_diff_eq!(state_pert[1], state_stm[1], epsilon = 1.0e-9);
        assert_abs_diff_eq!(state_pert[2], state_stm[2], epsilon = 1.0e-9);
        assert_abs_diff_eq!(state_pert[3], state_stm[3], epsilon = 1.0e-9);
        assert_abs_diff_eq!(state_pert[4], state_stm[4], epsilon = 1.0e-9);
        assert_abs_diff_eq!(state_pert[5], state_stm[5], epsilon = 1.0e-9);
    }

    // RKF45 Tests

    #[test]
    fn test_rkf45_integrator_parabola() {
        // Test RKF45 on simple parabola x' = 2t
        let f = |t: f64, _: SVector<f64, 1>| -> SVector<f64, 1> { SVector::<f64, 1>::new(2.0 * t) };

        let rkf45 = RKF45Integrator::new(Box::new(f), None);

        let mut t = 0.0;
        let mut state = SVector::<f64, 1>::new(0.0);
        let dt = 0.01;

        for _ in 0..100 {
            state = rkf45.step(t, state, dt);
            t += dt;
        }

        // At t=1.0, x should be 1.0 (integral of 2t from 0 to 1)
        assert_abs_diff_eq!(state[0], 1.0, epsilon = 1.0e-10);
    }

    #[test]
    fn test_rkf45_integrator_adaptive() {
        // Test adaptive stepping on parabola
        let f = |t: f64, _: SVector<f64, 1>| -> SVector<f64, 1> { SVector::<f64, 1>::new(2.0 * t) };

        use crate::integrators::IntegratorConfig;
        let config = IntegratorConfig::adaptive(1e-10, 1e-8);
        let rkf45 = RKF45Integrator::with_config(Box::new(f), None, config);

        let mut t = 0.0;
        let mut state = SVector::<f64, 1>::new(0.0);
        let t_end = 1.0;

        while t < t_end {
            let dt = f64::min(t_end - t, 0.1);
            let result = rkf45.step_adaptive(t, state, dt, 1e-10, 1e-8);
            state = result.state;
            t += result.dt_used;

            // Verify that error estimate is reasonable
            assert!(result.error_estimate >= 0.0);
        }

        // Should still get accurate result
        assert_abs_diff_eq!(state[0], 1.0, epsilon = 1.0e-8);
    }

    #[test]
    fn test_rkf45_integrator_orbit() {
        // Test RKF45 on orbital mechanics (more stringent than RK4)
        let rkf45 = RKF45Integrator::new(Box::new(point_earth), None);

        // Get start state
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 90.0, 0.0, 0.0, 0.0);
        let state0 = state_osculating_to_cartesian(oe0, RADIANS);
        let mut state = state0;

        // Get start and end times of propagation (1 orbit)
        let epc0 = Epoch::from_date(2024, 1, 1, TimeSystem::TAI);
        let epcf = epc0 + orbital_period(oe0[0]);
        let mut dt;
        let mut epc = epc0;

        while epc < epcf {
            dt = (epcf - epc).min(10.0); // Larger steps than RK4 due to higher order
            state = rkf45.step(epc - epc0, state, dt);
            epc += dt;
        }

        // RKF45 should achieve good accuracy
        assert_abs_diff_eq!(state.norm(), state0.norm(), epsilon = 1.0e-4);
        assert_abs_diff_eq!(state[0], state0[0], epsilon = 1.0e-3);
        assert_abs_diff_eq!(state[1], state0[1], epsilon = 1.0e-3);
        assert_abs_diff_eq!(state[2], state0[2], epsilon = 1.0e-3);
    }

    #[test]
    fn test_rkf45_accuracy() {
        // Verify RKF45 achieves expected 5th order accuracy
        let f =
            |t: f64, _: SVector<f64, 1>| -> SVector<f64, 1> { SVector::<f64, 1>::new(3.0 * t * t) };

        let rkf45 = RKF45Integrator::new(Box::new(f), None);

        // Use moderate step size
        let dt = 0.1;
        let mut state = SVector::<f64, 1>::new(0.0);

        for i in 0..100 {
            let t = i as f64 * dt;
            state = rkf45.step(t, state, dt);
        }

        let exact = 1000.0; // t^3 at t=10
        let error = (state[0] - exact).abs();

        // 5th order method should be very accurate
        assert!(error < 1.0e-5);
    }

    #[test]
    fn test_rkf45_step_size_increases() {
        // Verify that adaptive stepping increases step size when error is small
        let f = |t: f64, _: SVector<f64, 1>| -> SVector<f64, 1> { SVector::<f64, 1>::new(2.0 * t) };

        use crate::integrators::IntegratorConfig;
        let config = IntegratorConfig::adaptive(1e-6, 1e-4);
        let rkf45 = RKF45Integrator::with_config(Box::new(f), None, config);

        let state = SVector::<f64, 1>::new(0.0);
        let dt_initial = 0.01;

        // Take a step with loose tolerance - error should be small
        let result = rkf45.step_adaptive(0.0, state, dt_initial, 1e-6, 1e-4);

        // For this simple problem with loose tolerance, suggested step should be larger
        assert!(
            result.dt_next > dt_initial,
            "Expected dt_next ({}) > dt_initial ({})",
            result.dt_next,
            dt_initial
        );

        // Error should be very small for this simple problem
        assert!(result.error_estimate < 0.1);
    }

    #[test]
    fn test_rkf45_step_size_decreases() {
        // Verify that adaptive stepping decreases step size when error is large
        let f = |_t: f64, state: SVector<f64, 1>| -> SVector<f64, 1> {
            // Stiff problem: y' = -1000 * y
            SVector::<f64, 1>::new(-1000.0 * state[0])
        };

        use crate::integrators::IntegratorConfig;
        let config = IntegratorConfig::adaptive(1e-10, 1e-8);
        let rkf45 = RKF45Integrator::with_config(Box::new(f), None, config);

        let state = SVector::<f64, 1>::new(1.0);
        let dt_initial = 0.1; // Too large for this stiff problem

        // This should trigger step rejection and reduction
        let result = rkf45.step_adaptive(0.0, state, dt_initial, 1e-10, 1e-8);

        // Step should have been reduced from initial
        assert!(result.dt_used <= dt_initial);
    }

    #[test]
    fn test_rkf45_config_parameters() {
        // Verify that config parameters are actually used
        let f = |t: f64, _: SVector<f64, 1>| -> SVector<f64, 1> { SVector::<f64, 1>::new(2.0 * t) };

        use crate::integrators::IntegratorConfig;
        let mut config = IntegratorConfig::adaptive(1e-8, 1e-6);
        config.step_safety_factor = Some(0.5); // Very conservative
        config.max_step_scale_factor = Some(2.0); // Limit growth

        let rkf45 = RKF45Integrator::with_config(Box::new(f), None, config);

        let state = SVector::<f64, 1>::new(0.0);
        let result = rkf45.step_adaptive(0.0, state, 0.01, 1e-8, 1e-6);

        // With safety factor 0.5, growth should be limited
        assert!(result.dt_next <= 2.0 * result.dt_used);
    }

    #[test]
    fn test_rkf45_no_limits() {
        // Verify that setting limits to None removes protections
        let f = |t: f64, _: SVector<f64, 1>| -> SVector<f64, 1> { SVector::<f64, 1>::new(2.0 * t) };

        use crate::integrators::IntegratorConfig;
        let mut config = IntegratorConfig::adaptive(1e-8, 1e-6);
        config.min_step = None; // No minimum
        config.max_step = None; // No maximum
        config.min_step_scale_factor = None; // No limit on reduction
        config.max_step_scale_factor = None; // No limit on growth

        let rkf45 = RKF45Integrator::with_config(Box::new(f), None, config);

        let state = SVector::<f64, 1>::new(0.0);
        let result = rkf45.step_adaptive(0.0, state, 0.001, 1e-8, 1e-6);

        // With no max_step_scale_factor, step can grow beyond typical 10x limit
        // (though actual growth depends on error)
        assert!(result.dt_next > 0.0);
    }
}
