/*!
Implementation of the Runge-Kutta-Fehlberg 4(5) adaptive integration method.
 */

use nalgebra::{DMatrix, DVector, SMatrix, SVector};

use crate::integrators::butcher_tableau::{EmbeddedButcherTableau, RKF45_TABLEAU};
use crate::integrators::config::{AdaptiveStepSResult, IntegratorConfig};
use crate::integrators::traits::{
    AdaptiveStepDIntegrator, AdaptiveStepDResult, AdaptiveStepInternalResultD,
    AdaptiveStepInternalResultS, AdaptiveStepSIntegrator, ControlInput, ControlInputD, DIntegrator,
    SensitivityD, SensitivityS, StateDynamics, StateDynamicsD, VariationalMatrix,
    VariationalMatrixD, compute_next_step_size, compute_normalized_error,
    compute_normalized_error_s, compute_reduced_step_size,
};

/// Runge-Kutta-Fehlberg 4(5) adaptive integrator.
///
/// Embedded 5th/4th order method with automatic step size control. Uses error
/// estimation from embedded solution to adapt timestep for efficiency and accuracy.
///
/// # Example
///
/// ```
/// use nalgebra::SVector;
/// use brahe::integrators::{RKF45SIntegrator, AdaptiveStepSIntegrator, IntegratorConfig};
///
/// let f = |t: f64, state: SVector<f64, 1>| -> SVector<f64, 1> {
///     SVector::<f64, 1>::new(2.0 * t)
/// };
///
/// let config = IntegratorConfig::adaptive(1e-8, 1e-6);
/// let rkf45: RKF45SIntegrator<1, 0> = RKF45SIntegrator::with_config(Box::new(f), None, None, None, config);
///
/// let state = SVector::<f64, 1>::new(0.0);
/// let result = rkf45.step(0.0, state, 0.5);
/// ```
pub struct RKF45SIntegrator<const S: usize, const P: usize> {
    f: StateDynamics<S>,
    varmat: VariationalMatrix<S>,
    sensmat: SensitivityS<S, P>,
    control: ControlInput<S>,
    bt: EmbeddedButcherTableau<6>,
    config: IntegratorConfig,
}

impl<const S: usize, const P: usize> RKF45SIntegrator<S, P> {
    /// Create a new RKF45 integrator with default configuration.
    pub fn new(
        f: StateDynamics<S>,
        varmat: VariationalMatrix<S>,
        sensmat: SensitivityS<S, P>,
        control: ControlInput<S>,
    ) -> Self {
        Self::with_config(f, varmat, sensmat, control, IntegratorConfig::default())
    }

    /// Create a new RKF45 integrator with custom configuration.
    pub fn with_config(
        f: StateDynamics<S>,
        varmat: VariationalMatrix<S>,
        sensmat: SensitivityS<S, P>,
        control: ControlInput<S>,
        config: IntegratorConfig,
    ) -> Self {
        Self {
            f,
            varmat,
            sensmat,
            control,
            bt: RKF45_TABLEAU,
            config,
        }
    }

    /// Get a reference to the integrator configuration.
    pub fn config(&self) -> &IntegratorConfig {
        &self.config
    }

    /// Consolidated internal step method that handles all step variants.
    ///
    /// This method performs the core RKF45 integration with adaptive step control,
    /// optionally propagating the variational matrix (STM) and/or sensitivity matrix.
    fn step_internal(
        &self,
        t: f64,
        state: SVector<f64, S>,
        phi: Option<SMatrix<f64, S, S>>,
        sens: Option<SMatrix<f64, S, P>>,
        params: Option<&SVector<f64, P>>,
        dt: f64,
    ) -> AdaptiveStepInternalResultS<S, P> {
        let compute_phi = phi.is_some();
        let compute_sens = sens.is_some();

        let mut h = dt;
        let mut attempts = 0;

        loop {
            attempts += 1;
            if attempts > self.config.max_step_attempts {
                break;
            }

            // Compute RK stages
            let mut k = SMatrix::<f64, S, 6>::zeros();
            let mut k_phi = [SMatrix::<f64, S, S>::zeros(); 6];
            let mut k_sens = [SMatrix::<f64, S, P>::zeros(); 6];

            for i in 0..6 {
                let mut ksum = SVector::<f64, S>::zeros();
                let mut k_phi_sum = SMatrix::<f64, S, S>::zeros();
                let mut k_sens_sum = SMatrix::<f64, S, P>::zeros();

                for j in 0..i {
                    ksum += self.bt.a[(i, j)] * k.column(j);
                    if compute_phi {
                        k_phi_sum += self.bt.a[(i, j)] * k_phi[j];
                    }
                    if compute_sens {
                        k_sens_sum += self.bt.a[(i, j)] * k_sens[j];
                    }
                }

                let state_i = state + h * ksum;
                let t_i = t + self.bt.c[i] * h;
                let mut k_i = (self.f)(t_i, state_i);

                if let Some(ref ctrl) = self.control {
                    k_i += ctrl(t_i, state_i);
                }

                k.set_column(i, &k_i);

                if compute_phi || compute_sens {
                    let a_i = self
                        .varmat
                        .as_ref()
                        .expect("varmat required")
                        .compute(t_i, state_i);

                    if compute_phi {
                        k_phi[i] = a_i * (phi.unwrap() + h * k_phi_sum);
                    }

                    if compute_sens {
                        let b_i = self.sensmat.as_ref().expect("sensmat required").compute(
                            t_i,
                            &state_i,
                            params.unwrap(),
                        );
                        k_sens[i] = a_i * (sens.unwrap() + h * k_sens_sum) + b_i;
                    }
                }
            }

            // Compute solutions
            let mut state_high = SVector::<f64, S>::zeros();
            let mut state_low = SVector::<f64, S>::zeros();
            let mut phi_update = SMatrix::<f64, S, S>::zeros();
            let mut sens_update = SMatrix::<f64, S, P>::zeros();

            for i in 0..6 {
                state_high += h * self.bt.b_high[i] * k.column(i);
                state_low += h * self.bt.b_low[i] * k.column(i);
                if compute_phi {
                    phi_update += h * self.bt.b_high[i] * k_phi[i];
                }
                if compute_sens {
                    sens_update += h * self.bt.b_high[i] * k_sens[i];
                }
            }

            let state_high = state + state_high;
            let state_low = state + state_low;

            // Error estimation
            let error_vec = state_high - state_low;
            let error = compute_normalized_error_s(&error_vec, &state_high, &state, &self.config);

            let min_step_reached = self.config.min_step.is_some_and(|min| h.abs() <= min);

            if error <= 1.0 || min_step_reached {
                let dt_next = compute_next_step_size(error, h, 0.2, &self.config);

                return AdaptiveStepInternalResultS {
                    state: state_high,
                    phi: phi.map(|p| p + phi_update),
                    sens: sens.map(|s| s + sens_update),
                    dt_used: h,
                    error_estimate: error,
                    dt_next,
                };
            }

            // Step rejected - reduce step size
            h = compute_reduced_step_size(error, h, 0.25, &self.config);
        }

        panic!("RKF45S integrator exceeded maximum step attempts");
    }
}

impl<const S: usize, const P: usize> AdaptiveStepSIntegrator<S, P> for RKF45SIntegrator<S, P> {
    fn step(&self, t: f64, state: SVector<f64, S>, dt: f64) -> AdaptiveStepSResult<S> {
        let result = self.step_internal(t, state, None, None, None, dt);
        AdaptiveStepSResult {
            state: result.state,
            dt_used: result.dt_used,
            error_estimate: result.error_estimate,
            dt_next: result.dt_next,
        }
    }

    fn step_with_varmat(
        &self,
        t: f64,
        state: SVector<f64, S>,
        phi: SMatrix<f64, S, S>,
        dt: f64,
    ) -> (SVector<f64, S>, SMatrix<f64, S, S>, f64, f64, f64) {
        let result = self.step_internal(t, state, Some(phi), None, None, dt);
        (
            result.state,
            result.phi.unwrap(),
            result.dt_used,
            result.error_estimate,
            result.dt_next,
        )
    }

    fn step_with_sensmat(
        &self,
        t: f64,
        state: SVector<f64, S>,
        sens: SMatrix<f64, S, P>,
        params: &SVector<f64, P>,
        dt: f64,
    ) -> (SVector<f64, S>, SMatrix<f64, S, P>, f64, f64, f64) {
        let result = self.step_internal(t, state, None, Some(sens), Some(params), dt);
        (
            result.state,
            result.sens.unwrap(),
            result.dt_used,
            result.error_estimate,
            result.dt_next,
        )
    }

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
    ) {
        let result = self.step_internal(t, state, Some(phi), Some(sens), Some(params), dt);
        (
            result.state,
            result.phi.unwrap(),
            result.sens.unwrap(),
            result.dt_used,
            result.error_estimate,
            result.dt_next,
        )
    }
}

// ============================================================================
// Dynamic (runtime-sized) RKF45 Integrator
// ============================================================================

/// Runge-Kutta-Fehlberg 4(5) adaptive integrator with runtime-sized state vectors.
///
/// This is the dynamic-sized counterpart to `RKF45SIntegrator<S>`. Embedded 5th/4th order
/// method with automatic step size control for runtime-determined state dimensions.
///
/// # Example
///
/// ```
/// use nalgebra::DVector;
/// use brahe::integrators::{RKF45DIntegrator, AdaptiveStepDIntegrator, IntegratorConfig};
///
/// let f = |t: f64, state: DVector<f64>, _: Option<&DVector<f64>>| -> DVector<f64> {
///     DVector::from_vec(vec![2.0 * t])
/// };
///
/// let config = IntegratorConfig::adaptive(1e-8, 1e-6);
/// let rkf45 = RKF45DIntegrator::with_config(1, Box::new(f), None, None, None, config);
///
/// let state = DVector::from_vec(vec![0.0]);
/// let result = rkf45.step(0.0, state, 0.5);
/// ```
pub struct RKF45DIntegrator {
    dimension: usize,
    f: StateDynamicsD,
    varmat: VariationalMatrixD,
    sensmat: SensitivityD,
    control: ControlInputD,
    bt: EmbeddedButcherTableau<6>,
    config: IntegratorConfig,
}

impl RKF45DIntegrator {
    /// Create a new RKF45 integrator with default configuration.
    pub fn new(
        dimension: usize,
        f: StateDynamicsD,
        varmat: VariationalMatrixD,
        sensmat: SensitivityD,
        control: ControlInputD,
    ) -> Self {
        <Self as DIntegrator>::new(dimension, f, varmat, sensmat, control)
    }

    /// Create a new RKF45 integrator with custom configuration.
    pub fn with_config(
        dimension: usize,
        f: StateDynamicsD,
        varmat: VariationalMatrixD,
        sensmat: SensitivityD,
        control: ControlInputD,
        config: IntegratorConfig,
    ) -> Self {
        <Self as DIntegrator>::with_config(dimension, f, varmat, sensmat, control, config)
    }

    /// Get the state vector dimension for this integrator.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get a reference to the integrator configuration.
    pub fn config(&self) -> &IntegratorConfig {
        &self.config
    }
}

impl DIntegrator for RKF45DIntegrator {
    fn new(
        dimension: usize,
        f: StateDynamicsD,
        varmat: VariationalMatrixD,
        sensmat: SensitivityD,
        control: ControlInputD,
    ) -> Self {
        Self::with_config(
            dimension,
            f,
            varmat,
            sensmat,
            control,
            IntegratorConfig::default(),
        )
    }

    fn with_config(
        dimension: usize,
        f: StateDynamicsD,
        varmat: VariationalMatrixD,
        sensmat: SensitivityD,
        control: ControlInputD,
        config: IntegratorConfig,
    ) -> Self {
        Self {
            dimension,
            f,
            varmat,
            sensmat,
            control,
            bt: RKF45_TABLEAU,
            config,
        }
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn config(&self) -> &IntegratorConfig {
        &self.config
    }
}

impl RKF45DIntegrator {
    /// Consolidated internal step method that handles all step variants.
    ///
    /// This method performs the core RKF45 integration with adaptive step control,
    /// optionally propagating the variational matrix (STM) and/or sensitivity matrix.
    fn step_internal(
        &self,
        t: f64,
        state: DVector<f64>,
        phi: Option<DMatrix<f64>>,
        sens: Option<DMatrix<f64>>,
        params: Option<&DVector<f64>>,
        dt: f64,
    ) -> AdaptiveStepInternalResultD {
        // Validate dimensions
        assert_eq!(
            state.len(),
            self.dimension,
            "State dimension {} doesn't match integrator dimension {}",
            state.len(),
            self.dimension
        );

        if let Some(ref p) = phi {
            assert_eq!(p.nrows(), self.dimension);
            assert_eq!(p.ncols(), self.dimension);
        }

        if let Some(ref s) = sens {
            assert_eq!(s.nrows(), self.dimension);
        }

        let compute_phi = phi.is_some();
        let compute_sens = sens.is_some();
        let num_params = sens.as_ref().map(|s| s.ncols()).unwrap_or(0);

        let mut h = dt;
        let mut attempts = 0;

        loop {
            attempts += 1;
            if attempts > self.config.max_step_attempts {
                break;
            }

            // Compute RK stages
            let mut k = DMatrix::<f64>::zeros(self.dimension, 6);
            let mut k_phi = if compute_phi {
                vec![DMatrix::<f64>::zeros(self.dimension, self.dimension); 6]
            } else {
                vec![]
            };
            let mut k_sens = if compute_sens {
                vec![DMatrix::<f64>::zeros(self.dimension, num_params); 6]
            } else {
                vec![]
            };

            for i in 0..6 {
                let mut ksum = DVector::<f64>::zeros(self.dimension);
                let mut k_phi_sum = if compute_phi {
                    DMatrix::<f64>::zeros(self.dimension, self.dimension)
                } else {
                    DMatrix::<f64>::zeros(0, 0)
                };
                let mut k_sens_sum = if compute_sens {
                    DMatrix::<f64>::zeros(self.dimension, num_params)
                } else {
                    DMatrix::<f64>::zeros(0, 0)
                };

                for j in 0..i {
                    ksum += self.bt.a[(i, j)] * k.column(j);
                    if compute_phi {
                        k_phi_sum += self.bt.a[(i, j)] * &k_phi[j];
                    }
                    if compute_sens {
                        k_sens_sum += self.bt.a[(i, j)] * &k_sens[j];
                    }
                }

                let state_i = &state + h * &ksum;
                let t_i = t + self.bt.c[i] * h;

                let mut k_i = if compute_sens {
                    (self.f)(t_i, state_i.clone(), params)
                } else {
                    (self.f)(t_i, state_i.clone(), None)
                };

                if let Some(ref ctrl) = self.control {
                    k_i += ctrl(t_i, state_i.clone());
                }

                k.set_column(i, &k_i);

                if compute_phi || compute_sens {
                    let a_i = self
                        .varmat
                        .as_ref()
                        .expect("varmat required")
                        .compute(t_i, state_i.clone());

                    if compute_phi {
                        k_phi[i] = &a_i * (phi.as_ref().unwrap() + h * k_phi_sum);
                    }

                    if compute_sens {
                        let b_i = self.sensmat.as_ref().expect("sensmat required").compute(
                            t_i,
                            &state_i,
                            params.unwrap(),
                        );
                        k_sens[i] = &a_i * (sens.as_ref().unwrap() + h * k_sens_sum) + b_i;
                    }
                }
            }

            // Compute solutions
            let mut state_high = DVector::<f64>::zeros(self.dimension);
            let mut state_low = DVector::<f64>::zeros(self.dimension);
            let mut phi_update = if compute_phi {
                DMatrix::<f64>::zeros(self.dimension, self.dimension)
            } else {
                DMatrix::<f64>::zeros(0, 0)
            };
            let mut sens_update = if compute_sens {
                DMatrix::<f64>::zeros(self.dimension, num_params)
            } else {
                DMatrix::<f64>::zeros(0, 0)
            };

            for i in 0..6 {
                state_high += h * self.bt.b_high[i] * k.column(i);
                state_low += h * self.bt.b_low[i] * k.column(i);
                if compute_phi {
                    phi_update += h * self.bt.b_high[i] * &k_phi[i];
                }
                if compute_sens {
                    sens_update += h * self.bt.b_high[i] * &k_sens[i];
                }
            }

            let state_high = &state + state_high;
            let state_low = &state + state_low;

            // Error estimation
            let error_vec = &state_high - &state_low;
            let error = compute_normalized_error(&error_vec, &state_high, &state, &self.config);

            let min_step_reached = self.config.min_step.is_some_and(|min| h.abs() <= min);

            if error <= 1.0 || min_step_reached {
                let dt_next = compute_next_step_size(error, h, 0.2, &self.config);

                return AdaptiveStepInternalResultD {
                    state: state_high,
                    phi: phi.map(|p| p + phi_update),
                    sens: sens.map(|s| s + sens_update),
                    dt_used: h,
                    error_estimate: error,
                    dt_next,
                };
            }

            // Step rejected - reduce step size
            h = compute_reduced_step_size(error, h, 0.25, &self.config);
        }

        panic!("RKF45D integrator exceeded maximum step attempts");
    }
}

impl AdaptiveStepDIntegrator for RKF45DIntegrator {
    fn step(&self, t: f64, state: DVector<f64>, dt: f64) -> AdaptiveStepDResult {
        let result = self.step_internal(t, state, None, None, None, dt);
        AdaptiveStepDResult {
            state: result.state,
            dt_used: result.dt_used,
            error_estimate: result.error_estimate,
            dt_next: result.dt_next,
        }
    }

    fn step_with_varmat(
        &self,
        t: f64,
        state: DVector<f64>,
        phi: DMatrix<f64>,
        dt: f64,
    ) -> (DVector<f64>, DMatrix<f64>, f64, f64, f64) {
        let result = self.step_internal(t, state, Some(phi), None, None, dt);
        (
            result.state,
            result.phi.unwrap(),
            result.dt_used,
            result.error_estimate,
            result.dt_next,
        )
    }

    fn step_with_sensmat(
        &self,
        t: f64,
        state: DVector<f64>,
        sens: DMatrix<f64>,
        params: &DVector<f64>,
        dt: f64,
    ) -> (DVector<f64>, DMatrix<f64>, f64, f64, f64) {
        let result = self.step_internal(t, state, None, Some(sens), Some(params), dt);
        (
            result.state,
            result.sens.unwrap(),
            result.dt_used,
            result.error_estimate,
            result.dt_next,
        )
    }

    fn step_with_varmat_sensmat(
        &self,
        t: f64,
        state: DVector<f64>,
        phi: DMatrix<f64>,
        sens: DMatrix<f64>,
        params: &DVector<f64>,
        dt: f64,
    ) -> (DVector<f64>, DMatrix<f64>, DMatrix<f64>, f64, f64, f64) {
        let result = self.step_internal(t, state, Some(phi), Some(sens), Some(params), dt);
        (
            result.state,
            result.phi.unwrap(),
            result.sens.unwrap(),
            result.dt_used,
            result.error_estimate,
            result.dt_next,
        )
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::{DMatrix, DVector, SMatrix, SVector};

    use crate::constants::{DEGREES, RADIANS};
    use crate::integrators::IntegratorConfig;
    use crate::integrators::rkf45::{RKF45DIntegrator, RKF45SIntegrator};
    use crate::integrators::traits::{AdaptiveStepDIntegrator, AdaptiveStepSIntegrator};
    use crate::math::jacobian::{DNumericalJacobian, DifferenceMethod, SNumericalJacobian};
    use crate::time::{Epoch, TimeSystem};
    use crate::utils::testing::setup_global_test_eop;
    use crate::{GM_EARTH, R_EARTH, orbital_period, state_osculating_to_cartesian};

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
    fn test_rkf45s_integrator_parabola() {
        // Test RKF45 on simple parabola x' = 2t
        let f = |t: f64, _: SVector<f64, 1>| -> SVector<f64, 1> { SVector::<f64, 1>::new(2.0 * t) };

        let config = IntegratorConfig::adaptive(1e-10, 1e-8);
        let rkf45: RKF45SIntegrator<1, 0> =
            RKF45SIntegrator::with_config(Box::new(f), None, None, None, config);

        let mut t = 0.0;
        let mut state = SVector::<f64, 1>::new(0.0);

        while t < 1.0 {
            let dt = f64::min(1.0 - t, 0.1);
            let result = rkf45.step(t, state, dt);
            state = result.state;
            t += result.dt_used;
        }

        // At t=1.0, x should be 1.0 (integral of 2t from 0 to 1)
        assert_abs_diff_eq!(state[0], 1.0, epsilon = 1.0e-8);
    }

    #[test]
    fn test_rkf45s_integrator_adaptive() {
        // Test adaptive stepping on parabola
        let f = |t: f64, _: SVector<f64, 1>| -> SVector<f64, 1> { SVector::<f64, 1>::new(2.0 * t) };

        let config = IntegratorConfig::adaptive(1e-10, 1e-8);
        let rkf45: RKF45SIntegrator<1, 0> =
            RKF45SIntegrator::with_config(Box::new(f), None, None, None, config);

        let mut t = 0.0;
        let mut state = SVector::<f64, 1>::new(0.0);
        let t_end = 1.0;

        while t < t_end {
            let dt = f64::min(t_end - t, 0.1);
            let result = rkf45.step(t, state, dt);
            state = result.state;
            t += result.dt_used;

            // Verify that error estimate is reasonable
            assert!(result.error_estimate >= 0.0);
        }

        // Should still get accurate result
        assert_abs_diff_eq!(state[0], 1.0, epsilon = 1.0e-8);
    }

    #[test]
    fn test_rkf45s_integrator_orbit() {
        // Test RKF45 on orbital mechanics (more stringent than RK4)
        let config = IntegratorConfig::adaptive(1e-8, 1e-6);
        let rkf45: RKF45SIntegrator<6, 0> =
            RKF45SIntegrator::with_config(Box::new(point_earth), None, None, None, config);

        // Get start state
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 90.0, 0.0, 0.0, 0.0);
        let state0 = state_osculating_to_cartesian(oe0, RADIANS);
        let mut state = state0;

        // Get start and end times of propagation (1 orbit)
        let epc0 = Epoch::from_date(2024, 1, 1, TimeSystem::TAI);
        let epcf = epc0 + orbital_period(oe0[0]);
        let mut epc = epc0;

        while epc < epcf {
            let dt = (epcf - epc).min(10.0);
            let result = rkf45.step(epc - epc0, state, dt);
            state = result.state;
            epc += result.dt_used;
        }

        // RKF45 should achieve good accuracy
        assert_abs_diff_eq!(state.norm(), state0.norm(), epsilon = 1.0e-4);
        assert_abs_diff_eq!(state[0], state0[0], epsilon = 1.0e-3);
        assert_abs_diff_eq!(state[1], state0[1], epsilon = 1.0e-3);
        assert_abs_diff_eq!(state[2], state0[2], epsilon = 1.0e-3);
    }

    #[test]
    fn test_rkf45s_accuracy() {
        // Verify RKF45 achieves expected 5th order accuracy
        let f =
            |t: f64, _: SVector<f64, 1>| -> SVector<f64, 1> { SVector::<f64, 1>::new(3.0 * t * t) };

        let config = IntegratorConfig::adaptive(1e-8, 1e-6);
        let rkf45: RKF45SIntegrator<1, 0> =
            RKF45SIntegrator::with_config(Box::new(f), None, None, None, config);

        let mut t = 0.0;
        let mut state = SVector::<f64, 1>::new(0.0);

        while t < 10.0 {
            let dt = f64::min(10.0 - t, 0.1);
            let result = rkf45.step(t, state, dt);
            state = result.state;
            t += result.dt_used;
        }

        let exact = 1000.0; // t^3 at t=10
        let error = (state[0] - exact).abs();

        // 5th order method should be very accurate
        assert!(error < 1.0e-5);
    }

    #[test]
    fn test_rkf45s_step_size_increases() {
        // Verify that adaptive stepping increases step size when error is small
        let f = |t: f64, _: SVector<f64, 1>| -> SVector<f64, 1> { SVector::<f64, 1>::new(2.0 * t) };

        let config = IntegratorConfig::adaptive(1e-6, 1e-4);
        let rkf45: RKF45SIntegrator<1, 0> =
            RKF45SIntegrator::with_config(Box::new(f), None, None, None, config);

        let state = SVector::<f64, 1>::new(0.0);
        let dt_initial = 0.01;

        // Take a step with loose tolerance - error should be small
        let result = rkf45.step(0.0, state, dt_initial);

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
    fn test_rkf45s_step_size_decreases() {
        // Verify that adaptive stepping decreases step size when error is large
        let f = |_t: f64, state: SVector<f64, 1>| -> SVector<f64, 1> {
            // Stiff problem: y' = -1000 * y
            SVector::<f64, 1>::new(-1000.0 * state[0])
        };

        let config = IntegratorConfig::adaptive(1e-10, 1e-8);
        let rkf45: RKF45SIntegrator<1, 0> =
            RKF45SIntegrator::with_config(Box::new(f), None, None, None, config);

        let state = SVector::<f64, 1>::new(1.0);
        let dt_initial = 0.1; // Too large for this stiff problem

        // This should trigger step rejection and reduction
        let result = rkf45.step(0.0, state, dt_initial);

        // Step should have been reduced from initial
        assert!(result.dt_used <= dt_initial);
    }

    #[test]
    fn test_rkf45s_config_parameters() {
        // Verify that config parameters are actually used
        let f = |t: f64, _: SVector<f64, 1>| -> SVector<f64, 1> { SVector::<f64, 1>::new(2.0 * t) };

        let mut config = IntegratorConfig::adaptive(1e-8, 1e-6);
        config.step_safety_factor = Some(0.5); // Very conservative
        config.max_step_scale_factor = Some(2.0); // Limit growth

        let rkf45: RKF45SIntegrator<1, 0> =
            RKF45SIntegrator::with_config(Box::new(f), None, None, None, config);

        let state = SVector::<f64, 1>::new(0.0);
        let result = rkf45.step(0.0, state, 0.01);

        // With safety factor 0.5, growth should be limited
        assert!(result.dt_next <= 2.0 * result.dt_used);
    }

    #[test]
    fn test_rkf45s_no_limits() {
        // Verify that setting limits to None removes protections
        let f = |t: f64, _: SVector<f64, 1>| -> SVector<f64, 1> { SVector::<f64, 1>::new(2.0 * t) };

        let mut config = IntegratorConfig::adaptive(1e-8, 1e-6);
        config.min_step = None; // No minimum
        config.max_step = None; // No maximum
        config.min_step_scale_factor = None; // No limit on reduction
        config.max_step_scale_factor = None; // No limit on growth

        let rkf45: RKF45SIntegrator<1, 0> =
            RKF45SIntegrator::with_config(Box::new(f), None, None, None, config);

        let state = SVector::<f64, 1>::new(0.0);
        let result = rkf45.step(0.0, state, 0.001);

        // With no max_step_scale_factor, step can grow beyond typical 10x limit
        // (though actual growth depends on error)
        assert!(result.dt_next > 0.0);
    }

    #[test]
    fn test_rkf45s_stm_accuracy() {
        setup_global_test_eop();

        // Set up variational matrix computation with central differences
        let jacobian = SNumericalJacobian::new(Box::new(point_earth))
            .with_method(DifferenceMethod::Central)
            .with_fixed_offset(0.1);

        let config = IntegratorConfig::adaptive(1e-12, 1e-10);
        let rkf45: RKF45SIntegrator<6, 0> = RKF45SIntegrator::with_config(
            Box::new(point_earth),
            Some(Box::new(jacobian)),
            None,
            None,
            config,
        );

        // Circular orbit
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0);
        let state0 = state_osculating_to_cartesian(oe0, DEGREES);

        // Propagate single step
        let dt = 10.0; // 10 seconds
        let (state_new, phi, _dt_used, _error, _dt_next) =
            rkf45.step_with_varmat(0.0, state0, SMatrix::<f64, 6, 6>::identity(), dt);

        // Test STM accuracy by comparing with direct perturbation
        for i in 0..6 {
            let mut perturbation = SVector::<f64, 6>::zeros();
            perturbation[i] = 10.0; // 10m or 10mm/s perturbation

            // Propagate perturbed state
            let state0_pert = state0 + perturbation;
            let result_pert = rkf45.step(0.0, state0_pert, dt);

            // Predict perturbed state using STM
            let state_pert_predicted = state_new + phi * perturbation;

            // Compare each component
            for j in 0..6 {
                let direct = result_pert.state[j];
                let predicted = state_pert_predicted[j];
                let error = (direct - predicted).abs();
                let relative_error = error / perturbation[i].abs();

                println!(
                    "Component {} perturbation {}: error = {:.6e}, relative = {:.6e}",
                    j, i, error, relative_error
                );

                // RKF45 is 5th order, expect very good STM accuracy
                // Position components (0-2): ~1e-6 relative error
                // Velocity components (3-5): ~1e-5 relative error
                if j < 3 {
                    assert!(
                        relative_error < 1e-5,
                        "Position component {} failed: relative error = {:.3e}",
                        j,
                        relative_error
                    );
                } else {
                    assert!(
                        relative_error < 1e-4,
                        "Velocity component {} failed: relative error = {:.3e}",
                        j,
                        relative_error
                    );
                }
            }
        }
    }

    #[test]
    fn test_rkf45s_stm_vs_direct_perturbation() {
        setup_global_test_eop();

        let jacobian = SNumericalJacobian::new(Box::new(point_earth))
            .with_method(DifferenceMethod::Central)
            .with_fixed_offset(0.1);

        let config = IntegratorConfig::adaptive(1e-12, 1e-10);
        let rkf45: RKF45SIntegrator<6, 0> = RKF45SIntegrator::with_config(
            Box::new(point_earth),
            Some(Box::new(jacobian)),
            None,
            None,
            config,
        );

        // Circular orbit
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0);
        let state0 = state_osculating_to_cartesian(oe0, DEGREES);

        // Small perturbation in position
        let perturbation = SVector::<f64, 6>::new(10.0, 0.0, 0.0, 0.0, 0.0, 0.0);

        // Propagate over multiple steps
        let total_time = 100.0; // 100 seconds
        let num_steps = 10;
        let dt = total_time / num_steps as f64;

        let mut state = state0;
        let mut phi = SMatrix::<f64, 6, 6>::identity();
        let mut state_pert = state0 + perturbation;
        let mut t = 0.0;

        for step in 0..num_steps {
            // Propagate with STM
            let (state_new, phi_new, dt_used, _, _) = rkf45.step_with_varmat(t, state, phi, dt);

            // Propagate perturbed state directly
            let result_pert = rkf45.step(t, state_pert, dt);

            // Predict perturbed state using STM
            let state_pert_predicted = state_new + phi_new * perturbation;

            // Compare
            let error = (result_pert.state - state_pert_predicted).norm();
            println!("Step {}: error = {:.6e} m", step + 1, error);

            // Error should remain small and not accumulate excessively
            // RKF45 STM has ~1e-5 relative error, so for 10m perturbation
            // expect ~0.0001m error per step, growing to ~0.001m over 10 steps
            let max_error = 0.001 * (step + 1) as f64;
            assert!(
                error < max_error,
                "STM prediction diverged at step {}: error = {:.3e} m (max: {:.3e} m)",
                step + 1,
                error,
                max_error
            );

            // Update for next step
            state = state_new;
            phi = phi_new;
            state_pert = result_pert.state;
            t += dt_used;
        }
    }

    // ========================================================================
    // Dynamic RKF45 Tests
    // ========================================================================

    fn point_earth_dynamic(
        _: f64,
        x: DVector<f64>,
        _params: Option<&DVector<f64>>,
    ) -> DVector<f64> {
        assert_eq!(x.len(), 6, "State must be 6D for orbital mechanics");

        let r = x.rows(0, 3);
        let v = x.rows(3, 3);

        let r_norm = r.norm();
        let a = -GM_EARTH / r_norm.powi(3);

        let mut x_dot = DVector::<f64>::zeros(6);
        x_dot.rows_mut(0, 3).copy_from(&v);
        x_dot.rows_mut(3, 3).copy_from(&(a * r));

        x_dot
    }

    // Wrapper for Jacobian computation which expects a 2-argument function
    fn point_earth_dynamic_for_jacobian(t: f64, x: DVector<f64>) -> DVector<f64> {
        point_earth_dynamic(t, x, None)
    }

    #[test]
    fn test_rkf45d_integrator_parabola() {
        let f = |t: f64, _: DVector<f64>, _: Option<&DVector<f64>>| -> DVector<f64> {
            DVector::from_vec(vec![2.0 * t])
        };

        let config = IntegratorConfig::adaptive(1e-10, 1e-8);
        let rkf45 = RKF45DIntegrator::with_config(1, Box::new(f), None, None, None, config);

        let mut t = 0.0;
        let mut state = DVector::from_vec(vec![0.0]);

        while t < 1.0 {
            let dt = f64::min(1.0 - t, 0.1);
            let result = rkf45.step(t, state, dt);
            state = result.state;
            t += result.dt_used;
        }

        assert_abs_diff_eq!(state[0], 1.0, epsilon = 1.0e-8);
    }

    #[test]
    fn test_rkf45d_integrator_adaptive() {
        let f = |t: f64, _: DVector<f64>, _: Option<&DVector<f64>>| -> DVector<f64> {
            DVector::from_vec(vec![2.0 * t])
        };

        let config = IntegratorConfig::adaptive(1e-10, 1e-8);
        let rkf45 = RKF45DIntegrator::with_config(1, Box::new(f), None, None, None, config);

        let mut t = 0.0;
        let mut state = DVector::from_vec(vec![0.0]);
        let t_end = 1.0;

        while t < t_end {
            let dt = f64::min(t_end - t, 0.1);
            let result = rkf45.step(t, state, dt);
            state = result.state;
            t += result.dt_used;

            assert!(result.error_estimate >= 0.0);
        }

        assert_abs_diff_eq!(state[0], 1.0, epsilon = 1.0e-8);
    }

    #[test]
    fn test_rkf45d_integrator_orbit() {
        // Setup integrator
        let config = IntegratorConfig::adaptive(1e-8, 1e-6);
        let rkf45 = RKF45DIntegrator::with_config(
            6,
            Box::new(point_earth_dynamic),
            None,
            None,
            None,
            config,
        );

        // Setup initial state
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 90.0, 0.0, 0.0, 0.0);
        let state0_static = state_osculating_to_cartesian(oe0, RADIANS);
        let state0 = DVector::from_vec(state0_static.as_slice().to_vec());

        // Propagate for one orbital period
        let mut state = state0.clone();
        let epc0 = Epoch::from_date(2024, 1, 1, TimeSystem::TAI);
        let epcf = epc0 + orbital_period(oe0[0]);
        let mut epc = epc0;

        while epc < epcf {
            let dt = (epcf - epc).min(10.0);
            let result = rkf45.step(epc - epc0, state, dt);
            state = result.state;
            epc += result.dt_used;
        }

        // Verify energy conservation and position closure
        assert_abs_diff_eq!(state.norm(), state0.norm(), epsilon = 1.0e-4);
        assert_abs_diff_eq!(state[0], state0[0], epsilon = 1.0e-3);
        assert_abs_diff_eq!(state[1], state0[1], epsilon = 1.0e-3);
        assert_abs_diff_eq!(state[2], state0[2], epsilon = 1.0e-3);
    }

    #[test]
    fn test_rkf45d_accuracy() {
        let f = |t: f64, _: DVector<f64>, _: Option<&DVector<f64>>| -> DVector<f64> {
            DVector::from_vec(vec![3.0 * t * t])
        };

        let config = IntegratorConfig::adaptive(1e-8, 1e-6);
        let rkf45 = RKF45DIntegrator::with_config(1, Box::new(f), None, None, None, config);

        let mut t = 0.0;
        let mut state = DVector::from_vec(vec![0.0]);

        while t < 10.0 {
            let dt = f64::min(10.0 - t, 0.1);
            let result = rkf45.step(t, state, dt);
            state = result.state;
            t += result.dt_used;
        }

        let exact = 1000.0;
        let error = (state[0] - exact).abs();

        assert!(error < 1.0e-5);
    }

    #[test]
    fn test_rkf45d_step_size_increases() {
        let f = |t: f64, _: DVector<f64>, _: Option<&DVector<f64>>| -> DVector<f64> {
            DVector::from_vec(vec![2.0 * t])
        };

        let config = IntegratorConfig::adaptive(1e-6, 1e-4);
        let rkf45 = RKF45DIntegrator::with_config(1, Box::new(f), None, None, None, config);

        let state = DVector::from_vec(vec![0.0]);
        let dt_initial = 0.01;

        let result = rkf45.step(0.0, state, dt_initial);

        assert!(
            result.dt_next > dt_initial,
            "Expected dt_next ({}) > dt_initial ({})",
            result.dt_next,
            dt_initial
        );

        assert!(result.error_estimate < 0.1);
    }

    #[test]
    fn test_rkf45d_step_size_decreases() {
        let f = |_t: f64, state: DVector<f64>, _: Option<&DVector<f64>>| -> DVector<f64> {
            DVector::from_vec(vec![-1000.0 * state[0]])
        };

        let config = IntegratorConfig::adaptive(1e-10, 1e-8);
        let rkf45 = RKF45DIntegrator::with_config(1, Box::new(f), None, None, None, config);

        let state = DVector::from_vec(vec![1.0]);
        let dt_initial = 0.1;

        let result = rkf45.step(0.0, state, dt_initial);

        assert!(result.dt_used <= dt_initial);
    }

    #[test]
    fn test_rkf45d_config_parameters() {
        // Setup with custom configuration
        let f = |t: f64, _: DVector<f64>, _: Option<&DVector<f64>>| -> DVector<f64> {
            DVector::from_vec(vec![2.0 * t])
        };
        let mut config = IntegratorConfig::adaptive(1e-8, 1e-6);
        config.step_safety_factor = Some(0.5);
        config.max_step_scale_factor = Some(2.0);

        let rkf45 = RKF45DIntegrator::with_config(1, Box::new(f), None, None, None, config);
        let state = DVector::from_vec(vec![0.0]);

        // Take step
        let result = rkf45.step(0.0, state, 0.01);

        // Verify config parameters limit step size growth
        assert!(result.dt_next <= 2.0 * result.dt_used);
    }

    #[test]
    fn test_rkf45d_no_limits() {
        // Setup with all limits disabled
        let f = |t: f64, _: DVector<f64>, _: Option<&DVector<f64>>| -> DVector<f64> {
            DVector::from_vec(vec![2.0 * t])
        };
        let mut config = IntegratorConfig::adaptive(1e-8, 1e-6);
        config.min_step = None;
        config.max_step = None;
        config.min_step_scale_factor = None;
        config.max_step_scale_factor = None;

        let rkf45 = RKF45DIntegrator::with_config(1, Box::new(f), None, None, None, config);
        let state = DVector::from_vec(vec![0.0]);

        // Take step
        let result = rkf45.step(0.0, state, 0.001);

        // Verify step succeeds without limits
        assert!(result.dt_next > 0.0);
    }

    #[test]
    fn test_rkf45d_stm_accuracy() {
        setup_global_test_eop();

        // Setup integrator with variational matrix
        let jacobian = DNumericalJacobian::new(Box::new(point_earth_dynamic_for_jacobian))
            .with_method(DifferenceMethod::Central)
            .with_fixed_offset(0.1);
        let config = IntegratorConfig::adaptive(1e-12, 1e-10);
        let rkf45 = RKF45DIntegrator::with_config(
            6,
            Box::new(point_earth_dynamic),
            Some(Box::new(jacobian)),
            None,
            None,
            config,
        );

        // Setup circular orbit
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0);
        let state0_static = state_osculating_to_cartesian(oe0, DEGREES);
        let state0 = DVector::from_vec(state0_static.as_slice().to_vec());

        // Propagate with STM
        let dt = 10.0;
        let (state_new, phi, _dt_used, _error, _dt_next) =
            rkf45.step_with_varmat(0.0, state0.clone(), DMatrix::<f64>::identity(6, 6), dt);

        // Test STM accuracy by comparing with direct perturbation
        for i in 0..6 {
            let mut perturbation = DVector::<f64>::zeros(6);
            perturbation[i] = 10.0;

            // Propagate perturbed state
            let state0_pert = &state0 + &perturbation;
            let result_pert = rkf45.step(0.0, state0_pert, dt);

            // Predict perturbed state using STM
            let state_pert_predicted = &state_new + &phi * &perturbation;

            for j in 0..6 {
                let direct = result_pert.state[j];
                let predicted = state_pert_predicted[j];
                let error = (direct - predicted).abs();
                let relative_error = error / perturbation[i].abs();

                println!(
                    "Component {} perturbation {}: error = {:.6e}, relative = {:.6e}",
                    j, i, error, relative_error
                );

                if j < 3 {
                    assert!(
                        relative_error < 1e-5,
                        "Position component {} failed: relative error = {:.3e}",
                        j,
                        relative_error
                    );
                } else {
                    assert!(
                        relative_error < 1e-4,
                        "Velocity component {} failed: relative error = {:.3e}",
                        j,
                        relative_error
                    );
                }
            }
        }
    }

    #[test]
    fn test_rkf45d_stm_vs_direct_perturbation() {
        setup_global_test_eop();

        // Setup variational matrix computation
        let jacobian = DNumericalJacobian::new(Box::new(point_earth_dynamic_for_jacobian))
            .with_method(DifferenceMethod::Central)
            .with_fixed_offset(0.1);
        let config = IntegratorConfig::adaptive(1e-12, 1e-10);
        let rkf45 = RKF45DIntegrator::with_config(
            6,
            Box::new(point_earth_dynamic),
            Some(Box::new(jacobian)),
            None,
            None,
            config,
        );

        // Setup initial state and perturbation
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0);
        let state0_static = state_osculating_to_cartesian(oe0, DEGREES);
        let state0 = DVector::from_vec(state0_static.as_slice().to_vec());
        let perturbation = DVector::from_vec(vec![10.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // Setup multi-step propagation
        let total_time = 100.0;
        let num_steps = 10;
        let dt = total_time / num_steps as f64;

        let mut state = state0.clone();
        let mut phi = DMatrix::<f64>::identity(6, 6);
        let mut state_pert = &state0 + &perturbation;
        let mut t = 0.0;

        // Propagate both trajectories and verify STM prediction at each step
        for step in 0..num_steps {
            // Propagate nominal state with STM
            let (state_new, phi_new, dt_used, _, _) =
                rkf45.step_with_varmat(t, state.clone(), phi.clone(), dt);

            // Propagate perturbed state directly
            let result_pert = rkf45.step(t, state_pert.clone(), dt);

            // Predict perturbed state using STM
            let state_pert_predicted = &state_new + &phi_new * &perturbation;

            // Verify STM prediction accuracy (error grows linearly with time)
            let error = (&result_pert.state - &state_pert_predicted).norm();
            println!("Step {}: error = {:.6e} m", step + 1, error);

            let max_error = 0.001 * (step + 1) as f64;
            assert!(
                error < max_error,
                "STM prediction diverged at step {}: error = {:.3e} m (max: {:.3e} m)",
                step + 1,
                error,
                max_error
            );

            // Update for next step
            state = state_new;
            phi = phi_new;
            state_pert = result_pert.state;
            t += dt_used;
        }
    }

    #[test]
    fn test_rkf45_s_vs_d_consistency() {
        // Verify RKF45SIntegrator and RKF45DIntegrator produce identical results
        let f_static = |_t: f64, x: SVector<f64, 2>| -> SVector<f64, 2> {
            SVector::<f64, 2>::new(x[1], -x[0])
        };
        let f_dynamic = |_t: f64, x: DVector<f64>, _: Option<&DVector<f64>>| -> DVector<f64> {
            DVector::from_vec(vec![x[1], -x[0]])
        };

        let config = IntegratorConfig::adaptive(1e-10, 1e-8);
        let rkf45_s: RKF45SIntegrator<2, 0> =
            RKF45SIntegrator::with_config(Box::new(f_static), None, None, None, config.clone());
        let rkf45_d =
            RKF45DIntegrator::with_config(2, Box::new(f_dynamic), None, None, None, config);

        let state_s = SVector::<f64, 2>::new(1.0, 0.0);
        let state_d = DVector::from_vec(vec![1.0, 0.0]);
        let dt = 0.1;

        let result_s = rkf45_s.step(0.0, state_s, dt);
        let result_d = rkf45_d.step(0.0, state_d, dt);

        // State results should be identical to machine precision
        assert_abs_diff_eq!(result_s.state[0], result_d.state[0], epsilon = 1.0e-15);
        assert_abs_diff_eq!(result_s.state[1], result_d.state[1], epsilon = 1.0e-15);

        // Error estimates and step suggestions should also match
        assert_abs_diff_eq!(
            result_s.error_estimate,
            result_d.error_estimate,
            epsilon = 1.0e-15
        );
        assert_abs_diff_eq!(result_s.dt_used, result_d.dt_used, epsilon = 1.0e-15);
        assert_abs_diff_eq!(result_s.dt_next, result_d.dt_next, epsilon = 1.0e-15);
    }

    #[test]
    fn test_rkf45s_backward_integration() {
        // Test backward propagation with orbital mechanics
        let config = IntegratorConfig::adaptive(1e-10, 1e-8);
        let rkf45: RKF45SIntegrator<6, 0> =
            RKF45SIntegrator::with_config(Box::new(point_earth), None, None, None, config);

        // Setup initial state
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 90.0, 0.0, 0.0, 0.0);
        let state0 = state_osculating_to_cartesian(oe0, DEGREES);

        // Propagate forward for 100 seconds
        let dt_forward = 10.0;
        let mut state_fwd = state0;
        let mut t = 0.0;
        while t < 100.0 {
            let result = rkf45.step(t, state_fwd, dt_forward);
            state_fwd = result.state;
            t += result.dt_used;
        }
        let t_max = t;

        // Now propagate backward from the final state
        let mut state_back = state_fwd;
        let mut t = t_max;
        let mut dt_back: f64 = -10.0; // Initial negative timestep for backward integration
        while t > 0.0 {
            // Ensure we don't step past t=0
            dt_back = dt_back.max(-t);
            let result = rkf45.step(t, state_back, dt_back);
            state_back = result.state;
            t += result.dt_used;
            dt_back = result.dt_next; // Use adaptive timestep suggestion
        }

        // Should return close to initial state
        for i in 0..6 {
            assert_abs_diff_eq!(state_back[i], state0[i], epsilon = 1.0e-3);
        }
    }

    #[test]
    fn test_rkf45d_backward_integration() {
        // Test backward propagation with orbital mechanics (dynamic variant)
        let config = IntegratorConfig::adaptive(1e-10, 1e-8);
        let rkf45 = RKF45DIntegrator::with_config(
            6,
            Box::new(point_earth_dynamic),
            None,
            None,
            None,
            config,
        );

        // Setup initial state
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 90.0, 0.0, 0.0, 0.0);
        let state0_static = state_osculating_to_cartesian(oe0, DEGREES);
        let state0 = DVector::from_vec(state0_static.as_slice().to_vec());

        // Propagate forward for 100 seconds
        let dt_forward = 10.0;
        let mut state_fwd = state0.clone();
        let mut t = 0.0;
        while t < 100.0 {
            let result = rkf45.step(t, state_fwd, dt_forward);
            state_fwd = result.state;
            t += result.dt_used;
        }
        let t_max = t;

        // Now propagate backward from the final state
        let mut state_back = state_fwd;
        let mut t = t_max;
        let mut dt_back: f64 = -10.0; // Initial negative timestep for backward integration
        while t > 0.0 {
            // Ensure we don't step past t=0
            dt_back = dt_back.max(-t);
            let result = rkf45.step(t, state_back, dt_back);
            state_back = result.state;
            t += result.dt_used;
            dt_back = result.dt_next; // Use adaptive timestep suggestion
        }

        // Should return close to initial state
        for i in 0..6 {
            assert_abs_diff_eq!(state_back[i], state0[i], epsilon = 1.0e-3);
        }
    }

    #[test]
    fn test_rkf45d_varmat_sensmat() {
        // Test step_with_varmat_sensmat using simple exponential decay: dx/dt = -k*x
        // where k is a parameter. This has analytical solutions for both STM and sensitivity.
        //
        // For dx/dt = -k*x:
        // - State solution: x(t) = x0 * exp(-k*t)
        // - STM: Φ(t) = exp(-k*t) (since ∂f/∂x = -k)
        // - Sensitivity: S(t) = -x0 * t * exp(-k*t) (since ∂f/∂k = -x)

        use crate::math::sensitivity::DSensitivityProvider;

        // Dynamics: dx/dt = -k*x where k = params[0] if provided, else k=1.0
        let dynamics =
            |_t: f64, state: DVector<f64>, params: Option<&DVector<f64>>| -> DVector<f64> {
                let k = params.map_or(1.0, |p| p[0]);
                DVector::from_vec(vec![-k * state[0]])
            };

        // Jacobian provider: ∂f/∂x = -k
        struct DecayJacobian;
        impl crate::math::jacobian::DJacobianProvider for DecayJacobian {
            fn compute(&self, _t: f64, _state: DVector<f64>) -> DMatrix<f64> {
                // For simplicity, use k=1.0 for the Jacobian (this is approximate but works for testing)
                // In a real application, you'd pass k through or use numerical differentiation
                DMatrix::from_vec(1, 1, vec![-1.0])
            }
        }

        // Sensitivity provider: ∂f/∂k = -x
        struct DecaySensitivity;
        impl DSensitivityProvider for DecaySensitivity {
            fn compute(
                &self,
                _t: f64,
                state: &DVector<f64>,
                _params: &DVector<f64>,
            ) -> DMatrix<f64> {
                DMatrix::from_vec(1, 1, vec![-state[0]])
            }
        }

        let config = IntegratorConfig::adaptive(1e-12, 1e-10);
        let rkf45 = RKF45DIntegrator::with_config(
            1,
            Box::new(dynamics),
            Some(Box::new(DecayJacobian)),
            Some(Box::new(DecaySensitivity)),
            None,
            config.clone(),
        );

        // Initial conditions
        let x0 = 1.0;
        let state0 = DVector::from_vec(vec![x0]);
        let phi0 = DMatrix::identity(1, 1);
        let sens0 = DMatrix::zeros(1, 1); // Initial sensitivity is zero
        let params = DVector::from_vec(vec![1.0]); // k = 1.0
        let dt = 0.1;

        // Take a step with combined method
        let (state_combined, phi_combined, sens_combined, dt_used, _, _) = rkf45
            .step_with_varmat_sensmat(
                0.0,
                state0.clone(),
                phi0.clone(),
                sens0.clone(),
                &params,
                dt,
            );

        // Create separate integrator for comparison (to avoid cache issues)
        let rkf45_sensmat = RKF45DIntegrator::with_config(
            1,
            Box::new(dynamics),
            Some(Box::new(DecayJacobian)),
            Some(Box::new(DecaySensitivity)),
            None,
            config,
        );

        // Test 1: Compare with step_with_sensmat - states and sensitivity should match
        // Both use params, so the dynamics are identical
        let (state_sensmat, sens_sensmat, dt_sensmat, _, _) =
            rkf45_sensmat.step_with_sensmat(0.0, state0.clone(), sens0.clone(), &params, dt);

        // The dt_used should be the same since both methods have identical dynamics
        assert_abs_diff_eq!(dt_used, dt_sensmat, epsilon = 1e-14);
        assert_abs_diff_eq!(state_combined[0], state_sensmat[0], epsilon = 1e-14);
        assert_abs_diff_eq!(sens_combined[(0, 0)], sens_sensmat[(0, 0)], epsilon = 1e-14);

        // Test 2: Verify STM evolution is correct by comparing with numerical differentiation
        // Create a fresh integrator to test STM accuracy via direct perturbation
        let rkf45_pert = RKF45DIntegrator::with_config(
            1,
            Box::new(dynamics),
            None,
            None,
            None,
            IntegratorConfig::adaptive(1e-12, 1e-10),
        );

        // Perturb initial state
        let delta = 1e-6;
        let state0_pert = DVector::from_vec(vec![x0 + delta]);
        let result_pert = rkf45_pert.step(0.0, state0_pert, dt);

        // STM should predict the perturbed state
        let state_pert_predicted = state_combined[0] + phi_combined[(0, 0)] * delta;
        let relative_error = (result_pert.state[0] - state_pert_predicted).abs() / delta;

        // STM prediction should be accurate
        assert!(
            relative_error < 1e-4,
            "STM prediction error: {}",
            relative_error
        );

        // Test 3: Check against analytical solution
        // x(t) = x0 * exp(-k*t)
        let t = dt_used;
        let k = params[0];
        let x_analytical = x0 * (-k * t).exp();
        let phi_analytical = (-k * t).exp();
        // S(t) = -x0 * t * exp(-k*t) for zero initial sensitivity
        let sens_analytical = -x0 * t * (-k * t).exp();

        // State should match analytical solution
        assert_abs_diff_eq!(state_combined[0], x_analytical, epsilon = 1e-8);

        // STM should match (approximately due to using k=1.0 in Jacobian)
        assert_abs_diff_eq!(phi_combined[(0, 0)], phi_analytical, epsilon = 1e-6);

        // Sensitivity should match analytical solution
        assert_abs_diff_eq!(sens_combined[(0, 0)], sens_analytical, epsilon = 1e-6);

        // Test 4: Multiple steps accumulation
        let mut state = state0.clone();
        let mut phi = phi0;
        let mut sens = sens0;
        let mut t = 0.0;

        // Create fresh integrator for multi-step test
        let rkf45_multi = RKF45DIntegrator::with_config(
            1,
            Box::new(dynamics),
            Some(Box::new(DecayJacobian)),
            Some(Box::new(DecaySensitivity)),
            None,
            IntegratorConfig::adaptive(1e-12, 1e-10),
        );

        for _ in 0..10 {
            let (new_state, new_phi, new_sens, dt_used, _, _) =
                rkf45_multi.step_with_varmat_sensmat(t, state, phi, sens, &params, 0.1);
            state = new_state;
            phi = new_phi;
            sens = new_sens;
            t += dt_used;
        }

        // After 1 second with k=1: x should be ~e^(-1) = 0.368
        let x_expected = x0 * (-k * t).exp();
        assert_abs_diff_eq!(state[0], x_expected, epsilon = 1e-6);

        // STM should be ~e^(-1)
        assert_abs_diff_eq!(phi[(0, 0)], (-k * t).exp(), epsilon = 1e-4);
    }

    // ========================================================================
    // Static Integrator Sensitivity Tests
    // ========================================================================

    #[test]
    fn test_rkf45s_sensmat() {
        // Test sensitivity matrix propagation using exponential decay: dx/dt = -k*x

        use crate::math::jacobian::SJacobianProvider;
        use crate::math::sensitivity::SSensitivityProvider;

        // Jacobian provider: ∂f/∂x = -k (using k=1.0)
        struct DecayJacobian;
        impl SJacobianProvider<1> for DecayJacobian {
            fn compute(&self, _t: f64, _state: SVector<f64, 1>) -> SMatrix<f64, 1, 1> {
                SMatrix::<f64, 1, 1>::new(-1.0)
            }
        }

        // Sensitivity provider: ∂f/∂k = -x
        struct DecaySensitivity;
        impl SSensitivityProvider<1, 1> for DecaySensitivity {
            fn compute(
                &self,
                _t: f64,
                state: &SVector<f64, 1>,
                _params: &SVector<f64, 1>,
            ) -> SMatrix<f64, 1, 1> {
                SMatrix::<f64, 1, 1>::new(-state[0])
            }
        }

        // Dynamics: dx/dt = -k*x where k = 1.0
        let f = |_t: f64, x: SVector<f64, 1>| -> SVector<f64, 1> { -x };

        let config = IntegratorConfig::adaptive(1e-12, 1e-10);
        let rkf45: RKF45SIntegrator<1, 1> = RKF45SIntegrator::with_config(
            Box::new(f),
            Some(Box::new(DecayJacobian)),
            Some(Box::new(DecaySensitivity)),
            None,
            config,
        );

        // Initial conditions
        let x0 = 1.0;
        let state0 = SVector::<f64, 1>::new(x0);
        let sens0 = SMatrix::<f64, 1, 1>::zeros();
        let params = SVector::<f64, 1>::new(1.0);

        // Propagate for 1 second
        let mut state = state0;
        let mut sens = sens0;
        let mut t = 0.0_f64;

        while t < 1.0 {
            let dt = (1.0_f64 - t).min(0.1);
            let (new_state, new_sens, dt_used, _, _) =
                rkf45.step_with_sensmat(t, state, sens, &params, dt);
            state = new_state;
            sens = new_sens;
            t += dt_used;
        }

        // Check against analytical solution
        let k = params[0];
        let x_analytical = x0 * (-k * t).exp();
        let sens_analytical = -x0 * t * (-k * t).exp();

        assert_abs_diff_eq!(state[0], x_analytical, epsilon = 1e-6);
        assert_abs_diff_eq!(sens[(0, 0)], sens_analytical, epsilon = 1e-4);
    }

    #[test]
    fn test_rkf45d_sensmat() {
        // Test sensitivity matrix propagation (standalone dynamic version)

        use crate::math::sensitivity::DSensitivityProvider;

        struct DecayJacobian;
        impl crate::math::jacobian::DJacobianProvider for DecayJacobian {
            fn compute(&self, _t: f64, _state: DVector<f64>) -> DMatrix<f64> {
                DMatrix::from_vec(1, 1, vec![-1.0])
            }
        }

        struct DecaySensitivity;
        impl DSensitivityProvider for DecaySensitivity {
            fn compute(
                &self,
                _t: f64,
                state: &DVector<f64>,
                _params: &DVector<f64>,
            ) -> DMatrix<f64> {
                DMatrix::from_vec(1, 1, vec![-state[0]])
            }
        }

        let dynamics =
            |_t: f64, x: DVector<f64>, _params: Option<&DVector<f64>>| -> DVector<f64> { -x };

        let config = IntegratorConfig::adaptive(1e-12, 1e-10);
        let rkf45 = RKF45DIntegrator::with_config(
            1,
            Box::new(dynamics),
            Some(Box::new(DecayJacobian)),
            Some(Box::new(DecaySensitivity)),
            None,
            config,
        );

        let x0 = 1.0;
        let state0 = DVector::from_vec(vec![x0]);
        let sens0 = DMatrix::zeros(1, 1);
        let params = DVector::from_vec(vec![1.0]);

        let mut state = state0;
        let mut sens = sens0;
        let mut t = 0.0_f64;

        while t < 1.0 {
            let dt = (1.0_f64 - t).min(0.1);
            let (new_state, new_sens, dt_used, _, _) =
                rkf45.step_with_sensmat(t, state, sens, &params, dt);
            state = new_state;
            sens = new_sens;
            t += dt_used;
        }

        let k = params[0];
        let x_analytical = x0 * (-k * t).exp();
        let sens_analytical = -x0 * t * (-k * t).exp();

        assert_abs_diff_eq!(state[0], x_analytical, epsilon = 1e-6);
        assert_abs_diff_eq!(sens[(0, 0)], sens_analytical, epsilon = 1e-4);
    }

    #[test]
    fn test_rkf45s_varmat_sensmat() {
        // Test combined STM and sensitivity matrix propagation (static version)

        use crate::math::jacobian::SJacobianProvider;
        use crate::math::sensitivity::SSensitivityProvider;

        struct DecayJacobian;
        impl SJacobianProvider<1> for DecayJacobian {
            fn compute(&self, _t: f64, _state: SVector<f64, 1>) -> SMatrix<f64, 1, 1> {
                SMatrix::<f64, 1, 1>::new(-1.0)
            }
        }

        struct DecaySensitivity;
        impl SSensitivityProvider<1, 1> for DecaySensitivity {
            fn compute(
                &self,
                _t: f64,
                state: &SVector<f64, 1>,
                _params: &SVector<f64, 1>,
            ) -> SMatrix<f64, 1, 1> {
                SMatrix::<f64, 1, 1>::new(-state[0])
            }
        }

        let f = |_t: f64, x: SVector<f64, 1>| -> SVector<f64, 1> { -x };

        let config = IntegratorConfig::adaptive(1e-12, 1e-10);
        let rkf45: RKF45SIntegrator<1, 1> = RKF45SIntegrator::with_config(
            Box::new(f),
            Some(Box::new(DecayJacobian)),
            Some(Box::new(DecaySensitivity)),
            None,
            config,
        );

        let x0 = 1.0;
        let state0 = SVector::<f64, 1>::new(x0);
        let phi0 = SMatrix::<f64, 1, 1>::identity();
        let sens0 = SMatrix::<f64, 1, 1>::zeros();
        let params = SVector::<f64, 1>::new(1.0);

        let mut state = state0;
        let mut phi = phi0;
        let mut sens = sens0;
        let mut t = 0.0_f64;

        while t < 1.0 {
            let dt = (1.0_f64 - t).min(0.1);
            let (new_state, new_phi, new_sens, dt_used, _, _) =
                rkf45.step_with_varmat_sensmat(t, state, phi, sens, &params, dt);
            state = new_state;
            phi = new_phi;
            sens = new_sens;
            t += dt_used;
        }

        let k = params[0];
        let x_analytical = x0 * (-k * t).exp();
        let phi_analytical = (-k * t).exp();
        let sens_analytical = -x0 * t * (-k * t).exp();

        assert_abs_diff_eq!(state[0], x_analytical, epsilon = 1e-6);
        assert_abs_diff_eq!(phi[(0, 0)], phi_analytical, epsilon = 1e-4);
        assert_abs_diff_eq!(sens[(0, 0)], sens_analytical, epsilon = 1e-4);

        // Verify STM accuracy via direct perturbation
        let delta = 1e-6;
        let state0_pert = SVector::<f64, 1>::new(x0 + delta);
        let rkf45_pert: RKF45SIntegrator<1, 1> = RKF45SIntegrator::with_config(
            Box::new(f),
            None,
            None,
            None,
            IntegratorConfig::adaptive(1e-12, 1e-10),
        );

        let mut state_pert = state0_pert;
        let mut t_pert = 0.0_f64;
        while t_pert < 1.0 {
            let dt = (1.0_f64 - t_pert).min(0.1);
            let result = rkf45_pert.step(t_pert, state_pert, dt);
            state_pert = result.state;
            t_pert += result.dt_used;
        }

        let state_pert_predicted = state[0] + phi[(0, 0)] * delta;
        let relative_error = (state_pert[0] - state_pert_predicted).abs() / delta;
        assert!(
            relative_error < 1e-3,
            "STM prediction error: {}",
            relative_error
        );
    }

    // =============================================================================
    // Constructor and Config Tests
    // =============================================================================

    #[test]
    fn test_rkf45s_new_uses_default_config() {
        fn dynamics(_t: f64, state: SVector<f64, 1>) -> SVector<f64, 1> {
            state
        }

        let integrator: RKF45SIntegrator<1, 0> =
            RKF45SIntegrator::new(Box::new(dynamics), None, None, None);
        let config = integrator.config();

        let default_config = IntegratorConfig::default();
        assert_eq!(config.abs_tol, default_config.abs_tol);
        assert_eq!(config.rel_tol, default_config.rel_tol);
        assert_eq!(config.max_step_attempts, default_config.max_step_attempts);
        assert_eq!(config.min_step, default_config.min_step);
        assert_eq!(config.max_step, default_config.max_step);
    }

    #[test]
    fn test_rkf45s_with_config_stores_config() {
        fn dynamics(_t: f64, state: SVector<f64, 1>) -> SVector<f64, 1> {
            state
        }

        let custom_config = IntegratorConfig {
            abs_tol: 1e-10,
            rel_tol: 1e-8,
            initial_step: None,
            max_step_attempts: 20,
            min_step: Some(1e-15),
            max_step: Some(100.0),
            step_safety_factor: Some(0.9),
            max_step_scale_factor: Some(5.0),
            min_step_scale_factor: Some(0.1),
            fixed_step_size: None,
        };

        let integrator: RKF45SIntegrator<1, 0> = RKF45SIntegrator::with_config(
            Box::new(dynamics),
            None,
            None,
            None,
            custom_config.clone(),
        );
        let config = integrator.config();

        assert_eq!(config.abs_tol, 1e-10);
        assert_eq!(config.rel_tol, 1e-8);
        assert_eq!(config.max_step_attempts, 20);
        assert_eq!(config.min_step, Some(1e-15));
        assert_eq!(config.max_step, Some(100.0));
    }

    #[test]
    fn test_rkf45s_config_returns_reference() {
        fn dynamics(_t: f64, state: SVector<f64, 1>) -> SVector<f64, 1> {
            state
        }

        let integrator: RKF45SIntegrator<1, 0> =
            RKF45SIntegrator::new(Box::new(dynamics), None, None, None);

        let config1 = integrator.config();
        let config2 = integrator.config();

        assert_eq!(config1.abs_tol, config2.abs_tol);
        assert_eq!(config1.rel_tol, config2.rel_tol);
        assert_eq!(config1.max_step_attempts, config2.max_step_attempts);
    }

    #[test]
    fn test_rkf45d_new_uses_default_config() {
        fn dynamics(_t: f64, state: DVector<f64>, _params: Option<&DVector<f64>>) -> DVector<f64> {
            state
        }

        let integrator = RKF45DIntegrator::new(1, Box::new(dynamics), None, None, None);
        let config = integrator.config();

        let default_config = IntegratorConfig::default();
        assert_eq!(config.abs_tol, default_config.abs_tol);
        assert_eq!(config.rel_tol, default_config.rel_tol);
        assert_eq!(config.max_step_attempts, default_config.max_step_attempts);
        assert_eq!(config.min_step, default_config.min_step);
        assert_eq!(config.max_step, default_config.max_step);
    }

    #[test]
    fn test_rkf45d_with_config_stores_config() {
        fn dynamics(_t: f64, state: DVector<f64>, _params: Option<&DVector<f64>>) -> DVector<f64> {
            state
        }

        let custom_config = IntegratorConfig {
            abs_tol: 1e-10,
            rel_tol: 1e-8,
            initial_step: None,
            max_step_attempts: 20,
            min_step: Some(1e-15),
            max_step: Some(100.0),
            step_safety_factor: Some(0.9),
            max_step_scale_factor: Some(5.0),
            min_step_scale_factor: Some(0.1),
            fixed_step_size: None,
        };

        let integrator = RKF45DIntegrator::with_config(
            1,
            Box::new(dynamics),
            None,
            None,
            None,
            custom_config.clone(),
        );
        let config = integrator.config();

        assert_eq!(config.abs_tol, 1e-10);
        assert_eq!(config.rel_tol, 1e-8);
        assert_eq!(config.max_step_attempts, 20);
        assert_eq!(config.min_step, Some(1e-15));
        assert_eq!(config.max_step, Some(100.0));
    }

    #[test]
    fn test_rkf45d_config_returns_reference() {
        fn dynamics(_t: f64, state: DVector<f64>, _params: Option<&DVector<f64>>) -> DVector<f64> {
            state
        }

        let integrator = RKF45DIntegrator::new(1, Box::new(dynamics), None, None, None);

        let config1 = integrator.config();
        let config2 = integrator.config();

        assert_eq!(config1.abs_tol, config2.abs_tol);
        assert_eq!(config1.rel_tol, config2.rel_tol);
        assert_eq!(config1.max_step_attempts, config2.max_step_attempts);
    }

    #[test]
    fn test_rkf45d_dimension_method() {
        fn dynamics(_t: f64, state: DVector<f64>, _params: Option<&DVector<f64>>) -> DVector<f64> {
            state
        }

        let integrator = RKF45DIntegrator::new(6, Box::new(dynamics), None, None, None);
        assert_eq!(integrator.dimension(), 6);

        let integrator2 = RKF45DIntegrator::new(12, Box::new(dynamics), None, None, None);
        assert_eq!(integrator2.dimension(), 12);
    }

    // =============================================================================
    // Panic Tests - Max Step Attempts Exceeded
    // =============================================================================

    #[test]
    #[should_panic(expected = "exceeded maximum step attempts")]
    fn test_rkf45s_panics_on_max_attempts_exceeded() {
        // Create a "stiff" problem that will fail to converge
        // Use very tight tolerances and small max_step_attempts
        fn stiff_dynamics(_t: f64, state: SVector<f64, 1>) -> SVector<f64, 1> {
            // Rapidly varying dynamics that will produce large errors
            SVector::<f64, 1>::new(1e10 * state[0])
        }

        let config = IntegratorConfig {
            abs_tol: 1e-15,
            rel_tol: 1e-15,
            initial_step: None,
            max_step_attempts: 1, // Only allow 1 attempt
            min_step: None,       // No minimum step floor
            max_step: None,
            step_safety_factor: None,
            max_step_scale_factor: None,
            min_step_scale_factor: None,
            fixed_step_size: None,
        };

        let integrator: RKF45SIntegrator<1, 0> =
            RKF45SIntegrator::with_config(Box::new(stiff_dynamics), None, None, None, config);

        let state = SVector::<f64, 1>::new(1.0);
        // This should panic because the error will be too large and we only allow 1 attempt
        let _ = integrator.step(0.0, state, 1.0);
    }

    #[test]
    #[should_panic(expected = "exceeded maximum step attempts")]
    fn test_rkf45d_panics_on_max_attempts_exceeded() {
        fn stiff_dynamics(
            _t: f64,
            state: DVector<f64>,
            _params: Option<&DVector<f64>>,
        ) -> DVector<f64> {
            DVector::from_vec(vec![1e10 * state[0]])
        }

        let config = IntegratorConfig {
            abs_tol: 1e-15,
            rel_tol: 1e-15,
            initial_step: None,
            max_step_attempts: 1,
            min_step: None,
            max_step: None,
            step_safety_factor: None,
            max_step_scale_factor: None,
            min_step_scale_factor: None,
            fixed_step_size: None,
        };

        let integrator =
            RKF45DIntegrator::with_config(1, Box::new(stiff_dynamics), None, None, None, config);

        let state = DVector::from_vec(vec![1.0]);
        let _ = integrator.step(0.0, state, 1.0);
    }
}
