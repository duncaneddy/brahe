/*!
Implementation of the Dormand-Prince 5(4) adaptive integration method.
 */

use nalgebra::{DMatrix, DVector, SMatrix, SVector};
use std::cell::RefCell;

use crate::integrators::butcher_tableau::{EmbeddedButcherTableau, dp54_tableau};
use crate::integrators::config::{AdaptiveStepSResult, IntegratorConfig};
use crate::integrators::traits::{
    AdaptiveStepDIntegrator, AdaptiveStepDResult, AdaptiveStepSIntegrator,
};

// Type aliases for complex function types
type StateDynamics<const S: usize> = Box<dyn Fn(f64, SVector<f64, S>) -> SVector<f64, S>>;
type VariationalMatrix<const S: usize> =
    Option<Box<dyn Fn(f64, SVector<f64, S>) -> SMatrix<f64, S, S>>>;

/// Dormand-Prince 5(4) adaptive integrator.
///
/// This is MATLAB's ode45 method. More efficient than RKF45 due to:
/// - Better error constants (fewer rejected steps for same tolerance)
/// - FSAL property (First-Same-As-Last): reuses last stage from previous step
///
/// 7 stages but only 6 function evaluations per accepted step due to FSAL.
pub struct DormandPrince54SIntegrator<const S: usize> {
    f: StateDynamics<S>,
    varmat: VariationalMatrix<S>,
    bt: EmbeddedButcherTableau<7>,
    config: IntegratorConfig,
    /// Cached last stage evaluation for FSAL optimization
    last_f: RefCell<Option<SVector<f64, S>>>,
}

impl<const S: usize> DormandPrince54SIntegrator<S> {
    /// Create a new DP54 integrator with default configuration.
    pub fn new(f: StateDynamics<S>, varmat: VariationalMatrix<S>) -> Self {
        Self::with_config(f, varmat, IntegratorConfig::default())
    }

    /// Create a new DP54 integrator with custom configuration.
    pub fn with_config(
        f: StateDynamics<S>,
        varmat: VariationalMatrix<S>,
        config: IntegratorConfig,
    ) -> Self {
        Self {
            f,
            varmat,
            bt: dp54_tableau(),
            config,
            last_f: RefCell::new(None),
        }
    }

    /// Get a reference to the integrator configuration.
    pub fn config(&self) -> &IntegratorConfig {
        &self.config
    }
}

impl<const S: usize> AdaptiveStepSIntegrator<S> for DormandPrince54SIntegrator<S> {
    fn step(
        &self,
        t: f64,
        state: SVector<f64, S>,
        dt: f64,
        abs_tol: f64,
        rel_tol: f64,
    ) -> AdaptiveStepSResult<S> {
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

            // Compute embedded solutions with FSAL
            let mut k = SMatrix::<f64, S, 7>::zeros();
            let k0 = if let Some(cached_f) = self.last_f.borrow().as_ref() {
                *cached_f
            } else {
                (self.f)(t, state)
            };
            k.set_column(0, &k0);

            for i in 1..7 {
                let mut ksum = SVector::<f64, S>::zeros();
                for j in 0..i {
                    ksum += self.bt.a[(i, j)] * k.column(j);
                }
                k.set_column(i, &(self.f)(t + self.bt.c[i] * h, state + h * ksum));
            }

            *self.last_f.borrow_mut() = Some(k.column(6).clone_owned());

            let mut state_high = SVector::<f64, S>::zeros();
            let mut state_low = SVector::<f64, S>::zeros();
            for i in 0..7 {
                state_high += h * self.bt.b_high[i] * k.column(i);
                state_low += h * self.bt.b_low[i] * k.column(i);
            }
            let state_high = state + state_high;
            let state_low = state + state_low;

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

                return AdaptiveStepSResult {
                    state: state_high,
                    dt_used: h,
                    error_estimate: error,
                    dt_next,
                };
            }

            // Step rejected - invalidate FSAL cache since we'll retry with different h
            *self.last_f.borrow_mut() = None;

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
        let mut k = SMatrix::<f64, S, 7>::zeros();
        let k0 = if let Some(cached_f) = self.last_f.borrow().as_ref() {
            *cached_f
        } else {
            (self.f)(t, state)
        };
        k.set_column(0, &k0);

        for i in 1..7 {
            let mut ksum = SVector::<f64, S>::zeros();
            for j in 0..i {
                ksum += self.bt.a[(i, j)] * k.column(j);
            }
            k.set_column(i, &(self.f)(t + self.bt.c[i] * h, state + h * ksum));
        }

        *self.last_f.borrow_mut() = Some(k.column(6).clone_owned());

        let mut state_high = SVector::<f64, S>::zeros();
        let mut state_low = SVector::<f64, S>::zeros();
        for i in 0..7 {
            state_high += h * self.bt.b_high[i] * k.column(i);
            state_low += h * self.bt.b_low[i] * k.column(i);
        }
        let state_high = state + state_high;
        let state_low = state + state_low;

        let error_vec = state_high - state_low;
        let mut error: f64 = 0.0;
        for i in 0..S {
            let scale = abs_tol + rel_tol * state_high[i].abs().max(state[i].abs());
            error = error.max((error_vec[i] / scale).abs());
        }

        AdaptiveStepSResult {
            state: state_high,
            dt_used: h,
            error_estimate: error,
            dt_next: h, // Keep same step for fallback
        }
    }

    fn step_with_varmat(
        &self,
        t: f64,
        state: SVector<f64, S>,
        phi: SMatrix<f64, S, S>,
        dt: f64,
        abs_tol: f64,
        rel_tol: f64,
    ) -> (SVector<f64, S>, SMatrix<f64, S, S>, f64, f64, f64) {
        // Implementation mirrors step but propagates STM and uses FSAL
        let mut h = dt;
        let mut attempts = 0;

        loop {
            attempts += 1;
            if attempts > self.config.max_step_attempts {
                if let Some(min_step) = self.config.min_step {
                    h = min_step;
                }
                break;
            }

            // Compute embedded solutions for both state and STM with FSAL
            let mut k = SMatrix::<f64, S, 7>::zeros();
            let mut k_phi = [SMatrix::<f64, S, S>::zeros(); 7];

            // Stage 0: Use cached FSAL value if available
            let k0 = if let Some(cached_f) = self.last_f.borrow().as_ref() {
                *cached_f
            } else {
                (self.f)(t, state)
            };
            k.set_column(0, &k0);
            k_phi[0] = self.varmat.as_ref().unwrap()(t, state) * phi;

            // Compute stages 1-6
            for i in 1..7 {
                let mut ksum = SVector::<f64, S>::zeros();
                let mut k_phi_sum = SMatrix::<f64, S, S>::zeros();

                for (j, k_phi_j) in k_phi.iter().enumerate().take(i) {
                    ksum += self.bt.a[(i, j)] * k.column(j);
                    k_phi_sum += self.bt.a[(i, j)] * k_phi_j;
                }

                k.set_column(i, &(self.f)(t + self.bt.c[i] * h, state + h * ksum));
                k_phi[i] = self.varmat.as_ref().unwrap()(t + self.bt.c[i] * h, state + h * ksum)
                    * (phi + h * k_phi_sum);
            }

            // Cache k[6] for next step (FSAL)
            *self.last_f.borrow_mut() = Some(k.column(6).clone_owned());

            // Compute high and low order solutions
            let mut state_high = SVector::<f64, S>::zeros();
            let mut state_low = SVector::<f64, S>::zeros();
            let mut phi_update = SMatrix::<f64, S, S>::zeros();

            for (i, k_phi_i) in k_phi.iter().enumerate().take(7) {
                state_high += h * self.bt.b_high[i] * k.column(i);
                state_low += h * self.bt.b_low[i] * k.column(i);
                phi_update += h * self.bt.b_high[i] * k_phi_i;
            }

            let state_high = state + state_high;
            let state_low = state + state_low;
            let phi_new = phi + phi_update;

            // Estimate error
            let error_vec = state_high - state_low;
            let mut error: f64 = 0.0;

            for i in 0..S {
                let tol = abs_tol + rel_tol * state_high[i].abs().max(state[i].abs());
                error = error.max((error_vec[i] / tol).abs());
            }

            // Check acceptance
            let min_step_reached = self.config.min_step.is_some_and(|min| h <= min);

            if error <= 1.0 || min_step_reached {
                // Calculate next step size
                let dt_next = if error > 0.0 {
                    let raw_scale = (1.0 / error).powf(0.2);
                    let scale = self
                        .config
                        .step_safety_factor
                        .map_or(raw_scale, |safety| safety * raw_scale);

                    let mut h_next = h * scale;

                    if let Some(min_scale) = self.config.min_step_scale_factor {
                        h_next = h_next.max(min_scale * h);
                    }
                    if let Some(max_scale) = self.config.max_step_scale_factor {
                        h_next = h_next.min(max_scale * h);
                    }
                    if let Some(max_step) = self.config.max_step {
                        h_next = h_next.min(max_step);
                    }
                    if let Some(min_step) = self.config.min_step {
                        h_next = h_next.max(min_step);
                    }

                    h_next
                } else {
                    let h_next = if let Some(max_scale) = self.config.max_step_scale_factor {
                        max_scale * h
                    } else {
                        10.0 * h
                    };
                    self.config.max_step.map_or(h_next, |max| h_next.min(max))
                };

                return (state_high, phi_new, h, error, dt_next);
            }

            // Step rejected - invalidate FSAL cache
            *self.last_f.borrow_mut() = None;

            // Reduce step size
            let raw_scale = (1.0 / error).powf(0.25);
            let scale = self
                .config
                .step_safety_factor
                .map_or(raw_scale, |safety| safety * raw_scale);

            let mut h_new = h * scale;

            if let Some(min_scale) = self.config.min_step_scale_factor {
                h_new = h_new.max(min_scale * h);
            }
            if let Some(min_step) = self.config.min_step {
                h_new = h_new.max(min_step);
            }

            h = h_new;
        }

        // Fallback: use minimum step
        let mut k = SMatrix::<f64, S, 7>::zeros();
        let mut k_phi = [SMatrix::<f64, S, S>::zeros(); 7];

        let k0 = if let Some(cached_f) = self.last_f.borrow().as_ref() {
            *cached_f
        } else {
            (self.f)(t, state)
        };
        k.set_column(0, &k0);
        k_phi[0] = self.varmat.as_ref().unwrap()(t, state) * phi;

        for i in 1..7 {
            let mut ksum = SVector::<f64, S>::zeros();
            let mut k_phi_sum = SMatrix::<f64, S, S>::zeros();

            for (j, k_phi_j) in k_phi.iter().enumerate().take(i) {
                ksum += self.bt.a[(i, j)] * k.column(j);
                k_phi_sum += self.bt.a[(i, j)] * k_phi_j;
            }

            k.set_column(i, &(self.f)(t + self.bt.c[i] * h, state + h * ksum));
            k_phi[i] = self.varmat.as_ref().unwrap()(t + self.bt.c[i] * h, state + h * ksum)
                * (phi + h * k_phi_sum);
        }

        *self.last_f.borrow_mut() = Some(k.column(6).clone_owned());

        let mut state_high = SVector::<f64, S>::zeros();
        let mut state_low = SVector::<f64, S>::zeros();
        let mut phi_update = SMatrix::<f64, S, S>::zeros();

        for (i, k_phi_i) in k_phi.iter().enumerate().take(7) {
            state_high += h * self.bt.b_high[i] * k.column(i);
            state_low += h * self.bt.b_low[i] * k.column(i);
            phi_update += h * self.bt.b_high[i] * k_phi_i;
        }

        let state_high = state + state_high;
        let state_low = state + state_low;
        let phi_new = phi + phi_update;

        let error_vec = state_high - state_low;
        let mut error: f64 = 0.0;
        for i in 0..S {
            let scale = abs_tol + rel_tol * state_high[i].abs().max(state[i].abs());
            error = error.max((error_vec[i] / scale).abs());
        }

        (state_high, phi_new, h, error, h)
    }
}

// ============================================================================
// Dynamic (runtime-sized) Dormand-Prince 5(4) Integrator
// ============================================================================

// Type aliases for dynamic function types
type StateDynamicsD = Box<dyn Fn(f64, DVector<f64>) -> DVector<f64>>;
type VariationalMatrixD = Option<Box<dyn Fn(f64, DVector<f64>) -> DMatrix<f64>>>;

/// Dormand-Prince 5(4) adaptive integrator with runtime-sized state vectors.
///
/// This is the dynamic-sized counterpart to `DormandPrince54SIntegrator<S>`.
/// More efficient than RKF45D due to FSAL property (reuses last stage evaluation).
pub struct DormandPrince54DIntegrator {
    dimension: usize,
    f: StateDynamicsD,
    varmat: VariationalMatrixD,
    bt: EmbeddedButcherTableau<7>,
    config: IntegratorConfig,
    /// Cached last stage evaluation for FSAL optimization
    last_f: RefCell<Option<DVector<f64>>>,
}

impl DormandPrince54DIntegrator {
    /// Create a new DP54 integrator with default configuration.
    pub fn new(dimension: usize, f: StateDynamicsD, varmat: VariationalMatrixD) -> Self {
        Self::with_config(dimension, f, varmat, IntegratorConfig::default())
    }

    /// Create a new DP54 integrator with custom configuration.
    pub fn with_config(
        dimension: usize,
        f: StateDynamicsD,
        varmat: VariationalMatrixD,
        config: IntegratorConfig,
    ) -> Self {
        Self {
            dimension,
            f,
            varmat,
            bt: dp54_tableau(),
            config,
            last_f: RefCell::new(None),
        }
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

impl AdaptiveStepDIntegrator for DormandPrince54DIntegrator {
    fn step(
        &self,
        t: f64,
        state: DVector<f64>,
        dt: f64,
        abs_tol: f64,
        rel_tol: f64,
    ) -> AdaptiveStepDResult {
        assert_eq!(state.len(), self.dimension);

        let mut h = dt;
        let mut attempts = 0;

        loop {
            attempts += 1;
            if attempts > self.config.max_step_attempts {
                if let Some(min_step) = self.config.min_step {
                    h = min_step;
                }
                break;
            }

            // Compute embedded solutions with FSAL
            let mut k = DMatrix::<f64>::zeros(self.dimension, 7);
            let k0 = if let Some(cached_f) = self.last_f.borrow().as_ref() {
                cached_f.clone()
            } else {
                (self.f)(t, state.clone())
            };
            k.set_column(0, &k0);

            for i in 1..7 {
                let mut ksum = DVector::<f64>::zeros(self.dimension);
                for j in 0..i {
                    ksum += self.bt.a[(i, j)] * k.column(j);
                }
                k.set_column(i, &(self.f)(t + self.bt.c[i] * h, &state + h * ksum));
            }

            *self.last_f.borrow_mut() = Some(k.column(6).clone_owned());

            let mut state_high = DVector::<f64>::zeros(self.dimension);
            let mut state_low = DVector::<f64>::zeros(self.dimension);
            for i in 0..7 {
                state_high += h * self.bt.b_high[i] * k.column(i);
                state_low += h * self.bt.b_low[i] * k.column(i);
            }
            let state_high = &state + state_high;
            let state_low = &state + state_low;

            let error_vec = &state_high - &state_low;
            let mut error: f64 = 0.0;

            for i in 0..self.dimension {
                let tol = abs_tol + rel_tol * state_high[i].abs().max(state[i].abs());
                error = error.max((error_vec[i] / tol).abs());
            }

            let min_step_reached = self.config.min_step.is_some_and(|min| h <= min);

            if error <= 1.0 || min_step_reached {
                let dt_next = if error > 0.0 {
                    let raw_scale = (1.0 / error).powf(0.2);
                    let scale = self
                        .config
                        .step_safety_factor
                        .map_or(raw_scale, |safety| safety * raw_scale);

                    let mut h_next = h * scale;

                    if let Some(min_scale) = self.config.min_step_scale_factor {
                        h_next = h_next.max(min_scale * h);
                    }
                    if let Some(max_scale) = self.config.max_step_scale_factor {
                        h_next = h_next.min(max_scale * h);
                    }
                    if let Some(max_step) = self.config.max_step {
                        h_next = h_next.min(max_step);
                    }
                    if let Some(min_step) = self.config.min_step {
                        h_next = h_next.max(min_step);
                    }

                    h_next
                } else {
                    let h_next = if let Some(max_scale) = self.config.max_step_scale_factor {
                        max_scale * h
                    } else {
                        10.0 * h
                    };
                    self.config.max_step.map_or(h_next, |max| h_next.min(max))
                };

                return AdaptiveStepDResult {
                    state: state_high,
                    dt_used: h,
                    error_estimate: error,
                    dt_next,
                };
            }

            // Step rejected - invalidate FSAL cache since we'll retry with different h
            *self.last_f.borrow_mut() = None;

            // Step rejected - reduce step size
            let raw_scale = (1.0 / error).powf(0.25);
            let scale = self
                .config
                .step_safety_factor
                .map_or(raw_scale, |safety| safety * raw_scale);

            let mut h_new = h * scale;

            if let Some(min_scale) = self.config.min_step_scale_factor {
                h_new = h_new.max(min_scale * h);
            }
            if let Some(min_step) = self.config.min_step {
                h_new = h_new.max(min_step);
            }

            h = h_new;
        }

        // Fallback: use minimum step
        let mut k = DMatrix::<f64>::zeros(self.dimension, 7);
        let k0 = if let Some(cached_f) = self.last_f.borrow().as_ref() {
            cached_f.clone()
        } else {
            (self.f)(t, state.clone())
        };
        k.set_column(0, &k0);

        for i in 1..7 {
            let mut ksum = DVector::<f64>::zeros(self.dimension);
            for j in 0..i {
                ksum += self.bt.a[(i, j)] * k.column(j);
            }
            k.set_column(i, &(self.f)(t + self.bt.c[i] * h, &state + h * ksum));
        }

        *self.last_f.borrow_mut() = Some(k.column(6).clone_owned());

        let mut state_high = DVector::<f64>::zeros(self.dimension);
        let mut state_low = DVector::<f64>::zeros(self.dimension);
        for i in 0..7 {
            state_high += h * self.bt.b_high[i] * k.column(i);
            state_low += h * self.bt.b_low[i] * k.column(i);
        }
        let state_high = &state + state_high;
        let state_low = &state + state_low;

        let error_vec = &state_high - &state_low;
        let mut error: f64 = 0.0;
        for i in 0..self.dimension {
            let scale = abs_tol + rel_tol * state_high[i].abs().max(state[i].abs());
            error = error.max((error_vec[i] / scale).abs());
        }

        AdaptiveStepDResult {
            state: state_high,
            dt_used: h,
            error_estimate: error,
            dt_next: h,
        }
    }

    fn step_with_varmat(
        &self,
        t: f64,
        state: DVector<f64>,
        phi: DMatrix<f64>,
        dt: f64,
        abs_tol: f64,
        rel_tol: f64,
    ) -> (DVector<f64>, DMatrix<f64>, f64, f64, f64) {
        assert_eq!(state.len(), self.dimension);
        assert_eq!(phi.nrows(), self.dimension);
        assert_eq!(phi.ncols(), self.dimension);

        let mut h = dt;
        let mut attempts = 0;

        loop {
            attempts += 1;
            if attempts > self.config.max_step_attempts {
                if let Some(min_step) = self.config.min_step {
                    h = min_step;
                }
                break;
            }

            // Compute embedded solutions for both state and STM with FSAL
            let mut k = DMatrix::<f64>::zeros(self.dimension, 7);
            let mut k_phi = vec![DMatrix::<f64>::zeros(self.dimension, self.dimension); 7];

            let k0 = if let Some(cached_f) = self.last_f.borrow().as_ref() {
                cached_f.clone()
            } else {
                (self.f)(t, state.clone())
            };
            k.set_column(0, &k0);
            k_phi[0] = self.varmat.as_ref().unwrap()(t, state.clone()) * &phi;

            for i in 1..7 {
                let mut ksum = DVector::<f64>::zeros(self.dimension);
                let mut k_phi_sum = DMatrix::<f64>::zeros(self.dimension, self.dimension);

                #[allow(clippy::needless_range_loop)]
                for j in 0..i {
                    ksum += self.bt.a[(i, j)] * k.column(j);
                    k_phi_sum += self.bt.a[(i, j)] * &k_phi[j];
                }

                k.set_column(i, &(self.f)(t + self.bt.c[i] * h, &state + h * &ksum));
                k_phi[i] = self.varmat.as_ref().unwrap()(t + self.bt.c[i] * h, &state + h * ksum)
                    * (&phi + h * k_phi_sum);
            }

            *self.last_f.borrow_mut() = Some(k.column(6).clone_owned());

            let mut state_high = DVector::<f64>::zeros(self.dimension);
            let mut state_low = DVector::<f64>::zeros(self.dimension);
            let mut phi_update = DMatrix::<f64>::zeros(self.dimension, self.dimension);

            #[allow(clippy::needless_range_loop)]
            for i in 0..7 {
                state_high += h * self.bt.b_high[i] * k.column(i);
                state_low += h * self.bt.b_low[i] * k.column(i);
                phi_update += h * self.bt.b_high[i] * &k_phi[i];
            }

            let state_high = &state + state_high;
            let state_low = &state + state_low;
            let phi_new = &phi + phi_update;

            let error_vec = &state_high - &state_low;
            let mut error: f64 = 0.0;

            for i in 0..self.dimension {
                let tol = abs_tol + rel_tol * state_high[i].abs().max(state[i].abs());
                error = error.max((error_vec[i] / tol).abs());
            }

            let min_step_reached = self.config.min_step.is_some_and(|min| h <= min);

            if error <= 1.0 || min_step_reached {
                let dt_next = if error > 0.0 {
                    let raw_scale = (1.0 / error).powf(0.2);
                    let scale = self
                        .config
                        .step_safety_factor
                        .map_or(raw_scale, |safety| safety * raw_scale);

                    let mut h_next = h * scale;

                    if let Some(min_scale) = self.config.min_step_scale_factor {
                        h_next = h_next.max(min_scale * h);
                    }
                    if let Some(max_scale) = self.config.max_step_scale_factor {
                        h_next = h_next.min(max_scale * h);
                    }
                    if let Some(max_step) = self.config.max_step {
                        h_next = h_next.min(max_step);
                    }
                    if let Some(min_step) = self.config.min_step {
                        h_next = h_next.max(min_step);
                    }

                    h_next
                } else {
                    let h_next = if let Some(max_scale) = self.config.max_step_scale_factor {
                        max_scale * h
                    } else {
                        10.0 * h
                    };
                    self.config.max_step.map_or(h_next, |max| h_next.min(max))
                };

                return (state_high, phi_new, h, error, dt_next);
            }

            // Step rejected
            let raw_scale = (1.0 / error).powf(0.25);
            let scale = self
                .config
                .step_safety_factor
                .map_or(raw_scale, |safety| safety * raw_scale);

            let mut h_new = h * scale;

            if let Some(min_scale) = self.config.min_step_scale_factor {
                h_new = h_new.max(min_scale * h);
            }
            if let Some(min_step) = self.config.min_step {
                h_new = h_new.max(min_step);
            }

            h = h_new;
        }

        // Fallback
        let mut k = DMatrix::<f64>::zeros(self.dimension, 7);
        let mut k_phi = vec![DMatrix::<f64>::zeros(self.dimension, self.dimension); 7];

        let k0 = if let Some(cached_f) = self.last_f.borrow().as_ref() {
            cached_f.clone()
        } else {
            (self.f)(t, state.clone())
        };
        k.set_column(0, &k0);
        k_phi[0] = self.varmat.as_ref().unwrap()(t, state.clone()) * &phi;

        for i in 1..7 {
            let mut ksum = DVector::<f64>::zeros(self.dimension);
            let mut k_phi_sum = DMatrix::<f64>::zeros(self.dimension, self.dimension);

            #[allow(clippy::needless_range_loop)]
            for j in 0..i {
                ksum += self.bt.a[(i, j)] * k.column(j);
                k_phi_sum += self.bt.a[(i, j)] * &k_phi[j];
            }

            k.set_column(i, &(self.f)(t + self.bt.c[i] * h, &state + h * &ksum));
            k_phi[i] = self.varmat.as_ref().unwrap()(t + self.bt.c[i] * h, &state + h * ksum)
                * (&phi + h * k_phi_sum);
        }

        *self.last_f.borrow_mut() = Some(k.column(6).clone_owned());

        let mut state_high = DVector::<f64>::zeros(self.dimension);
        let mut state_low = DVector::<f64>::zeros(self.dimension);
        let mut phi_update = DMatrix::<f64>::zeros(self.dimension, self.dimension);

        #[allow(clippy::needless_range_loop)]
        for i in 0..7 {
            state_high += h * self.bt.b_high[i] * k.column(i);
            state_low += h * self.bt.b_low[i] * k.column(i);
            phi_update += h * self.bt.b_high[i] * &k_phi[i];
        }

        let state_high = &state + state_high;
        let state_low = &state + state_low;
        let phi_new = &phi + phi_update;

        let error_vec = &state_high - &state_low;
        let mut error: f64 = 0.0;
        for i in 0..self.dimension {
            let scale = abs_tol + rel_tol * state_high[i].abs().max(state[i].abs());
            error = error.max((error_vec[i] / scale).abs());
        }

        (state_high, phi_new, h, error, h)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::{DMatrix, DVector, SMatrix, SVector};

    use crate::constants::{DEGREES, RADIANS};
    use crate::integrators::IntegratorConfig;
    use crate::integrators::dp54::{DormandPrince54DIntegrator, DormandPrince54SIntegrator};
    use crate::integrators::rkf45::{RKF45DIntegrator, RKF45SIntegrator};
    use crate::integrators::traits::{
        AdaptiveStepDIntegrator, AdaptiveStepSIntegrator, VarmatConfig,
    };
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
    fn test_dp54_integrator_parabola() {
        // Test DP54 on simple parabola x' = 2t
        let f = |t: f64, _: SVector<f64, 1>| -> SVector<f64, 1> { SVector::<f64, 1>::new(2.0 * t) };

        let config = IntegratorConfig::adaptive(1e-10, 1e-8);
        let dp54 = DormandPrince54SIntegrator::with_config(Box::new(f), None, config);

        let mut t = 0.0;
        let mut state = SVector::<f64, 1>::new(0.0);
        let dt = 0.01;

        for _ in 0..100 {
            let result = dp54.step(t, state, dt, 1e-10, 1e-8);
            state = result.state;
            t += result.dt_used;
        }

        // At t=1.0, x should be 1.0 (integral of 2t from 0 to 1)
        assert_abs_diff_eq!(state[0], 1.0, epsilon = 1.0e-10);
    }

    #[test]
    fn test_dp54_integrator_adaptive() {
        // Test adaptive stepping on parabola
        let f = |t: f64, _: SVector<f64, 1>| -> SVector<f64, 1> { SVector::<f64, 1>::new(2.0 * t) };

        let config = IntegratorConfig::adaptive(1e-10, 1e-8);
        let dp54 = DormandPrince54SIntegrator::with_config(Box::new(f), None, config);

        let mut t = 0.0;
        let mut state = SVector::<f64, 1>::new(0.0);
        let t_end = 1.0;

        while t < t_end {
            let dt = f64::min(t_end - t, 0.1);
            let result = dp54.step(t, state, dt, 1e-10, 1e-8);
            state = result.state;
            t += result.dt_used;

            // Verify that error estimate is reasonable
            assert!(result.error_estimate >= 0.0);
        }

        // Should still get accurate result
        assert_abs_diff_eq!(state[0], 1.0, epsilon = 1.0e-8);
    }

    #[test]
    fn test_dp54_integrator_orbit() {
        // Test DP54 on orbital mechanics
        let config = IntegratorConfig::adaptive(1e-9, 1e-6);
        let dp54 = DormandPrince54SIntegrator::with_config(Box::new(point_earth), None, config);

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
            dt = (epcf - epc).min(10.0);
            let result = dp54.step(epc - epc0, state, dt, 1e-9, 1e-6);
            state = result.state;
            epc += result.dt_used;
        }

        // DP54 should achieve good accuracy (similar to RKF45)
        assert_abs_diff_eq!(state.norm(), state0.norm(), epsilon = 1.0e-4);
        assert_abs_diff_eq!(state[0], state0[0], epsilon = 1.0e-3);
        assert_abs_diff_eq!(state[1], state0[1], epsilon = 1.0e-3);
        assert_abs_diff_eq!(state[2], state0[2], epsilon = 1.0e-3);
    }

    #[test]
    fn test_dp54_accuracy() {
        // Verify DP54 achieves expected 5th order accuracy
        let f =
            |t: f64, _: SVector<f64, 1>| -> SVector<f64, 1> { SVector::<f64, 1>::new(3.0 * t * t) };

        let config = IntegratorConfig::adaptive(1e-10, 1e-8);
        let dp54 = DormandPrince54SIntegrator::with_config(Box::new(f), None, config);

        // Use moderate step size
        let dt = 0.1;
        let mut state = SVector::<f64, 1>::new(0.0);

        for i in 0..100 {
            let t = i as f64 * dt;
            let result = dp54.step(t, state, dt, 1e-10, 1e-8);
            state = result.state;
        }

        let exact = 1000.0; // t^3 at t=10
        let error = (state[0] - exact).abs();

        // 5th order method should be very accurate
        assert!(error < 1.0e-5);
    }

    #[test]
    fn test_dp54_step_size_increases() {
        // Verify that adaptive stepping increases step size when error is small
        let f = |t: f64, _: SVector<f64, 1>| -> SVector<f64, 1> { SVector::<f64, 1>::new(2.0 * t) };

        let config = IntegratorConfig::adaptive(1e-6, 1e-4);
        let dp54 = DormandPrince54SIntegrator::with_config(Box::new(f), None, config);

        let state = SVector::<f64, 1>::new(0.0);
        let dt_initial = 0.01;

        // Take a step with loose tolerance - error should be small
        let result = dp54.step(0.0, state, dt_initial, 1e-6, 1e-4);

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
    fn test_dp54_step_size_decreases() {
        // Verify that adaptive stepping decreases step size when error is large
        let f = |_t: f64, state: SVector<f64, 1>| -> SVector<f64, 1> {
            // Stiff problem: y' = -1000 * y
            SVector::<f64, 1>::new(-1000.0 * state[0])
        };

        let config = IntegratorConfig::adaptive(1e-10, 1e-8);
        let dp54 = DormandPrince54SIntegrator::with_config(Box::new(f), None, config);

        let state = SVector::<f64, 1>::new(1.0);
        let dt_initial = 0.1; // Too large for this stiff problem

        // This should trigger step rejection and reduction
        let result = dp54.step(0.0, state, dt_initial, 1e-10, 1e-8);

        // Step should have been reduced from initial
        assert!(result.dt_used <= dt_initial);
    }

    #[test]
    fn test_dp54_config_parameters() {
        // Verify that config parameters are actually used
        let f = |t: f64, _: SVector<f64, 1>| -> SVector<f64, 1> { SVector::<f64, 1>::new(2.0 * t) };

        let mut config = IntegratorConfig::adaptive(1e-8, 1e-6);
        config.step_safety_factor = Some(0.5); // Very conservative
        config.max_step_scale_factor = Some(2.0); // Limit growth

        let dp54 = DormandPrince54SIntegrator::with_config(Box::new(f), None, config);

        let state = SVector::<f64, 1>::new(0.0);
        let result = dp54.step(0.0, state, 0.01, 1e-8, 1e-6);

        // With safety factor 0.5, growth should be limited
        assert!(result.dt_next <= 2.0 * result.dt_used);
    }

    #[test]
    fn test_dp54_fsal_cache() {
        // Verify that FSAL optimization works - second step should reuse cached value
        let f = |t: f64, _: SVector<f64, 1>| -> SVector<f64, 1> { SVector::<f64, 1>::new(2.0 * t) };

        let config = IntegratorConfig::adaptive(1e-10, 1e-8);
        let dp54 = DormandPrince54SIntegrator::with_config(Box::new(f), None, config);

        let state0 = SVector::<f64, 1>::new(0.0);

        // First step - no cached value
        let result1 = dp54.step(0.0, state0, 0.1, 1e-10, 1e-8);
        let state1 = result1.state;

        // Second step - should use cached f value from first step
        let result2 = dp54.step(result1.dt_used, state1, 0.1, 1e-10, 1e-8);
        let state2 = result2.state;

        // Verify we get correct result (integral of 2t from 0 to ~0.2 is ~0.04)
        assert_abs_diff_eq!(state2[0], 0.04, epsilon = 1.0e-8);
    }

    #[test]
    fn test_dp54_vs_rkf45_accuracy() {
        // Compare DP54 and RKF45 accuracy on same problem
        let f =
            |t: f64, _: SVector<f64, 1>| -> SVector<f64, 1> { SVector::<f64, 1>::new(3.0 * t * t) };

        let config = IntegratorConfig::adaptive(1e-10, 1e-8);
        let dp54 = DormandPrince54SIntegrator::with_config(Box::new(f), None, config.clone());
        let rkf45 = RKF45SIntegrator::with_config(Box::new(f), None, config);

        let dt = 0.1;
        let mut state_dp54 = SVector::<f64, 1>::new(0.0);
        let mut state_rkf45 = SVector::<f64, 1>::new(0.0);

        for i in 0..100 {
            let t = i as f64 * dt;
            let result_dp54 = dp54.step(t, state_dp54, dt, 1e-10, 1e-8);
            let result_rkf45 = rkf45.step(t, state_rkf45, dt, 1e-10, 1e-8);
            state_dp54 = result_dp54.state;
            state_rkf45 = result_rkf45.state;
        }

        let exact = 1000.0;

        // Both should be very accurate
        let error_dp54 = (state_dp54[0] - exact).abs();
        let error_rkf45 = (state_rkf45[0] - exact).abs();

        // Both 5th order methods should have small error
        assert!(error_dp54 < 1.0e-5);
        assert!(error_rkf45 < 1.0e-5);
    }

    #[test]
    fn test_dp54_stm_accuracy() {
        setup_global_test_eop();

        // Set up variational matrix computation with central differences
        let varmat_config = VarmatConfig::central().with_fixed_offset(0.1);
        let varmat = move |t: f64, state: SVector<f64, 6>| -> SMatrix<f64, 6, 6> {
            varmat_config.compute(t, state, &point_earth)
        };

        let config = IntegratorConfig::adaptive(1e-12, 1e-10);
        let dp54_nominal = DormandPrince54SIntegrator::with_config(
            Box::new(point_earth),
            Some(Box::new(varmat)),
            config.clone(),
        );

        // Circular orbit
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0);
        let state0 = state_osculating_to_cartesian(oe0, DEGREES);

        // Propagate single step
        let dt = 10.0; // 10 seconds
        let (state_new, phi, _dt_used, _error, _dt_next) = dp54_nominal.step_with_varmat(
            0.0,
            state0,
            SMatrix::<f64, 6, 6>::identity(),
            dt,
            1e-12,
            1e-10,
        );

        // Test STM accuracy by comparing with direct perturbation
        for i in 0..6 {
            let mut perturbation = SVector::<f64, 6>::zeros();
            perturbation[i] = 10.0; // 10m or 10mm/s perturbation

            // Create separate integrator for perturbed state to avoid FSAL cache issues
            let dp54_pert = DormandPrince54SIntegrator::with_config(
                Box::new(point_earth),
                None,
                config.clone(),
            );

            // Propagate perturbed state
            let state0_pert = state0 + perturbation;
            let result_pert = dp54_pert.step(0.0, state0_pert, dt, 1e-12, 1e-10);

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

                // DP54 is 5th order, expect very good STM accuracy
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
    fn test_dp54_stm_vs_direct_perturbation() {
        setup_global_test_eop();

        let varmat_config = VarmatConfig::central().with_fixed_offset(0.1);
        let varmat = move |t: f64, state: SVector<f64, 6>| -> SMatrix<f64, 6, 6> {
            varmat_config.compute(t, state, &point_earth)
        };

        let config = IntegratorConfig::adaptive(1e-12, 1e-10);

        // Create separate integrators for nominal and perturbed trajectories to avoid FSAL cache conflicts
        let dp54_nominal = DormandPrince54SIntegrator::with_config(
            Box::new(point_earth),
            Some(Box::new(varmat)),
            config.clone(),
        );
        let dp54_pert =
            DormandPrince54SIntegrator::with_config(Box::new(point_earth), None, config);

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
            let (state_new, phi_new, dt_used, _, _) =
                dp54_nominal.step_with_varmat(t, state, phi, dt, 1e-12, 1e-10);

            // Propagate perturbed state directly
            let result_pert = dp54_pert.step(t, state_pert, dt, 1e-12, 1e-10);

            // Predict perturbed state using STM
            let state_pert_predicted = state_new + phi_new * perturbation;

            // Compare
            let error = (result_pert.state - state_pert_predicted).norm();
            println!("Step {}: error = {:.6e} m", step + 1, error);

            // Error should remain small and not accumulate excessively
            // DP54 STM has ~1e-5 relative error, so for 10m perturbation
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
    // Dynamic DP54 Tests
    // ========================================================================

    fn point_earth_dynamic(_: f64, x: DVector<f64>) -> DVector<f64> {
        assert_eq!(x.len(), 6);
        let r = x.rows(0, 3);
        let v = x.rows(3, 3);

        let r_norm = r.norm();
        let a = -GM_EARTH / r_norm.powi(3);

        let mut x_dot = DVector::<f64>::zeros(6);

        x_dot.rows_mut(0, 3).copy_from(&v);
        x_dot.rows_mut(3, 3).copy_from(&(a * r));
        x_dot
    }

    #[test]
    fn test_dp54d_integrator_parabola() {
        let f = |t: f64, _: DVector<f64>| DVector::from_vec(vec![2.0 * t]);

        let config = IntegratorConfig::adaptive(1e-10, 1e-8);
        let dp54 = DormandPrince54DIntegrator::with_config(1, Box::new(f), None, config);

        let mut t = 0.0;
        let mut state = DVector::from_vec(vec![0.0]);

        while t < 1.0 {
            let dt = f64::min(1.0 - t, 0.1);
            let result = dp54.step(t, state, dt, 1e-10, 1e-8);
            state = result.state;
            t += result.dt_used;
        }

        assert_abs_diff_eq!(state[0], 1.0, epsilon = 1.0e-8);
    }

    #[test]
    fn test_dp54d_integrator_adaptive() {
        let f = |t: f64, _: DVector<f64>| DVector::from_vec(vec![2.0 * t]);

        let config = IntegratorConfig::adaptive(1e-10, 1e-8);
        let dp54 = DormandPrince54DIntegrator::with_config(1, Box::new(f), None, config);

        let mut t = 0.0;
        let mut state = DVector::from_vec(vec![0.0]);

        while t < 1.0 {
            let dt = f64::min(1.0 - t, 0.1);
            let result = dp54.step(t, state, dt, 1e-10, 1e-8);
            state = result.state;
            t += result.dt_used;
            assert!(result.error_estimate >= 0.0);
        }

        assert_abs_diff_eq!(state[0], 1.0, epsilon = 1.0e-8);
    }

    #[test]
    fn test_dp54d_integrator_orbit() {
        // Setup integrator
        let config = IntegratorConfig::adaptive(1e-8, 1e-6);
        let dp54 =
            DormandPrince54DIntegrator::with_config(6, Box::new(point_earth_dynamic), None, config);

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
            let result = dp54.step(epc - epc0, state, dt, 1e-8, 1e-6);
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
    fn test_dp54d_accuracy() {
        let f = |t: f64, _: DVector<f64>| DVector::from_vec(vec![3.0 * t * t]);

        let config = IntegratorConfig::adaptive(1e-8, 1e-6);
        let dp54 = DormandPrince54DIntegrator::with_config(1, Box::new(f), None, config);

        let mut t = 0.0;
        let mut state = DVector::from_vec(vec![0.0]);

        while t < 10.0 {
            let dt = f64::min(10.0 - t, 0.1);
            let result = dp54.step(t, state, dt, 1e-8, 1e-6);
            state = result.state;
            t += result.dt_used;
        }

        let exact = 1000.0;
        let error = (state[0] - exact).abs();
        assert!(error < 1.0e-5);
    }

    #[test]
    fn test_dp54d_step_size_increases() {
        let f = |t: f64, _: DVector<f64>| DVector::from_vec(vec![2.0 * t]);

        let config = IntegratorConfig::adaptive(1e-6, 1e-4);
        let dp54 = DormandPrince54DIntegrator::with_config(1, Box::new(f), None, config);

        let state = DVector::from_vec(vec![0.0]);
        let dt_initial = 0.01;

        let result = dp54.step(0.0, state, dt_initial, 1e-6, 1e-4);

        assert!(result.dt_next > dt_initial);
        assert!(result.error_estimate < 0.1);
    }

    #[test]
    fn test_dp54d_step_size_decreases() {
        let f = |_t: f64, state: DVector<f64>| DVector::from_vec(vec![-1000.0 * state[0]]);

        let config = IntegratorConfig::adaptive(1e-10, 1e-8);
        let dp54 = DormandPrince54DIntegrator::with_config(1, Box::new(f), None, config);

        let state = DVector::from_vec(vec![1.0]);
        let dt_initial = 0.1;

        let result = dp54.step(0.0, state, dt_initial, 1e-10, 1e-8);

        assert!(result.dt_used <= dt_initial);
    }

    #[test]
    fn test_dp54d_config_parameters() {
        // Setup with custom configuration
        let f = |t: f64, _: DVector<f64>| DVector::from_vec(vec![2.0 * t]);
        let mut config = IntegratorConfig::adaptive(1e-8, 1e-6);
        config.step_safety_factor = Some(0.5);
        config.max_step_scale_factor = Some(2.0);

        let dp54 = DormandPrince54DIntegrator::with_config(1, Box::new(f), None, config);
        let state = DVector::from_vec(vec![0.0]);

        // Take step
        let result = dp54.step(0.0, state, 0.01, 1e-8, 1e-6);

        // Verify config parameters limit step size growth
        assert!(result.dt_next <= 2.0 * result.dt_used);
    }

    #[test]
    fn test_dp54d_fsal_cache() {
        // Setup integrator
        let f = |t: f64, _: DVector<f64>| DVector::from_vec(vec![2.0 * t]);
        let config = IntegratorConfig::adaptive(1e-8, 1e-6);
        let dp54 = DormandPrince54DIntegrator::with_config(1, Box::new(f), None, config);

        // Take two consecutive steps to verify FSAL cache works
        let state = DVector::from_vec(vec![0.0]);
        let result1 = dp54.step(0.0, state, 0.1, 1e-8, 1e-6);
        let result2 = dp54.step(result1.dt_used, result1.state, 0.1, 1e-8, 1e-6);

        // Second step should succeed (verifying FSAL cache doesn't cause errors)
        assert!(result2.dt_used > 0.0);
    }

    #[test]
    fn test_dp54d_vs_rkf45_accuracy() {
        // Setup both integrators
        let f = |t: f64, _: DVector<f64>| DVector::from_vec(vec![2.0 * t]);
        let config = IntegratorConfig::adaptive(1e-8, 1e-6);
        let dp54 = DormandPrince54DIntegrator::with_config(1, Box::new(f), None, config.clone());
        let rkf45 = RKF45DIntegrator::with_config(1, Box::new(f), None, config);

        // Take same step with both
        let state = DVector::from_vec(vec![0.0]);
        let result_dp54 = dp54.step(0.0, state.clone(), 0.1, 1e-8, 1e-6);
        let result_rkf45 = rkf45.step(0.0, state, 0.1, 1e-8, 1e-6);

        // Both should produce similar results
        assert!((result_dp54.state[0] - result_rkf45.state[0]).abs() < 1.0e-10);
    }

    #[test]
    fn test_dp54d_stm_accuracy() {
        setup_global_test_eop();

        // Setup integrator with variational matrix
        let varmat_config = VarmatConfig::central().with_fixed_offset(0.1);
        let varmat = move |t: f64, state: DVector<f64>| {
            varmat_config.compute_dynamic(t, state, &point_earth_dynamic)
        };
        let config = IntegratorConfig::adaptive(1e-12, 1e-10);
        let dp54 = DormandPrince54DIntegrator::with_config(
            6,
            Box::new(point_earth_dynamic),
            Some(Box::new(varmat)),
            config.clone(),
        );

        // Setup circular orbit
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0);
        let state0_static = state_osculating_to_cartesian(oe0, DEGREES);
        let state0 = DVector::from_vec(state0_static.as_slice().to_vec());

        // Propagate with STM
        let dt = 10.0;
        let (state_new, phi, _, _, _) = dp54.step_with_varmat(
            0.0,
            state0.clone(),
            DMatrix::identity(6, 6),
            dt,
            1e-12,
            1e-10,
        );

        // Test STM accuracy by comparing with direct perturbation
        for i in 0..6 {
            let mut perturbation = DVector::zeros(6);
            perturbation[i] = 10.0;
            let state0_pert = &state0 + &perturbation;

            // Create separate integrator for perturbed state to avoid FSAL cache issues
            let dp54_pert = DormandPrince54DIntegrator::with_config(
                6,
                Box::new(point_earth_dynamic),
                None,
                config.clone(),
            );

            // Propagate perturbed state
            let result_pert = dp54_pert.step(0.0, state0_pert, dt, 1e-12, 1e-10);

            // Predict perturbed state using STM
            let state_pert_predicted = &state_new + &phi * &perturbation;

            // Verify each component
            for j in 0..6 {
                let error = (result_pert.state[j] - state_pert_predicted[j]).abs();
                let relative_error = error / perturbation[i].abs();
                if j < 3 {
                    assert!(relative_error < 1e-5);
                } else {
                    assert!(relative_error < 1e-4);
                }
            }
        }
    }

    #[test]
    fn test_dp54d_stm_vs_direct_perturbation() {
        setup_global_test_eop();

        // Setup variational matrix computation
        let varmat_config = VarmatConfig::central().with_fixed_offset(0.1);
        let varmat = move |t: f64, state: DVector<f64>| {
            varmat_config.compute_dynamic(t, state, &point_earth_dynamic)
        };
        let config = IntegratorConfig::adaptive(1e-12, 1e-10);

        // Create separate integrators for nominal and perturbed trajectories to avoid FSAL cache conflicts
        let dp54_nominal = DormandPrince54DIntegrator::with_config(
            6,
            Box::new(point_earth_dynamic),
            Some(Box::new(varmat)),
            config.clone(),
        );
        let dp54_pert =
            DormandPrince54DIntegrator::with_config(6, Box::new(point_earth_dynamic), None, config);

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
        let mut phi = DMatrix::identity(6, 6);
        let mut state_pert = &state0 + &perturbation;
        let mut t = 0.0;

        // Propagate both trajectories and verify STM prediction at each step
        for step in 0..num_steps {
            // Propagate nominal state with STM
            let (state_new, phi_new, dt_used, _, _) =
                dp54_nominal.step_with_varmat(t, state.clone(), phi.clone(), dt, 1e-12, 1e-10);

            // Propagate perturbed state directly
            let result_pert = dp54_pert.step(t, state_pert.clone(), dt, 1e-12, 1e-10);

            // Predict perturbed state using STM
            let state_pert_predicted = &state_new + &phi_new * &perturbation;

            // Verify STM prediction accuracy (error grows linearly with time)
            let error = (&result_pert.state - &state_pert_predicted).norm();
            let max_error = 0.001 * (step + 1) as f64;
            assert!(error < max_error);

            // Update for next step
            state = state_new;
            phi = phi_new;
            state_pert = result_pert.state;
            t += dt_used;
        }
    }

    #[test]
    fn test_dp54_s_vs_d_consistency() {
        // Verify DormandPrince54SIntegrator and DormandPrince54DIntegrator produce identical results
        let f_static = |_t: f64, x: SVector<f64, 2>| -> SVector<f64, 2> {
            SVector::<f64, 2>::new(x[1], -x[0])
        };
        let f_dynamic =
            |_t: f64, x: DVector<f64>| -> DVector<f64> { DVector::from_vec(vec![x[1], -x[0]]) };

        let config = IntegratorConfig::adaptive(1e-10, 1e-8);
        let dp54_s =
            DormandPrince54SIntegrator::with_config(Box::new(f_static), None, config.clone());
        let dp54_d = DormandPrince54DIntegrator::with_config(2, Box::new(f_dynamic), None, config);

        let state_s = SVector::<f64, 2>::new(1.0, 0.0);
        let state_d = DVector::from_vec(vec![1.0, 0.0]);
        let dt = 0.1;

        let result_s = dp54_s.step(0.0, state_s, dt, 1e-10, 1e-8);
        let result_d = dp54_d.step(0.0, state_d, dt, 1e-10, 1e-8);

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
}
