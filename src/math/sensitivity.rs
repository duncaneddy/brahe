/*!
 * Sensitivity matrix computation for parameter estimation and consider covariance analysis.
 *
 * This module provides trait-based interfaces for computing sensitivity matrices (∂f/∂p),
 * with implementations for both analytical and numerical (finite difference) methods.
 *
 * Sensitivity matrices describe how a function's output changes with respect to consider
 * parameters, which is essential for orbit determination with consider parameters and
 * covariance analysis.
 */

use crate::math::jacobian::{DifferenceMethod, PerturbationStrategy};
use nalgebra::{DMatrix, DVector, SMatrix, SVector};

// ============================================================================
// Type Aliases for Sensitivity Functions
// ============================================================================

/// Sensitivity function type for static-sized systems.
type SSensitivityFn<const S: usize, const P: usize> =
    Box<dyn Fn(f64, &SVector<f64, S>, &SVector<f64, P>) -> SMatrix<f64, S, P> + Send>;

/// Sensitivity function type for dynamic-sized systems.
type DSensitivityFn = Box<dyn Fn(f64, &DVector<f64>, &DVector<f64>) -> DMatrix<f64> + Send>;

/// Dynamics function type for static-sized sensitivity computation.
type SDynamicsWithParams<const S: usize, const P: usize> =
    Box<dyn Fn(f64, &SVector<f64, S>, &SVector<f64, P>) -> SVector<f64, S> + Send>;

/// Dynamics function type for dynamic-sized sensitivity computation.
type DDynamicsWithParams = Box<dyn Fn(f64, &DVector<f64>, &DVector<f64>) -> DVector<f64> + Send>;

/// Trait for static-sized sensitivity providers.
///x
/// Computes the sensitivity matrix ∂f/∂p where f is the dynamics function
/// and p are the consider parameters.
pub trait SSensitivityProvider<const S: usize, const P: usize>: Send {
    /// Compute the sensitivity matrix at the given time, state, and parameters.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `params`: Consider parameters
    ///
    /// # Returns
    /// Sensitivity matrix ∂f/∂p (S×P matrix)
    fn compute(
        &self,
        t: f64,
        state: &SVector<f64, S>,
        params: &SVector<f64, P>,
    ) -> SMatrix<f64, S, P>;
}

/// Trait for dynamic-sized sensitivity providers.
///
/// Computes the sensitivity matrix ∂f/∂p where f is the dynamics function
/// and p are the consider parameters.
pub trait DSensitivityProvider: Send {
    /// Compute the sensitivity matrix at the given time, state, and parameters.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `params`: Consider parameters
    ///
    /// # Returns
    /// Sensitivity matrix ∂f/∂p (S×P matrix)
    fn compute(&self, t: f64, state: &DVector<f64>, params: &DVector<f64>) -> DMatrix<f64>;
}

/// Analytical sensitivity provider for static-sized systems.
///
/// Uses a user-provided function that directly computes the analytical sensitivity matrix.
pub struct SAnalyticSensitivity<const S: usize, const P: usize> {
    sensitivity_fn: SSensitivityFn<S, P>,
}

impl<const S: usize, const P: usize> SAnalyticSensitivity<S, P> {
    /// Create a new analytical sensitivity provider.
    pub fn new(sensitivity_fn: SSensitivityFn<S, P>) -> Self {
        Self { sensitivity_fn }
    }
}

impl<const S: usize, const P: usize> SSensitivityProvider<S, P> for SAnalyticSensitivity<S, P> {
    fn compute(
        &self,
        t: f64,
        state: &SVector<f64, S>,
        params: &SVector<f64, P>,
    ) -> SMatrix<f64, S, P> {
        (self.sensitivity_fn)(t, state, params)
    }
}

/// Analytical sensitivity provider for dynamic-sized systems.
///
/// Uses a user-provided function that directly computes the analytical sensitivity matrix.
pub struct DAnalyticSensitivity {
    sensitivity_fn: DSensitivityFn,
}

impl DAnalyticSensitivity {
    /// Create a new analytical sensitivity provider.
    pub fn new(sensitivity_fn: DSensitivityFn) -> Self {
        Self { sensitivity_fn }
    }
}

impl DSensitivityProvider for DAnalyticSensitivity {
    fn compute(&self, t: f64, state: &DVector<f64>, params: &DVector<f64>) -> DMatrix<f64> {
        (self.sensitivity_fn)(t, state, params)
    }
}

/// Numerical sensitivity provider for static-sized systems using finite differences.
///
/// Computes the sensitivity matrix numerically by perturbing the parameters
/// and evaluating the dynamics function.
pub struct SNumericalSensitivity<const S: usize, const P: usize> {
    dynamics_fn: SDynamicsWithParams<S, P>,
    method: DifferenceMethod,
    strategy: PerturbationStrategy,
}

impl<const S: usize, const P: usize> SNumericalSensitivity<S, P> {
    /// Create a new numerical sensitivity provider with default settings.
    ///
    /// Uses central differences with adaptive perturbation strategy.
    pub fn new(dynamics_fn: SDynamicsWithParams<S, P>) -> Self {
        Self {
            dynamics_fn,
            method: DifferenceMethod::Central,
            strategy: PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_threshold: 1.0,
            },
        }
    }

    /// Create with central differences (default, most accurate).
    pub fn central(dynamics_fn: SDynamicsWithParams<S, P>) -> Self {
        Self::new(dynamics_fn)
    }

    /// Create with forward differences (faster but less accurate).
    pub fn forward(dynamics_fn: SDynamicsWithParams<S, P>) -> Self {
        Self {
            dynamics_fn,
            method: DifferenceMethod::Forward,
            strategy: PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_threshold: 1.0,
            },
        }
    }

    /// Create with backward differences.
    pub fn backward(dynamics_fn: SDynamicsWithParams<S, P>) -> Self {
        Self {
            dynamics_fn,
            method: DifferenceMethod::Backward,
            strategy: PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_threshold: 1.0,
            },
        }
    }

    /// Set the perturbation strategy.
    pub fn with_strategy(mut self, strategy: PerturbationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    fn compute_perturbation(&self, value: f64) -> f64 {
        match self.strategy {
            PerturbationStrategy::Adaptive {
                scale_factor,
                min_threshold,
            } => {
                let eps = f64::EPSILON;
                scale_factor * eps.sqrt() * value.abs().max(min_threshold)
            }
            PerturbationStrategy::Fixed(h) => h,
            PerturbationStrategy::Percentage(pct) => value.abs() * pct,
        }
    }
}

impl<const S: usize, const P: usize> SSensitivityProvider<S, P> for SNumericalSensitivity<S, P> {
    fn compute(
        &self,
        t: f64,
        state: &SVector<f64, S>,
        params: &SVector<f64, P>,
    ) -> SMatrix<f64, S, P> {
        let mut sensitivity = SMatrix::<f64, S, P>::zeros();

        for j in 0..P {
            let h = self.compute_perturbation(params[j]);

            let column = match self.method {
                DifferenceMethod::Forward => {
                    let mut params_plus = *params;
                    params_plus[j] += h;
                    let f_plus = (self.dynamics_fn)(t, state, &params_plus);
                    let f_0 = (self.dynamics_fn)(t, state, params);
                    (f_plus - f_0) / h
                }
                DifferenceMethod::Central => {
                    let mut params_plus = *params;
                    let mut params_minus = *params;
                    params_plus[j] += h;
                    params_minus[j] -= h;
                    let f_plus = (self.dynamics_fn)(t, state, &params_plus);
                    let f_minus = (self.dynamics_fn)(t, state, &params_minus);
                    (f_plus - f_minus) / (2.0 * h)
                }
                DifferenceMethod::Backward => {
                    let mut params_minus = *params;
                    params_minus[j] -= h;
                    let f_0 = (self.dynamics_fn)(t, state, params);
                    let f_minus = (self.dynamics_fn)(t, state, &params_minus);
                    (f_0 - f_minus) / h
                }
            };

            sensitivity.set_column(j, &column);
        }

        sensitivity
    }
}

/// Numerical sensitivity provider for dynamic-sized systems using finite differences.
///
/// Computes the sensitivity matrix numerically by perturbing the parameters
/// and evaluating the dynamics function.
pub struct DNumericalSensitivity {
    dynamics_fn: DDynamicsWithParams,
    method: DifferenceMethod,
    strategy: PerturbationStrategy,
}

impl DNumericalSensitivity {
    /// Create a new numerical sensitivity provider with default settings.
    ///
    /// Uses central differences with adaptive perturbation strategy.
    pub fn new(dynamics_fn: DDynamicsWithParams) -> Self {
        Self {
            dynamics_fn,
            method: DifferenceMethod::Central,
            strategy: PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_threshold: 1.0,
            },
        }
    }

    /// Create with central differences (default, most accurate).
    pub fn central(dynamics_fn: DDynamicsWithParams) -> Self {
        Self::new(dynamics_fn)
    }

    /// Create with forward differences (faster but less accurate).
    pub fn forward(dynamics_fn: DDynamicsWithParams) -> Self {
        Self {
            dynamics_fn,
            method: DifferenceMethod::Forward,
            strategy: PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_threshold: 1.0,
            },
        }
    }

    /// Create with backward differences.
    pub fn backward(dynamics_fn: DDynamicsWithParams) -> Self {
        Self {
            dynamics_fn,
            method: DifferenceMethod::Backward,
            strategy: PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_threshold: 1.0,
            },
        }
    }

    /// Set the perturbation strategy.
    pub fn with_strategy(mut self, strategy: PerturbationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    fn compute_perturbation(&self, value: f64) -> f64 {
        match self.strategy {
            PerturbationStrategy::Adaptive {
                scale_factor,
                min_threshold,
            } => {
                let eps = f64::EPSILON;
                scale_factor * eps.sqrt() * value.abs().max(min_threshold)
            }
            PerturbationStrategy::Fixed(h) => h,
            PerturbationStrategy::Percentage(pct) => value.abs() * pct,
        }
    }
}

impl DSensitivityProvider for DNumericalSensitivity {
    fn compute(&self, t: f64, state: &DVector<f64>, params: &DVector<f64>) -> DMatrix<f64> {
        let s = state.len();
        let p = params.len();
        let mut sensitivity = DMatrix::<f64>::zeros(s, p);

        for j in 0..p {
            let h = self.compute_perturbation(params[j]);

            let column = match self.method {
                DifferenceMethod::Forward => {
                    let mut params_plus = params.clone();
                    params_plus[j] += h;
                    let f_plus = (self.dynamics_fn)(t, state, &params_plus);
                    let f_0 = (self.dynamics_fn)(t, state, params);
                    (f_plus - f_0) / h
                }
                DifferenceMethod::Central => {
                    let mut params_plus = params.clone();
                    let mut params_minus = params.clone();
                    params_plus[j] += h;
                    params_minus[j] -= h;
                    let f_plus = (self.dynamics_fn)(t, state, &params_plus);
                    let f_minus = (self.dynamics_fn)(t, state, &params_minus);
                    (f_plus - f_minus) / (2.0 * h)
                }
                DifferenceMethod::Backward => {
                    let mut params_minus = params.clone();
                    params_minus[j] -= h;
                    let f_0 = (self.dynamics_fn)(t, state, params);
                    let f_minus = (self.dynamics_fn)(t, state, &params_minus);
                    (f_0 - f_minus) / h
                }
            };

            sensitivity.set_column(j, &column);
        }

        sensitivity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_dynamic_numerical_sensitivity() {
        // Dynamics: f(t, x, p) = p[0] * x
        let dynamics = |_t: f64, state: &DVector<f64>, params: &DVector<f64>| -> DVector<f64> {
            params[0] * state
        };

        let provider = DNumericalSensitivity::central(Box::new(dynamics));

        let state = DVector::from_vec(vec![1.0, 2.0]);
        let params = DVector::from_vec(vec![3.0]);

        let sens = provider.compute(0.0, &state, &params);

        // ∂f/∂p = x, so sensitivity should be [[1], [2]]
        assert_eq!(sens.nrows(), 2);
        assert_eq!(sens.ncols(), 1);
        assert_abs_diff_eq!(sens[(0, 0)], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(sens[(1, 0)], 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_dynamic_analytical_sensitivity() {
        // Analytical sensitivity: ∂f/∂p = x
        let sensitivity_fn =
            |_t: f64, state: &DVector<f64>, _params: &DVector<f64>| -> DMatrix<f64> {
                DMatrix::from_column_slice(state.len(), 1, state.as_slice())
            };

        let provider = DAnalyticSensitivity::new(Box::new(sensitivity_fn));

        let state = DVector::from_vec(vec![1.0, 2.0]);
        let params = DVector::from_vec(vec![3.0]);

        let sens = provider.compute(0.0, &state, &params);

        assert_eq!(sens.nrows(), 2);
        assert_eq!(sens.ncols(), 1);
        assert_abs_diff_eq!(sens[(0, 0)], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sens[(1, 0)], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_static_numerical_sensitivity() {
        // Dynamics: f(t, x, p) = [p[0] * x[0], p[1] * x[1]]
        let dynamics =
            |_t: f64, state: &SVector<f64, 2>, params: &SVector<f64, 2>| -> SVector<f64, 2> {
                SVector::<f64, 2>::new(params[0] * state[0], params[1] * state[1])
            };

        let provider = SNumericalSensitivity::central(Box::new(dynamics));

        let state = SVector::<f64, 2>::new(1.0, 2.0);
        let params = SVector::<f64, 2>::new(3.0, 4.0);

        let sens = provider.compute(0.0, &state, &params);

        // ∂f/∂p = [[x[0], 0], [0, x[1]]]
        assert_abs_diff_eq!(sens[(0, 0)], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(sens[(0, 1)], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(sens[(1, 0)], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(sens[(1, 1)], 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_static_analytical_sensitivity() {
        // Analytical sensitivity
        let sensitivity_fn =
            |_t: f64, state: &SVector<f64, 2>, _params: &SVector<f64, 2>| -> SMatrix<f64, 2, 2> {
                SMatrix::<f64, 2, 2>::new(state[0], 0.0, 0.0, state[1])
            };

        let provider = SAnalyticSensitivity::new(Box::new(sensitivity_fn));

        let state = SVector::<f64, 2>::new(1.0, 2.0);
        let params = SVector::<f64, 2>::new(3.0, 4.0);

        let sens = provider.compute(0.0, &state, &params);

        assert_abs_diff_eq!(sens[(0, 0)], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sens[(0, 1)], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sens[(1, 0)], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sens[(1, 1)], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_forward_vs_central_difference() {
        // Simple quadratic dynamics to test difference methods
        let dynamics = |_t: f64, state: &DVector<f64>, params: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![params[0] * params[0] * state[0]])
        };

        let state = DVector::from_vec(vec![1.0]);
        let params = DVector::from_vec(vec![2.0]);

        let forward = DNumericalSensitivity::forward(Box::new(dynamics));
        let central = DNumericalSensitivity::central(Box::new(dynamics));

        let sens_forward = forward.compute(0.0, &state, &params);
        let sens_central = central.compute(0.0, &state, &params);

        // Analytical: ∂f/∂p = 2*p*x = 4
        // Central should be more accurate
        assert_abs_diff_eq!(sens_central[(0, 0)], 4.0, epsilon = 1e-8);
        assert_abs_diff_eq!(sens_forward[(0, 0)], 4.0, epsilon = 1e-4);
    }
}
