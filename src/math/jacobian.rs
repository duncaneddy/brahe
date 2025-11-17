/*!
 * Jacobian computation for numerical analysis and integration.
 *
 * This module provides trait-based interfaces for computing Jacobian matrices,
 * with implementations for both analytical and numerical (finite difference) methods.
 */

use nalgebra::{DMatrix, DVector, SMatrix, SVector};

/// Finite difference method for numerical Jacobian approximation.
///
/// Different methods trade off accuracy vs computational cost:
/// - **Forward**: O(h) error, S+1 function evaluations
/// - **Central**: O(h²) error, 2S function evaluations (more accurate)
/// - **Backward**: O(h) error, S+1 function evaluations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DifferenceMethod {
    /// Forward finite difference: df/dx ≈ (f(x+h) - f(x)) / h
    ///
    /// First-order accurate but cheaper (S+1 evaluations).
    Forward,

    /// Central finite difference: df/dx ≈ (f(x+h) - f(x-h)) / (2h)
    ///
    /// Second-order accurate but more expensive (2S evaluations).
    /// Generally preferred for high-precision applications.
    Central,

    /// Backward finite difference: df/dx ≈ (f(x) - f(x-h)) / h
    ///
    /// First-order accurate, same cost as forward differences (S+1 evaluations).
    /// Useful when forward perturbations are problematic.
    Backward,
}

/// Strategy for computing perturbation sizes in finite differences.
///
/// The choice of perturbation size balances truncation error (wants large h)
/// vs roundoff error (wants small h). Different strategies suit different problems.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PerturbationStrategy {
    /// Automatic perturbation sizing: h_i = scale_factor * sqrt(ε) * max(|x_i|, min_threshold)
    ///
    /// Industry standard approach that adapts to state magnitude. The sqrt(ε) scaling
    /// (where ε is machine epsilon) optimally balances truncation vs roundoff error.
    Adaptive {
        /// Multiplier on sqrt(ε), typically 1.0
        scale_factor: f64,
        /// Minimum reference value (prevents tiny perturbations near zero)
        min_threshold: f64,
    },

    /// Fixed absolute perturbation for all state components.
    ///
    /// Use when all components have similar units/scales (e.g., purely position coordinates).
    Fixed(f64),

    /// Percentage-based perturbation: h_i = |x_i| * percentage
    ///
    /// Use when components have vastly different scales. Fails for near-zero values.
    Percentage(f64),
}

/// Trait for static-sized (compile-time known) Jacobian providers.
///
/// Implementors of this trait can compute Jacobian matrices for systems with
/// compile-time known dimensionality.
pub trait SJacobianProvider<const S: usize>: Send {
    /// Compute the Jacobian matrix at the given time and state.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    ///
    /// # Returns
    /// Jacobian matrix ∂f/∂x
    fn compute(&self, t: f64, state: SVector<f64, S>) -> SMatrix<f64, S, S>;
}

/// Trait for dynamic-sized (runtime-known) Jacobian providers.
///
/// Implementors of this trait can compute Jacobian matrices for systems with
/// runtime-determined dimensionality.
pub trait DJacobianProvider: Send {
    /// Compute the Jacobian matrix at the given time and state.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    ///
    /// # Returns
    /// Jacobian matrix ∂f/∂x
    fn compute(&self, t: f64, state: DVector<f64>) -> DMatrix<f64>;
}

/// Analytical Jacobian provider for static-sized systems.
///
/// Uses a user-provided function that directly computes the analytical Jacobian.
/// This is the most accurate and efficient method when the analytical Jacobian is known.
///
/// # Examples
///
/// ```rust
/// use brahe::math::jacobian::{SAnalyticJacobian, SJacobianProvider};
/// use nalgebra::{SVector, SMatrix};
///
/// // Simple harmonic oscillator: dx/dt = v, dv/dt = -x
/// // Jacobian is [[0, 1], [-1, 0]]
/// let jacobian_fn = |_t: f64, _state: SVector<f64, 2>| {
///     SMatrix::<f64, 2, 2>::new(
///         0.0,  1.0,
///        -1.0,  0.0
///     )
/// };
///
/// let provider = SAnalyticJacobian::new(Box::new(jacobian_fn));
/// let state = SVector::<f64, 2>::new(1.0, 0.0);
/// let jacobian = provider.compute(0.0, state);
/// ```
pub struct SAnalyticJacobian<const S: usize> {
    jacobian_fn: Box<dyn Fn(f64, SVector<f64, S>) -> SMatrix<f64, S, S> + Send>,
}

impl<const S: usize> SAnalyticJacobian<S> {
    /// Create a new analytical Jacobian provider.
    ///
    /// # Arguments
    /// - `jacobian_fn`: Function that computes the analytical Jacobian
    pub fn new(
        jacobian_fn: Box<dyn Fn(f64, SVector<f64, S>) -> SMatrix<f64, S, S> + Send>,
    ) -> Self {
        Self { jacobian_fn }
    }
}

impl<const S: usize> SJacobianProvider<S> for SAnalyticJacobian<S> {
    fn compute(&self, t: f64, state: SVector<f64, S>) -> SMatrix<f64, S, S> {
        (self.jacobian_fn)(t, state)
    }
}

/// Analytical Jacobian provider for dynamic-sized systems.
///
/// Uses a user-provided function that directly computes the analytical Jacobian.
/// This is the most accurate and efficient method when the analytical Jacobian is known.
///
/// # Examples
///
/// ```rust
/// use brahe::math::jacobian::{DAnalyticJacobian, DJacobianProvider};
/// use nalgebra::{DVector, DMatrix};
///
/// // Simple harmonic oscillator: dx/dt = v, dv/dt = -x
/// // Jacobian is [[0, 1], [-1, 0]]
/// let jacobian_fn = |_t: f64, _state: DVector<f64>| {
///     DMatrix::<f64>::from_row_slice(2, 2, &[
///         0.0,  1.0,
///        -1.0,  0.0
///     ])
/// };
///
/// let provider = DAnalyticJacobian::new(Box::new(jacobian_fn));
/// let state = DVector::from_vec(vec![1.0, 0.0]);
/// let jacobian = provider.compute(0.0, state);
/// ```
pub struct DAnalyticJacobian {
    jacobian_fn: Box<dyn Fn(f64, DVector<f64>) -> DMatrix<f64> + Send>,
}

impl DAnalyticJacobian {
    /// Create a new analytical Jacobian provider.
    ///
    /// # Arguments
    /// - `jacobian_fn`: Function that computes the analytical Jacobian
    pub fn new(jacobian_fn: Box<dyn Fn(f64, DVector<f64>) -> DMatrix<f64> + Send>) -> Self {
        Self { jacobian_fn }
    }
}

impl DJacobianProvider for DAnalyticJacobian {
    fn compute(&self, t: f64, state: DVector<f64>) -> DMatrix<f64> {
        (self.jacobian_fn)(t, state)
    }
}

/// Numerical Jacobian provider for static-sized systems using finite differences.
///
/// Computes the Jacobian numerically by perturbing the state and evaluating the dynamics.
/// Supports forward, central, and backward finite difference methods with various
/// perturbation strategies.
///
/// # Examples
///
/// ```rust
/// use brahe::math::jacobian::{SNumericalJacobian, SJacobianProvider};
/// use nalgebra::SVector;
///
/// // Simple harmonic oscillator dynamics
/// let dynamics = |_t: f64, state: SVector<f64, 2>| {
///     SVector::<f64, 2>::new(state[1], -state[0])
/// };
///
/// // Default: central differences with adaptive perturbations
/// let provider = SNumericalJacobian::new(Box::new(dynamics));
///
/// // Or with custom settings:
/// let provider = SNumericalJacobian::forward(Box::new(dynamics))
///     .with_fixed_offset(1e-6);
///
/// let state = SVector::<f64, 2>::new(1.0, 0.0);
/// let jacobian = provider.compute(0.0, state);
/// ```
pub struct SNumericalJacobian<const S: usize> {
    dynamics_fn: Box<dyn Fn(f64, SVector<f64, S>) -> SVector<f64, S> + Send>,
    method: DifferenceMethod,
    perturbation: PerturbationStrategy,
}

impl<const S: usize> SNumericalJacobian<S> {
    /// Create a numerical Jacobian provider with default settings.
    ///
    /// Default: central differences with adaptive perturbations.
    pub fn new(dynamics_fn: Box<dyn Fn(f64, SVector<f64, S>) -> SVector<f64, S> + Send>) -> Self {
        Self {
            dynamics_fn,
            method: DifferenceMethod::Central,
            perturbation: PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_threshold: 1.0,
            },
        }
    }

    /// Create with forward finite differences.
    pub fn forward(
        dynamics_fn: Box<dyn Fn(f64, SVector<f64, S>) -> SVector<f64, S> + Send>,
    ) -> Self {
        Self {
            dynamics_fn,
            method: DifferenceMethod::Forward,
            perturbation: PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_threshold: 1.0,
            },
        }
    }

    /// Create with central finite differences.
    pub fn central(
        dynamics_fn: Box<dyn Fn(f64, SVector<f64, S>) -> SVector<f64, S> + Send>,
    ) -> Self {
        Self::new(dynamics_fn)
    }

    /// Create with backward finite differences.
    pub fn backward(
        dynamics_fn: Box<dyn Fn(f64, SVector<f64, S>) -> SVector<f64, S> + Send>,
    ) -> Self {
        Self {
            dynamics_fn,
            method: DifferenceMethod::Backward,
            perturbation: PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_threshold: 1.0,
            },
        }
    }

    /// Set fixed absolute perturbation for all components.
    pub fn with_fixed_offset(mut self, offset: f64) -> Self {
        self.perturbation = PerturbationStrategy::Fixed(offset);
        self
    }

    /// Set percentage-based perturbation.
    pub fn with_percentage(mut self, percentage: f64) -> Self {
        self.perturbation = PerturbationStrategy::Percentage(percentage);
        self
    }

    /// Set adaptive perturbation with custom parameters.
    pub fn with_adaptive(mut self, scale_factor: f64, min_threshold: f64) -> Self {
        self.perturbation = PerturbationStrategy::Adaptive {
            scale_factor,
            min_threshold,
        };
        self
    }

    /// Set the difference method.
    pub fn with_method(mut self, method: DifferenceMethod) -> Self {
        self.method = method;
        self
    }

    /// Compute perturbation offsets for each state component.
    fn compute_offsets(&self, state: &SVector<f64, S>) -> SVector<f64, S> {
        match self.perturbation {
            PerturbationStrategy::Adaptive {
                scale_factor,
                min_threshold,
            } => {
                // Industry standard: h = sqrt(eps) * max(|x|, threshold)
                #[allow(non_snake_case)]
                let SQRT_EPS: f64 = f64::EPSILON.sqrt();
                let base_offset = scale_factor * SQRT_EPS;

                state.map(|x| base_offset * x.abs().max(min_threshold))
            }
            PerturbationStrategy::Fixed(offset) => SVector::from_element(offset),
            PerturbationStrategy::Percentage(pct) => state.map(|x| x.abs() * pct),
        }
    }

    /// Compute Jacobian using forward finite differences.
    ///
    /// Cost: S + 1 function evaluations
    fn compute_forward(&self, t: f64, state: SVector<f64, S>) -> SMatrix<f64, S, S> {
        let offsets = self.compute_offsets(&state);

        // Evaluate unperturbed dynamics
        let f0 = (self.dynamics_fn)(t, state);

        // Initialize Jacobian matrix
        let mut jacobian = SMatrix::<f64, S, S>::zeros();

        // Compute each column: ∂f/∂x_i ≈ (f(x + h*e_i) - f(x)) / h
        for i in 0..S {
            let mut perturbed = state;
            perturbed[i] += offsets[i];

            let fp = (self.dynamics_fn)(t, perturbed);
            jacobian.set_column(i, &((fp - f0) / offsets[i]));
        }

        jacobian
    }

    /// Compute Jacobian using central finite differences.
    ///
    /// Cost: 2S function evaluations (more accurate than forward differences)
    fn compute_central(&self, t: f64, state: SVector<f64, S>) -> SMatrix<f64, S, S> {
        let offsets = self.compute_offsets(&state);

        // Initialize Jacobian matrix
        let mut jacobian = SMatrix::<f64, S, S>::zeros();

        // Compute each column: ∂f/∂x_i ≈ (f(x + h*e_i) - f(x - h*e_i)) / (2h)
        for i in 0..S {
            let mut perturbed_plus = state;
            let mut perturbed_minus = state;
            perturbed_plus[i] += offsets[i];
            perturbed_minus[i] -= offsets[i];

            let fp = (self.dynamics_fn)(t, perturbed_plus);
            let fm = (self.dynamics_fn)(t, perturbed_minus);
            jacobian.set_column(i, &((fp - fm) / (2.0 * offsets[i])));
        }

        jacobian
    }

    /// Compute Jacobian using backward finite differences.
    ///
    /// Cost: S + 1 function evaluations
    fn compute_backward(&self, t: f64, state: SVector<f64, S>) -> SMatrix<f64, S, S> {
        let offsets = self.compute_offsets(&state);

        // Evaluate unperturbed dynamics
        let f0 = (self.dynamics_fn)(t, state);

        // Initialize Jacobian matrix
        let mut jacobian = SMatrix::<f64, S, S>::zeros();

        // Compute each column: ∂f/∂x_i ≈ (f(x) - f(x - h*e_i)) / h
        for i in 0..S {
            let mut perturbed = state;
            perturbed[i] -= offsets[i];

            let fm = (self.dynamics_fn)(t, perturbed);
            jacobian.set_column(i, &((f0 - fm) / offsets[i]));
        }

        jacobian
    }
}

impl<const S: usize> SJacobianProvider<S> for SNumericalJacobian<S> {
    fn compute(&self, t: f64, state: SVector<f64, S>) -> SMatrix<f64, S, S> {
        match self.method {
            DifferenceMethod::Forward => self.compute_forward(t, state),
            DifferenceMethod::Central => self.compute_central(t, state),
            DifferenceMethod::Backward => self.compute_backward(t, state),
        }
    }
}

/// Numerical Jacobian provider for dynamic-sized systems using finite differences.
///
/// Computes the Jacobian numerically by perturbing the state and evaluating the dynamics.
/// Supports forward, central, and backward finite difference methods with various
/// perturbation strategies.
///
/// # Examples
///
/// ```rust
/// use brahe::math::jacobian::{DNumericalJacobian, DJacobianProvider};
/// use nalgebra::DVector;
///
/// // Simple harmonic oscillator dynamics
/// let dynamics = |_t: f64, state: DVector<f64>| {
///     DVector::from_vec(vec![state[1], -state[0]])
/// };
///
/// // Default: central differences with adaptive perturbations
/// let provider = DNumericalJacobian::new(Box::new(dynamics));
///
/// // Or with custom settings:
/// let provider = DNumericalJacobian::forward(Box::new(dynamics))
///     .with_fixed_offset(1e-6);
///
/// let state = DVector::from_vec(vec![1.0, 0.0]);
/// let jacobian = provider.compute(0.0, state);
/// ```
pub struct DNumericalJacobian {
    dynamics_fn: Box<dyn Fn(f64, DVector<f64>) -> DVector<f64> + Send>,
    method: DifferenceMethod,
    perturbation: PerturbationStrategy,
}

impl DNumericalJacobian {
    /// Create a numerical Jacobian provider with default settings.
    ///
    /// Default: central differences with adaptive perturbations.
    pub fn new(dynamics_fn: Box<dyn Fn(f64, DVector<f64>) -> DVector<f64> + Send>) -> Self {
        Self {
            dynamics_fn,
            method: DifferenceMethod::Central,
            perturbation: PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_threshold: 1.0,
            },
        }
    }

    /// Create with forward finite differences.
    pub fn forward(dynamics_fn: Box<dyn Fn(f64, DVector<f64>) -> DVector<f64> + Send>) -> Self {
        Self {
            dynamics_fn,
            method: DifferenceMethod::Forward,
            perturbation: PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_threshold: 1.0,
            },
        }
    }

    /// Create with central finite differences.
    pub fn central(dynamics_fn: Box<dyn Fn(f64, DVector<f64>) -> DVector<f64> + Send>) -> Self {
        Self::new(dynamics_fn)
    }

    /// Create with backward finite differences.
    pub fn backward(dynamics_fn: Box<dyn Fn(f64, DVector<f64>) -> DVector<f64> + Send>) -> Self {
        Self {
            dynamics_fn,
            method: DifferenceMethod::Backward,
            perturbation: PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_threshold: 1.0,
            },
        }
    }

    /// Set fixed absolute perturbation for all components.
    pub fn with_fixed_offset(mut self, offset: f64) -> Self {
        self.perturbation = PerturbationStrategy::Fixed(offset);
        self
    }

    /// Set percentage-based perturbation.
    pub fn with_percentage(mut self, percentage: f64) -> Self {
        self.perturbation = PerturbationStrategy::Percentage(percentage);
        self
    }

    /// Set adaptive perturbation with custom parameters.
    pub fn with_adaptive(mut self, scale_factor: f64, min_threshold: f64) -> Self {
        self.perturbation = PerturbationStrategy::Adaptive {
            scale_factor,
            min_threshold,
        };
        self
    }

    /// Set the difference method.
    pub fn with_method(mut self, method: DifferenceMethod) -> Self {
        self.method = method;
        self
    }

    /// Compute perturbation offsets for each state component.
    fn compute_offsets(&self, state: &DVector<f64>) -> DVector<f64> {
        match self.perturbation {
            PerturbationStrategy::Adaptive {
                scale_factor,
                min_threshold,
            } => {
                // Industry standard: h = sqrt(eps) * max(|x|, threshold)
                #[allow(non_snake_case)]
                let SQRT_EPS: f64 = f64::EPSILON.sqrt();
                let base_offset = scale_factor * SQRT_EPS;

                state.map(|x| base_offset * x.abs().max(min_threshold))
            }
            PerturbationStrategy::Fixed(offset) => DVector::from_element(state.len(), offset),
            PerturbationStrategy::Percentage(pct) => state.map(|x| x.abs() * pct),
        }
    }

    /// Compute Jacobian using forward finite differences.
    ///
    /// Cost: dimension + 1 function evaluations
    fn compute_forward(&self, t: f64, state: DVector<f64>) -> DMatrix<f64> {
        let offsets = self.compute_offsets(&state);
        let dimension = state.len();

        // Evaluate unperturbed dynamics
        let f0 = (self.dynamics_fn)(t, state.clone());

        // Initialize Jacobian matrix
        let mut jacobian = DMatrix::<f64>::zeros(dimension, dimension);

        // Compute each column: ∂f/∂x_i ≈ (f(x + h*e_i) - f(x)) / h
        for i in 0..dimension {
            let mut perturbed = state.clone();
            perturbed[i] += offsets[i];

            let fp = (self.dynamics_fn)(t, perturbed);
            jacobian.set_column(i, &((fp - &f0) / offsets[i]));
        }

        jacobian
    }

    /// Compute Jacobian using central finite differences.
    ///
    /// Cost: 2*dimension function evaluations (more accurate than forward differences)
    fn compute_central(&self, t: f64, state: DVector<f64>) -> DMatrix<f64> {
        let offsets = self.compute_offsets(&state);
        let dimension = state.len();

        // Initialize Jacobian matrix
        let mut jacobian = DMatrix::<f64>::zeros(dimension, dimension);

        // Compute each column: ∂f/∂x_i ≈ (f(x + h*e_i) - f(x - h*e_i)) / (2h)
        for i in 0..dimension {
            let mut perturbed_plus = state.clone();
            let mut perturbed_minus = state.clone();
            perturbed_plus[i] += offsets[i];
            perturbed_minus[i] -= offsets[i];

            let fp = (self.dynamics_fn)(t, perturbed_plus);
            let fm = (self.dynamics_fn)(t, perturbed_minus);
            jacobian.set_column(i, &((fp - fm) / (2.0 * offsets[i])));
        }

        jacobian
    }

    /// Compute Jacobian using backward finite differences.
    ///
    /// Cost: dimension + 1 function evaluations
    fn compute_backward(&self, t: f64, state: DVector<f64>) -> DMatrix<f64> {
        let offsets = self.compute_offsets(&state);
        let dimension = state.len();

        // Evaluate unperturbed dynamics
        let f0 = (self.dynamics_fn)(t, state.clone());

        // Initialize Jacobian matrix
        let mut jacobian = DMatrix::<f64>::zeros(dimension, dimension);

        // Compute each column: ∂f/∂x_i ≈ (f(x) - f(x - h*e_i)) / h
        for i in 0..dimension {
            let mut perturbed = state.clone();
            perturbed[i] -= offsets[i];

            let fm = (self.dynamics_fn)(t, perturbed);
            jacobian.set_column(i, &((f0.clone() - fm) / offsets[i]));
        }

        jacobian
    }
}

impl DJacobianProvider for DNumericalJacobian {
    fn compute(&self, t: f64, state: DVector<f64>) -> DMatrix<f64> {
        match self.method {
            DifferenceMethod::Forward => self.compute_forward(t, state),
            DifferenceMethod::Central => self.compute_central(t, state),
            DifferenceMethod::Backward => self.compute_backward(t, state),
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // Simple linear system: dx/dt = Ax where A = [[0, 1], [-1, 0]]
    // Analytical Jacobian is constant: A
    fn linear_dynamics_static(_t: f64, state: SVector<f64, 2>) -> SVector<f64, 2> {
        SVector::<f64, 2>::new(state[1], -state[0])
    }

    fn linear_dynamics_dynamic(_t: f64, state: DVector<f64>) -> DVector<f64> {
        DVector::from_vec(vec![state[1], -state[0]])
    }

    fn analytical_jacobian_static(_t: f64, _state: SVector<f64, 2>) -> SMatrix<f64, 2, 2> {
        SMatrix::<f64, 2, 2>::new(0.0, 1.0, -1.0, 0.0)
    }

    fn analytical_jacobian_dynamic(_t: f64, _state: DVector<f64>) -> DMatrix<f64> {
        DMatrix::<f64>::from_row_slice(2, 2, &[0.0, 1.0, -1.0, 0.0])
    }

    #[test]
    fn test_sanalytic_jacobian() {
        let provider = SAnalyticJacobian::new(Box::new(analytical_jacobian_static));
        let state = SVector::<f64, 2>::new(1.0, 0.5);
        let jacobian = provider.compute(0.0, state);

        let expected = SMatrix::<f64, 2, 2>::new(0.0, 1.0, -1.0, 0.0);

        assert_abs_diff_eq!(jacobian, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_danalytic_jacobian() {
        let provider = DAnalyticJacobian::new(Box::new(analytical_jacobian_dynamic));
        let state = DVector::from_vec(vec![1.0, 0.5]);
        let jacobian = provider.compute(0.0, state);

        let expected = DMatrix::<f64>::from_row_slice(2, 2, &[0.0, 1.0, -1.0, 0.0]);

        assert_abs_diff_eq!(jacobian, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_snumerical_jacobian_central() {
        let provider =
            SNumericalJacobian::central(Box::new(linear_dynamics_static)).with_fixed_offset(1e-6);

        let state = SVector::<f64, 2>::new(1.0, 0.5);
        let jacobian = provider.compute(0.0, state);

        let expected = SMatrix::<f64, 2, 2>::new(0.0, 1.0, -1.0, 0.0);

        assert_abs_diff_eq!(jacobian, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_snumerical_jacobian_forward() {
        let provider =
            SNumericalJacobian::forward(Box::new(linear_dynamics_static)).with_fixed_offset(1e-6);

        let state = SVector::<f64, 2>::new(1.0, 0.5);
        let jacobian = provider.compute(0.0, state);

        let expected = SMatrix::<f64, 2, 2>::new(0.0, 1.0, -1.0, 0.0);

        // Forward differences less accurate than central
        assert_abs_diff_eq!(jacobian, expected, epsilon = 1e-5);
    }

    #[test]
    fn test_snumerical_jacobian_backward() {
        let provider =
            SNumericalJacobian::backward(Box::new(linear_dynamics_static)).with_fixed_offset(1e-6);

        let state = SVector::<f64, 2>::new(1.0, 0.5);
        let jacobian = provider.compute(0.0, state);

        let expected = SMatrix::<f64, 2, 2>::new(0.0, 1.0, -1.0, 0.0);

        // Backward differences less accurate than central
        assert_abs_diff_eq!(jacobian, expected, epsilon = 1e-5);
    }

    #[test]
    fn test_dnumerical_jacobian_central() {
        let provider =
            DNumericalJacobian::central(Box::new(linear_dynamics_dynamic)).with_fixed_offset(1e-6);

        let state = DVector::from_vec(vec![1.0, 0.5]);
        let jacobian = provider.compute(0.0, state);

        let expected = DMatrix::<f64>::from_row_slice(2, 2, &[0.0, 1.0, -1.0, 0.0]);

        assert_abs_diff_eq!(jacobian, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_dnumerical_jacobian_forward() {
        let provider =
            DNumericalJacobian::forward(Box::new(linear_dynamics_dynamic)).with_fixed_offset(1e-6);

        let state = DVector::from_vec(vec![1.0, 0.5]);
        let jacobian = provider.compute(0.0, state);

        let expected = DMatrix::<f64>::from_row_slice(2, 2, &[0.0, 1.0, -1.0, 0.0]);

        // Forward differences less accurate than central
        assert_abs_diff_eq!(jacobian, expected, epsilon = 1e-5);
    }

    #[test]
    fn test_dnumerical_jacobian_backward() {
        let provider =
            DNumericalJacobian::backward(Box::new(linear_dynamics_dynamic)).with_fixed_offset(1e-6);

        let state = DVector::from_vec(vec![1.0, 0.5]);
        let jacobian = provider.compute(0.0, state);

        let expected = DMatrix::<f64>::from_row_slice(2, 2, &[0.0, 1.0, -1.0, 0.0]);

        // Backward differences less accurate than central
        assert_abs_diff_eq!(jacobian, expected, epsilon = 1e-5);
    }

    #[test]
    fn test_perturbation_strategies() {
        // Test adaptive perturbation
        let provider =
            SNumericalJacobian::central(Box::new(linear_dynamics_static)).with_adaptive(1.0, 1.0);

        let state = SVector::<f64, 2>::new(1.0, 0.5);
        let jacobian = provider.compute(0.0, state);

        let expected = SMatrix::<f64, 2, 2>::new(0.0, 1.0, -1.0, 0.0);
        assert_abs_diff_eq!(jacobian, expected, epsilon = 1e-6);

        // Test percentage perturbation
        let provider =
            SNumericalJacobian::central(Box::new(linear_dynamics_static)).with_percentage(1e-6);

        let jacobian = provider.compute(0.0, state);
        assert_abs_diff_eq!(jacobian, expected, epsilon = 1e-5);
    }
}
