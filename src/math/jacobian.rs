/*!
 * Jacobian computation for numerical analysis and integration.
 *
 * This module provides trait-based interfaces for computing Jacobian matrices,
 * with implementations for both analytical and numerical (finite difference) methods.
 */

#![allow(clippy::type_complexity)]

use nalgebra::{DMatrix, DVector, SMatrix, SVector};

use crate::utils::errors::BraheError;

/// Finite difference method for numerical Jacobian approximation.
///
/// Different methods trade off accuracy vs computational cost:
/// - **Forward**: O(h) error, S+1 function evaluations
/// - **Central**: O(h^2) error, 2S function evaluations (more accurate)
/// - **Backward**: O(h) error, S+1 function evaluations
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
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
    /// Automatic perturbation sizing: h_i = scale_factor * sqrt(ε) * max(|x_i|, min_value)
    ///
    /// Industry standard approach that adapts to state magnitude. The sqrt(ε) scaling
    /// (where ε is machine epsilon) optimally balances truncation vs roundoff error.
    Adaptive {
        /// Multiplier on sqrt(ε), typically 1.0
        scale_factor: f64,
        /// Minimum reference value (prevents tiny perturbations near zero)
        min_value: f64,
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
///
/// # Thread Safety
///
/// Requires `Send + Sync` for thread-safe integrator usage.
///
/// # Type Parameters
/// - `S`: State vector dimension
/// - `P`: Parameter vector dimension
pub trait SJacobianProvider<const S: usize, const P: usize>: Send + Sync {
    /// Compute the Jacobian matrix at the given time and state.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `params`: Optional parameter vector for parameter-dependent dynamics
    ///
    /// # Returns
    /// Jacobian matrix ∂f/∂x
    fn compute(
        &self,
        t: f64,
        state: &SVector<f64, S>,
        params: Option<&SVector<f64, P>>,
    ) -> Result<SMatrix<f64, S, S>, BraheError>;
}

/// Trait for dynamic-sized (runtime-known) Jacobian providers.
///
/// Implementors of this trait can compute Jacobian matrices for systems with
/// runtime-determined dimensionality.
pub trait DJacobianProvider: Send + Sync {
    /// Compute the Jacobian matrix at the given time and state.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `params`: Optional parameter vector for parameter-dependent dynamics
    ///
    /// # Returns
    /// Jacobian matrix ∂f/∂x
    fn compute(
        &self,
        t: f64,
        state: &DVector<f64>,
        params: Option<&DVector<f64>>,
    ) -> Result<DMatrix<f64>, BraheError>;
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
/// let jacobian_fn = |_t: f64, _state: &SVector<f64, 2>, _params: Option<&SVector<f64, 0>>| {
///     Ok(SMatrix::<f64, 2, 2>::new(
///         0.0,  1.0,
///        -1.0,  0.0
///     ))
/// };
///
/// let provider = SAnalyticJacobian::new(Box::new(jacobian_fn));
/// let state = SVector::<f64, 2>::new(1.0, 0.0);
/// let jacobian = provider.compute(0.0, &state, None).unwrap();
/// ```
pub struct SAnalyticJacobian<const S: usize, const P: usize> {
    jacobian_fn: Box<
        dyn Fn(
                f64,
                &SVector<f64, S>,
                Option<&SVector<f64, P>>,
            ) -> Result<SMatrix<f64, S, S>, BraheError>
            + Send
            + Sync,
    >,
}

impl<const S: usize, const P: usize> SAnalyticJacobian<S, P> {
    /// Create a new analytical Jacobian provider.
    ///
    /// # Arguments
    /// - `jacobian_fn`: Function that computes the analytical Jacobian
    pub fn new(
        jacobian_fn: Box<
            dyn Fn(
                    f64,
                    &SVector<f64, S>,
                    Option<&SVector<f64, P>>,
                ) -> Result<SMatrix<f64, S, S>, BraheError>
                + Send
                + Sync,
        >,
    ) -> Self {
        Self { jacobian_fn }
    }
}

impl<const S: usize, const P: usize> SJacobianProvider<S, P> for SAnalyticJacobian<S, P> {
    fn compute(
        &self,
        t: f64,
        state: &SVector<f64, S>,
        params: Option<&SVector<f64, P>>,
    ) -> Result<SMatrix<f64, S, S>, BraheError> {
        (self.jacobian_fn)(t, state, params)
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
/// let jacobian_fn = |_t: f64, _state: &DVector<f64>, _params: Option<&DVector<f64>>| {
///     Ok(DMatrix::<f64>::from_row_slice(2, 2, &[
///         0.0,  1.0,
///        -1.0,  0.0
///     ]))
/// };
///
/// let provider = DAnalyticJacobian::new(Box::new(jacobian_fn));
/// let state = DVector::from_vec(vec![1.0, 0.0]);
/// let jacobian = provider.compute(0.0, &state, None).unwrap();
/// ```
pub struct DAnalyticJacobian {
    jacobian_fn: Box<
        dyn Fn(f64, &DVector<f64>, Option<&DVector<f64>>) -> Result<DMatrix<f64>, BraheError>
            + Send
            + Sync,
    >,
}

impl DAnalyticJacobian {
    /// Create a new analytical Jacobian provider.
    ///
    /// # Arguments
    /// - `jacobian_fn`: Function that computes the analytical Jacobian
    pub fn new(
        jacobian_fn: Box<
            dyn Fn(f64, &DVector<f64>, Option<&DVector<f64>>) -> Result<DMatrix<f64>, BraheError>
                + Send
                + Sync,
        >,
    ) -> Self {
        Self { jacobian_fn }
    }
}

impl DJacobianProvider for DAnalyticJacobian {
    fn compute(
        &self,
        t: f64,
        state: &DVector<f64>,
        params: Option<&DVector<f64>>,
    ) -> Result<DMatrix<f64>, BraheError> {
        (self.jacobian_fn)(t, state, params)
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
/// let dynamics = |_t: f64, state: &SVector<f64, 2>, _params: Option<&SVector<f64, 0>>| {
///     Ok(SVector::<f64, 2>::new(state[1], -state[0]))
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
/// let jacobian = provider.compute(0.0, &state, None).unwrap();
/// ```
pub struct SNumericalJacobian<const S: usize, const P: usize> {
    dynamics_fn: Box<
        dyn Fn(
                f64,
                &SVector<f64, S>,
                Option<&SVector<f64, P>>,
            ) -> Result<SVector<f64, S>, BraheError>
            + Send
            + Sync,
    >,
    method: DifferenceMethod,
    perturbation: PerturbationStrategy,
}

impl<const S: usize, const P: usize> SNumericalJacobian<S, P> {
    /// Create a numerical Jacobian provider with default settings.
    ///
    /// Default: central differences with adaptive perturbations.
    pub fn new(
        dynamics_fn: Box<
            dyn Fn(
                    f64,
                    &SVector<f64, S>,
                    Option<&SVector<f64, P>>,
                ) -> Result<SVector<f64, S>, BraheError>
                + Send
                + Sync,
        >,
    ) -> Self {
        Self {
            dynamics_fn,
            method: DifferenceMethod::Central,
            perturbation: PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_value: 1.0,
            },
        }
    }

    /// Create with forward finite differences.
    pub fn forward(
        dynamics_fn: Box<
            dyn Fn(
                    f64,
                    &SVector<f64, S>,
                    Option<&SVector<f64, P>>,
                ) -> Result<SVector<f64, S>, BraheError>
                + Send
                + Sync,
        >,
    ) -> Self {
        Self {
            dynamics_fn,
            method: DifferenceMethod::Forward,
            perturbation: PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_value: 1.0,
            },
        }
    }

    /// Create with central finite differences.
    pub fn central(
        dynamics_fn: Box<
            dyn Fn(
                    f64,
                    &SVector<f64, S>,
                    Option<&SVector<f64, P>>,
                ) -> Result<SVector<f64, S>, BraheError>
                + Send
                + Sync,
        >,
    ) -> Self {
        Self::new(dynamics_fn)
    }

    /// Create with backward finite differences.
    pub fn backward(
        dynamics_fn: Box<
            dyn Fn(
                    f64,
                    &SVector<f64, S>,
                    Option<&SVector<f64, P>>,
                ) -> Result<SVector<f64, S>, BraheError>
                + Send
                + Sync,
        >,
    ) -> Self {
        Self {
            dynamics_fn,
            method: DifferenceMethod::Backward,
            perturbation: PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_value: 1.0,
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
    pub fn with_adaptive(mut self, scale_factor: f64, min_value: f64) -> Self {
        self.perturbation = PerturbationStrategy::Adaptive {
            scale_factor,
            min_value,
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
                min_value,
            } => {
                // Industry standard: h = sqrt(eps) * max(|x|, value)
                #[allow(non_snake_case)]
                let SQRT_EPS: f64 = f64::EPSILON.sqrt();
                let base_offset = scale_factor * SQRT_EPS;

                state.map(|x| base_offset * x.abs().max(min_value))
            }
            PerturbationStrategy::Fixed(offset) => SVector::from_element(offset),
            PerturbationStrategy::Percentage(pct) => state.map(|x| x.abs() * pct),
        }
    }

    /// Compute Jacobian using forward finite differences.
    ///
    /// Cost: S + 1 function evaluations
    fn compute_forward(
        &self,
        t: f64,
        state: &SVector<f64, S>,
        params: Option<&SVector<f64, P>>,
    ) -> Result<SMatrix<f64, S, S>, BraheError> {
        let offsets = self.compute_offsets(state);

        // Evaluate unperturbed dynamics
        let f0 = (self.dynamics_fn)(t, state, params)?;

        // Initialize Jacobian matrix
        let mut jacobian = SMatrix::<f64, S, S>::zeros();

        // Compute each column: ∂f/∂x_i ≈ (f(x + h*e_i) - f(x)) / h
        for i in 0..S {
            let mut perturbed = *state;
            perturbed[i] += offsets[i];

            let fp = (self.dynamics_fn)(t, &perturbed, params)?;
            jacobian.set_column(i, &((fp - f0) / offsets[i]));
        }

        Ok(jacobian)
    }

    /// Compute Jacobian using central finite differences.
    ///
    /// Cost: 2S function evaluations (more accurate than forward differences)
    fn compute_central(
        &self,
        t: f64,
        state: &SVector<f64, S>,
        params: Option<&SVector<f64, P>>,
    ) -> Result<SMatrix<f64, S, S>, BraheError> {
        let offsets = self.compute_offsets(state);

        // Initialize Jacobian matrix
        let mut jacobian = SMatrix::<f64, S, S>::zeros();

        // Compute each column: ∂f/∂x_i ≈ (f(x + h*e_i) - f(x - h*e_i)) / (2h)
        for i in 0..S {
            let mut perturbed_plus = *state;
            let mut perturbed_minus = *state;
            perturbed_plus[i] += offsets[i];
            perturbed_minus[i] -= offsets[i];

            let fp = (self.dynamics_fn)(t, &perturbed_plus, params)?;
            let fm = (self.dynamics_fn)(t, &perturbed_minus, params)?;
            jacobian.set_column(i, &((fp - fm) / (2.0 * offsets[i])));
        }

        Ok(jacobian)
    }

    /// Compute Jacobian using backward finite differences.
    ///
    /// Cost: S + 1 function evaluations
    fn compute_backward(
        &self,
        t: f64,
        state: &SVector<f64, S>,
        params: Option<&SVector<f64, P>>,
    ) -> Result<SMatrix<f64, S, S>, BraheError> {
        let offsets = self.compute_offsets(state);

        // Evaluate unperturbed dynamics
        let f0 = (self.dynamics_fn)(t, state, params)?;

        // Initialize Jacobian matrix
        let mut jacobian = SMatrix::<f64, S, S>::zeros();

        // Compute each column: ∂f/∂x_i ≈ (f(x) - f(x - h*e_i)) / h
        for i in 0..S {
            let mut perturbed = *state;
            perturbed[i] -= offsets[i];

            let fm = (self.dynamics_fn)(t, &perturbed, params)?;
            jacobian.set_column(i, &((f0 - fm) / offsets[i]));
        }

        Ok(jacobian)
    }
}

impl<const S: usize, const P: usize> SJacobianProvider<S, P> for SNumericalJacobian<S, P> {
    fn compute(
        &self,
        t: f64,
        state: &SVector<f64, S>,
        params: Option<&SVector<f64, P>>,
    ) -> Result<SMatrix<f64, S, S>, BraheError> {
        match self.method {
            DifferenceMethod::Forward => self.compute_forward(t, state, params),
            DifferenceMethod::Central => self.compute_central(t, state, params),
            DifferenceMethod::Backward => self.compute_backward(t, state, params),
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
/// let dynamics = |_t: f64, state: &DVector<f64>, _params: Option<&DVector<f64>>| {
///     Ok(DVector::from_vec(vec![state[1], -state[0]]))
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
/// let jacobian = provider.compute(0.0, &state, None).unwrap();
/// ```
pub struct DNumericalJacobian {
    dynamics_fn: Box<
        dyn Fn(f64, &DVector<f64>, Option<&DVector<f64>>) -> Result<DVector<f64>, BraheError>
            + Send
            + Sync,
    >,
    method: DifferenceMethod,
    perturbation: PerturbationStrategy,
}

impl DNumericalJacobian {
    /// Create a numerical Jacobian provider with default settings.
    ///
    /// Default: central differences with adaptive perturbations.
    pub fn new(
        dynamics_fn: Box<
            dyn Fn(f64, &DVector<f64>, Option<&DVector<f64>>) -> Result<DVector<f64>, BraheError>
                + Send
                + Sync,
        >,
    ) -> Self {
        Self {
            dynamics_fn,
            method: DifferenceMethod::Central,
            perturbation: PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_value: 1.0,
            },
        }
    }

    /// Create with forward finite differences.
    pub fn forward(
        dynamics_fn: Box<
            dyn Fn(f64, &DVector<f64>, Option<&DVector<f64>>) -> Result<DVector<f64>, BraheError>
                + Send
                + Sync,
        >,
    ) -> Self {
        Self {
            dynamics_fn,
            method: DifferenceMethod::Forward,
            perturbation: PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_value: 1.0,
            },
        }
    }

    /// Create with central finite differences.
    pub fn central(
        dynamics_fn: Box<
            dyn Fn(f64, &DVector<f64>, Option<&DVector<f64>>) -> Result<DVector<f64>, BraheError>
                + Send
                + Sync,
        >,
    ) -> Self {
        Self::new(dynamics_fn)
    }

    /// Create with backward finite differences.
    pub fn backward(
        dynamics_fn: Box<
            dyn Fn(f64, &DVector<f64>, Option<&DVector<f64>>) -> Result<DVector<f64>, BraheError>
                + Send
                + Sync,
        >,
    ) -> Self {
        Self {
            dynamics_fn,
            method: DifferenceMethod::Backward,
            perturbation: PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_value: 1.0,
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
    pub fn with_adaptive(mut self, scale_factor: f64, min_value: f64) -> Self {
        self.perturbation = PerturbationStrategy::Adaptive {
            scale_factor,
            min_value,
        };
        self
    }

    /// Set the difference method.
    pub fn with_method(mut self, method: DifferenceMethod) -> Self {
        self.method = method;
        self
    }
}

impl DJacobianProvider for DNumericalJacobian {
    fn compute(
        &self,
        t: f64,
        state: &DVector<f64>,
        params: Option<&DVector<f64>>,
    ) -> Result<DMatrix<f64>, BraheError> {
        numerical_jacobian(
            |x| (self.dynamics_fn)(t, x, params),
            state,
            self.method,
            self.perturbation,
        )
    }
}

/// Compute perturbation offsets for each component of `x` under a
/// [`PerturbationStrategy`].
pub fn compute_perturbation_offsets(
    x: &DVector<f64>,
    perturbation: PerturbationStrategy,
) -> DVector<f64> {
    match perturbation {
        PerturbationStrategy::Adaptive {
            scale_factor,
            min_value,
        } => {
            // Industry standard: h = sqrt(eps) * max(|x|, value)
            let sqrt_eps: f64 = f64::EPSILON.sqrt();
            let base_offset = scale_factor * sqrt_eps;

            x.map(|v| base_offset * v.abs().max(min_value))
        }
        PerturbationStrategy::Fixed(offset) => DVector::from_element(x.len(), offset),
        PerturbationStrategy::Percentage(pct) => x.map(|v| v.abs() * pct),
    }
}

/// Compute the Jacobian ∂f/∂x of an arbitrary vector-valued function via
/// finite differences.
///
/// The function may be rectangular: the output dimension `m` is inferred from
/// the function's output and need not equal the input dimension `n = x.len()`.
/// This is the shared finite-difference engine behind [`DNumericalJacobian`]
/// (dynamics Jacobians, n×n), `DNumericalSensitivity` (parameter
/// sensitivities, n×p, differentiated with respect to the parameter vector),
/// and the estimation module's measurement Jacobians (m×n).
///
/// # Arguments
///
/// * `f` - Function to differentiate. May fail; errors propagate to the caller.
/// * `x` - Point at which to evaluate the Jacobian (also the perturbed vector)
/// * `method` - Finite difference method
/// * `perturbation` - Perturbation sizing strategy
///
/// # Returns
///
/// The m×n Jacobian matrix, or an error if `f` fails or returns vectors of
/// inconsistent dimensions.
pub fn numerical_jacobian<F>(
    f: F,
    x: &DVector<f64>,
    method: DifferenceMethod,
    perturbation: PerturbationStrategy,
) -> Result<DMatrix<f64>, BraheError>
where
    F: Fn(&DVector<f64>) -> Result<DVector<f64>, BraheError>,
{
    let n = x.len();
    let offsets = compute_perturbation_offsets(x, perturbation);

    let mut output_dim: Option<usize> = None;
    let check_dim = |v: &DVector<f64>, dim: &mut Option<usize>| -> Result<(), BraheError> {
        match dim {
            None => {
                *dim = Some(v.len());
                Ok(())
            }
            Some(expected) if v.len() == *expected => Ok(()),
            Some(expected) => Err(BraheError::Error(format!(
                "Function returned inconsistent output dimension during finite \
                 differencing: got {}, expected {}",
                v.len(),
                expected
            ))),
        }
    };

    let mut columns: Vec<DVector<f64>> = Vec::with_capacity(n);
    match method {
        DifferenceMethod::Central => {
            // ∂f/∂x_j ≈ (f(x + h*e_j) - f(x - h*e_j)) / (2h)
            for j in 0..n {
                let mut x_plus = x.clone();
                x_plus[j] += offsets[j];
                let fp = f(&x_plus)?;
                check_dim(&fp, &mut output_dim)?;

                let mut x_minus = x.clone();
                x_minus[j] -= offsets[j];
                let fm = f(&x_minus)?;
                check_dim(&fm, &mut output_dim)?;

                columns.push((fp - fm) / (2.0 * offsets[j]));
            }
        }
        DifferenceMethod::Forward => {
            // ∂f/∂x_j ≈ (f(x + h*e_j) - f(x)) / h
            let f0 = f(x)?;
            check_dim(&f0, &mut output_dim)?;
            for j in 0..n {
                let mut x_plus = x.clone();
                x_plus[j] += offsets[j];
                let fp = f(&x_plus)?;
                check_dim(&fp, &mut output_dim)?;

                columns.push((fp - &f0) / offsets[j]);
            }
        }
        DifferenceMethod::Backward => {
            // ∂f/∂x_j ≈ (f(x) - f(x - h*e_j)) / h
            let f0 = f(x)?;
            check_dim(&f0, &mut output_dim)?;
            for j in 0..n {
                let mut x_minus = x.clone();
                x_minus[j] -= offsets[j];
                let fm = f(&x_minus)?;
                check_dim(&fm, &mut output_dim)?;

                columns.push((&f0 - fm) / offsets[j]);
            }
        }
    }

    let m = output_dim.unwrap_or(0);
    let mut jacobian = DMatrix::<f64>::zeros(m, n);
    for (j, column) in columns.iter().enumerate() {
        jacobian.set_column(j, column);
    }
    Ok(jacobian)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // Simple linear system: dx/dt = Ax where A = [[0, 1], [-1, 0]]
    // Analytical Jacobian is constant: A
    fn linear_dynamics_static(
        _t: f64,
        state: &SVector<f64, 2>,
        _params: Option<&SVector<f64, 0>>,
    ) -> Result<SVector<f64, 2>, BraheError> {
        Ok(SVector::<f64, 2>::new(state[1], -state[0]))
    }

    fn linear_dynamics_dynamic(
        _t: f64,
        state: &DVector<f64>,
        _params: Option<&DVector<f64>>,
    ) -> Result<DVector<f64>, BraheError> {
        Ok(DVector::from_vec(vec![state[1], -state[0]]))
    }

    fn analytical_jacobian_static(
        _t: f64,
        _state: &SVector<f64, 2>,
        _params: Option<&SVector<f64, 0>>,
    ) -> Result<SMatrix<f64, 2, 2>, BraheError> {
        Ok(SMatrix::<f64, 2, 2>::new(0.0, 1.0, -1.0, 0.0))
    }

    fn analytical_jacobian_dynamic(
        _t: f64,
        _state: &DVector<f64>,
        _params: Option<&DVector<f64>>,
    ) -> Result<DMatrix<f64>, BraheError> {
        Ok(DMatrix::<f64>::from_row_slice(2, 2, &[0.0, 1.0, -1.0, 0.0]))
    }

    #[test]
    fn test_sanalytic_jacobian() {
        let provider = SAnalyticJacobian::new(Box::new(analytical_jacobian_static));
        let state = SVector::<f64, 2>::new(1.0, 0.5);
        let jacobian = provider.compute(0.0, &state, None).unwrap();

        let expected = SMatrix::<f64, 2, 2>::new(0.0, 1.0, -1.0, 0.0);

        assert_abs_diff_eq!(jacobian, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_danalytic_jacobian() {
        let provider = DAnalyticJacobian::new(Box::new(analytical_jacobian_dynamic));
        let state = DVector::from_vec(vec![1.0, 0.5]);
        let jacobian = provider.compute(0.0, &state, None).unwrap();

        let expected = DMatrix::<f64>::from_row_slice(2, 2, &[0.0, 1.0, -1.0, 0.0]);

        assert_abs_diff_eq!(jacobian, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_snumerical_jacobian_central() {
        let provider =
            SNumericalJacobian::central(Box::new(linear_dynamics_static)).with_fixed_offset(1e-6);

        let state = SVector::<f64, 2>::new(1.0, 0.5);
        let jacobian = provider.compute(0.0, &state, None).unwrap();

        let expected = SMatrix::<f64, 2, 2>::new(0.0, 1.0, -1.0, 0.0);

        assert_abs_diff_eq!(jacobian, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_snumerical_jacobian_forward() {
        let provider =
            SNumericalJacobian::forward(Box::new(linear_dynamics_static)).with_fixed_offset(1e-6);

        let state = SVector::<f64, 2>::new(1.0, 0.5);
        let jacobian = provider.compute(0.0, &state, None).unwrap();

        let expected = SMatrix::<f64, 2, 2>::new(0.0, 1.0, -1.0, 0.0);

        // Forward differences less accurate than central
        assert_abs_diff_eq!(jacobian, expected, epsilon = 1e-5);
    }

    #[test]
    fn test_snumerical_jacobian_backward() {
        let provider =
            SNumericalJacobian::backward(Box::new(linear_dynamics_static)).with_fixed_offset(1e-6);

        let state = SVector::<f64, 2>::new(1.0, 0.5);
        let jacobian = provider.compute(0.0, &state, None).unwrap();

        let expected = SMatrix::<f64, 2, 2>::new(0.0, 1.0, -1.0, 0.0);

        // Backward differences less accurate than central
        assert_abs_diff_eq!(jacobian, expected, epsilon = 1e-5);
    }

    #[test]
    fn test_dnumerical_jacobian_central() {
        let provider =
            DNumericalJacobian::central(Box::new(linear_dynamics_dynamic)).with_fixed_offset(1e-6);

        let state = DVector::from_vec(vec![1.0, 0.5]);
        let jacobian = provider.compute(0.0, &state, None).unwrap();

        let expected = DMatrix::<f64>::from_row_slice(2, 2, &[0.0, 1.0, -1.0, 0.0]);

        assert_abs_diff_eq!(jacobian, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_dnumerical_jacobian_forward() {
        let provider =
            DNumericalJacobian::forward(Box::new(linear_dynamics_dynamic)).with_fixed_offset(1e-6);

        let state = DVector::from_vec(vec![1.0, 0.5]);
        let jacobian = provider.compute(0.0, &state, None).unwrap();

        let expected = DMatrix::<f64>::from_row_slice(2, 2, &[0.0, 1.0, -1.0, 0.0]);

        // Forward differences less accurate than central
        assert_abs_diff_eq!(jacobian, expected, epsilon = 1e-5);
    }

    #[test]
    fn test_dnumerical_jacobian_backward() {
        let provider =
            DNumericalJacobian::backward(Box::new(linear_dynamics_dynamic)).with_fixed_offset(1e-6);

        let state = DVector::from_vec(vec![1.0, 0.5]);
        let jacobian = provider.compute(0.0, &state, None).unwrap();

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
        let jacobian = provider.compute(0.0, &state, None).unwrap();

        let expected = SMatrix::<f64, 2, 2>::new(0.0, 1.0, -1.0, 0.0);
        assert_abs_diff_eq!(jacobian, expected, epsilon = 1e-6);

        // Test percentage perturbation
        let provider =
            SNumericalJacobian::central(Box::new(linear_dynamics_static)).with_percentage(1e-6);

        let jacobian = provider.compute(0.0, &state, None).unwrap();
        assert_abs_diff_eq!(jacobian, expected, epsilon = 1e-5);
    }

    #[test]
    fn test_numerical_jacobian_rectangular() {
        // f: R^3 -> R^2, f(x) = [x0 + 2*x1, x1 * x2]
        // Jacobian: [[1, 2, 0], [0, x2, x1]]
        let f = |x: &DVector<f64>| -> Result<DVector<f64>, BraheError> {
            Ok(DVector::from_vec(vec![x[0] + 2.0 * x[1], x[1] * x[2]]))
        };
        let x = DVector::from_vec(vec![1.0, 2.0, 3.0]);

        for method in [
            DifferenceMethod::Central,
            DifferenceMethod::Forward,
            DifferenceMethod::Backward,
        ] {
            let jac = numerical_jacobian(
                f,
                &x,
                method,
                PerturbationStrategy::Adaptive {
                    scale_factor: 1.0,
                    min_value: 1.0,
                },
            )
            .unwrap();

            assert_eq!(jac.nrows(), 2);
            assert_eq!(jac.ncols(), 3);
            let expected = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 0.0, 0.0, 3.0, 2.0]);
            assert_abs_diff_eq!(jac, expected, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_numerical_jacobian_propagates_function_error() {
        let f = |_x: &DVector<f64>| -> Result<DVector<f64>, BraheError> {
            Err(BraheError::Error("function failure".to_string()))
        };
        let x = DVector::from_vec(vec![1.0, 2.0]);

        let result = numerical_jacobian(
            f,
            &x,
            DifferenceMethod::Central,
            PerturbationStrategy::Fixed(1e-6),
        );
        match result {
            Err(e) => assert!(e.to_string().contains("function failure")),
            Ok(_) => panic!("Expected the function error to propagate"),
        }
    }

    #[test]
    fn test_numerical_jacobian_inconsistent_output_dim_errors() {
        // Function whose output length depends on the sign of a perturbation
        let f = |x: &DVector<f64>| -> Result<DVector<f64>, BraheError> {
            if x[0] > 1.0 {
                Ok(DVector::from_vec(vec![x[0], x[1]]))
            } else {
                Ok(DVector::from_vec(vec![x[0]]))
            }
        };
        let x = DVector::from_vec(vec![1.0, 2.0]);

        let result = numerical_jacobian(
            f,
            &x,
            DifferenceMethod::Central,
            PerturbationStrategy::Fixed(0.5),
        );
        match result {
            Err(e) => assert!(e.to_string().contains("inconsistent output dimension")),
            Ok(_) => panic!("Expected inconsistent-dimension error"),
        }
    }
}
