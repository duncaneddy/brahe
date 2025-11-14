/*!
Defines the `NumericalIntegrator` trait and its implementations, which provides a common interface
for numerical integration routines.
 */

use nalgebra::{SMatrix, SVector};

use crate::integrators::config::AdaptiveStepResult;

/// Trait defining interface for fixed-step numerical integration methods.
///
/// Provides basic integration functionality with fixed timesteps.
/// All numerical integrators must implement this trait.
pub trait FixedStepIntegrator<const S: usize> {
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

/// Trait defining interface for adaptive-step numerical integration methods.
///
/// Provides automatic step size control based on embedded error estimation.
/// Typically implemented by embedded Runge-Kutta methods (RKF45, DP54, etc.).
pub trait AdaptiveStepIntegrator<const S: usize> {
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
    /// AdaptiveStepResult containing new state, actual dt used, error estimate, and suggested next dt
    fn step(
        &self,
        t: f64,
        state: SVector<f64, S>,
        dt: f64,
        abs_tol: f64,
        rel_tol: f64,
    ) -> AdaptiveStepResult<S>;

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
// Variational Matrix (Jacobian) Computation Configuration
// ============================================================================

/// Finite difference method for numerical Jacobian approximation.
///
/// Different methods trade off accuracy vs computational cost:
/// - **Forward**: O(h) error, S+1 function evaluations
/// - **Central**: O(h²) error, 2S function evaluations (more accurate)
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

/// Configuration for variational matrix (Jacobian) computation via finite differences.
///
/// Provides a unified interface for computing state transition matrices with sensible defaults
/// (central differences with adaptive perturbation sizing) and full configurability for
/// advanced users.
///
/// # Examples
///
/// ```rust
/// use brahe::integrators::{VarmatConfig, DifferenceMethod, PerturbationStrategy};
/// use nalgebra::SVector;
///
/// // Simple: use smart defaults (central differences, adaptive perturbations)
/// let config = VarmatConfig::default();
///
/// // Custom: forward differences with fixed offset
/// let config = VarmatConfig::forward().with_fixed_offset(1e-6);
///
/// // Power user: central differences with custom adaptive scaling
/// let config = VarmatConfig {
///     method: DifferenceMethod::Central,
///     perturbation: PerturbationStrategy::Adaptive {
///         scale_factor: 2.0,
///         min_threshold: 0.1,
///     },
/// };
///
/// // Compute Jacobian
/// let dynamics = |t: f64, state: SVector<f64, 6>| -> SVector<f64, 6> {
///     // ... your dynamics here
///     state
/// };
///
/// let t = 0.0;
/// let state = SVector::<f64, 6>::zeros();
/// let jacobian = config.compute(t, state, &dynamics);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct VarmatConfig {
    /// Finite difference method (Forward or Central)
    pub method: DifferenceMethod,

    /// Perturbation sizing strategy
    pub perturbation: PerturbationStrategy,
}

impl Default for VarmatConfig {
    /// Default configuration: central differences with adaptive perturbation sizing.
    ///
    /// This provides the best accuracy-to-cost ratio for most applications.
    fn default() -> Self {
        Self {
            method: DifferenceMethod::Central,
            perturbation: PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_threshold: 1.0,
            },
        }
    }
}

impl VarmatConfig {
    /// Create configuration with forward differences and adaptive perturbations.
    pub fn forward() -> Self {
        Self {
            method: DifferenceMethod::Forward,
            perturbation: PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_threshold: 1.0,
            },
        }
    }

    /// Create configuration with central differences and adaptive perturbations.
    pub fn central() -> Self {
        Self::default()
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
    fn compute_offsets<const S: usize>(&self, state: &SVector<f64, S>) -> SVector<f64, S> {
        match &self.perturbation {
            PerturbationStrategy::Adaptive {
                scale_factor,
                min_threshold,
            } => {
                // Industry standard: h = sqrt(eps) * max(|x|, threshold)
                #[allow(non_snake_case)]
                let SQRT_EPS: f64 = f64::EPSILON.sqrt();
                let base_offset = *scale_factor * SQRT_EPS;

                state.map(|x| base_offset * x.abs().max(*min_threshold))
            }
            PerturbationStrategy::Fixed(offset) => SVector::from_element(*offset),
            PerturbationStrategy::Percentage(pct) => state.map(|x| x.abs() * pct),
        }
    }

    /// Compute Jacobian using forward finite differences.
    ///
    /// Cost: S + 1 function evaluations
    fn compute_forward<const S: usize>(
        &self,
        t: f64,
        state: SVector<f64, S>,
        f: &dyn Fn(f64, SVector<f64, S>) -> SVector<f64, S>,
        offsets: &SVector<f64, S>,
    ) -> SMatrix<f64, S, S> {
        // Evaluate unperturbed dynamics
        let f0 = f(t, state);

        // Initialize Jacobian matrix
        let mut jacobian = SMatrix::<f64, S, S>::zeros();

        // Compute each column: ∂f/∂x_i ≈ (f(x + h*e_i) - f(x)) / h
        for i in 0..S {
            let mut perturbed = state;
            perturbed[i] += offsets[i];

            let fp = f(t, perturbed);
            jacobian.set_column(i, &((fp - f0) / offsets[i]));
        }

        jacobian
    }

    /// Compute Jacobian using central finite differences.
    ///
    /// Cost: 2S function evaluations (more accurate than forward differences)
    fn compute_central<const S: usize>(
        &self,
        t: f64,
        state: SVector<f64, S>,
        f: &dyn Fn(f64, SVector<f64, S>) -> SVector<f64, S>,
        offsets: &SVector<f64, S>,
    ) -> SMatrix<f64, S, S> {
        // Initialize Jacobian matrix
        let mut jacobian = SMatrix::<f64, S, S>::zeros();

        // Compute each column: ∂f/∂x_i ≈ (f(x + h*e_i) - f(x - h*e_i)) / (2h)
        for i in 0..S {
            let mut perturbed_plus = state;
            let mut perturbed_minus = state;
            perturbed_plus[i] += offsets[i];
            perturbed_minus[i] -= offsets[i];

            let fp = f(t, perturbed_plus);
            let fm = f(t, perturbed_minus);
            jacobian.set_column(i, &((fp - fm) / (2.0 * offsets[i])));
        }

        jacobian
    }

    /// Compute the state transition matrix (Jacobian) using configured method.
    ///
    /// This is the main entry point for Jacobian computation. It automatically
    /// computes appropriate perturbation sizes and applies the selected finite
    /// difference method.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: State vector at time t
    /// - `f`: State derivative function (dynamics)
    ///
    /// # Returns
    /// Jacobian matrix ∂f/∂x approximated via finite differences
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::integrators::VarmatConfig;
    /// use nalgebra::SVector;
    ///
    /// let config = VarmatConfig::default();
    /// let dynamics = |t: f64, state: SVector<f64, 2>| {
    ///     SVector::<f64, 2>::new(state[1], -state[0])  // Simple oscillator
    /// };
    ///
    /// let state = SVector::<f64, 2>::new(1.0, 0.0);
    /// let jacobian = config.compute(0.0, state, &dynamics);
    /// ```
    pub fn compute<const S: usize>(
        &self,
        t: f64,
        state: SVector<f64, S>,
        f: &dyn Fn(f64, SVector<f64, S>) -> SVector<f64, S>,
    ) -> SMatrix<f64, S, S> {
        let offsets = self.compute_offsets(&state);

        match self.method {
            DifferenceMethod::Forward => self.compute_forward(t, state, f, &offsets),
            DifferenceMethod::Central => self.compute_central(t, state, f, &offsets),
        }
    }
}

/// Compute variational matrix with custom perturbation offsets.
///
/// Power user function that allows complete control over perturbation sizes.
/// Useful when you have domain-specific knowledge about appropriate step sizes
/// for different state components (e.g., position vs velocity in orbital mechanics).
///
/// # Arguments
/// - `t`: Current time
/// - `state`: State vector at time t
/// - `f`: State derivative function (dynamics)
/// - `method`: Finite difference method to use
/// - `offsets`: Custom perturbation size for each state component
///
/// # Returns
/// Jacobian matrix ∂f/∂x
///
/// # Examples
///
/// ```rust
/// use brahe::integrators::{varmat_custom, DifferenceMethod};
/// use nalgebra::SVector;
///
/// let dynamics = |t: f64, state: SVector<f64, 6>| state;
/// let state = SVector::<f64, 6>::zeros();
///
/// // Different perturbations for position (first 3) vs velocity (last 3)
/// let offsets = SVector::<f64, 6>::new(
///     1.0, 1.0, 1.0,      // position: 1m
///     0.001, 0.001, 0.001 // velocity: 1mm/s
/// );
///
/// let jacobian = varmat_custom(
///     0.0, state, &dynamics,
///     DifferenceMethod::Central,
///     offsets
/// );
/// ```
pub fn varmat_custom<const S: usize>(
    t: f64,
    state: SVector<f64, S>,
    f: &dyn Fn(f64, SVector<f64, S>) -> SVector<f64, S>,
    method: DifferenceMethod,
    offsets: SVector<f64, S>,
) -> SMatrix<f64, S, S> {
    let config = VarmatConfig {
        method,
        perturbation: PerturbationStrategy::Fixed(0.0), // Dummy, won't be used
    };

    match method {
        DifferenceMethod::Forward => config.compute_forward(t, state, f, &offsets),
        DifferenceMethod::Central => config.compute_central(t, state, f, &offsets),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use nalgebra::{SMatrix, SVector};

    // ========================================================================
    // Test: Analytical Jacobian comparison
    // ========================================================================

    /// Linear system: dx/dt = A*x has exact Jacobian = A
    #[test]
    fn test_varmat_linear_system() {
        // Define linear system: x' = [0 1; -1 0] * x (simple harmonic oscillator)
        let dynamics = |_t: f64, state: SVector<f64, 2>| -> SVector<f64, 2> {
            SVector::<f64, 2>::new(state[1], -state[0])
        };

        // Analytical Jacobian
        let analytical = SMatrix::<f64, 2, 2>::new(0.0, 1.0, -1.0, 0.0);

        let state = SVector::<f64, 2>::new(1.0, 0.5);

        // Test central differences (should be very accurate)
        let config = VarmatConfig::central().with_fixed_offset(1e-6);
        let jacobian = config.compute(0.0, state, &dynamics);
        assert_abs_diff_eq!(jacobian, analytical, epsilon = 1e-8);

        // Test forward differences (less accurate but still good)
        let config = VarmatConfig::forward().with_fixed_offset(1e-6);
        let jacobian = config.compute(0.0, state, &dynamics);
        assert_abs_diff_eq!(jacobian, analytical, epsilon = 1e-5);
    }

    // ========================================================================
    // Test: Convergence order
    // ========================================================================

    /// Verify forward differences are O(h) and central are O(h²)
    #[test]
    fn test_varmat_convergence_order() {
        // Nonlinear system: x' = [x1^2, x0*x1]
        let dynamics = |_t: f64, state: SVector<f64, 2>| -> SVector<f64, 2> {
            SVector::<f64, 2>::new(state[1].powi(2), state[0] * state[1])
        };

        // Analytical Jacobian: [[0, 2*x1], [x1, x0]]
        let state = SVector::<f64, 2>::new(2.0, 3.0);
        let analytical = SMatrix::<f64, 2, 2>::new(0.0, 6.0, 3.0, 2.0);

        // Test forward differences with decreasing step sizes
        let forward_steps = [1e-2, 1e-3, 1e-4];
        let mut forward_errors = Vec::new();

        for &h in &forward_steps {
            let config = VarmatConfig::forward().with_fixed_offset(h);
            let jacobian = config.compute(0.0, state, &dynamics);
            let error = (jacobian - analytical).norm();
            forward_errors.push(error);
        }

        // Verify forward differences show O(h) convergence
        let ratio_1 = forward_errors[0] / forward_errors[1];
        let ratio_2 = forward_errors[1] / forward_errors[2];
        assert!(ratio_1 > 5.0 && ratio_1 < 15.0); // ~10x reduction
        assert!(ratio_2 > 5.0 && ratio_2 < 15.0);

        // Test that central differences are significantly more accurate than forward
        // at the same step size
        for &h in &[1e-2, 1e-3, 1e-4] {
            let forward_config = VarmatConfig::forward().with_fixed_offset(h);
            let central_config = VarmatConfig::central().with_fixed_offset(h);

            let forward_jac = forward_config.compute(0.0, state, &dynamics);
            let central_jac = central_config.compute(0.0, state, &dynamics);

            let forward_error = (forward_jac - analytical).norm();
            let central_error = (central_jac - analytical).norm();

            // Central should be at least 10x more accurate (often much better)
            assert!(central_error < forward_error / 10.0);
        }
    }

    // ========================================================================
    // Test: Adaptive perturbation strategy
    // ========================================================================

    #[test]
    fn test_varmat_adaptive_perturbation() {
        let config = VarmatConfig::default(); // Uses adaptive strategy

        // Test state with varying magnitudes
        let state = SVector::<f64, 4>::new(1000.0, 10.0, 0.1, 0.001);
        let offsets = config.compute_offsets(&state);

        // All offsets should be proportional to state magnitude
        // h_i = sqrt(eps) * max(|x_i|, threshold)
        #[allow(non_snake_case)]
        let SQRT_EPS: f64 = f64::EPSILON.sqrt();
        const THRESHOLD: f64 = 1.0;

        assert_abs_diff_eq!(offsets[0], SQRT_EPS * 1000.0, epsilon = 1e-10);
        assert_abs_diff_eq!(offsets[1], SQRT_EPS * 10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(offsets[2], SQRT_EPS * THRESHOLD, epsilon = 1e-10); // Falls back to threshold
        assert_abs_diff_eq!(offsets[3], SQRT_EPS * THRESHOLD, epsilon = 1e-10); // Falls back to threshold

        // Test with custom scale factor
        let config = VarmatConfig::default().with_adaptive(2.0, 0.5);
        let offsets = config.compute_offsets(&state);

        assert_abs_diff_eq!(offsets[0], 2.0 * SQRT_EPS * 1000.0, epsilon = 1e-10);
        assert_abs_diff_eq!(offsets[1], 2.0 * SQRT_EPS * 10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(offsets[2], 2.0 * SQRT_EPS * 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(offsets[3], 2.0 * SQRT_EPS * 0.5, epsilon = 1e-10);
    }

    // ========================================================================
    // Test: Fixed and percentage perturbations
    // ========================================================================

    #[test]
    fn test_varmat_fixed_perturbation() {
        let config = VarmatConfig::default().with_fixed_offset(0.001);
        let state = SVector::<f64, 3>::new(100.0, 1.0, 0.01);
        let offsets = config.compute_offsets(&state);

        // All offsets should be identical
        assert_eq!(offsets[0], 0.001);
        assert_eq!(offsets[1], 0.001);
        assert_eq!(offsets[2], 0.001);
    }

    #[test]
    fn test_varmat_percentage_perturbation() {
        let config = VarmatConfig::default().with_percentage(0.01); // 1%
        let state = SVector::<f64, 3>::new(100.0, 10.0, 1.0);
        let offsets = config.compute_offsets(&state);

        // Each offset should be 1% of the state component
        assert_abs_diff_eq!(offsets[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(offsets[1], 0.1, epsilon = 1e-10);
        assert_abs_diff_eq!(offsets[2], 0.01, epsilon = 1e-10);
    }

    // ========================================================================
    // Test: Builder methods
    // ========================================================================

    #[test]
    fn test_varmat_config_builders() {
        // Test forward()
        let config = VarmatConfig::forward();
        assert_eq!(config.method, DifferenceMethod::Forward);
        assert!(matches!(
            config.perturbation,
            PerturbationStrategy::Adaptive { .. }
        ));

        // Test central()
        let config = VarmatConfig::central();
        assert_eq!(config.method, DifferenceMethod::Central);

        // Test chaining
        let config = VarmatConfig::forward()
            .with_fixed_offset(1e-6)
            .with_method(DifferenceMethod::Central);
        assert_eq!(config.method, DifferenceMethod::Central);
        assert_eq!(config.perturbation, PerturbationStrategy::Fixed(1e-6));
    }

    // ========================================================================
    // Test: Custom offsets
    // ========================================================================

    #[test]
    fn test_varmat_custom_offsets() {
        let dynamics = |_t: f64, state: SVector<f64, 2>| -> SVector<f64, 2> {
            SVector::<f64, 2>::new(state[1], -state[0])
        };

        let state = SVector::<f64, 2>::new(1.0, 0.5);
        let offsets = SVector::<f64, 2>::new(1e-6, 1e-7); // Custom per-component

        // Test with central differences
        let jacobian_central =
            varmat_custom(0.0, state, &dynamics, DifferenceMethod::Central, offsets);

        // Should match analytical Jacobian closely
        let analytical = SMatrix::<f64, 2, 2>::new(0.0, 1.0, -1.0, 0.0);
        assert_abs_diff_eq!(jacobian_central, analytical, epsilon = 1e-6);

        // Test with forward differences
        let jacobian_forward =
            varmat_custom(0.0, state, &dynamics, DifferenceMethod::Forward, offsets);
        assert_abs_diff_eq!(jacobian_forward, analytical, epsilon = 1e-4);
    }

    // ========================================================================
    // Test: Edge case - near-zero state components
    // ========================================================================

    #[test]
    fn test_varmat_near_zero_states() {
        let dynamics = |_t: f64, state: SVector<f64, 2>| -> SVector<f64, 2> {
            SVector::<f64, 2>::new(state[1], -state[0])
        };

        // State with very small components
        let state = SVector::<f64, 2>::new(1e-15, 1e-16);

        // Adaptive strategy should handle this gracefully using min_threshold
        let config = VarmatConfig::default();
        let jacobian = config.compute(0.0, state, &dynamics);

        // Should still produce reasonable Jacobian (not NaN or Inf)
        assert!(jacobian.iter().all(|&x| x.is_finite()));

        // Should be close to analytical result
        let analytical = SMatrix::<f64, 2, 2>::new(0.0, 1.0, -1.0, 0.0);
        assert_abs_diff_eq!(jacobian, analytical, epsilon = 1e-4);
    }

    // ========================================================================
    // Test: Default configuration
    // ========================================================================

    #[test]
    fn test_varmat_default_config() {
        let config = VarmatConfig::default();
        assert_eq!(config.method, DifferenceMethod::Central);
        assert_eq!(
            config.perturbation,
            PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_threshold: 1.0
            }
        );
    }

    // ========================================================================
    // Test: Orbital mechanics example (6D state)
    // ========================================================================

    #[test]
    fn test_varmat_orbital_mechanics() {
        use crate::constants::{GM_EARTH, R_EARTH};

        // Two-body dynamics
        let dynamics = |_t: f64, state: SVector<f64, 6>| -> SVector<f64, 6> {
            let r = state.fixed_rows::<3>(0).norm();
            let a = -GM_EARTH / r.powi(3);
            SVector::<f64, 6>::new(
                state[3],
                state[4],
                state[5],
                a * state[0],
                a * state[1],
                a * state[2],
            )
        };

        // Circular orbit state
        let r = R_EARTH + 500e3;
        let v = (GM_EARTH / r).sqrt();
        let state = SVector::<f64, 6>::new(r, 0.0, 0.0, 0.0, v, 0.0);

        // Compute with default config (should handle mixed position/velocity scales)
        let config = VarmatConfig::default();
        let jacobian = config.compute(0.0, state, &dynamics);

        // Verify Jacobian structure (should be block matrix)
        // Top-left 3x3 should be ~zero (dx_dot/dx ~ 0)
        for i in 0..3 {
            for j in 0..3 {
                assert!(jacobian[(i, j)].abs() < 1e-6);
            }
        }

        // Top-right 3x3 should be ~identity (dx_dot/dv = I)
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_abs_diff_eq!(jacobian[(i, j + 3)], 1.0, epsilon = 1e-6);
                } else {
                    assert!(jacobian[(i, j + 3)].abs() < 1e-6);
                }
            }
        }

        // Bottom blocks should be non-zero (gravity gradient)
        assert!(jacobian[(3, 0)].abs() > 1e-9);
        assert!(jacobian[(4, 1)].abs() > 1e-9);
        assert!(jacobian[(5, 2)].abs() > 1e-9);
    }
}
