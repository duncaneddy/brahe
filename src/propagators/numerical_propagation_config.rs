/*!
 * Numerical orbit propagation configuration
 *
 * This module provides configuration for the numerical integrator settings
 * used in numerical orbit propagation, including:
 * - Integrator selection (RK4, RKF45, DP54, RKN1210)
 * - Integrator tolerances and step sizes
 * - Jacobian and sensitivity computation methods
 *
 * Note: Force model configuration is handled separately in `force_model_config.rs`.
 *
 * # Example
 *
 * ```rust,ignore
 * use brahe::propagators::{NumericalPropagationConfig, IntegratorMethod};
 *
 * // Default configuration (DP54 integrator with standard tolerances)
 * let config = NumericalPropagationConfig::default();
 *
 * // Custom configuration with specific integrator
 * let config = NumericalPropagationConfig::with_method(IntegratorMethod::RKF45);
 * ```
 */

use serde::{Deserialize, Serialize};

use crate::integrators::IntegratorConfig;
use crate::math::jacobian::DifferenceMethod;

// =============================================================================
// Integrator Method Selection
// =============================================================================

/// Integration method for numerical orbit propagation
///
/// Specifies which numerical integrator to use. Different methods trade off
/// accuracy, efficiency, and applicability.
///
/// # Available Methods
///
/// | Method   | Order | Adaptive | Function Evals | Best For |
/// |----------|-------|----------|----------------|----------|
/// | RK4      | 4     | No       | 4              | Simple problems, debugging |
/// | RKF45    | 4(5)  | Yes      | 6              | General purpose |
/// | DP54     | 5(4)  | Yes      | 7 (6 w/FSAL)   | General purpose (default) |
/// | RKN1210  | 12(10)| Yes      | 17             | High precision |
///
/// # Example
///
/// ```rust
/// use brahe::propagators::IntegratorMethod;
///
/// // Use default (DP54)
/// let method = IntegratorMethod::default();
///
/// // Check if method uses adaptive stepping
/// assert!(method.is_adaptive());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum IntegratorMethod {
    /// Classical 4th-order Runge-Kutta (fixed-step)
    ///
    /// The "standard" RK4 method. Fixed step size only.
    /// Good for simple problems or when you want to control step size manually.
    RK4,

    /// Runge-Kutta-Fehlberg 4(5) adaptive
    ///
    /// Uses 4th and 5th order solutions for error estimation.
    /// 6 function evaluations per step.
    RKF45,

    /// Dormand-Prince 5(4) adaptive (default)
    ///
    /// MATLAB's ode45. Uses FSAL (First-Same-As-Last) optimization
    /// for only 6 effective function evaluations per accepted step.
    /// Generally preferred over RKF45.
    #[default]
    DP54,

    /// Runge-Kutta-Nystrom 12(10) adaptive
    ///
    /// Very high-order integrator optimized for second-order ODEs.
    /// 17 function evaluations per step but achieves extreme accuracy.
    /// Best for high-precision applications with tight tolerances.
    ///
    /// **Note**: This integrator is experimental.
    RKN1210,
}

impl IntegratorMethod {
    /// Returns true if this integrator uses adaptive step size control.
    ///
    /// Adaptive integrators automatically adjust step size to maintain
    /// error tolerances.
    ///
    /// # Returns
    /// `true` for RKF45, DP54, and RKN1210; `false` for RK4.
    pub fn is_adaptive(&self) -> bool {
        !matches!(self, IntegratorMethod::RK4)
    }
}

// =============================================================================
// Numerical Propagation Configuration
// =============================================================================

/// Configuration for the numerical integrator
///
/// This struct contains the integrator-specific configuration options:
/// - Integration method selection
/// - Integrator tolerances and step sizes
/// - Methods for computing Jacobians and sensitivities
///
/// Note: Force model configuration (gravity, drag, SRP, etc.) is handled
/// separately via `ForceModelConfiguration`.
///
/// # Example
///
/// ```rust,ignore
/// use brahe::propagators::{NumericalPropagationConfig, IntegratorMethod};
/// use brahe::integrators::IntegratorConfig;
///
/// // Create custom configuration
/// let config = NumericalPropagationConfig {
///     method: IntegratorMethod::RKF45,
///     integrator: IntegratorConfig::adaptive(1e-10, 1e-8),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericalPropagationConfig {
    /// Integration method to use
    pub method: IntegratorMethod,

    /// Integrator configuration (tolerances, step sizes)
    pub integrator: IntegratorConfig,

    /// Finite difference method for Jacobian computation
    ///
    /// Used when computing the state transition matrix (STM) numerically.
    pub jacobian_method: DifferenceMethod,

    /// Finite difference method for sensitivity matrix computation
    ///
    /// Used when computing parameter sensitivities numerically.
    pub sensitivity_method: DifferenceMethod,
}

impl Default for NumericalPropagationConfig {
    /// Default configuration suitable for most applications
    ///
    /// Uses:
    /// - Dormand-Prince 5(4) integrator
    /// - Default tolerances (abs=1e-6, rel=1e-3)
    /// - Central differences for Jacobian/sensitivity
    fn default() -> Self {
        Self {
            method: IntegratorMethod::default(),
            integrator: IntegratorConfig::default(),
            jacobian_method: DifferenceMethod::Central,
            sensitivity_method: DifferenceMethod::Central,
        }
    }
}

impl NumericalPropagationConfig {
    /// Create configuration with a specific integrator method
    ///
    /// # Arguments
    /// * `method` - The integrator method to use
    ///
    /// # Returns
    /// Configuration with the specified method and default settings
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use brahe::propagators::{NumericalPropagationConfig, IntegratorMethod};
    ///
    /// let config = NumericalPropagationConfig::with_method(IntegratorMethod::RKF45);
    /// ```
    pub fn with_method(method: IntegratorMethod) -> Self {
        Self {
            method,
            ..Default::default()
        }
    }

    /// Create high-precision configuration with tight tolerances
    ///
    /// Uses:
    /// - Dormand-Prince 5(4) integrator
    /// - Tight tolerances (abs=1e-10, rel=1e-8)
    /// - Central differences for Jacobian/sensitivity
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use brahe::propagators::NumericalPropagationConfig;
    ///
    /// let config = NumericalPropagationConfig::high_precision();
    /// ```
    pub fn high_precision() -> Self {
        Self {
            method: IntegratorMethod::DP54,
            integrator: IntegratorConfig::adaptive(1e-10, 1e-8),
            jacobian_method: DifferenceMethod::Central,
            sensitivity_method: DifferenceMethod::Central,
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_integrator_method_default() {
        let method = IntegratorMethod::default();
        assert_eq!(method, IntegratorMethod::DP54);
    }

    #[test]
    fn test_integrator_method_is_adaptive() {
        assert!(!IntegratorMethod::RK4.is_adaptive());
        assert!(IntegratorMethod::RKF45.is_adaptive());
        assert!(IntegratorMethod::DP54.is_adaptive());
        assert!(IntegratorMethod::RKN1210.is_adaptive());
    }

    #[test]
    fn test_numerical_propagation_config_default() {
        let config = NumericalPropagationConfig::default();

        assert_eq!(config.method, IntegratorMethod::DP54);
        assert_eq!(config.jacobian_method, DifferenceMethod::Central);
        assert_eq!(config.sensitivity_method, DifferenceMethod::Central);
    }

    #[test]
    fn test_numerical_propagation_config_with_method() {
        let config = NumericalPropagationConfig::with_method(IntegratorMethod::RKF45);
        assert_eq!(config.method, IntegratorMethod::RKF45);
    }

    #[test]
    fn test_numerical_propagation_config_high_precision() {
        let config = NumericalPropagationConfig::high_precision();

        assert_eq!(config.method, IntegratorMethod::DP54);
        // Tolerances are set tighter
        assert!(config.integrator.abs_tol <= 1e-9);
        assert!(config.integrator.rel_tol <= 1e-7);
    }
}
