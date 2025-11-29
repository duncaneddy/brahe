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
use crate::math::interpolation::InterpolationMethod;
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
// Trajectory Mode
// =============================================================================

/// Trajectory storage mode for numerical propagators
///
/// Controls when and whether state data is stored in the trajectory during propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TrajectoryMode {
    /// Store state at requested output epochs only (default)
    ///
    /// Most memory-efficient. Only stores at times explicitly requested
    /// via `propagate_to_epochs()` or similar methods.
    #[default]
    OutputStepsOnly,

    /// Store state at every integration step
    ///
    /// Useful for debugging or high-resolution trajectory analysis.
    /// May use significantly more memory for long propagations.
    AllSteps,

    /// Disable trajectory storage entirely
    ///
    /// Only the current state is maintained. Useful when only the
    /// final state matters and memory is constrained.
    Disabled,
}

// =============================================================================
// Variational Configuration
// =============================================================================

/// Configuration for STM and sensitivity matrix propagation
///
/// Controls whether the propagator computes and stores variational matrices
/// (State Transition Matrix and Sensitivity Matrix) during propagation.
///
/// # Example
///
/// ```rust
/// use brahe::propagators::VariationalConfig;
/// use brahe::math::jacobian::DifferenceMethod;
///
/// // Default: no variational matrices
/// let config = VariationalConfig::default();
///
/// // Enable STM and sensitivity with history storage
/// let config = VariationalConfig::new(
///     true, true, true, true,
///     DifferenceMethod::Central, DifferenceMethod::Central,
/// );
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariationalConfig {
    /// Enable State Transition Matrix (STM) propagation
    ///
    /// When enabled, the propagator computes the STM (Φ) which maps
    /// initial state perturbations to final state perturbations:
    /// δx(t) = Φ(t, t₀) · δx(t₀)
    ///
    /// Default: `false`
    /// Note: Automatically enabled if `initial_covariance` is provided to constructor.
    pub enable_stm: bool,

    /// Enable sensitivity matrix propagation
    ///
    /// When enabled, the propagator computes the sensitivity matrix (S)
    /// which maps parameter perturbations to state perturbations:
    /// δx(t) = S(t) · δp
    ///
    /// Default: `false`
    /// Note: Requires `params` to be provided to constructor when enabled.
    pub enable_sensitivity: bool,

    /// Store STM at output times in trajectory
    ///
    /// When enabled, the STM is stored at requested output epochs
    /// and can be retrieved from the trajectory.
    ///
    /// Default: `false`
    pub store_stm_history: bool,

    /// Store sensitivity matrix at output times in trajectory
    ///
    /// When enabled, the sensitivity matrix is stored at requested
    /// output epochs and can be retrieved from the trajectory.
    ///
    /// Default: `false`
    pub store_sensitivity_history: bool,

    /// Finite difference method for Jacobian computation
    ///
    /// Used when computing the state transition matrix (STM) numerically.
    ///
    /// Default: `DifferenceMethod::Central`
    pub jacobian_method: DifferenceMethod,

    /// Finite difference method for sensitivity matrix computation
    ///
    /// Used when computing parameter sensitivities numerically.
    ///
    /// Default: `DifferenceMethod::Central`
    pub sensitivity_method: DifferenceMethod,
}

impl Default for VariationalConfig {
    fn default() -> Self {
        Self {
            enable_stm: false,
            enable_sensitivity: false,
            store_stm_history: false,
            store_sensitivity_history: false,
            jacobian_method: DifferenceMethod::Central,
            sensitivity_method: DifferenceMethod::Central,
        }
    }
}

impl VariationalConfig {
    /// Create configuration with specified STM and sensitivity settings
    ///
    /// # Arguments
    /// * `enable_stm` - Enable STM propagation
    /// * `enable_sensitivity` - Enable sensitivity matrix propagation
    /// * `store_stm_history` - Store STM at each step in trajectory
    /// * `store_sensitivity_history` - Store sensitivity at each step in trajectory
    /// * `jacobian_method` - Finite difference method for Jacobian computation
    /// * `sensitivity_method` - Finite difference method for sensitivity computation
    ///
    /// # Example
    ///
    /// ```rust
    /// use brahe::propagators::VariationalConfig;
    /// use brahe::math::jacobian::DifferenceMethod;
    ///
    /// // Full uncertainty quantification with history
    /// let config = VariationalConfig::new(
    ///     true, true, true, true,
    ///     DifferenceMethod::Central, DifferenceMethod::Central,
    /// );
    /// ```
    pub fn new(
        enable_stm: bool,
        enable_sensitivity: bool,
        store_stm_history: bool,
        store_sensitivity_history: bool,
        jacobian_method: DifferenceMethod,
        sensitivity_method: DifferenceMethod,
    ) -> Self {
        Self {
            enable_stm,
            enable_sensitivity,
            store_stm_history,
            store_sensitivity_history,
            jacobian_method,
            sensitivity_method,
        }
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
/// - Variational equation settings (STM, sensitivity, finite difference methods)
///
/// Note: Force model configuration (gravity, drag, SRP, etc.) is handled
/// separately via `ForceModelConfig`.
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

    /// STM and sensitivity propagation configuration
    pub variational: VariationalConfig,

    /// Whether to store accelerations in trajectory
    ///
    /// When enabled, the propagator computes and stores acceleration vectors
    /// at each trajectory point. This enables higher-order interpolation
    /// methods like Hermite quintic that use acceleration data.
    ///
    /// Default: `true`
    pub store_accelerations: bool,

    /// Interpolation method for trajectory queries
    ///
    /// Controls how states are interpolated between stored trajectory points.
    /// Higher-order methods (Lagrange, Hermite) provide better accuracy but
    /// may require more stored points or acceleration data.
    ///
    /// Default: `InterpolationMethod::HermiteCubic`
    pub interpolation_method: InterpolationMethod,
}

impl Default for NumericalPropagationConfig {
    /// Default configuration suitable for most applications
    ///
    /// Uses:
    /// - Dormand-Prince 5(4) integrator
    /// - Default tolerances (abs=1e-6, rel=1e-3)
    /// - No variational matrix propagation (central differences when enabled)
    /// - Acceleration storage enabled
    /// - Hermite cubic interpolation
    fn default() -> Self {
        Self {
            method: IntegratorMethod::default(),
            integrator: IntegratorConfig::default(),
            variational: VariationalConfig::default(),
            store_accelerations: true,
            interpolation_method: InterpolationMethod::HermiteCubic,
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

    /// Creates a new numerical propagation configuration with all components specified.
    ///
    /// # Arguments
    ///
    /// * `method` - Integration method to use (RK4, RKF45, DP54, or RKN1210)
    /// * `integrator` - Integrator configuration (tolerances, step sizes)
    /// * `variational` - Variational configuration (STM and sensitivity settings)
    ///
    /// # Returns
    ///
    /// A new `NumericalPropagationConfig` with the specified settings.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::propagators::{
    ///     NumericalPropagationConfig, IntegratorMethod, VariationalConfig
    /// };
    /// use brahe::integrators::IntegratorConfig;
    ///
    /// let config = NumericalPropagationConfig::new(
    ///     IntegratorMethod::DP54,
    ///     IntegratorConfig::adaptive(1e-10, 1e-8),
    ///     VariationalConfig::default(),
    /// );
    /// ```
    pub fn new(
        method: IntegratorMethod,
        integrator: IntegratorConfig,
        variational: VariationalConfig,
    ) -> Self {
        Self {
            method,
            integrator,
            variational,
            store_accelerations: true,
            interpolation_method: InterpolationMethod::HermiteCubic,
        }
    }

    /// Create high-precision configuration with tight tolerances
    ///
    /// Uses:
    /// - RKN1210 integrator
    /// - Tight tolerances (abs=1e-10, rel=1e-8)
    /// - No variational matrix propagation (central differences when enabled)
    /// - Acceleration storage enabled
    /// - Hermite cubic interpolation
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
            method: IntegratorMethod::RKN1210,
            integrator: IntegratorConfig::adaptive(1e-10, 1e-8),
            variational: VariationalConfig::default(),
            store_accelerations: true,
            interpolation_method: InterpolationMethod::HermiteCubic,
        }
    }

    /// Set whether to store accelerations in trajectory
    ///
    /// # Arguments
    /// * `store` - Whether to store accelerations
    ///
    /// # Returns
    /// Self for method chaining
    pub fn with_store_accelerations(mut self, store: bool) -> Self {
        self.store_accelerations = store;
        self
    }

    /// Set the interpolation method for trajectory queries
    ///
    /// # Arguments
    /// * `method` - The interpolation method to use
    ///
    /// # Returns
    /// Self for method chaining
    pub fn with_interpolation_method(mut self, method: InterpolationMethod) -> Self {
        self.interpolation_method = method;
        self
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
        assert_eq!(
            config.variational.jacobian_method,
            DifferenceMethod::Central
        );
        assert_eq!(
            config.variational.sensitivity_method,
            DifferenceMethod::Central
        );
    }

    #[test]
    fn test_numerical_propagation_config_with_method() {
        let config = NumericalPropagationConfig::with_method(IntegratorMethod::RKF45);
        assert_eq!(config.method, IntegratorMethod::RKF45);
    }

    #[test]
    fn test_numerical_propagation_config_high_precision() {
        let config = NumericalPropagationConfig::high_precision();

        assert_eq!(config.method, IntegratorMethod::RKN1210);
        // Tolerances are set tighter
        assert!(config.integrator.abs_tol <= 1e-9);
        assert!(config.integrator.rel_tol <= 1e-7);
    }

    #[test]
    fn test_variational_config_default() {
        let config = VariationalConfig::default();

        assert!(!config.enable_stm);
        assert!(!config.enable_sensitivity);
        assert!(!config.store_stm_history);
        assert!(!config.store_sensitivity_history);
        assert_eq!(config.jacobian_method, DifferenceMethod::Central);
        assert_eq!(config.sensitivity_method, DifferenceMethod::Central);
    }

    #[test]
    fn test_variational_config_new() {
        let config = VariationalConfig::new(
            true,
            true,
            true,
            true,
            DifferenceMethod::Forward,
            DifferenceMethod::Backward,
        );

        assert!(config.enable_stm);
        assert!(config.enable_sensitivity);
        assert!(config.store_stm_history);
        assert!(config.store_sensitivity_history);
        assert_eq!(config.jacobian_method, DifferenceMethod::Forward);
        assert_eq!(config.sensitivity_method, DifferenceMethod::Backward);
    }

    #[test]
    fn test_variational_config_partial() {
        let config = VariationalConfig::new(
            true,
            false,
            true,
            false,
            DifferenceMethod::Central,
            DifferenceMethod::Central,
        );

        assert!(config.enable_stm);
        assert!(!config.enable_sensitivity);
        assert!(config.store_stm_history);
        assert!(!config.store_sensitivity_history);
        assert_eq!(config.jacobian_method, DifferenceMethod::Central);
        assert_eq!(config.sensitivity_method, DifferenceMethod::Central);
    }

    #[test]
    fn test_numerical_propagation_config_default_variational() {
        let config = NumericalPropagationConfig::default();

        assert!(!config.variational.enable_stm);
        assert!(!config.variational.enable_sensitivity);
    }

    #[test]
    fn test_numerical_propagation_config_new() {
        let config = NumericalPropagationConfig::new(
            IntegratorMethod::RKN1210,
            IntegratorConfig::adaptive(1e-12, 1e-10),
            VariationalConfig {
                enable_stm: true,
                ..Default::default()
            },
        );

        assert_eq!(config.method, IntegratorMethod::RKN1210);
        assert_eq!(config.integrator.abs_tol, 1e-12);
        assert_eq!(config.integrator.rel_tol, 1e-10);
        assert!(config.variational.enable_stm);
        assert!(!config.variational.enable_sensitivity);
        // New fields should have defaults
        assert!(config.store_accelerations);
        assert_eq!(
            config.interpolation_method,
            InterpolationMethod::HermiteCubic
        );
    }

    #[test]
    fn test_numerical_propagation_config_default_accelerations() {
        let config = NumericalPropagationConfig::default();

        // Acceleration storage should be enabled by default
        assert!(config.store_accelerations);
        assert_eq!(
            config.interpolation_method,
            InterpolationMethod::HermiteCubic
        );
    }

    #[test]
    fn test_numerical_propagation_config_with_store_accelerations() {
        let config = NumericalPropagationConfig::default().with_store_accelerations(false);
        assert!(!config.store_accelerations);

        let config = NumericalPropagationConfig::default().with_store_accelerations(true);
        assert!(config.store_accelerations);
    }

    #[test]
    fn test_numerical_propagation_config_with_interpolation_method() {
        let config = NumericalPropagationConfig::default()
            .with_interpolation_method(InterpolationMethod::Lagrange { degree: 5 });
        assert_eq!(
            config.interpolation_method,
            InterpolationMethod::Lagrange { degree: 5 }
        );

        let config = NumericalPropagationConfig::default()
            .with_interpolation_method(InterpolationMethod::HermiteCubic);
        assert_eq!(
            config.interpolation_method,
            InterpolationMethod::HermiteCubic
        );

        let config = NumericalPropagationConfig::default()
            .with_interpolation_method(InterpolationMethod::HermiteQuintic);
        assert_eq!(
            config.interpolation_method,
            InterpolationMethod::HermiteQuintic
        );
    }

    #[test]
    fn test_numerical_propagation_config_builder_chaining() {
        let config = NumericalPropagationConfig::default()
            .with_store_accelerations(false)
            .with_interpolation_method(InterpolationMethod::Lagrange { degree: 7 });

        assert!(!config.store_accelerations);
        assert_eq!(
            config.interpolation_method,
            InterpolationMethod::Lagrange { degree: 7 }
        );
    }

    #[test]
    fn test_numerical_propagation_config_high_precision_new_fields() {
        let config = NumericalPropagationConfig::high_precision();

        // High precision should also have defaults for new fields
        assert!(config.store_accelerations);
        assert_eq!(
            config.interpolation_method,
            InterpolationMethod::HermiteCubic
        );
    }
}
