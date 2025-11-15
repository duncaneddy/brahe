/*!
Configuration structures for numerical integrators.
*/

/// Stepping mode for numerical integrators.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StepMode {
    /// Fixed timestep integration - use provided dt for all steps
    Fixed(f64),
    /// Adaptive timestep integration - automatically adjust dt based on error estimate
    Adaptive,
}

/// Configuration options for numerical integrators.
///
/// # Example
///
/// ```
/// use brahe::integrators::IntegratorConfig;
///
/// // Create config with tight tolerances for high-precision integration
/// let config = IntegratorConfig {
///     abs_tol: 1e-12,
///     rel_tol: 1e-9,
///     max_step: Some(100.0),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct IntegratorConfig {
    /// Absolute error tolerance for adaptive stepping
    pub abs_tol: f64,

    /// Relative error tolerance for adaptive stepping
    pub rel_tol: f64,

    /// Initial step size (if None, integrator determines automatically)
    pub initial_step: Option<f64>,

    /// Minimum allowed step size (if None, no minimum enforced)
    pub min_step: Option<f64>,

    /// Maximum allowed step size (if None, no maximum enforced)
    pub max_step: Option<f64>,

    /// Stepping mode: fixed or adaptive
    pub step_mode: StepMode,

    /// Safety factor for adaptive step size control (typically 0.8-0.9)
    /// If None, no safety factor applied (uses raw error-based scaling)
    pub step_safety_factor: Option<f64>,

    /// Minimum step size scaling factor (prevents too-aggressive decreases)
    /// If None, no minimum limit on step reduction
    pub min_step_scale_factor: Option<f64>,

    /// Maximum step size scaling factor (prevents too-aggressive increases)
    /// If None, no maximum limit on step growth
    pub max_step_scale_factor: Option<f64>,

    /// Maximum attempts to find acceptable step size
    pub max_step_attempts: usize,
}

impl Default for IntegratorConfig {
    /// Default configuration matching typical defaults
    ///
    /// - abs_tol: 1e-6
    /// - rel_tol: 1e-3
    /// - initial_step: None (auto-determined)
    /// - min_step: Some(1e-12)
    /// - max_step: Some(900.0) (15 minutes)
    /// - step_mode: Adaptive
    /// - step_safety_factor: Some(0.9)
    /// - min_step_scale_factor: Some(0.2)
    /// - max_step_scale_factor: Some(10.0)
    /// - max_step_attempts: 10
    fn default() -> Self {
        Self {
            abs_tol: 1e-6,
            rel_tol: 1e-3,
            initial_step: None,
            min_step: Some(1e-12),
            max_step: Some(900.0),
            step_mode: StepMode::Adaptive,
            step_safety_factor: Some(0.9),
            min_step_scale_factor: Some(0.2),
            max_step_scale_factor: Some(10.0),
            max_step_attempts: 10,
        }
    }
}

impl IntegratorConfig {
    /// Create a new configuration for fixed-step integration.
    ///
    /// # Arguments
    /// - `step_size`: Fixed timestep to use for all integration steps
    ///
    /// # Returns
    /// IntegratorConfig configured for fixed-step mode
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::integrators::IntegratorConfig;
    ///
    /// let config = IntegratorConfig::fixed_step(1.0);
    /// ```
    pub fn fixed_step(step_size: f64) -> Self {
        Self {
            step_mode: StepMode::Fixed(step_size),
            ..Default::default()
        }
    }

    /// Create a new configuration for adaptive-step integration.
    ///
    /// # Arguments
    /// - `abs_tol`: Absolute error tolerance
    /// - `rel_tol`: Relative error tolerance
    ///
    /// # Returns
    /// IntegratorConfig configured for adaptive stepping
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::integrators::IntegratorConfig;
    ///
    /// let config = IntegratorConfig::adaptive(1e-8, 1e-6);
    /// ```
    pub fn adaptive(abs_tol: f64, rel_tol: f64) -> Self {
        Self {
            abs_tol,
            rel_tol,
            step_mode: StepMode::Adaptive,
            ..Default::default()
        }
    }
}

/// Result from an adaptive integration step.
///
/// Contains the new state, the timestep actually used, the estimated error,
/// and a suggested step size for the next iteration.
/// Used by adaptive static integrators to report step information.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveStepSResult<const S: usize> {
    /// State vector after the integration step
    pub state: nalgebra::SVector<f64, S>,

    /// Timestep actually used (may differ from requested dt)
    pub dt_used: f64,

    /// Estimated local truncation error (normalized by tolerance)
    pub error_estimate: f64,

    /// Suggested step size for next iteration (based on error estimate)
    pub dt_next: f64,
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = IntegratorConfig::default();
        assert_eq!(config.abs_tol, 1e-6);
        assert_eq!(config.rel_tol, 1e-3);
        assert_eq!(config.initial_step, None);
        assert_eq!(config.min_step, Some(1e-12));
        assert_eq!(config.max_step, Some(900.0));
        assert_eq!(config.step_mode, StepMode::Adaptive);
        assert_eq!(config.step_safety_factor, Some(0.9));
        assert_eq!(config.min_step_scale_factor, Some(0.2));
        assert_eq!(config.max_step_scale_factor, Some(10.0));
        assert_eq!(config.max_step_attempts, 10);
    }

    #[test]
    fn test_fixed_step_config() {
        let config = IntegratorConfig::fixed_step(0.1);
        assert_eq!(config.step_mode, StepMode::Fixed(0.1));
    }

    #[test]
    fn test_adaptive_config() {
        let config = IntegratorConfig::adaptive(1e-9, 1e-6);
        assert_eq!(config.abs_tol, 1e-9);
        assert_eq!(config.rel_tol, 1e-6);
        assert_eq!(config.step_mode, StepMode::Adaptive);
    }
}
