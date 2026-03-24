/*!
 * Configuration types for estimation filters.
 *
 * Provides configuration structs for process noise, EKF, UKF, and batch
 * least squares estimators.
 */

use nalgebra::DMatrix;

/// Process noise configuration for sequential filters.
///
/// Controls how process noise Q is applied to the predicted covariance
/// during the time update step: P_predicted += Q (or Q * dt).
#[derive(Clone, Debug)]
pub struct ProcessNoiseConfig {
    /// Process noise matrix Q (state_dim x state_dim)
    pub q_matrix: DMatrix<f64>,

    /// Whether Q scales with the time step duration.
    ///
    /// If true: Q_effective = Q * dt (continuous-time process noise model)
    /// If false: Q_effective = Q (discrete-time, applied as-is)
    pub scale_with_dt: bool,
}

/// Configuration for the Extended Kalman Filter.
#[derive(Clone, Debug)]
pub struct EKFConfig {
    /// Process noise configuration (None for no process noise)
    pub process_noise: Option<ProcessNoiseConfig>,

    /// Whether to store filter records (pre/post-fit residuals, Kalman gain)
    /// in memory for each processed observation. Defaults to `true`.
    ///
    /// Set to `false` for long-running filters where unbounded record growth
    /// is undesirable. Individual `process_observation()` calls still return
    /// the `FilterRecord` regardless of this setting.
    pub store_records: bool,
}

impl Default for EKFConfig {
    fn default() -> Self {
        Self {
            process_noise: None,
            store_records: true,
        }
    }
}

/// Configuration for the Unscented Kalman Filter.
#[derive(Clone, Debug)]
pub struct UKFConfig {
    /// Process noise configuration
    pub process_noise: Option<ProcessNoiseConfig>,

    /// State dimension
    pub state_dim: usize,

    /// Alpha parameter for sigma point spread (typically 1e-3)
    pub alpha: f64,

    /// Beta parameter for distribution (2.0 for Gaussian)
    pub beta: f64,

    /// Kappa parameter (typically 0.0 or 3 - state_dim)
    pub kappa: f64,
}

impl Default for UKFConfig {
    fn default() -> Self {
        Self {
            process_noise: None,
            state_dim: 6,
            alpha: 1e-3,
            beta: 2.0,
            kappa: 0.0,
        }
    }
}

/// Configuration for Batch Least Squares.
#[derive(Clone, Debug)]
pub struct BLSConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,

    /// Convergence threshold on state correction norm
    pub convergence_threshold: f64,

    /// State dimension
    pub state_dim: usize,

    /// Whether to use weighted least squares (using R^-1 as weights)
    pub weighted: bool,
}

impl Default for BLSConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            convergence_threshold: 1e-8,
            state_dim: 6,
            weighted: true,
        }
    }
}
