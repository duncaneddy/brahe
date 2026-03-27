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

    /// Whether to store filter records in memory for each processed observation.
    /// Defaults to `true`.
    pub store_records: bool,
}

impl Default for UKFConfig {
    fn default() -> Self {
        Self {
            process_noise: None,
            state_dim: 6,
            alpha: 1e-3,
            beta: 2.0,
            kappa: 0.0,
            store_records: true,
        }
    }
}

/// Solver formulation for Batch Least Squares.
#[derive(Clone, Debug)]
pub enum BLSSolverMethod {
    /// Build full H matrix and residual vector, solve via QR decomposition.
    /// Better numerical conditioning. Memory: O(m_total × n_solve).
    StackedObservationMatrix,
    /// Accumulate information matrix Λ and normal vector N, solve via Cholesky.
    /// Memory-efficient: O(n_solve²). Standard Tapley/Schutz/Born formulation.
    NormalEquations,
}

/// Configuration for consider parameters in batch estimation.
#[derive(Clone, Debug)]
pub struct ConsiderParameterConfig {
    /// Number of solve-for parameters (first n_solve elements of the state vector)
    pub n_solve: usize,
    /// A priori covariance for the consider parameters (n_c × n_c)
    pub consider_covariance: DMatrix<f64>,
}

/// Configuration for Batch Least Squares.
#[derive(Clone, Debug)]
pub struct BLSConfig {
    /// Solver formulation to use
    pub solver_method: BLSSolverMethod,
    /// Maximum number of Gauss-Newton iterations
    pub max_iterations: usize,
    /// Convergence threshold on state correction norm ||δx||.
    pub state_correction_threshold: Option<f64>,
    /// Convergence threshold on relative cost function change |ΔJ|/|J|.
    pub cost_convergence_threshold: Option<f64>,
    /// Consider parameter configuration (None = all state elements are solve-for).
    pub consider_params: Option<ConsiderParameterConfig>,
    /// Whether to store per-iteration diagnostic records. Defaults to true.
    pub store_iteration_records: bool,
    /// Whether to store per-observation residuals at each iteration. Defaults to true.
    pub store_observation_residuals: bool,
}

impl Default for BLSConfig {
    fn default() -> Self {
        Self {
            solver_method: BLSSolverMethod::NormalEquations,
            max_iterations: 10,
            state_correction_threshold: Some(1e-8),
            cost_convergence_threshold: None,
            consider_params: None,
            store_iteration_records: true,
            store_observation_residuals: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_bls_config_default() {
        let config = BLSConfig::default();
        assert_eq!(config.max_iterations, 10);
        assert_eq!(config.state_correction_threshold, Some(1e-8));
        assert_eq!(config.cost_convergence_threshold, None);
        assert!(config.consider_params.is_none());
        assert!(config.store_iteration_records);
        assert!(config.store_observation_residuals);
        assert!(matches!(
            config.solver_method,
            BLSSolverMethod::NormalEquations
        ));
    }

    #[test]
    fn test_bls_solver_method_clone_debug() {
        let method = BLSSolverMethod::StackedObservationMatrix;
        let cloned = method.clone();
        assert!(matches!(cloned, BLSSolverMethod::StackedObservationMatrix));
        let _ = format!("{:?}", method);
    }

    #[test]
    fn test_consider_parameter_config() {
        let cov = DMatrix::identity(2, 2) * 100.0;
        let config = ConsiderParameterConfig {
            n_solve: 6,
            consider_covariance: cov.clone(),
        };
        assert_eq!(config.n_solve, 6);
        assert_eq!(config.consider_covariance.nrows(), 2);
    }
}
