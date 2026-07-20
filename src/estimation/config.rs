/*!
 * Configuration types for estimation filters.
 *
 * Provides configuration structs for process noise, EKF, UKF, and batch
 * least squares estimators.
 */

use nalgebra::DMatrix;

use crate::time::Epoch;
use crate::utils::errors::BraheError;

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
    /// If true: Q_effective = Q * |dt|, where dt is the time since the last
    /// update. Q must then be a noise *rate* (units of state² per second).
    /// If false: Q_effective = Q (discrete-time, applied as-is per update)
    pub scale_with_dt: bool,

    /// Maximum time step (seconds) over which continuous process noise is
    /// accumulated in a single chunk.
    ///
    /// `None` (the default) preserves the single-shot behavior exactly: the
    /// process noise for a predict interval is added once, scaled by the full
    /// `|dt|` when `scale_with_dt` is `true`.
    ///
    /// `Some(h)` (with `h > 0`) sub-steps the predict interval: when the
    /// interval `|dt|` exceeds `h`, the filter splits it into chunks of at most
    /// `h` seconds, propagates the covariance through each chunk, and adds
    /// `q_matrix · δt_chunk` per chunk. This more faithfully accumulates the
    /// covariance growth of the underlying (nonlinear) dynamics over long coast
    /// arcs than a single end-to-end step, because the process noise is injected
    /// along the trajectory rather than all at the start.
    ///
    /// Sub-stepping applies **only** when `scale_with_dt` is `true`: `q_matrix`
    /// is then a continuous noise *rate* accumulated as `Q · δt`. When
    /// `scale_with_dt` is `false`, `q_matrix` is a discrete per-update
    /// covariance and is applied once per predict interval regardless of
    /// `max_noise_dt` (sub-stepping a per-update `Q` would multiply the noise by
    /// the number of chunks).
    ///
    /// This is an approximation: it uses the same constant rate `Q` on every
    /// chunk and relies on the state-transition-matrix propagation of `P`
    /// between chunk boundaries. The exact variational discrete-time `Q_d`
    /// integration is tracked separately in issue #408.
    pub max_noise_dt: Option<f64>,
}

impl ProcessNoiseConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if `max_noise_dt` is `Some(h)` with `h` non-positive or
    /// non-finite.
    pub fn validate(&self) -> Result<(), BraheError> {
        if let Some(h) = self.max_noise_dt
            && (!h.is_finite() || h <= 0.0)
        {
            return Err(BraheError::Error(format!(
                "ProcessNoiseConfig max_noise_dt must be finite and positive, got {}",
                h
            )));
        }
        Ok(())
    }
}

/// Compute process-noise sub-step chunk boundaries over `[start, target]`.
///
/// Returns `Some(boundaries)` only when continuous sub-stepping applies: the
/// config carries `scale_with_dt == true` and `max_noise_dt == Some(h)` with
/// `h > 0`, and the signed interval `|target − start|` strictly exceeds `h`.
/// The boundaries step from `start` toward `target` in the signed direction in
/// increments of `h` (direction-aware, consistent with bidirectional
/// `propagate_to`), with the final entry clamped to exactly `target`.
///
/// Returns `None` (single-shot: apply the process noise once) otherwise —
/// including when there is no process noise, when `scale_with_dt` is `false`
/// (a discrete per-update `Q`), when `max_noise_dt` is unset, or when the
/// interval already fits in one chunk.
pub(crate) fn noise_chunk_boundaries(
    process_noise: Option<&ProcessNoiseConfig>,
    start: Epoch,
    target: Epoch,
) -> Option<Vec<Epoch>> {
    let pn = process_noise?;
    if !pn.scale_with_dt {
        return None;
    }
    let h = pn.max_noise_dt?;
    let dt = target - start;
    if dt.abs() <= h {
        return None;
    }

    let sign = if dt >= 0.0 { 1.0 } else { -1.0 };
    let n_full = (dt.abs() / h).floor() as usize;
    let mut boundaries = Vec::with_capacity(n_full + 1);
    for i in 1..=n_full {
        boundaries.push(start + (i as f64) * h * sign);
    }
    // Clamp the final chunk to exactly the target: replace a boundary that
    // lands on the target (exact multiple of h) or append the remainder chunk.
    match boundaries.last() {
        Some(&last) if (target - last).abs() <= 1e-9 => {
            *boundaries.last_mut().unwrap() = target;
        }
        _ => boundaries.push(target),
    }
    Some(boundaries)
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
///
/// The state dimension is derived from the propagator at filter construction.
#[derive(Clone, Debug)]
pub struct UKFConfig {
    /// Process noise configuration
    pub process_noise: Option<ProcessNoiseConfig>,

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
            alpha: 1e-3,
            beta: 2.0,
            kappa: 0.0,
            store_records: true,
        }
    }
}

/// Solver formulation for Batch Least Squares.
///
/// Both formulations apply the full (possibly correlated) measurement noise
/// covariance R and produce the same estimate; they differ in numerical
/// conditioning and memory footprint.
#[derive(Clone, Debug)]
pub enum BLSSolverMethod {
    /// Whiten and stack all observation blocks into one matrix, then solve the
    /// least squares problem directly via QR decomposition. Avoids forming the
    /// normal equations, so the effective condition number is not squared.
    /// Memory: O(m_total × n_solve).
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
    use crate::time::TimeSystem;
    use nalgebra::DMatrix;
    use serial_test::parallel;

    #[test]
    #[parallel]
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
    #[parallel]
    fn test_bls_solver_method_clone_debug() {
        let method = BLSSolverMethod::StackedObservationMatrix;
        let cloned = method.clone();
        assert!(matches!(cloned, BLSSolverMethod::StackedObservationMatrix));
        let _ = format!("{:?}", method);
    }

    #[test]
    #[parallel]
    fn test_consider_parameter_config() {
        let cov = DMatrix::identity(2, 2) * 100.0;
        let config = ConsiderParameterConfig {
            n_solve: 6,
            consider_covariance: cov.clone(),
        };
        assert_eq!(config.n_solve, 6);
        assert_eq!(config.consider_covariance.nrows(), 2);
    }

    #[test]
    #[parallel]
    fn test_noise_chunk_boundaries() {
        let pn = ProcessNoiseConfig {
            q_matrix: DMatrix::identity(6, 6),
            scale_with_dt: true,
            max_noise_dt: Some(10.0),
        };
        let start = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        // Exact multiple of the chunk size: last boundary lands on the target.
        let b = noise_chunk_boundaries(Some(&pn), start, start + 30.0).unwrap();
        assert_eq!(b.len(), 3);
        assert!((b[0] - start - 10.0).abs() < 1e-9);
        assert!((b[2] - start - 30.0).abs() < 1e-9);

        // Non-multiple interval: a remainder chunk is appended.
        let b = noise_chunk_boundaries(Some(&pn), start, start + 25.0).unwrap();
        assert_eq!(b.len(), 3);
        assert!((b[1] - start - 20.0).abs() < 1e-9);
        assert!((b[2] - start - 25.0).abs() < 1e-9);

        // Backward interval steps in the negative direction.
        let b = noise_chunk_boundaries(Some(&pn), start, start - 25.0).unwrap();
        assert_eq!(b.len(), 3);
        assert!((b[0] - start + 10.0).abs() < 1e-9);
        assert!((b[2] - start + 25.0).abs() < 1e-9);

        // Single-shot cases: interval fits in one chunk, discrete-time Q,
        // no chunk cap, or no process noise at all.
        assert!(noise_chunk_boundaries(Some(&pn), start, start + 5.0).is_none());
        let discrete = ProcessNoiseConfig {
            scale_with_dt: false,
            ..pn.clone()
        };
        assert!(noise_chunk_boundaries(Some(&discrete), start, start + 30.0).is_none());
        let uncapped = ProcessNoiseConfig {
            max_noise_dt: None,
            ..pn.clone()
        };
        assert!(noise_chunk_boundaries(Some(&uncapped), start, start + 30.0).is_none());
        assert!(noise_chunk_boundaries(None, start, start + 30.0).is_none());
    }
}
