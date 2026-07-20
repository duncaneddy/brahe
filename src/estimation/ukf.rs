/*!
 * Unscented Kalman Filter for sequential state estimation.
 *
 * The UKF uses the unscented transform to propagate state statistics through
 * nonlinear dynamics and measurement models without linearization. It generates
 * deterministic sigma points that capture the mean and covariance of the state
 * distribution, propagates them through the true nonlinear functions, and
 * reconstructs the statistics from the transformed points.
 *
 * # Algorithm
 *
 * **Predict** (time update to measurement epoch):
 * 1. Generate 2n+1 sigma points from current state and covariance
 * 2. Propagate each sigma point through dynamics to observation epoch
 * 3. Compute predicted mean and covariance from propagated sigma points
 * 4. Add process noise: P_pred += Q
 *
 * **Update** (measurement incorporation):
 * 1. Generate sigma points from predicted state and covariance
 * 2. Transform each through measurement model: Z_i = h(χ_i)
 * 3. Compute predicted measurement mean and innovation covariance
 * 4. Compute cross-covariance P_xz and gain K = P_xz * S⁻¹
 * 5. Update state: x_upd = x_pred + K * (z - z_pred)
 * 6. Update covariance: P_upd = P_pred - K * S * Kᵀ
 *
 * # Advantages over EKF
 *
 * - No Jacobian computation required (neither analytical nor numerical)
 * - Captures nonlinearity to second order (vs first order for EKF)
 * - Works with non-differentiable measurement models
 * - Does not require STM propagation (faster per-step dynamics)
 *
 * The trade-off is 2n+1 dynamics evaluations per predict step instead of one.
 *
 * Use [`UnscentedKalmanFilter::new()`] for the common orbit-propagator case,
 * or [`UnscentedKalmanFilter::from_propagator()`] when you have a pre-built
 * propagator with custom dynamics.
 */

use nalgebra::{DMatrix, DVector};

use crate::integrators::traits::{DControlInput, DStateDynamics};
use crate::propagators::force_model_config::ForceModelConfig;
use crate::propagators::{DNumericalOrbitPropagator, NumericalPropagationConfig, TrajectoryMode};
use crate::time::Epoch;
use crate::utils::errors::BraheError;

use super::config::{UKFConfig, noise_chunk_boundaries};
use super::dynamics_source::DynamicsSource;
use super::traits::{MeasurementModel, compute_residual, validate_model_outputs};
use super::types::{FilterRecord, Observation, sort_by_epoch};

/// Unscented Kalman Filter for sequential state estimation.
///
/// Processes observations one at a time using the unscented transform to
/// capture nonlinear effects without linearization. Sigma points are
/// propagated through the true dynamics and measurement models.
///
/// # Examples
///
/// ```no_run
/// use brahe::estimation::*;
/// use brahe::propagators::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::{DMatrix, DVector};
///
/// let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let state = DVector::from_vec(vec![6878e3, 0.0, 0.0, 0.0, 7612.0, 0.0]);
/// let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
///
/// let mut ukf = UnscentedKalmanFilter::new(
///     epoch,
///     state,
///     p0,
///     NumericalPropagationConfig::default(),
///     ForceModelConfig::two_body_gravity(),
///     None,
///     None,
///     None,
///     vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
///     UKFConfig::default(),
/// ).unwrap();
///
/// let obs = Observation::new(
///     epoch + 60.0,
///     DVector::from_vec(vec![6877.8e3, 456.7e3, 0.0]),
///     0,
/// );
/// let record = ukf.process_observation(&obs).unwrap();
/// println!("post-fit residual: {}", record.postfit_residual.norm());
/// ```
pub struct UnscentedKalmanFilter {
    /// Dynamics source (propagator).
    dynamics: DynamicsSource,

    /// Measurement models (supports multiple types)
    measurement_models: Vec<Box<dyn MeasurementModel>>,

    /// Filter configuration
    config: UKFConfig,

    /// History of filter records
    records: Vec<FilterRecord>,

    /// State dimension (from the propagator)
    state_dim: usize,

    /// Cached mean weights (2n+1)
    weights_mean: DVector<f64>,

    /// Cached covariance weights (2n+1)
    weights_cov: DVector<f64>,

    /// Cached lambda = alpha^2 * (n + kappa) - n
    lambda: f64,

    /// Internal covariance (managed by UKF directly, not by propagator)
    covariance: DMatrix<f64>,
}

impl UnscentedKalmanFilter {
    /// Create an Unscented Kalman Filter with orbit propagator dynamics.
    ///
    /// Builds a numerical orbit propagator internally. Unlike the EKF, no STM
    /// propagation is enabled: the UKF captures uncertainty via sigma-point
    /// propagation, so each sigma point integrates only the state equations.
    /// The full range of orbit propagator configuration is available: force
    /// model parameters, additional dynamics, and control inputs pass through
    /// to the propagator. For a generic propagator (`DNumericalPropagator`)
    /// use [`UnscentedKalmanFilter::from_propagator`].
    ///
    /// # Arguments
    ///
    /// * `epoch` - Initial epoch
    /// * `state` - Initial state vector (meters, m/s)
    /// * `initial_covariance` - Initial covariance matrix (state_dim x state_dim)
    /// * `propagation_config` - Numerical propagation configuration
    /// * `force_config` - Force model configuration
    /// * `params` - Optional parameter vector for force models
    /// * `additional_dynamics` - Optional additional dynamics function
    /// * `control_input` - Optional control input function
    /// * `measurement_models` - List of measurement models
    /// * `config` - UKF configuration (alpha, beta, kappa)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        epoch: Epoch,
        state: DVector<f64>,
        initial_covariance: DMatrix<f64>,
        propagation_config: NumericalPropagationConfig,
        force_config: ForceModelConfig,
        params: Option<DVector<f64>>,
        additional_dynamics: Option<DStateDynamics>,
        control_input: DControlInput,
        measurement_models: Vec<Box<dyn MeasurementModel>>,
        config: UKFConfig,
    ) -> Result<Self, BraheError> {
        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            propagation_config,
            force_config,
            params,
            additional_dynamics,
            control_input,
            None,
        )
        .map_err(|e| BraheError::Error(format!("Failed to create propagator: {}", e)))?;

        Self::from_parts(prop.into(), initial_covariance, measurement_models, config)
    }

    /// Create an Unscented Kalman Filter from an existing propagator.
    ///
    /// Use this when you have a pre-built propagator — custom dynamics
    /// (`DNumericalPropagator`), force model parameters, control inputs, or
    /// any other propagator configuration not covered by
    /// [`UnscentedKalmanFilter::new`]. Both propagator types convert into
    /// [`DynamicsSource`] automatically.
    ///
    /// The filter's initial covariance is taken from the propagator's
    /// `initial_covariance`; construct the propagator with the covariance you
    /// want the filter to start from.
    ///
    /// Providing a covariance to a propagator auto-enables its STM
    /// propagation, which the UKF does not use; the filter disables STM
    /// propagation on the propagator at construction so sigma-point
    /// propagation only integrates the state equations.
    ///
    /// Trajectory recording on the propagator is disabled: sigma-point
    /// propagation re-propagates each time span 2n+1 times, which would
    /// otherwise accumulate unbounded, interleaved trajectory data.
    ///
    /// # Arguments
    ///
    /// * `propagator` - Pre-built propagator (must have an initial covariance)
    /// * `measurement_models` - List of measurement models
    /// * `config` - UKF configuration (alpha, beta, kappa)
    ///
    /// # Errors
    ///
    /// Returns error if the propagator was not initialized with an initial
    /// covariance, if no measurement models are provided, or if the
    /// sigma-point parameters are invalid (`alpha <= 0` or
    /// `state_dim + kappa <= 0`).
    pub fn from_propagator(
        propagator: impl Into<DynamicsSource>,
        measurement_models: Vec<Box<dyn MeasurementModel>>,
        config: UKFConfig,
    ) -> Result<Self, BraheError> {
        let dynamics = propagator.into();

        let initial_covariance = dynamics
            .current_covariance()
            .ok_or_else(|| {
                BraheError::Error(
                    "UnscentedKalmanFilter requires an initial covariance on the propagator. \
                     Initialize the propagator with an initial_covariance."
                        .to_string(),
                )
            })?
            .clone();

        Self::from_parts(dynamics, initial_covariance, measurement_models, config)
    }

    /// Shared constructor: validates and assembles the filter from a dynamics
    /// source and an explicit initial covariance.
    fn from_parts(
        mut dynamics: DynamicsSource,
        initial_covariance: DMatrix<f64>,
        measurement_models: Vec<Box<dyn MeasurementModel>>,
        config: UKFConfig,
    ) -> Result<Self, BraheError> {
        let n = dynamics.state_dim();
        if initial_covariance.nrows() != n || initial_covariance.ncols() != n {
            return Err(BraheError::Error(format!(
                "Initial covariance dimensions ({}x{}) do not match state dimension ({})",
                initial_covariance.nrows(),
                initial_covariance.ncols(),
                n
            )));
        }

        if measurement_models.is_empty() {
            return Err(BraheError::Error(
                "At least one measurement model is required".to_string(),
            ));
        }

        if let Some(ref pn) = config.process_noise {
            pn.validate()?;
        }

        // Validate sigma-point parameters: the scaling factor n + lambda =
        // alpha^2 * (n + kappa) must be positive and finite for the
        // Cholesky-based sigma-point generation and the weight divisions to
        // be well-defined. NaN compares false against any threshold, so
        // finiteness is checked explicitly; a tiny alpha can underflow
        // alpha^2 to zero, so the check is on the computed scale itself.
        let alpha = config.alpha;
        let beta = config.beta;
        let kappa = config.kappa;
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(BraheError::Error(format!(
                "UKF alpha must be positive and finite, got {}",
                alpha
            )));
        }
        if !beta.is_finite() {
            return Err(BraheError::Error(format!(
                "UKF beta must be finite, got {}",
                beta
            )));
        }
        if !kappa.is_finite() {
            return Err(BraheError::Error(format!(
                "UKF kappa must be finite, got {}",
                kappa
            )));
        }
        let scale = alpha * alpha * (n as f64 + kappa);
        if !scale.is_finite() || scale <= 0.0 {
            return Err(BraheError::Error(format!(
                "UKF sigma-point scale alpha^2 * (state_dim + kappa) must be positive \
                 and finite, got {} (alpha={}, kappa={}, state_dim={})",
                scale, alpha, kappa, n
            )));
        }

        // Compute UKF weights
        let lambda = scale - n as f64;
        let n_sigma = 2 * n + 1;

        let mut weights_mean = DVector::zeros(n_sigma);
        let mut weights_cov = DVector::zeros(n_sigma);

        // W_0^m = lambda / (n + lambda)
        weights_mean[0] = lambda / (n as f64 + lambda);
        // W_0^c = lambda / (n + lambda) + (1 - alpha^2 + beta)
        weights_cov[0] = lambda / (n as f64 + lambda) + (1.0 - alpha * alpha + beta);

        // W_i^m = W_i^c = 1 / (2(n + lambda)) for i = 1..2n
        let w_i = 1.0 / (2.0 * (n as f64 + lambda));
        for i in 1..n_sigma {
            weights_mean[i] = w_i;
            weights_cov[i] = w_i;
        }

        dynamics.set_trajectory_mode(TrajectoryMode::Disabled);

        // The UKF captures uncertainty via sigma points and never reads the
        // STM; disable its propagation (auto-enabled when the propagator was
        // constructed with a covariance) so sigma-point propagation only
        // integrates the state equations.
        dynamics.disable_stm_propagation();

        Ok(Self {
            dynamics,
            measurement_models,
            config,
            records: Vec::new(),
            state_dim: n,
            weights_mean,
            weights_cov,
            lambda,
            covariance: initial_covariance,
        })
    }

    /// Generate 2n+1 sigma points from mean and covariance.
    ///
    /// Uses Cholesky decomposition of (n + lambda) * P for the matrix square root.
    fn generate_sigma_points(
        mean: &DVector<f64>,
        cov: &DMatrix<f64>,
        lambda: f64,
        n: usize,
    ) -> Result<Vec<DVector<f64>>, BraheError> {
        let scaled_cov = (n as f64 + lambda) * cov;

        // Cholesky decomposition: scaled_cov = L * L^T
        let chol = scaled_cov.clone().cholesky().ok_or_else(|| {
            BraheError::NumericalError(
                "Cholesky decomposition failed: covariance is not positive-definite. \
                 The filter may have diverged."
                    .to_string(),
            )
        })?;
        let l = chol.l();

        let mut sigma_points = Vec::with_capacity(2 * n + 1);

        // χ_0 = mean
        sigma_points.push(mean.clone());

        // χ_i = mean + L_col_i for i = 1..n
        // χ_{i+n} = mean - L_col_i for i = 1..n
        for i in 0..n {
            let col = l.column(i);
            sigma_points.push(mean + col);
            sigma_points.push(mean - col);
        }

        Ok(sigma_points)
    }

    /// Process a single observation.
    ///
    /// Performs predict (propagate sigma points to observation epoch) then
    /// update (incorporate measurement via unscented transform).
    ///
    /// # Arguments
    ///
    /// * `observation` - The observation to process
    ///
    /// # Returns
    ///
    /// Filter record containing pre/post-fit residuals, gain, etc.
    ///
    /// # Errors
    ///
    /// Returns error if observation epoch is before current filter epoch,
    /// if model_index is out of bounds, or if a numerical error occurs.
    pub fn process_observation(
        &mut self,
        observation: &Observation,
    ) -> Result<FilterRecord, BraheError> {
        // Validate time ordering
        let current_epoch = self.dynamics.current_epoch();
        let dt: f64 = observation.epoch - current_epoch;
        if dt < -1e-9 {
            return Err(BraheError::Error(format!(
                "Observation epoch is before current filter epoch. \
                 Filter epoch: {}, observation epoch: {}. \
                 Use process_observations() which auto-sorts, or ensure \
                 observations are in chronological order.",
                current_epoch, observation.epoch
            )));
        }

        // Validate model index
        if observation.model_index >= self.measurement_models.len() {
            return Err(BraheError::Error(format!(
                "Observation model_index {} is out of bounds (have {} models)",
                observation.model_index,
                self.measurement_models.len()
            )));
        }

        // Snapshot for rollback: any error after the propagator has been
        // advanced would otherwise leave a propagated sigma point — not the
        // state estimate — as the filter state.
        let entry_state = self.dynamics.current_state();

        match self.predict_and_update(observation, current_epoch, dt) {
            Ok(record) => {
                if self.config.store_records {
                    self.records.push(record.clone());
                }
                Ok(record)
            }
            Err(e) => {
                // Roll back the propagator; self.covariance is only written
                // on the success path, so restoring the propagator restores
                // the consistent pre-observation filter state.
                self.dynamics.reinitialize(current_epoch, entry_state, None);
                Err(e)
            }
        }
    }

    /// Fallible predict/update body of [`process_observation`]. Mutates the
    /// propagator; the caller rolls it back if this returns an error.
    /// Sub-stepped predict: run a full unscented transform per chunk across the
    /// provided `boundaries`, injecting continuous process noise `Q · δt_chunk`
    /// at each chunk boundary rather than all at once.
    ///
    /// Used only when [`ProcessNoiseConfig::max_noise_dt`] sub-stepping is
    /// active (see [`noise_chunk_boundaries`]); the process noise is therefore
    /// always a continuous rate (`scale_with_dt == true`). Each chunk costs a
    /// full sigma-point pass — `2n + 1` propagations — so the total cost scales
    /// with the number of chunks. The propagator must be positioned at the
    /// interval start on entry; on success it is left at the final boundary.
    /// Returns the predicted `(state, P)`.
    ///
    /// This is an approximation using a constant `Q` per chunk; the exact
    /// variational discrete-time `Q_d` integration is tracked in issue #408.
    fn predict_substepped(
        &mut self,
        boundaries: &[Epoch],
        stopped_short: impl Fn(Epoch, Epoch) -> BraheError,
    ) -> Result<(DVector<f64>, DMatrix<f64>), BraheError> {
        let n = self.state_dim;
        let q = self
            .config
            .process_noise
            .as_ref()
            .expect("sub-stepping requires a process-noise configuration")
            .q_matrix
            .clone();
        let target = *boundaries
            .last()
            .expect("noise_chunk_boundaries never returns an empty Vec");

        let mut chunk_start = self.dynamics.current_epoch();
        let mut state = self.dynamics.current_state();
        let mut p = self.covariance.clone();

        for &chunk_target in boundaries {
            let dt_chunk = (chunk_target - chunk_start).abs();
            let sigma_points = Self::generate_sigma_points(&state, &p, self.lambda, n)?;

            let mut propagated_sigmas = Vec::with_capacity(2 * n + 1);
            for sp in &sigma_points {
                self.dynamics.reinitialize(chunk_start, sp.clone(), None);
                self.dynamics.propagate_to(chunk_target);

                let reached = self.dynamics.current_epoch();
                if (chunk_target - reached).abs() > 1e-6 {
                    return Err(stopped_short(reached, target));
                }
                propagated_sigmas.push(self.dynamics.current_state());
            }

            let mut new_state = DVector::zeros(n);
            for (i, sp) in propagated_sigmas.iter().enumerate() {
                new_state += self.weights_mean[i] * sp;
            }

            let mut new_p = DMatrix::zeros(n, n);
            for (i, sp) in propagated_sigmas.iter().enumerate() {
                let diff = sp - &new_state;
                new_p += self.weights_cov[i] * &diff * diff.transpose();
            }
            new_p += &q * dt_chunk;

            state = new_state;
            p = 0.5 * (&new_p + new_p.transpose());
            chunk_start = chunk_target;
        }

        Ok((state, p))
    }

    fn predict_and_update(
        &mut self,
        observation: &Observation,
        current_epoch: Epoch,
        dt: f64,
    ) -> Result<FilterRecord, BraheError> {
        let n = self.state_dim;
        let current_state = self.dynamics.current_state();

        // Propagator force-model / consider parameters, passed through to the
        // measurement model so consider values can affect the measurement.
        let params = self.dynamics.params().cloned();

        // === PREDICT (time update via sigma point propagation) ===
        // When continuous process-noise sub-stepping is active the interval is
        // split into chunks (a full unscented transform per chunk, noise
        // injected along the trajectory); otherwise a single end-to-end
        // transform reproduces the original behavior exactly.
        let (state_predicted, p_predicted) = match noise_chunk_boundaries(
            self.config.process_noise.as_ref(),
            current_epoch,
            observation.epoch,
        ) {
            Some(boundaries) => self.predict_substepped(&boundaries, |reached, target| {
                BraheError::Error(format!(
                    "Propagation stopped at {} before reaching observation epoch {} \
                     (a terminal event may have fired); the observation was not processed",
                    reached, target
                ))
            })?,
            None => {
                // Generate sigma points from current state and covariance
                let sigma_points =
                    Self::generate_sigma_points(&current_state, &self.covariance, self.lambda, n)?;

                // Propagate each sigma point through dynamics
                let mut propagated_sigmas = Vec::with_capacity(2 * n + 1);
                for sp in &sigma_points {
                    self.dynamics.reinitialize(current_epoch, sp.clone(), None);
                    self.dynamics.propagate_to(observation.epoch);

                    // Guard against the propagator stopping short of the
                    // observation epoch (e.g., a terminal event fired).
                    let reached_epoch = self.dynamics.current_epoch();
                    let epoch_gap: f64 = observation.epoch - reached_epoch;
                    if epoch_gap.abs() > 1e-6 {
                        return Err(BraheError::Error(format!(
                            "Propagation stopped at {} before reaching observation epoch {} \
                             (a terminal event may have fired); the observation was not processed",
                            reached_epoch, observation.epoch
                        )));
                    }

                    propagated_sigmas.push(self.dynamics.current_state());
                }

                // Compute predicted mean: x_pred = sum(W_m_i * chi_i)
                let mut state_predicted = DVector::zeros(n);
                for (i, sp) in propagated_sigmas.iter().enumerate() {
                    state_predicted += self.weights_mean[i] * sp;
                }

                // Compute predicted covariance:
                // P_pred = sum(W_c_i * (chi_i - x_pred)(chi_i - x_pred)^T) + Q
                let mut p_predicted = DMatrix::zeros(n, n);
                for (i, sp) in propagated_sigmas.iter().enumerate() {
                    let diff = sp - &state_predicted;
                    p_predicted += self.weights_cov[i] * &diff * diff.transpose();
                }

                // Add process noise
                if let Some(ref pn) = self.config.process_noise {
                    if pn.scale_with_dt {
                        let dt_abs = dt.abs().max(1e-12);
                        p_predicted += &pn.q_matrix * dt_abs;
                    } else {
                        p_predicted += &pn.q_matrix;
                    }
                }
                (state_predicted, p_predicted)
            }
        };

        // === UPDATE (measurement incorporation via unscented transform) ===

        let model = &self.measurement_models[observation.model_index];
        let m = model.measurement_dim();

        // Generate new sigma points from predicted state + covariance
        let update_sigmas =
            Self::generate_sigma_points(&state_predicted, &p_predicted, self.lambda, n)?;

        // Transform sigma points through measurement model. Validate every
        // prediction's length here, not just z_sigmas[0]: a model returning
        // inconsistent dimensions across sigma points would otherwise reach
        // the wrapped-deviation loops and panic in nalgebra rather than
        // surfacing as a structured error.
        let mut z_sigmas = Vec::with_capacity(2 * n + 1);
        for sp in &update_sigmas {
            let z_i = model.predict(&observation.epoch, sp, params.as_ref())?;
            if z_i.len() != m {
                return Err(BraheError::Error(format!(
                    "Model '{}' predict() returned {} elements, expected measurement_dim {}",
                    model.name(),
                    z_i.len(),
                    m
                )));
            }
            z_sigmas.push(z_i);
        }

        // Measurement noise covariance
        let r = model.noise_covariance();

        // Measurement models are a user-extension boundary; validate the
        // observation and noise-covariance shapes so mistakes surface as
        // errors rather than dimension panics.
        validate_model_outputs(
            model.as_ref(),
            &observation.measurement,
            &z_sigmas[0],
            None,
            &r,
            n,
        )?;

        // Predicted measurement mean via the reference-point trick:
        // z_pred = z_0 + sum(W_m_i * residual(Z_i, z_0)). Angular components
        // (azimuth) can straddle a wrap; differencing through residual() keeps
        // the mean well-defined. For plain-subtraction residuals this is
        // algebraically identical to the weighted sum since the weights sum
        // to 1, so linear-model UKF results are unchanged.
        let z_0 = z_sigmas[0].clone();
        let mut z_predicted = z_0.clone();
        for (i, z_i) in z_sigmas.iter().enumerate() {
            let dz = compute_residual(model.as_ref(), z_i, &z_0)?;
            z_predicted += self.weights_mean[i] * dz;
        }

        // Pre-fit residual (innovation)
        let prefit_residual =
            compute_residual(model.as_ref(), &observation.measurement, &z_predicted)?;

        // Wrapped measurement deviations dz_i = residual(Z_i, z_pred), computed
        // once per sigma point and reused in both S and the cross-covariance.
        let dz_all: Vec<DVector<f64>> = z_sigmas
            .iter()
            .map(|z_i| compute_residual(model.as_ref(), z_i, &z_predicted))
            .collect::<Result<_, _>>()?;

        // Innovation covariance: S = sum(W_c_i * dz_i * dz_i^T) + R
        let mut s = DMatrix::zeros(m, m);
        for (i, dz) in dz_all.iter().enumerate() {
            s += self.weights_cov[i] * dz * dz.transpose();
        }
        s += &r;

        // Cross-covariance: P_xz = sum(W_c_i * (chi_i - x_pred) * dz_i^T)
        let mut p_xz = DMatrix::zeros(n, m);
        for (i, (sp, dz)) in update_sigmas.iter().zip(dz_all.iter()).enumerate() {
            let dx = sp - &state_predicted;
            p_xz += self.weights_cov[i] * &dx * dz.transpose();
        }

        // Gain: K = P_xz * S⁻¹, computed via Cholesky solve of the symmetric
        // positive-definite S: Kᵀ = S⁻¹ · P_xzᵀ
        let s_chol = s.clone().cholesky().ok_or_else(|| {
            BraheError::NumericalError(
                "Innovation covariance matrix S is not positive-definite. Check \
                 measurement noise covariance R and predicted covariance P."
                    .to_string(),
            )
        })?;
        let k = s_chol.solve(&p_xz.transpose()).transpose();

        // State update: x_upd = x_pred + K * innovation
        let state_updated = &state_predicted + &k * &prefit_residual;

        // Covariance update: P_upd = P_pred - K * S * K^T
        let p_updated = &p_predicted - &k * &s * k.transpose();
        // Symmetrize to remove roundoff asymmetry from the matrix products
        let p_updated = 0.5 * (&p_updated + p_updated.transpose());

        // Post-fit residual
        let z_postfit = model.predict(&observation.epoch, &state_updated, params.as_ref())?;
        let postfit_residual =
            compute_residual(model.as_ref(), &observation.measurement, &z_postfit)?;

        // Build record
        let record = FilterRecord {
            epoch: observation.epoch,
            state_predicted: state_predicted.clone(),
            covariance_predicted: p_predicted,
            state_updated: state_updated.clone(),
            covariance_updated: p_updated.clone(),
            prefit_residual,
            postfit_residual,
            kalman_gain: k,
            measurement_name: model.name().to_string(),
        };

        // Update internal state. The covariance stays with the filter; the
        // propagator only carries the state between observations.
        self.covariance = p_updated;
        self.dynamics
            .reinitialize(observation.epoch, state_updated, None);

        Ok(record)
    }

    /// Process a batch of observations sequentially.
    ///
    /// Observations are auto-sorted by epoch before processing.
    pub fn process_observations(&mut self, observations: &[Observation]) -> Result<(), BraheError> {
        for obs in sort_by_epoch(observations) {
            self.process_observation(obs)?;
        }
        Ok(())
    }

    /// Propagate the filter to an epoch without a measurement update.
    ///
    /// Runs the sigma-point prediction step only: sigma points are
    /// propagated to `epoch`, the predicted mean and covariance are
    /// recomputed, and process noise is applied. Use this to advance the
    /// filter across measurement gaps while recording covariance growth.
    ///
    /// # Arguments
    ///
    /// * `epoch` - Target epoch. May be before or after the current filter
    ///   epoch; backwards propagation is supported (e.g. for smoothing).
    ///
    /// # Returns
    ///
    /// Filter record for the prediction step (measurement fields empty,
    /// `measurement_name` = `"Propagation"`).
    ///
    /// # Errors
    ///
    /// Returns an error if propagation stops short of the target epoch
    /// (e.g., a terminal event fired).
    pub fn propagate_to(&mut self, epoch: Epoch) -> Result<FilterRecord, BraheError> {
        let current_epoch = self.dynamics.current_epoch();
        let entry_state = self.dynamics.current_state();

        match self.predict_only(current_epoch, epoch) {
            Ok(record) => {
                if self.config.store_records {
                    self.records.push(record.clone());
                }
                Ok(record)
            }
            Err(e) => {
                self.dynamics.reinitialize(current_epoch, entry_state, None);
                Err(e)
            }
        }
    }

    /// Fallible body of [`propagate_to`](Self::propagate_to). Mutates the
    /// propagator; the caller rolls it back if this returns an error.
    ///
    /// Keep in sync with the predict half of `predict_and_update`.
    fn predict_only(
        &mut self,
        current_epoch: Epoch,
        target_epoch: Epoch,
    ) -> Result<FilterRecord, BraheError> {
        let dt: f64 = target_epoch - current_epoch;
        let n = self.state_dim;

        // When continuous process-noise sub-stepping is active the interval is
        // split into chunks (a full unscented transform per chunk); otherwise a
        // single end-to-end transform reproduces the original behavior exactly.
        let (state_predicted, p_predicted) = match noise_chunk_boundaries(
            self.config.process_noise.as_ref(),
            current_epoch,
            target_epoch,
        ) {
            Some(boundaries) => self.predict_substepped(&boundaries, |reached, target| {
                BraheError::Error(format!(
                    "Propagation stopped at {} before reaching target epoch {} \
                     (a terminal event may have fired)",
                    reached, target
                ))
            })?,
            None => {
                let current_state = self.dynamics.current_state();

                let sigma_points =
                    Self::generate_sigma_points(&current_state, &self.covariance, self.lambda, n)?;

                let mut propagated_sigmas = Vec::with_capacity(2 * n + 1);
                for sp in &sigma_points {
                    self.dynamics.reinitialize(current_epoch, sp.clone(), None);
                    self.dynamics.propagate_to(target_epoch);

                    let reached_epoch = self.dynamics.current_epoch();
                    let epoch_gap: f64 = target_epoch - reached_epoch;
                    if epoch_gap.abs() > 1e-6 {
                        return Err(BraheError::Error(format!(
                            "Propagation stopped at {} before reaching target epoch {} \
                             (a terminal event may have fired)",
                            reached_epoch, target_epoch
                        )));
                    }
                    propagated_sigmas.push(self.dynamics.current_state());
                }

                let mut state_predicted = DVector::zeros(n);
                for (i, sp) in propagated_sigmas.iter().enumerate() {
                    state_predicted += self.weights_mean[i] * sp;
                }

                let mut p_predicted = DMatrix::zeros(n, n);
                for (i, sp) in propagated_sigmas.iter().enumerate() {
                    let diff = sp - &state_predicted;
                    p_predicted += self.weights_cov[i] * &diff * diff.transpose();
                }

                // Apply process noise only for a nonzero-duration step; a
                // same-epoch propagate_to adds zero noise. Scaled Q uses |dt|
                // so backwards propagation accumulates the same process noise
                // as forwards.
                if dt.abs() > 0.0
                    && let Some(ref pn) = self.config.process_noise
                {
                    if pn.scale_with_dt {
                        p_predicted += &pn.q_matrix * dt.abs();
                    } else {
                        p_predicted += &pn.q_matrix;
                    }
                }
                let p_predicted = 0.5 * (&p_predicted + p_predicted.transpose());
                (state_predicted, p_predicted)
            }
        };

        let record = FilterRecord {
            epoch: target_epoch,
            state_predicted: state_predicted.clone(),
            covariance_predicted: p_predicted.clone(),
            state_updated: state_predicted.clone(),
            covariance_updated: p_predicted.clone(),
            prefit_residual: DVector::zeros(0),
            postfit_residual: DVector::zeros(0),
            kalman_gain: DMatrix::zeros(0, 0),
            measurement_name: "Propagation".to_string(),
        };

        self.covariance = p_predicted;
        self.dynamics
            .reinitialize(target_epoch, state_predicted, None);

        Ok(record)
    }

    /// Get current state estimate.
    pub fn current_state(&self) -> DVector<f64> {
        self.dynamics.current_state()
    }

    /// Get current covariance estimate.
    pub fn current_covariance(&self) -> &DMatrix<f64> {
        &self.covariance
    }

    /// Get current epoch.
    pub fn current_epoch(&self) -> Epoch {
        self.dynamics.current_epoch()
    }

    /// Get all stored filter records.
    ///
    /// Only populated when `config.store_records` is `true`.
    pub fn records(&self) -> &[FilterRecord] {
        &self.records
    }

    /// Immutable access to the underlying dynamics source.
    pub fn dynamics(&self) -> &DynamicsSource {
        &self.dynamics
    }

    /// Consume the filter, returning the underlying dynamics source.
    pub fn into_dynamics(self) -> DynamicsSource {
        self.dynamics
    }

    /// Create a builder for [`UnscentedKalmanFilter`].
    ///
    /// Takes the required inputs directly; optional inputs are provided
    /// through chained setters on the returned builder and default to `None`
    /// / empty ([`NumericalPropagationConfig::default`] for the propagation
    /// configuration, no measurement models).
    ///
    /// # Arguments
    ///
    /// * `epoch` - Initial epoch
    /// * `state` - Initial state vector (meters, m/s)
    /// * `initial_covariance` - Initial covariance matrix (state_dim x state_dim)
    /// * `force_config` - Force model configuration
    /// * `config` - UKF configuration (alpha, beta, kappa)
    ///
    /// # Returns
    ///
    /// Builder with the required fields set and all optional fields defaulted
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::estimation::*;
    /// use brahe::propagators::*;
    /// use brahe::eop::{StaticEOPProvider, set_global_eop_provider};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use nalgebra::{DMatrix, DVector};
    ///
    /// let eop = StaticEOPProvider::from_zero();
    /// set_global_eop_provider(eop);
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = DVector::from_vec(vec![6878e3, 0.0, 0.0, 0.0, 7612.0, 0.0]);
    /// let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
    ///
    /// let ukf = UnscentedKalmanFilter::builder(
    ///     epoch, state, p0, ForceModelConfig::two_body_gravity(), UKFConfig::default(),
    /// )
    /// .measurement_model(Box::new(InertialPositionMeasurementModel::new(10.0)))
    /// .build()
    /// .unwrap();
    /// ```
    pub fn builder(
        epoch: Epoch,
        state: DVector<f64>,
        initial_covariance: DMatrix<f64>,
        force_config: ForceModelConfig,
        config: UKFConfig,
    ) -> UnscentedKalmanFilterBuilder {
        UnscentedKalmanFilterBuilder {
            epoch,
            state,
            initial_covariance,
            force_config,
            config,
            propagation_config: NumericalPropagationConfig::default(),
            params: None,
            additional_dynamics: None,
            control_input: None,
            measurement_models: Vec::new(),
        }
    }
}

/// Builder for [`UnscentedKalmanFilter`].
///
/// Created by [`UnscentedKalmanFilter::builder`], which takes the required
/// inputs (`epoch`, `state`, `initial_covariance`, `force_config`, `config`).
/// Remaining inputs are provided through chained setters and default to
/// `None` / empty ([`NumericalPropagationConfig::default`] for the
/// propagation configuration, no measurement models).
/// [`UnscentedKalmanFilterBuilder::build`] delegates to [`UnscentedKalmanFilter::new`].
///
/// # Example
///
/// ```rust
/// use brahe::estimation::*;
/// use brahe::propagators::*;
/// use brahe::eop::{StaticEOPProvider, set_global_eop_provider};
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::{DMatrix, DVector};
///
/// let eop = StaticEOPProvider::from_zero();
/// set_global_eop_provider(eop);
///
/// let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let state = DVector::from_vec(vec![6878e3, 0.0, 0.0, 0.0, 7612.0, 0.0]);
/// let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
///
/// let ukf = UnscentedKalmanFilter::builder(
///     epoch, state, p0, ForceModelConfig::two_body_gravity(), UKFConfig::default(),
/// )
/// .measurement_model(Box::new(InertialPositionMeasurementModel::new(10.0)))
/// .build()
/// .unwrap();
/// ```
pub struct UnscentedKalmanFilterBuilder {
    epoch: Epoch,
    state: DVector<f64>,
    initial_covariance: DMatrix<f64>,
    force_config: ForceModelConfig,
    config: UKFConfig,
    propagation_config: NumericalPropagationConfig,
    params: Option<DVector<f64>>,
    additional_dynamics: Option<DStateDynamics>,
    control_input: DControlInput,
    measurement_models: Vec<Box<dyn MeasurementModel>>,
}

impl UnscentedKalmanFilterBuilder {
    /// Set the propagation configuration (integrator method, tolerances, and step sizes).
    ///
    /// Defaults to [`NumericalPropagationConfig::default`] if not called.
    ///
    /// # Arguments
    /// * `config` - Numerical propagation configuration
    ///
    /// # Returns
    /// Builder for method chaining
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::estimation::*;
    /// use brahe::propagators::*;
    /// use brahe::eop::{StaticEOPProvider, set_global_eop_provider};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use nalgebra::{DMatrix, DVector};
    ///
    /// let eop = StaticEOPProvider::from_zero();
    /// set_global_eop_provider(eop);
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = DVector::from_vec(vec![6878e3, 0.0, 0.0, 0.0, 7612.0, 0.0]);
    /// let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
    ///
    /// let ukf = UnscentedKalmanFilter::builder(
    ///     epoch, state, p0, ForceModelConfig::two_body_gravity(), UKFConfig::default(),
    /// )
    /// .propagation_config(NumericalPropagationConfig::default())
    /// .measurement_model(Box::new(InertialPositionMeasurementModel::new(10.0)))
    /// .build()
    /// .unwrap();
    /// ```
    pub fn propagation_config(mut self, config: NumericalPropagationConfig) -> Self {
        self.propagation_config = config;
        self
    }

    /// Set the parameter vector `[mass, drag_area, Cd, srp_area, Cr, ...]`.
    ///
    /// Required when `force_config` references parameter indices for drag or SRP.
    ///
    /// # Arguments
    /// * `params` - Parameter vector
    ///
    /// # Returns
    /// Builder for method chaining
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::estimation::*;
    /// use brahe::propagators::*;
    /// use brahe::eop::{StaticEOPProvider, set_global_eop_provider};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use nalgebra::{DMatrix, DVector};
    ///
    /// let eop = StaticEOPProvider::from_zero();
    /// set_global_eop_provider(eop);
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = DVector::from_vec(vec![6878e3, 0.0, 0.0, 0.0, 7612.0, 0.0]);
    /// let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
    ///
    /// let ukf = UnscentedKalmanFilter::builder(
    ///     epoch, state, p0, ForceModelConfig::two_body_gravity(), UKFConfig::default(),
    /// )
    /// .params(DVector::from_vec(vec![10.0, 2.2]))
    /// .measurement_model(Box::new(InertialPositionMeasurementModel::new(10.0)))
    /// .build()
    /// .unwrap();
    /// ```
    pub fn params(mut self, params: DVector<f64>) -> Self {
        self.params = Some(params);
        self
    }

    /// Set additional dynamics for extended state dimensions beyond the orbital state.
    ///
    /// # Arguments
    /// * `dynamics` - Function computing derivatives for extra state elements
    ///
    /// # Returns
    /// Builder for method chaining
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::estimation::*;
    /// use brahe::propagators::*;
    /// use brahe::eop::{StaticEOPProvider, set_global_eop_provider};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use nalgebra::{DMatrix, DVector};
    ///
    /// let eop = StaticEOPProvider::from_zero();
    /// set_global_eop_provider(eop);
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    /// // 6D orbital state + 1 mass state
    /// let state = DVector::from_vec(vec![6878e3, 0.0, 0.0, 0.0, 7612.0, 0.0, 1000.0]);
    /// let p0 = DMatrix::<f64>::identity(7, 7);
    ///
    /// let dynamics: brahe::integrators::traits::DStateDynamics = Box::new(|_t, state, _params| {
    ///     let mut dx = DVector::zeros(state.len());
    ///     dx[6] = -0.1; // dm/dt = -0.1 kg/s
    ///     dx
    /// });
    ///
    /// let ukf = UnscentedKalmanFilter::builder(
    ///     epoch, state, p0, ForceModelConfig::two_body_gravity(), UKFConfig::default(),
    /// )
    /// .additional_dynamics(dynamics)
    /// .measurement_model(Box::new(InertialPositionMeasurementModel::new(10.0)))
    /// .build()
    /// .unwrap();
    /// ```
    pub fn additional_dynamics(mut self, dynamics: DStateDynamics) -> Self {
        self.additional_dynamics = Some(dynamics);
        self
    }

    /// Set a continuous control-input function that adds an acceleration perturbation.
    ///
    /// # Arguments
    /// * `control` - Control function returning an acceleration perturbation vector
    ///
    /// # Returns
    /// Builder for method chaining
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::estimation::*;
    /// use brahe::propagators::*;
    /// use brahe::eop::{StaticEOPProvider, set_global_eop_provider};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use nalgebra::{DMatrix, DVector};
    ///
    /// let eop = StaticEOPProvider::from_zero();
    /// set_global_eop_provider(eop);
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = DVector::from_vec(vec![6878e3, 0.0, 0.0, 0.0, 7612.0, 0.0]);
    /// let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
    ///
    /// let control: brahe::integrators::traits::DStateDynamics = Box::new(|_t, state, _params| {
    ///     let mut dx = DVector::zeros(state.len());
    ///     dx[3] = 0.0001; // small perturbing acceleration
    ///     dx
    /// });
    ///
    /// let ukf = UnscentedKalmanFilter::builder(
    ///     epoch, state, p0, ForceModelConfig::two_body_gravity(), UKFConfig::default(),
    /// )
    /// .control_input(control)
    /// .measurement_model(Box::new(InertialPositionMeasurementModel::new(10.0)))
    /// .build()
    /// .unwrap();
    /// ```
    pub fn control_input(mut self, control: DStateDynamics) -> Self {
        self.control_input = Some(control);
        self
    }

    /// Append a measurement model.
    ///
    /// Call multiple times to register multiple measurement types;
    /// [`Observation`](super::types::Observation)'s `model_index` selects among them.
    ///
    /// # Arguments
    /// * `model` - Measurement model to append
    ///
    /// # Returns
    /// Builder for method chaining
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::estimation::*;
    /// use brahe::propagators::*;
    /// use brahe::eop::{StaticEOPProvider, set_global_eop_provider};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use nalgebra::{DMatrix, DVector};
    ///
    /// let eop = StaticEOPProvider::from_zero();
    /// set_global_eop_provider(eop);
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = DVector::from_vec(vec![6878e3, 0.0, 0.0, 0.0, 7612.0, 0.0]);
    /// let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
    ///
    /// let ukf = UnscentedKalmanFilter::builder(
    ///     epoch, state, p0, ForceModelConfig::two_body_gravity(), UKFConfig::default(),
    /// )
    /// .measurement_model(Box::new(InertialPositionMeasurementModel::new(10.0)))
    /// .measurement_model(Box::new(InertialStateMeasurementModel::new(10.0, 0.01)))
    /// .build()
    /// .unwrap();
    /// ```
    pub fn measurement_model(mut self, model: Box<dyn MeasurementModel>) -> Self {
        self.measurement_models.push(model);
        self
    }

    /// Replace the full list of measurement models.
    ///
    /// # Arguments
    /// * `models` - Measurement models, replacing any previously set
    ///
    /// # Returns
    /// Builder for method chaining
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::estimation::*;
    /// use brahe::propagators::*;
    /// use brahe::eop::{StaticEOPProvider, set_global_eop_provider};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use nalgebra::{DMatrix, DVector};
    ///
    /// let eop = StaticEOPProvider::from_zero();
    /// set_global_eop_provider(eop);
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = DVector::from_vec(vec![6878e3, 0.0, 0.0, 0.0, 7612.0, 0.0]);
    /// let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
    ///
    /// let ukf = UnscentedKalmanFilter::builder(
    ///     epoch, state, p0, ForceModelConfig::two_body_gravity(), UKFConfig::default(),
    /// )
    /// .measurement_models(vec![Box::new(InertialPositionMeasurementModel::new(10.0))])
    /// .build()
    /// .unwrap();
    /// ```
    pub fn measurement_models(mut self, models: Vec<Box<dyn MeasurementModel>>) -> Self {
        self.measurement_models = models;
        self
    }

    /// Construct the filter from the accumulated configuration.
    ///
    /// # Returns
    /// Initialized filter ready to process observations
    ///
    /// # Errors
    /// Returns `BraheError` if no measurement models were provided, if the
    /// covariance dimensions do not match the state, if the sigma-point
    /// parameters are invalid, or if the underlying propagator construction
    /// fails (see [`UnscentedKalmanFilter::new`]).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::estimation::*;
    /// use brahe::propagators::*;
    /// use brahe::eop::{StaticEOPProvider, set_global_eop_provider};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use nalgebra::{DMatrix, DVector};
    ///
    /// let eop = StaticEOPProvider::from_zero();
    /// set_global_eop_provider(eop);
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = DVector::from_vec(vec![6878e3, 0.0, 0.0, 0.0, 7612.0, 0.0]);
    /// let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
    ///
    /// let ukf = UnscentedKalmanFilter::builder(
    ///     epoch, state, p0, ForceModelConfig::two_body_gravity(), UKFConfig::default(),
    /// )
    /// .measurement_model(Box::new(InertialPositionMeasurementModel::new(10.0)))
    /// .build()
    /// .unwrap();
    /// ```
    pub fn build(self) -> Result<UnscentedKalmanFilter, BraheError> {
        UnscentedKalmanFilter::new(
            self.epoch,
            self.state,
            self.initial_covariance,
            self.propagation_config,
            self.force_config,
            self.params,
            self.additional_dynamics,
            self.control_input,
            self.measurement_models,
            self.config,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::physical::GM_EARTH;
    use crate::eop::{EOPExtrapolation, FileEOPProvider, set_global_eop_provider};
    use crate::estimation::{
        InertialPositionMeasurementModel, InertialStateMeasurementModel, ProcessNoiseConfig,
    };
    use crate::propagators::DNumericalOrbitPropagator;
    use crate::propagators::NumericalPropagationConfig;
    use crate::propagators::force_model_config::ForceModelConfig;
    use crate::propagators::traits::DStatePropagator;
    use crate::time::TimeSystem;
    use approx::assert_abs_diff_eq;
    use serial_test::serial;

    fn setup_global_test_eop() {
        let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
        set_global_eop_provider(eop);
    }

    fn two_body_leo() -> (Epoch, DVector<f64>) {
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let r = 6878.0e3;
        let v = (GM_EARTH / r).sqrt();
        (epoch, DVector::from_vec(vec![r, 0.0, 0.0, 0.0, v, 0.0]))
    }

    fn generate_position_observations(
        epoch: Epoch,
        true_state: &DVector<f64>,
        num_obs: usize,
        interval_s: f64,
    ) -> Vec<Observation> {
        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            true_state.clone(),
            NumericalPropagationConfig::default(),
            ForceModelConfig::two_body_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        (1..=num_obs)
            .map(|i| {
                let t = epoch + (i as f64) * interval_s;
                prop.propagate_to(t);
                Observation::new(t, prop.current_state().rows(0, 3).into_owned(), 0)
            })
            .collect()
    }

    fn create_two_body_ukf(
        epoch: Epoch,
        state: DVector<f64>,
        p0: DMatrix<f64>,
        models: Vec<Box<dyn MeasurementModel>>,
    ) -> UnscentedKalmanFilter {
        UnscentedKalmanFilter::new(
            epoch,
            state,
            p0,
            NumericalPropagationConfig::default(),
            ForceModelConfig::two_body_gravity(),
            None,
            None,
            None,
            models,
            UKFConfig::default(),
        )
        .unwrap()
    }

    /// Create a plain propagator (no STM, no covariance) for from_propagator tests.
    fn create_plain_propagator(
        epoch: Epoch,
        state: DVector<f64>,
        covariance: Option<DMatrix<f64>>,
    ) -> DNumericalOrbitPropagator {
        DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfig::two_body_gravity(),
            None,
            None,
            None,
            covariance,
        )
        .unwrap()
    }

    fn truth_at(epoch: Epoch, true_state: &DVector<f64>, target: Epoch) -> DVector<f64> {
        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            true_state.clone(),
            NumericalPropagationConfig::default(),
            ForceModelConfig::two_body_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        prop.propagate_to(target);
        prop.current_state()
    }

    // =========================================================================
    // Construction tests
    // =========================================================================

    #[test]
    #[serial]
    fn test_ukf_construction_valid() {
        setup_global_test_eop();
        let (epoch, true_state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        let models: Vec<Box<dyn MeasurementModel>> =
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))];

        let ukf = create_two_body_ukf(epoch, true_state.clone(), p0, models);
        assert_eq!(ukf.current_state().len(), 6);
        assert_eq!(ukf.current_covariance().nrows(), 6);
    }

    #[test]
    #[serial]
    fn test_ukf_from_propagator_no_covariance_errors() {
        // A propagator without an initial covariance cannot seed the filter
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let prop = create_plain_propagator(epoch, state, None);

        let models: Vec<Box<dyn MeasurementModel>> =
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))];

        let result = UnscentedKalmanFilter::from_propagator(prop, models, UKFConfig::default());
        match result {
            Err(e) => assert!(
                e.to_string().contains("initial covariance"),
                "Error should mention initial covariance: {}",
                e
            ),
            Ok(_) => panic!("Expected error for missing covariance"),
        }
    }

    #[test]
    #[serial]
    fn test_ukf_covariance_dim_mismatch_errors() {
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();

        let result = UnscentedKalmanFilter::new(
            epoch,
            state,
            DMatrix::identity(4, 4), // wrong dimensions for 6D state
            NumericalPropagationConfig::default(),
            ForceModelConfig::two_body_gravity(),
            None,
            None,
            None,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            UKFConfig::default(),
        );
        match result {
            Err(e) => assert!(
                e.to_string().contains("dimensions"),
                "Error should mention dimensions: {}",
                e
            ),
            Ok(_) => panic!("Expected error for covariance dimension mismatch"),
        }
    }

    #[test]
    #[serial]
    fn test_ukf_from_propagator_no_models_errors() {
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let prop = create_plain_propagator(epoch, state, Some(p0));

        let result = UnscentedKalmanFilter::from_propagator(
            prop,
            vec![], // no models
            UKFConfig::default(),
        );
        match result {
            Err(e) => assert!(
                e.to_string().contains("measurement model"),
                "Error should mention measurement model: {}",
                e
            ),
            Ok(_) => panic!("Expected error for no models"),
        }
    }

    #[test]
    #[serial]
    fn test_ukf_invalid_sigma_parameters_error() {
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        // alpha <= 0
        let result = UnscentedKalmanFilter::from_propagator(
            create_plain_propagator(epoch, state.clone(), Some(p0.clone())),
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            UKFConfig {
                alpha: 0.0,
                ..UKFConfig::default()
            },
        );
        match result {
            Err(e) => assert!(
                e.to_string().contains("alpha"),
                "Error should mention alpha: {}",
                e
            ),
            Ok(_) => panic!("Expected error for alpha <= 0"),
        }

        // n + kappa <= 0 (kappa = -6 for a 6D state)
        let result = UnscentedKalmanFilter::from_propagator(
            create_plain_propagator(epoch, state, Some(p0)),
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            UKFConfig {
                kappa: -6.0,
                ..UKFConfig::default()
            },
        );
        match result {
            Err(e) => assert!(
                e.to_string().contains("kappa"),
                "Error should mention kappa: {}",
                e
            ),
            Ok(_) => panic!("Expected error for kappa"),
        }
    }

    #[test]
    #[serial]
    fn test_ukf_non_finite_sigma_parameters_error() {
        // NaN compares false against every guard threshold, so explicit
        // finiteness checks are required; a tiny alpha underflows alpha^2 to
        // zero, collapsing the sigma-point scaling denominator.
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let models = || -> Vec<Box<dyn MeasurementModel>> {
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))]
        };

        for config in [
            UKFConfig {
                alpha: f64::NAN,
                ..UKFConfig::default()
            },
            UKFConfig {
                kappa: f64::NAN,
                ..UKFConfig::default()
            },
            UKFConfig {
                beta: f64::NAN,
                ..UKFConfig::default()
            },
            UKFConfig {
                alpha: 1e-200, // alpha^2 underflows to 0.0
                ..UKFConfig::default()
            },
        ] {
            let result = UnscentedKalmanFilter::from_propagator(
                create_plain_propagator(epoch, state.clone(), Some(p0.clone())),
                models(),
                config.clone(),
            );
            assert!(
                result.is_err(),
                "Expected construction error for alpha={}, beta={}, kappa={}",
                config.alpha,
                config.beta,
                config.kappa
            );
        }
    }

    #[test]
    #[serial]
    fn test_ukf_from_propagator_disables_stm() {
        // Providing a covariance to a propagator auto-enables STM propagation,
        // which the UKF never uses; from_propagator must disable it so
        // sigma-point propagation does not integrate variational equations.
        setup_global_test_eop();
        let (epoch, true_state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        let prop = create_plain_propagator(epoch, true_state.clone(), Some(p0));
        assert!(prop.has_stm(), "covariance should have auto-enabled STM");

        let mut ukf = UnscentedKalmanFilter::from_propagator(
            prop,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            UKFConfig::default(),
        )
        .unwrap();
        assert!(
            !ukf.dynamics().has_stm(),
            "UKF must disable STM propagation on its propagator"
        );

        // The filter still works end-to-end
        let obs = generate_position_observations(epoch, &true_state, 3, 60.0);
        ukf.process_observations(&obs).unwrap();
        assert_eq!(ukf.current_covariance().nrows(), 6);
    }

    #[test]
    #[serial]
    fn test_ukf_terminal_event_before_observation_errors() {
        use crate::events::DTimeEvent;

        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let mut prop = create_plain_propagator(epoch, state.clone(), Some(p0));
        prop.add_event_detector(Box::new(
            DTimeEvent::new(epoch + 30.0, "Terminal").set_terminal(),
        ));

        let mut ukf = UnscentedKalmanFilter::from_propagator(
            prop,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            UKFConfig::default(),
        )
        .unwrap();

        let obs = Observation::new(epoch + 60.0, state.rows(0, 3).into_owned(), 0);
        match ukf.process_observation(&obs) {
            Err(e) => assert!(
                e.to_string().contains("before reaching observation epoch"),
                "Error should mention stopping early: {}",
                e
            ),
            Ok(_) => panic!("Expected error when terminal event stops propagation"),
        }

        // The failed observation must not have moved the filter, and the
        // propagator must hold the mean state, not a sigma point
        let epoch_drift: f64 = ukf.current_epoch() - epoch;
        assert!(
            epoch_drift.abs() < 1e-9,
            "Filter epoch must be unchanged after a terminal-event error, drifted {} s",
            epoch_drift
        );
        assert_abs_diff_eq!(ukf.current_state(), state, epsilon = 1e-9);
    }

    /// Model that mis-shapes one of its outputs relative to its declared
    /// measurement_dim of 3 — exercises the model-output shape validation.
    struct MisshapenModel {
        bad_predict: bool,
        bad_noise: bool,
    }
    impl MeasurementModel for MisshapenModel {
        fn predict(
            &self,
            _epoch: &Epoch,
            state: &DVector<f64>,
            _params: Option<&DVector<f64>>,
        ) -> Result<DVector<f64>, BraheError> {
            let m = if self.bad_predict { 2 } else { 3 };
            Ok(state.rows(0, m).into_owned())
        }
        fn noise_covariance(&self) -> DMatrix<f64> {
            let m = if self.bad_noise { 2 } else { 3 };
            DMatrix::identity(m, m) * 100.0
        }
        fn measurement_dim(&self) -> usize {
            3
        }
        fn name(&self) -> &str {
            "Misshapen"
        }
    }

    #[test]
    #[serial]
    fn test_ukf_model_output_shape_errors() {
        // Malformed measurement-model outputs must surface as structured
        // errors, not dimension panics, and must leave the filter rolled
        // back at its pre-observation epoch.
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        let cases: Vec<(MisshapenModel, &str)> = vec![
            (
                MisshapenModel {
                    bad_predict: true,
                    bad_noise: false,
                },
                "predict() returned",
            ),
            (
                MisshapenModel {
                    bad_predict: false,
                    bad_noise: true,
                },
                "noise_covariance() is",
            ),
        ];

        for (model, expected_msg) in cases {
            let mut ukf =
                create_two_body_ukf(epoch, state.clone(), p0.clone(), vec![Box::new(model)]);

            let obs = Observation::new(epoch + 60.0, state.rows(0, 3).into_owned(), 0);
            match ukf.process_observation(&obs) {
                Err(e) => assert!(
                    e.to_string().contains(expected_msg),
                    "Error should contain '{}': {}",
                    expected_msg,
                    e
                ),
                Ok(_) => panic!("Expected shape validation error for '{}'", expected_msg),
            }

            let epoch_drift: f64 = ukf.current_epoch() - epoch;
            assert!(
                epoch_drift.abs() < 1e-9,
                "Filter must be rolled back after shape error, drifted {} s",
                epoch_drift
            );
        }
    }

    /// Model that returns the declared dimension for the first sigma-point
    /// prediction but a shorter vector for later ones — exercises the
    /// per-sigma-point length validation (a bug would panic in the wrapped
    /// deviation loops instead of returning an error).
    struct InconsistentSigmaDimModel {
        calls: std::sync::atomic::AtomicUsize,
    }
    impl MeasurementModel for InconsistentSigmaDimModel {
        fn predict(
            &self,
            _epoch: &Epoch,
            state: &DVector<f64>,
            _params: Option<&DVector<f64>>,
        ) -> Result<DVector<f64>, BraheError> {
            let call = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            // First sigma point (z_0) is the declared length; later ones short.
            let m = if call == 0 { 3 } else { 2 };
            Ok(state.rows(0, m).into_owned())
        }
        fn noise_covariance(&self) -> DMatrix<f64> {
            DMatrix::identity(3, 3) * 100.0
        }
        fn measurement_dim(&self) -> usize {
            3
        }
        fn name(&self) -> &str {
            "InconsistentSigmaDim"
        }
    }

    #[test]
    #[serial]
    fn test_ukf_inconsistent_sigma_point_dim_errors() {
        // A model whose predict() length varies across sigma points must
        // surface as a structured error naming the model, not an nalgebra
        // dimension panic, and must leave the filter rolled back.
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        let mut ukf = create_two_body_ukf(
            epoch,
            state.clone(),
            p0,
            vec![Box::new(InconsistentSigmaDimModel {
                calls: std::sync::atomic::AtomicUsize::new(0),
            })],
        );

        let obs = Observation::new(epoch + 60.0, state.rows(0, 3).into_owned(), 0);
        match ukf.process_observation(&obs) {
            Err(e) => assert!(
                e.to_string().contains("predict() returned 2 elements"),
                "Error should name the mismatched sigma-point dimension: {}",
                e
            ),
            Ok(_) => panic!("Expected an error for inconsistent sigma-point dimensions"),
        }

        let epoch_drift: f64 = ukf.current_epoch() - epoch;
        assert!(
            epoch_drift.abs() < 1e-9,
            "Filter must be rolled back after the error, drifted {} s",
            epoch_drift
        );
    }

    /// Model that requires the propagator's parameter vector to be passed
    /// through — errors if params are missing or wrong.
    struct ParamsAssertingModel {
        expected: DVector<f64>,
    }
    impl MeasurementModel for ParamsAssertingModel {
        fn predict(
            &self,
            _epoch: &Epoch,
            state: &DVector<f64>,
            params: Option<&DVector<f64>>,
        ) -> Result<DVector<f64>, BraheError> {
            match params {
                Some(p) if p == &self.expected => Ok(state.rows(0, 3).into_owned()),
                other => Err(BraheError::Error(format!(
                    "expected params Some({:?}), got {:?}",
                    self.expected, other
                ))),
            }
        }
        fn noise_covariance(&self) -> DMatrix<f64> {
            DMatrix::identity(3, 3) * 100.0
        }
        fn measurement_dim(&self) -> usize {
            3
        }
        fn name(&self) -> &str {
            "ParamsAsserting"
        }
    }

    #[test]
    #[serial]
    fn test_ukf_passes_propagator_params_to_measurement_models() {
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let expected = DVector::from_vec(vec![2.2, 1.3]);

        let mut ukf = UnscentedKalmanFilter::new(
            epoch,
            state.clone(),
            p0,
            NumericalPropagationConfig::default(),
            ForceModelConfig::two_body_gravity(),
            Some(expected.clone()),
            None,
            None,
            vec![Box::new(ParamsAssertingModel { expected })],
            UKFConfig::default(),
        )
        .unwrap();

        let obs = Observation::new(epoch + 60.0, state.rows(0, 3).into_owned(), 0);
        ukf.process_observation(&obs)
            .expect("model should receive the propagator params");
    }

    /// Model that always fails predict() — exercises error paths that occur
    /// after the propagator has been advanced (mid sigma-point update the
    /// propagator otherwise holds an arbitrary sigma point, not the mean).
    struct FailingModel;
    impl MeasurementModel for FailingModel {
        fn predict(
            &self,
            _epoch: &Epoch,
            _state: &DVector<f64>,
            _params: Option<&DVector<f64>>,
        ) -> Result<DVector<f64>, BraheError> {
            Err(BraheError::Error("intentional model failure".to_string()))
        }
        fn noise_covariance(&self) -> DMatrix<f64> {
            DMatrix::identity(3, 3)
        }
        fn measurement_dim(&self) -> usize {
            3
        }
        fn name(&self) -> &str {
            "Failing"
        }
    }

    #[test]
    #[serial]
    fn test_ukf_failed_observation_leaves_filter_consistent() {
        // A failed observation must not desynchronize the filter: the
        // propagator must be rolled back (otherwise it holds a sigma point,
        // not the state estimate), and a retry must produce the same result
        // as a filter that never saw the failure.
        setup_global_test_eop();
        let (epoch, true_state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let obs = generate_position_observations(epoch, &true_state, 2, 60.0);

        let mut perturbed = true_state.clone();
        perturbed[0] += 500.0;

        let make_ukf = |models: Vec<Box<dyn MeasurementModel>>| {
            UnscentedKalmanFilter::new(
                epoch,
                perturbed.clone(),
                p0.clone(),
                NumericalPropagationConfig::default(),
                ForceModelConfig::two_body_gravity(),
                None,
                None,
                None,
                models,
                UKFConfig::default(),
            )
            .unwrap()
        };

        let mut ukf = make_ukf(vec![
            Box::new(InertialPositionMeasurementModel::new(10.0)),
            Box::new(FailingModel),
        ]);
        let mut reference = make_ukf(vec![Box::new(InertialPositionMeasurementModel::new(10.0))]);

        ukf.process_observation(&obs[0]).unwrap();
        reference.process_observation(&obs[0]).unwrap();

        // Failing observation at the next epoch must error and leave the
        // filter at the previous epoch with the mean state
        let bad_obs = Observation::new(obs[1].epoch, obs[1].measurement.clone(), 1);
        assert!(ukf.process_observation(&bad_obs).is_err());
        let epoch_drift: f64 = ukf.current_epoch() - obs[0].epoch;
        assert!(
            epoch_drift.abs() < 1e-9,
            "Filter epoch must be unchanged after a failed observation, drifted {} s",
            epoch_drift
        );

        // Retrying with the working model must match the reference filter
        let rec = ukf.process_observation(&obs[1]).unwrap();
        let rec_ref = reference.process_observation(&obs[1]).unwrap();
        assert_abs_diff_eq!(rec.state_updated, rec_ref.state_updated, epsilon = 1e-9);
        assert_abs_diff_eq!(
            rec.covariance_updated,
            rec_ref.covariance_updated,
            epsilon = 1e-9
        );
    }

    #[test]
    #[serial]
    fn test_ukf_does_not_enable_stm() {
        // The UKF captures uncertainty via sigma points; its internal
        // propagator must not integrate variational equations.
        setup_global_test_eop();
        let (epoch, true_state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        let models: Vec<Box<dyn MeasurementModel>> =
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))];

        let ukf = create_two_body_ukf(epoch, true_state, p0, models);
        assert!(
            !ukf.dynamics().has_stm(),
            "UKF-built propagator must not have STM propagation enabled"
        );
    }

    // =========================================================================
    // Convergence tests
    // =========================================================================

    #[test]
    #[serial]
    fn test_ukf_converges_from_position_offset() {
        setup_global_test_eop();
        let (epoch, true_state) = two_body_leo();

        // Add 1 km position offset
        let mut state = true_state.clone();
        state[0] += 1000.0;

        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        let models: Vec<Box<dyn MeasurementModel>> =
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))];

        let mut ukf = create_two_body_ukf(epoch, state, p0, models);
        let observations = generate_position_observations(epoch, &true_state, 20, 60.0);
        ukf.process_observations(&observations).unwrap();

        // Check convergence
        let truth_final = truth_at(epoch, &true_state, ukf.current_epoch());
        let pos_error = (ukf.current_state().rows(0, 3) - truth_final.rows(0, 3)).norm();
        assert!(
            pos_error < 100.0,
            "Position error {} m should be < 100 m after 20 observations",
            pos_error
        );
    }

    #[test]
    #[serial]
    fn test_ukf_converges_with_state_measurements() {
        setup_global_test_eop();
        let (epoch, true_state) = two_body_leo();

        let mut state = true_state.clone();
        state[0] += 1000.0;
        state[4] += 1.0;

        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        let models: Vec<Box<dyn MeasurementModel>> =
            vec![Box::new(InertialStateMeasurementModel::new(10.0, 0.1))];

        let mut ukf = create_two_body_ukf(epoch, state, p0, models);

        // Generate state observations
        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            true_state.clone(),
            NumericalPropagationConfig::default(),
            ForceModelConfig::two_body_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let observations: Vec<Observation> = (1..=20)
            .map(|i| {
                let t = epoch + (i as f64) * 60.0;
                prop.propagate_to(t);
                Observation::new(t, prop.current_state().rows(0, 6).into_owned(), 0)
            })
            .collect();

        ukf.process_observations(&observations).unwrap();

        let truth_final = truth_at(epoch, &true_state, ukf.current_epoch());
        let state_error = (ukf.current_state() - truth_final).norm();
        assert!(
            state_error < 10.0,
            "State error {} should be < 10 after full state observations",
            state_error
        );
    }

    // =========================================================================
    // API behavior tests
    // =========================================================================

    #[test]
    #[serial]
    fn test_ukf_backward_observation_rejected() {
        setup_global_test_eop();
        let (epoch, true_state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        let models: Vec<Box<dyn MeasurementModel>> =
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))];

        let mut ukf = create_two_body_ukf(epoch, true_state.clone(), p0, models);

        // Process a forward observation first
        let obs1 = Observation::new(epoch + 60.0, true_state.rows(0, 3).into_owned(), 0);
        ukf.process_observation(&obs1).unwrap();

        // Try a backward observation
        let obs2 = Observation::new(epoch + 30.0, true_state.rows(0, 3).into_owned(), 0);
        assert!(ukf.process_observation(&obs2).is_err());
    }

    #[test]
    #[serial]
    fn test_ukf_records_stored_and_accessible() {
        setup_global_test_eop();
        let (epoch, true_state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        let models: Vec<Box<dyn MeasurementModel>> =
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))];

        let mut ukf = create_two_body_ukf(epoch, true_state.clone(), p0, models);
        let observations = generate_position_observations(epoch, &true_state, 5, 60.0);

        for obs in &observations {
            let record = ukf.process_observation(obs).unwrap();
            assert_eq!(record.state_predicted.len(), 6);
            assert_eq!(record.prefit_residual.len(), 3);
            assert_eq!(record.postfit_residual.len(), 3);
            assert_eq!(record.measurement_name, "InertialPosition");
        }

        assert_eq!(ukf.records().len(), 5);
    }

    // =========================================================================
    // Builder tests
    // =========================================================================

    #[test]
    #[serial]
    fn test_ukf_builder_equivalence() {
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        let built = UnscentedKalmanFilter::builder(
            epoch,
            state.clone(),
            p0.clone(),
            ForceModelConfig::two_body_gravity(),
            UKFConfig::default(),
        )
        .measurement_model(Box::new(InertialPositionMeasurementModel::new(10.0)))
        .build()
        .unwrap();

        let flat = UnscentedKalmanFilter::new(
            epoch,
            state,
            p0,
            NumericalPropagationConfig::default(),
            ForceModelConfig::two_body_gravity(),
            None,
            None,
            None,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            UKFConfig::default(),
        )
        .unwrap();

        assert_eq!(built.current_state(), flat.current_state());
        assert_eq!(built.current_covariance(), flat.current_covariance());
    }

    #[test]
    #[serial]
    fn test_ukf_builder_measurement_model_accumulates() {
        // Calling measurement_model() twice should register two models; an
        // observation targeting model_index 1 should be accepted rather than
        // rejected as out-of-bounds.
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        let mut ukf = UnscentedKalmanFilter::builder(
            epoch,
            state.clone(),
            p0,
            ForceModelConfig::two_body_gravity(),
            UKFConfig::default(),
        )
        .measurement_model(Box::new(InertialPositionMeasurementModel::new(10.0)))
        .measurement_model(Box::new(InertialPositionMeasurementModel::new(10.0)))
        .build()
        .unwrap();

        let obs = Observation::new(epoch + 60.0, state.rows(0, 3).into_owned(), 1);
        assert!(ukf.process_observation(&obs).is_ok());
    }

    #[test]
    #[serial]
    fn test_ukf_propagate_to_grows_covariance() {
        setup_global_test_eop();
        let (epoch, true_state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        let models: Vec<Box<dyn MeasurementModel>> =
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))];

        let mut ukf = create_two_body_ukf(epoch, true_state, p0, models);

        let p0_trace = ukf.current_covariance().trace();
        let record = ukf.propagate_to(epoch + 600.0).unwrap();
        assert_eq!(ukf.current_epoch(), epoch + 600.0);
        assert_eq!(record.measurement_name, "Propagation");
        assert_eq!(record.prefit_residual.len(), 0);
        assert_eq!(record.postfit_residual.len(), 0);
        assert_eq!(record.kalman_gain.nrows(), 0);
        assert_eq!(record.kalman_gain.ncols(), 0);
        assert_eq!(record.state_updated, record.state_predicted);
        assert_eq!(record.covariance_updated, record.covariance_predicted);
        // Covariance grows without measurements (Keplerian shear stretches
        // the along-track uncertainty even with no explicit process noise).
        assert!(ukf.current_covariance().trace() > p0_trace);
        // Backwards propagation is supported (e.g. for smoothing): the filter
        // epoch moves back to the target.
        ukf.propagate_to(epoch).unwrap();
        assert_eq!(ukf.current_epoch(), epoch);
    }

    #[test]
    #[serial]
    fn test_ukf_propagate_to_respects_store_records() {
        setup_global_test_eop();
        let (epoch, true_state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        // store_records = true: records() grows by one per propagate_to call.
        let models: Vec<Box<dyn MeasurementModel>> =
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))];
        let mut ukf = create_two_body_ukf(epoch, true_state.clone(), p0.clone(), models);
        assert_eq!(ukf.records().len(), 0);
        ukf.propagate_to(epoch + 300.0).unwrap();
        assert_eq!(ukf.records().len(), 1);
        ukf.propagate_to(epoch + 600.0).unwrap();
        assert_eq!(ukf.records().len(), 2);

        // store_records = false: records() stays empty after propagate_to.
        let prop = create_plain_propagator(epoch, true_state, Some(p0));
        let mut ukf_no_records = UnscentedKalmanFilter::from_propagator(
            prop,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            UKFConfig {
                store_records: false,
                ..UKFConfig::default()
            },
        )
        .unwrap();
        ukf_no_records.propagate_to(epoch + 300.0).unwrap();
        assert_eq!(ukf_no_records.records().len(), 0);
    }

    #[test]
    #[serial]
    fn test_ukf_propagate_to_applies_process_noise() {
        // With process_noise (scale_with_dt = true) configured, propagate_to
        // must add Q * dt to the predicted covariance on top of the dynamics-
        // driven growth; without process noise the same propagation should
        // leave a smaller trace.
        setup_global_test_eop();
        let (epoch, true_state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let dt = 600.0;

        let q_diag = vec![1.0, 1.0, 1.0, 1e-4, 1e-4, 1e-4];
        let q_matrix = DMatrix::from_diagonal(&DVector::from_vec(q_diag.clone()));
        let q_trace: f64 = q_diag.iter().sum();

        let models_no_noise: Vec<Box<dyn MeasurementModel>> =
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))];
        let mut ukf_no_noise =
            create_two_body_ukf(epoch, true_state.clone(), p0.clone(), models_no_noise);
        ukf_no_noise.propagate_to(epoch + dt).unwrap();
        let trace_no_noise = ukf_no_noise.current_covariance().trace();

        let mut ukf_with_noise = UnscentedKalmanFilter::new(
            epoch,
            true_state,
            p0,
            NumericalPropagationConfig::default(),
            ForceModelConfig::two_body_gravity(),
            None,
            None,
            None,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            UKFConfig {
                process_noise: Some(ProcessNoiseConfig {
                    q_matrix,
                    scale_with_dt: true,
                    max_noise_dt: None,
                }),
                ..UKFConfig::default()
            },
        )
        .unwrap();
        ukf_with_noise.propagate_to(epoch + dt).unwrap();
        let trace_with_noise = ukf_with_noise.current_covariance().trace();

        let expected_increment = q_trace * dt;
        let actual_increment = trace_with_noise - trace_no_noise;
        assert!(
            actual_increment > 0.0,
            "process noise should increase the covariance trace, got increment {:.3}",
            actual_increment
        );
        assert_abs_diff_eq!(
            actual_increment,
            expected_increment,
            epsilon = expected_increment * 0.05
        );

        // A same-epoch propagate_to must add no process noise: with dt == 0 the
        // scaled Q contributes nothing, so the covariance trace is unchanged.
        let trace_before = ukf_with_noise.current_covariance().trace();
        ukf_with_noise
            .propagate_to(ukf_with_noise.current_epoch())
            .unwrap();
        let trace_after = ukf_with_noise.current_covariance().trace();
        assert_abs_diff_eq!(trace_after, trace_before, epsilon = trace_before * 1e-9);

        // Backwards propagation applies the same |dt|-scaled process noise as
        // forward propagation: over a backward step the noisy filter's trace
        // grows by Q * |dt| relative to the noise-free filter.
        let (epoch_b, state_b) = two_body_leo();
        let p0_b = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let mut back_no_noise = create_two_body_ukf(
            epoch_b,
            state_b.clone(),
            p0_b.clone(),
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
        );
        back_no_noise.propagate_to(epoch_b - dt).unwrap();
        let back_trace_no_noise = back_no_noise.current_covariance().trace();

        let mut back_with_noise = UnscentedKalmanFilter::new(
            epoch_b,
            state_b,
            p0_b,
            NumericalPropagationConfig::default(),
            ForceModelConfig::two_body_gravity(),
            None,
            None,
            None,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            UKFConfig {
                process_noise: Some(ProcessNoiseConfig {
                    q_matrix: DMatrix::from_diagonal(&DVector::from_vec(q_diag.clone())),
                    scale_with_dt: true,
                    max_noise_dt: None,
                }),
                ..UKFConfig::default()
            },
        )
        .unwrap();
        back_with_noise.propagate_to(epoch_b - dt).unwrap();
        assert_eq!(back_with_noise.current_epoch(), epoch_b - dt);
        let back_increment = back_with_noise.current_covariance().trace() - back_trace_no_noise;
        assert_abs_diff_eq!(back_increment, q_trace * dt, epsilon = q_trace * dt * 0.05);
    }

    #[test]
    #[serial]
    fn test_ukf_wrap_aware_measurement_near_north() {
        // With an AzElRangeMeasurementModel and a near-due-north pass, the
        // sigma-point azimuths straddle the 0/360 wrap. The reference-point
        // measurement mean and wrapped S / P_xz deviations must keep the
        // update sane; the pre-fix raw-average mean lands mid-circle (~180°)
        // and blows the innovation and state correction up by orders of
        // magnitude.
        use crate::constants::AngleFormat;
        use crate::coordinates::position_geodetic_to_ecef;
        use crate::estimation::AzElRangeMeasurementModel;
        use crate::frames::position_ecef_to_eci;
        use nalgebra::Vector3;

        setup_global_test_eop();
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        // Station at (lon, lat) = (0, 0); target due north at lat = 3°,
        // 500 km altitude — predicted azimuth sits on the 0/360 wrap.
        let model =
            AzElRangeMeasurementModel::new(0.0, 0.0, 0.0, 0.01, 0.01, 10.0, AngleFormat::Degrees)
                .unwrap();
        let target_ecef =
            position_geodetic_to_ecef(Vector3::new(0.0, 3.0, 500e3), AngleFormat::Degrees).unwrap();
        let target_eci = position_ecef_to_eci(epoch, target_ecef);
        let r_mag = target_eci.norm();
        let v = (GM_EARTH / r_mag).sqrt();
        let state = DVector::from_vec(vec![
            target_eci[0],
            target_eci[1],
            target_eci[2],
            0.0,
            v,
            0.0,
        ]);

        // Noise-free observation at the true geometry.
        let z_true = model.predict(&epoch, &state, None).unwrap();
        assert!(
            !(1.0..=359.0).contains(&z_true[0]),
            "test geometry azimuth should sit near 0/360, got {}",
            z_true[0]
        );

        // Inflated position covariance so sigma points straddle the wrap;
        // alpha = 1 keeps the sigma-point weights well-conditioned.
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let mut ukf = UnscentedKalmanFilter::new(
            epoch,
            state.clone(),
            p0,
            NumericalPropagationConfig::default(),
            ForceModelConfig::two_body_gravity(),
            None,
            None,
            None,
            vec![Box::new(model)],
            UKFConfig {
                alpha: 1.0,
                beta: 2.0,
                kappa: 0.0,
                store_records: true,
                process_noise: None,
            },
        )
        .unwrap();

        let obs = Observation::new(epoch, z_true.clone(), 0);
        let record = ukf.process_observation(&obs).unwrap();

        // Innovation azimuth component stays small (a few degrees), not ~180°.
        assert!(
            record.prefit_residual[0].abs() < 5.0,
            "azimuth innovation should be small near the wrap, got {}",
            record.prefit_residual[0]
        );

        // Updated covariance remains positive-definite (Cholesky succeeds).
        assert!(
            record.covariance_updated.clone().cholesky().is_some(),
            "updated covariance must remain positive-definite"
        );

        // State correction is bounded — the broken mean drives it to
        // orders of magnitude beyond this.
        let correction =
            (record.state_updated.rows(0, 3) - record.state_predicted.rows(0, 3)).norm();
        assert!(
            correction < 100_000.0,
            "position correction should be bounded, got {} m",
            correction
        );
    }

    #[test]
    #[serial]
    fn test_ukf_store_records_disabled() {
        setup_global_test_eop();
        let (epoch, true_state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        let models: Vec<Box<dyn MeasurementModel>> =
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))];

        let prop = create_plain_propagator(epoch, true_state.clone(), Some(p0));

        let mut ukf = UnscentedKalmanFilter::from_propagator(
            prop,
            models,
            UKFConfig {
                store_records: false,
                ..UKFConfig::default()
            },
        )
        .unwrap();

        let observations = generate_position_observations(epoch, &true_state, 3, 60.0);
        for obs in &observations {
            ukf.process_observation(obs).unwrap();
        }

        assert_eq!(ukf.records().len(), 0);
    }

    #[test]
    #[serial]
    fn test_ukf_process_observations_auto_sorts() {
        setup_global_test_eop();
        let (epoch, true_state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        let models: Vec<Box<dyn MeasurementModel>> =
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))];

        let mut ukf = create_two_body_ukf(epoch, true_state.clone(), p0, models);

        // Create observations in reverse order
        let mut observations = generate_position_observations(epoch, &true_state, 5, 60.0);
        observations.reverse();

        // Should work despite reverse order
        ukf.process_observations(&observations).unwrap();
        assert_eq!(ukf.records().len(), 5);
    }

    #[test]
    #[serial]
    fn test_ukf_non_positive_definite_covariance_errors() {
        // The constructor validates only dimensions, not positive-definiteness;
        // a negative-diagonal initial covariance must surface at the first
        // sigma-point generation as a Cholesky failure.
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let bad_p0 =
            DMatrix::from_diagonal(&DVector::from_vec(vec![-1.0, 1e6, 1e6, 1e2, 1e2, 1e2]));

        let mut ukf = create_two_body_ukf(
            epoch,
            state.clone(),
            bad_p0,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
        );

        let obs = Observation::new(epoch + 60.0, state.rows(0, 3).into_owned(), 0);
        match ukf.process_observation(&obs) {
            Err(e) => assert!(
                e.to_string().contains("positive-definite"),
                "Error should mention positive-definite: {}",
                e
            ),
            Ok(_) => panic!("Expected Cholesky failure for non-positive-definite covariance"),
        }
    }

    #[test]
    #[serial]
    fn test_ukf_model_index_out_of_bounds_errors() {
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        let mut ukf = create_two_body_ukf(
            epoch,
            state.clone(),
            p0,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
        );

        let obs = Observation::new(epoch + 60.0, state.rows(0, 3).into_owned(), 5);
        match ukf.process_observation(&obs) {
            Err(e) => assert!(
                e.to_string().contains("out of bounds"),
                "Error should mention out of bounds: {}",
                e
            ),
            Ok(_) => panic!("Expected error for out-of-bounds model_index"),
        }
    }

    /// Model with a strongly negative-definite noise covariance R, so the
    /// unscented innovation covariance S loses positive-definiteness and the
    /// Cholesky-based gain solve fails.
    struct NegativeNoiseModel;
    impl MeasurementModel for NegativeNoiseModel {
        fn predict(
            &self,
            _epoch: &Epoch,
            state: &DVector<f64>,
            _params: Option<&DVector<f64>>,
        ) -> Result<DVector<f64>, BraheError> {
            Ok(state.rows(0, 3).into_owned())
        }
        fn noise_covariance(&self) -> DMatrix<f64> {
            DMatrix::identity(3, 3) * -1e9
        }
        fn measurement_dim(&self) -> usize {
            3
        }
        fn name(&self) -> &str {
            "NegativeNoise"
        }
    }

    #[test]
    #[serial]
    fn test_ukf_innovation_covariance_not_positive_definite_errors() {
        // A negative-definite R drives S non-positive-definite; the gain solve
        // must surface a structured error and the filter must roll back.
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        let mut ukf =
            create_two_body_ukf(epoch, state.clone(), p0, vec![Box::new(NegativeNoiseModel)]);

        let obs = Observation::new(epoch + 60.0, state.rows(0, 3).into_owned(), 0);
        match ukf.process_observation(&obs) {
            Err(e) => assert!(
                e.to_string().contains("positive-definite"),
                "Error should mention positive-definite: {}",
                e
            ),
            Ok(_) => panic!("Expected non-positive-definite innovation covariance error"),
        }

        let epoch_drift: f64 = ukf.current_epoch() - epoch;
        assert!(
            epoch_drift.abs() < 1e-9,
            "Filter must be rolled back after the error, drifted {} s",
            epoch_drift
        );
    }

    #[test]
    #[serial]
    fn test_ukf_propagate_to_terminal_event_rolls_back() {
        // A terminal event that stops sigma-point propagation short of the
        // target epoch must make propagate_to error and roll the filter back.
        use crate::events::DTimeEvent;

        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let mut prop = create_plain_propagator(epoch, state.clone(), Some(p0));
        prop.add_event_detector(Box::new(
            DTimeEvent::new(epoch + 30.0, "Terminal").set_terminal(),
        ));

        let mut ukf = UnscentedKalmanFilter::from_propagator(
            prop,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            UKFConfig::default(),
        )
        .unwrap();

        let entry_state = ukf.current_state();
        match ukf.propagate_to(epoch + 60.0) {
            Err(e) => assert!(
                e.to_string().contains("before reaching target epoch"),
                "Error should mention stopping early: {}",
                e
            ),
            Ok(_) => panic!("Expected error when terminal event stops propagation"),
        }

        let epoch_drift: f64 = ukf.current_epoch() - epoch;
        assert!(
            epoch_drift.abs() < 1e-9,
            "Filter epoch must be unchanged after a terminal-event error, drifted {} s",
            epoch_drift
        );
        assert_abs_diff_eq!(ukf.current_state(), entry_state, epsilon = 1e-9);
    }

    #[test]
    #[serial]
    fn test_ukf_observation_applies_process_noise_both_branches() {
        // The sigma-point predict step adds process noise Q for both
        // scale_with_dt = true (Q·dt) and false (Q). The sampled predicted
        // covariance is identical across the three filters, so the
        // predicted-covariance trace difference vs a no-noise twin is exactly
        // the Q contribution.
        setup_global_test_eop();
        let (epoch, true_state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let obs = generate_position_observations(epoch, &true_state, 1, 60.0);
        let dt = 60.0;

        let q_diag = vec![1.0, 1.0, 1.0, 1e-4, 1e-4, 1e-4];
        let q_matrix = DMatrix::from_diagonal(&DVector::from_vec(q_diag.clone()));
        let q_trace: f64 = q_diag.iter().sum();

        let make = |pn: Option<ProcessNoiseConfig>| {
            UnscentedKalmanFilter::new(
                epoch,
                true_state.clone(),
                p0.clone(),
                NumericalPropagationConfig::default(),
                ForceModelConfig::two_body_gravity(),
                None,
                None,
                None,
                vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
                UKFConfig {
                    process_noise: pn,
                    ..UKFConfig::default()
                },
            )
            .unwrap()
        };

        let trace_none = make(None)
            .process_observation(&obs[0])
            .unwrap()
            .covariance_predicted
            .trace();

        let trace_scaled = make(Some(ProcessNoiseConfig {
            q_matrix: q_matrix.clone(),
            scale_with_dt: true,
            max_noise_dt: None,
        }))
        .process_observation(&obs[0])
        .unwrap()
        .covariance_predicted
        .trace();
        assert_abs_diff_eq!(
            trace_scaled - trace_none,
            q_trace * dt,
            epsilon = q_trace * dt * 1e-6
        );

        let trace_fixed = make(Some(ProcessNoiseConfig {
            q_matrix,
            scale_with_dt: false,
            max_noise_dt: None,
        }))
        .process_observation(&obs[0])
        .unwrap()
        .covariance_predicted
        .trace();
        assert_abs_diff_eq!(trace_fixed - trace_none, q_trace, epsilon = q_trace * 1e-6);
    }

    #[test]
    #[serial]
    fn test_ukf_propagate_to_fixed_process_noise_independent_of_dt() {
        // With scale_with_dt = false, propagate_to adds the same Q regardless of
        // the propagation duration.
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        let q_diag = vec![1.0, 1.0, 1.0, 1e-4, 1e-4, 1e-4];
        let q_matrix = DMatrix::from_diagonal(&DVector::from_vec(q_diag.clone()));
        let q_trace: f64 = q_diag.iter().sum();

        let increment_at = |target_dt: f64| -> f64 {
            let mut with_q = UnscentedKalmanFilter::new(
                epoch,
                state.clone(),
                p0.clone(),
                NumericalPropagationConfig::default(),
                ForceModelConfig::two_body_gravity(),
                None,
                None,
                None,
                vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
                UKFConfig {
                    process_noise: Some(ProcessNoiseConfig {
                        q_matrix: q_matrix.clone(),
                        scale_with_dt: false,
                        max_noise_dt: None,
                    }),
                    ..UKFConfig::default()
                },
            )
            .unwrap();
            let mut without_q = create_two_body_ukf(
                epoch,
                state.clone(),
                p0.clone(),
                vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            );
            with_q.propagate_to(epoch + target_dt).unwrap();
            without_q.propagate_to(epoch + target_dt).unwrap();
            with_q.current_covariance().trace() - without_q.current_covariance().trace()
        };

        assert_abs_diff_eq!(increment_at(300.0), q_trace, epsilon = q_trace * 1e-6);
        assert_abs_diff_eq!(increment_at(600.0), q_trace, epsilon = q_trace * 1e-6);
    }

    #[test]
    #[serial]
    fn test_ukf_into_dynamics_returns_propagator() {
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let ukf = create_two_body_ukf(
            epoch,
            state,
            p0,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
        );
        let dynamics = ukf.into_dynamics();
        assert_eq!(dynamics.current_epoch(), epoch);
    }

    // =========================================================================
    // Sub-stepped process noise (max_noise_dt)
    // =========================================================================

    fn substep_ukf(
        epoch: Epoch,
        state: DVector<f64>,
        p0: DMatrix<f64>,
        q: DMatrix<f64>,
        max_noise_dt: Option<f64>,
    ) -> UnscentedKalmanFilter {
        UnscentedKalmanFilter::new(
            epoch,
            state,
            p0,
            NumericalPropagationConfig::default(),
            ForceModelConfig::two_body_gravity(),
            None,
            None,
            None,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            UKFConfig {
                process_noise: Some(ProcessNoiseConfig {
                    q_matrix: q,
                    scale_with_dt: true,
                    max_noise_dt,
                }),
                ..UKFConfig::default()
            },
        )
        .unwrap()
    }

    fn assert_cov_close(a: &DMatrix<f64>, b: &DMatrix<f64>, eps: f64) {
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                assert_abs_diff_eq!(a[(i, j)], b[(i, j)], epsilon = eps);
            }
        }
    }

    #[test]
    #[serial]
    fn test_ukf_substep_single_chunk_equals_none() {
        // max_noise_dt >= interval takes the single-shot path: identical to None.
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let q = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0, 1.0, 1e-4, 1e-4, 1e-4]));
        let dt = 300.0;

        let mut f_none = substep_ukf(epoch, state.clone(), p0.clone(), q.clone(), None);
        f_none.propagate_to(epoch + dt).unwrap();
        let mut f_big = substep_ukf(epoch, state, p0, q, Some(dt * 2.0));
        f_big.propagate_to(epoch + dt).unwrap();

        assert_abs_diff_eq!(
            f_none.current_state(),
            f_big.current_state(),
            epsilon = 1e-9
        );
        assert_cov_close(
            f_none.current_covariance(),
            f_big.current_covariance(),
            1e-6,
        );
    }

    #[test]
    #[serial]
    fn test_ukf_substep_definitional_equivalence() {
        // One sub-stepped propagate_to equals the chained single-shot sequence
        // at cadence h over the same span.
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let q = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0, 1.0, 1e-4, 1e-4, 1e-4]));
        let h = 60.0;

        let mut f_sub = substep_ukf(epoch, state.clone(), p0.clone(), q.clone(), Some(h));
        f_sub.propagate_to(epoch + 300.0).unwrap();

        let mut f_chain = substep_ukf(epoch, state, p0, q, None);
        for steps in 1..=5 {
            f_chain.propagate_to(epoch + (steps as f64) * h).unwrap();
        }

        assert_abs_diff_eq!(
            f_sub.current_state(),
            f_chain.current_state(),
            epsilon = 1e-6
        );
        assert_cov_close(
            f_sub.current_covariance(),
            f_chain.current_covariance(),
            1e-3,
        );
    }

    #[test]
    #[serial]
    fn test_ukf_substep_shear_increases_along_track_variance() {
        // Radial process noise injected along the arc (small h) shears into
        // greater along-track variance than the same noise injected at once.
        use crate::math::linalg::SVector6;
        use crate::relative_motion::rotation_eci_to_rtn;
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e2, 1e2, 1e2, 1e-2, 1e-2, 1e-2]));
        let mut q = DMatrix::zeros(6, 6);
        q[(0, 0)] = 1e2;
        let span = 1500.0;

        let mut f_single = substep_ukf(epoch, state.clone(), p0.clone(), q.clone(), None);
        f_single.propagate_to(epoch + span).unwrap();
        let mut f_sub = substep_ukf(epoch, state, p0, q, Some(60.0));
        f_sub.propagate_to(epoch + span).unwrap();

        let x = f_sub.current_state();
        let x6 = SVector6::from_iterator(x.iter().copied());
        let r = rotation_eci_to_rtn(x6);
        let t_var = |f: &UnscentedKalmanFilter| -> f64 {
            let p = f.current_covariance();
            let pos = p.view((0, 0), (3, 3)).into_owned();
            let pos3 = nalgebra::Matrix3::from_iterator(pos.iter().copied());
            let rtn = r * pos3 * r.transpose();
            rtn[(1, 1)]
        };

        assert!(
            t_var(&f_sub) > t_var(&f_single),
            "sub-stepped along-track variance ({}) should exceed single-shot ({})",
            t_var(&f_sub),
            t_var(&f_single)
        );
    }
}
