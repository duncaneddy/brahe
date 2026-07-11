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

use crate::propagators::force_model_config::ForceModelConfig;
use crate::propagators::{DNumericalOrbitPropagator, NumericalPropagationConfig, TrajectoryMode};
use crate::time::Epoch;
use crate::utils::errors::BraheError;

use super::config::UKFConfig;
use super::dynamics_source::DynamicsSource;
use super::traits::{MeasurementModel, validate_model_outputs};
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
    /// For custom dynamics, force model parameters, control inputs, or a
    /// generic propagator, build the propagator yourself and use
    /// [`UnscentedKalmanFilter::from_propagator`].
    ///
    /// # Arguments
    ///
    /// * `epoch` - Initial epoch
    /// * `state` - Initial state vector (meters, m/s)
    /// * `initial_covariance` - Initial covariance matrix (state_dim x state_dim)
    /// * `propagation_config` - Numerical propagation configuration
    /// * `force_config` - Force model configuration
    /// * `measurement_models` - List of measurement models
    /// * `config` - UKF configuration (alpha, beta, kappa)
    pub fn new(
        epoch: Epoch,
        state: DVector<f64>,
        initial_covariance: DMatrix<f64>,
        propagation_config: NumericalPropagationConfig,
        force_config: ForceModelConfig,
        measurement_models: Vec<Box<dyn MeasurementModel>>,
        config: UKFConfig,
    ) -> Result<Self, BraheError> {
        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            propagation_config,
            force_config,
            None,
            None,
            None,
            None,
        )
        .map_err(|e| BraheError::Error(format!("Failed to create propagator: {}", e)))?;

        Self::from_propagator(prop, initial_covariance, measurement_models, config)
    }

    /// Create an Unscented Kalman Filter from an existing propagator.
    ///
    /// Use this when you have a pre-built propagator — custom dynamics
    /// (`DNumericalPropagator`), force model parameters, control inputs, or
    /// any other propagator configuration not covered by
    /// [`UnscentedKalmanFilter::new`]. Both propagator types convert into
    /// [`DynamicsSource`] automatically.
    ///
    /// The UKF does not require STM propagation. If the propagator has STM
    /// enabled it will still work, but every sigma-point propagation pays the
    /// cost of integrating the variational equations for no benefit.
    ///
    /// Trajectory recording on the propagator is disabled: sigma-point
    /// propagation re-propagates each time span 2n+1 times, which would
    /// otherwise accumulate unbounded, interleaved trajectory data.
    ///
    /// # Arguments
    ///
    /// * `propagator` - Pre-built propagator
    /// * `initial_covariance` - Initial covariance matrix (state_dim x state_dim)
    /// * `measurement_models` - List of measurement models
    /// * `config` - UKF configuration (alpha, beta, kappa)
    ///
    /// # Errors
    ///
    /// Returns error if the covariance dimensions do not match the state
    /// dimension, if no measurement models are provided, or if the sigma-point
    /// parameters are invalid (`alpha <= 0` or `state_dim + kappa <= 0`).
    pub fn from_propagator(
        propagator: impl Into<DynamicsSource>,
        initial_covariance: DMatrix<f64>,
        measurement_models: Vec<Box<dyn MeasurementModel>>,
        config: UKFConfig,
    ) -> Result<Self, BraheError> {
        let mut dynamics = propagator.into();

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
    fn predict_and_update(
        &mut self,
        observation: &Observation,
        current_epoch: Epoch,
        dt: f64,
    ) -> Result<FilterRecord, BraheError> {
        let n = self.state_dim;
        let current_state = self.dynamics.current_state();

        // === PREDICT (time update via sigma point propagation) ===

        // Generate sigma points from current state and covariance
        let sigma_points =
            Self::generate_sigma_points(&current_state, &self.covariance, self.lambda, n)?;

        // Propagate each sigma point through dynamics
        let mut propagated_sigmas = Vec::with_capacity(2 * n + 1);
        for sp in &sigma_points {
            self.dynamics.reinitialize(current_epoch, sp.clone(), None);
            self.dynamics.propagate_to(observation.epoch);

            // Guard against the propagator stopping short of the observation
            // epoch (e.g., a terminal event fired during propagation).
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

        // Compute predicted covariance: P_pred = sum(W_c_i * (chi_i - x_pred)(chi_i - x_pred)^T) + Q
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

        // === UPDATE (measurement incorporation via unscented transform) ===

        let model = &self.measurement_models[observation.model_index];
        let m = model.measurement_dim();

        // Generate new sigma points from predicted state + covariance
        let update_sigmas =
            Self::generate_sigma_points(&state_predicted, &p_predicted, self.lambda, n)?;

        // Transform sigma points through measurement model
        let mut z_sigmas = Vec::with_capacity(2 * n + 1);
        for sp in &update_sigmas {
            let z_i = model.predict(&observation.epoch, sp)?;
            z_sigmas.push(z_i);
        }

        // Measurement noise covariance
        let r = model.noise_covariance();

        // Measurement models are a user-extension boundary; validate output
        // shapes so mistakes surface as errors rather than dimension panics.
        validate_model_outputs(
            model.as_ref(),
            &observation.measurement,
            &z_sigmas[0],
            None,
            &r,
            n,
        )?;

        // Predicted measurement mean: z_pred = sum(W_m_i * Z_i)
        let mut z_predicted = DVector::zeros(m);
        for (i, z_i) in z_sigmas.iter().enumerate() {
            z_predicted += self.weights_mean[i] * z_i;
        }

        // Pre-fit residual (innovation)
        let prefit_residual = &observation.measurement - &z_predicted;

        // Innovation covariance: S = sum(W_c_i * (Z_i - z_pred)(Z_i - z_pred)^T) + R
        let mut s = DMatrix::zeros(m, m);
        for (i, z_i) in z_sigmas.iter().enumerate() {
            let dz = z_i - &z_predicted;
            s += self.weights_cov[i] * &dz * dz.transpose();
        }
        s += &r;

        // Cross-covariance: P_xz = sum(W_c_i * (chi_i - x_pred)(Z_i - z_pred)^T)
        let mut p_xz = DMatrix::zeros(n, m);
        for (i, (sp, z_i)) in update_sigmas.iter().zip(z_sigmas.iter()).enumerate() {
            let dx = sp - &state_predicted;
            let dz = z_i - &z_predicted;
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
        let z_postfit = model.predict(&observation.epoch, &state_updated)?;
        let postfit_residual = &observation.measurement - &z_postfit;

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::physical::GM_EARTH;
    use crate::eop::{EOPExtrapolation, FileEOPProvider, set_global_eop_provider};
    use crate::estimation::{InertialPositionMeasurementModel, InertialStateMeasurementModel};
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
            models,
            UKFConfig::default(),
        )
        .unwrap()
    }

    /// Create a plain propagator (no STM, no covariance) for from_propagator tests.
    fn create_plain_propagator(epoch: Epoch, state: DVector<f64>) -> DNumericalOrbitPropagator {
        DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfig::two_body_gravity(),
            None,
            None,
            None,
            None,
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
    fn test_ukf_covariance_dim_mismatch_errors() {
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let prop = create_plain_propagator(epoch, state);

        let models: Vec<Box<dyn MeasurementModel>> =
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))];

        let result = UnscentedKalmanFilter::from_propagator(
            prop,
            DMatrix::identity(4, 4), // wrong dimensions for 6D state
            models,
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
        let prop = create_plain_propagator(epoch, state);

        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let result = UnscentedKalmanFilter::from_propagator(
            prop,
            p0,
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
            create_plain_propagator(epoch, state.clone()),
            p0.clone(),
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
            create_plain_propagator(epoch, state),
            p0,
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
                create_plain_propagator(epoch, state.clone()),
                p0.clone(),
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

    /// Model that always fails predict() — exercises error paths that occur
    /// after the propagator has been advanced (mid sigma-point update the
    /// propagator otherwise holds an arbitrary sigma point, not the mean).
    struct FailingModel;
    impl MeasurementModel for FailingModel {
        fn predict(
            &self,
            _epoch: &Epoch,
            _state: &DVector<f64>,
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

    #[test]
    #[serial]
    fn test_ukf_store_records_disabled() {
        setup_global_test_eop();
        let (epoch, true_state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        let models: Vec<Box<dyn MeasurementModel>> =
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))];

        let prop = create_plain_propagator(epoch, true_state.clone());

        let mut ukf = UnscentedKalmanFilter::from_propagator(
            prop,
            p0,
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
}
