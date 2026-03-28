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
use crate::propagators::{DNumericalOrbitPropagator, NumericalPropagationConfig};
use crate::time::Epoch;
use crate::utils::errors::BraheError;

use super::config::UKFConfig;
use super::dynamics_source::DynamicsSource;
use super::traits::MeasurementModel;
use super::types::{FilterRecord, Observation};

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
/// // let mut ukf = UnscentedKalmanFilter::new(
/// //     epoch, state, p0,
/// //     NumericalPropagationConfig::default(),
/// //     ForceModelConfig::two_body_gravity(),
/// //     None, None, None,
/// //     models, UKFConfig::default(),
/// // )?;
/// // ukf.process_observations(&observations)?;
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
    /// Builds a numerical orbit propagator internally. Unlike the EKF, STM
    /// propagation is not required since the UKF uses sigma-point propagation.
    ///
    /// # Arguments
    ///
    /// * `epoch` - Initial epoch
    /// * `state` - Initial state vector (meters, m/s)
    /// * `initial_covariance` - Initial covariance matrix
    /// * `propagation_config` - Numerical propagation configuration
    /// * `force_config` - Force model configuration
    /// * `params` - Optional parameter vector for force models
    /// * `additional_dynamics` - Optional additional dynamics function
    /// * `control_input` - Optional control input function
    /// * `measurement_models` - List of measurement models
    /// * `config` - UKF configuration (alpha, beta, kappa, state_dim)
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
            Some(initial_covariance),
        )
        .map_err(|e| BraheError::Error(format!("Failed to create propagator: {}", e)))?;

        Self::from_propagator(
            DynamicsSource::OrbitPropagator(prop),
            measurement_models,
            config,
        )
    }

    /// Create an Unscented Kalman Filter from an existing propagator.
    ///
    /// Use this when you have a pre-built propagator with custom dynamics
    /// (e.g., `NumericalPropagator`) or need full control over propagator
    /// configuration.
    ///
    /// # Arguments
    ///
    /// * `propagator` - Pre-built dynamics source (must have covariance set)
    /// * `measurement_models` - List of measurement models
    /// * `config` - UKF configuration (alpha, beta, kappa, state_dim)
    ///
    /// # Errors
    ///
    /// Returns error if the propagator has no initial covariance, if no
    /// measurement models are provided, or if state_dim doesn't match.
    pub fn from_propagator(
        propagator: DynamicsSource,
        measurement_models: Vec<Box<dyn MeasurementModel>>,
        config: UKFConfig,
    ) -> Result<Self, BraheError> {
        // Validate initial covariance
        let initial_cov = propagator.current_covariance().ok_or_else(|| {
            BraheError::Error(
                "UnscentedKalmanFilter requires an initial covariance on the propagator. \
                 Provide initial_covariance when constructing the propagator."
                    .to_string(),
            )
        })?;

        if measurement_models.is_empty() {
            return Err(BraheError::Error(
                "At least one measurement model is required".to_string(),
            ));
        }

        let n = config.state_dim;
        if propagator.state_dim() != n {
            return Err(BraheError::Error(format!(
                "UKFConfig state_dim ({}) does not match propagator state dimension ({})",
                n,
                propagator.state_dim()
            )));
        }

        // Compute UKF weights
        let alpha = config.alpha;
        let beta = config.beta;
        let kappa = config.kappa;
        let lambda = alpha * alpha * (n as f64 + kappa) - n as f64;
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

        let covariance = initial_cov.clone();

        Ok(Self {
            dynamics: propagator,
            measurement_models,
            config,
            records: Vec::new(),
            weights_mean,
            weights_cov,
            lambda,
            covariance,
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

        let n = self.config.state_dim;
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
            let z_i = model.predict(&observation.epoch, sp, None)?;
            z_sigmas.push(z_i);
        }

        // Predicted measurement mean: z_pred = sum(W_m_i * Z_i)
        let mut z_predicted = DVector::zeros(m);
        for (i, z_i) in z_sigmas.iter().enumerate() {
            z_predicted += self.weights_mean[i] * z_i;
        }

        // Pre-fit residual (innovation)
        let prefit_residual = &observation.measurement - &z_predicted;

        // Innovation covariance: S = sum(W_c_i * (Z_i - z_pred)(Z_i - z_pred)^T) + R
        let r = model.noise_covariance();
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

        // Gain: K = P_xz * S^-1
        let s_inv = s.clone().try_inverse().ok_or_else(|| {
            BraheError::NumericalError(
                "Innovation covariance matrix S is singular. Check measurement \
                 noise covariance R and predicted covariance P."
                    .to_string(),
            )
        })?;
        let k = &p_xz * &s_inv;

        // State update: x_upd = x_pred + K * innovation
        let state_updated = &state_predicted + &k * &prefit_residual;

        // Covariance update: P_upd = P_pred - K * S * K^T
        let p_updated = &p_predicted - &k * &s * k.transpose();

        // Post-fit residual
        let z_postfit = model.predict(&observation.epoch, &state_updated, None)?;
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

        // Update internal state
        self.covariance = p_updated;
        self.dynamics.reinitialize(
            observation.epoch,
            state_updated,
            Some(self.covariance.clone()),
        );

        if self.config.store_records {
            self.records.push(record.clone());
        }
        Ok(record)
    }

    /// Process a batch of observations sequentially.
    ///
    /// Observations are auto-sorted by epoch before processing.
    pub fn process_observations(&mut self, observations: &[Observation]) -> Result<(), BraheError> {
        let mut sorted_obs: Vec<&Observation> = observations.iter().collect();
        sorted_obs.sort_by(|a, b| {
            let dt: f64 = a.epoch - b.epoch;
            dt.partial_cmp(&0.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(std::cmp::Ordering::Equal)
        });

        for obs in sorted_obs {
            self.process_observation(obs)?;
        }

        Ok(())
    }

    /// Get current state estimate.
    pub fn current_state(&self) -> DVector<f64> {
        self.dynamics.current_state()
    }

    /// Get current covariance.
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
            UKFConfig {
                state_dim: 6,
                ..UKFConfig::default()
            },
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
    fn test_ukf_construction_no_covariance_errors() {
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();

        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfig::two_body_gravity(),
            None,
            None,
            None,
            None, // No covariance
        )
        .unwrap();

        let models: Vec<Box<dyn MeasurementModel>> =
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))];

        let result = UnscentedKalmanFilter::from_propagator(
            DynamicsSource::OrbitPropagator(prop),
            models,
            UKFConfig::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn test_ukf_from_propagator_no_models_errors() {
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();

        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfig::two_body_gravity(),
            None,
            None,
            None,
            Some(p0),
        )
        .unwrap();

        let result = UnscentedKalmanFilter::from_propagator(
            DynamicsSource::OrbitPropagator(prop),
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
    fn test_ukf_from_propagator_state_dim_mismatch_errors() {
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();

        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfig::two_body_gravity(),
            None,
            None,
            None,
            Some(p0),
        )
        .unwrap();

        // state_dim=3 doesn't match propagator's 6
        let config = UKFConfig {
            state_dim: 3,
            ..UKFConfig::default()
        };
        let result = UnscentedKalmanFilter::from_propagator(
            DynamicsSource::OrbitPropagator(prop),
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            config,
        );
        match result {
            Err(e) => assert!(
                e.to_string().contains("state_dim"),
                "Error should mention state_dim: {}",
                e
            ),
            Ok(_) => panic!("Expected error for state_dim mismatch"),
        }
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

        let prop = DNumericalOrbitPropagator::new(
            epoch,
            true_state.clone(),
            NumericalPropagationConfig::default(),
            ForceModelConfig::two_body_gravity(),
            None,
            None,
            None,
            Some(p0),
        )
        .unwrap();

        let mut ukf = UnscentedKalmanFilter::from_propagator(
            DynamicsSource::OrbitPropagator(prop),
            models,
            UKFConfig {
                state_dim: 6,
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
