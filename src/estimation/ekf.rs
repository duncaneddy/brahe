/*!
 * Extended Kalman Filter for sequential state estimation.
 *
 * The EKF linearizes the dynamics and measurement models around the current
 * state estimate. It uses the propagator's built-in STM for the state
 * prediction (time update) and the measurement model's Jacobian for the
 * measurement update. Use [`ExtendedKalmanFilter::new()`] for the common
 * orbit-propagator case, or [`ExtendedKalmanFilter::from_propagator()`]
 * when you have a pre-built propagator with custom dynamics.
 *
 * # Algorithm
 *
 * **Predict** (time update to measurement epoch):
 * 1. Reinitialize propagator with current state and covariance
 * 2. Propagate to observation epoch (propagator computes P = Φ·P₀·Φᵀ internally)
 * 3. Add process noise: P_pred = P_propagated + Q
 *
 * **Update** (measurement incorporation):
 * 1. Compute predicted measurement: z_pred = h(x_pred)
 * 2. Compute innovation: y = z - z_pred
 * 3. Compute innovation covariance: S = H * P_pred * Hᵀ + R
 * 4. Compute Kalman gain: K = P_pred * Hᵀ * S⁻¹
 * 5. Update state: x_upd = x_pred + K * y
 * 6. Update covariance: P_upd = (I - K*H) * P_pred * (I - K*H)ᵀ + K*R*Kᵀ
 *    (Joseph form for numerical stability)
 */

use nalgebra::{DMatrix, DVector};

use crate::integrators::traits::{DControlInput, DStateDynamics};
use crate::propagators::force_model_config::ForceModelConfig;
use crate::propagators::{DNumericalOrbitPropagator, NumericalPropagationConfig};
use crate::time::Epoch;
use crate::utils::errors::BraheError;

use super::config::EKFConfig;
use super::dynamics_source::DynamicsSource;
use super::traits::MeasurementModel;
use super::types::{FilterRecord, Observation};

/// Extended Kalman Filter for sequential state estimation.
///
/// Processes observations one at a time, propagating state and covariance
/// between observation epochs and incorporating measurements via linearized
/// updates.
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
/// // let mut ekf = ExtendedKalmanFilter::new(
/// //     epoch, state, p0,
/// //     NumericalPropagationConfig::default(),
/// //     ForceModelConfig::two_body_gravity(),
/// //     None, None, None,
/// //     models, EKFConfig::default(),
/// // )?;
/// // ekf.process_observations(&observations)?;
/// ```
pub struct ExtendedKalmanFilter {
    /// Dynamics source (propagator). Holds the current epoch, state, and
    /// covariance — the EKF accesses them via pass-through methods.
    dynamics: DynamicsSource,

    /// Measurement models (supports multiple types)
    measurement_models: Vec<Box<dyn MeasurementModel>>,

    /// Filter configuration
    config: EKFConfig,

    /// History of filter records (only populated when `config.store_records` is true)
    records: Vec<FilterRecord>,
}

impl ExtendedKalmanFilter {
    /// Create an Extended Kalman Filter with orbit propagator dynamics.
    ///
    /// Builds a numerical orbit propagator internally with STM enabled for
    /// covariance propagation.
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
    /// * `config` - EKF configuration
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
        config: EKFConfig,
    ) -> Result<Self, BraheError> {
        // Force STM enabled for EKF covariance propagation
        let mut prop_config = propagation_config;
        prop_config.variational.enable_stm = true;

        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            prop_config,
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

    /// Create an Extended Kalman Filter from an existing propagator.
    ///
    /// Use this when you have a pre-built propagator with custom dynamics
    /// (e.g., `NumericalPropagator`) or need full control over propagator
    /// configuration.
    ///
    /// # Arguments
    ///
    /// * `propagator` - Pre-built dynamics source (must have STM and covariance enabled)
    /// * `measurement_models` - List of measurement models
    /// * `config` - EKF configuration
    ///
    /// # Errors
    ///
    /// Returns error if the propagator does not have STM enabled, has no
    /// initial covariance, or if no measurement models are provided.
    pub fn from_propagator(
        propagator: DynamicsSource,
        measurement_models: Vec<Box<dyn MeasurementModel>>,
        config: EKFConfig,
    ) -> Result<Self, BraheError> {
        // Validate propagation mode includes STM
        if !propagator.has_stm() {
            return Err(BraheError::Error(
                "ExtendedKalmanFilter requires STM propagation to be enabled. \
                 Set propagation_config.variational.enable_stm = true or provide \
                 an initial_covariance to the propagator constructor."
                    .to_string(),
            ));
        }

        // Validate initial covariance is set on the propagator
        if propagator.current_covariance().is_none() {
            return Err(BraheError::Error(
                "ExtendedKalmanFilter requires an initial covariance on the propagator. \
                 Provide initial_covariance when constructing the propagator."
                    .to_string(),
            ));
        }

        if measurement_models.is_empty() {
            return Err(BraheError::Error(
                "At least one measurement model is required".to_string(),
            ));
        }

        Ok(Self {
            dynamics: propagator,
            measurement_models,
            config,
            records: Vec::new(),
        })
    }

    /// Process a single observation.
    ///
    /// Performs predict (propagate to observation epoch) then update
    /// (incorporate measurement). The observation epoch must be at or after
    /// the current filter epoch.
    ///
    /// # Arguments
    ///
    /// * `observation` - The observation to process
    ///
    /// # Returns
    ///
    /// Filter record containing pre/post-fit residuals, Kalman gain, etc.
    ///
    /// # Errors
    ///
    /// Returns error if observation epoch is before current filter epoch,
    /// if model_index is out of bounds, or if a numerical error occurs.
    pub fn process_observation(
        &mut self,
        observation: &Observation,
    ) -> Result<FilterRecord, BraheError> {
        // Validate time ordering (forward-only)
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

        let model = &self.measurement_models[observation.model_index];

        // === PREDICT (time update) ===
        // The propagator already holds the current state and covariance from
        // construction or the previous update. Reinitialize to reset the STM
        // to identity, then propagate to the observation epoch.
        let current_state = self.dynamics.current_state();
        let current_cov = self
            .dynamics
            .current_covariance()
            .expect("Covariance lost from propagator")
            .clone();
        self.dynamics
            .reinitialize(current_epoch, current_state, Some(current_cov));

        // Propagate state and covariance to observation epoch
        self.dynamics.propagate_to(observation.epoch);
        let state_predicted = self.dynamics.current_state();

        // Read propagated covariance from the propagator
        let mut p_predicted = self
            .dynamics
            .current_covariance()
            .expect("Covariance lost during propagation")
            .clone();

        // Add process noise Q
        if let Some(ref pn) = self.config.process_noise {
            if pn.scale_with_dt {
                let dt_abs = dt.abs().max(1e-12); // avoid zero
                p_predicted += &pn.q_matrix * dt_abs;
            } else {
                p_predicted += &pn.q_matrix;
            }
        }

        // === UPDATE (measurement incorporation) ===
        // Predicted measurement
        let z_predicted = model.predict(&observation.epoch, &state_predicted, None)?;

        // Pre-fit residual (innovation)
        let prefit_residual = &observation.measurement - &z_predicted;

        // Measurement Jacobian
        let h = model.jacobian(&observation.epoch, &state_predicted, None)?;

        // Innovation covariance: S = H * P_pred * Hᵀ + R
        let r = model.noise_covariance();
        let s = &h * &p_predicted * h.transpose() + &r;

        // Kalman gain: K = P_pred * Hᵀ * S⁻¹
        let s_inv = s.clone().try_inverse().ok_or_else(|| {
            BraheError::NumericalError(
                "Innovation covariance matrix S is singular. Check measurement \
                 noise covariance R and predicted covariance P."
                    .to_string(),
            )
        })?;
        let k = &p_predicted * h.transpose() * &s_inv;

        // State update: x_upd = x_pred + K * y
        let state_updated = &state_predicted + &k * &prefit_residual;

        // Covariance update (Joseph form for numerical stability):
        // P_upd = (I - K*H) * P_pred * (I - K*H)ᵀ + K*R*Kᵀ
        let n = state_predicted.len();
        let i_kh = DMatrix::identity(n, n) - &k * &h;
        let p_updated = &i_kh * &p_predicted * i_kh.transpose() + &k * &r * k.transpose();

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

        // Update propagator with post-update state and covariance
        self.dynamics
            .reinitialize(observation.epoch, state_updated, Some(p_updated));

        if self.config.store_records {
            self.records.push(record.clone());
        }
        Ok(record)
    }

    /// Process a batch of observations sequentially.
    ///
    /// Observations are auto-sorted by epoch before processing. This makes
    /// the batch API forgiving while the single-observation API
    /// (`process_observation`) enforces strict chronological ordering.
    ///
    /// Access results after processing via [`current_state()`], [`current_covariance()`],
    /// [`current_epoch()`], and [`records()`].
    pub fn process_observations(&mut self, observations: &[Observation]) -> Result<(), BraheError> {
        // Sort by epoch
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

    /// Get current state estimate (pass-through to propagator).
    pub fn current_state(&self) -> DVector<f64> {
        self.dynamics.current_state()
    }

    /// Get current covariance (pass-through to propagator).
    pub fn current_covariance(&self) -> Option<&DMatrix<f64>> {
        self.dynamics.current_covariance()
    }

    /// Get current epoch (pass-through to propagator).
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
    use crate::estimation::{
        EKFConfig, InertialPositionMeasurementModel, InertialStateMeasurementModel,
        ProcessNoiseConfig,
    };
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

    // =========================================================================
    // Test helpers — two-body point-mass scenario (exact dynamics)
    // =========================================================================

    /// Two-body LEO circular orbit. Point-mass gravity only so filter dynamics
    /// are exact — any state error is purely due to the initial offset.
    fn two_body_leo() -> (Epoch, DVector<f64>) {
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let r = 6878.0e3;
        let v = (GM_EARTH / r).sqrt();
        (epoch, DVector::from_vec(vec![r, 0.0, 0.0, 0.0, v, 0.0]))
    }

    /// Generate noise-free inertial position observations from a truth trajectory.
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

    /// Generate noise-free inertial state (pos+vel) observations.
    fn generate_state_observations(
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
                Observation::new(t, prop.current_state().rows(0, 6).into_owned(), 0)
            })
            .collect()
    }

    /// Create an EKF with two-body point-mass gravity.
    fn create_two_body_ekf(
        epoch: Epoch,
        state: DVector<f64>,
        p0: DMatrix<f64>,
        models: Vec<Box<dyn MeasurementModel>>,
        process_noise: Option<ProcessNoiseConfig>,
    ) -> ExtendedKalmanFilter {
        ExtendedKalmanFilter::new(
            epoch,
            state,
            p0,
            NumericalPropagationConfig::default(),
            ForceModelConfig::two_body_gravity(),
            None,
            None,
            None,
            models,
            EKFConfig {
                process_noise,
                store_records: true,
            },
        )
        .unwrap()
    }

    /// Propagate truth to a target epoch using two-body gravity.
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
    fn test_ekf_construction_valid() {
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let ekf = create_two_body_ekf(
            epoch,
            state,
            p0,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            None,
        );
        assert_eq!(ekf.current_state().len(), 6);
    }

    #[test]
    #[serial]
    fn test_ekf_construction_no_stm_errors() {
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();

        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state.clone(),
            NumericalPropagationConfig::default(), // no STM
            ForceModelConfig::two_body_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let result = ExtendedKalmanFilter::from_propagator(
            DynamicsSource::OrbitPropagator(prop),
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            EKFConfig::default(),
        );
        assert!(result.is_err());
        match result {
            Err(e) => assert!(
                e.to_string().contains("STM"),
                "Error should mention STM: {}",
                e
            ),
            Ok(_) => panic!("Expected error"),
        }
    }

    // =========================================================================
    // Convergence tests — two-body point-mass, noise-free measurements
    //
    // All tests start the filter at a PERTURBED state (offset from truth) and
    // feed noise-free measurements generated from the true trajectory. With
    // exact dynamics (same model for truth and filter), the EKF should converge
    // to the truth.
    // =========================================================================

    #[test]
    #[serial]
    fn test_ekf_converges_from_position_offset() {
        // 1km position offset, position-only measurements every 30s for 10 min.
        // Expected: final position error < 1% of initial error, within 3-sigma.
        setup_global_test_eop();

        let (epoch, true_state) = two_body_leo();
        let mut perturbed = true_state.clone();
        perturbed[0] += 1000.0;
        perturbed[1] += 500.0;

        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let obs = generate_position_observations(epoch, &true_state, 20, 30.0);

        let mut ekf = create_two_body_ekf(
            epoch,
            perturbed.clone(),
            p0,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            None,
        );
        ekf.process_observations(&obs).unwrap();

        let final_state = ekf.current_state();
        let final_cov = ekf.current_covariance().unwrap();
        let final_truth = truth_at(epoch, &true_state, ekf.current_epoch());
        let pos_error = (final_state.rows(0, 3) - final_truth.rows(0, 3)).norm();
        let initial_error = (perturbed.rows(0, 3) - true_state.rows(0, 3)).norm();

        assert!(
            pos_error < initial_error * 0.01,
            "Position error should decrease >100x: initial={:.1}m, final={:.3}m",
            initial_error,
            pos_error
        );

        let pos_sigma = final_cov[(0, 0)].sqrt();
        assert!(
            pos_error < 3.0 * pos_sigma,
            "Position error ({:.3}m) should be within 3-sigma ({:.3}m)",
            pos_error,
            3.0 * pos_sigma
        );
    }

    #[test]
    #[serial]
    fn test_ekf_converges_with_state_measurements() {
        // Full 6D state measurements should converge both position and velocity.
        setup_global_test_eop();

        let (epoch, true_state) = two_body_leo();
        let mut perturbed = true_state.clone();
        perturbed[0] += 500.0;
        perturbed[4] += 1.0;

        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let obs = generate_state_observations(epoch, &true_state, 10, 60.0);

        let mut ekf = create_two_body_ekf(
            epoch,
            perturbed,
            p0,
            vec![Box::new(InertialStateMeasurementModel::new(10.0, 0.1))],
            None,
        );
        ekf.process_observations(&obs).unwrap();

        let final_state = ekf.current_state();
        let final_truth = truth_at(epoch, &true_state, ekf.current_epoch());
        let pos_error = (final_state.rows(0, 3) - final_truth.rows(0, 3)).norm();
        let vel_error = (final_state.rows(3, 3) - final_truth.rows(3, 3)).norm();

        assert!(
            pos_error < 1.0,
            "Position error should be <1m, got {:.3}m",
            pos_error
        );
        assert!(
            vel_error < 0.01,
            "Velocity error should be <0.01m/s, got {:.6}m/s",
            vel_error
        );
    }

    #[test]
    #[serial]
    fn test_ekf_covariance_monotonically_decreases() {
        // No process noise + noise-free measurements: position covariance trace
        // should decrease (or stay flat) at every measurement update.
        setup_global_test_eop();

        let (epoch, true_state) = two_body_leo();
        let mut perturbed = true_state.clone();
        perturbed[0] += 100.0;

        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let obs = generate_position_observations(epoch, &true_state, 15, 30.0);

        let mut ekf = create_two_body_ekf(
            epoch,
            perturbed,
            p0,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            None,
        );
        ekf.process_observations(&obs).unwrap();

        let mut prev_trace = f64::MAX;
        for record in ekf.records() {
            let trace = record.covariance_updated[(0, 0)]
                + record.covariance_updated[(1, 1)]
                + record.covariance_updated[(2, 2)];
            assert!(
                trace <= prev_trace + 1e-6,
                "Position covariance trace should not increase: prev={:.2}, curr={:.2}",
                prev_trace,
                trace
            );
            prev_trace = trace;
        }
    }

    #[test]
    #[serial]
    fn test_ekf_postfit_residuals_converge_to_zero() {
        // Exact dynamics + noise-free measurements: post-fit residuals should
        // converge toward zero as the state estimate converges to truth.
        setup_global_test_eop();

        let (epoch, true_state) = two_body_leo();
        let mut perturbed = true_state.clone();
        perturbed[0] += 200.0;
        perturbed[1] -= 100.0;

        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let obs = generate_position_observations(epoch, &true_state, 20, 30.0);

        let mut ekf = create_two_body_ekf(
            epoch,
            perturbed,
            p0,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            None,
        );
        ekf.process_observations(&obs).unwrap();

        // Last 5 post-fit residuals should be near zero (< 1m)
        for (i, record) in ekf.records()[15..].iter().enumerate() {
            let norm = record.postfit_residual.norm();
            assert!(
                norm < 1.0,
                "Post-fit residual {} should be <1m after convergence, got {:.3}m",
                i + 15,
                norm
            );
        }

        // First pre-fit should be much larger than last post-fit
        let first_prefit = ekf.records()[0].prefit_residual.norm();
        let last_postfit = ekf.records().last().unwrap().postfit_residual.norm();
        assert!(
            first_prefit > last_postfit * 10.0,
            "First pre-fit ({:.1}m) should be >> last post-fit ({:.3}m)",
            first_prefit,
            last_postfit
        );
    }

    #[test]
    #[serial]
    fn test_ekf_state_error_within_covariance_bounds() {
        // After convergence, the state error vs truth should be within 3-sigma
        // of the filter's own covariance estimate for each position component.
        setup_global_test_eop();

        let (epoch, true_state) = two_body_leo();
        let mut perturbed = true_state.clone();
        perturbed[0] += 500.0;
        perturbed[1] -= 300.0;
        perturbed[4] += 0.5;

        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let obs = generate_position_observations(epoch, &true_state, 15, 30.0);

        let mut ekf = create_two_body_ekf(
            epoch,
            perturbed,
            p0,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            None,
        );
        ekf.process_observations(&obs).unwrap();

        // Check last 5 records
        for record in &ekf.records()[10..] {
            let truth = truth_at(epoch, &true_state, record.epoch);
            for i in 0..3 {
                let error = (record.state_updated[i] - truth[i]).abs();
                let sigma = record.covariance_updated[(i, i)].sqrt();
                assert!(
                    error < 3.0 * sigma,
                    "Component {} error ({:.1}m) exceeds 3-sigma ({:.1}m) at epoch {}",
                    i,
                    error,
                    3.0 * sigma,
                    record.epoch
                );
            }
        }
    }

    // =========================================================================
    // API behavior tests
    // =========================================================================

    #[test]
    #[serial]
    fn test_ekf_backward_observation_rejected() {
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        let mut ekf = create_two_body_ekf(
            epoch,
            state,
            p0,
            vec![Box::new(InertialPositionMeasurementModel::new(50.0))],
            None,
        );

        let obs1 = Observation::new(epoch + 60.0, DVector::from_vec(vec![6878e3, 0.0, 0.0]), 0);
        ekf.process_observation(&obs1).unwrap();

        let obs_back = Observation::new(epoch + 30.0, DVector::from_vec(vec![6878e3, 0.0, 0.0]), 0);
        assert!(ekf.process_observation(&obs_back).is_err());
    }

    #[test]
    #[serial]
    fn test_ekf_process_observations_auto_sorts() {
        setup_global_test_eop();
        let (epoch, true_state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let obs = generate_position_observations(epoch, &true_state, 5, 60.0);

        let mut ekf = create_two_body_ekf(
            epoch,
            true_state,
            p0,
            vec![Box::new(InertialPositionMeasurementModel::new(50.0))],
            None,
        );

        let mut reversed = obs.clone();
        reversed.reverse();
        ekf.process_observations(&reversed).unwrap();

        let records = ekf.records();
        for i in 1..records.len() {
            let dt: f64 = records[i].epoch - records[i - 1].epoch;
            assert!(dt >= 0.0);
        }
    }

    #[test]
    #[serial]
    fn test_ekf_records_stored_and_accessible() {
        setup_global_test_eop();
        let (epoch, true_state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let obs = generate_position_observations(epoch, &true_state, 5, 60.0);

        let mut ekf = create_two_body_ekf(
            epoch,
            true_state,
            p0,
            vec![Box::new(InertialPositionMeasurementModel::new(50.0))],
            None,
        );
        ekf.process_observations(&obs).unwrap();

        // Records should be stored (store_records defaults to true in helper)
        assert_eq!(ekf.records().len(), 5);

        // Each record should have the correct measurement name
        for record in ekf.records() {
            assert_eq!(record.measurement_name, "InertialPosition");
            assert_eq!(record.prefit_residual.len(), 3);
            assert_eq!(record.postfit_residual.len(), 3);
        }

        // Current state/covariance/epoch accessible via pass-through
        assert_eq!(ekf.current_state().len(), 6);
        assert!(ekf.current_covariance().is_some());
    }

    #[test]
    #[serial]
    fn test_ekf_store_records_disabled() {
        setup_global_test_eop();
        let (epoch, true_state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let obs = generate_position_observations(epoch, &true_state, 5, 60.0);

        let mut cfg = NumericalPropagationConfig::default();
        cfg.variational.enable_stm = true;

        let prop = DNumericalOrbitPropagator::new(
            epoch,
            true_state,
            cfg,
            ForceModelConfig::two_body_gravity(),
            None,
            None,
            None,
            Some(p0),
        )
        .unwrap();

        let mut ekf = ExtendedKalmanFilter::from_propagator(
            DynamicsSource::OrbitPropagator(prop),
            vec![Box::new(InertialPositionMeasurementModel::new(50.0))],
            EKFConfig {
                process_noise: None,
                store_records: false,
            },
        )
        .unwrap();

        ekf.process_observations(&obs).unwrap();

        // Records should NOT be stored
        assert_eq!(ekf.records().len(), 0);

        // But state/covariance should still be updated
        assert_eq!(ekf.current_state().len(), 6);
        assert!(ekf.current_covariance().is_some());
    }
}
