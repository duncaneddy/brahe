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
use crate::propagators::{DNumericalOrbitPropagator, NumericalPropagationConfig, TrajectoryMode};
use crate::time::Epoch;
use crate::utils::errors::BraheError;

use super::config::{EKFConfig, noise_chunk_boundaries};
use super::dynamics_source::DynamicsSource;
use super::traits::{MeasurementModel, compute_residual, validate_model_outputs};
use super::types::{FilterRecord, Observation, sort_by_epoch};

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
/// let mut ekf = ExtendedKalmanFilter::new(
///     epoch,
///     state,
///     p0,
///     NumericalPropagationConfig::default(),
///     ForceModelConfig::two_body_gravity(),
///     None,
///     None,
///     None,
///     vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
///     EKFConfig::default(),
/// ).unwrap();
///
/// let obs = Observation::new(
///     epoch + 60.0,
///     DVector::from_vec(vec![6877.8e3, 456.7e3, 0.0]),
///     0,
/// );
/// let record = ekf.process_observation(&obs).unwrap();
/// println!("post-fit residual: {}", record.postfit_residual.norm());
/// ```
pub struct ExtendedKalmanFilter {
    /// Dynamics source (propagator). Holds the current epoch and state; the
    /// covariance is owned by the filter and seeded into the propagator for
    /// each predict step.
    dynamics: DynamicsSource,

    /// Current state covariance estimate
    covariance: DMatrix<f64>,

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
    /// covariance propagation. The full range of orbit propagator
    /// configuration is available: force model parameters, additional
    /// dynamics, and control inputs pass through to the propagator. For a
    /// generic propagator (`DNumericalPropagator`) use
    /// [`ExtendedKalmanFilter::from_propagator`].
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
            None,
        )
        .map_err(|e| BraheError::Error(format!("Failed to create propagator: {}", e)))?;

        Self::from_parts(prop.into(), initial_covariance, measurement_models, config)
    }

    /// Create an Extended Kalman Filter from an existing propagator.
    ///
    /// Use this when you have a pre-built propagator — custom dynamics
    /// (`DNumericalPropagator`), force model parameters, control inputs, or
    /// any other propagator configuration not covered by
    /// [`ExtendedKalmanFilter::new`]. Both propagator types convert into
    /// [`DynamicsSource`] automatically.
    ///
    /// The filter's initial covariance is taken from the propagator's
    /// `initial_covariance`; construct the propagator with the covariance you
    /// want the filter to start from.
    ///
    /// Trajectory recording on the propagator is disabled: the filter
    /// re-propagates overlapping time spans, which would otherwise accumulate
    /// unbounded trajectory data. The estimation history is available via
    /// [`records()`](Self::records) instead.
    ///
    /// # Arguments
    ///
    /// * `propagator` - Pre-built propagator (must have an initial covariance;
    ///   providing one enables the STM propagation the EKF requires)
    /// * `measurement_models` - List of measurement models
    /// * `config` - EKF configuration
    ///
    /// # Errors
    ///
    /// Returns error if the propagator was not initialized with an initial
    /// covariance, if STM propagation is not enabled, or if no measurement
    /// models are provided.
    pub fn from_propagator(
        propagator: impl Into<DynamicsSource>,
        measurement_models: Vec<Box<dyn MeasurementModel>>,
        config: EKFConfig,
    ) -> Result<Self, BraheError> {
        let dynamics = propagator.into();

        let initial_covariance = dynamics
            .current_covariance()
            .ok_or_else(|| {
                BraheError::Error(
                    "ExtendedKalmanFilter requires an initial covariance on the propagator. \
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
        config: EKFConfig,
    ) -> Result<Self, BraheError> {
        // Validate propagation mode includes STM
        if !dynamics.has_stm() {
            return Err(BraheError::Error(
                "ExtendedKalmanFilter requires STM propagation to be enabled. \
                 Set propagation_config.variational.enable_stm = true on the propagator."
                    .to_string(),
            ));
        }

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

        dynamics.set_trajectory_mode(TrajectoryMode::Disabled);

        Ok(Self {
            dynamics,
            covariance: initial_covariance,
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

        // Snapshot for rollback: any error after the propagator has been
        // advanced must not leave the filter with a state/covariance pair
        // from two different epochs.
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

    /// Sub-stepped predict: propagate the state and covariance across the
    /// provided chunk `boundaries`, injecting continuous process noise
    /// `Q · δt_chunk` at each chunk boundary rather than all at once.
    ///
    /// Used only when [`ProcessNoiseConfig::max_noise_dt`] sub-stepping is
    /// active (see [`noise_chunk_boundaries`]); the process noise is therefore
    /// always a continuous rate (`scale_with_dt == true`). The propagator must
    /// be positioned at the interval start on entry; on success it is left at
    /// the final boundary (the target). Returns the predicted `(state, P)`.
    ///
    /// This is an approximation using a constant `Q` per chunk; the exact
    /// variational discrete-time `Q_d` integration is tracked in issue #408.
    fn predict_substepped(
        &mut self,
        boundaries: &[Epoch],
        stopped_short: impl Fn(Epoch, Epoch) -> BraheError,
    ) -> Result<(DVector<f64>, DMatrix<f64>), BraheError> {
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
            // Seed the propagator with the running state/covariance and reset
            // the STM to identity for this chunk.
            self.dynamics
                .reinitialize(chunk_start, state.clone(), Some(p.clone()));
            self.dynamics.propagate_to(chunk_target);

            let reached = self.dynamics.current_epoch();
            if (chunk_target - reached).abs() > 1e-6 {
                return Err(stopped_short(reached, target));
            }

            state = self.dynamics.current_state();
            let mut p_prop = self
                .dynamics
                .current_covariance()
                .ok_or_else(|| {
                    BraheError::NumericalError(
                        "Propagator did not provide a propagated covariance during the \
                         EKF predict step"
                            .to_string(),
                    )
                })?
                .clone();
            p_prop += &q * dt_chunk;
            p = 0.5 * (&p_prop + p_prop.transpose());
            chunk_start = chunk_target;
        }

        Ok((state, p))
    }

    /// Fallible predict/update body of [`process_observation`]. Mutates the
    /// propagator; the caller rolls it back if this returns an error.
    fn predict_and_update(
        &mut self,
        observation: &Observation,
        current_epoch: Epoch,
        dt: f64,
    ) -> Result<FilterRecord, BraheError> {
        // Propagator force-model / consider parameters, passed through to the
        // measurement model so consider values can affect the measurement.
        let params = self.dynamics.params().cloned();

        // === PREDICT (time update) ===
        // When continuous process-noise sub-stepping is active the interval is
        // split into chunks (noise injected along the trajectory); otherwise a
        // single end-to-end step reproduces the original behavior exactly.
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
                // Seed the propagator with the filter's current covariance and
                // reset the STM to identity, so it computes
                // P(t) = Φ(t,t_k)·P_k·Φ(t,t_k)ᵀ relative to the current filter
                // epoch during propagation.
                let current_state = self.dynamics.current_state();
                self.dynamics.reinitialize(
                    current_epoch,
                    current_state,
                    Some(self.covariance.clone()),
                );

                // Propagate state and covariance to observation epoch
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

                let state_predicted = self.dynamics.current_state();

                // Read propagated covariance from the propagator
                let mut p_predicted = self
                    .dynamics
                    .current_covariance()
                    .ok_or_else(|| {
                        BraheError::NumericalError(
                            "Propagator did not provide a propagated covariance during the \
                             EKF predict step"
                                .to_string(),
                        )
                    })?
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
                (state_predicted, p_predicted)
            }
        };

        // === UPDATE (measurement incorporation) ===
        let model = &self.measurement_models[observation.model_index];

        // Predicted measurement
        let z_predicted = model.predict(&observation.epoch, &state_predicted, params.as_ref())?;

        // Measurement Jacobian
        let h = model.jacobian(&observation.epoch, &state_predicted, params.as_ref())?;

        // Measurement noise covariance
        let r = model.noise_covariance();

        // Measurement models are a user-extension boundary; validate output
        // shapes so mistakes surface as errors rather than dimension panics.
        validate_model_outputs(
            model.as_ref(),
            &observation.measurement,
            &z_predicted,
            Some(&h),
            &r,
            state_predicted.len(),
        )?;

        // Pre-fit residual (innovation)
        let prefit_residual =
            compute_residual(model.as_ref(), &observation.measurement, &z_predicted)?;

        // Innovation covariance: S = H * P_pred * Hᵀ + R
        let s = &h * &p_predicted * h.transpose() + &r;

        // Kalman gain: K = P_pred * Hᵀ * S⁻¹, computed via Cholesky solve of
        // the symmetric positive-definite S: Kᵀ = S⁻¹ · (H · P_pred)
        let s_chol = s.cholesky().ok_or_else(|| {
            BraheError::NumericalError(
                "Innovation covariance matrix S is not positive-definite. Check \
                 measurement noise covariance R and predicted covariance P."
                    .to_string(),
            )
        })?;
        let k = s_chol.solve(&(&h * &p_predicted)).transpose();

        // State update: x_upd = x_pred + K * y
        let state_updated = &state_predicted + &k * &prefit_residual;

        // Covariance update (Joseph form for numerical stability):
        // P_upd = (I - K*H) * P_pred * (I - K*H)ᵀ + K*R*Kᵀ
        let n = state_predicted.len();
        let i_kh = DMatrix::identity(n, n) - &k * &h;
        let p_updated = &i_kh * &p_predicted * i_kh.transpose() + &k * &r * k.transpose();
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

        // Store the updated covariance in the filter and move the propagator
        // to the post-update state. No covariance is left on the propagator;
        // it is re-seeded on the next predict step.
        self.covariance = p_updated;
        self.dynamics
            .reinitialize(observation.epoch, state_updated, None);

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
        for obs in sort_by_epoch(observations) {
            self.process_observation(obs)?;
        }
        Ok(())
    }

    /// Propagate the filter to an epoch without a measurement update.
    ///
    /// Runs the prediction (time-update) step only: state and covariance are
    /// propagated to `epoch` via the internal dynamics (the numerical
    /// integrator takes its own internal steps and lands exactly on the
    /// target epoch) and process noise is applied. Use this to advance the
    /// filter across measurement gaps — e.g. between tracking passes — while
    /// recording covariance growth, or to step a filter forward in
    /// sequential-planning loops.
    ///
    /// # Arguments
    ///
    /// * `epoch` - Target epoch. May be before or after the current filter
    ///   epoch; backwards propagation is supported (e.g. for smoothing).
    ///
    /// # Returns
    ///
    /// Filter record for the prediction step. Measurement-specific fields
    /// are empty (`measurement_name` is `"Propagation"`, residuals are
    /// zero-length, Kalman gain is 0×0); `state_updated` and
    /// `covariance_updated` equal the predicted values.
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

        // When continuous process-noise sub-stepping is active the interval is
        // split into chunks (noise injected along the trajectory); otherwise a
        // single end-to-end step reproduces the original behavior exactly.
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
                self.dynamics.reinitialize(
                    current_epoch,
                    current_state,
                    Some(self.covariance.clone()),
                );
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

                let state_predicted = self.dynamics.current_state();
                let mut p_predicted = self
                    .dynamics
                    .current_covariance()
                    .ok_or_else(|| {
                        BraheError::NumericalError(
                            "Propagator did not provide a propagated covariance during the \
                             EKF predict step"
                                .to_string(),
                        )
                    })?
                    .clone();

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

    /// Create a builder for [`ExtendedKalmanFilter`].
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
    /// * `config` - EKF configuration
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
    /// let ekf = ExtendedKalmanFilter::builder(
    ///     epoch, state, p0, ForceModelConfig::two_body_gravity(), EKFConfig::default(),
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
        config: EKFConfig,
    ) -> ExtendedKalmanFilterBuilder {
        ExtendedKalmanFilterBuilder {
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

/// Builder for [`ExtendedKalmanFilter`].
///
/// Created by [`ExtendedKalmanFilter::builder`], which takes the required
/// inputs (`epoch`, `state`, `initial_covariance`, `force_config`, `config`).
/// Remaining inputs are provided through chained setters and default to
/// `None` / empty ([`NumericalPropagationConfig::default`] for the
/// propagation configuration, no measurement models).
/// [`ExtendedKalmanFilterBuilder::build`] delegates to [`ExtendedKalmanFilter::new`].
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
/// let ekf = ExtendedKalmanFilter::builder(
///     epoch, state, p0, ForceModelConfig::two_body_gravity(), EKFConfig::default(),
/// )
/// .measurement_model(Box::new(InertialPositionMeasurementModel::new(10.0)))
/// .build()
/// .unwrap();
/// ```
pub struct ExtendedKalmanFilterBuilder {
    epoch: Epoch,
    state: DVector<f64>,
    initial_covariance: DMatrix<f64>,
    force_config: ForceModelConfig,
    config: EKFConfig,
    propagation_config: NumericalPropagationConfig,
    params: Option<DVector<f64>>,
    additional_dynamics: Option<DStateDynamics>,
    control_input: DControlInput,
    measurement_models: Vec<Box<dyn MeasurementModel>>,
}

impl ExtendedKalmanFilterBuilder {
    /// Set the propagation configuration (integrator method, tolerances, and step sizes).
    ///
    /// Defaults to [`NumericalPropagationConfig::default`] if not called. STM
    /// propagation is force-enabled regardless, since the EKF requires it for
    /// covariance propagation.
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
    /// let ekf = ExtendedKalmanFilter::builder(
    ///     epoch, state, p0, ForceModelConfig::two_body_gravity(), EKFConfig::default(),
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
    /// let ekf = ExtendedKalmanFilter::builder(
    ///     epoch, state, p0, ForceModelConfig::two_body_gravity(), EKFConfig::default(),
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
    /// let ekf = ExtendedKalmanFilter::builder(
    ///     epoch, state, p0, ForceModelConfig::two_body_gravity(), EKFConfig::default(),
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
    /// let ekf = ExtendedKalmanFilter::builder(
    ///     epoch, state, p0, ForceModelConfig::two_body_gravity(), EKFConfig::default(),
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
    /// let ekf = ExtendedKalmanFilter::builder(
    ///     epoch, state, p0, ForceModelConfig::two_body_gravity(), EKFConfig::default(),
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
    /// let ekf = ExtendedKalmanFilter::builder(
    ///     epoch, state, p0, ForceModelConfig::two_body_gravity(), EKFConfig::default(),
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
    /// covariance dimensions do not match the state, or if the underlying
    /// propagator construction fails (see [`ExtendedKalmanFilter::new`]).
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
    /// let ekf = ExtendedKalmanFilter::builder(
    ///     epoch, state, p0, ForceModelConfig::two_body_gravity(), EKFConfig::default(),
    /// )
    /// .measurement_model(Box::new(InertialPositionMeasurementModel::new(10.0)))
    /// .build()
    /// .unwrap();
    /// ```
    pub fn build(self) -> Result<ExtendedKalmanFilter, BraheError> {
        ExtendedKalmanFilter::new(
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
        EKFConfig, InertialPositionMeasurementModel, InertialStateMeasurementModel,
        ProcessNoiseConfig,
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

    /// Create a propagator with STM enabled (no covariance) for from_propagator tests.
    fn create_stm_propagator(
        epoch: Epoch,
        state: DVector<f64>,
        covariance: Option<DMatrix<f64>>,
    ) -> DNumericalOrbitPropagator {
        let mut prop_config = NumericalPropagationConfig::default();
        prop_config.variational.enable_stm = true;
        DNumericalOrbitPropagator::new(
            epoch,
            state,
            prop_config,
            ForceModelConfig::two_body_gravity(),
            None,
            None,
            None,
            covariance,
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
    fn test_ekf_from_propagator_no_covariance_errors() {
        // A propagator without an initial covariance cannot seed the filter
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let prop = create_stm_propagator(epoch, state, None);

        let result = ExtendedKalmanFilter::from_propagator(
            prop,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            EKFConfig::default(),
        );
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
    fn test_ekf_covariance_dim_mismatch_errors() {
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();

        let result = ExtendedKalmanFilter::new(
            epoch,
            state,
            DMatrix::identity(4, 4), // wrong dimensions for 6D state
            NumericalPropagationConfig::default(),
            ForceModelConfig::two_body_gravity(),
            None,
            None,
            None,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            EKFConfig::default(),
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
    fn test_ekf_from_propagator_no_models_errors() {
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let prop = create_stm_propagator(epoch, state, Some(p0));

        let result = ExtendedKalmanFilter::from_propagator(
            prop,
            vec![], // no models
            EKFConfig::default(),
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
    fn test_ekf_trajectory_recording_disabled() {
        // Filters re-propagate overlapping spans; trajectory recording must be
        // disabled or the propagator accumulates unbounded interleaved data.
        use crate::trajectories::traits::Trajectory;

        setup_global_test_eop();
        let (epoch, true_state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let obs = generate_position_observations(epoch, &true_state, 5, 60.0);

        let mut ekf = create_two_body_ekf(
            epoch,
            true_state,
            p0,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            None,
        );
        ekf.process_observations(&obs).unwrap();

        match ekf.dynamics() {
            DynamicsSource::OrbitPropagator(p) => {
                assert_eq!(
                    p.trajectory().len(),
                    1,
                    "Trajectory should only contain the construction-time state"
                );
            }
            _ => panic!("Expected orbit propagator"),
        }
    }

    #[test]
    #[serial]
    fn test_ekf_terminal_event_before_observation_errors() {
        use crate::events::DTimeEvent;

        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let mut prop = create_stm_propagator(epoch, state.clone(), Some(p0));
        prop.add_event_detector(Box::new(
            DTimeEvent::new(epoch + 30.0, "Terminal").set_terminal(),
        ));

        let mut ekf = ExtendedKalmanFilter::from_propagator(
            prop,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            EKFConfig::default(),
        )
        .unwrap();

        let obs = Observation::new(epoch + 60.0, state.rows(0, 3).into_owned(), 0);
        let result = ekf.process_observation(&obs);
        match result {
            Err(e) => assert!(
                e.to_string().contains("before reaching observation epoch"),
                "Error should mention stopping early: {}",
                e
            ),
            Ok(_) => panic!("Expected error when terminal event stops propagation"),
        }

        // The failed observation must not have moved the filter
        let epoch_drift: f64 = ekf.current_epoch() - epoch;
        assert!(
            epoch_drift.abs() < 1e-9,
            "Filter epoch must be unchanged after a terminal-event error, drifted {} s",
            epoch_drift
        );
    }

    /// Model that mis-shapes one of its outputs relative to its declared
    /// measurement_dim of 3 — exercises the model-output shape validation.
    struct MisshapenModel {
        bad_predict: bool,
        bad_jacobian: bool,
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
        fn jacobian(
            &self,
            _epoch: &Epoch,
            state: &DVector<f64>,
            _params: Option<&DVector<f64>>,
        ) -> Result<DMatrix<f64>, BraheError> {
            let rows = if self.bad_jacobian { 2 } else { 3 };
            let mut h = DMatrix::zeros(rows, state.len());
            for i in 0..rows {
                h[(i, i)] = 1.0;
            }
            Ok(h)
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
    fn test_ekf_model_output_shape_errors() {
        // Malformed measurement-model outputs must surface as structured
        // errors naming the model, not nalgebra dimension panics — and must
        // leave the filter rolled back at its pre-observation epoch.
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        let cases: Vec<(MisshapenModel, &str)> = vec![
            (
                MisshapenModel {
                    bad_predict: true,
                    bad_jacobian: false,
                    bad_noise: false,
                },
                "predict() returned",
            ),
            (
                MisshapenModel {
                    bad_predict: false,
                    bad_jacobian: true,
                    bad_noise: false,
                },
                "jacobian() returned",
            ),
            (
                MisshapenModel {
                    bad_predict: false,
                    bad_jacobian: false,
                    bad_noise: true,
                },
                "noise_covariance() is",
            ),
        ];

        for (model, expected_msg) in cases {
            let mut ekf = ExtendedKalmanFilter::new(
                epoch,
                state.clone(),
                p0.clone(),
                NumericalPropagationConfig::default(),
                ForceModelConfig::two_body_gravity(),
                None,
                None,
                None,
                vec![Box::new(model)],
                EKFConfig::default(),
            )
            .unwrap();

            let obs = Observation::new(epoch + 60.0, state.rows(0, 3).into_owned(), 0);
            match ekf.process_observation(&obs) {
                Err(e) => assert!(
                    e.to_string().contains(expected_msg),
                    "Error should contain '{}': {}",
                    expected_msg,
                    e
                ),
                Ok(_) => panic!("Expected shape validation error for '{}'", expected_msg),
            }

            let epoch_drift: f64 = ekf.current_epoch() - epoch;
            assert!(
                epoch_drift.abs() < 1e-9,
                "Filter must be rolled back after shape error, drifted {} s",
                epoch_drift
            );
        }
    }

    #[test]
    #[serial]
    fn test_ekf_measurement_dimension_mismatch_errors() {
        // An observation whose measurement length disagrees with the model's
        // measurement_dim must produce a structured error.
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        let mut ekf = create_two_body_ekf(
            epoch,
            state.clone(),
            p0,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            None,
        );

        // 2-element measurement against a 3D position model
        let obs = Observation::new(epoch + 60.0, DVector::from_vec(vec![6878e3, 0.0]), 0);
        match ekf.process_observation(&obs) {
            Err(e) => assert!(
                e.to_string().contains("Observation measurement has"),
                "Error should mention the measurement dimension: {}",
                e
            ),
            Ok(_) => panic!("Expected error for measurement dimension mismatch"),
        }
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
    fn test_ekf_passes_propagator_params_to_measurement_models() {
        // The propagator's force-model / consider parameters must reach the
        // measurement models so consider values can affect measurements.
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let expected = DVector::from_vec(vec![2.2, 1.3]);

        let mut ekf = ExtendedKalmanFilter::new(
            epoch,
            state.clone(),
            p0,
            NumericalPropagationConfig::default(),
            ForceModelConfig::two_body_gravity(),
            Some(expected.clone()),
            None,
            None,
            vec![Box::new(ParamsAssertingModel { expected })],
            EKFConfig::default(),
        )
        .unwrap();

        let obs = Observation::new(epoch + 60.0, state.rows(0, 3).into_owned(), 0);
        let record = ekf
            .process_observation(&obs)
            .expect("model should receive the propagator params");
        assert_eq!(record.measurement_name, "ParamsAsserting");
    }

    /// Model that always fails predict() — exercises error paths that occur
    /// after the propagator has been advanced.
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
    fn test_ekf_failed_observation_leaves_filter_consistent() {
        // A failed observation must not desynchronize the filter: the
        // propagator must be rolled back so state, covariance, and epoch
        // remain the consistent pre-observation triple, and a retry must
        // produce the same result as a filter that never saw the failure.
        setup_global_test_eop();
        let (epoch, true_state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let obs = generate_position_observations(epoch, &true_state, 2, 60.0);

        let mut perturbed = true_state.clone();
        perturbed[0] += 500.0;

        let make_ekf = |models: Vec<Box<dyn MeasurementModel>>| {
            ExtendedKalmanFilter::new(
                epoch,
                perturbed.clone(),
                p0.clone(),
                NumericalPropagationConfig::default(),
                ForceModelConfig::two_body_gravity(),
                None,
                None,
                None,
                models,
                EKFConfig::default(),
            )
            .unwrap()
        };

        let mut ekf = make_ekf(vec![
            Box::new(InertialPositionMeasurementModel::new(10.0)),
            Box::new(FailingModel),
        ]);
        let mut reference = make_ekf(vec![Box::new(InertialPositionMeasurementModel::new(10.0))]);

        ekf.process_observation(&obs[0]).unwrap();
        reference.process_observation(&obs[0]).unwrap();

        // Failing observation at the next epoch must error and leave the
        // filter at the previous epoch
        let bad_obs = Observation::new(obs[1].epoch, obs[1].measurement.clone(), 1);
        assert!(ekf.process_observation(&bad_obs).is_err());
        let epoch_drift: f64 = ekf.current_epoch() - obs[0].epoch;
        assert!(
            epoch_drift.abs() < 1e-9,
            "Filter epoch must be unchanged after a failed observation, drifted {} s",
            epoch_drift
        );

        // Retrying with the working model must match the reference filter
        let rec = ekf.process_observation(&obs[1]).unwrap();
        let rec_ref = reference.process_observation(&obs[1]).unwrap();
        assert_abs_diff_eq!(rec.state_updated, rec_ref.state_updated, epsilon = 1e-9);
        assert_abs_diff_eq!(
            rec.covariance_updated,
            rec_ref.covariance_updated,
            epsilon = 1e-9
        );
    }

    /// Model whose `residual()` override returns a wrong-length vector —
    /// exercises the residual length validation.
    struct BadResidualModel;
    impl MeasurementModel for BadResidualModel {
        fn predict(
            &self,
            _epoch: &Epoch,
            state: &DVector<f64>,
            _params: Option<&DVector<f64>>,
        ) -> Result<DVector<f64>, BraheError> {
            Ok(state.rows(0, 3).into_owned())
        }
        fn noise_covariance(&self) -> DMatrix<f64> {
            DMatrix::identity(3, 3) * 100.0
        }
        fn residual(
            &self,
            _measured: &DVector<f64>,
            _predicted: &DVector<f64>,
        ) -> Result<DVector<f64>, BraheError> {
            // Wrong length (2 instead of the declared measurement_dim 3)
            Ok(DVector::from_vec(vec![0.0, 0.0]))
        }
        fn measurement_dim(&self) -> usize {
            3
        }
        fn name(&self) -> &str {
            "BadResidual"
        }
    }

    #[test]
    #[serial]
    fn test_ekf_residual_wrong_length_errors() {
        // A residual() override returning the wrong length must surface as a
        // structured error, not an nalgebra dimension panic during the update.
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        let mut ekf = create_two_body_ekf(
            epoch,
            state.clone(),
            p0,
            vec![Box::new(BadResidualModel)],
            None,
        );

        let obs = Observation::new(epoch + 60.0, state.rows(0, 3).into_owned(), 0);
        match ekf.process_observation(&obs) {
            Err(e) => assert!(
                e.to_string().contains("residual() returned"),
                "Error should mention the residual length: {}",
                e
            ),
            Ok(_) => panic!("Expected residual length validation error"),
        }

        let epoch_drift: f64 = ekf.current_epoch() - epoch;
        assert!(
            epoch_drift.abs() < 1e-9,
            "Filter must be rolled back after residual error, drifted {} s",
            epoch_drift
        );
    }

    #[test]
    #[serial]
    fn test_ekf_updated_covariance_is_symmetric() {
        setup_global_test_eop();
        let (epoch, true_state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let obs = generate_position_observations(epoch, &true_state, 5, 60.0);

        let mut ekf = create_two_body_ekf(
            epoch,
            true_state,
            p0,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            None,
        );
        ekf.process_observations(&obs).unwrap();

        for record in ekf.records() {
            let p = &record.covariance_updated;
            for i in 0..p.nrows() {
                for j in 0..p.ncols() {
                    assert_eq!(
                        p[(i, j)],
                        p[(j, i)],
                        "Updated covariance must be exactly symmetric"
                    );
                }
            }
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
        let final_cov = ekf.current_covariance();
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
        assert_eq!(ekf.current_covariance().nrows(), 6);
    }

    // =========================================================================
    // Builder tests
    // =========================================================================

    #[test]
    #[serial]
    fn test_ekf_builder_equivalence() {
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        let built = ExtendedKalmanFilter::builder(
            epoch,
            state.clone(),
            p0.clone(),
            ForceModelConfig::two_body_gravity(),
            EKFConfig::default(),
        )
        .measurement_model(Box::new(InertialPositionMeasurementModel::new(10.0)))
        .build()
        .unwrap();

        let flat = ExtendedKalmanFilter::new(
            epoch,
            state,
            p0,
            NumericalPropagationConfig::default(),
            ForceModelConfig::two_body_gravity(),
            None,
            None,
            None,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            EKFConfig::default(),
        )
        .unwrap();

        assert_eq!(built.current_state(), flat.current_state());
        assert_eq!(built.current_covariance(), flat.current_covariance());
    }

    #[test]
    #[serial]
    fn test_ekf_builder_measurement_model_accumulates() {
        // Calling measurement_model() twice should register two models; an
        // observation targeting model_index 1 should be accepted rather than
        // rejected as out-of-bounds.
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        let mut ekf = ExtendedKalmanFilter::builder(
            epoch,
            state.clone(),
            p0,
            ForceModelConfig::two_body_gravity(),
            EKFConfig::default(),
        )
        .measurement_model(Box::new(InertialPositionMeasurementModel::new(10.0)))
        .measurement_model(Box::new(InertialPositionMeasurementModel::new(10.0)))
        .build()
        .unwrap();

        let obs = Observation::new(epoch + 60.0, state.rows(0, 3).into_owned(), 1);
        assert!(ekf.process_observation(&obs).is_ok());
    }

    #[test]
    #[serial]
    fn test_ekf_propagate_to_grows_covariance() {
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        let mut ekf = create_two_body_ekf(
            epoch,
            state,
            p0,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            None,
        );

        let p0_trace = ekf.current_covariance().trace();
        let record = ekf.propagate_to(epoch + 600.0).unwrap();
        assert_eq!(ekf.current_epoch(), epoch + 600.0);
        assert_eq!(record.measurement_name, "Propagation");
        assert_eq!(record.prefit_residual.len(), 0);
        assert_eq!(record.postfit_residual.len(), 0);
        assert_eq!(record.kalman_gain.nrows(), 0);
        assert_eq!(record.kalman_gain.ncols(), 0);
        assert_eq!(record.state_updated, record.state_predicted);
        assert_eq!(record.covariance_updated, record.covariance_predicted);
        // Covariance grows without measurements (Keplerian shear stretches
        // the along-track uncertainty even with no explicit process noise).
        assert!(ekf.current_covariance().trace() > p0_trace);
        // Backwards propagation is supported (e.g. for smoothing): the filter
        // epoch moves back to the target.
        ekf.propagate_to(epoch).unwrap();
        assert_eq!(ekf.current_epoch(), epoch);
    }

    #[test]
    #[serial]
    fn test_ekf_propagate_to_respects_store_records() {
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        // store_records = true: records() grows by one per propagate_to call.
        let mut ekf = create_two_body_ekf(
            epoch,
            state.clone(),
            p0.clone(),
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            None,
        );
        assert_eq!(ekf.records().len(), 0);
        ekf.propagate_to(epoch + 300.0).unwrap();
        assert_eq!(ekf.records().len(), 1);
        ekf.propagate_to(epoch + 600.0).unwrap();
        assert_eq!(ekf.records().len(), 2);

        // store_records = false: records() stays empty after propagate_to.
        let prop = create_stm_propagator(epoch, state, Some(p0));
        let mut ekf_no_records = ExtendedKalmanFilter::from_propagator(
            prop,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            EKFConfig {
                process_noise: None,
                store_records: false,
            },
        )
        .unwrap();
        ekf_no_records.propagate_to(epoch + 300.0).unwrap();
        assert_eq!(ekf_no_records.records().len(), 0);
    }

    #[test]
    #[serial]
    fn test_ekf_propagate_to_applies_process_noise() {
        // With process_noise (scale_with_dt = true) configured, propagate_to
        // must add Q * dt to the predicted covariance on top of the dynamics-
        // driven growth; without process noise the same propagation should
        // leave a smaller trace.
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let dt = 600.0;

        let q_diag = vec![1.0, 1.0, 1.0, 1e-4, 1e-4, 1e-4];
        let q_matrix = DMatrix::from_diagonal(&DVector::from_vec(q_diag.clone()));
        let q_trace: f64 = q_diag.iter().sum();

        let mut ekf_no_noise = create_two_body_ekf(
            epoch,
            state.clone(),
            p0.clone(),
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            None,
        );
        ekf_no_noise.propagate_to(epoch + dt).unwrap();
        let trace_no_noise = ekf_no_noise.current_covariance().trace();

        let mut ekf_with_noise = create_two_body_ekf(
            epoch,
            state,
            p0,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            Some(ProcessNoiseConfig {
                q_matrix,
                scale_with_dt: true,
                max_noise_dt: None,
            }),
        );
        ekf_with_noise.propagate_to(epoch + dt).unwrap();
        let trace_with_noise = ekf_with_noise.current_covariance().trace();

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
        let trace_before = ekf_with_noise.current_covariance().trace();
        ekf_with_noise
            .propagate_to(ekf_with_noise.current_epoch())
            .unwrap();
        let trace_after = ekf_with_noise.current_covariance().trace();
        assert_abs_diff_eq!(trace_after, trace_before, epsilon = trace_before * 1e-9);

        // Backwards propagation applies the same |dt|-scaled process noise as
        // forward propagation: over a backward step the noisy filter's trace
        // grows by Q * |dt| relative to the noise-free filter.
        let (epoch_b, state_b) = two_body_leo();
        let p0_b = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let mut back_no_noise = create_two_body_ekf(
            epoch_b,
            state_b.clone(),
            p0_b.clone(),
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            None,
        );
        back_no_noise.propagate_to(epoch_b - dt).unwrap();
        let back_trace_no_noise = back_no_noise.current_covariance().trace();

        let mut back_with_noise = create_two_body_ekf(
            epoch_b,
            state_b,
            p0_b,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            Some(ProcessNoiseConfig {
                q_matrix: DMatrix::from_diagonal(&DVector::from_vec(q_diag.clone())),
                scale_with_dt: true,
                max_noise_dt: None,
            }),
        );
        back_with_noise.propagate_to(epoch_b - dt).unwrap();
        assert_eq!(back_with_noise.current_epoch(), epoch_b - dt);
        let back_increment = back_with_noise.current_covariance().trace() - back_trace_no_noise;
        assert_abs_diff_eq!(back_increment, q_trace * dt, epsilon = q_trace * dt * 0.05);
    }

    #[test]
    #[serial]
    fn test_ekf_store_records_disabled() {
        setup_global_test_eop();
        let (epoch, true_state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let obs = generate_position_observations(epoch, &true_state, 5, 60.0);

        let prop = create_stm_propagator(epoch, true_state, Some(p0));

        let mut ekf = ExtendedKalmanFilter::from_propagator(
            prop,
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
        assert_eq!(ekf.current_covariance().nrows(), 6);
    }

    #[test]
    #[serial]
    fn test_ekf_model_index_out_of_bounds_errors() {
        // An observation tagged with a model_index beyond the configured models
        // must error before any propagation.
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        let mut ekf = create_two_body_ekf(
            epoch,
            state.clone(),
            p0,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            None,
        );

        let obs = Observation::new(epoch + 60.0, state.rows(0, 3).into_owned(), 5);
        match ekf.process_observation(&obs) {
            Err(e) => assert!(
                e.to_string().contains("out of bounds"),
                "Error should mention out of bounds: {}",
                e
            ),
            Ok(_) => panic!("Expected error for out-of-bounds model_index"),
        }
    }

    /// Model with a strongly negative-definite noise covariance R, so the
    /// innovation covariance S = H·P·Hᵀ + R loses positive-definiteness and the
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
    fn test_ekf_innovation_covariance_not_positive_definite_errors() {
        // A negative-definite R drives S non-positive-definite; the Cholesky
        // solve must surface a structured error and the filter must roll back.
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        let mut ekf = create_two_body_ekf(
            epoch,
            state.clone(),
            p0,
            vec![Box::new(NegativeNoiseModel)],
            None,
        );

        let obs = Observation::new(epoch + 60.0, state.rows(0, 3).into_owned(), 0);
        match ekf.process_observation(&obs) {
            Err(e) => assert!(
                e.to_string().contains("positive-definite"),
                "Error should mention positive-definite: {}",
                e
            ),
            Ok(_) => panic!("Expected non-positive-definite innovation covariance error"),
        }

        // The failed observation must have rolled the filter back.
        let epoch_drift: f64 = ekf.current_epoch() - epoch;
        assert!(
            epoch_drift.abs() < 1e-9,
            "Filter must be rolled back after the error, drifted {} s",
            epoch_drift
        );
    }

    #[test]
    #[serial]
    fn test_ekf_propagate_to_terminal_event_rolls_back() {
        // A terminal event that stops propagation short of the target epoch
        // must make propagate_to error and roll the filter back to its entry
        // epoch and state.
        use crate::events::DTimeEvent;

        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let mut prop = create_stm_propagator(epoch, state.clone(), Some(p0));
        prop.add_event_detector(Box::new(
            DTimeEvent::new(epoch + 30.0, "Terminal").set_terminal(),
        ));

        let mut ekf = ExtendedKalmanFilter::from_propagator(
            prop,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            EKFConfig::default(),
        )
        .unwrap();

        let entry_state = ekf.current_state();
        match ekf.propagate_to(epoch + 60.0) {
            Err(e) => assert!(
                e.to_string().contains("before reaching target epoch"),
                "Error should mention stopping early: {}",
                e
            ),
            Ok(_) => panic!("Expected error when terminal event stops propagation"),
        }

        let epoch_drift: f64 = ekf.current_epoch() - epoch;
        assert!(
            epoch_drift.abs() < 1e-9,
            "Filter epoch must be unchanged after a terminal-event error, drifted {} s",
            epoch_drift
        );
        assert_abs_diff_eq!(ekf.current_state(), entry_state, epsilon = 1e-9);
    }

    #[test]
    #[serial]
    fn test_ekf_observation_applies_process_noise_both_branches() {
        // The measurement predict step adds process noise Q to the predicted
        // covariance for both scale_with_dt = true (Q·dt) and false (Q). The
        // propagated covariance is identical across the three filters, so the
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

        let mut ekf_none = create_two_body_ekf(
            epoch,
            true_state.clone(),
            p0.clone(),
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            None,
        );
        let trace_none = ekf_none
            .process_observation(&obs[0])
            .unwrap()
            .covariance_predicted
            .trace();

        let mut ekf_scaled = create_two_body_ekf(
            epoch,
            true_state.clone(),
            p0.clone(),
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            Some(ProcessNoiseConfig {
                q_matrix: q_matrix.clone(),
                scale_with_dt: true,
                max_noise_dt: None,
            }),
        );
        let trace_scaled = ekf_scaled
            .process_observation(&obs[0])
            .unwrap()
            .covariance_predicted
            .trace();
        assert_abs_diff_eq!(
            trace_scaled - trace_none,
            q_trace * dt,
            epsilon = q_trace * dt * 1e-6
        );

        let mut ekf_fixed = create_two_body_ekf(
            epoch,
            true_state,
            p0,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            Some(ProcessNoiseConfig {
                q_matrix,
                scale_with_dt: false,
                max_noise_dt: None,
            }),
        );
        let trace_fixed = ekf_fixed
            .process_observation(&obs[0])
            .unwrap()
            .covariance_predicted
            .trace();
        assert_abs_diff_eq!(trace_fixed - trace_none, q_trace, epsilon = q_trace * 1e-6);
    }

    #[test]
    #[serial]
    fn test_ekf_propagate_to_fixed_process_noise_independent_of_dt() {
        // With scale_with_dt = false, propagate_to adds the same Q regardless of
        // the propagation duration. The covariance trace increment over a
        // no-noise twin equals trace(Q) for two different dt values.
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

        let q_diag = vec![1.0, 1.0, 1.0, 1e-4, 1e-4, 1e-4];
        let q_matrix = DMatrix::from_diagonal(&DVector::from_vec(q_diag.clone()));
        let q_trace: f64 = q_diag.iter().sum();

        let increment_at = |target_dt: f64| -> f64 {
            let mut with_q = create_two_body_ekf(
                epoch,
                state.clone(),
                p0.clone(),
                vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
                Some(ProcessNoiseConfig {
                    q_matrix: q_matrix.clone(),
                    scale_with_dt: false,
                    max_noise_dt: None,
                }),
            );
            let mut without_q = create_two_body_ekf(
                epoch,
                state.clone(),
                p0.clone(),
                vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
                None,
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
    fn test_ekf_into_dynamics_returns_propagator() {
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
        let dynamics = ekf.into_dynamics();
        assert_eq!(dynamics.current_epoch(), epoch);
    }

    // =========================================================================
    // Sub-stepped process noise (max_noise_dt)
    // =========================================================================

    fn substep_setup() -> (Epoch, DVector<f64>, DMatrix<f64>, DMatrix<f64>) {
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));
        let q = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0, 1.0, 1e-4, 1e-4, 1e-4]));
        (epoch, state, p0, q)
    }

    fn substep_ekf(
        epoch: Epoch,
        state: DVector<f64>,
        p0: DMatrix<f64>,
        q: DMatrix<f64>,
        max_noise_dt: Option<f64>,
    ) -> ExtendedKalmanFilter {
        create_two_body_ekf(
            epoch,
            state,
            p0,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            Some(ProcessNoiseConfig {
                q_matrix: q,
                scale_with_dt: true,
                max_noise_dt,
            }),
        )
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
    fn test_ekf_substep_single_chunk_equals_none_propagate() {
        // max_noise_dt >= interval takes the single-shot path: identical to None.
        setup_global_test_eop();
        let (epoch, state, p0, q) = substep_setup();
        let dt = 300.0;

        let mut f_none = substep_ekf(epoch, state.clone(), p0.clone(), q.clone(), None);
        f_none.propagate_to(epoch + dt).unwrap();

        let mut f_big = substep_ekf(epoch, state, p0, q, Some(dt * 2.0));
        f_big.propagate_to(epoch + dt).unwrap();

        assert_abs_diff_eq!(
            f_none.current_state(),
            f_big.current_state(),
            epsilon = 1e-12
        );
        assert_cov_close(
            f_none.current_covariance(),
            f_big.current_covariance(),
            1e-9,
        );
    }

    #[test]
    #[serial]
    fn test_ekf_substep_single_chunk_equals_none_observation() {
        // Single-chunk equality also holds through process_observation.
        setup_global_test_eop();
        let (epoch, state, p0, q) = substep_setup();
        let obs = generate_position_observations(epoch, &state, 1, 300.0);

        let mut f_none = substep_ekf(epoch, state.clone(), p0.clone(), q.clone(), None);
        f_none.process_observation(&obs[0]).unwrap();

        let mut f_big = substep_ekf(epoch, state, p0, q, Some(600.0));
        f_big.process_observation(&obs[0]).unwrap();

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
    fn test_ekf_substep_definitional_equivalence() {
        // One sub-stepped propagate_to(target) equals the chained sequence of
        // single-shot propagate_to calls at cadence h over the same span.
        setup_global_test_eop();
        let (epoch, state, p0, q) = substep_setup();
        let h = 60.0;
        let span = 300.0; // exactly 5 chunks

        let mut f_sub = substep_ekf(epoch, state.clone(), p0.clone(), q.clone(), Some(h));
        f_sub.propagate_to(epoch + span).unwrap();

        let mut f_chain = substep_ekf(epoch, state, p0, q, None);
        let mut steps = 0;
        while steps < 5 {
            steps += 1;
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
    fn test_ekf_substep_shear_increases_along_track_variance() {
        // Over a long coast arc, radial process noise injected along the
        // trajectory (small h) shears into greater along-track variance than
        // the same noise injected all at once at the end (single shot).
        use crate::math::linalg::SVector6;
        use crate::relative_motion::rotation_eci_to_rtn;
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e2, 1e2, 1e2, 1e-2, 1e-2, 1e-2]));
        // Process noise concentrated on the radial (x) position axis.
        let mut q = DMatrix::zeros(6, 6);
        q[(0, 0)] = 1e2;
        let span = 1500.0;

        let mut f_single = substep_ekf(epoch, state.clone(), p0.clone(), q.clone(), None);
        f_single.propagate_to(epoch + span).unwrap();

        let mut f_sub = substep_ekf(epoch, state, p0, q, Some(60.0));
        f_sub.propagate_to(epoch + span).unwrap();

        let x = f_sub.current_state();
        let x6 = SVector6::from_iterator(x.iter().copied());
        let r = rotation_eci_to_rtn(x6);

        let t_var = |f: &ExtendedKalmanFilter| -> f64 {
            let p = f.current_covariance();
            let pos = p.view((0, 0), (3, 3)).into_owned();
            let pos3 = nalgebra::Matrix3::from_iterator(pos.iter().copied());
            let rtn = r * pos3 * r.transpose();
            rtn[(1, 1)] // along-track (transverse) variance
        };

        let t_single = t_var(&f_single);
        let t_sub = t_var(&f_sub);
        assert!(
            t_sub > t_single,
            "sub-stepped along-track variance ({}) should exceed single-shot ({})",
            t_sub,
            t_single
        );
    }

    #[test]
    #[serial_test::parallel]
    fn test_process_noise_config_validate_rejects_bad_max_noise_dt() {
        let q = DMatrix::identity(6, 6);
        for bad in [0.0, -1.0, f64::NAN] {
            let pn = ProcessNoiseConfig {
                q_matrix: q.clone(),
                scale_with_dt: true,
                max_noise_dt: Some(bad),
            };
            assert!(
                pn.validate().is_err(),
                "max_noise_dt = {} should be rejected",
                bad
            );
        }
        let ok = ProcessNoiseConfig {
            q_matrix: q,
            scale_with_dt: true,
            max_noise_dt: Some(60.0),
        };
        assert!(ok.validate().is_ok());
    }

    #[test]
    #[serial]
    fn test_ekf_construction_rejects_bad_max_noise_dt() {
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let p0 = DMatrix::identity(6, 6);
        let result = ExtendedKalmanFilter::new(
            epoch,
            state,
            p0,
            NumericalPropagationConfig::default(),
            ForceModelConfig::two_body_gravity(),
            None,
            None,
            None,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            EKFConfig {
                process_noise: Some(ProcessNoiseConfig {
                    q_matrix: DMatrix::identity(6, 6),
                    scale_with_dt: true,
                    max_noise_dt: Some(-1.0),
                }),
                store_records: true,
            },
        );
        assert!(result.is_err());
    }
}
