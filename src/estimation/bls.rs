/*!
 * Batch Least Squares (BLS) estimator for orbit determination.
 *
 * The BLS estimator processes all observations simultaneously through an
 * iterative Gauss-Newton algorithm. At each iteration it propagates the
 * current state estimate through all observation epochs, accumulates
 * measurement residuals and their Jacobians mapped back to the reference
 * epoch via the State Transition Matrix (STM), and solves for a state
 * correction that minimizes the weighted sum of squared residuals.
 *
 * # Algorithm (Gauss-Newton iteration)
 *
 * 1. Reinitialize the propagator at (t₀, x_k)
 * 2. Propagate sequentially through each observation epoch t_i, collecting:
 *    - Residual: δy_i = y_i - h(x_k(t_i))
 *    - Mapped Jacobian: H_i = H̃_i * Φ(t_i, t₀) (solve-for partition)
 * 3. Solve for state correction δx via either:
 *    - **Normal Equations**: Λ = P̄₀⁻¹ + Σ Hᵢᵀ Rᵢ⁻¹ Hᵢ, N = Σ Hᵢᵀ Rᵢ⁻¹ δyᵢ
 *    - **Stacked Observation Matrix**: Build full H and δy, solve HᵀWH δx = HᵀW δy
 * 4. Apply correction: x_{k+1} = x_k + δx
 * 5. Check convergence (state correction norm and/or cost function change)
 *
 * Two solver formulations are available via [`BLSSolverMethod`]:
 * - [`NormalEquations`]: Memory-efficient O(n²), standard Tapley/Schutz/Born
 * - [`StackedObservationMatrix`]: Better numerical conditioning, O(m*n) memory
 *
 * Optional **consider parameters** allow partitioning the state into solve-for
 * and consider subsets, with the consider covariance contribution computed via
 * cross-correlation terms.
 *
 * [`BLSSolverMethod`]: crate::estimation::BLSSolverMethod
 * [`NormalEquations`]: crate::estimation::BLSSolverMethod::NormalEquations
 * [`StackedObservationMatrix`]: crate::estimation::BLSSolverMethod::StackedObservationMatrix
 */

use nalgebra::{DMatrix, DVector};

use crate::integrators::traits::{DControlInput, DStateDynamics};
use crate::propagators::force_model_config::ForceModelConfig;
use crate::propagators::{DNumericalOrbitPropagator, NumericalPropagationConfig};
use crate::time::Epoch;
use crate::utils::errors::BraheError;

use super::config::{BLSConfig, BLSSolverMethod};
use super::dynamics_source::DynamicsSource;
use super::traits::MeasurementModel;
use super::types::{BLSIterationRecord, BLSObservationResidual, Observation};

/// Batch Least Squares estimator for orbit determination.
///
/// Processes all observations simultaneously through an iterative
/// Gauss-Newton algorithm. The state correction at each iteration
/// minimizes the weighted sum of squared residuals subject to the
/// a priori constraint.
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
/// // let mut bls = BatchLeastSquares::new(
/// //     epoch, state, p0,
/// //     NumericalPropagationConfig::default(),
/// //     ForceModelConfig::two_body_gravity(),
/// //     None, None, None,
/// //     models, BLSConfig::default(),
/// // )?;
/// // bls.solve(&observations)?;
/// ```
pub struct BatchLeastSquares {
    /// Dynamics source (propagator).
    dynamics: DynamicsSource,

    /// Measurement models (supports multiple types)
    measurement_models: Vec<Box<dyn MeasurementModel>>,

    /// BLS configuration
    config: BLSConfig,

    /// A priori epoch (reference epoch for the batch)
    apriori_epoch: Epoch,

    /// A priori state estimate
    apriori_state: DVector<f64>,

    /// A priori covariance matrix
    apriori_covariance: DMatrix<f64>,

    /// Current state estimate (updated each iteration)
    current_state: DVector<f64>,

    /// Current covariance (formal, solve-for partition)
    current_covariance: DMatrix<f64>,

    /// Current epoch (same as apriori_epoch for BLS)
    current_epoch: Epoch,

    /// Whether the solver has converged
    converged: bool,

    /// Number of iterations completed
    iterations_completed: usize,

    /// Final cost function value J
    final_cost: f64,

    /// Per-iteration diagnostic records
    iteration_records: Vec<BLSIterationRecord>,

    /// Per-observation residuals for each iteration
    observation_residuals: Vec<Vec<BLSObservationResidual>>,

    /// Cached consider parameter information (Λ_ss_inv, Λ_sc) for
    /// computing the consider covariance contribution.
    consider_info: Option<(DMatrix<f64>, DMatrix<f64>)>,
}

impl std::fmt::Debug for BatchLeastSquares {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchLeastSquares")
            .field("apriori_epoch", &self.apriori_epoch)
            .field("state_dim", &self.current_state.len())
            .field("converged", &self.converged)
            .field("iterations_completed", &self.iterations_completed)
            .field("final_cost", &self.final_cost)
            .field("num_models", &self.measurement_models.len())
            .finish()
    }
}

/// Internal result type for solver formulations.
struct SolverResult {
    /// State correction vector δx
    state_correction: DVector<f64>,
    /// Formal covariance (solve-for partition)
    formal_covariance: DMatrix<f64>,
    /// Consider info: (Λ_ss_inv, Λ_sc) if consider params configured
    consider_info: Option<(DMatrix<f64>, DMatrix<f64>)>,
}

impl BatchLeastSquares {
    /// Create a Batch Least Squares estimator with orbit propagator dynamics.
    ///
    /// Builds a numerical orbit propagator internally with STM enabled for
    /// the state transition matrix computation required by BLS.
    ///
    /// # Arguments
    ///
    /// * `epoch` - Initial (reference) epoch
    /// * `state` - Initial state vector (meters, m/s)
    /// * `initial_covariance` - A priori covariance matrix
    /// * `propagation_config` - Numerical propagation configuration
    /// * `force_config` - Force model configuration
    /// * `params` - Optional parameter vector for force models
    /// * `additional_dynamics` - Optional additional dynamics function
    /// * `control_input` - Optional control input function
    /// * `measurement_models` - List of measurement models
    /// * `config` - BLS configuration
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
        config: BLSConfig,
    ) -> Result<Self, BraheError> {
        // Force STM enabled for BLS Jacobian mapping
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

    /// Create a Batch Least Squares estimator from an existing propagator.
    ///
    /// Use this when you have a pre-built propagator with custom dynamics
    /// or need full control over propagator configuration.
    ///
    /// # Arguments
    ///
    /// * `propagator` - Pre-built dynamics source (must have STM enabled and covariance set)
    /// * `measurement_models` - List of measurement models
    /// * `config` - BLS configuration
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - STM propagation is not enabled
    /// - No measurement models are provided
    /// - Covariance dimensions do not match state dimensions
    /// - No convergence threshold is configured
    /// - Consider parameter dimensions are inconsistent
    pub fn from_propagator(
        propagator: DynamicsSource,
        measurement_models: Vec<Box<dyn MeasurementModel>>,
        config: BLSConfig,
    ) -> Result<Self, BraheError> {
        // Validate STM enabled
        if !propagator.has_stm() {
            return Err(BraheError::Error(
                "BatchLeastSquares requires STM propagation to be enabled. \
                 Set propagation_config.variational.enable_stm = true or provide \
                 an initial_covariance to the propagator constructor."
                    .to_string(),
            ));
        }

        // Validate at least one measurement model
        if measurement_models.is_empty() {
            return Err(BraheError::Error(
                "At least one measurement model is required".to_string(),
            ));
        }

        // Validate covariance is set and dimensions match
        let covariance = propagator
            .current_covariance()
            .ok_or_else(|| {
                BraheError::Error(
                    "BatchLeastSquares requires an initial covariance on the propagator. \
                     Provide initial_covariance when constructing the propagator."
                        .to_string(),
                )
            })?
            .clone();

        let state_dim = propagator.state_dim();
        if covariance.nrows() != state_dim || covariance.ncols() != state_dim {
            return Err(BraheError::Error(format!(
                "Covariance dimensions ({}x{}) do not match state dimension ({})",
                covariance.nrows(),
                covariance.ncols(),
                state_dim
            )));
        }

        // Validate at least one convergence threshold is set
        if config.state_correction_threshold.is_none()
            && config.cost_convergence_threshold.is_none()
        {
            return Err(BraheError::Error(
                "At least one convergence threshold must be set: \
                 state_correction_threshold and/or cost_convergence_threshold"
                    .to_string(),
            ));
        }

        // Validate consider parameter dimensions if configured
        if let Some(ref consider) = config.consider_params {
            if consider.n_solve >= state_dim {
                return Err(BraheError::Error(format!(
                    "n_solve ({}) must be less than state_dim ({}) when consider \
                     parameters are configured",
                    consider.n_solve, state_dim
                )));
            }
            let n_consider = state_dim - consider.n_solve;
            if consider.consider_covariance.nrows() != n_consider
                || consider.consider_covariance.ncols() != n_consider
            {
                return Err(BraheError::Error(format!(
                    "Consider covariance dimensions ({}x{}) do not match expected \
                     consider parameter count ({})",
                    consider.consider_covariance.nrows(),
                    consider.consider_covariance.ncols(),
                    n_consider
                )));
            }
        }

        let epoch = propagator.current_epoch();
        let state = propagator.current_state();

        Ok(Self {
            dynamics: propagator,
            measurement_models,
            config,
            apriori_epoch: epoch,
            apriori_state: state.clone(),
            apriori_covariance: covariance.clone(),
            current_state: state,
            current_covariance: covariance,
            current_epoch: epoch,
            converged: false,
            iterations_completed: 0,
            final_cost: f64::INFINITY,
            iteration_records: Vec::new(),
            observation_residuals: Vec::new(),
            consider_info: None,
        })
    }

    // =========================================================================
    // Accessor methods
    // =========================================================================

    /// Get current state estimate at the reference epoch.
    pub fn current_state(&self) -> &DVector<f64> {
        &self.current_state
    }

    /// Get current formal covariance (solve-for partition).
    pub fn current_covariance(&self) -> &DMatrix<f64> {
        &self.current_covariance
    }

    /// Get current epoch (reference epoch for the batch).
    pub fn current_epoch(&self) -> Epoch {
        self.current_epoch
    }

    /// Whether the solver has converged.
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// Number of Gauss-Newton iterations completed.
    pub fn iterations_completed(&self) -> usize {
        self.iterations_completed
    }

    /// Final cost function value J.
    pub fn final_cost(&self) -> f64 {
        self.final_cost
    }

    /// Per-iteration diagnostic records.
    ///
    /// Only populated when `config.store_iteration_records` is true.
    pub fn iteration_records(&self) -> &[BLSIterationRecord] {
        &self.iteration_records
    }

    /// Per-observation residuals for each iteration.
    ///
    /// Only populated when `config.store_observation_residuals` is true.
    /// Outer index is iteration, inner index is observation.
    pub fn observation_residuals(&self) -> &[Vec<BLSObservationResidual>] {
        &self.observation_residuals
    }

    /// Formal covariance (same as `current_covariance()`).
    ///
    /// This is the inverse of the information matrix, representing the
    /// uncertainty from the solve-for parameters only.
    pub fn formal_covariance(&self) -> &DMatrix<f64> {
        &self.current_covariance
    }

    /// Consider covariance contribution P_consider.
    ///
    /// Computes P_consider = Λ_ss⁻¹ Λ_sc P̄_c Λ_cs Λ_ss⁻¹ from cached
    /// consider information. Returns None if consider parameters are not
    /// configured or solve() has not been called.
    pub fn consider_covariance(&self) -> Option<DMatrix<f64>> {
        let (lambda_ss_inv, lambda_sc) = self.consider_info.as_ref()?;
        let consider_cov = &self.config.consider_params.as_ref()?.consider_covariance;

        // P_consider = Λ_ss⁻¹ * Λ_sc * P̄_c * Λ_csᵀ * Λ_ss⁻¹
        // Note: Λ_cs = Λ_scᵀ
        let p_consider =
            lambda_ss_inv * lambda_sc * consider_cov * lambda_sc.transpose() * lambda_ss_inv;
        Some(p_consider)
    }

    /// Total covariance: formal + consider contribution.
    ///
    /// P_total = P_formal + P_consider
    ///
    /// Returns formal covariance if consider parameters are not configured.
    pub fn total_covariance(&self) -> DMatrix<f64> {
        match self.consider_covariance() {
            Some(p_consider) => &self.current_covariance + p_consider,
            None => self.current_covariance.clone(),
        }
    }

    // =========================================================================
    // solve() — main entry point
    // =========================================================================

    /// Solve the batch least squares problem.
    ///
    /// Iteratively processes all observations to find the state that minimizes
    /// the weighted sum of squared residuals subject to the a priori constraint.
    ///
    /// # Arguments
    ///
    /// * `observations` - Slice of observations to process
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - No observations are provided
    /// - Any observation's model_index is out of bounds
    /// - A numerical error occurs (singular matrix, propagation failure)
    pub fn solve(&mut self, observations: &[Observation]) -> Result<(), BraheError> {
        // Validate observations non-empty
        if observations.is_empty() {
            return Err(BraheError::Error(
                "No observations provided to BLS solver".to_string(),
            ));
        }

        // Validate model indices
        for (i, obs) in observations.iter().enumerate() {
            if obs.model_index >= self.measurement_models.len() {
                return Err(BraheError::Error(format!(
                    "Observation {} has model_index {} but only {} models are configured",
                    i,
                    obs.model_index,
                    self.measurement_models.len()
                )));
            }
        }

        // Sort observations by epoch
        let mut sorted_obs: Vec<&Observation> = observations.iter().collect();
        sorted_obs.sort_by(|a, b| {
            let dt: f64 = a.epoch - b.epoch;
            dt.partial_cmp(&0.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(std::cmp::Ordering::Equal)
        });

        // Determine solve-for dimension
        let state_dim = self.current_state.len();
        let n_solve = self
            .config
            .consider_params
            .as_ref()
            .map(|c| c.n_solve)
            .unwrap_or(state_dim);

        // Compute a priori information matrix (solve-for partition)
        let p0_solve = self.apriori_covariance.view((0, 0), (n_solve, n_solve));
        let p0_solve_inv = p0_solve.clone_owned().try_inverse().ok_or_else(|| {
            BraheError::NumericalError(
                "A priori covariance (solve-for partition) is singular".to_string(),
            )
        })?;

        let mut prev_cost = f64::INFINITY;

        for iteration in 0..self.config.max_iterations {
            // Reinitialize propagator at reference epoch with current state
            self.dynamics.reinitialize(
                self.apriori_epoch,
                self.current_state.clone(),
                Some(self.current_covariance.clone()),
            );

            // Collect residuals and mapped Jacobians at each observation epoch
            let mut residuals: Vec<DVector<f64>> = Vec::with_capacity(sorted_obs.len());
            let mut h_mapped_solve: Vec<DMatrix<f64>> = Vec::with_capacity(sorted_obs.len());
            let mut h_mapped_full: Vec<DMatrix<f64>> = Vec::with_capacity(sorted_obs.len());
            let mut r_matrices: Vec<DMatrix<f64>> = Vec::with_capacity(sorted_obs.len());
            let mut obs_epochs: Vec<Epoch> = Vec::with_capacity(sorted_obs.len());
            let mut obs_model_names: Vec<String> = Vec::with_capacity(sorted_obs.len());

            for obs in &sorted_obs {
                // Propagate to observation epoch
                self.dynamics.propagate_to(obs.epoch);

                let state_at_obs = self.dynamics.current_state();
                let model = &self.measurement_models[obs.model_index];

                // Compute predicted measurement
                let z_predicted = model.predict(&obs.epoch, &state_at_obs, None)?;

                // Compute residual: δy = y - h(x_k(t_i))
                let residual = &obs.measurement - &z_predicted;

                // Get local Jacobian H̃ (m x n_state)
                let h_local = model.jacobian(&obs.epoch, &state_at_obs, None)?;

                // Get STM Φ(t_i, t_0) from propagator
                let stm = self
                    .dynamics
                    .stm()
                    .ok_or_else(|| {
                        BraheError::NumericalError(
                            "STM not available from propagator during BLS solve".to_string(),
                        )
                    })?
                    .clone();

                // Mapped Jacobian: H_i = H̃_i * Φ(t_i, t_0)
                let h_full = &h_local * &stm;

                // Extract solve-for columns
                let h_solve = h_full.columns(0, n_solve).into_owned();

                residuals.push(residual);
                h_mapped_solve.push(h_solve);
                if self.config.consider_params.is_some() {
                    h_mapped_full.push(h_full);
                }
                r_matrices.push(model.noise_covariance());
                obs_epochs.push(obs.epoch);
                obs_model_names.push(model.name().to_string());
            }

            // Dispatch to formulation-specific solver
            let solver_result = match self.config.solver_method {
                BLSSolverMethod::NormalEquations => self.solve_normal_equations(
                    &residuals,
                    &h_mapped_solve,
                    &h_mapped_full,
                    &r_matrices,
                    &p0_solve_inv,
                    n_solve,
                )?,
                BLSSolverMethod::StackedObservationMatrix => self.solve_stacked_matrix(
                    &residuals,
                    &h_mapped_solve,
                    &h_mapped_full,
                    &r_matrices,
                    &p0_solve_inv,
                    n_solve,
                )?,
            };

            // Compute cost function J = Σ δy_iᵀ R_i⁻¹ δy_i + (x̄₀ - x_k)ᵀ P̄₀⁻¹ (x̄₀ - x_k)
            let mut cost = 0.0;
            for (i, residual) in residuals.iter().enumerate() {
                let r_inv = r_matrices[i].clone().try_inverse().ok_or_else(|| {
                    BraheError::NumericalError(format!(
                        "Measurement noise covariance R is singular for observation {}",
                        i
                    ))
                })?;
                cost += (residual.transpose() * &r_inv * residual)[(0, 0)];
            }
            let apriori_diff = self.apriori_state.rows(0, n_solve).into_owned()
                - self.current_state.rows(0, n_solve).into_owned();
            cost += (apriori_diff.transpose() * &p0_solve_inv * &apriori_diff)[(0, 0)];

            // Compute pre-fit RMS
            let total_meas: usize = residuals.iter().map(|r| r.len()).sum();
            let prefit_rms = if total_meas > 0 {
                let sum_sq: f64 = residuals.iter().map(|r| r.dot(r)).sum();
                (sum_sq / total_meas as f64).sqrt()
            } else {
                0.0
            };

            // Apply state correction to solve-for parameters
            let dx = &solver_result.state_correction;
            let dx_norm = dx.norm();
            for i in 0..n_solve {
                self.current_state[i] += dx[i];
            }
            self.current_covariance = solver_result.formal_covariance;
            self.consider_info = solver_result.consider_info;

            // Compute post-fit residuals by re-propagating with updated state
            self.dynamics.reinitialize(
                self.apriori_epoch,
                self.current_state.clone(),
                Some(self.current_covariance.clone()),
            );

            let mut postfit_residuals: Vec<DVector<f64>> = Vec::with_capacity(sorted_obs.len());
            for obs in &sorted_obs {
                self.dynamics.propagate_to(obs.epoch);
                let state_at_obs = self.dynamics.current_state();
                let model = &self.measurement_models[obs.model_index];
                let z_predicted = model.predict(&obs.epoch, &state_at_obs, None)?;
                postfit_residuals.push(&obs.measurement - &z_predicted);
            }

            let postfit_rms = if total_meas > 0 {
                let sum_sq: f64 = postfit_residuals.iter().map(|r| r.dot(r)).sum();
                (sum_sq / total_meas as f64).sqrt()
            } else {
                0.0
            };

            // Store iteration record
            if self.config.store_iteration_records {
                self.iteration_records.push(BLSIterationRecord {
                    iteration,
                    epoch: self.apriori_epoch,
                    state: self.current_state.clone(),
                    covariance: self.current_covariance.clone(),
                    state_correction: dx.clone(),
                    state_correction_norm: dx_norm,
                    cost,
                    rms_prefit_residual: prefit_rms,
                    rms_postfit_residual: postfit_rms,
                });
            }

            // Store per-observation residuals
            if self.config.store_observation_residuals {
                let obs_residuals: Vec<BLSObservationResidual> = (0..sorted_obs.len())
                    .map(|i| BLSObservationResidual {
                        epoch: obs_epochs[i],
                        model_name: obs_model_names[i].clone(),
                        prefit_residual: residuals[i].clone(),
                        postfit_residual: postfit_residuals[i].clone(),
                    })
                    .collect();
                self.observation_residuals.push(obs_residuals);
            }

            self.iterations_completed = iteration + 1;
            self.final_cost = cost;

            // Check convergence
            let state_converged = self
                .config
                .state_correction_threshold
                .map(|thresh| dx_norm < thresh)
                .unwrap_or(false);

            let cost_converged = self
                .config
                .cost_convergence_threshold
                .map(|thresh| {
                    if prev_cost.is_infinite() {
                        false
                    } else {
                        let rel_change = (prev_cost - cost).abs() / prev_cost.abs().max(1e-30);
                        rel_change < thresh
                    }
                })
                .unwrap_or(false);

            if state_converged || cost_converged {
                self.converged = true;
                break;
            }

            prev_cost = cost;
        }

        self.current_epoch = self.apriori_epoch;
        Ok(())
    }

    // =========================================================================
    // Normal Equations solver
    // =========================================================================

    /// Solve using the Normal Equations formulation.
    ///
    /// Accumulates:
    /// - Λ = P̄₀⁻¹ + Σ Hᵢᵀ Rᵢ⁻¹ Hᵢ (information matrix)
    /// - N = P̄₀⁻¹(x̄₀ - xₖ) + Σ Hᵢᵀ Rᵢ⁻¹ δyᵢ (normal vector)
    ///
    /// Solves δx = Λ⁻¹N via Cholesky decomposition.
    fn solve_normal_equations(
        &self,
        residuals: &[DVector<f64>],
        h_solve: &[DMatrix<f64>],
        h_full: &[DMatrix<f64>],
        r_matrices: &[DMatrix<f64>],
        p0_solve_inv: &DMatrix<f64>,
        n_solve: usize,
    ) -> Result<SolverResult, BraheError> {
        let state_dim = self.current_state.len();

        // Initialize information matrix and normal vector with a priori
        let mut lambda = p0_solve_inv.clone();
        let apriori_diff = self.apriori_state.rows(0, n_solve).into_owned()
            - self.current_state.rows(0, n_solve).into_owned();
        let mut n_vec = p0_solve_inv * &apriori_diff;

        // Also accumulate cross-term if consider params are configured
        let has_consider = self.config.consider_params.is_some();
        let n_consider = state_dim - n_solve;
        let mut lambda_sc = if has_consider {
            DMatrix::zeros(n_solve, n_consider)
        } else {
            DMatrix::zeros(0, 0)
        };

        // Accumulate over observations
        for i in 0..residuals.len() {
            let r_inv = r_matrices[i].clone().try_inverse().ok_or_else(|| {
                BraheError::NumericalError(format!(
                    "Measurement noise covariance R is singular for observation {}",
                    i
                ))
            })?;

            let h_t = h_solve[i].transpose();
            let h_t_r_inv = &h_t * &r_inv;

            lambda += &h_t_r_inv * &h_solve[i];
            n_vec += &h_t_r_inv * &residuals[i];

            // Cross-term for consider parameters
            if has_consider {
                let h_c = h_full[i].columns(n_solve, n_consider);
                lambda_sc += &h_t_r_inv * h_c;
            }
        }

        // Solve via Cholesky decomposition
        let chol = lambda.clone().cholesky().ok_or_else(|| {
            BraheError::NumericalError(
                "Cholesky decomposition of information matrix Λ failed. \
                 The information matrix may not be positive definite."
                    .to_string(),
            )
        })?;

        let dx = chol.solve(&n_vec);
        let lambda_inv = chol.inverse();

        let consider_info = if has_consider {
            Some((lambda_inv.clone(), lambda_sc))
        } else {
            None
        };

        Ok(SolverResult {
            state_correction: dx,
            formal_covariance: lambda_inv,
            consider_info,
        })
    }

    // =========================================================================
    // Stacked Observation Matrix solver
    // =========================================================================

    /// Solve using the Stacked Observation Matrix formulation.
    ///
    /// Builds the augmented system:
    /// - Stack all Hᵢ and δyᵢ into H (m_total x n_solve) and δy (m_total)
    /// - Augment with a priori: append √(P̄₀⁻¹) rows and √(P̄₀⁻¹)(x̄₀ - xₖ)
    /// - Apply weights √(Rᵢ⁻¹) to observation rows
    /// - Solve via Cholesky on HᵀWH
    fn solve_stacked_matrix(
        &self,
        residuals: &[DVector<f64>],
        h_solve: &[DMatrix<f64>],
        h_full: &[DMatrix<f64>],
        r_matrices: &[DMatrix<f64>],
        p0_solve_inv: &DMatrix<f64>,
        n_solve: usize,
    ) -> Result<SolverResult, BraheError> {
        let state_dim = self.current_state.len();

        // Total measurement dimension
        let m_total: usize = residuals.iter().map(|r| r.len()).sum();

        // Build weighted H and δy matrices
        // Total rows = m_total (observations) + n_solve (a priori)
        let total_rows = m_total + n_solve;
        let mut h_stacked = DMatrix::zeros(total_rows, n_solve);
        let mut dy_stacked = DVector::zeros(total_rows);

        let mut row = 0;
        for i in 0..residuals.len() {
            let m_i = residuals[i].len();

            // Compute √(R⁻¹) via diagonal (valid for diagonal R)
            let r_inv = r_matrices[i].clone().try_inverse().ok_or_else(|| {
                BraheError::NumericalError(format!(
                    "Measurement noise covariance R is singular for observation {}",
                    i
                ))
            })?;

            // For diagonal R, √(R⁻¹) is element-wise sqrt of diagonal
            let sqrt_r_inv =
                DMatrix::from_fn(
                    m_i,
                    m_i,
                    |r, c| {
                        if r == c { r_inv[(r, c)].sqrt() } else { 0.0 }
                    },
                );

            // Weighted H row block: √(R⁻¹) * H_i
            let weighted_h = &sqrt_r_inv * &h_solve[i];
            h_stacked
                .view_mut((row, 0), (m_i, n_solve))
                .copy_from(&weighted_h);

            // Weighted residual: √(R⁻¹) * δy_i
            let weighted_dy = &sqrt_r_inv * &residuals[i];
            dy_stacked.rows_mut(row, m_i).copy_from(&weighted_dy);

            row += m_i;
        }

        // Augment with a priori: √(P̄₀⁻¹)
        let p0_chol = p0_solve_inv.clone().cholesky().ok_or_else(|| {
            BraheError::NumericalError(
                "Cholesky decomposition of a priori information matrix failed".to_string(),
            )
        })?;
        // Use L^T (upper factor) so that (L^T)^T * L^T = L * L^T = P0_inv
        // when accumulated via H^T H in the stacked system
        let sqrt_p0_inv = p0_chol.l().transpose();

        h_stacked
            .view_mut((row, 0), (n_solve, n_solve))
            .copy_from(&sqrt_p0_inv);

        let apriori_diff = self.apriori_state.rows(0, n_solve).into_owned()
            - self.current_state.rows(0, n_solve).into_owned();
        let weighted_apriori = &sqrt_p0_inv * &apriori_diff;
        dy_stacked
            .rows_mut(row, n_solve)
            .copy_from(&weighted_apriori);

        // Solve via HᵀH δx = Hᵀ δy (the weighting is already in H and δy)
        let hth = h_stacked.transpose() * &h_stacked;
        let htdy = h_stacked.transpose() * &dy_stacked;

        let chol = hth.clone().cholesky().ok_or_else(|| {
            BraheError::NumericalError(
                "Cholesky decomposition of HᵀWH failed in stacked matrix formulation".to_string(),
            )
        })?;

        let dx = chol.solve(&htdy);
        let lambda_inv = chol.inverse();

        // Compute cross-term for consider parameters if configured
        let has_consider = self.config.consider_params.is_some();
        let n_consider = state_dim - n_solve;
        let consider_info = if has_consider {
            let mut lambda_sc = DMatrix::zeros(n_solve, n_consider);
            for i in 0..residuals.len() {
                let r_inv = r_matrices[i].clone().try_inverse().ok_or_else(|| {
                    BraheError::NumericalError(
                        "Measurement noise covariance R is singular (consider cross-term)"
                            .to_string(),
                    )
                })?;
                let h_s_t = h_solve[i].transpose();
                let h_c = h_full[i].columns(n_solve, n_consider);
                lambda_sc += &h_s_t * &r_inv * h_c;
            }
            Some((lambda_inv.clone(), lambda_sc))
        } else {
            None
        };

        Ok(SolverResult {
            state_correction: dx,
            formal_covariance: lambda_inv,
            consider_info,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::physical::GM_EARTH;
    use crate::eop::{EOPExtrapolation, FileEOPProvider, set_global_eop_provider};
    use crate::estimation::{BLSConfig, BLSSolverMethod, InertialPositionMeasurementModel};
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

    /// Two-body LEO circular orbit. Point-mass gravity only.
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

    fn default_p0() -> DMatrix<f64> {
        DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]))
    }

    /// Create a BLS estimator with two-body point-mass gravity.
    fn create_two_body_bls(
        epoch: Epoch,
        state: DVector<f64>,
        p0: DMatrix<f64>,
        models: Vec<Box<dyn MeasurementModel>>,
        config: BLSConfig,
    ) -> BatchLeastSquares {
        BatchLeastSquares::new(
            epoch,
            state,
            p0,
            NumericalPropagationConfig::default(),
            ForceModelConfig::two_body_gravity(),
            None,
            None,
            None,
            models,
            config,
        )
        .unwrap()
    }

    // =========================================================================
    // Construction tests
    // =========================================================================

    #[test]
    #[serial]
    fn test_bls_construction_valid() {
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();
        let bls = create_two_body_bls(
            epoch,
            state.clone(),
            default_p0(),
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            BLSConfig::default(),
        );
        assert_eq!(bls.current_state().len(), 6);
        assert_eq!(bls.current_covariance().nrows(), 6);
        assert!(!bls.converged());
        assert_eq!(bls.iterations_completed(), 0);
    }

    #[test]
    #[serial]
    fn test_bls_construction_no_stm_errors() {
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();

        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(), // no STM
            ForceModelConfig::two_body_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let result = BatchLeastSquares::from_propagator(
            DynamicsSource::OrbitPropagator(prop),
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            BLSConfig::default(),
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("STM"));
    }

    #[test]
    #[serial]
    fn test_bls_construction_no_models_errors() {
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();

        let mut prop_config = NumericalPropagationConfig::default();
        prop_config.variational.enable_stm = true;

        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            prop_config,
            ForceModelConfig::two_body_gravity(),
            None,
            None,
            None,
            Some(default_p0()),
        )
        .unwrap();

        let result = BatchLeastSquares::from_propagator(
            DynamicsSource::OrbitPropagator(prop),
            vec![], // no models
            BLSConfig::default(),
        );
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("measurement model")
        );
    }

    #[test]
    #[serial]
    fn test_bls_construction_no_convergence_threshold_errors() {
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();

        let config = BLSConfig {
            state_correction_threshold: None,
            cost_convergence_threshold: None,
            ..BLSConfig::default()
        };

        let result = BatchLeastSquares::new(
            epoch,
            state,
            default_p0(),
            NumericalPropagationConfig::default(),
            ForceModelConfig::two_body_gravity(),
            None,
            None,
            None,
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            config,
        );
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("convergence threshold")
        );
    }

    #[test]
    #[serial]
    fn test_bls_construction_bad_covariance_dims_errors() {
        setup_global_test_eop();
        let (epoch, state) = two_body_leo();

        // The propagator validates covariance dimensions internally, so this
        // will fail at the propagator level. We verify the error propagates up.
        let p_wrong = DMatrix::identity(4, 4);
        let result = std::panic::catch_unwind(|| {
            BatchLeastSquares::new(
                epoch,
                state,
                p_wrong,
                NumericalPropagationConfig::default(),
                ForceModelConfig::two_body_gravity(),
                None,
                None,
                None,
                vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
                BLSConfig::default(),
            )
        });
        // The propagator panics on dimension mismatch, so we expect a panic or error
        assert!(
            result.is_err() || result.unwrap().is_err(),
            "Should fail with mismatched covariance dimensions"
        );
    }

    // =========================================================================
    // Solve tests — Normal Equations
    // =========================================================================

    #[test]
    #[serial]
    fn test_bls_normal_equations_converges() {
        setup_global_test_eop();

        let (epoch, true_state) = two_body_leo();
        let mut perturbed = true_state.clone();
        perturbed[0] += 1000.0; // 1km offset in x
        perturbed[1] += 500.0; // 500m offset in y

        let obs = generate_position_observations(epoch, &true_state, 20, 30.0);

        let config = BLSConfig {
            solver_method: BLSSolverMethod::NormalEquations,
            max_iterations: 10,
            state_correction_threshold: Some(1e-6),
            cost_convergence_threshold: None,
            consider_params: None,
            store_iteration_records: true,
            store_observation_residuals: true,
        };

        let mut bls = create_two_body_bls(
            epoch,
            perturbed.clone(),
            default_p0(),
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            config,
        );

        bls.solve(&obs).unwrap();

        // Verify convergence
        assert!(bls.converged(), "BLS should converge");

        // Verify position error reduced significantly
        let final_state = bls.current_state();
        let pos_error = (final_state.rows(0, 3) - true_state.rows(0, 3)).norm();
        let initial_error = (perturbed.rows(0, 3) - true_state.rows(0, 3)).norm();

        assert!(
            pos_error < initial_error * 0.001,
            "Position error should decrease >1000x: initial={:.1}m, final={:.6}m",
            initial_error,
            pos_error
        );
    }

    #[test]
    #[serial]
    fn test_bls_empty_observations_errors() {
        setup_global_test_eop();

        let (epoch, state) = two_body_leo();
        let mut bls = create_two_body_bls(
            epoch,
            state,
            default_p0(),
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            BLSConfig::default(),
        );

        let result = bls.solve(&[]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No observations"));
    }

    #[test]
    #[serial]
    fn test_bls_cost_decreases_monotonically() {
        setup_global_test_eop();

        let (epoch, true_state) = two_body_leo();
        let mut perturbed = true_state.clone();
        perturbed[0] += 500.0;

        let obs = generate_position_observations(epoch, &true_state, 15, 30.0);

        let config = BLSConfig {
            solver_method: BLSSolverMethod::NormalEquations,
            max_iterations: 10,
            state_correction_threshold: Some(1e-12), // tight threshold to get multiple iterations
            cost_convergence_threshold: None,
            consider_params: None,
            store_iteration_records: true,
            store_observation_residuals: false,
        };

        let mut bls = create_two_body_bls(
            epoch,
            perturbed,
            default_p0(),
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            config,
        );

        bls.solve(&obs).unwrap();

        let records = bls.iteration_records();
        assert!(
            records.len() >= 2,
            "Need at least 2 iterations to check monotonicity, got {}",
            records.len()
        );

        for i in 1..records.len() {
            assert!(
                records[i].cost <= records[i - 1].cost + 1e-6,
                "Cost should not increase: iteration {} cost={:.6}, iteration {} cost={:.6}",
                i - 1,
                records[i - 1].cost,
                i,
                records[i].cost
            );
        }
    }

    #[test]
    #[serial]
    fn test_bls_iteration_records_stored() {
        setup_global_test_eop();

        let (epoch, true_state) = two_body_leo();
        let mut perturbed = true_state.clone();
        perturbed[0] += 200.0;

        let obs = generate_position_observations(epoch, &true_state, 10, 30.0);

        let config = BLSConfig {
            solver_method: BLSSolverMethod::NormalEquations,
            max_iterations: 5,
            state_correction_threshold: Some(1e-12), // tight to force iterations
            cost_convergence_threshold: None,
            consider_params: None,
            store_iteration_records: true,
            store_observation_residuals: false,
        };

        let mut bls = create_two_body_bls(
            epoch,
            perturbed,
            default_p0(),
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            config,
        );

        bls.solve(&obs).unwrap();

        let records = bls.iteration_records();
        assert!(
            !records.is_empty(),
            "Should have at least one iteration record"
        );
        assert_eq!(records.len(), bls.iterations_completed());

        // Verify iteration numbers
        for (i, record) in records.iter().enumerate() {
            assert_eq!(record.iteration, i);
            assert_eq!(record.state.len(), 6);
            assert_eq!(record.covariance.nrows(), 6);
            assert_eq!(record.state_correction.len(), 6);
        }
    }

    #[test]
    #[serial]
    fn test_bls_observation_residuals_stored() {
        setup_global_test_eop();

        let (epoch, true_state) = two_body_leo();
        let mut perturbed = true_state.clone();
        perturbed[0] += 200.0;

        let num_obs = 10;
        let obs = generate_position_observations(epoch, &true_state, num_obs, 30.0);

        let config = BLSConfig {
            solver_method: BLSSolverMethod::NormalEquations,
            max_iterations: 3,
            state_correction_threshold: Some(1e-12),
            cost_convergence_threshold: None,
            consider_params: None,
            store_iteration_records: true,
            store_observation_residuals: true,
        };

        let mut bls = create_two_body_bls(
            epoch,
            perturbed,
            default_p0(),
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            config,
        );

        bls.solve(&obs).unwrap();

        let obs_residuals = bls.observation_residuals();
        assert_eq!(obs_residuals.len(), bls.iterations_completed());

        for iteration_residuals in obs_residuals {
            assert_eq!(iteration_residuals.len(), num_obs);
            for residual in iteration_residuals {
                assert_eq!(residual.prefit_residual.len(), 3);
                assert_eq!(residual.postfit_residual.len(), 3);
                assert_eq!(residual.model_name, "InertialPosition");
            }
        }
    }

    #[test]
    #[serial]
    fn test_bls_max_iterations_respected() {
        setup_global_test_eop();

        let (epoch, true_state) = two_body_leo();
        let mut perturbed = true_state.clone();
        perturbed[0] += 1000.0;

        let obs = generate_position_observations(epoch, &true_state, 10, 30.0);

        let config = BLSConfig {
            solver_method: BLSSolverMethod::NormalEquations,
            max_iterations: 2,
            state_correction_threshold: Some(1e-30), // impossible to reach
            cost_convergence_threshold: None,
            consider_params: None,
            store_iteration_records: true,
            store_observation_residuals: false,
        };

        let mut bls = create_two_body_bls(
            epoch,
            perturbed,
            default_p0(),
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            config,
        );

        bls.solve(&obs).unwrap();

        assert_eq!(
            bls.iterations_completed(),
            2,
            "Should stop at max_iterations=2"
        );
        assert!(
            !bls.converged(),
            "Should not converge with impossible threshold"
        );
    }

    // =========================================================================
    // Solve tests — Stacked Observation Matrix
    // =========================================================================

    #[test]
    #[serial]
    fn test_bls_stacked_matrix_converges() {
        setup_global_test_eop();

        let (epoch, true_state) = two_body_leo();
        let mut perturbed = true_state.clone();
        perturbed[0] += 1000.0;
        perturbed[1] += 500.0;

        let obs = generate_position_observations(epoch, &true_state, 20, 30.0);

        let config = BLSConfig {
            solver_method: BLSSolverMethod::StackedObservationMatrix,
            max_iterations: 10,
            state_correction_threshold: Some(1e-6),
            cost_convergence_threshold: None,
            consider_params: None,
            store_iteration_records: true,
            store_observation_residuals: false,
        };

        let mut bls = create_two_body_bls(
            epoch,
            perturbed.clone(),
            default_p0(),
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            config,
        );

        bls.solve(&obs).unwrap();

        assert!(bls.converged(), "Stacked matrix BLS should converge");

        let final_state = bls.current_state();
        let pos_error = (final_state.rows(0, 3) - true_state.rows(0, 3)).norm();
        let initial_error = (perturbed.rows(0, 3) - true_state.rows(0, 3)).norm();

        assert!(
            pos_error < initial_error * 0.001,
            "Position error should decrease >1000x: initial={:.1}m, final={:.6}m",
            initial_error,
            pos_error
        );
    }

    #[test]
    #[serial]
    fn test_bls_formulation_equivalence() {
        setup_global_test_eop();

        let (epoch, true_state) = two_body_leo();
        let mut perturbed = true_state.clone();
        perturbed[0] += 500.0;
        perturbed[1] += 200.0;

        let obs = generate_position_observations(epoch, &true_state, 15, 30.0);

        // Normal Equations
        let config_ne = BLSConfig {
            solver_method: BLSSolverMethod::NormalEquations,
            max_iterations: 10,
            state_correction_threshold: Some(1e-8),
            cost_convergence_threshold: None,
            consider_params: None,
            store_iteration_records: true,
            store_observation_residuals: false,
        };

        let mut bls_ne = create_two_body_bls(
            epoch,
            perturbed.clone(),
            default_p0(),
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            config_ne,
        );
        bls_ne.solve(&obs).unwrap();

        // Stacked Matrix
        let config_sm = BLSConfig {
            solver_method: BLSSolverMethod::StackedObservationMatrix,
            max_iterations: 10,
            state_correction_threshold: Some(1e-8),
            cost_convergence_threshold: None,
            consider_params: None,
            store_iteration_records: true,
            store_observation_residuals: false,
        };

        let mut bls_sm = create_two_body_bls(
            epoch,
            perturbed,
            default_p0(),
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            config_sm,
        );
        bls_sm.solve(&obs).unwrap();

        // Both should converge
        assert!(bls_ne.converged());
        assert!(bls_sm.converged());

        // States should match within tolerance
        let state_diff = (bls_ne.current_state() - bls_sm.current_state()).norm();
        assert!(
            state_diff < 1e-3,
            "Formulations should produce same state: diff={:.6}m",
            state_diff
        );

        // Costs should match within tolerance
        let cost_diff = (bls_ne.final_cost() - bls_sm.final_cost()).abs();
        assert!(
            cost_diff < 1e-3,
            "Formulations should produce same cost: NE={:.6}, SM={:.6}",
            bls_ne.final_cost(),
            bls_sm.final_cost()
        );
    }

    #[test]
    #[serial]
    fn test_bls_cost_convergence_criterion() {
        setup_global_test_eop();

        let (epoch, true_state) = two_body_leo();
        let mut perturbed = true_state.clone();
        perturbed[0] += 500.0;

        let obs = generate_position_observations(epoch, &true_state, 15, 30.0);

        // Use only cost convergence (no state correction threshold)
        let config = BLSConfig {
            solver_method: BLSSolverMethod::NormalEquations,
            max_iterations: 20,
            state_correction_threshold: None,
            cost_convergence_threshold: Some(1e-6),
            consider_params: None,
            store_iteration_records: true,
            store_observation_residuals: false,
        };

        let mut bls = create_two_body_bls(
            epoch,
            perturbed,
            default_p0(),
            vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
            config,
        );

        bls.solve(&obs).unwrap();

        assert!(
            bls.converged(),
            "Should converge via cost criterion. Iterations: {}",
            bls.iterations_completed()
        );

        // Cost should be very small (noise-free measurements with exact dynamics)
        assert!(
            bls.final_cost() < 1.0,
            "Final cost should be small: {:.6}",
            bls.final_cost()
        );
    }
}
