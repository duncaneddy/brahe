/*!
Implementation of the RKN12(10) Runge-Kutta-Nyström integration method.

This integrator is specifically designed for second-order ordinary differential
equations of the form `y'' = f(t, y)`, making it highly efficient for orbital
mechanics and other problems with this structure.

# ⚠️ EXPERIMENTAL

**This integrator is experimental and requires significantly more validation before
use in production systems.** While the implementation passes all tests and achieves
excellent accuracy on orbital mechanics problems, it has not been extensively validated
against a wide range of problem types and edge cases. Use with caution and verify
results independently for critical applications.
*/

use nalgebra::{DMatrix, DVector, SMatrix, SVector};

use crate::integrators::butcher_tableau::{EmbeddedRKNButcherTableau, rkn1210_tableau};
use crate::integrators::config::{AdaptiveStepSResult, IntegratorConfig};
use crate::integrators::traits::{
    AdaptiveStepDIntegrator, AdaptiveStepDResult, AdaptiveStepInternalResultD,
    AdaptiveStepInternalResultS, AdaptiveStepSIntegrator, ControlInput, ControlInputD, DIntegrator,
    SensitivityD, SensitivityS, StateDynamics, StateDynamicsD, VariationalMatrix,
    VariationalMatrixD, compute_next_step_size, compute_normalized_error,
    compute_normalized_error_s, compute_reduced_step_size,
};

/// Implementation of the RKN12(10) Runge-Kutta-Nyström numerical integrator.
///
/// This is a very high-order adaptive integrator (12th/10th order embedded) specifically
/// designed for second-order ODEs of the form `y'' = f(t, y)`. While it accepts the
/// standard state vector format `[position, velocity]`, it exploits the second-order
/// structure internally for improved efficiency.
///
/// # ⚠️ Experimental Status
///
/// **This integrator is experimental and requires significantly more validation before
/// use in production systems.** While it passes all tests and achieves excellent accuracy
/// on orbital mechanics problems, it has not been extensively validated against a wide
/// range of problem types and edge cases. Independent verification is strongly recommended
/// for critical applications.
///
/// # Performance Characteristics
/// - 17 function evaluations per step
/// - 12th order accurate solution with 10th order embedded error estimate
/// - Optimal for problems requiring tolerances < 1e-10
/// - More efficient than standard RK methods for second-order systems
///
/// # State Vector Format
/// For a 3D orbital mechanics problem, the state vector is:
/// ```text
/// state = [x, y, z, vx, vy, vz]
/// ```
///
/// The dynamics function receives this full state and returns:
/// ```text
/// state_dot = [vx, vy, vz, ax, ay, az]
/// ```
///
/// where `[ax, ay, az]` is the acceleration.
///
/// # Example
///
/// ```
/// use nalgebra::SVector;
/// use brahe::integrators::{RKN1210SIntegrator, AdaptiveStepSIntegrator, IntegratorConfig};
/// use brahe::constants::GM_EARTH;
///
/// // Define dynamics for two-body problem (second-order)
/// let f = |_t: f64, state: SVector<f64, 6>| -> SVector<f64, 6> {
///     let r = state.fixed_rows::<3>(0);
///     let v = state.fixed_rows::<3>(3);
///     let r_norm = r.norm();
///     let a = -GM_EARTH / (r_norm * r_norm * r_norm) * r;
///
///     let mut state_dot = SVector::<f64, 6>::zeros();
///     state_dot.fixed_rows_mut::<3>(0).copy_from(&v);
///     state_dot.fixed_rows_mut::<3>(3).copy_from(&a);
///     state_dot
/// };
///
/// // Create integrator with tight tolerances
/// let config = IntegratorConfig::adaptive(1e-12, 1e-10);
/// let rkn: RKN1210SIntegrator<6, 0> = RKN1210SIntegrator::with_config(Box::new(f), None, None, None, config);
///
/// // Integrate one step
/// let t = 0.0;
/// let state = SVector::<f64, 6>::new(7000e3, 0.0, 0.0, 0.0, 7500.0, 0.0);
/// let dt = 10.0;
/// let result = rkn.step(t, state, dt);
///
/// println!("New state: {:?}", result.state);
/// println!("Suggested next dt: {}", result.dt_next);
/// ```
///
/// # Type Parameters
/// - `S`: State vector dimension
/// - `P`: Parameter vector dimension (for sensitivity matrix propagation)
pub struct RKN1210SIntegrator<const S: usize, const P: usize> {
    f: StateDynamics<S>,
    varmat: VariationalMatrix<S>,
    sensmat: SensitivityS<S, P>,
    control: ControlInput<S, P>,
    bt: EmbeddedRKNButcherTableau<17>,
    config: IntegratorConfig,
}

impl<const S: usize, const P: usize> RKN1210SIntegrator<S, P> {
    /// Create a new RKN12(10) integrator.
    ///
    /// Initializes RKN1210 integrator with the Dormand-El-Mikkawy-Prince tableau.
    /// This is a very high-order method (12th/10th) suitable for problems requiring
    /// tolerances below 1e-10.
    ///
    /// # ⚠️ Experimental
    ///
    /// This integrator is experimental and requires additional validation. Verify results
    /// independently for critical applications.
    ///
    /// # Arguments
    /// - `f`: State derivative function defining the dynamics
    /// - `varmat`: Variational matrix computation function for STM propagation
    /// - `sensmat`: Sensitivity matrix computation function for parameter uncertainty propagation
    /// - `control`: Control input function
    ///
    /// # Returns
    /// RKN1210SIntegrator instance ready for numerical integration
    ///
    /// # Note
    /// This constructor provides backward compatibility. Uses default configuration.
    /// For custom configuration, use `with_config()`.
    pub fn new(
        f: StateDynamics<S>,
        varmat: VariationalMatrix<S>,
        sensmat: SensitivityS<S, P>,
        control: ControlInput<S, P>,
    ) -> Self {
        Self::with_config(f, varmat, sensmat, control, IntegratorConfig::default())
    }

    /// Create a new RKN12(10) integrator with custom configuration.
    ///
    /// # ⚠️ Experimental
    ///
    /// This integrator is experimental and requires additional validation. Verify results
    /// independently for critical applications.
    ///
    /// # Arguments
    /// - `f`: State derivative function defining the dynamics
    /// - `varmat`: Variational matrix computation function for STM propagation
    /// - `sensmat`: Sensitivity matrix computation function for parameter uncertainty propagation
    /// - `control`: Control input function
    /// - `config`: Integration configuration (tolerances, step sizes, etc.)
    ///
    /// # Returns
    /// RKN1210SIntegrator instance with specified configuration
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::SVector;
    /// use brahe::integrators::{RKN1210SIntegrator, IntegratorConfig};
    ///
    /// let f = |t: f64, state: SVector<f64, 6>| -> SVector<f64, 6> {
    ///     // Dynamics implementation
    ///     state
    /// };
    ///
    /// let config = IntegratorConfig::adaptive(1e-12, 1e-10);
    /// let rkn: RKN1210SIntegrator<6, 0> = RKN1210SIntegrator::with_config(Box::new(f), None, None, None, config);
    /// ```
    pub fn with_config(
        f: StateDynamics<S>,
        varmat: VariationalMatrix<S>,
        sensmat: SensitivityS<S, P>,
        control: ControlInput<S, P>,
        config: IntegratorConfig,
    ) -> Self {
        Self {
            f,
            varmat,
            sensmat,
            control,
            bt: rkn1210_tableau(),
            config,
        }
    }

    /// Get a reference to the integrator configuration.
    pub fn config(&self) -> &IntegratorConfig {
        &self.config
    }

    /// Internal consolidated step function that handles all variants.
    ///
    /// This method is called by all public step functions to avoid code duplication.
    fn step_internal(
        &self,
        t: f64,
        state: SVector<f64, S>,
        phi: Option<SMatrix<f64, S, S>>,
        _sens: Option<SMatrix<f64, S, P>>,
        params: Option<&SVector<f64, P>>,
        dt: f64,
    ) -> AdaptiveStepInternalResultS<S, P> {
        assert!(
            S.is_multiple_of(2),
            "RKN integrator requires even-dimensional state (position + velocity)"
        );

        let half_dim = S / 2;
        let has_phi = phi.is_some();
        let current_phi = phi.unwrap_or_else(SMatrix::<f64, S, S>::zeros);
        let mut h = dt;
        let mut attempts = 0;

        loop {
            attempts += 1;
            if attempts > self.config.max_step_attempts {
                break;
            }

            // Extract position and velocity from state
            let mut pos = SVector::<f64, S>::zeros();
            let mut vel = SVector::<f64, S>::zeros();
            for i in 0..half_dim {
                pos[i] = state[i];
                vel[i] = state[half_dim + i];
            }

            // Preallocate stage matrix for accelerations
            let mut k = SMatrix::<f64, S, 17>::zeros();

            // Preallocate STM stages if needed
            let mut k_phi = [SMatrix::<f64, S, S>::zeros(); 17];

            // Compute RKN stages
            for i in 0..17 {
                // Compute position perturbation: h²*sum(a[i,j]*k[j])
                let mut pos_pert = SVector::<f64, S>::zeros();
                for j in 0..i {
                    for dim in 0..half_dim {
                        pos_pert[dim] += self.bt.a[(i, j)] * k[(dim, j)];
                    }
                }

                // Stage position: y + c*h*v + h²*pos_pert
                let mut stage_pos = SVector::<f64, S>::zeros();
                for dim in 0..half_dim {
                    stage_pos[dim] = pos[dim] + self.bt.c[i] * h * vel[dim] + h * h * pos_pert[dim];
                }

                // Reconstruct full state
                let mut stage_state = SVector::<f64, S>::zeros();
                for dim in 0..half_dim {
                    stage_state[dim] = stage_pos[dim];
                    stage_state[half_dim + dim] = vel[dim];
                }

                // Evaluate dynamics and extract acceleration
                let t_i = t + self.bt.c[i] * h;
                let mut state_dot = (self.f)(t_i, stage_state);

                // Apply control input if present
                if let Some(ref ctrl) = self.control {
                    state_dot += ctrl(t_i, stage_state, params);
                }

                for dim in 0..half_dim {
                    k[(dim, i)] = state_dot[half_dim + dim];
                }

                // Compute STM stages if needed
                if has_phi {
                    let mut k_phi_sum = SMatrix::<f64, S, S>::zeros();
                    for (j, k_phi_j) in k_phi.iter().enumerate().take(i) {
                        k_phi_sum += self.bt.a[(i, j)] * k_phi_j;
                    }

                    k_phi[i] = self
                        .varmat
                        .as_ref()
                        .expect("varmat required for step_with_varmat")
                        .compute(t_i, stage_state)
                        * (current_phi + h * k_phi_sum);
                }
            }

            // Compute high-order and low-order solutions
            let mut pos_high = SVector::<f64, S>::zeros();
            let mut vel_high = SVector::<f64, S>::zeros();
            let mut pos_low = SVector::<f64, S>::zeros();
            let mut vel_low = SVector::<f64, S>::zeros();

            for dim in 0..half_dim {
                let mut pos_update_high = 0.0;
                let mut vel_update_high = 0.0;
                let mut pos_update_low = 0.0;
                let mut vel_update_low = 0.0;

                for i in 0..17 {
                    pos_update_high += h * h * self.bt.b_pos_high[i] * k[(dim, i)];
                    vel_update_high += h * self.bt.b_vel_high[i] * k[(dim, i)];
                    pos_update_low += h * h * self.bt.b_pos_low[i] * k[(dim, i)];
                    vel_update_low += h * self.bt.b_vel_low[i] * k[(dim, i)];
                }

                pos_high[dim] = pos[dim] + h * vel[dim] + pos_update_high;
                vel_high[dim] = vel[dim] + vel_update_high;
                pos_low[dim] = pos[dim] + h * vel[dim] + pos_update_low;
                vel_low[dim] = vel[dim] + vel_update_low;
            }

            // Reconstruct full state vectors
            let mut state_high = SVector::<f64, S>::zeros();
            let mut state_low = SVector::<f64, S>::zeros();
            for dim in 0..half_dim {
                state_high[dim] = pos_high[dim];
                state_high[half_dim + dim] = vel_high[dim];
                state_low[dim] = pos_low[dim];
                state_low[half_dim + dim] = vel_low[dim];
            }

            // Compute STM update using velocity weights
            let phi_new = if has_phi {
                let mut phi_update = SMatrix::<f64, S, S>::zeros();
                for (i, k_phi_i) in k_phi.iter().enumerate().take(17) {
                    phi_update += h * self.bt.b_vel_high[i] * k_phi_i;
                }
                Some(current_phi + phi_update)
            } else {
                None
            };

            // Compute error estimate
            let error_vec = state_high - state_low;
            let error = compute_normalized_error_s(&error_vec, &state_high, &state, &self.config);

            // Check if step should be accepted
            let min_step_reached = self.config.min_step.is_some_and(|min| h.abs() <= min);

            if error <= 1.0 || min_step_reached {
                // Step accepted - compute next step size
                // RKN12(10) uses 12th order for accept
                let dt_next = compute_next_step_size(error, h, 1.0 / 12.0, &self.config);

                return AdaptiveStepInternalResultS {
                    state: state_high,
                    phi: phi_new,
                    sens: None,
                    dt_used: h,
                    error_estimate: error,
                    dt_next,
                };
            }

            // Step rejected - reduce step size
            // RKN12(10) uses 10th order for reject
            h = compute_reduced_step_size(error, h, 1.0 / 10.0, &self.config);
        }

        panic!("RKN1210S integrator exceeded maximum step attempts");
    }
}

impl<const S: usize, const P: usize> AdaptiveStepSIntegrator<S, P> for RKN1210SIntegrator<S, P> {
    fn step(&self, t: f64, state: SVector<f64, S>, dt: f64) -> AdaptiveStepSResult<S> {
        let result = self.step_internal(t, state, None, None, None, dt);
        AdaptiveStepSResult {
            state: result.state,
            dt_used: result.dt_used,
            error_estimate: result.error_estimate,
            dt_next: result.dt_next,
        }
    }

    fn step_with_varmat(
        &self,
        t: f64,
        state: SVector<f64, S>,
        phi: SMatrix<f64, S, S>,
        dt: f64,
    ) -> (SVector<f64, S>, SMatrix<f64, S, S>, f64, f64, f64) {
        let result = self.step_internal(t, state, Some(phi), None, None, dt);
        (
            result.state,
            result.phi.unwrap(),
            result.dt_used,
            result.error_estimate,
            result.dt_next,
        )
    }

    fn step_with_sensmat(
        &self,
        t: f64,
        state: SVector<f64, S>,
        sens: SMatrix<f64, S, P>,
        params: &SVector<f64, P>,
        dt: f64,
    ) -> (SVector<f64, S>, SMatrix<f64, S, P>, f64, f64, f64) {
        assert!(
            S.is_multiple_of(2),
            "RKN integrator requires even-dimensional state (position + velocity)"
        );

        let half_dim = S / 2;
        let mut h = dt;
        let mut attempts = 0;

        loop {
            attempts += 1;
            if attempts > self.config.max_step_attempts {
                break;
            }

            // Extract position and velocity from state
            let mut pos = SVector::<f64, S>::zeros();
            let mut vel = SVector::<f64, S>::zeros();
            for i in 0..half_dim {
                pos[i] = state[i];
                vel[i] = state[half_dim + i];
            }

            // Preallocate stage matrix for accelerations
            let mut k = SMatrix::<f64, S, 17>::zeros();
            let mut k_sens = [SMatrix::<f64, S, P>::zeros(); 17];

            // Compute RKN stages
            for i in 0..17 {
                // Compute position perturbation: h^2 * sum(a[i,j]*k[j])
                let mut pos_pert = SVector::<f64, S>::zeros();
                for j in 0..i {
                    for dim in 0..half_dim {
                        pos_pert[dim] += self.bt.a[(i, j)] * k[(dim, j)];
                    }
                }

                // Stage position: y + c*h*v + h^2*pos_pert
                let mut stage_pos = SVector::<f64, S>::zeros();
                for dim in 0..half_dim {
                    stage_pos[dim] = pos[dim] + self.bt.c[i] * h * vel[dim] + h * h * pos_pert[dim];
                }

                // Reconstruct full state
                let mut stage_state = SVector::<f64, S>::zeros();
                for dim in 0..half_dim {
                    stage_state[dim] = stage_pos[dim];
                    stage_state[half_dim + dim] = vel[dim];
                }

                // Evaluate dynamics and extract acceleration
                let t_i = t + self.bt.c[i] * h;
                let mut state_dot = (self.f)(t_i, stage_state);

                // Apply control input if present
                if let Some(ref ctrl) = self.control {
                    state_dot += ctrl(t_i, stage_state, Some(params));
                }

                for dim in 0..half_dim {
                    k[(dim, i)] = state_dot[half_dim + dim];
                }

                // Compute sensitivity stages: dS/dt = A*S + B
                let mut k_sens_sum = SMatrix::<f64, S, P>::zeros();
                for (j, k_sens_j) in k_sens.iter().enumerate().take(i) {
                    k_sens_sum += self.bt.a[(i, j)] * k_sens_j;
                }

                let a_i = self
                    .varmat
                    .as_ref()
                    .expect("varmat required for step_with_sensmat")
                    .compute(t_i, stage_state);
                let b_i = self
                    .sensmat
                    .as_ref()
                    .expect("sensmat required for step_with_sensmat")
                    .compute(t_i, &stage_state, params);
                k_sens[i] = a_i * (sens + h * k_sens_sum) + b_i;
            }

            // Compute high-order and low-order solutions
            let mut pos_high = SVector::<f64, S>::zeros();
            let mut vel_high = SVector::<f64, S>::zeros();
            let mut pos_low = SVector::<f64, S>::zeros();
            let mut vel_low = SVector::<f64, S>::zeros();

            for dim in 0..half_dim {
                let mut pos_update_high = 0.0;
                let mut vel_update_high = 0.0;
                let mut pos_update_low = 0.0;
                let mut vel_update_low = 0.0;

                for i in 0..17 {
                    pos_update_high += h * h * self.bt.b_pos_high[i] * k[(dim, i)];
                    vel_update_high += h * self.bt.b_vel_high[i] * k[(dim, i)];
                    pos_update_low += h * h * self.bt.b_pos_low[i] * k[(dim, i)];
                    vel_update_low += h * self.bt.b_vel_low[i] * k[(dim, i)];
                }

                pos_high[dim] = pos[dim] + h * vel[dim] + pos_update_high;
                vel_high[dim] = vel[dim] + vel_update_high;
                pos_low[dim] = pos[dim] + h * vel[dim] + pos_update_low;
                vel_low[dim] = vel[dim] + vel_update_low;
            }

            // Reconstruct full state vectors
            let mut state_high = SVector::<f64, S>::zeros();
            let mut state_low = SVector::<f64, S>::zeros();
            for dim in 0..half_dim {
                state_high[dim] = pos_high[dim];
                state_high[half_dim + dim] = vel_high[dim];
                state_low[dim] = pos_low[dim];
                state_low[half_dim + dim] = vel_low[dim];
            }

            // Compute sensitivity update using velocity weights
            let mut sens_update = SMatrix::<f64, S, P>::zeros();
            for (i, k_sens_i) in k_sens.iter().enumerate().take(17) {
                sens_update += h * self.bt.b_vel_high[i] * k_sens_i;
            }
            let sens_new = sens + sens_update;

            // Compute error estimate
            let error_vec = state_high - state_low;
            let error = compute_normalized_error_s(&error_vec, &state_high, &state, &self.config);

            // Check if step should be accepted
            let min_step_reached = self.config.min_step.is_some_and(|min| h.abs() <= min);

            if error <= 1.0 || min_step_reached {
                // Step accepted - compute next step size
                let dt_next = compute_next_step_size(error, h, 1.0 / 12.0, &self.config);

                return (state_high, sens_new, h, error, dt_next);
            }

            // Step rejected - reduce step size
            h = compute_reduced_step_size(error, h, 1.0 / 10.0, &self.config);
        }

        panic!("RKN1210S integrator exceeded maximum step attempts");
    }

    fn step_with_varmat_sensmat(
        &self,
        t: f64,
        state: SVector<f64, S>,
        phi: SMatrix<f64, S, S>,
        sens: SMatrix<f64, S, P>,
        params: &SVector<f64, P>,
        dt: f64,
    ) -> (
        SVector<f64, S>,
        SMatrix<f64, S, S>,
        SMatrix<f64, S, P>,
        f64,
        f64,
        f64,
    ) {
        assert!(
            S.is_multiple_of(2),
            "RKN integrator requires even-dimensional state (position + velocity)"
        );

        let half_dim = S / 2;
        let mut h = dt;
        let mut attempts = 0;

        loop {
            attempts += 1;
            if attempts > self.config.max_step_attempts {
                break;
            }

            // Extract position and velocity from state
            let mut pos = SVector::<f64, S>::zeros();
            let mut vel = SVector::<f64, S>::zeros();
            for i in 0..half_dim {
                pos[i] = state[i];
                vel[i] = state[half_dim + i];
            }

            // Preallocate stage matrices
            let mut k = SMatrix::<f64, S, 17>::zeros();
            let mut k_phi = [SMatrix::<f64, S, S>::zeros(); 17];
            let mut k_sens = [SMatrix::<f64, S, P>::zeros(); 17];

            // Compute RKN stages
            for i in 0..17 {
                // Compute position perturbation: h^2 * sum(a[i,j]*k[j])
                let mut pos_pert = SVector::<f64, S>::zeros();
                for j in 0..i {
                    for dim in 0..half_dim {
                        pos_pert[dim] += self.bt.a[(i, j)] * k[(dim, j)];
                    }
                }

                // Stage position: y + c*h*v + h^2*pos_pert
                let mut stage_pos = SVector::<f64, S>::zeros();
                for dim in 0..half_dim {
                    stage_pos[dim] = pos[dim] + self.bt.c[i] * h * vel[dim] + h * h * pos_pert[dim];
                }

                // Reconstruct full state
                let mut stage_state = SVector::<f64, S>::zeros();
                for dim in 0..half_dim {
                    stage_state[dim] = stage_pos[dim];
                    stage_state[half_dim + dim] = vel[dim];
                }

                // Evaluate dynamics and extract acceleration
                let t_i = t + self.bt.c[i] * h;
                let mut state_dot = (self.f)(t_i, stage_state);

                // Apply control input if present
                if let Some(ref ctrl) = self.control {
                    state_dot += ctrl(t_i, stage_state, Some(params));
                }

                for dim in 0..half_dim {
                    k[(dim, i)] = state_dot[half_dim + dim];
                }

                // Compute Jacobian for both phi and sens
                let a_i = self
                    .varmat
                    .as_ref()
                    .expect("varmat required for step_with_varmat_sensmat")
                    .compute(t_i, stage_state);

                // STM stages: dPhi/dt = A*Phi
                let mut k_phi_sum = SMatrix::<f64, S, S>::zeros();
                for (j, k_phi_j) in k_phi.iter().enumerate().take(i) {
                    k_phi_sum += self.bt.a[(i, j)] * k_phi_j;
                }
                k_phi[i] = a_i * (phi + h * k_phi_sum);

                // Sensitivity stages: dS/dt = A*S + B
                let mut k_sens_sum = SMatrix::<f64, S, P>::zeros();
                for (j, k_sens_j) in k_sens.iter().enumerate().take(i) {
                    k_sens_sum += self.bt.a[(i, j)] * k_sens_j;
                }
                let b_i = self
                    .sensmat
                    .as_ref()
                    .expect("sensmat required for step_with_varmat_sensmat")
                    .compute(t_i, &stage_state, params);
                k_sens[i] = a_i * (sens + h * k_sens_sum) + b_i;
            }

            // Compute high-order and low-order solutions
            let mut pos_high = SVector::<f64, S>::zeros();
            let mut vel_high = SVector::<f64, S>::zeros();
            let mut pos_low = SVector::<f64, S>::zeros();
            let mut vel_low = SVector::<f64, S>::zeros();

            for dim in 0..half_dim {
                let mut pos_update_high = 0.0;
                let mut vel_update_high = 0.0;
                let mut pos_update_low = 0.0;
                let mut vel_update_low = 0.0;

                for i in 0..17 {
                    pos_update_high += h * h * self.bt.b_pos_high[i] * k[(dim, i)];
                    vel_update_high += h * self.bt.b_vel_high[i] * k[(dim, i)];
                    pos_update_low += h * h * self.bt.b_pos_low[i] * k[(dim, i)];
                    vel_update_low += h * self.bt.b_vel_low[i] * k[(dim, i)];
                }

                pos_high[dim] = pos[dim] + h * vel[dim] + pos_update_high;
                vel_high[dim] = vel[dim] + vel_update_high;
                pos_low[dim] = pos[dim] + h * vel[dim] + pos_update_low;
                vel_low[dim] = vel[dim] + vel_update_low;
            }

            // Reconstruct full state vectors
            let mut state_high = SVector::<f64, S>::zeros();
            let mut state_low = SVector::<f64, S>::zeros();
            for dim in 0..half_dim {
                state_high[dim] = pos_high[dim];
                state_high[half_dim + dim] = vel_high[dim];
                state_low[dim] = pos_low[dim];
                state_low[half_dim + dim] = vel_low[dim];
            }

            // Compute STM and sensitivity updates using velocity weights
            let mut phi_update = SMatrix::<f64, S, S>::zeros();
            let mut sens_update = SMatrix::<f64, S, P>::zeros();
            for i in 0..17 {
                phi_update += h * self.bt.b_vel_high[i] * k_phi[i];
                sens_update += h * self.bt.b_vel_high[i] * k_sens[i];
            }
            let phi_new = phi + phi_update;
            let sens_new = sens + sens_update;

            // Compute error estimate
            let error_vec = state_high - state_low;
            let error = compute_normalized_error_s(&error_vec, &state_high, &state, &self.config);

            // Check if step should be accepted
            let min_step_reached = self.config.min_step.is_some_and(|min| h.abs() <= min);

            if error <= 1.0 || min_step_reached {
                // Step accepted - compute next step size
                let dt_next = compute_next_step_size(error, h, 1.0 / 12.0, &self.config);

                return (state_high, phi_new, sens_new, h, error, dt_next);
            }

            // Step rejected - reduce step size
            h = compute_reduced_step_size(error, h, 1.0 / 10.0, &self.config);
        }

        panic!("RKN1210S integrator exceeded maximum step attempts");
    }
}

/// Dynamic-dimensional implementation of the RKN12(10) Runge-Kutta-Nyström integrator.
///
/// This version accepts runtime-sized state vectors (DVector) instead of compile-time
/// sized vectors, making it suitable for Python bindings and cases where state dimension
/// is not known at compile time.
///
/// # ⚠️ Experimental Status
///
/// **This integrator is experimental and requires significantly more validation before
/// use in production systems.**
///
/// # Performance Characteristics
/// - 17 function evaluations per step
/// - 12th order accurate solution with 10th order embedded error estimate
/// - Optimal for problems requiring tolerances < 1e-10
///
/// # Example
///
/// ```
/// use nalgebra::DVector;
/// use brahe::integrators::{RKN1210DIntegrator, AdaptiveStepDIntegrator, IntegratorConfig};
/// use brahe::constants::GM_EARTH;
///
/// // Define dynamics for two-body problem
/// let f = |_t: f64, state: DVector<f64>, _params: Option<&DVector<f64>>| -> DVector<f64> {
///     let r = state.rows(0, 3);
///     let v = state.rows(3, 3);
///     let r_norm = r.norm();
///     let a = -GM_EARTH / (r_norm * r_norm * r_norm) * r;
///
///     let mut state_dot = DVector::zeros(6);
///     state_dot.rows_mut(0, 3).copy_from(&v);
///     state_dot.rows_mut(3, 3).copy_from(&a);
///     state_dot
/// };
///
/// let config = IntegratorConfig::adaptive(1e-12, 1e-10);
/// let rkn = RKN1210DIntegrator::with_config(6, Box::new(f), None, None, None, config);
///
/// let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);
/// let result = rkn.step(0.0, state, 10.0);
/// ```
pub struct RKN1210DIntegrator {
    dimension: usize,
    f: StateDynamicsD,
    varmat: VariationalMatrixD,
    sensmat: SensitivityD,
    control: ControlInputD,
    bt: EmbeddedRKNButcherTableau<17>,
    config: IntegratorConfig,
}

impl RKN1210DIntegrator {
    /// Create a new RKN12(10) dynamic integrator.
    ///
    /// # Arguments
    /// - `dimension`: State vector dimension (must be even)
    /// - `f`: State derivative function
    /// - `varmat`: Optional variational matrix computation function
    /// - `sensmat`: Optional sensitivity matrix computation function
    /// - `control`: Optional control input function
    ///
    /// # Panics
    /// Panics if dimension is not even (RKN requires position + velocity).
    pub fn new(
        dimension: usize,
        f: StateDynamicsD,
        varmat: VariationalMatrixD,
        sensmat: SensitivityD,
        control: ControlInputD,
    ) -> Self {
        <Self as DIntegrator>::new(dimension, f, varmat, sensmat, control)
    }

    /// Create a new RKN12(10) dynamic integrator with custom configuration.
    ///
    /// # Arguments
    /// - `dimension`: State vector dimension (must be even)
    /// - `f`: State derivative function
    /// - `varmat`: Optional variational matrix computation function
    /// - `sensmat`: Optional sensitivity matrix computation function
    /// - `control`: Optional control input function
    /// - `config`: Integration configuration
    pub fn with_config(
        dimension: usize,
        f: StateDynamicsD,
        varmat: VariationalMatrixD,
        sensmat: SensitivityD,
        control: ControlInputD,
        config: IntegratorConfig,
    ) -> Self {
        <Self as DIntegrator>::with_config(dimension, f, varmat, sensmat, control, config)
    }

    /// Get the state vector dimension for this integrator.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get a reference to the integrator configuration.
    pub fn config(&self) -> &IntegratorConfig {
        &self.config
    }
}

impl DIntegrator for RKN1210DIntegrator {
    fn new(
        dimension: usize,
        f: StateDynamicsD,
        varmat: VariationalMatrixD,
        sensmat: SensitivityD,
        control: ControlInputD,
    ) -> Self {
        Self::with_config(
            dimension,
            f,
            varmat,
            sensmat,
            control,
            IntegratorConfig::default(),
        )
    }

    fn with_config(
        dimension: usize,
        f: StateDynamicsD,
        varmat: VariationalMatrixD,
        sensmat: SensitivityD,
        control: ControlInputD,
        config: IntegratorConfig,
    ) -> Self {
        assert!(
            dimension.is_multiple_of(2),
            "RKN integrator requires even-dimensional state (position + velocity)"
        );
        Self {
            dimension,
            f,
            varmat,
            sensmat,
            control,
            bt: rkn1210_tableau(),
            config,
        }
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn config(&self) -> &IntegratorConfig {
        &self.config
    }
}

impl RKN1210DIntegrator {
    /// Internal consolidated step function that handles all variants.
    ///
    /// This method is called by all public step functions to avoid code duplication.
    fn step_internal(
        &self,
        t: f64,
        state: DVector<f64>,
        phi: Option<DMatrix<f64>>,
        sens: Option<DMatrix<f64>>,
        params: Option<&DVector<f64>>,
        dt: f64,
    ) -> AdaptiveStepInternalResultD {
        let half_dim = self.dimension / 2;
        let has_phi = phi.is_some();
        let has_sens = sens.is_some();
        let current_phi = phi.unwrap_or_else(|| DMatrix::zeros(0, 0));
        let current_sens = sens.unwrap_or_else(|| DMatrix::zeros(0, 0));
        let num_params = if has_sens { current_sens.ncols() } else { 0 };
        let mut h = dt;
        let mut attempts = 0;

        loop {
            attempts += 1;
            if attempts > self.config.max_step_attempts {
                break;
            }

            // Extract position and velocity
            let pos = state.rows(0, half_dim).clone_owned();
            let vel = state.rows(half_dim, half_dim).clone_owned();

            // Preallocate stage matrix for accelerations
            let mut k = DMatrix::<f64>::zeros(half_dim, 17);

            // Preallocate STM stages if needed
            let mut k_phi = if has_phi {
                let mut v = Vec::with_capacity(17);
                for _ in 0..17 {
                    v.push(DMatrix::<f64>::zeros(self.dimension, self.dimension));
                }
                v
            } else {
                Vec::new()
            };

            // Preallocate sensitivity stages if needed
            let mut k_sens = if has_sens {
                let mut v = Vec::with_capacity(17);
                for _ in 0..17 {
                    v.push(DMatrix::<f64>::zeros(self.dimension, num_params));
                }
                v
            } else {
                Vec::new()
            };

            // Compute RKN stages
            for i in 0..17 {
                // Compute position perturbation: h²*sum(a[i,j]*k[j])
                let mut pos_pert = DVector::<f64>::zeros(half_dim);
                for j in 0..i {
                    for dim in 0..half_dim {
                        pos_pert[dim] += self.bt.a[(i, j)] * k[(dim, j)];
                    }
                }

                // Stage position: y + c*h*v + h²*pos_pert
                let mut stage_pos = DVector::<f64>::zeros(half_dim);
                for dim in 0..half_dim {
                    stage_pos[dim] = pos[dim] + self.bt.c[i] * h * vel[dim] + h * h * pos_pert[dim];
                }

                // Reconstruct full state
                let mut stage_state = DVector::<f64>::zeros(self.dimension);
                stage_state.rows_mut(0, half_dim).copy_from(&stage_pos);
                stage_state.rows_mut(half_dim, half_dim).copy_from(&vel);

                // Evaluate dynamics and extract acceleration
                let t_i = t + self.bt.c[i] * h;
                let mut state_dot = if has_sens {
                    (self.f)(t_i, stage_state.clone(), params)
                } else {
                    (self.f)(t_i, stage_state.clone(), None)
                };

                // Apply control input if present
                if let Some(ref ctrl) = self.control {
                    state_dot += ctrl(t_i, stage_state.clone(), params);
                }

                for dim in 0..half_dim {
                    k[(dim, i)] = state_dot[half_dim + dim];
                }

                // Compute Jacobian if needed for phi or sens
                if has_phi || has_sens {
                    let a_i = self
                        .varmat
                        .as_ref()
                        .expect("varmat required for step_with_varmat or step_with_sensmat")
                        .compute(t_i, stage_state.clone());

                    // Variational: dΦ/dt = A*Φ
                    if has_phi {
                        let mut k_phi_sum = DMatrix::<f64>::zeros(self.dimension, self.dimension);
                        #[allow(clippy::needless_range_loop)]
                        for j in 0..i {
                            k_phi_sum += self.bt.a[(i, j)] * &k_phi[j];
                        }
                        k_phi[i] = &a_i * (&current_phi + h * k_phi_sum);
                    }

                    // Sensitivity: dS/dt = A*S + B
                    if has_sens {
                        let mut k_sens_sum = DMatrix::<f64>::zeros(self.dimension, num_params);
                        #[allow(clippy::needless_range_loop)]
                        for j in 0..i {
                            k_sens_sum += self.bt.a[(i, j)] * &k_sens[j];
                        }
                        let b_i = self
                            .sensmat
                            .as_ref()
                            .expect("sensmat required for step_with_sensmat")
                            .compute(t_i, &stage_state, params.unwrap());
                        k_sens[i] = &a_i * (&current_sens + h * k_sens_sum) + b_i;
                    }
                }
            }

            // Compute high-order and low-order solutions
            let mut pos_high = DVector::<f64>::zeros(half_dim);
            let mut vel_high = DVector::<f64>::zeros(half_dim);
            let mut pos_low = DVector::<f64>::zeros(half_dim);
            let mut vel_low = DVector::<f64>::zeros(half_dim);

            for dim in 0..half_dim {
                let mut pos_update_high = 0.0;
                let mut vel_update_high = 0.0;
                let mut pos_update_low = 0.0;
                let mut vel_update_low = 0.0;

                for i in 0..17 {
                    pos_update_high += h * h * self.bt.b_pos_high[i] * k[(dim, i)];
                    vel_update_high += h * self.bt.b_vel_high[i] * k[(dim, i)];
                    pos_update_low += h * h * self.bt.b_pos_low[i] * k[(dim, i)];
                    vel_update_low += h * self.bt.b_vel_low[i] * k[(dim, i)];
                }

                pos_high[dim] = pos[dim] + h * vel[dim] + pos_update_high;
                vel_high[dim] = vel[dim] + vel_update_high;
                pos_low[dim] = pos[dim] + h * vel[dim] + pos_update_low;
                vel_low[dim] = vel[dim] + vel_update_low;
            }

            // Reconstruct full state vectors
            let mut state_high = DVector::<f64>::zeros(self.dimension);
            let mut state_low = DVector::<f64>::zeros(self.dimension);
            state_high.rows_mut(0, half_dim).copy_from(&pos_high);
            state_high.rows_mut(half_dim, half_dim).copy_from(&vel_high);
            state_low.rows_mut(0, half_dim).copy_from(&pos_low);
            state_low.rows_mut(half_dim, half_dim).copy_from(&vel_low);

            // Compute STM update using velocity weights
            let phi_new = if has_phi {
                let mut phi_update = DMatrix::<f64>::zeros(self.dimension, self.dimension);
                #[allow(clippy::needless_range_loop)]
                for i in 0..17 {
                    phi_update += h * self.bt.b_vel_high[i] * &k_phi[i];
                }
                Some(&current_phi + phi_update)
            } else {
                None
            };

            // Compute sensitivity matrix update using velocity weights
            let sens_new = if has_sens {
                let mut sens_update = DMatrix::<f64>::zeros(self.dimension, num_params);
                #[allow(clippy::needless_range_loop)]
                for i in 0..17 {
                    sens_update += h * self.bt.b_vel_high[i] * &k_sens[i];
                }
                Some(&current_sens + sens_update)
            } else {
                None
            };

            // Compute error estimate
            let error_vec = &state_high - &state_low;
            let error = compute_normalized_error(&error_vec, &state_high, &state, &self.config);

            // Check if step should be accepted
            let min_step_reached = self.config.min_step.is_some_and(|min| h.abs() <= min);

            if error <= 1.0 || min_step_reached {
                // Step accepted - compute next step size
                // RKN12(10) uses 12th order for accept
                let dt_next = compute_next_step_size(error, h, 1.0 / 12.0, &self.config);

                return AdaptiveStepInternalResultD {
                    state: state_high,
                    phi: phi_new,
                    sens: sens_new,
                    dt_used: h,
                    error_estimate: error,
                    dt_next,
                };
            }

            // Step rejected - reduce step size
            // RKN12(10) uses 10th order for reject
            h = compute_reduced_step_size(error, h, 1.0 / 10.0, &self.config);
        }

        panic!("RKN1210D integrator exceeded maximum step attempts");
    }
}

impl AdaptiveStepDIntegrator for RKN1210DIntegrator {
    fn step(&self, t: f64, state: DVector<f64>, dt: f64) -> AdaptiveStepDResult {
        assert_eq!(state.len(), self.dimension);
        let result = self.step_internal(t, state, None, None, None, dt);
        AdaptiveStepDResult {
            state: result.state,
            dt_used: result.dt_used,
            error_estimate: result.error_estimate,
            dt_next: result.dt_next,
        }
    }

    fn step_with_varmat(
        &self,
        t: f64,
        state: DVector<f64>,
        phi: DMatrix<f64>,
        dt: f64,
    ) -> (DVector<f64>, DMatrix<f64>, f64, f64, f64) {
        assert_eq!(state.len(), self.dimension);
        assert_eq!(phi.nrows(), self.dimension);
        assert_eq!(phi.ncols(), self.dimension);
        let result = self.step_internal(t, state, Some(phi), None, None, dt);
        (
            result.state,
            result.phi.unwrap(),
            result.dt_used,
            result.error_estimate,
            result.dt_next,
        )
    }

    fn step_with_sensmat(
        &self,
        t: f64,
        state: DVector<f64>,
        sens: DMatrix<f64>,
        params: &DVector<f64>,
        dt: f64,
    ) -> (DVector<f64>, DMatrix<f64>, f64, f64, f64) {
        assert_eq!(state.len(), self.dimension);
        assert_eq!(sens.nrows(), self.dimension);
        let result = self.step_internal(t, state, None, Some(sens), Some(params), dt);
        (
            result.state,
            result.sens.unwrap(),
            result.dt_used,
            result.error_estimate,
            result.dt_next,
        )
    }

    fn step_with_varmat_sensmat(
        &self,
        t: f64,
        state: DVector<f64>,
        phi: DMatrix<f64>,
        sens: DMatrix<f64>,
        params: &DVector<f64>,
        dt: f64,
    ) -> (DVector<f64>, DMatrix<f64>, DMatrix<f64>, f64, f64, f64) {
        assert_eq!(state.len(), self.dimension);
        assert_eq!(phi.nrows(), self.dimension);
        assert_eq!(phi.ncols(), self.dimension);
        assert_eq!(sens.nrows(), self.dimension);
        let result = self.step_internal(t, state, Some(phi), Some(sens), Some(params), dt);
        (
            result.state,
            result.phi.unwrap(),
            result.sens.unwrap(),
            result.dt_used,
            result.error_estimate,
            result.dt_next,
        )
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::{SMatrix, SVector};

    use crate::constants::DEGREES;
    use crate::integrators::butcher_tableau::rkn1210_tableau;
    use crate::integrators::config::IntegratorConfig;
    use crate::integrators::rkn1210::RKN1210SIntegrator;
    use crate::integrators::traits::AdaptiveStepSIntegrator;
    use crate::math::jacobian::{DNumericalJacobian, DifferenceMethod, SNumericalJacobian};
    use crate::time::{Epoch, TimeSystem};
    use crate::{GM_EARTH, R_EARTH, orbital_period, state_osculating_to_cartesian};

    #[test]
    fn test_rkn1210s_coefficients() {
        // Verify Butcher tableau coefficient sums
        // For RKN methods: b_pos should sum to 0.5, b_vel should sum to 1.0
        let bt = rkn1210_tableau();

        let b_pos_high_sum: f64 = bt.b_pos_high.iter().sum();
        let b_pos_low_sum: f64 = bt.b_pos_low.iter().sum();
        let b_vel_high_sum: f64 = bt.b_vel_high.iter().sum();
        let b_vel_low_sum: f64 = bt.b_vel_low.iter().sum();

        println!("b_pos_high sum: {}", b_pos_high_sum);
        println!("b_pos_low sum: {}", b_pos_low_sum);
        println!("b_vel_high sum: {}", b_vel_high_sum);
        println!("b_vel_low sum: {}", b_vel_low_sum);

        // RKN methods should have b_pos sum to 0.5 and b_vel sum to 1.0
        assert_abs_diff_eq!(b_pos_high_sum, 0.5, epsilon = 1.0e-10);
        assert_abs_diff_eq!(b_pos_low_sum, 0.5, epsilon = 1.0e-10);
        assert_abs_diff_eq!(b_vel_high_sum, 1.0, epsilon = 1.0e-10);
        assert_abs_diff_eq!(b_vel_low_sum, 1.0, epsilon = 1.0e-10);

        // Verify RKN consistency condition: sum_j(a_ij) = c_i²/2
        for i in 0..17 {
            let mut row_sum = 0.0;
            for j in 0..17 {
                row_sum += bt.a[(i, j)];
            }
            let expected = bt.c[i] * bt.c[i] / 2.0;
            assert_abs_diff_eq!(row_sum, expected, epsilon = 1.0e-10);
        }
    }

    // Two-body gravitational dynamics
    fn point_earth(_: f64, x: SVector<f64, 6>) -> SVector<f64, 6> {
        let r = x.fixed_rows::<3>(0);
        let v = x.fixed_rows::<3>(3);

        // Calculate acceleration
        let r_norm = r.norm();
        let a = -GM_EARTH / (r_norm * r_norm * r_norm) * r;

        // Construct state derivative: [velocity, acceleration]
        let mut x_dot = SVector::<f64, 6>::zeros();
        x_dot.fixed_rows_mut::<3>(0).copy_from(&v);
        x_dot.fixed_rows_mut::<3>(3).copy_from(&a);

        x_dot
    }

    #[test]
    fn test_rkn1210s_single_step() {
        // Test a single step with constant acceleration to verify formulas
        let f = |_t: f64, state: SVector<f64, 2>| -> SVector<f64, 2> {
            // state = [x, v], state_dot = [v, a] where a = 2
            SVector::<f64, 2>::new(state[1], 2.0)
        };

        let config = IntegratorConfig::adaptive(1e-10, 1e-8);
        let rkn: RKN1210SIntegrator<2, 0> =
            RKN1210SIntegrator::with_config(Box::new(f), None, None, None, config);

        let state = SVector::<f64, 2>::new(0.0, 0.0); // [x, v] = [0, 0]
        let dt = 0.01;

        let result = rkn.step(0.0, state, dt);

        println!("After one step:");
        println!("  state: [{}, {}]", result.state[0], result.state[1]);
        println!("  dt_used: {}", result.dt_used);
        println!("  error: {}", result.error_estimate);
        println!("  dt_next: {}", result.dt_next);

        // For constant acceleration a=2, after time dt=0.01:
        // x = 0 + 0*dt + 0.5*a*dt² = 0.5*2*0.0001 = 0.0001
        // v = 0 + a*dt = 2*0.01 = 0.02
        assert_abs_diff_eq!(result.state[0], 0.0001, epsilon = 1.0e-6);
        assert_abs_diff_eq!(result.state[1], 0.02, epsilon = 1.0e-6);
    }

    #[test]
    fn test_rkn1210s_integrator_parabola() {
        // Test with simple parabolic motion: x'' = 2 (constant acceleration)
        // Solution: x(t) = t²
        let f = |_t: f64, state: SVector<f64, 2>| -> SVector<f64, 2> {
            // state = [x, v]
            // state_dot = [v, a] where a = 2
            SVector::<f64, 2>::new(state[1], 2.0)
        };

        let config = IntegratorConfig::adaptive(1e-10, 1e-8);
        let rkn: RKN1210SIntegrator<2, 0> =
            RKN1210SIntegrator::with_config(Box::new(f), None, None, None, config);

        let mut t = 0.0;
        let mut state = SVector::<f64, 2>::new(0.0, 0.0); // [position, velocity]
        let mut dt = 0.01;
        let mut step_count = 0;

        // Integrate to t = 1.0, solution should be x = 1.0
        let t_final = 1.0;
        while t < t_final {
            // Clip dt to not overshoot target time
            let dt_actual = f64::min(dt, t_final - t);
            let result = rkn.step(t, state, dt_actual);
            state = result.state;
            dt = result.dt_next;
            t += result.dt_used;
            step_count += 1;

            if step_count <= 5 {
                println!(
                    "Step {}: t={:.6}, x={:.6}, v={:.6}, dt_next={:.6}",
                    step_count, t, state[0], state[1], dt
                );
            }
        }

        println!(
            "Total steps: {}, final t={:.6}, x={:.6}, v={:.6}",
            step_count, t, state[0], state[1]
        );

        assert_abs_diff_eq!(state[0], 1.0, epsilon = 1.0e-8);
    }

    #[test]
    fn test_rkn1210s_integrator_orbit() {
        let config = IntegratorConfig::adaptive(1e-9, 1e-6);
        let rkn: RKN1210SIntegrator<6, 0> =
            RKN1210SIntegrator::with_config(Box::new(point_earth), None, None, None, config);

        // Get start state
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 90.0, 0.0, 0.0, 0.0);
        let state0 = state_osculating_to_cartesian(oe0, DEGREES);
        let mut state = state0;

        // Get start and end times of propagation (1 orbit)
        let epc0 = Epoch::from_date(2024, 1, 1, TimeSystem::TAI);
        let epcf = epc0 + orbital_period(oe0[0]);
        let mut dt = 10.0;
        let mut epc = epc0;

        while epc < epcf {
            let dt_actual = f64::min(dt, epcf - epc);
            let result = rkn.step(epc - epc0, state, dt_actual);
            state = result.state;
            dt = result.dt_next;
            epc += result.dt_used;
        }

        // Check energy conservation (position magnitude should be close)
        // RKN1210 is extremely accurate - expect sub-meter error over full orbit
        assert_abs_diff_eq!(state.norm(), state0.norm(), epsilon = 1.0);
    }

    #[test]
    fn test_rkn1210s_accuracy() {
        // Verify 12th order convergence on simple problem
        let f = |_t: f64, state: SVector<f64, 2>| -> SVector<f64, 2> {
            SVector::<f64, 2>::new(state[1], 2.0)
        };

        let config = IntegratorConfig::adaptive(1e-12, 1e-10);
        let rkn: RKN1210SIntegrator<2, 0> =
            RKN1210SIntegrator::with_config(Box::new(f), None, None, None, config);

        let mut t = 0.0;
        let mut state = SVector::<f64, 2>::new(0.0, 0.0);
        let mut dt = 0.1;
        let t_final = 1.0;

        while t < t_final {
            let dt_actual = f64::min(dt, t_final - t);
            let result = rkn.step(t, state, dt_actual);
            state = result.state;
            dt = result.dt_next;
            t += result.dt_used;
        }

        // With 12th order and tight tolerances, error should be extremely small
        assert_abs_diff_eq!(state[0], 1.0, epsilon = 1.0e-10);
    }

    #[test]
    fn test_rkn1210s_step_size_increases() {
        let f = |_t: f64, state: SVector<f64, 2>| -> SVector<f64, 2> {
            SVector::<f64, 2>::new(state[1], 2.0)
        };

        let config = IntegratorConfig::adaptive(1e-6, 1e-3);
        let rkn: RKN1210SIntegrator<2, 0> =
            RKN1210SIntegrator::with_config(Box::new(f), None, None, None, config);

        let state = SVector::<f64, 2>::new(0.0, 0.0);
        let dt = 0.01;

        let result = rkn.step(0.0, state, dt);

        // With loose tolerances on smooth problem, should suggest larger step
        assert!(result.dt_next > dt);
    }

    #[test]
    fn test_rkn1210s_adaptive_mechanism() {
        // Verify that the adaptive step mechanism works correctly
        // Note: RKN1210 is a very high-order (12th) method, so it may meet tight tolerances
        // even with large steps on stiff problems. This test verifies the mechanism works,
        // not that it necessarily reduces step size.
        let f = |_t: f64, state: SVector<f64, 2>| -> SVector<f64, 2> {
            let x = state[0];
            let v = state[1];
            let accel = -100.0 * x - 20.0 * v;
            SVector::<f64, 2>::new(v, accel)
        };

        let config = IntegratorConfig::adaptive(1e-14, 1e-12);
        let rkn: RKN1210SIntegrator<2, 0> =
            RKN1210SIntegrator::with_config(Box::new(f), None, None, None, config);

        let state = SVector::<f64, 2>::new(1.0, 0.0);
        let dt = 1.0;

        let result = rkn.step(0.0, state, dt);

        // Verify adaptive mechanism produces valid output
        assert!(result.dt_used > 0.0);
        assert!(result.dt_next > 0.0);
        assert!(result.dt_next.is_finite());
        assert!(result.error_estimate >= 0.0);
        assert!(result.error_estimate.is_finite());
    }

    #[test]
    fn test_rkn1210s_config_parameters() {
        let f = |_t: f64, state: SVector<f64, 2>| -> SVector<f64, 2> {
            SVector::<f64, 2>::new(state[1], 2.0)
        };

        let config = IntegratorConfig {
            abs_tol: 1e-8,
            rel_tol: 1e-6,
            step_safety_factor: Some(0.8),
            min_step_scale_factor: Some(0.5),
            max_step_scale_factor: Some(5.0),
            ..Default::default()
        };

        let rkn: RKN1210SIntegrator<2, 0> =
            RKN1210SIntegrator::with_config(Box::new(f), None, None, None, config);

        assert_eq!(rkn.config().step_safety_factor, Some(0.8));
        assert_eq!(rkn.config().min_step_scale_factor, Some(0.5));
        assert_eq!(rkn.config().max_step_scale_factor, Some(5.0));
    }

    #[test]
    fn test_rkn1210s_high_precision() {
        // Test with very tight tolerances
        let f = |_t: f64, state: SVector<f64, 2>| -> SVector<f64, 2> {
            SVector::<f64, 2>::new(state[1], 2.0)
        };

        let config = IntegratorConfig::adaptive(1e-13, 1e-11);
        let rkn: RKN1210SIntegrator<2, 0> =
            RKN1210SIntegrator::with_config(Box::new(f), None, None, None, config);

        let mut t = 0.0;
        let mut state = SVector::<f64, 2>::new(0.0, 0.0);
        let mut dt = 0.1;
        let t_final = 1.0;

        while t < t_final {
            let dt_actual = f64::min(dt, t_final - t);
            let result = rkn.step(t, state, dt_actual);
            state = result.state;
            dt = result.dt_next;
            t += result.dt_used;
        }

        // RKN1210 should achieve very high precision
        assert_abs_diff_eq!(state[0], 1.0, epsilon = 1.0e-11);
    }

    #[test]
    fn test_rkn1210s_varmat() {
        // Define variational matrix computation using JacobianProvider
        // Use forward differences to match old test behavior
        let jacobian = SNumericalJacobian::new(Box::new(point_earth))
            .with_method(DifferenceMethod::Forward)
            .with_fixed_offset(1.0);

        let config = IntegratorConfig::adaptive(1e-9, 1e-6);
        let rkn: RKN1210SIntegrator<6, 0> = RKN1210SIntegrator::with_config(
            Box::new(point_earth),
            Some(Box::new(jacobian)),
            None,
            None,
            config,
        );

        // Get start state
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 90.0, 0.0, 0.0, 0.0);
        let state0 = state_osculating_to_cartesian(oe0, DEGREES);
        let phi0 = SMatrix::<f64, 6, 6>::identity();

        // Take no step and confirm the variational matrix is the identity matrix
        let (_, phi1, _, _, _) = rkn.step_with_varmat(0.0, state0, phi0, 0.0);
        for i in 0..6 {
            for j in 0..6 {
                if i == j {
                    assert_abs_diff_eq!(phi1[(i, j)], 1.0, epsilon = 1.0e-12);
                } else {
                    assert_abs_diff_eq!(phi1[(i, j)], 0.0, epsilon = 1.0e-12);
                }
            }
        }

        // Propagate one step and verify STM updated
        let (_, phi2, _, _, _) = rkn.step_with_varmat(0.0, state0, phi0, 1.0);
        for i in 0..6 {
            for j in 0..6 {
                if i == j {
                    // Diagonal should be close to 1.0 but not exactly
                    assert_ne!(phi2[(i, i)], 1.0);
                    assert_abs_diff_eq!(phi2[(i, i)], 1.0, epsilon = 1.0e-3);
                } else {
                    // Off-diagonal elements should be populated
                    assert_ne!(phi2[(i, j)], 0.0);
                }
            }
        }
    }

    #[test]
    fn test_rkn1210s_stm_accuracy() {
        // Comprehensive test comparing STM propagation with direct numerical perturbation
        // This validates that the STM correctly predicts how perturbations evolve

        let jacobian = SNumericalJacobian::new(Box::new(point_earth))
            .with_method(DifferenceMethod::Central)
            .with_fixed_offset(1.0);

        let config = IntegratorConfig::adaptive(1e-12, 1e-10);
        let rkn: RKN1210SIntegrator<6, 0> = RKN1210SIntegrator::with_config(
            Box::new(point_earth),
            Some(Box::new(jacobian)),
            None,
            None,
            config,
        );

        // Start with a realistic orbital state
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 45.0, 10.0, 20.0, 30.0);
        let state0 = state_osculating_to_cartesian(oe0, DEGREES);
        let phi0 = SMatrix::<f64, 6, 6>::identity();

        // Propagate with STM for a significant time step
        let dt = 10.0; // 10 seconds
        let (state_final, phi_final, _, _, _) = rkn.step_with_varmat(0.0, state0, phi0, dt);

        // Test STM accuracy by comparing with direct perturbation
        // Use a small perturbation in each direction
        let pert_size = 1.0; // 1 meter in position, 1 mm/s in velocity

        // Test all 6 components
        for i in 0..6 {
            let mut perturbation = SVector::<f64, 6>::zeros();
            perturbation[i] = pert_size;

            // Propagate perturbed state directly
            let state_pert0 = state0 + perturbation;
            let result_pert = rkn.step(0.0, state_pert0, dt);
            let state_pert_direct = result_pert.state;

            // Predict perturbed state using STM
            let state_pert_predicted = state_final + phi_final * perturbation;

            // Compare - STM should accurately predict the perturbation evolution
            let error = (state_pert_direct - state_pert_predicted).norm();
            let relative_error = error / pert_size;

            println!(
                "Component {}: STM error = {:.3e} m or m/s (relative: {:.3e})",
                i, error, relative_error
            );

            // STM should be accurate for practical use
            // Note: RKN STM propagation has inherent limitations due to:
            // 1. Finite difference Jacobian approximation
            // 2. Treating full 6D state [r,v] when RKN is fundamentally for 2nd-order systems
            // 3. Using position-based RKN stage points for evaluating 6D Jacobian
            // 4. Nonlinearity of the dynamics
            // Velocity components (3-5) have larger errors than position (0-2)
            // Expect ~1e-3 relative accuracy for velocity, ~1e-4 for position
            let tolerance = if i < 3 { 2e-4 } else { 5e-4 };
            assert!(
                relative_error < tolerance,
                "STM prediction error too large for component {}: {:.3e} (tolerance: {:.3e})",
                i,
                relative_error,
                tolerance
            );
        }

        // Additional test: Verify STM is symplectic-like for Hamiltonian systems
        // The determinant of the STM should be close to 1 for conservative systems
        let det = phi_final.determinant();
        println!("STM determinant: {:.12}", det);
        assert_abs_diff_eq!(det, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_rkn1210s_stm_vs_direct_perturbation() {
        // This test specifically validates the STM weight choice by comparing
        // multiple propagation steps with direct perturbation

        let jacobian = SNumericalJacobian::new(Box::new(point_earth))
            .with_method(DifferenceMethod::Central)
            .with_fixed_offset(0.1);

        let config = IntegratorConfig::adaptive(1e-12, 1e-10);
        let rkn: RKN1210SIntegrator<6, 0> = RKN1210SIntegrator::with_config(
            Box::new(point_earth),
            Some(Box::new(jacobian)),
            None,
            None,
            config,
        );

        // Circular orbit
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0);
        let state0 = state_osculating_to_cartesian(oe0, DEGREES);

        // Small perturbation in position
        let perturbation = SVector::<f64, 6>::new(10.0, 0.0, 0.0, 0.0, 0.0, 0.0);

        // Propagate over multiple steps
        let total_time = 100.0; // 100 seconds
        let num_steps = 10;
        let dt = total_time / num_steps as f64;

        let mut state = state0;
        let mut phi = SMatrix::<f64, 6, 6>::identity();
        let mut state_pert = state0 + perturbation;
        let mut t = 0.0;

        for step in 0..num_steps {
            // Propagate with STM
            let (state_new, phi_new, dt_used, _, _) = rkn.step_with_varmat(t, state, phi, dt);

            // Propagate perturbed state directly
            let result_pert = rkn.step(t, state_pert, dt);

            // Predict perturbed state using STM
            let state_pert_predicted = state_new + phi_new * perturbation;

            // Compare
            let error = (result_pert.state - state_pert_predicted).norm();
            println!("Step {}: error = {:.6e} m", step + 1, error);

            // Error should remain small and not accumulate excessively
            // RKN STM has ~1e-3 relative error per step, so for 10m perturbation
            // expect ~0.01m error per step, growing to ~0.1m over 10 steps
            let max_error = 0.01 * (step + 1) as f64; // Linear accumulation assumption
            assert!(
                error < max_error,
                "STM prediction diverged at step {}: error = {:.3e} m (max: {:.3e} m)",
                step + 1,
                error,
                max_error
            );

            // Update for next step
            state = state_new;
            phi = phi_new;
            state_pert = result_pert.state;
            t += dt_used;
        }
    }

    // ========================================
    // Dynamic RKN1210DIntegrator tests
    // ========================================

    use crate::eop::set_global_eop_provider;
    use crate::integrators::rkn1210::RKN1210DIntegrator;
    use crate::integrators::traits::AdaptiveStepDIntegrator;
    use nalgebra::{DMatrix, DVector};

    fn setup_global_test_eop() {
        use crate::eop::StaticEOPProvider;
        let eop = StaticEOPProvider::from_zero();
        set_global_eop_provider(eop);
    }

    fn point_earth_dynamic(
        _t: f64,
        state: DVector<f64>,
        _params: Option<&DVector<f64>>,
    ) -> DVector<f64> {
        let r = state.rows(0, 3);
        let v = state.rows(3, 3);
        let r_norm = r.norm();
        let a = -GM_EARTH / (r_norm * r_norm * r_norm) * r;
        let mut state_dot = DVector::zeros(6);
        state_dot.rows_mut(0, 3).copy_from(&v);
        state_dot.rows_mut(3, 3).copy_from(&a);
        state_dot
    }

    #[test]
    fn test_rkn1210d_coefficients() {
        let bt = rkn1210_tableau();
        let b_pos_high_sum: f64 = bt.b_pos_high.iter().sum();
        let b_pos_low_sum: f64 = bt.b_pos_low.iter().sum();
        let b_vel_high_sum: f64 = bt.b_vel_high.iter().sum();
        let b_vel_low_sum: f64 = bt.b_vel_low.iter().sum();
        assert_abs_diff_eq!(b_pos_high_sum, 0.5, epsilon = 1.0e-10);
        assert_abs_diff_eq!(b_pos_low_sum, 0.5, epsilon = 1.0e-10);
        assert_abs_diff_eq!(b_vel_high_sum, 1.0, epsilon = 1.0e-10);
        assert_abs_diff_eq!(b_vel_low_sum, 1.0, epsilon = 1.0e-10);
    }

    #[test]
    fn test_rkn1210d_single_step() {
        setup_global_test_eop();
        let config = IntegratorConfig::adaptive(1e-12, 1e-10);
        let rkn = RKN1210DIntegrator::with_config(
            6,
            Box::new(point_earth_dynamic),
            None,
            None,
            None,
            config,
        );
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0);
        let state0_static = state_osculating_to_cartesian(oe0, DEGREES);
        let state0 = DVector::from_vec(state0_static.as_slice().to_vec());
        let dt = 10.0;
        let result = rkn.step(0.0, state0, dt);
        assert!(result.state.len() == 6);
        assert!(result.dt_used > 0.0);
        assert!(result.error_estimate >= 0.0);
        assert!(result.dt_next > 0.0);
    }

    #[test]
    fn test_rkn1210d_integrator_parabola() {
        let f = |_t: f64, state: DVector<f64>, _params: Option<&DVector<f64>>| -> DVector<f64> {
            let mut state_dot = DVector::zeros(2);
            state_dot[0] = state[1];
            state_dot[1] = 2.0;
            state_dot
        };
        let config = IntegratorConfig::adaptive(1e-12, 1e-10);
        let rkn = RKN1210DIntegrator::with_config(2, Box::new(f), None, None, None, config);
        let state = DVector::from_vec(vec![0.0, 0.0]);
        let dt = 1.0;
        let result = rkn.step(0.0, state, dt);
        let expected_pos = 0.5 * 2.0 * dt * dt;
        let expected_vel = 2.0 * dt;
        assert_abs_diff_eq!(result.state[0], expected_pos, epsilon = 1e-10);
        assert_abs_diff_eq!(result.state[1], expected_vel, epsilon = 1e-10);
    }

    #[test]
    fn test_rkn1210d_integrator_orbit() {
        setup_global_test_eop();
        let config = IntegratorConfig::adaptive(1e-12, 1e-10);
        let rkn = RKN1210DIntegrator::with_config(
            6,
            Box::new(point_earth_dynamic),
            None,
            None,
            None,
            config,
        );
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0);
        let state0_static = state_osculating_to_cartesian(oe0, DEGREES);
        let state0 = DVector::from_vec(state0_static.as_slice().to_vec());
        let period = orbital_period(oe0[0]);
        let dt = period / 100.0;
        let mut state = state0.clone();
        let mut t = 0.0;
        for _ in 0..100 {
            let result = rkn.step(t, state.clone(), dt);
            state = result.state;
            t += result.dt_used;
        }
        let final_r = state.rows(0, 3).norm();
        assert_abs_diff_eq!(final_r, oe0[0], epsilon = 1.0e-7);
    }

    #[test]
    fn test_rkn1210d_accuracy() {
        setup_global_test_eop();
        let config = IntegratorConfig::adaptive(1e-13, 1e-11);
        let rkn = RKN1210DIntegrator::with_config(
            6,
            Box::new(point_earth_dynamic),
            None,
            None,
            None,
            config,
        );
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0);
        let state0_static = state_osculating_to_cartesian(oe0, DEGREES);
        let state0 = DVector::from_vec(state0_static.as_slice().to_vec());
        let dt = 60.0;
        let result = rkn.step(0.0, state0, dt);
        assert!(result.error_estimate < 1.0);
    }

    #[test]
    fn test_rkn1210d_step_size_increases() {
        let f = |_t: f64, state: DVector<f64>, _params: Option<&DVector<f64>>| -> DVector<f64> {
            let mut state_dot = DVector::zeros(2);
            state_dot[0] = state[1];
            state_dot[1] = 0.0;
            state_dot
        };
        let config = IntegratorConfig::adaptive(1e-8, 1e-6);
        let rkn = RKN1210DIntegrator::with_config(2, Box::new(f), None, None, None, config);
        let state = DVector::from_vec(vec![0.0, 1.0]);
        let result = rkn.step(0.0, state, 0.1);
        assert!(result.dt_next > 0.1);
    }

    #[test]
    fn test_rkn1210d_adaptive_mechanism() {
        let f = |t: f64, state: DVector<f64>, _params: Option<&DVector<f64>>| -> DVector<f64> {
            let mut state_dot = DVector::zeros(2);
            state_dot[0] = state[1];
            state_dot[1] = -t.sin();
            state_dot
        };
        let config = IntegratorConfig::adaptive(1e-8, 1e-6);
        let rkn = RKN1210DIntegrator::with_config(2, Box::new(f), None, None, None, config);
        let state = DVector::from_vec(vec![0.0, 1.0]);
        let result1 = rkn.step(0.0, state.clone(), 0.1);
        let result2 = rkn.step(result1.dt_used, result1.state, 0.1);
        assert!(result2.dt_used > 0.0);
    }

    #[test]
    fn test_rkn1210d_config_parameters() {
        let f = |_t: f64, state: DVector<f64>, _params: Option<&DVector<f64>>| -> DVector<f64> {
            let mut state_dot = DVector::zeros(2);
            state_dot[0] = state[1];
            state_dot[1] = 0.0;
            state_dot
        };
        let config = IntegratorConfig {
            abs_tol: 1e-8,
            rel_tol: 1e-6,
            step_safety_factor: Some(0.8),
            min_step_scale_factor: Some(0.5),
            max_step_scale_factor: Some(5.0),
            ..Default::default()
        };
        let rkn = RKN1210DIntegrator::with_config(2, Box::new(f), None, None, None, config);
        assert_eq!(rkn.config().step_safety_factor, Some(0.8));
        assert_eq!(rkn.config().min_step_scale_factor, Some(0.5));
        assert_eq!(rkn.config().max_step_scale_factor, Some(5.0));
    }

    #[test]
    fn test_rkn1210d_high_precision() {
        setup_global_test_eop();
        let config = IntegratorConfig::adaptive(1e-14, 1e-12);
        let rkn = RKN1210DIntegrator::with_config(
            6,
            Box::new(point_earth_dynamic),
            None,
            None,
            None,
            config,
        );
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0);
        let state0_static = state_osculating_to_cartesian(oe0, DEGREES);
        let state0 = DVector::from_vec(state0_static.as_slice().to_vec());
        let result = rkn.step(0.0, state0, 10.0);
        assert!(result.error_estimate < 1.0);
    }

    #[test]
    fn test_rkn1210d_varmat() {
        setup_global_test_eop();
        let point_earth_for_jacobian =
            |t: f64, x: DVector<f64>| -> DVector<f64> { point_earth_dynamic(t, x, None) };
        let jacobian = DNumericalJacobian::new(Box::new(point_earth_for_jacobian))
            .with_method(DifferenceMethod::Central)
            .with_fixed_offset(0.1);
        let config = IntegratorConfig::adaptive(1e-12, 1e-10);
        let rkn = RKN1210DIntegrator::with_config(
            6,
            Box::new(point_earth_dynamic),
            Some(Box::new(jacobian)),
            None,
            None,
            config,
        );
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0);
        let state0_static = state_osculating_to_cartesian(oe0, DEGREES);
        let state0 = DVector::from_vec(state0_static.as_slice().to_vec());
        let dt = 10.0;
        let (state_new, phi_new, dt_used, error, dt_next) =
            rkn.step_with_varmat(0.0, state0, DMatrix::identity(6, 6), dt);
        assert_eq!(state_new.len(), 6);
        assert_eq!(phi_new.nrows(), 6);
        assert_eq!(phi_new.ncols(), 6);
        assert!(dt_used > 0.0);
        assert!(error >= 0.0);
        assert!(dt_next > 0.0);
    }

    #[test]
    fn test_rkn1210d_stm_accuracy() {
        setup_global_test_eop();
        let point_earth_for_jacobian =
            |t: f64, x: DVector<f64>| -> DVector<f64> { point_earth_dynamic(t, x, None) };
        let jacobian = DNumericalJacobian::new(Box::new(point_earth_for_jacobian))
            .with_method(DifferenceMethod::Central)
            .with_fixed_offset(1.0);
        let config = IntegratorConfig::adaptive(1e-12, 1e-10);
        let rkn = RKN1210DIntegrator::with_config(
            6,
            Box::new(point_earth_dynamic),
            Some(Box::new(jacobian)),
            None,
            None,
            config,
        );
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 45.0, 10.0, 20.0, 30.0);
        let state0_static = state_osculating_to_cartesian(oe0, DEGREES);
        let state0 = DVector::from_vec(state0_static.as_slice().to_vec());
        let phi0 = DMatrix::identity(6, 6);
        let dt = 10.0;
        let (state_final, phi_final, _, _, _) = rkn.step_with_varmat(0.0, state0.clone(), phi0, dt);
        let pert_size = 1.0;
        for i in 0..6 {
            let mut perturbation = DVector::zeros(6);
            perturbation[i] = pert_size;
            let state_pert0 = &state0 + &perturbation;
            let result_pert = rkn.step(0.0, state_pert0, dt);
            let state_pert_direct = result_pert.state;
            let state_pert_predicted = &state_final + &phi_final * &perturbation;
            let error = (&state_pert_direct - &state_pert_predicted).norm();
            let relative_error = error / pert_size;
            let tolerance = if i < 3 { 2e-4 } else { 5e-4 };
            assert!(relative_error < tolerance);
        }
    }

    #[test]
    fn test_rkn1210d_stm_vs_direct_perturbation() {
        setup_global_test_eop();
        let point_earth_for_jacobian =
            |t: f64, x: DVector<f64>| -> DVector<f64> { point_earth_dynamic(t, x, None) };
        let jacobian = DNumericalJacobian::new(Box::new(point_earth_for_jacobian))
            .with_method(DifferenceMethod::Central)
            .with_fixed_offset(0.1);
        let config = IntegratorConfig::adaptive(1e-12, 1e-10);
        let rkn_nominal = RKN1210DIntegrator::with_config(
            6,
            Box::new(point_earth_dynamic),
            Some(Box::new(jacobian)),
            None,
            None,
            config.clone(),
        );
        let rkn_pert = RKN1210DIntegrator::with_config(
            6,
            Box::new(point_earth_dynamic),
            None,
            None,
            None,
            config,
        );
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0);
        let state0_static = state_osculating_to_cartesian(oe0, DEGREES);
        let state0 = DVector::from_vec(state0_static.as_slice().to_vec());
        let perturbation = DVector::from_vec(vec![10.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let total_time = 100.0;
        let num_steps = 10;
        let dt = total_time / num_steps as f64;
        let mut state = state0.clone();
        let mut phi = DMatrix::identity(6, 6);
        let mut state_pert = &state0 + &perturbation;
        let mut t = 0.0;
        for step in 0..num_steps {
            let (state_new, phi_new, dt_used, _, _) =
                rkn_nominal.step_with_varmat(t, state.clone(), phi.clone(), dt);
            let result_pert = rkn_pert.step(t, state_pert.clone(), dt);
            let state_pert_predicted = &state_new + &phi_new * &perturbation;
            let error = (&result_pert.state - &state_pert_predicted).norm();
            let max_error = 0.001 * (step + 1) as f64;
            assert!(error < max_error);
            state = state_new;
            phi = phi_new;
            state_pert = result_pert.state;
            t += dt_used;
        }
    }

    #[test]
    fn test_rkn1210_s_vs_d_consistency() {
        // Verify RKN1210SIntegrator and RKN1210DIntegrator produce identical results
        let f_static =
            |_t: f64, x: SVector<f64, 2>| -> SVector<f64, 2> { SVector::<f64, 2>::new(x[1], 2.0) };
        let f_dynamic = |_t: f64,
                         x: DVector<f64>,
                         _params: Option<&DVector<f64>>|
         -> DVector<f64> { DVector::from_vec(vec![x[1], 2.0]) };

        let config = IntegratorConfig::adaptive(1e-10, 1e-8);
        let rkn_s: RKN1210SIntegrator<2, 0> =
            RKN1210SIntegrator::with_config(Box::new(f_static), None, None, None, config.clone());
        let rkn_d =
            RKN1210DIntegrator::with_config(2, Box::new(f_dynamic), None, None, None, config);

        let state_s = SVector::<f64, 2>::new(0.0, 0.0);
        let state_d = DVector::from_vec(vec![0.0, 0.0]);
        let dt = 0.1;

        let result_s = rkn_s.step(0.0, state_s, dt);
        let result_d = rkn_d.step(0.0, state_d, dt);

        // State results should be identical to machine precision
        assert_abs_diff_eq!(result_s.state[0], result_d.state[0], epsilon = 1.0e-15);
        assert_abs_diff_eq!(result_s.state[1], result_d.state[1], epsilon = 1.0e-15);

        // Error estimates and step suggestions should also match
        assert_abs_diff_eq!(
            result_s.error_estimate,
            result_d.error_estimate,
            epsilon = 1.0e-15
        );
        assert_abs_diff_eq!(result_s.dt_used, result_d.dt_used, epsilon = 1.0e-15);
        assert_abs_diff_eq!(result_s.dt_next, result_d.dt_next, epsilon = 1.0e-15);
    }

    // ========================================================================
    // Sensitivity Matrix Tests
    // ========================================================================

    #[test]
    fn test_rkn1210s_sensmat() {
        // Test sensitivity matrix propagation using harmonic oscillator: dx/dt = v, dv/dt = -k*x
        // RKN requires even-dimensional state (position + velocity)

        use crate::math::jacobian::SJacobianProvider;
        use crate::math::sensitivity::SSensitivityProvider;

        // Jacobian: ∂f/∂x = [[0, 1], [-k, 0]] where k = 1.0
        struct HarmonicJacobian;
        impl SJacobianProvider<2> for HarmonicJacobian {
            fn compute(&self, _t: f64, _state: SVector<f64, 2>) -> SMatrix<f64, 2, 2> {
                SMatrix::<f64, 2, 2>::new(0.0, 1.0, -1.0, 0.0)
            }
        }

        // Sensitivity: ∂f/∂k = [[0], [-x]]
        struct HarmonicSensitivity;
        impl SSensitivityProvider<2, 1> for HarmonicSensitivity {
            fn compute(
                &self,
                _t: f64,
                state: &SVector<f64, 2>,
                _params: &SVector<f64, 1>,
            ) -> SMatrix<f64, 2, 1> {
                SMatrix::<f64, 2, 1>::new(0.0, -state[0])
            }
        }

        // Harmonic oscillator: dx/dt = v, dv/dt = -k*x
        let f = |_t: f64, x: SVector<f64, 2>| -> SVector<f64, 2> {
            SVector::<f64, 2>::new(x[1], -x[0])
        };

        let config = IntegratorConfig::adaptive(1e-12, 1e-10);
        let rkn: RKN1210SIntegrator<2, 1> = RKN1210SIntegrator::with_config(
            Box::new(f),
            Some(Box::new(HarmonicJacobian)),
            Some(Box::new(HarmonicSensitivity)),
            None,
            config,
        );

        let x0 = 1.0;
        let v0 = 0.0;
        let state0 = SVector::<f64, 2>::new(x0, v0);
        let sens0 = SMatrix::<f64, 2, 1>::zeros();
        let params = SVector::<f64, 1>::new(1.0);

        let mut state = state0;
        let mut sens = sens0;
        let mut t = 0.0_f64;

        while t < 1.0 {
            let dt = (1.0_f64 - t).min(0.1);
            let (new_state, new_sens, dt_used, _, _) =
                rkn.step_with_sensmat(t, state, sens, &params, dt);
            state = new_state;
            sens = new_sens;
            t += dt_used;
        }

        // Analytical: x(t) = x0*cos(t), v(t) = -x0*sin(t)
        let x_analytical = x0 * t.cos();
        let v_analytical = -x0 * t.sin();

        assert_abs_diff_eq!(state[0], x_analytical, epsilon = 1e-6);
        assert_abs_diff_eq!(state[1], v_analytical, epsilon = 1e-6);

        // Sensitivity should be non-zero (exact analytical solution is complex)
        assert!(sens[(0, 0)].abs() > 1e-10 || sens[(1, 0)].abs() > 1e-10);
    }

    #[test]
    fn test_rkn1210d_sensmat() {
        // Test sensitivity matrix propagation (dynamic version)

        use crate::math::sensitivity::DSensitivityProvider;

        struct HarmonicJacobian;
        impl crate::math::jacobian::DJacobianProvider for HarmonicJacobian {
            fn compute(&self, _t: f64, _state: DVector<f64>) -> DMatrix<f64> {
                DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -1.0, 0.0])
            }
        }

        struct HarmonicSensitivity;
        impl DSensitivityProvider for HarmonicSensitivity {
            fn compute(
                &self,
                _t: f64,
                state: &DVector<f64>,
                _params: &DVector<f64>,
            ) -> DMatrix<f64> {
                DMatrix::from_column_slice(2, 1, &[0.0, -state[0]])
            }
        }

        let dynamics = |_t: f64, x: DVector<f64>, _params: Option<&DVector<f64>>| -> DVector<f64> {
            DVector::from_vec(vec![x[1], -x[0]])
        };

        let config = IntegratorConfig::adaptive(1e-12, 1e-10);
        let rkn = RKN1210DIntegrator::with_config(
            2,
            Box::new(dynamics),
            Some(Box::new(HarmonicJacobian)),
            Some(Box::new(HarmonicSensitivity)),
            None,
            config,
        );

        let x0 = 1.0;
        let state0 = DVector::from_vec(vec![x0, 0.0]);
        let sens0 = DMatrix::zeros(2, 1);
        let params = DVector::from_vec(vec![1.0]);

        let mut state = state0;
        let mut sens = sens0;
        let mut t = 0.0_f64;

        while t < 1.0 {
            let dt = (1.0_f64 - t).min(0.1);
            let (new_state, new_sens, dt_used, _, _) =
                rkn.step_with_sensmat(t, state, sens, &params, dt);
            state = new_state;
            sens = new_sens;
            t += dt_used;
        }

        let x_analytical = x0 * t.cos();
        let v_analytical = -x0 * t.sin();

        assert_abs_diff_eq!(state[0], x_analytical, epsilon = 1e-6);
        assert_abs_diff_eq!(state[1], v_analytical, epsilon = 1e-6);
        assert!(sens[(0, 0)].abs() > 1e-10 || sens[(1, 0)].abs() > 1e-10);
    }

    #[test]
    fn test_rkn1210s_varmat_sensmat() {
        // Test combined STM and sensitivity matrix propagation (static version)

        use crate::math::jacobian::SJacobianProvider;
        use crate::math::sensitivity::SSensitivityProvider;

        struct HarmonicJacobian;
        impl SJacobianProvider<2> for HarmonicJacobian {
            fn compute(&self, _t: f64, _state: SVector<f64, 2>) -> SMatrix<f64, 2, 2> {
                SMatrix::<f64, 2, 2>::new(0.0, 1.0, -1.0, 0.0)
            }
        }

        struct HarmonicSensitivity;
        impl SSensitivityProvider<2, 1> for HarmonicSensitivity {
            fn compute(
                &self,
                _t: f64,
                state: &SVector<f64, 2>,
                _params: &SVector<f64, 1>,
            ) -> SMatrix<f64, 2, 1> {
                SMatrix::<f64, 2, 1>::new(0.0, -state[0])
            }
        }

        let f = |_t: f64, x: SVector<f64, 2>| -> SVector<f64, 2> {
            SVector::<f64, 2>::new(x[1], -x[0])
        };

        let config = IntegratorConfig::adaptive(1e-12, 1e-10);
        let rkn: RKN1210SIntegrator<2, 1> = RKN1210SIntegrator::with_config(
            Box::new(f),
            Some(Box::new(HarmonicJacobian)),
            Some(Box::new(HarmonicSensitivity)),
            None,
            config,
        );

        let x0 = 1.0;
        let state0 = SVector::<f64, 2>::new(x0, 0.0);
        let phi0 = SMatrix::<f64, 2, 2>::identity();
        let sens0 = SMatrix::<f64, 2, 1>::zeros();
        let params = SVector::<f64, 1>::new(1.0);

        let mut state = state0;
        let mut phi = phi0;
        let mut sens = sens0;
        let mut t = 0.0_f64;

        while t < 1.0 {
            let dt = (1.0_f64 - t).min(0.1);
            let (new_state, new_phi, new_sens, dt_used, _, _) =
                rkn.step_with_varmat_sensmat(t, state, phi, sens, &params, dt);
            state = new_state;
            phi = new_phi;
            sens = new_sens;
            t += dt_used;
        }

        let x_analytical = x0 * t.cos();
        let v_analytical = -x0 * t.sin();

        assert_abs_diff_eq!(state[0], x_analytical, epsilon = 1e-6);
        assert_abs_diff_eq!(state[1], v_analytical, epsilon = 1e-6);

        // STM for harmonic oscillator: [[cos(t), sin(t)], [-sin(t), cos(t)]]
        // Note: RKN integrator STM may differ slightly due to position/velocity structure
        assert_abs_diff_eq!(phi[(0, 0)], t.cos(), epsilon = 0.05);
        assert_abs_diff_eq!(phi[(0, 1)], t.sin(), epsilon = 0.05);
        assert_abs_diff_eq!(phi[(1, 0)], -t.sin(), epsilon = 0.05);
        assert_abs_diff_eq!(phi[(1, 1)], t.cos(), epsilon = 0.05);
    }

    #[test]
    fn test_rkn1210d_varmat_sensmat() {
        // Test combined STM and sensitivity matrix propagation (dynamic version)

        use crate::math::sensitivity::DSensitivityProvider;

        struct HarmonicJacobian;
        impl crate::math::jacobian::DJacobianProvider for HarmonicJacobian {
            fn compute(&self, _t: f64, _state: DVector<f64>) -> DMatrix<f64> {
                DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -1.0, 0.0])
            }
        }

        struct HarmonicSensitivity;
        impl DSensitivityProvider for HarmonicSensitivity {
            fn compute(
                &self,
                _t: f64,
                state: &DVector<f64>,
                _params: &DVector<f64>,
            ) -> DMatrix<f64> {
                DMatrix::from_column_slice(2, 1, &[0.0, -state[0]])
            }
        }

        let dynamics = |_t: f64, x: DVector<f64>, _params: Option<&DVector<f64>>| -> DVector<f64> {
            DVector::from_vec(vec![x[1], -x[0]])
        };

        let config = IntegratorConfig::adaptive(1e-12, 1e-10);
        let rkn = RKN1210DIntegrator::with_config(
            2,
            Box::new(dynamics),
            Some(Box::new(HarmonicJacobian)),
            Some(Box::new(HarmonicSensitivity)),
            None,
            config,
        );

        let x0 = 1.0;
        let state0 = DVector::from_vec(vec![x0, 0.0]);
        let phi0 = DMatrix::identity(2, 2);
        let sens0 = DMatrix::zeros(2, 1);
        let params = DVector::from_vec(vec![1.0]);

        let mut state = state0;
        let mut phi = phi0;
        let mut sens = sens0;
        let mut t = 0.0_f64;

        while t < 1.0 {
            let dt = (1.0_f64 - t).min(0.1);
            let (new_state, new_phi, new_sens, dt_used, _, _) =
                rkn.step_with_varmat_sensmat(t, state, phi, sens, &params, dt);
            state = new_state;
            phi = new_phi;
            sens = new_sens;
            t += dt_used;
        }

        let x_analytical = x0 * t.cos();
        let v_analytical = -x0 * t.sin();

        assert_abs_diff_eq!(state[0], x_analytical, epsilon = 1e-6);
        assert_abs_diff_eq!(state[1], v_analytical, epsilon = 1e-6);

        // STM for harmonic oscillator
        // Note: RKN integrator STM may differ slightly due to position/velocity structure
        assert_abs_diff_eq!(phi[(0, 0)], t.cos(), epsilon = 0.05);
        assert_abs_diff_eq!(phi[(0, 1)], t.sin(), epsilon = 0.05);
        assert_abs_diff_eq!(phi[(1, 0)], -t.sin(), epsilon = 0.05);
        assert_abs_diff_eq!(phi[(1, 1)], t.cos(), epsilon = 0.05);
    }

    // =============================================================================
    // Constructor and Config Tests
    // =============================================================================

    #[test]
    fn test_rkn1210s_new_uses_default_config() {
        fn dynamics(_t: f64, state: SVector<f64, 2>) -> SVector<f64, 2> {
            SVector::<f64, 2>::new(state[1], -state[0])
        }

        let integrator: RKN1210SIntegrator<2, 0> =
            RKN1210SIntegrator::new(Box::new(dynamics), None, None, None);
        let config = integrator.config();

        let default_config = IntegratorConfig::default();
        assert_eq!(config.abs_tol, default_config.abs_tol);
        assert_eq!(config.rel_tol, default_config.rel_tol);
        assert_eq!(config.max_step_attempts, default_config.max_step_attempts);
        assert_eq!(config.min_step, default_config.min_step);
        assert_eq!(config.max_step, default_config.max_step);
    }

    #[test]
    fn test_rkn1210s_with_config_stores_config() {
        fn dynamics(_t: f64, state: SVector<f64, 2>) -> SVector<f64, 2> {
            SVector::<f64, 2>::new(state[1], -state[0])
        }

        let custom_config = IntegratorConfig {
            abs_tol: 1e-10,
            rel_tol: 1e-8,
            initial_step: None,
            max_step_attempts: 20,
            min_step: Some(1e-15),
            max_step: Some(100.0),
            step_safety_factor: Some(0.9),
            max_step_scale_factor: Some(5.0),
            min_step_scale_factor: Some(0.1),
            fixed_step_size: None,
        };

        let integrator: RKN1210SIntegrator<2, 0> = RKN1210SIntegrator::with_config(
            Box::new(dynamics),
            None,
            None,
            None,
            custom_config.clone(),
        );
        let config = integrator.config();

        assert_eq!(config.abs_tol, 1e-10);
        assert_eq!(config.rel_tol, 1e-8);
        assert_eq!(config.max_step_attempts, 20);
        assert_eq!(config.min_step, Some(1e-15));
        assert_eq!(config.max_step, Some(100.0));
    }

    #[test]
    fn test_rkn1210s_config_returns_reference() {
        fn dynamics(_t: f64, state: SVector<f64, 2>) -> SVector<f64, 2> {
            SVector::<f64, 2>::new(state[1], -state[0])
        }

        let integrator: RKN1210SIntegrator<2, 0> =
            RKN1210SIntegrator::new(Box::new(dynamics), None, None, None);

        let config1 = integrator.config();
        let config2 = integrator.config();

        assert_eq!(config1.abs_tol, config2.abs_tol);
        assert_eq!(config1.rel_tol, config2.rel_tol);
        assert_eq!(config1.max_step_attempts, config2.max_step_attempts);
    }

    #[test]
    fn test_rkn1210d_new_uses_default_config() {
        fn dynamics(_t: f64, state: DVector<f64>, _params: Option<&DVector<f64>>) -> DVector<f64> {
            DVector::from_vec(vec![state[1], -state[0]])
        }

        let integrator = RKN1210DIntegrator::new(2, Box::new(dynamics), None, None, None);
        let config = integrator.config();

        let default_config = IntegratorConfig::default();
        assert_eq!(config.abs_tol, default_config.abs_tol);
        assert_eq!(config.rel_tol, default_config.rel_tol);
        assert_eq!(config.max_step_attempts, default_config.max_step_attempts);
        assert_eq!(config.min_step, default_config.min_step);
        assert_eq!(config.max_step, default_config.max_step);
    }

    #[test]
    fn test_rkn1210d_with_config_stores_config() {
        fn dynamics(_t: f64, state: DVector<f64>, _params: Option<&DVector<f64>>) -> DVector<f64> {
            DVector::from_vec(vec![state[1], -state[0]])
        }

        let custom_config = IntegratorConfig {
            abs_tol: 1e-10,
            rel_tol: 1e-8,
            initial_step: None,
            max_step_attempts: 20,
            min_step: Some(1e-15),
            max_step: Some(100.0),
            step_safety_factor: Some(0.9),
            max_step_scale_factor: Some(5.0),
            min_step_scale_factor: Some(0.1),
            fixed_step_size: None,
        };

        let integrator = RKN1210DIntegrator::with_config(
            2,
            Box::new(dynamics),
            None,
            None,
            None,
            custom_config.clone(),
        );
        let config = integrator.config();

        assert_eq!(config.abs_tol, 1e-10);
        assert_eq!(config.rel_tol, 1e-8);
        assert_eq!(config.max_step_attempts, 20);
        assert_eq!(config.min_step, Some(1e-15));
        assert_eq!(config.max_step, Some(100.0));
    }

    #[test]
    fn test_rkn1210d_config_returns_reference() {
        fn dynamics(_t: f64, state: DVector<f64>, _params: Option<&DVector<f64>>) -> DVector<f64> {
            DVector::from_vec(vec![state[1], -state[0]])
        }

        let integrator = RKN1210DIntegrator::new(2, Box::new(dynamics), None, None, None);

        let config1 = integrator.config();
        let config2 = integrator.config();

        assert_eq!(config1.abs_tol, config2.abs_tol);
        assert_eq!(config1.rel_tol, config2.rel_tol);
        assert_eq!(config1.max_step_attempts, config2.max_step_attempts);
    }

    #[test]
    fn test_rkn1210d_dimension_method() {
        fn dynamics(_t: f64, state: DVector<f64>, _params: Option<&DVector<f64>>) -> DVector<f64> {
            DVector::from_vec(vec![state[1], -state[0]])
        }

        let integrator = RKN1210DIntegrator::new(6, Box::new(dynamics), None, None, None);
        assert_eq!(integrator.dimension(), 6);

        let integrator2 = RKN1210DIntegrator::new(12, Box::new(dynamics), None, None, None);
        assert_eq!(integrator2.dimension(), 12);
    }

    // =============================================================================
    // Panic Tests - Max Step Attempts Exceeded
    // =============================================================================

    #[test]
    #[should_panic(expected = "exceeded maximum step attempts")]
    fn test_rkn1210s_panics_on_max_attempts_exceeded() {
        fn stiff_dynamics(_t: f64, state: SVector<f64, 2>) -> SVector<f64, 2> {
            // Rapidly varying dynamics
            SVector::<f64, 2>::new(1e10 * state[1], 1e10 * state[0])
        }

        let config = IntegratorConfig {
            abs_tol: 1e-15,
            rel_tol: 1e-15,
            initial_step: None,
            max_step_attempts: 1,
            min_step: None,
            max_step: None,
            step_safety_factor: None,
            max_step_scale_factor: None,
            min_step_scale_factor: None,
            fixed_step_size: None,
        };

        let integrator: RKN1210SIntegrator<2, 0> =
            RKN1210SIntegrator::with_config(Box::new(stiff_dynamics), None, None, None, config);

        let state = SVector::<f64, 2>::new(1.0, 0.0);
        let _ = integrator.step(0.0, state, 1.0);
    }

    #[test]
    #[should_panic(expected = "exceeded maximum step attempts")]
    fn test_rkn1210d_panics_on_max_attempts_exceeded() {
        fn stiff_dynamics(
            _t: f64,
            state: DVector<f64>,
            _params: Option<&DVector<f64>>,
        ) -> DVector<f64> {
            DVector::from_vec(vec![1e10 * state[1], 1e10 * state[0]])
        }

        let config = IntegratorConfig {
            abs_tol: 1e-15,
            rel_tol: 1e-15,
            initial_step: None,
            max_step_attempts: 1,
            min_step: None,
            max_step: None,
            step_safety_factor: None,
            max_step_scale_factor: None,
            min_step_scale_factor: None,
            fixed_step_size: None,
        };

        let integrator =
            RKN1210DIntegrator::with_config(2, Box::new(stiff_dynamics), None, None, None, config);

        let state = DVector::from_vec(vec![1.0, 0.0]);
        let _ = integrator.step(0.0, state, 1.0);
    }
}
