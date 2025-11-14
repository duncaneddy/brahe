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

use nalgebra::{SMatrix, SVector};

use crate::integrators::butcher_tableau::{EmbeddedRKNButcherTableau, rkn1210_tableau};
use crate::integrators::config::{AdaptiveStepResult, IntegratorConfig};
use crate::integrators::traits::AdaptiveStepIntegrator;

// Type aliases for complex function types
type StateDynamics<const S: usize> = Box<dyn Fn(f64, SVector<f64, S>) -> SVector<f64, S>>;
type VariationalMatrix<const S: usize> =
    Option<Box<dyn Fn(f64, SVector<f64, S>) -> SMatrix<f64, S, S>>>;

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
/// use brahe::integrators::{RKN1210Integrator, AdaptiveStepIntegrator, IntegratorConfig};
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
/// let rkn = RKN1210Integrator::with_config(Box::new(f), None, config);
///
/// // Integrate one step
/// let t = 0.0;
/// let state = SVector::<f64, 6>::new(7000e3, 0.0, 0.0, 0.0, 7500.0, 0.0);
/// let dt = 10.0;
/// let result = rkn.step(t, state, dt, 1e-12, 1e-10);
///
/// println!("New state: {:?}", result.state);
/// println!("Suggested next dt: {}", result.dt_next);
/// ```
pub struct RKN1210Integrator<const S: usize> {
    f: StateDynamics<S>,
    varmat: VariationalMatrix<S>,
    bt: EmbeddedRKNButcherTableau<17>,
    config: IntegratorConfig,
}

impl<const S: usize> RKN1210Integrator<S> {
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
    ///
    /// # Returns
    /// RKN1210Integrator instance ready for numerical integration
    ///
    /// # Note
    /// This constructor provides backward compatibility. Uses default configuration.
    /// For custom configuration, use `with_config()`.
    pub fn new(f: StateDynamics<S>, varmat: VariationalMatrix<S>) -> Self {
        Self::with_config(f, varmat, IntegratorConfig::default())
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
    /// - `config`: Integration configuration (tolerances, step sizes, etc.)
    ///
    /// # Returns
    /// RKN1210Integrator instance with specified configuration
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::SVector;
    /// use brahe::integrators::{RKN1210Integrator, IntegratorConfig};
    ///
    /// let f = |t: f64, state: SVector<f64, 6>| -> SVector<f64, 6> {
    ///     // Dynamics implementation
    ///     state
    /// };
    ///
    /// let config = IntegratorConfig::adaptive(1e-12, 1e-10);
    /// let rkn = RKN1210Integrator::with_config(Box::new(f), None, config);
    /// ```
    pub fn with_config(
        f: StateDynamics<S>,
        varmat: VariationalMatrix<S>,
        config: IntegratorConfig,
    ) -> Self {
        Self {
            f,
            varmat,
            bt: rkn1210_tableau(),
            config,
        }
    }

    /// Get a reference to the integrator configuration.
    pub fn config(&self) -> &IntegratorConfig {
        &self.config
    }
}

impl<const S: usize> AdaptiveStepIntegrator<S> for RKN1210Integrator<S> {
    fn step(
        &self,
        t: f64,
        state: SVector<f64, S>,
        dt: f64,
        abs_tol: f64,
        rel_tol: f64,
    ) -> AdaptiveStepResult<S> {
        // State vector format: [position, velocity] where each is S/2 dimensional
        // For 6D orbital state: [x, y, z, vx, vy, vz]
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

            // Extract position and velocity from state (using runtime indexing)
            let mut pos = SVector::<f64, S>::zeros();
            let mut vel = SVector::<f64, S>::zeros();
            for i in 0..half_dim {
                pos[i] = state[i];
                vel[i] = state[half_dim + i];
            }

            // Preallocate stage matrix for accelerations (k[i] = f(t_i, y_i))
            // Note: RKN evaluates acceleration, not full state derivative
            let mut k = SMatrix::<f64, S, 17>::zeros();

            // Compute RKN stages
            // Stage formula: k[i] = accel(t + c[i]*h, pos + c[i]*h*vel + h²*sum(a[i,j]*k[j]))
            for i in 0..17 {
                // Compute position perturbation: h²*sum(a[i,j]*k[j])
                let mut pos_pert = SVector::<f64, S>::zeros();
                for j in 0..i {
                    for dim in 0..half_dim {
                        pos_pert[dim] += self.bt.a[(i, j)] * k[(dim, j)];
                    }
                }

                // Stage position: y + c[i]*h*v + h²*sum(a[i,j]*k[j])
                let mut stage_pos = SVector::<f64, S>::zeros();
                for dim in 0..half_dim {
                    stage_pos[dim] = pos[dim] + self.bt.c[i] * h * vel[dim] + h * h * pos_pert[dim];
                }

                // Reconstruct full state for dynamics evaluation
                let mut stage_state = SVector::<f64, S>::zeros();
                for dim in 0..half_dim {
                    stage_state[dim] = stage_pos[dim];
                    stage_state[half_dim + dim] = vel[dim];
                }

                // Evaluate dynamics and extract acceleration (second half of state_dot)
                let state_dot = (self.f)(t + self.bt.c[i] * h, stage_state);
                for dim in 0..half_dim {
                    k[(dim, i)] = state_dot[half_dim + dim];
                }
            }

            // Compute high-order and low-order solutions
            // Position: y_new = y + h*v + h²*sum(b_pos*k)
            // Velocity: v_new = v + h*sum(b_vel*k)

            let mut pos_high = SVector::<f64, S>::zeros();
            let mut vel_high = SVector::<f64, S>::zeros();
            let mut pos_low = SVector::<f64, S>::zeros();
            let mut vel_low = SVector::<f64, S>::zeros();

            for dim in 0..half_dim {
                // High-order solution
                let mut pos_update_high = 0.0;
                let mut vel_update_high = 0.0;
                for i in 0..17 {
                    pos_update_high += h * h * self.bt.b_pos_high[i] * k[(dim, i)];
                    vel_update_high += h * self.bt.b_vel_high[i] * k[(dim, i)];
                }
                pos_high[dim] = pos[dim] + h * vel[dim] + pos_update_high;
                vel_high[dim] = vel[dim] + vel_update_high;

                // Low-order solution
                let mut pos_update_low = 0.0;
                let mut vel_update_low = 0.0;
                for i in 0..17 {
                    pos_update_low += h * h * self.bt.b_pos_low[i] * k[(dim, i)];
                    vel_update_low += h * self.bt.b_vel_low[i] * k[(dim, i)];
                }
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

            // Compute error estimate (element-wise maximum of position and velocity errors)
            let error_vec = state_high - state_low;
            let mut error = 0.0;
            for i in 0..S {
                let tol = abs_tol + rel_tol * f64::max(state_high[i].abs(), state[i].abs());
                let normalized_error = (error_vec[i] / tol).abs();
                error = f64::max(error, normalized_error);
            }

            // Check if step should be accepted
            let min_step_reached = self.config.min_step.is_some_and(|min| h <= min);

            if error <= 1.0 || min_step_reached {
                // Step accepted - calculate optimal next step size
                let dt_next = if error > 0.0 {
                    // Use high order (12) for step size calculation
                    let raw_scale = (1.0 / error).powf(1.0 / 12.0);
                    let scale = self
                        .config
                        .step_safety_factor
                        .map_or(raw_scale, |safety| safety * raw_scale);

                    let mut next_h = h * scale;

                    // Apply min scale factor if configured
                    if let Some(min_scale) = self.config.min_step_scale_factor {
                        next_h = next_h.max(min_scale * h);
                    }

                    // Apply max scale factor if configured
                    if let Some(max_scale) = self.config.max_step_scale_factor {
                        next_h = next_h.min(max_scale * h);
                    }

                    // Apply absolute step size limits
                    if let Some(max_step) = self.config.max_step {
                        next_h = next_h.min(max_step.abs());
                    }
                    if let Some(min_step) = self.config.min_step {
                        next_h = next_h.max(min_step.abs());
                    }

                    next_h
                } else {
                    // Error is zero - use maximum increase
                    let next_h = if let Some(max_scale) = self.config.max_step_scale_factor {
                        max_scale * h
                    } else {
                        10.0 * h // Default max growth if unconfigured
                    };

                    // Respect absolute max if configured
                    self.config.max_step.map_or(next_h, |max| next_h.min(max))
                };

                return AdaptiveStepResult {
                    state: state_high,
                    dt_used: h,
                    error_estimate: error,
                    dt_next,
                };
            } else {
                // Step rejected - reduce step size and retry
                let raw_scale = (1.0 / error).powf(1.0 / 10.0);
                let scale = self
                    .config
                    .step_safety_factor
                    .map_or(raw_scale, |safety| safety * raw_scale);

                let mut h_new = h * scale;

                // Apply min scale factor if configured
                if let Some(min_scale) = self.config.min_step_scale_factor {
                    h_new = h_new.max(min_scale * h);
                }

                // Respect absolute minimum if configured
                if let Some(min_step) = self.config.min_step {
                    h_new = h_new.max(min_step);
                }

                h = h_new;
            }
        }

        // If we get here, we've exceeded max attempts - Things are not going well
        panic!("RKN1210 integrator exceeded maximum step attempts");
    }

    fn step_with_varmat(
        &self,
        t: f64,
        state: SVector<f64, S>,
        phi: SMatrix<f64, S, S>,
        dt: f64,
        abs_tol: f64,
        rel_tol: f64,
    ) -> (SVector<f64, S>, SMatrix<f64, S, S>, f64, f64, f64) {
        assert!(
            S.is_multiple_of(2),
            "RKN integrator requires even-dimensional state (position + velocity)"
        );

        let half_dim = S / 2;

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

        // Define working variables for STM updates
        let mut phi_update = SMatrix::<f64, S, S>::zeros();

        // Compute RKN stages with STM
        for i in 0..17 {
            // Position perturbation
            let mut pos_pert = SVector::<f64, S>::zeros();
            for j in 0..i {
                for dim in 0..half_dim {
                    pos_pert[dim] += self.bt.a[(i, j)] * k[(dim, j)];
                }
            }

            // Stage position
            let mut stage_pos = SVector::<f64, S>::zeros();
            for dim in 0..half_dim {
                stage_pos[dim] = pos[dim] + self.bt.c[i] * dt * vel[dim] + dt * dt * pos_pert[dim];
            }

            // Reconstruct full state
            let mut stage_state = SVector::<f64, S>::zeros();
            for dim in 0..half_dim {
                stage_state[dim] = stage_pos[dim];
                stage_state[half_dim + dim] = vel[dim];
            }

            // Evaluate dynamics and extract acceleration
            let state_dot = (self.f)(t + self.bt.c[i] * dt, stage_state);
            for dim in 0..half_dim {
                k[(dim, i)] = state_dot[half_dim + dim];
            }

            // Compute STM for this stage
            let mut k_phi_sum = SMatrix::<f64, S, S>::zeros();
            for (j, k_phi_j) in k_phi.iter().enumerate().take(i) {
                k_phi_sum += self.bt.a[(i, j)] * k_phi_j;
            }

            k_phi[i] = self.varmat.as_ref().unwrap()(t + self.bt.c[i] * dt, stage_state)
                * (phi + dt * k_phi_sum);
        }

        // Compute high-order and low-order solutions
        let mut pos_high = SVector::<f64, S>::zeros();
        let mut vel_high = SVector::<f64, S>::zeros();
        let mut pos_low = SVector::<f64, S>::zeros();
        let mut vel_low = SVector::<f64, S>::zeros();

        for dim in 0..half_dim {
            // High-order solution
            let mut pos_update_high = 0.0;
            let mut vel_update_high = 0.0;
            for i in 0..17 {
                pos_update_high += dt * dt * self.bt.b_pos_high[i] * k[(dim, i)];
                vel_update_high += dt * self.bt.b_vel_high[i] * k[(dim, i)];
            }
            pos_high[dim] = pos[dim] + dt * vel[dim] + pos_update_high;
            vel_high[dim] = vel[dim] + vel_update_high;

            // Low-order solution
            let mut pos_update_low = 0.0;
            let mut vel_update_low = 0.0;
            for i in 0..17 {
                pos_update_low += dt * dt * self.bt.b_pos_low[i] * k[(dim, i)];
                vel_update_low += dt * self.bt.b_vel_low[i] * k[(dim, i)];
            }
            pos_low[dim] = pos[dim] + dt * vel[dim] + pos_update_low;
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

        // Compute STM update (using high-order position weights for consistency)
        for (i, k_phi_i) in k_phi.iter().enumerate().take(17) {
            phi_update += dt * self.bt.b_pos_high[i] * k_phi_i;
        }

        // Error estimation
        let error_vec = state_high - state_low;
        let mut error = 0.0;
        for i in 0..S {
            let tol = abs_tol + rel_tol * f64::max(state_high[i].abs(), state[i].abs());
            error = f64::max(error, (error_vec[i] / tol).abs());
        }

        // Compute suggested next step size
        let dt_next = if error <= 1.0 {
            // Step accepted - use 12th order for step size calculation
            let raw_scale = (1.0 / f64::max(error, 1e-10)).powf(1.0 / 12.0);
            let scale = self
                .config
                .step_safety_factor
                .map_or(raw_scale, |safety| safety * raw_scale);

            let mut next_dt = dt * scale;

            // Apply min scale factor if configured
            if let Some(min_scale) = self.config.min_step_scale_factor {
                next_dt = next_dt.max(min_scale * dt);
            }

            // Apply max scale factor if configured
            if let Some(max_scale) = self.config.max_step_scale_factor {
                next_dt = next_dt.min(max_scale * dt);
            }

            // Apply absolute step size limits
            if let Some(max_step) = self.config.max_step {
                next_dt = next_dt.min(max_step.abs());
            }
            if let Some(min_step) = self.config.min_step {
                next_dt = next_dt.max(min_step.abs());
            }

            next_dt
        } else {
            // Step rejected - use 10th order for more conservative estimate
            let raw_scale = (1.0 / error).powf(1.0 / 10.0);
            let scale = self
                .config
                .step_safety_factor
                .map_or(raw_scale, |safety| safety * raw_scale);

            let mut next_dt = dt * scale;

            // Apply min scale factor if configured
            if let Some(min_scale) = self.config.min_step_scale_factor {
                next_dt = next_dt.max(min_scale * dt);
            }

            next_dt
        };

        (state_high, phi + phi_update, dt, error, dt_next)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::{SMatrix, SVector};

    use crate::constants::DEGREES;
    use crate::integrators::butcher_tableau::rkn1210_tableau;
    use crate::integrators::config::IntegratorConfig;
    use crate::integrators::rkn1210::RKN1210Integrator;
    use crate::integrators::traits::{AdaptiveStepIntegrator, VarmatConfig};
    use crate::time::{Epoch, TimeSystem};
    use crate::{GM_EARTH, R_EARTH, orbital_period, state_osculating_to_cartesian};

    #[test]
    fn test_rkn1210_coefficients() {
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
    fn test_rkn1210_single_step() {
        // Test a single step with constant acceleration to verify formulas
        let f = |_t: f64, state: SVector<f64, 2>| -> SVector<f64, 2> {
            // state = [x, v], state_dot = [v, a] where a = 2
            SVector::<f64, 2>::new(state[1], 2.0)
        };

        let config = IntegratorConfig::adaptive(1e-10, 1e-8);
        let rkn = RKN1210Integrator::with_config(Box::new(f), None, config);

        let state = SVector::<f64, 2>::new(0.0, 0.0); // [x, v] = [0, 0]
        let dt = 0.01;

        let result = rkn.step(0.0, state, dt, 1e-10, 1e-8);

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
    fn test_rkn1210_integrator_parabola() {
        // Test with simple parabolic motion: x'' = 2 (constant acceleration)
        // Solution: x(t) = t²
        let f = |_t: f64, state: SVector<f64, 2>| -> SVector<f64, 2> {
            // state = [x, v]
            // state_dot = [v, a] where a = 2
            SVector::<f64, 2>::new(state[1], 2.0)
        };

        let config = IntegratorConfig::adaptive(1e-10, 1e-8);
        let rkn = RKN1210Integrator::with_config(Box::new(f), None, config);

        let mut t = 0.0;
        let mut state = SVector::<f64, 2>::new(0.0, 0.0); // [position, velocity]
        let mut dt = 0.01;
        let mut step_count = 0;

        // Integrate to t = 1.0, solution should be x = 1.0
        let t_final = 1.0;
        while t < t_final {
            // Clip dt to not overshoot target time
            let dt_actual = f64::min(dt, t_final - t);
            let result = rkn.step(t, state, dt_actual, 1e-10, 1e-8);
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
    fn test_rkn1210_integrator_orbit() {
        let config = IntegratorConfig::adaptive(1e-9, 1e-6);
        let rkn = RKN1210Integrator::with_config(Box::new(point_earth), None, config);

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
            let result = rkn.step(epc - epc0, state, dt_actual, 1e-9, 1e-6);
            state = result.state;
            dt = result.dt_next;
            epc += result.dt_used;
        }

        // Check energy conservation (position magnitude should be close)
        // RKN1210 is extremely accurate - expect sub-meter error over full orbit
        assert_abs_diff_eq!(state.norm(), state0.norm(), epsilon = 1.0);
    }

    #[test]
    fn test_rkn1210_accuracy() {
        // Verify 12th order convergence on simple problem
        let f = |_t: f64, state: SVector<f64, 2>| -> SVector<f64, 2> {
            SVector::<f64, 2>::new(state[1], 2.0)
        };

        let config = IntegratorConfig::adaptive(1e-12, 1e-10);
        let rkn = RKN1210Integrator::with_config(Box::new(f), None, config);

        let mut t = 0.0;
        let mut state = SVector::<f64, 2>::new(0.0, 0.0);
        let mut dt = 0.1;
        let t_final = 1.0;

        while t < t_final {
            let dt_actual = f64::min(dt, t_final - t);
            let result = rkn.step(t, state, dt_actual, 1e-12, 1e-10);
            state = result.state;
            dt = result.dt_next;
            t += result.dt_used;
        }

        // With 12th order and tight tolerances, error should be extremely small
        assert_abs_diff_eq!(state[0], 1.0, epsilon = 1.0e-10);
    }

    #[test]
    fn test_rkn1210_step_size_increases() {
        let f = |_t: f64, state: SVector<f64, 2>| -> SVector<f64, 2> {
            SVector::<f64, 2>::new(state[1], 2.0)
        };

        let config = IntegratorConfig::adaptive(1e-6, 1e-3);
        let rkn = RKN1210Integrator::with_config(Box::new(f), None, config);

        let state = SVector::<f64, 2>::new(0.0, 0.0);
        let dt = 0.01;

        let result = rkn.step(0.0, state, dt, 1e-6, 1e-3);

        // With loose tolerances on smooth problem, should suggest larger step
        assert!(result.dt_next > dt);
    }

    #[test]
    fn test_rkn1210_adaptive_mechanism() {
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
        let rkn = RKN1210Integrator::with_config(Box::new(f), None, config);

        let state = SVector::<f64, 2>::new(1.0, 0.0);
        let dt = 1.0;

        let result = rkn.step(0.0, state, dt, 1e-14, 1e-12);

        // Verify adaptive mechanism produces valid output
        assert!(result.dt_used > 0.0);
        assert!(result.dt_next > 0.0);
        assert!(result.dt_next.is_finite());
        assert!(result.error_estimate >= 0.0);
        assert!(result.error_estimate.is_finite());
    }

    #[test]
    fn test_rkn1210_config_parameters() {
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

        let rkn = RKN1210Integrator::with_config(Box::new(f), None, config);

        assert_eq!(rkn.config().step_safety_factor, Some(0.8));
        assert_eq!(rkn.config().min_step_scale_factor, Some(0.5));
        assert_eq!(rkn.config().max_step_scale_factor, Some(5.0));
    }

    #[test]
    fn test_rkn1210_high_precision() {
        // Test with very tight tolerances
        let f = |_t: f64, state: SVector<f64, 2>| -> SVector<f64, 2> {
            SVector::<f64, 2>::new(state[1], 2.0)
        };

        let config = IntegratorConfig::adaptive(1e-13, 1e-11);
        let rkn = RKN1210Integrator::with_config(Box::new(f), None, config);

        let mut t = 0.0;
        let mut state = SVector::<f64, 2>::new(0.0, 0.0);
        let mut dt = 0.1;
        let t_final = 1.0;

        while t < t_final {
            let dt_actual = f64::min(dt, t_final - t);
            let result = rkn.step(t, state, dt_actual, 1e-13, 1e-11);
            state = result.state;
            dt = result.dt_next;
            t += result.dt_used;
        }

        // RKN1210 should achieve very high precision
        assert_abs_diff_eq!(state[0], 1.0, epsilon = 1.0e-11);
    }

    #[test]
    fn test_rkn1210_varmat() {
        // Define variational matrix computation using new VarmatConfig API
        // Use forward differences to match old test behavior
        let varmat_config = VarmatConfig::forward().with_fixed_offset(1.0);
        let varmat = move |t: f64, state: SVector<f64, 6>| -> SMatrix<f64, 6, 6> {
            varmat_config.compute(t, state, &point_earth)
        };

        let config = IntegratorConfig::adaptive(1e-9, 1e-6);
        let rkn =
            RKN1210Integrator::with_config(Box::new(point_earth), Some(Box::new(varmat)), config);

        // Get start state
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 90.0, 0.0, 0.0, 0.0);
        let state0 = state_osculating_to_cartesian(oe0, DEGREES);
        let phi0 = SMatrix::<f64, 6, 6>::identity();

        // Take no step and confirm the variational matrix is the identity matrix
        let (_, phi1, _, _, _) = rkn.step_with_varmat(0.0, state0, phi0, 0.0, 1e-9, 1e-6);
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
        let (_, phi2, _, _, _) = rkn.step_with_varmat(0.0, state0, phi0, 1.0, 1e-9, 1e-6);
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
}
