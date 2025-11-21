/*!
Implementation of the 4th order Runge-Kutta integration method.
 */

use nalgebra::{DMatrix, DVector, SMatrix, SVector};

use crate::integrators::butcher_tableau::{ButcherTableau, RK4_TABLEAU};
use crate::integrators::config::IntegratorConfig;
use crate::integrators::traits::{
    ControlInput, ControlInputD, DIntegrator, FixedStepDIntegrator, FixedStepInternalResultD,
    FixedStepInternalResultS, FixedStepSIntegrator, SensitivityD, SensitivityS, StateDynamics,
    StateDynamicsD, VariationalMatrix, VariationalMatrixD, get_step_size,
};

/// Implementation of the 4th order Runge-Kutta numerical integrator. This implementation is generic
/// over the size of the state vector and parameter vector.
///
/// # Type Parameters
/// - `S`: State vector dimension
/// - `P`: Parameter vector dimension (for sensitivity matrix propagation)
///
/// # Example
///
/// ```
/// use nalgebra::{SVector, SMatrix};
/// use brahe::integrators::{RK4SIntegrator, FixedStepSIntegrator};
///
/// // Define a simple function for testing x' = 2x,
/// let f = |t: f64, state: SVector<f64, 1>| -> SVector<f64, 1> {
///    let mut state_new = SVector::<f64, 1>::zeros();
///     state_new[0] = 2.0*t;
///     state_new
/// };
///
///
/// // Create a new RK4 integrator
/// let rk4: RK4SIntegrator<1, 0> = RK4SIntegrator::new(Box::new(f), None, None, None);
///
/// // Define the initial state and time step
/// let mut t = 0.0;
/// let mut state = SVector::<f64, 1>::new(0.0);
/// let dt = 0.01;
///
/// // Integrate the system forward in time to t = 1.0 (analytic solution is x = 1.0)
/// for i in 0..100{
///    state = rk4.step(t, state, Some(dt));
///    t += dt;
/// }
///
/// assert!(state[0] - 1.0 < 1.0e-12);
///
/// // Now integrate the system forward in time to t = 10.0 (analytic solution is x = 100.0)
/// for i in 100..1000{
///     state = rk4.step(t, state, Some(dt));
///     t += dt;
/// }
///
/// assert!(state[0] - 100.0 < 1.0e-12);
/// ```
pub struct RK4SIntegrator<const S: usize, const P: usize> {
    f: StateDynamics<S>,
    varmat: VariationalMatrix<S>,
    sensmat: SensitivityS<S, P>,
    control: ControlInput<S, P>,
    bt: ButcherTableau<4>,
    config: IntegratorConfig,
}

impl<const S: usize, const P: usize> RK4SIntegrator<S, P> {
    /// Create a new 4th-order Runge-Kutta integrator.
    ///
    /// Initializes RK4 integrator with classical Butcher tableau. Fourth-order accuracy
    /// provides good balance between accuracy and computational cost for most ODE systems.
    ///
    /// # Arguments
    /// - `f`: State derivative function defining the dynamics (closure or function pointer)
    /// - `varmat`: Variational matrix computation function for state transition matrix propagation
    /// - `sensmat`: Sensitivity matrix computation function for parameter uncertainty propagation
    /// - `control`: Control input function
    ///
    /// # Returns
    /// RK4SIntegrator instance ready for numerical integration
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

    /// Create a new 4th-order Runge-Kutta integrator with custom configuration.
    ///
    /// # Arguments
    /// - `f`: State derivative function defining the dynamics
    /// - `varmat`: Variational matrix computation function for STM propagation
    /// - `sensmat`: Sensitivity matrix computation function for parameter uncertainty propagation
    /// - `control`: Control input function
    /// - `config`: Integration configuration (tolerances, step sizes, etc.)
    ///
    /// # Returns
    /// RK4SIntegrator instance with specified configuration
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::SVector;
    /// use brahe::integrators::{RK4SIntegrator, IntegratorConfig};
    ///
    /// let f = |t: f64, state: SVector<f64, 1>| -> SVector<f64, 1> {
    ///     SVector::<f64, 1>::new(2.0 * t)
    /// };
    ///
    /// let config = IntegratorConfig::fixed_step(0.01);
    /// let rk4: RK4SIntegrator<1, 0> = RK4SIntegrator::with_config(Box::new(f), None, None, None, config);
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
            bt: RK4_TABLEAU,
            config,
        }
    }

    /// Get a reference to the integrator configuration.
    pub fn config(&self) -> &IntegratorConfig {
        &self.config
    }

    /// Consolidated internal step method that handles all step variants.
    ///
    /// This method performs the core RK4 integration and optionally propagates
    /// the variational matrix (STM) and/or sensitivity matrix.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: Current state vector
    /// - `phi`: Optional state transition matrix
    /// - `sens`: Optional sensitivity matrix
    /// - `params`: Optional parameter vector (required if sens is Some)
    /// - `dt`: Step size
    fn step_internal(
        &self,
        t: f64,
        state: SVector<f64, S>,
        phi: Option<SMatrix<f64, S, S>>,
        sens: Option<SMatrix<f64, S, P>>,
        params: Option<&SVector<f64, P>>,
        dt: f64,
    ) -> FixedStepInternalResultS<S, P> {
        let compute_phi = phi.is_some();
        let compute_sens = sens.is_some();

        // Initialize working variables for RK stages
        let mut k = SMatrix::<f64, S, 4>::zeros();
        let mut k_phi = [SMatrix::<f64, S, S>::zeros(); 4];
        let mut k_sens = [SMatrix::<f64, S, P>::zeros(); 4];

        // Compute RK4 stages
        for i in 0..4 {
            let mut ksum = SVector::<f64, S>::zeros();
            let mut k_phi_sum = SMatrix::<f64, S, S>::zeros();
            let mut k_sens_sum = SMatrix::<f64, S, P>::zeros();

            // Sum previous stages
            for j in 0..i {
                ksum += self.bt.a[(i, j)] * k.column(j);
                if compute_phi {
                    k_phi_sum += self.bt.a[(i, j)] * k_phi[j];
                }
                if compute_sens {
                    k_sens_sum += self.bt.a[(i, j)] * k_sens[j];
                }
            }

            let state_i = state + dt * ksum;
            let t_i = t + self.bt.c[i] * dt;
            let mut k_i = (self.f)(t_i, state_i);

            // Apply control input if present
            if let Some(ref ctrl) = self.control {
                k_i += ctrl(t_i, state_i, params);
            }

            k.set_column(i, &k_i);

            // Compute Jacobian if needed for phi or sens
            if compute_phi || compute_sens {
                let a_i = self
                    .varmat
                    .as_ref()
                    .expect("varmat required for step_with_varmat or step_with_sensmat")
                    .compute(t_i, state_i);

                // Variational: dΦ/dt = A*Φ
                if compute_phi {
                    k_phi[i] = a_i * (phi.unwrap() + dt * k_phi_sum);
                }

                // Sensitivity: dS/dt = A*S + B
                if compute_sens {
                    let b_i = self
                        .sensmat
                        .as_ref()
                        .expect("sensmat required for step_with_sensmat")
                        .compute(t_i, &state_i, params.unwrap());
                    k_sens[i] = a_i * (sens.unwrap() + dt * k_sens_sum) + b_i;
                }
            }
        }

        // Compute updates from all stages
        let mut state_update = SVector::<f64, S>::zeros();
        let mut phi_update = SMatrix::<f64, S, S>::zeros();
        let mut sens_update = SMatrix::<f64, S, P>::zeros();

        for i in 0..4 {
            state_update += dt * self.bt.b[i] * k.column(i);
            if compute_phi {
                phi_update += dt * self.bt.b[i] * k_phi[i];
            }
            if compute_sens {
                sens_update += dt * self.bt.b[i] * k_sens[i];
            }
        }

        // Build result
        FixedStepInternalResultS {
            state: state + state_update,
            phi: phi.map(|p| p + phi_update),
            sens: sens.map(|s| s + sens_update),
        }
    }
}

impl<const S: usize, const P: usize> FixedStepSIntegrator<S, P> for RK4SIntegrator<S, P> {
    fn step(&self, t: f64, state: SVector<f64, S>, dt: Option<f64>) -> SVector<f64, S> {
        let dt = get_step_size(dt, &self.config);
        self.step_internal(t, state, None, None, None, dt).state
    }

    fn step_with_varmat(
        &self,
        t: f64,
        state: SVector<f64, S>,
        phi: SMatrix<f64, S, S>,
        dt: Option<f64>,
    ) -> (SVector<f64, S>, SMatrix<f64, S, S>) {
        let dt = get_step_size(dt, &self.config);
        let result = self.step_internal(t, state, Some(phi), None, None, dt);
        (result.state, result.phi.unwrap())
    }

    fn step_with_sensmat(
        &self,
        t: f64,
        state: SVector<f64, S>,
        sens: SMatrix<f64, S, P>,
        params: &SVector<f64, P>,
        dt: Option<f64>,
    ) -> (SVector<f64, S>, SMatrix<f64, S, P>) {
        let dt = get_step_size(dt, &self.config);
        let result = self.step_internal(t, state, None, Some(sens), Some(params), dt);
        (result.state, result.sens.unwrap())
    }

    fn step_with_varmat_sensmat(
        &self,
        t: f64,
        state: SVector<f64, S>,
        phi: SMatrix<f64, S, S>,
        sens: SMatrix<f64, S, P>,
        params: &SVector<f64, P>,
        dt: Option<f64>,
    ) -> (SVector<f64, S>, SMatrix<f64, S, S>, SMatrix<f64, S, P>) {
        let dt = get_step_size(dt, &self.config);
        let result = self.step_internal(t, state, Some(phi), Some(sens), Some(params), dt);
        (result.state, result.phi.unwrap(), result.sens.unwrap())
    }
}

// ============================================================================
// Dynamic (runtime-sized) RK4 Integrator
// ============================================================================

/// Implementation of the 4th order Runge-Kutta numerical integrator with runtime-sized state vectors.
///
/// This is the dynamic-sized counterpart to `RK4SIntegrator<S>`, using `DVector` and `DMatrix`
/// instead of compile-time sized vectors. This makes it ideal for Python bindings and applications
/// where state dimension needs to be determined at runtime.
///
/// # Example
///
/// ```
/// use nalgebra::DVector;
/// use brahe::integrators::{RK4DIntegrator, FixedStepDIntegrator};
///
/// // Define a simple exponential decay: x' = -x
/// let f = |t: f64, state: DVector<f64>, _: Option<&DVector<f64>>| -> DVector<f64> {
///     state.map(|x| -x)
/// };
///
/// // Create a new RK4 integrator for 2D system
/// let rk4 = RK4DIntegrator::new(2, Box::new(f), None, None, None);
///
/// // Define the initial state and time step
/// let mut t = 0.0;
/// let mut state = DVector::from_vec(vec![1.0, 2.0]);
/// let dt = 0.1;
///
/// // Integrate forward in time
/// for _ in 0..10 {
///     state = rk4.step(t, state, Some(dt));
///     t += dt;
/// }
/// ```
pub struct RK4DIntegrator {
    dimension: usize,
    f: StateDynamicsD,
    varmat: VariationalMatrixD,
    sensmat: SensitivityD,
    control: ControlInputD,
    bt: ButcherTableau<4>,
    config: IntegratorConfig,
}

impl RK4DIntegrator {
    /// Create a new 4th-order Runge-Kutta integrator with runtime-sized state vectors.
    ///
    /// Initializes RK4 integrator with classical Butcher tableau. Fourth-order accuracy
    /// provides good balance between accuracy and computational cost for most ODE systems.
    ///
    /// # Arguments
    /// - `dimension`: State vector dimension (runtime-determined)
    /// - `f`: State derivative function defining the dynamics
    /// - `varmat`: Optional variational matrix computation function for STM propagation
    ///
    /// # Returns
    /// RK4DIntegrator instance ready for numerical integration
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::DVector;
    /// use brahe::integrators::RK4DIntegrator;
    ///
    /// let f = |t: f64, state: DVector<f64>, _: Option<&DVector<f64>>| -> DVector<f64> {
    ///     DVector::from_vec(vec![state[1], -state[0]])  // Oscillator
    /// };
    ///
    /// let integrator = RK4DIntegrator::new(2, Box::new(f), None, None, None);
    /// ```
    pub fn new(
        dimension: usize,
        f: StateDynamicsD,
        varmat: VariationalMatrixD,
        sensmat: SensitivityD,
        control: ControlInputD,
    ) -> Self {
        <Self as DIntegrator>::new(dimension, f, varmat, sensmat, control)
    }

    /// Create a new 4th-order Runge-Kutta integrator with custom configuration.
    ///
    /// # Arguments
    /// - `dimension`: State vector dimension
    /// - `f`: State derivative function defining the dynamics
    /// - `varmat`: Optional variational matrix computation function
    /// - `sensmat`: Optional sensitivity matrix computation function
    /// - `control`: Optional control input function
    /// - `config`: Integration configuration
    ///
    /// # Returns
    /// RK4DIntegrator instance with specified configuration
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

impl DIntegrator for RK4DIntegrator {
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
        Self {
            dimension,
            f,
            varmat,
            sensmat,
            control,
            bt: RK4_TABLEAU,
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

impl RK4DIntegrator {
    /// Consolidated internal step method that handles all step variants.
    ///
    /// This method performs the core RK4 integration and optionally propagates
    /// the variational matrix (STM) and/or sensitivity matrix.
    ///
    /// # Arguments
    /// - `t`: Current time
    /// - `state`: Current state vector
    /// - `phi`: Optional state transition matrix
    /// - `sens`: Optional sensitivity matrix
    /// - `params`: Optional parameter vector (required if sens is Some)
    /// - `dt`: Step size
    fn step_internal(
        &self,
        t: f64,
        state: DVector<f64>,
        phi: Option<DMatrix<f64>>,
        sens: Option<DMatrix<f64>>,
        params: Option<&DVector<f64>>,
        dt: f64,
    ) -> FixedStepInternalResultD {
        // Validate dimensions
        assert_eq!(
            state.len(),
            self.dimension,
            "State dimension {} doesn't match integrator dimension {}",
            state.len(),
            self.dimension
        );

        if let Some(ref p) = phi {
            assert_eq!(
                p.nrows(),
                self.dimension,
                "STM rows {} doesn't match integrator dimension {}",
                p.nrows(),
                self.dimension
            );
            assert_eq!(
                p.ncols(),
                self.dimension,
                "STM cols {} doesn't match integrator dimension {}",
                p.ncols(),
                self.dimension
            );
        }

        if let Some(ref s) = sens {
            assert_eq!(
                s.nrows(),
                self.dimension,
                "Sensitivity rows {} doesn't match integrator dimension {}",
                s.nrows(),
                self.dimension
            );
        }

        let compute_phi = phi.is_some();
        let compute_sens = sens.is_some();
        let num_params = sens.as_ref().map(|s| s.ncols()).unwrap_or(0);

        // Initialize working variables for RK stages
        let mut k = DMatrix::<f64>::zeros(self.dimension, 4);

        // Only allocate if needed
        let mut k_phi = if compute_phi {
            vec![DMatrix::<f64>::zeros(self.dimension, self.dimension); 4]
        } else {
            vec![]
        };

        let mut k_sens = if compute_sens {
            vec![DMatrix::<f64>::zeros(self.dimension, num_params); 4]
        } else {
            vec![]
        };

        // Compute RK4 stages
        for i in 0..4 {
            let mut ksum = DVector::<f64>::zeros(self.dimension);
            let mut k_phi_sum = if compute_phi {
                DMatrix::<f64>::zeros(self.dimension, self.dimension)
            } else {
                DMatrix::<f64>::zeros(0, 0)
            };
            let mut k_sens_sum = if compute_sens {
                DMatrix::<f64>::zeros(self.dimension, num_params)
            } else {
                DMatrix::<f64>::zeros(0, 0)
            };

            // Sum previous stages
            for j in 0..i {
                ksum += self.bt.a[(i, j)] * k.column(j);
                if compute_phi {
                    k_phi_sum += self.bt.a[(i, j)] * &k_phi[j];
                }
                if compute_sens {
                    k_sens_sum += self.bt.a[(i, j)] * &k_sens[j];
                }
            }

            let state_i = &state + dt * &ksum;
            let t_i = t + self.bt.c[i] * dt;

            // Compute dynamics (pass params if computing sensitivity)
            let mut k_i = if compute_sens {
                (self.f)(t_i, state_i.clone(), params)
            } else {
                (self.f)(t_i, state_i.clone(), None)
            };

            // Apply control input if present
            if let Some(ref ctrl) = self.control {
                k_i += ctrl(t_i, state_i.clone(), params);
            }

            k.set_column(i, &k_i);

            // Compute Jacobian if needed for phi or sens
            if compute_phi || compute_sens {
                let a_i = self
                    .varmat
                    .as_ref()
                    .expect("varmat required for step_with_varmat or step_with_sensmat")
                    .compute(t_i, state_i.clone());

                // Variational: dΦ/dt = A*Φ
                if compute_phi {
                    k_phi[i] = &a_i * (phi.as_ref().unwrap() + dt * k_phi_sum);
                }

                // Sensitivity: dS/dt = A*S + B
                if compute_sens {
                    let b_i = self
                        .sensmat
                        .as_ref()
                        .expect("sensmat required for step_with_sensmat")
                        .compute(t_i, &state_i, params.unwrap());
                    k_sens[i] = &a_i * (sens.as_ref().unwrap() + dt * k_sens_sum) + b_i;
                }
            }
        }

        // Compute updates from all stages
        let mut state_update = DVector::<f64>::zeros(self.dimension);
        let mut phi_update = if compute_phi {
            DMatrix::<f64>::zeros(self.dimension, self.dimension)
        } else {
            DMatrix::<f64>::zeros(0, 0)
        };
        let mut sens_update = if compute_sens {
            DMatrix::<f64>::zeros(self.dimension, num_params)
        } else {
            DMatrix::<f64>::zeros(0, 0)
        };

        for i in 0..4 {
            state_update += dt * self.bt.b[i] * k.column(i);
            if compute_phi {
                phi_update += dt * self.bt.b[i] * &k_phi[i];
            }
            if compute_sens {
                sens_update += dt * self.bt.b[i] * &k_sens[i];
            }
        }

        // Build result
        FixedStepInternalResultD {
            state: state + state_update,
            phi: phi.map(|p| p + phi_update),
            sens: sens.map(|s| s + sens_update),
        }
    }
}

impl FixedStepDIntegrator for RK4DIntegrator {
    fn step(&self, t: f64, state: DVector<f64>, dt: Option<f64>) -> DVector<f64> {
        let dt = get_step_size(dt, &self.config);
        self.step_internal(t, state, None, None, None, dt).state
    }

    fn step_with_varmat(
        &self,
        t: f64,
        state: DVector<f64>,
        phi: DMatrix<f64>,
        dt: Option<f64>,
    ) -> (DVector<f64>, DMatrix<f64>) {
        let dt = get_step_size(dt, &self.config);
        let result = self.step_internal(t, state, Some(phi), None, None, dt);
        (result.state, result.phi.unwrap())
    }

    fn step_with_sensmat(
        &self,
        t: f64,
        state: DVector<f64>,
        sens: DMatrix<f64>,
        params: &DVector<f64>,
        dt: Option<f64>,
    ) -> (DVector<f64>, DMatrix<f64>) {
        let dt = get_step_size(dt, &self.config);
        let result = self.step_internal(t, state, None, Some(sens), Some(params), dt);
        (result.state, result.sens.unwrap())
    }

    fn step_with_varmat_sensmat(
        &self,
        t: f64,
        state: DVector<f64>,
        phi: DMatrix<f64>,
        sens: DMatrix<f64>,
        params: &DVector<f64>,
        dt: Option<f64>,
    ) -> (DVector<f64>, DMatrix<f64>, DMatrix<f64>) {
        let dt = get_step_size(dt, &self.config);
        let result = self.step_internal(t, state, Some(phi), Some(sens), Some(params), dt);
        (result.state, result.phi.unwrap(), result.sens.unwrap())
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::{DMatrix, DVector, SMatrix, SVector};

    use crate::constants::{DEGREES, RADIANS};
    use crate::integrators::rk4::{RK4DIntegrator, RK4SIntegrator};
    use crate::integrators::traits::{FixedStepDIntegrator, FixedStepSIntegrator};
    use crate::math::jacobian::{DNumericalJacobian, SNumericalJacobian};
    use crate::time::{Epoch, TimeSystem};
    use crate::{GM_EARTH, R_EARTH, orbital_period, state_osculating_to_cartesian};

    fn point_earth(_: f64, x: SVector<f64, 6>) -> SVector<f64, 6> {
        let r = x.fixed_rows::<3>(0);
        let v = x.fixed_rows::<3>(3);

        // Calculate acceleration
        let a = -GM_EARTH / r.norm().powi(3);

        // Construct state derivative
        let r_dot = v;
        let v_dot = a * r;

        let mut x_dot = SVector::<f64, 6>::zeros();
        x_dot.fixed_rows_mut::<3>(0).copy_from(&r_dot);
        x_dot.fixed_rows_mut::<3>(3).copy_from(&v_dot);

        x_dot
    }

    #[test]
    fn test_rk4s_integrator_cubic() {
        // Define a simple function for testing x' = 2x,
        let f = |t: f64, _: SVector<f64, 1>| -> SVector<f64, 1> {
            let mut state_new = SVector::<f64, 1>::zeros();
            state_new[0] = 3.0 * t * t;
            state_new
        };

        let rk4: RK4SIntegrator<1, 0> = RK4SIntegrator::new(Box::new(f), None, None, None);

        let mut state = SVector::<f64, 1>::new(0.0);
        let dt = 1.0;

        for i in 0..10 {
            state = rk4.step(i as f64, state, Some(dt));
        }

        assert_abs_diff_eq!(state[0], 1000.0, epsilon = 1.0e-12);
    }

    #[test]
    fn test_rk4s_integrator_parabola() {
        // Define a simple function for testing x' = 2x,
        let f = |t: f64, _: SVector<f64, 1>| -> SVector<f64, 1> {
            let mut state_new = SVector::<f64, 1>::zeros();
            state_new[0] = 2.0 * t;
            state_new
        };

        let rk4: RK4SIntegrator<1, 0> = RK4SIntegrator::new(Box::new(f), None, None, None);

        let mut t = 0.0;
        let mut state = SVector::<f64, 1>::new(0.0);
        let dt = 0.01;

        for _ in 0..100 {
            state = rk4.step(t, state, Some(dt));
            t += dt;
        }

        assert_abs_diff_eq!(state[0], 1.0, epsilon = 1.0e-12);
    }

    #[test]
    fn test_rk4s_integrator_orbit() {
        let rk4: RK4SIntegrator<6, 0> =
            RK4SIntegrator::new(Box::new(point_earth), None, None, None);

        // Get start state
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 90.0, 0.0, 0.0, 0.0);
        let state0 = state_osculating_to_cartesian(oe0, RADIANS);
        let mut state = state0;

        // Get start and end times of propagation (1 orbit)
        let epc0 = Epoch::from_date(2024, 1, 1, TimeSystem::TAI);
        let epcf = epc0 + orbital_period(oe0[0]);
        let mut dt;
        let mut epc = epc0;

        while epc < epcf {
            dt = (epcf - epc).min(1.0);
            state = rk4.step(epc - epc0, state, Some(dt));
            epc += dt;
        }

        assert_abs_diff_eq!(state.norm(), state0.norm(), epsilon = 1.0e-7);
        assert_abs_diff_eq!(state[0], state0[0], epsilon = 1.0e-5);
        assert_abs_diff_eq!(state[1], state0[1], epsilon = 1.0e-5);
        assert_abs_diff_eq!(state[2], state0[2], epsilon = 1.0e-5);
        assert_abs_diff_eq!(state[3], state0[3], epsilon = 1.0e-5);
        assert_abs_diff_eq!(state[4], state0[4], epsilon = 1.0e-5);
        assert_abs_diff_eq!(state[5], state0[5], epsilon = 1.0e-5);
    }

    #[test]
    fn test_rk4s_integrator_varmat() {
        // Define how we want to calculate the variational matrix for the RK4 integrator
        // Use SNumericalJacobian with fixed offset
        let jacobian = SNumericalJacobian::new(Box::new(point_earth)).with_fixed_offset(1.0);

        let rk4: RK4SIntegrator<6, 0> =
            RK4SIntegrator::new(Box::new(point_earth), Some(Box::new(jacobian)), None, None);

        // Get start state
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 90.0, 0.0, 0.0, 0.0);
        let state0 = state_osculating_to_cartesian(oe0, RADIANS);
        let phi0 = SMatrix::<f64, 6, 6>::identity();

        // Take no step and confirm the variational matrix is the identity matrix
        let (_, phi1) = rk4.step_with_varmat(0.0, state0, phi0, Some(0.0));
        for i in 0..6 {
            for j in 0..6 {
                if i == j {
                    assert_abs_diff_eq!(phi1[(i, j)], 1.0, epsilon = 1.0e-12);
                } else {
                    assert_abs_diff_eq!(phi1[(i, j)], 0.0, epsilon = 1.0e-12);
                }
            }
        }

        // Propagate one step and independently confirm the variational matrix update
        let (_, phi2) = rk4.step_with_varmat(0.0, state0, phi0, Some(1.0));
        for i in 0..6 {
            for j in 0..6 {
                if i == j {
                    assert_ne!(phi2[(i, i)], 1.0);
                    assert_ne!(phi2[(i, i)], 0.0);
                    assert_abs_diff_eq!(phi2[(i, i)], 1.0, epsilon = 1.0e-5);
                } else {
                    // Ensure there are off-diagonal elements are now populated
                    assert_ne!(phi2[(i, j)], 0.0);
                }
            }
        }

        // Compare updating the state with a perturbation and the result from using the variational matrix
        // Define a simple perturbation to simplify the tests
        let pert = SVector::<f64, 6>::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);

        // Create a new jacobian provider with central differences and custom perturbations
        // This demonstrates using custom offset for each component
        let jacobian2 = SNumericalJacobian::central(Box::new(point_earth)).with_fixed_offset(1.0);
        let rk4: RK4SIntegrator<6, 0> =
            RK4SIntegrator::new(Box::new(point_earth), Some(Box::new(jacobian2)), None, None);

        // Get the state with a perturbation
        let (state_pert, _) = rk4.step_with_varmat(0.0, state0 + pert, phi0, Some(1.0));

        // Get the state with a perturbation by using the integrated variational matrix
        let state_stm = rk4.step(0.0, state0, Some(1.0)) + phi2 * pert;

        // Compare the two states - they should be the same
        assert_abs_diff_eq!(state_pert[0], state_stm[0], epsilon = 1.0e-9);
        assert_abs_diff_eq!(state_pert[1], state_stm[1], epsilon = 1.0e-9);
        assert_abs_diff_eq!(state_pert[2], state_stm[2], epsilon = 1.0e-9);
        assert_abs_diff_eq!(state_pert[3], state_stm[3], epsilon = 1.0e-9);
        assert_abs_diff_eq!(state_pert[4], state_stm[4], epsilon = 1.0e-9);
        assert_abs_diff_eq!(state_pert[5], state_stm[5], epsilon = 1.0e-9);
    }

    // ========================================================================
    // Dynamic RK4 Tests
    // ========================================================================

    fn point_earth_dynamic(
        _: f64,
        x: DVector<f64>,
        _params: Option<&DVector<f64>>,
    ) -> DVector<f64> {
        assert_eq!(x.len(), 6, "State must be 6D for orbital mechanics");

        let r = x.rows(0, 3);
        let v = x.rows(3, 3);

        // Calculate acceleration
        let r_norm = r.norm();
        let a = -GM_EARTH / r_norm.powi(3);

        // Construct state derivative
        let mut x_dot = DVector::<f64>::zeros(6);
        x_dot.rows_mut(0, 3).copy_from(&v);
        x_dot.rows_mut(3, 3).copy_from(&(a * r));

        x_dot
    }

    #[test]
    fn test_rk4d_integrator_cubic() {
        // Define a simple function for testing x' = 3t²
        let f = |t: f64, _: DVector<f64>, _: Option<&DVector<f64>>| -> DVector<f64> {
            DVector::from_vec(vec![3.0 * t * t])
        };

        let rk4 = RK4DIntegrator::new(1, Box::new(f), None, None, None);

        let mut state = DVector::from_vec(vec![0.0]);
        let dt = 1.0;

        for i in 0..10 {
            state = rk4.step(i as f64, state, Some(dt));
        }

        assert_abs_diff_eq!(state[0], 1000.0, epsilon = 1.0e-12);
    }

    #[test]
    fn test_rk4d_integrator_parabola() {
        // Define a simple function for testing x' = 2t
        let f = |t: f64, _: DVector<f64>, _: Option<&DVector<f64>>| -> DVector<f64> {
            DVector::from_vec(vec![2.0 * t])
        };

        let rk4 = RK4DIntegrator::new(1, Box::new(f), None, None, None);

        let mut t = 0.0;
        let mut state = DVector::from_vec(vec![0.0]);
        let dt = 0.01;

        for _ in 0..100 {
            state = rk4.step(t, state, Some(dt));
            t += dt;
        }

        assert_abs_diff_eq!(state[0], 1.0, epsilon = 1.0e-12);
    }

    #[test]
    fn test_rk4d_integrator_orbit() {
        let rk4 = RK4DIntegrator::new(6, Box::new(point_earth_dynamic), None, None, None);

        // Get start state
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 90.0, 0.0, 0.0, 0.0);
        let state0_static = state_osculating_to_cartesian(oe0, RADIANS);
        let state0 = DVector::from_vec(state0_static.as_slice().to_vec());
        let mut state = state0.clone();

        // Get start and end times of propagation (1 orbit)
        let epc0 = Epoch::from_date(2024, 1, 1, TimeSystem::TAI);
        let epcf = epc0 + orbital_period(oe0[0]);
        let mut dt;
        let mut epc = epc0;

        while epc < epcf {
            dt = (epcf - epc).min(1.0);
            state = rk4.step(epc - epc0, state, Some(dt));
            epc += dt;
        }

        assert_abs_diff_eq!(state.norm(), state0.norm(), epsilon = 1.0e-7);
        assert_abs_diff_eq!(state[0], state0[0], epsilon = 1.0e-5);
        assert_abs_diff_eq!(state[1], state0[1], epsilon = 1.0e-5);
        assert_abs_diff_eq!(state[2], state0[2], epsilon = 1.0e-5);
        assert_abs_diff_eq!(state[3], state0[3], epsilon = 1.0e-5);
        assert_abs_diff_eq!(state[4], state0[4], epsilon = 1.0e-5);
        assert_abs_diff_eq!(state[5], state0[5], epsilon = 1.0e-5);
    }

    #[test]
    fn test_rk4d_integrator_varmat() {
        // Define a 2-argument wrapper for the jacobian (it doesn't need params)
        let point_earth_for_jacobian =
            |t: f64, x: DVector<f64>| -> DVector<f64> { point_earth_dynamic(t, x, None) };

        // Define how we want to calculate the variational matrix for the RK4 integrator
        let jacobian =
            DNumericalJacobian::new(Box::new(point_earth_for_jacobian)).with_fixed_offset(1.0);

        let rk4 = RK4DIntegrator::new(
            6,
            Box::new(point_earth_dynamic),
            Some(Box::new(jacobian)),
            None,
            None,
        );

        // Get start state
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 90.0, 0.0, 0.0, 0.0);
        let state0_static = state_osculating_to_cartesian(oe0, RADIANS);
        let state0 = DVector::from_vec(state0_static.as_slice().to_vec());
        let phi0 = DMatrix::<f64>::identity(6, 6);

        // Take no step and confirm the variational matrix is the identity matrix
        let (_, phi1) = rk4.step_with_varmat(0.0, state0.clone(), phi0.clone(), Some(0.0));
        for i in 0..6 {
            for j in 0..6 {
                if i == j {
                    assert_abs_diff_eq!(phi1[(i, j)], 1.0, epsilon = 1.0e-12);
                } else {
                    assert_abs_diff_eq!(phi1[(i, j)], 0.0, epsilon = 1.0e-12);
                }
            }
        }

        // Propagate one step and independently confirm the variational matrix update
        let (_, phi2) = rk4.step_with_varmat(0.0, state0.clone(), phi0.clone(), Some(1.0));
        for i in 0..6 {
            for j in 0..6 {
                if i == j {
                    assert_ne!(phi2[(i, i)], 1.0);
                    assert_ne!(phi2[(i, i)], 0.0);
                    assert_abs_diff_eq!(phi2[(i, i)], 1.0, epsilon = 1.0e-5);
                } else {
                    // Ensure there are off-diagonal elements are now populated
                    assert_ne!(phi2[(i, j)], 0.0);
                }
            }
        }

        // Compare updating the state with a perturbation and the result from using the variational matrix
        let pert = DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // Create a new jacobian provider with central differences
        let point_earth_for_jacobian2 =
            |t: f64, x: DVector<f64>| -> DVector<f64> { point_earth_dynamic(t, x, None) };
        let jacobian2 =
            DNumericalJacobian::central(Box::new(point_earth_for_jacobian2)).with_fixed_offset(1.0);
        let rk4 = RK4DIntegrator::new(
            6,
            Box::new(point_earth_dynamic),
            Some(Box::new(jacobian2)),
            None,
            None,
        );

        // Get the state with a perturbation
        let (state_pert, _) = rk4.step_with_varmat(0.0, &state0 + &pert, phi0, Some(1.0));

        // Get the state with a perturbation by using the integrated variational matrix
        let state_stm = rk4.step(0.0, state0.clone(), Some(1.0)) + &phi2 * &pert;

        // Compare the two states - they should be the same
        assert_abs_diff_eq!(state_pert[0], state_stm[0], epsilon = 1.0e-9);
        assert_abs_diff_eq!(state_pert[1], state_stm[1], epsilon = 1.0e-9);
        assert_abs_diff_eq!(state_pert[2], state_stm[2], epsilon = 1.0e-9);
        assert_abs_diff_eq!(state_pert[3], state_stm[3], epsilon = 1.0e-9);
        assert_abs_diff_eq!(state_pert[4], state_stm[4], epsilon = 1.0e-9);
        assert_abs_diff_eq!(state_pert[5], state_stm[5], epsilon = 1.0e-9);
    }

    #[test]
    fn test_rk4_s_vs_d_consistency() {
        // Verify RK4SIntegrator and RK4DIntegrator produce identical results
        let f_static = |_t: f64, x: SVector<f64, 3>| -> SVector<f64, 3> {
            SVector::<f64, 3>::new(-x[0], -x[1], -x[2])
        };
        let f_dynamic = |_t: f64, x: DVector<f64>, _: Option<&DVector<f64>>| -> DVector<f64> {
            DVector::from_vec(vec![-x[0], -x[1], -x[2]])
        };

        let rk4_s: RK4SIntegrator<3, 0> = RK4SIntegrator::new(Box::new(f_static), None, None, None);
        let rk4_d = RK4DIntegrator::new(3, Box::new(f_dynamic), None, None, None);

        let state_s = SVector::<f64, 3>::new(1.0, 2.0, 3.0);
        let state_d = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let dt = 0.1;

        let result_s = rk4_s.step(0.0, state_s, Some(dt));
        let result_d = rk4_d.step(0.0, state_d, Some(dt));

        // Results should be identical to machine precision
        assert_abs_diff_eq!(result_s[0], result_d[0], epsilon = 1.0e-15);
        assert_abs_diff_eq!(result_s[1], result_d[1], epsilon = 1.0e-15);
        assert_abs_diff_eq!(result_s[2], result_d[2], epsilon = 1.0e-15);
    }

    #[test]
    fn test_rk4s_backward_integration() {
        // Test backward propagation with orbital mechanics
        let rk4: RK4SIntegrator<6, 0> =
            RK4SIntegrator::new(Box::new(point_earth), None, None, None);

        // Setup initial state
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 90.0, 0.0, 0.0, 0.0);
        let state0 = state_osculating_to_cartesian(oe0, DEGREES);

        // Propagate forward for 100 seconds with timestep 1 second
        let dt_forward = 1.0;
        let mut state_fwd = state0;
        for _ in 0..100 {
            state_fwd = rk4.step(0.0, state_fwd, Some(dt_forward));
        }

        // Now propagate backward from the final state
        let dt_back = -1.0; // Negative timestep for backward integration
        let mut state_back = state_fwd;
        for _ in 0..100 {
            state_back = rk4.step(0.0, state_back, Some(dt_back));
        }

        // Should return close to initial state
        for i in 0..6 {
            assert_abs_diff_eq!(state_back[i], state0[i], epsilon = 1.0e-9);
        }
    }

    #[test]
    fn test_rk4d_backward_integration() {
        // Test backward propagation with orbital mechanics (dynamic variant)
        let rk4 = RK4DIntegrator::new(6, Box::new(point_earth_dynamic), None, None, None);

        // Setup initial state
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 90.0, 0.0, 0.0, 0.0);
        let state0_static = state_osculating_to_cartesian(oe0, DEGREES);
        let state0 = DVector::from_vec(state0_static.as_slice().to_vec());

        // Propagate forward for 100 seconds with timestep 1 second
        let dt_forward = 1.0;
        let mut state_fwd = state0.clone();
        for _ in 0..100 {
            state_fwd = rk4.step(0.0, state_fwd, Some(dt_forward));
        }

        // Now propagate backward from the final state
        let dt_back = -1.0; // Negative timestep for backward integration
        let mut state_back = state_fwd;
        for _ in 0..100 {
            state_back = rk4.step(0.0, state_back, Some(dt_back));
        }

        // Should return close to initial state
        for i in 0..6 {
            assert_abs_diff_eq!(state_back[i], state0[i], epsilon = 1.0e-9);
        }
    }

    // ========================================================================
    // Control Input Tests
    // ========================================================================

    #[test]
    fn test_rk4s_integrator_with_control_input() {
        // Simple dynamics: x' = 0 (constant state without control)
        let f = |_t: f64, _x: SVector<f64, 1>| -> SVector<f64, 1> { SVector::<f64, 1>::zeros() };

        // Control input adds constant rate: u = 1.0
        let control = |_t: f64,
                       _x: SVector<f64, 1>,
                       _p: Option<&SVector<f64, 0>>|
         -> SVector<f64, 1> { SVector::<f64, 1>::new(1.0) };

        // Without control: state stays at 0
        let rk4_no_ctrl: RK4SIntegrator<1, 0> = RK4SIntegrator::new(Box::new(f), None, None, None);
        let state0 = SVector::<f64, 1>::new(0.0);
        let state_no_ctrl = rk4_no_ctrl.step(0.0, state0, Some(1.0));
        assert_abs_diff_eq!(state_no_ctrl[0], 0.0, epsilon = 1.0e-12);

        // With control: x' = 1, so x = t after integration
        let rk4_ctrl: RK4SIntegrator<1, 0> =
            RK4SIntegrator::new(Box::new(f), None, None, Some(Box::new(control)));
        let state_ctrl = rk4_ctrl.step(0.0, state0, Some(1.0));
        assert_abs_diff_eq!(state_ctrl[0], 1.0, epsilon = 1.0e-12);
    }

    #[test]
    fn test_rk4d_integrator_with_control_input() {
        // Simple dynamics: x' = 0 (constant state without control)
        let f = |_t: f64, x: DVector<f64>, _: Option<&DVector<f64>>| -> DVector<f64> {
            DVector::zeros(x.len())
        };

        // Control input adds constant rate: u = [1.0, 2.0]
        let control = |_t: f64, _x: DVector<f64>, _p: Option<&DVector<f64>>| -> DVector<f64> {
            DVector::from_vec(vec![1.0, 2.0])
        };

        // Without control: state stays constant
        let rk4_no_ctrl = RK4DIntegrator::new(2, Box::new(f), None, None, None);
        let state0 = DVector::from_vec(vec![0.0, 0.0]);
        let state_no_ctrl = rk4_no_ctrl.step(0.0, state0.clone(), Some(1.0));
        assert_abs_diff_eq!(state_no_ctrl[0], 0.0, epsilon = 1.0e-12);
        assert_abs_diff_eq!(state_no_ctrl[1], 0.0, epsilon = 1.0e-12);

        // With control: x' = [1, 2], so x = [t, 2t] after integration
        let rk4_ctrl = RK4DIntegrator::new(2, Box::new(f), None, None, Some(Box::new(control)));
        let state_ctrl = rk4_ctrl.step(0.0, state0, Some(1.0));
        assert_abs_diff_eq!(state_ctrl[0], 1.0, epsilon = 1.0e-12);
        assert_abs_diff_eq!(state_ctrl[1], 2.0, epsilon = 1.0e-12);
    }

    #[test]
    fn test_rk4s_integrator_control_with_dynamics() {
        // Dynamics: x' = -x (exponential decay)
        let f = |_t: f64, x: SVector<f64, 1>| -> SVector<f64, 1> { -x };

        // Control input: u = 1 (constant forcing)
        let control = |_t: f64,
                       _x: SVector<f64, 1>,
                       _p: Option<&SVector<f64, 0>>|
         -> SVector<f64, 1> { SVector::<f64, 1>::new(1.0) };

        // With control: x' = -x + 1, equilibrium at x = 1
        let rk4: RK4SIntegrator<1, 0> =
            RK4SIntegrator::new(Box::new(f), None, None, Some(Box::new(control)));

        let mut state = SVector::<f64, 1>::new(0.0);
        let dt = 0.1;

        // Integrate for many steps - should approach equilibrium at x = 1
        for _ in 0..100 {
            state = rk4.step(0.0, state, Some(dt));
        }

        // State should approach equilibrium value of 1.0
        assert_abs_diff_eq!(state[0], 1.0, epsilon = 1.0e-3);
    }

    #[test]
    fn test_rk4s_integrator_state_dependent_control() {
        // Dynamics: x' = 0
        let f = |_t: f64, _x: SVector<f64, 1>| -> SVector<f64, 1> { SVector::<f64, 1>::zeros() };

        // State-dependent control: u = -x (proportional feedback)
        let control =
            |_t: f64, x: SVector<f64, 1>, _p: Option<&SVector<f64, 0>>| -> SVector<f64, 1> { -x };

        // With this control: x' = -x, so exponential decay
        let rk4: RK4SIntegrator<1, 0> =
            RK4SIntegrator::new(Box::new(f), None, None, Some(Box::new(control)));

        let mut state = SVector::<f64, 1>::new(1.0);
        let dt = 0.1;

        // Integrate and check decay
        for _ in 0..50 {
            state = rk4.step(0.0, state, Some(dt));
        }

        // State should decay toward 0
        assert!(state[0] < 0.1);
    }

    // ========================================================================
    // Sensitivity Matrix Tests
    // ========================================================================

    #[test]
    fn test_rk4s_integrator_sensmat() {
        // Test sensitivity matrix propagation using exponential decay: dx/dt = -k*x
        // where k is a parameter.
        //
        // Analytical sensitivity: S(t) = -x0 * t * exp(-k*t) for S(0) = 0

        use crate::math::jacobian::SJacobianProvider;
        use crate::math::sensitivity::SSensitivityProvider;

        // Jacobian provider: ∂f/∂x = -k (using k=1.0)
        struct DecayJacobian;
        impl SJacobianProvider<1> for DecayJacobian {
            fn compute(&self, _t: f64, _state: SVector<f64, 1>) -> SMatrix<f64, 1, 1> {
                SMatrix::<f64, 1, 1>::new(-1.0)
            }
        }

        // Sensitivity provider: ∂f/∂k = -x
        struct DecaySensitivity;
        impl SSensitivityProvider<1, 1> for DecaySensitivity {
            fn compute(
                &self,
                _t: f64,
                state: &SVector<f64, 1>,
                _params: &SVector<f64, 1>,
            ) -> SMatrix<f64, 1, 1> {
                SMatrix::<f64, 1, 1>::new(-state[0])
            }
        }

        // Dynamics: dx/dt = -k*x where k = params[0]
        let f = |_t: f64, x: SVector<f64, 1>| -> SVector<f64, 1> { -x };

        let rk4: RK4SIntegrator<1, 1> = RK4SIntegrator::new(
            Box::new(f),
            Some(Box::new(DecayJacobian)),
            Some(Box::new(DecaySensitivity)),
            None,
        );

        // Initial conditions
        let x0 = 1.0;
        let state0 = SVector::<f64, 1>::new(x0);
        let sens0 = SMatrix::<f64, 1, 1>::zeros(); // Initial sensitivity is zero
        let params = SVector::<f64, 1>::new(1.0); // k = 1.0
        let dt = 0.01;

        // Propagate for 1 second
        let mut state = state0;
        let mut sens = sens0;
        let mut t = 0.0;

        for _ in 0..100 {
            let (new_state, new_sens) = rk4.step_with_sensmat(t, state, sens, &params, Some(dt));
            state = new_state;
            sens = new_sens;
            t += dt;
        }

        // Check against analytical solution
        // x(t) = x0 * exp(-k*t)
        let k = params[0];
        let x_analytical = x0 * (-k * t).exp();
        // S(t) = -x0 * t * exp(-k*t)
        let sens_analytical = -x0 * t * (-k * t).exp();

        // State should match analytical solution
        assert_abs_diff_eq!(state[0], x_analytical, epsilon = 1e-6);

        // Sensitivity should match analytical solution
        assert_abs_diff_eq!(sens[(0, 0)], sens_analytical, epsilon = 1e-4);
    }

    #[test]
    fn test_rk4d_integrator_sensmat() {
        // Test sensitivity matrix propagation using exponential decay (dynamic version)

        use crate::math::jacobian::DJacobianProvider;
        use crate::math::sensitivity::DSensitivityProvider;

        // Jacobian provider: ∂f/∂x = -k (using k=1.0)
        struct DecayJacobian;
        impl DJacobianProvider for DecayJacobian {
            fn compute(&self, _t: f64, _state: DVector<f64>) -> DMatrix<f64> {
                DMatrix::from_vec(1, 1, vec![-1.0])
            }
        }

        // Sensitivity provider: ∂f/∂k = -x
        struct DecaySensitivity;
        impl DSensitivityProvider for DecaySensitivity {
            fn compute(
                &self,
                _t: f64,
                state: &DVector<f64>,
                _params: &DVector<f64>,
            ) -> DMatrix<f64> {
                DMatrix::from_vec(1, 1, vec![-state[0]])
            }
        }

        // Dynamics: dx/dt = -k*x
        let f = |_t: f64, x: DVector<f64>, _: Option<&DVector<f64>>| -> DVector<f64> { -x };

        let rk4 = RK4DIntegrator::new(
            1,
            Box::new(f),
            Some(Box::new(DecayJacobian)),
            Some(Box::new(DecaySensitivity)),
            None,
        );

        // Initial conditions
        let x0 = 1.0;
        let state0 = DVector::from_vec(vec![x0]);
        let sens0 = DMatrix::zeros(1, 1);
        let params = DVector::from_vec(vec![1.0]); // k = 1.0
        let dt = 0.01;

        // Propagate for 1 second
        let mut state = state0;
        let mut sens = sens0;
        let mut t = 0.0;

        for _ in 0..100 {
            let (new_state, new_sens) = rk4.step_with_sensmat(t, state, sens, &params, Some(dt));
            state = new_state;
            sens = new_sens;
            t += dt;
        }

        // Check against analytical solution
        let k = params[0];
        let x_analytical = x0 * (-k * t).exp();
        let sens_analytical = -x0 * t * (-k * t).exp();

        assert_abs_diff_eq!(state[0], x_analytical, epsilon = 1e-6);
        assert_abs_diff_eq!(sens[(0, 0)], sens_analytical, epsilon = 1e-4);
    }

    #[test]
    fn test_rk4s_integrator_varmat_sensmat() {
        // Test combined STM and sensitivity matrix propagation

        use crate::math::jacobian::SJacobianProvider;
        use crate::math::sensitivity::SSensitivityProvider;

        struct DecayJacobian;
        impl SJacobianProvider<1> for DecayJacobian {
            fn compute(&self, _t: f64, _state: SVector<f64, 1>) -> SMatrix<f64, 1, 1> {
                SMatrix::<f64, 1, 1>::new(-1.0)
            }
        }

        struct DecaySensitivity;
        impl SSensitivityProvider<1, 1> for DecaySensitivity {
            fn compute(
                &self,
                _t: f64,
                state: &SVector<f64, 1>,
                _params: &SVector<f64, 1>,
            ) -> SMatrix<f64, 1, 1> {
                SMatrix::<f64, 1, 1>::new(-state[0])
            }
        }

        let f = |_t: f64, x: SVector<f64, 1>| -> SVector<f64, 1> { -x };

        let rk4: RK4SIntegrator<1, 1> = RK4SIntegrator::new(
            Box::new(f),
            Some(Box::new(DecayJacobian)),
            Some(Box::new(DecaySensitivity)),
            None,
        );

        // Initial conditions
        let x0 = 1.0;
        let state0 = SVector::<f64, 1>::new(x0);
        let phi0 = SMatrix::<f64, 1, 1>::identity();
        let sens0 = SMatrix::<f64, 1, 1>::zeros();
        let params = SVector::<f64, 1>::new(1.0);
        let dt = 0.01;

        // Propagate for 1 second
        let mut state = state0;
        let mut phi = phi0;
        let mut sens = sens0;
        let mut t = 0.0;

        for _ in 0..100 {
            let (new_state, new_phi, new_sens) =
                rk4.step_with_varmat_sensmat(t, state, phi, sens, &params, Some(dt));
            state = new_state;
            phi = new_phi;
            sens = new_sens;
            t += dt;
        }

        // Check against analytical solutions
        let k = params[0];
        let x_analytical = x0 * (-k * t).exp();
        let phi_analytical = (-k * t).exp();
        let sens_analytical = -x0 * t * (-k * t).exp();

        assert_abs_diff_eq!(state[0], x_analytical, epsilon = 1e-6);
        assert_abs_diff_eq!(phi[(0, 0)], phi_analytical, epsilon = 1e-4);
        assert_abs_diff_eq!(sens[(0, 0)], sens_analytical, epsilon = 1e-4);

        // Test 2: Verify STM accuracy via direct perturbation
        let delta = 1e-6;
        let state0_pert = SVector::<f64, 1>::new(x0 + delta);
        let mut state_pert = state0_pert;
        for _ in 0..100 {
            state_pert = rk4.step(0.0, state_pert, Some(dt));
        }

        // STM should predict the perturbed state
        let state_pert_predicted = state[0] + phi[(0, 0)] * delta;
        let relative_error = (state_pert[0] - state_pert_predicted).abs() / delta;
        assert!(
            relative_error < 1e-3,
            "STM prediction error: {}",
            relative_error
        );
    }

    #[test]
    fn test_rk4d_integrator_varmat_sensmat() {
        // Test combined STM and sensitivity matrix propagation (dynamic version)

        use crate::math::jacobian::DJacobianProvider;
        use crate::math::sensitivity::DSensitivityProvider;

        struct DecayJacobian;
        impl DJacobianProvider for DecayJacobian {
            fn compute(&self, _t: f64, _state: DVector<f64>) -> DMatrix<f64> {
                DMatrix::from_vec(1, 1, vec![-1.0])
            }
        }

        struct DecaySensitivity;
        impl DSensitivityProvider for DecaySensitivity {
            fn compute(
                &self,
                _t: f64,
                state: &DVector<f64>,
                _params: &DVector<f64>,
            ) -> DMatrix<f64> {
                DMatrix::from_vec(1, 1, vec![-state[0]])
            }
        }

        let f = |_t: f64, x: DVector<f64>, _: Option<&DVector<f64>>| -> DVector<f64> { -x };

        let rk4 = RK4DIntegrator::new(
            1,
            Box::new(f),
            Some(Box::new(DecayJacobian)),
            Some(Box::new(DecaySensitivity)),
            None,
        );

        // Initial conditions
        let x0 = 1.0;
        let state0 = DVector::from_vec(vec![x0]);
        let phi0 = DMatrix::identity(1, 1);
        let sens0 = DMatrix::zeros(1, 1);
        let params = DVector::from_vec(vec![1.0]);
        let dt = 0.01;

        // Propagate for 1 second
        let mut state = state0.clone();
        let mut phi = phi0;
        let mut sens = sens0;
        let mut t = 0.0;

        for _ in 0..100 {
            let (new_state, new_phi, new_sens) =
                rk4.step_with_varmat_sensmat(t, state, phi, sens, &params, Some(dt));
            state = new_state;
            phi = new_phi;
            sens = new_sens;
            t += dt;
        }

        // Check against analytical solutions
        let k = params[0];
        let x_analytical = x0 * (-k * t).exp();
        let phi_analytical = (-k * t).exp();
        let sens_analytical = -x0 * t * (-k * t).exp();

        assert_abs_diff_eq!(state[0], x_analytical, epsilon = 1e-6);
        assert_abs_diff_eq!(phi[(0, 0)], phi_analytical, epsilon = 1e-4);
        assert_abs_diff_eq!(sens[(0, 0)], sens_analytical, epsilon = 1e-4);

        // Test STM accuracy via direct perturbation
        let delta = 1e-6;
        let state0_pert = DVector::from_vec(vec![x0 + delta]);
        let mut state_pert = state0_pert;
        for _ in 0..100 {
            state_pert = rk4.step(0.0, state_pert, Some(dt));
        }

        let state_pert_predicted = state[0] + phi[(0, 0)] * delta;
        let relative_error = (state_pert[0] - state_pert_predicted).abs() / delta;
        assert!(
            relative_error < 1e-3,
            "STM prediction error: {}",
            relative_error
        );
    }

    // =============================================================================
    // Constructor and Config Tests
    // =============================================================================

    #[test]
    fn test_rk4s_new_uses_default_config() {
        // Simple linear ODE: dx/dt = x
        fn dynamics(_t: f64, state: SVector<f64, 1>) -> SVector<f64, 1> {
            state
        }

        let integrator: RK4SIntegrator<1, 0> =
            RK4SIntegrator::new(Box::new(dynamics), None, None, None);
        let config = integrator.config();

        // Verify that new() uses default config values
        let default_config = crate::integrators::config::IntegratorConfig::default();
        assert_eq!(config.abs_tol, default_config.abs_tol);
        assert_eq!(config.rel_tol, default_config.rel_tol);
        assert_eq!(config.max_step_attempts, default_config.max_step_attempts);
        assert_eq!(config.min_step, default_config.min_step);
        assert_eq!(config.max_step, default_config.max_step);
    }

    #[test]
    fn test_rk4s_with_config_stores_config() {
        fn dynamics(_t: f64, state: SVector<f64, 1>) -> SVector<f64, 1> {
            state
        }

        let custom_config = crate::integrators::config::IntegratorConfig {
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

        let integrator: RK4SIntegrator<1, 0> = RK4SIntegrator::with_config(
            Box::new(dynamics),
            None,
            None,
            None,
            custom_config.clone(),
        );
        let config = integrator.config();

        // Verify the custom config was stored
        assert_eq!(config.abs_tol, 1e-10);
        assert_eq!(config.rel_tol, 1e-8);
        assert_eq!(config.max_step_attempts, 20);
        assert_eq!(config.min_step, Some(1e-15));
        assert_eq!(config.max_step, Some(100.0));
        assert_eq!(config.max_step_scale_factor, Some(5.0));
        assert_eq!(config.min_step_scale_factor, Some(0.1));
    }

    #[test]
    fn test_rk4s_config_returns_reference() {
        fn dynamics(_t: f64, state: SVector<f64, 1>) -> SVector<f64, 1> {
            state
        }

        let integrator: RK4SIntegrator<1, 0> =
            RK4SIntegrator::new(Box::new(dynamics), None, None, None);

        // Call config() multiple times and verify it returns the same values
        let config1 = integrator.config();
        let config2 = integrator.config();

        assert_eq!(config1.abs_tol, config2.abs_tol);
        assert_eq!(config1.rel_tol, config2.rel_tol);
        assert_eq!(config1.max_step_attempts, config2.max_step_attempts);
    }

    #[test]
    fn test_rk4d_new_uses_default_config() {
        fn dynamics(_t: f64, state: DVector<f64>, _params: Option<&DVector<f64>>) -> DVector<f64> {
            state
        }

        let integrator = RK4DIntegrator::new(1, Box::new(dynamics), None, None, None);
        let config = integrator.config();

        // Verify that new() uses default config values
        let default_config = crate::integrators::config::IntegratorConfig::default();
        assert_eq!(config.abs_tol, default_config.abs_tol);
        assert_eq!(config.rel_tol, default_config.rel_tol);
        assert_eq!(config.max_step_attempts, default_config.max_step_attempts);
        assert_eq!(config.min_step, default_config.min_step);
        assert_eq!(config.max_step, default_config.max_step);
    }

    #[test]
    fn test_rk4d_with_config_stores_config() {
        fn dynamics(_t: f64, state: DVector<f64>, _params: Option<&DVector<f64>>) -> DVector<f64> {
            state
        }

        let custom_config = crate::integrators::config::IntegratorConfig {
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

        let integrator = RK4DIntegrator::with_config(
            1,
            Box::new(dynamics),
            None,
            None,
            None,
            custom_config.clone(),
        );
        let config = integrator.config();

        // Verify the custom config was stored
        assert_eq!(config.abs_tol, 1e-10);
        assert_eq!(config.rel_tol, 1e-8);
        assert_eq!(config.max_step_attempts, 20);
        assert_eq!(config.min_step, Some(1e-15));
        assert_eq!(config.max_step, Some(100.0));
        assert_eq!(config.max_step_scale_factor, Some(5.0));
        assert_eq!(config.min_step_scale_factor, Some(0.1));
    }

    #[test]
    fn test_rk4d_config_returns_reference() {
        fn dynamics(_t: f64, state: DVector<f64>, _params: Option<&DVector<f64>>) -> DVector<f64> {
            state
        }

        let integrator = RK4DIntegrator::new(1, Box::new(dynamics), None, None, None);

        // Call config() multiple times and verify it returns the same values
        let config1 = integrator.config();
        let config2 = integrator.config();

        assert_eq!(config1.abs_tol, config2.abs_tol);
        assert_eq!(config1.rel_tol, config2.rel_tol);
        assert_eq!(config1.max_step_attempts, config2.max_step_attempts);
    }

    #[test]
    fn test_rk4d_dimension_method() {
        fn dynamics(_t: f64, state: DVector<f64>, _params: Option<&DVector<f64>>) -> DVector<f64> {
            state
        }

        let integrator = RK4DIntegrator::new(6, Box::new(dynamics), None, None, None);
        assert_eq!(integrator.dimension(), 6);

        let integrator2 = RK4DIntegrator::new(12, Box::new(dynamics), None, None, None);
        assert_eq!(integrator2.dimension(), 12);
    }
}
