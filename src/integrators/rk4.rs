/*!
Implementation of the 4th order Runge-Kutta integration method.
 */

use nalgebra::{DMatrix, DVector, SMatrix, SVector};

use crate::integrators::butcher_tableau::{ButcherTableau, RK4_TABLEAU};
use crate::integrators::config::IntegratorConfig;
use crate::integrators::traits::{FixedStepDIntegrator, FixedStepSIntegrator};
use crate::math::jacobian::{DJacobianProvider, SJacobianProvider};

// Type aliases for complex function types
type StateDynamics<const S: usize> = Box<dyn Fn(f64, SVector<f64, S>) -> SVector<f64, S>>;
type VariationalMatrix<const S: usize> = Option<Box<dyn SJacobianProvider<S>>>;

/// Implementation of the 4th order Runge-Kutta numerical integrator. This implementation is generic
/// over the size of the state vector.
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
/// let rk4 = RK4SIntegrator::new(Box::new(f), None);
///
/// // Define the initial state and time step
/// let mut t = 0.0;
/// let mut state = SVector::<f64, 1>::new(0.0);
/// let dt = 0.01;
///
/// // Integrate the system forward in time to t = 1.0 (analytic solution is x = 1.0)
/// for i in 0..100{
///    state = rk4.step(t, state, dt);
///    t += dt;
/// }
///
/// assert!(state[0] - 1.0 < 1.0e-12);
///
/// // Now integrate the system forward in time to t = 10.0 (analytic solution is x = 100.0)
/// for i in 100..1000{
///     state = rk4.step(t, state, dt);
///     t += dt;
/// }
///
/// assert!(state[0] - 100.0 < 1.0e-12);
/// ```
pub struct RK4SIntegrator<const S: usize> {
    f: StateDynamics<S>,
    varmat: VariationalMatrix<S>,
    bt: ButcherTableau<4>,
    config: IntegratorConfig,
}

impl<const S: usize> RK4SIntegrator<S> {
    /// Create a new 4th-order Runge-Kutta integrator.
    ///
    /// Initializes RK4 integrator with classical Butcher tableau. Fourth-order accuracy
    /// provides good balance between accuracy and computational cost for most ODE systems.
    ///
    /// # Arguments
    /// - `f`: State derivative function defining the dynamics (closure or function pointer)
    /// - `varmat`: Variational matrix computation function for state transition matrix propagation
    ///
    /// # Returns
    /// RK4SIntegrator instance ready for numerical integration
    ///
    /// # Note
    /// This constructor provides backward compatibility. Uses default configuration.
    /// For custom configuration, use `with_config()`.
    pub fn new(f: StateDynamics<S>, varmat: VariationalMatrix<S>) -> Self {
        Self::with_config(f, varmat, IntegratorConfig::default())
    }

    /// Create a new 4th-order Runge-Kutta integrator with custom configuration.
    ///
    /// # Arguments
    /// - `f`: State derivative function defining the dynamics
    /// - `varmat`: Variational matrix computation function for STM propagation
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
    /// let rk4 = RK4SIntegrator::with_config(Box::new(f), None, config);
    /// ```
    pub fn with_config(
        f: StateDynamics<S>,
        varmat: VariationalMatrix<S>,
        config: IntegratorConfig,
    ) -> Self {
        Self {
            f,
            varmat,
            bt: RK4_TABLEAU,
            config,
        }
    }

    /// Get a reference to the integrator configuration.
    pub fn config(&self) -> &IntegratorConfig {
        &self.config
    }
}

impl<const S: usize> FixedStepSIntegrator<S> for RK4SIntegrator<S> {
    fn step(&self, t: f64, state: SVector<f64, S>, dt: f64) -> SVector<f64, S> {
        let mut k = SMatrix::<f64, S, 4>::zeros();
        let mut state_update = SVector::<f64, S>::zeros();

        // Compute internal steps based on the Butcher tableau
        for i in 0..4 {
            let mut ksum = SVector::<f64, S>::zeros();
            for j in 0..i {
                ksum += self.bt.a[(i, j)] * k.column(j);
            }

            k.set_column(i, &(self.f)(t + self.bt.c[i] * dt, state + dt * ksum));
        }

        // Compute the state update from each internal step
        for i in 0..4 {
            state_update += dt * self.bt.b[i] * k.column(i);
        }

        // Combine the state and the state update to get the new state
        state + state_update
    }

    fn step_with_varmat(
        &self,
        t: f64,
        state: SVector<f64, S>,
        phi: SMatrix<f64, S, S>,
        dt: f64,
    ) -> (SVector<f64, S>, SMatrix<f64, S, S>) {
        // Define working variables to hold internal step state
        let mut k = SMatrix::<f64, S, 4>::zeros();
        let mut k_phi = [SMatrix::<f64, S, S>::zeros(); 4];

        // Define working variables to hold the state and variational matrix updates
        let mut state_update = SVector::<f64, S>::zeros();
        let mut phi_update = SMatrix::<f64, S, S>::zeros();

        // Compute internal steps based on the Butcher tableau
        for i in 0..4 {
            let mut ksum = SVector::<f64, S>::zeros();
            let mut k_phi_sum = SMatrix::<f64, S, S>::zeros();

            for (j, k_phi_j) in k_phi.iter().enumerate().take(i) {
                ksum += self.bt.a[(i, j)] * k.column(j);
                k_phi_sum += self.bt.a[(i, j)] * k_phi_j;
            }

            k.set_column(i, &(self.f)(t + self.bt.c[i] * dt, state + dt * ksum));
            let state_i = state + dt * ksum;
            k_phi[i] = self
                .varmat
                .as_ref()
                .unwrap()
                .compute(t + self.bt.c[i] * dt, state_i)
                * (phi + dt * k_phi_sum);
        }

        // Compute the state update from each internal step
        for (i, k_phi_i) in k_phi.iter().enumerate().take(4) {
            state_update += dt * self.bt.b[i] * k.column(i);
            phi_update += dt * self.bt.b[i] * k_phi_i;
        }

        // Combine the state and the state update to get the new state
        (state + state_update, phi + phi_update)
    }
}

// ============================================================================
// Dynamic (runtime-sized) RK4 Integrator
// ============================================================================

// Type aliases for dynamic function types
type StateDynamicsD = Box<dyn Fn(f64, DVector<f64>) -> DVector<f64>>;
type VariationalMatrixD = Option<Box<dyn DJacobianProvider>>;

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
/// let f = |t: f64, state: DVector<f64>| -> DVector<f64> {
///     state.map(|x| -x)
/// };
///
/// // Create a new RK4 integrator for 2D system
/// let rk4 = RK4DIntegrator::new(2, Box::new(f), None);
///
/// // Define the initial state and time step
/// let mut t = 0.0;
/// let mut state = DVector::from_vec(vec![1.0, 2.0]);
/// let dt = 0.1;
///
/// // Integrate forward in time
/// for _ in 0..10 {
///     state = rk4.step(t, state, dt);
///     t += dt;
/// }
/// ```
pub struct RK4DIntegrator {
    dimension: usize,
    f: StateDynamicsD,
    varmat: VariationalMatrixD,
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
    /// let f = |t: f64, state: DVector<f64>| -> DVector<f64> {
    ///     DVector::from_vec(vec![state[1], -state[0]])  // Oscillator
    /// };
    ///
    /// let integrator = RK4DIntegrator::new(2, Box::new(f), None);
    /// ```
    pub fn new(dimension: usize, f: StateDynamicsD, varmat: VariationalMatrixD) -> Self {
        Self::with_config(dimension, f, varmat, IntegratorConfig::default())
    }

    /// Create a new 4th-order Runge-Kutta integrator with custom configuration.
    ///
    /// # Arguments
    /// - `dimension`: State vector dimension
    /// - `f`: State derivative function defining the dynamics
    /// - `varmat`: Optional variational matrix computation function
    /// - `config`: Integration configuration
    ///
    /// # Returns
    /// RK4DIntegrator instance with specified configuration
    pub fn with_config(
        dimension: usize,
        f: StateDynamicsD,
        varmat: VariationalMatrixD,
        config: IntegratorConfig,
    ) -> Self {
        Self {
            dimension,
            f,
            varmat,
            bt: RK4_TABLEAU,
            config,
        }
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

impl FixedStepDIntegrator for RK4DIntegrator {
    fn step(&self, t: f64, state: DVector<f64>, dt: f64) -> DVector<f64> {
        assert_eq!(
            state.len(),
            self.dimension,
            "State dimension {} doesn't match integrator dimension {}",
            state.len(),
            self.dimension
        );

        let mut k = DMatrix::<f64>::zeros(self.dimension, 4);
        let mut state_update = DVector::<f64>::zeros(self.dimension);

        // Compute internal steps based on the Butcher tableau
        for i in 0..4 {
            let mut ksum = DVector::<f64>::zeros(self.dimension);
            for j in 0..i {
                ksum += self.bt.a[(i, j)] * k.column(j);
            }

            k.set_column(i, &(self.f)(t + self.bt.c[i] * dt, &state + dt * ksum));
        }

        // Compute the state update from each internal step
        for i in 0..4 {
            state_update += dt * self.bt.b[i] * k.column(i);
        }

        // Combine the state and the state update to get the new state
        state + state_update
    }

    fn step_with_varmat(
        &self,
        t: f64,
        state: DVector<f64>,
        phi: DMatrix<f64>,
        dt: f64,
    ) -> (DVector<f64>, DMatrix<f64>) {
        assert_eq!(
            state.len(),
            self.dimension,
            "State dimension {} doesn't match integrator dimension {}",
            state.len(),
            self.dimension
        );
        assert_eq!(
            phi.nrows(),
            self.dimension,
            "STM rows {} doesn't match integrator dimension {}",
            phi.nrows(),
            self.dimension
        );
        assert_eq!(
            phi.ncols(),
            self.dimension,
            "STM cols {} doesn't match integrator dimension {}",
            phi.ncols(),
            self.dimension
        );

        // Define working variables to hold internal step state
        let mut k = DMatrix::<f64>::zeros(self.dimension, 4);
        let mut k_phi = vec![DMatrix::<f64>::zeros(self.dimension, self.dimension); 4];

        // Define working variables to hold the state and variational matrix updates
        let mut state_update = DVector::<f64>::zeros(self.dimension);
        let mut phi_update = DMatrix::<f64>::zeros(self.dimension, self.dimension);

        // Compute internal steps based on the Butcher tableau
        for i in 0..4 {
            let mut ksum = DVector::<f64>::zeros(self.dimension);
            let mut k_phi_sum = DMatrix::<f64>::zeros(self.dimension, self.dimension);

            #[allow(clippy::needless_range_loop)]
            for j in 0..i {
                ksum += self.bt.a[(i, j)] * k.column(j);
                k_phi_sum += self.bt.a[(i, j)] * &k_phi[j];
            }

            k.set_column(i, &(self.f)(t + self.bt.c[i] * dt, &state + dt * &ksum));
            let state_i = &state + dt * ksum;
            k_phi[i] = self
                .varmat
                .as_ref()
                .unwrap()
                .compute(t + self.bt.c[i] * dt, state_i)
                * (&phi + dt * k_phi_sum);
        }

        // Compute the state update from each internal step
        #[allow(clippy::needless_range_loop)]
        for i in 0..4 {
            state_update += dt * self.bt.b[i] * k.column(i);
            phi_update += dt * self.bt.b[i] * &k_phi[i];
        }

        // Combine the state and the state update to get the new state
        (state + state_update, phi + phi_update)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::{DMatrix, DVector, SMatrix, SVector};

    use crate::constants::RADIANS;
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

        let rk4 = RK4SIntegrator::new(Box::new(f), None);

        let mut state = SVector::<f64, 1>::new(0.0);
        let dt = 1.0;

        for i in 0..10 {
            state = rk4.step(i as f64, state, dt);
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

        let rk4 = RK4SIntegrator::new(Box::new(f), None);

        let mut t = 0.0;
        let mut state = SVector::<f64, 1>::new(0.0);
        let dt = 0.01;

        for _ in 0..100 {
            state = rk4.step(t, state, dt);
            t += dt;
        }

        assert_abs_diff_eq!(state[0], 1.0, epsilon = 1.0e-12);
    }

    #[test]
    fn test_rk4s_integrator_orbit() {
        let rk4 = RK4SIntegrator::new(Box::new(point_earth), None);

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
            state = rk4.step(epc - epc0, state, dt);
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

        let rk4 = RK4SIntegrator::new(Box::new(point_earth), Some(Box::new(jacobian)));

        // Get start state
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 90.0, 0.0, 0.0, 0.0);
        let state0 = state_osculating_to_cartesian(oe0, RADIANS);
        let phi0 = SMatrix::<f64, 6, 6>::identity();

        // Take no step and confirm the variational matrix is the identity matrix
        let (_, phi1) = rk4.step_with_varmat(0.0, state0, phi0, 0.0);
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
        let (_, phi2) = rk4.step_with_varmat(0.0, state0, phi0, 1.0);
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
        let rk4 = RK4SIntegrator::new(Box::new(point_earth), Some(Box::new(jacobian2)));

        // Get the state with a perturbation
        let (state_pert, _) = rk4.step_with_varmat(0.0, state0 + pert, phi0, 1.0);

        // Get the state with a perturbation by using the integrated variational matrix
        let state_stm = rk4.step(0.0, state0, 1.0) + phi2 * pert;

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

    fn point_earth_dynamic(_: f64, x: DVector<f64>) -> DVector<f64> {
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
        let f = |t: f64, _: DVector<f64>| -> DVector<f64> { DVector::from_vec(vec![3.0 * t * t]) };

        let rk4 = RK4DIntegrator::new(1, Box::new(f), None);

        let mut state = DVector::from_vec(vec![0.0]);
        let dt = 1.0;

        for i in 0..10 {
            state = rk4.step(i as f64, state, dt);
        }

        assert_abs_diff_eq!(state[0], 1000.0, epsilon = 1.0e-12);
    }

    #[test]
    fn test_rk4d_integrator_parabola() {
        // Define a simple function for testing x' = 2t
        let f = |t: f64, _: DVector<f64>| -> DVector<f64> { DVector::from_vec(vec![2.0 * t]) };

        let rk4 = RK4DIntegrator::new(1, Box::new(f), None);

        let mut t = 0.0;
        let mut state = DVector::from_vec(vec![0.0]);
        let dt = 0.01;

        for _ in 0..100 {
            state = rk4.step(t, state, dt);
            t += dt;
        }

        assert_abs_diff_eq!(state[0], 1.0, epsilon = 1.0e-12);
    }

    #[test]
    fn test_rk4d_integrator_orbit() {
        let rk4 = RK4DIntegrator::new(6, Box::new(point_earth_dynamic), None);

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
            state = rk4.step(epc - epc0, state, dt);
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
        // Define how we want to calculate the variational matrix for the RK4 integrator
        let jacobian =
            DNumericalJacobian::new(Box::new(point_earth_dynamic)).with_fixed_offset(1.0);

        let rk4 = RK4DIntegrator::new(6, Box::new(point_earth_dynamic), Some(Box::new(jacobian)));

        // Get start state
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 90.0, 0.0, 0.0, 0.0);
        let state0_static = state_osculating_to_cartesian(oe0, RADIANS);
        let state0 = DVector::from_vec(state0_static.as_slice().to_vec());
        let phi0 = DMatrix::<f64>::identity(6, 6);

        // Take no step and confirm the variational matrix is the identity matrix
        let (_, phi1) = rk4.step_with_varmat(0.0, state0.clone(), phi0.clone(), 0.0);
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
        let (_, phi2) = rk4.step_with_varmat(0.0, state0.clone(), phi0.clone(), 1.0);
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
        let jacobian2 =
            DNumericalJacobian::central(Box::new(point_earth_dynamic)).with_fixed_offset(1.0);
        let rk4 = RK4DIntegrator::new(6, Box::new(point_earth_dynamic), Some(Box::new(jacobian2)));

        // Get the state with a perturbation
        let (state_pert, _) = rk4.step_with_varmat(0.0, &state0 + &pert, phi0, 1.0);

        // Get the state with a perturbation by using the integrated variational matrix
        let state_stm = rk4.step(0.0, state0.clone(), 1.0) + &phi2 * &pert;

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
        let f_dynamic = |_t: f64, x: DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![-x[0], -x[1], -x[2]])
        };

        let rk4_s = RK4SIntegrator::new(Box::new(f_static), None);
        let rk4_d = RK4DIntegrator::new(3, Box::new(f_dynamic), None);

        let state_s = SVector::<f64, 3>::new(1.0, 2.0, 3.0);
        let state_d = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let dt = 0.1;

        let result_s = rk4_s.step(0.0, state_s, dt);
        let result_d = rk4_d.step(0.0, state_d, dt);

        // Results should be identical to machine precision
        assert_abs_diff_eq!(result_s[0], result_d[0], epsilon = 1.0e-15);
        assert_abs_diff_eq!(result_s[1], result_d[1], epsilon = 1.0e-15);
        assert_abs_diff_eq!(result_s[2], result_d[2], epsilon = 1.0e-15);
    }
}
