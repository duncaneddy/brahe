/*!
Implementation of Runge-Kutta integration methods.
 */

use nalgebra::{SMatrix, SVector};

#[cfg(test)]
use crate::constants::RADIANS;
use crate::integrators::butcher_tableau::{ButcherTableau, RK4_TABLEAU};
use crate::integrators::numerical_integrator::NumericalIntegrator;

// Type aliases for complex function types
type StateDynamics<const S: usize> = Box<dyn Fn(f64, SVector<f64, S>) -> SVector<f64, S>>;
type VariationalMatrix<const S: usize> =
    Option<Box<dyn Fn(f64, SVector<f64, S>) -> SMatrix<f64, S, S>>>;

/// Implementation of the 4th order Runge-Kutta numerical integrator. This implementation is generic
/// over the size of the state vector.
///
/// # Example
///
/// ```
/// use nalgebra::{SVector, SMatrix};
/// use brahe::integrators::{RK4Integrator, NumericalIntegrator};
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
/// let rk4 = RK4Integrator::new(Box::new(f), None);
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
pub struct RK4Integrator<const S: usize> {
    f: StateDynamics<S>,
    varmat: VariationalMatrix<S>,
    bt: ButcherTableau<4>,
}

impl<const S: usize> RK4Integrator<S> {
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
    /// RK4Integrator instance ready for numerical integration
    pub fn new(f: StateDynamics<S>, varmat: VariationalMatrix<S>) -> Self {
        Self {
            f,
            varmat,
            bt: RK4_TABLEAU,
        }
    }
}

impl<const S: usize> NumericalIntegrator<S> for RK4Integrator<S> {
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
            k_phi[i] = self.varmat.as_ref().unwrap()(t + self.bt.c[i] * dt, state + dt * ksum)
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

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use crate::{
        Epoch, GM_EARTH, R_EARTH, TimeSystem, orbital_period, state_osculating_to_cartesian,
        varmat_from_fixed_offset, varmat_from_offset_vector,
    };

    use super::*;

    #[test]
    fn test_rk4_integrator_cubic() {
        // Define a simple function for testing x' = 2x,
        let f = |t: f64, _: SVector<f64, 1>| -> SVector<f64, 1> {
            let mut state_new = SVector::<f64, 1>::zeros();
            state_new[0] = 3.0 * t * t;
            state_new
        };

        let rk4 = RK4Integrator::new(Box::new(f), None);

        let mut state = SVector::<f64, 1>::new(0.0);
        let dt = 1.0;

        for i in 0..10 {
            state = rk4.step(i as f64, state, dt);
        }

        assert_abs_diff_eq!(state[0], 1000.0, epsilon = 1.0e-12);
    }

    #[test]
    fn test_rk4_integrator_parabola() {
        // Define a simple function for testing x' = 2x,
        let f = |t: f64, _: SVector<f64, 1>| -> SVector<f64, 1> {
            let mut state_new = SVector::<f64, 1>::zeros();
            state_new[0] = 2.0 * t;
            state_new
        };

        let rk4 = RK4Integrator::new(Box::new(f), None);

        let mut t = 0.0;
        let mut state = SVector::<f64, 1>::new(0.0);
        let dt = 0.01;

        for _ in 0..100 {
            state = rk4.step(t, state, dt);
            t += dt;
        }

        assert_abs_diff_eq!(state[0], 1.0, epsilon = 1.0e-12);
    }

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
    fn test_rk4_integrator_orbit() {
        let rk4 = RK4Integrator::new(Box::new(point_earth), None);

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
    fn test_rk4_integrator_varmat() {
        // Define how we want to calculate the variational matrix for the RK4 integrator
        let varmat = |t: f64, state: SVector<f64, 6>| -> SMatrix<f64, 6, 6> {
            varmat_from_fixed_offset(t, state, &point_earth, 1.0)
        };

        let rk4 = RK4Integrator::new(Box::new(point_earth), Some(Box::new(varmat)));

        // Get start state
        let oe0 = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 90.0, 0.0, 0.0, 0.0);
        let state0 = state_osculating_to_cartesian(oe0, RADIANS);
        let phi0 = SMatrix::<f64, 6, 6>::identity();

        // Take no setp and confirm the variational matrix is the identity matrix
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

        // Propagate one step and indecently confirm the variational matrix update
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
        let varmat = |t: f64, state: SVector<f64, 6>| -> SMatrix<f64, 6, 6> {
            varmat_from_offset_vector(
                t,
                state,
                &point_earth,
                SVector::<f64, 6>::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            )
        };
        let rk4 = RK4Integrator::new(Box::new(point_earth), Some(Box::new(varmat)));

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
}
