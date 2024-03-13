/*!
The numerical orbit propagator module provides a numerical integrator for propagating the state of a
system forward in time. This module provides the `NumericalOrbitPropagator` struct, which is the
concrete implementation of the propagator. This provides a high-acccuracy numerical simulation of
satellite motion that should be generally accurate over long periods. However, care should be taken
to ensure that the force model parameters and force model selection properly matches the use case.
If it does not, the code in the module provides a good starting point for developing a custom
propagator.
 */

use std::collections::BTreeMap;

use nalgebra::{SVector, Vector6};

use crate::{acceleration_drag, acceleration_gravity_spherical_harmonics, acceleration_relativity, acceleration_solar_radiation_pressure, acceleration_third_body_moon, acceleration_third_body_sun, bias_precession_nutation, density_harris_priester, earth_rotation, eclipse_conical, get_global_gravity_model, P_SUN, polar_motion, RK4Integrator, sun_position};
use crate::time::Epoch;

/// The `NumericalOrbitPropagatorParams` struct provides a set of parameters that can be used to
/// configure the numerical orbit propagator. This struct is used to configure tunable parameters
/// used by force models, along with enabling and disabling force models entirely.
#[derive(Debug, Clone)]
pub struct NumericalOrbitPropagatorParams {
    mass: f64,
    n_gravity: usize,
    m_gravity: usize,
    enable_drag: bool,
    drag_coefficient: f64,
    drag_area: f64,
    enable_srp: bool,
    srp_coefficient: f64,
    srp_area: f64,
    enable_third_body_sun: bool,
    enable_third_body_moon: bool,
    enable_relativity: bool,
    // TODO: Enable eclipse model selection
    // TODO: Enable atmospheric density model selection
}

/// The `NumericalOrbitPropagator` struct provides a numerical integrator for propagating the state
/// of a system forward in time. This provides a high-accuracy numerical simulation of satellite
/// motion that should be generally accurate over long periods. However, care should be taken to
/// ensure that the force model parameters and force model selection properly matches the use case.
/// The propagator state is the Cartesian position and velocity of the satellite in the GCRF, internal,
/// frame. The state is represented as a `SVector<f64, 6>`, where the first three elements are the
/// position and the last three elements are the velocity.
pub struct NumericalOrbitPropagator {
    initial_epoch: Epoch,
    initial_state: SVector<f64, 6>,
    last_epoch: Epoch,
    last_state: SVector<f64, 6>,
    final_epoch: Option<Epoch>,
    step_size: Option<f64>,
    last_step_size: Option<f64>,
    states: BTreeMap<Epoch, SVector<f64, 6>>,
    integrator: RK4Integrator<6>,
    params: NumericalOrbitPropagatorParams,
}

impl NumericalOrbitPropagator {
    /// Compute the state derivative at a given time and state. This is the function that effectively
    /// defines the force model for the numerical orbit propagator. The function should return the
    ///
    #[allow(non_snake_case)]
    fn deriv(&self, epc: Epoch, state: SVector<f64, 6>) -> SVector<f64, 6> {

        // Extract position and velocity
        let r = state.fixed_rows::<3>(0).into_owned();
        let v = state.fixed_rows::<3>(3).into_owned();

        // Compute components of ECI to ECEF transformation matrix
        // This is needed since different force models are defined in different frames
        // and the state is propagated in the GCRF frame
        let PN = bias_precession_nutation(epc);
        let E = earth_rotation(epc);
        let W = polar_motion(epc);
        let R = W * E * PN;

        // Compute position of the sun just in case it is needed
        let r_sun;
        if self.params.enable_drag || self.params.enable_srp {
            r_sun = sun_position(epc);
        } else {
            r_sun = SVector::<f64, 3>::zeros();
        }

        // Compute acceleration due to gravity

        // Initialize working vector containing acceleration due to gravity
        let mut a = SVector::<f64, 3>::zeros();

        // Compute acceleration due to gravity
        // This force cannot be disabled, however if set to 0x0 it will simulate a point mass
        // dynamics model, the simplest possible model
        a += acceleration_gravity_spherical_harmonics(r, PN, &get_global_gravity_model(), self.params.n_gravity, self.params.m_gravity);

        // Compute third-body gravitational perturbations
        if self.params.enable_third_body_sun {
            a += acceleration_third_body_sun(epc, r);
        }

        if self.params.enable_third_body_moon {
            a += acceleration_third_body_moon(epc, r);
        }

        // Compute acceleration due to drag
        if self.params.enable_drag {

            // TODO: Allow configuration of atmospheric density model
            let rho = density_harris_priester(r, r_sun);
            a += acceleration_drag(state, rho, self.params.mass, self.params.drag_area, self.params.drag_coefficient, PN);
        }

        // Compute acceleration due to solar radiation pressure
        if self.params.enable_srp {
            // TODO: Allow configuration of eclipse model
            let illum = eclipse_conical(r, r_sun);
            a += illum * acceleration_solar_radiation_pressure(r, r_sun, self.params.mass, self.params.srp_area, self.params.srp_coefficient, P_SUN);
        }

        if self.params.enable_relativity {
            a += acceleration_relativity(state);
        }

        // Return the state derivative
        Vector6::new(v[0], v[1], v[2], a[0], a[1], a[2])
    }
}