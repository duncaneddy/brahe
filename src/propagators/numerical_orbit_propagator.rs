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

use crate::constants::P_SUN;
use crate::frames::{bias_precession_nutation, earth_rotation, polar_motion};
use crate::integrators::RK4Integrator;
use crate::NumericalIntegrator;
use crate::orbit_dynamics::{acceleration_drag, acceleration_gravity_spherical_harmonics, acceleration_relativity, acceleration_solar_radiation_pressure, acceleration_third_body_moon, acceleration_third_body_sun, density_harris_priester, eclipse_conical, get_global_gravity_model, sun_position};
use crate::propagators::StatePropagator;
use crate::time::Epoch;
use crate::utils::BraheError;

fn create_deriv_function(first_epoch: Epoch, params: NumericalOrbitPropagatorParams) -> Box<dyn Fn(f64, SVector<f64, 6>) -> SVector<f64, 6>> {
    Box::new(move |t: f64, state: SVector<f64, 6>| -> SVector<f64, 6> {
        earth_orbit_deriv(first_epoch + t, state, params)
    })
}

/// The `NumericalOrbitPropagatorParams` struct provides a set of parameters that can be used to
/// configure the numerical orbit propagator. This struct is used to configure tunable parameters
/// used by force models, along with enabling and disabling force models entirely.
#[derive(Debug, Clone, Copy)]
pub struct NumericalOrbitPropagatorParams {
    pub mass: f64,
    pub n_gravity: usize,
    pub m_gravity: usize,
    pub enable_drag: bool,
    pub drag_coefficient: f64,
    pub drag_area: f64,
    pub enable_srp: bool,
    pub srp_coefficient: f64,
    pub srp_area: f64,
    pub enable_third_body_sun: bool,
    pub enable_third_body_moon: bool,
    pub enable_relativity: bool,
    // TODO: Enable eclipse model selection
    // TODO: Enable atmospheric density model selection
}

/// This is where the state derivative function is defined. This function is used by the integrator
/// to propagate the state forward in time. This function is the heart of the numerical orbit
/// propagator, and is where the force models are defined and applied to the state. When implementing
/// a custom propagator, this is the function that should be modified to change the force models and
/// their parameters.
#[allow(non_snake_case)]
fn earth_orbit_deriv(epc: Epoch, state: SVector<f64, 6>, params: NumericalOrbitPropagatorParams) -> SVector<f64, 6> {

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
    if params.enable_drag || params.enable_srp {
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
    a += acceleration_gravity_spherical_harmonics(r, R, &get_global_gravity_model(), params.n_gravity, params.m_gravity);

    // Compute third-body gravitational perturbations
    if params.enable_third_body_sun {
        a += acceleration_third_body_sun(epc, r);
    }

    if params.enable_third_body_moon {
        a += acceleration_third_body_moon(epc, r);
    }

    // Compute acceleration due to drag
    if params.enable_drag {

        // TODO: Allow configuration of atmospheric density model
        let rho = density_harris_priester(r, r_sun);
        a += acceleration_drag(state, rho, params.mass, params.drag_area, params.drag_coefficient, PN);
    }

    // Compute acceleration due to solar radiation pressure
    if params.enable_srp {
        // TODO: Allow configuration of eclipse model
        let illum = eclipse_conical(r, r_sun);
        a += illum * acceleration_solar_radiation_pressure(r, r_sun, params.mass, params.srp_area, params.srp_coefficient, P_SUN);
    }

    if params.enable_relativity {
        a += acceleration_relativity(state);
    }

    // Return the state derivative
    Vector6::new(v[0], v[1], v[2], a[0], a[1], a[2])
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
    last_step: Option<f64>,
    final_epoch: Option<Epoch>,
    step_size: Option<f64>,
    states: Vec<SVector<f64, 6>>,
    state: SVector<f64, 6>,
    epoch_index: BTreeMap<Epoch, usize>,
    integrator: RK4Integrator<6>,
    params: NumericalOrbitPropagatorParams,
}

impl NumericalOrbitPropagator {
    /// Create a new `NumericalOrbitPropagator` with the given initial epoch and state.
    ///
    /// # Arguments
    ///
    /// - `initial_epoch`: The initial epoch of the state.
    /// - `initial_state`: The initial state of the system, represented as a `SVector<f64, 6>` where
    ///    the elements are the Keplerian orbital elements. The elements are:
    ///     - `a`: Semi-major axis [m]
    ///     - `e`: Eccentricity
    ///     - `i`: Inclination [rad]
    ///     - `Ω`: Right ascension of the ascending node [rad]
    ///     - `ω`: Argument of periapsis [rad]
    ///     - `M`: Mean anomaly [rad]
    ///
    /// # Returns
    ///
    /// A new `NumericalOrbitPropagator` with the given initial epoch and state.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{NumericalOrbitPropagator, StatePropagator, NumericalOrbitPropagatorParams};
    /// use brahe::time::Epoch;
    /// use brahe::{R_EARTH, RAD2DEG, TimeSystem};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let params = NumericalOrbitPropagatorParams {
    ///    mass: 100.0,
    ///    n_gravity: 20,
    ///    m_gravity: 20,
    ///    enable_drag: true,
    ///    drag_coefficient: 2.3,
    ///    drag_area: 1.0,
    ///    enable_srp: true,
    ///    srp_coefficient: 1.8,
    ///    srp_area: 1.0,
    ///    enable_third_body_sun: true,
    ///    enable_third_body_moon: true,
    ///    enable_relativity: true,
    /// };
    ///
    /// let propagator = NumericalOrbitPropagator::new(epoch, state, params);
    /// ```
    pub fn new(initial_epoch: Epoch, initial_state: SVector<f64, 6>, params: NumericalOrbitPropagatorParams) -> Self {
        let mut states = Vec::new();
        let mut epoch_index = BTreeMap::new();
        states.push(initial_state.clone());
        epoch_index.insert(initial_epoch.clone(), 0);

        let f = create_deriv_function(initial_epoch, params);

        NumericalOrbitPropagator {
            initial_epoch,
            initial_state,
            last_epoch: initial_epoch,
            last_step: None,
            final_epoch: None,
            step_size: None,
            states,
            state: initial_state,
            epoch_index,
            integrator: RK4Integrator::new(f, None),
            params,
        }
    }
}

impl StatePropagator<6> for NumericalOrbitPropagator {
    /// Get the size of the state vector.
    ///
    /// # Returns
    ///
    /// The size of the state vector.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{NumericalOrbitPropagator, StatePropagator, NumericalOrbitPropagatorParams};
    /// use brahe::time::Epoch;
    /// use brahe::{R_EARTH, RAD2DEG, TimeSystem};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let params = NumericalOrbitPropagatorParams {
    ///    mass: 100.0,
    ///    n_gravity: 20,
    ///    m_gravity: 20,
    ///    enable_drag: true,
    ///    drag_coefficient: 2.3,
    ///    drag_area: 1.0,
    ///    enable_srp: true,
    ///    srp_coefficient: 1.8,
    ///    srp_area: 1.0,
    ///    enable_third_body_sun: true,
    ///    enable_third_body_moon: true,
    ///    enable_relativity: true,
    /// };
    ///
    /// let propagator = NumericalOrbitPropagator::new(epoch, state, params);
    ///
    /// assert_eq!(propagator.get_state_size(), 6);
    /// ```
    fn get_state_size(&self) -> usize {
        6
    }

    /// Get the number of states stored in the propagator.
    ///
    /// # Returns
    ///
    /// The number of states stored in the propagator.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{NumericalOrbitPropagator, StatePropagator, NumericalOrbitPropagatorParams};
    /// use brahe::time::Epoch;
    /// use brahe::{R_EARTH, RAD2DEG, TimeSystem};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let params = NumericalOrbitPropagatorParams {
    ///    mass: 100.0,
    ///    n_gravity: 20,
    ///    m_gravity: 20,
    ///    enable_drag: true,
    ///    drag_coefficient: 2.3,
    ///    drag_area: 1.0,
    ///    enable_srp: true,
    ///    srp_coefficient: 1.8,
    ///    srp_area: 1.0,
    ///    enable_third_body_sun: true,
    ///    enable_third_body_moon: true,
    ///    enable_relativity: true,
    /// };
    ///
    /// let propagator = NumericalOrbitPropagator::new(epoch, state, params);
    ///
    /// assert_eq!(propagator.get_num_states(), 1);
    /// ```
    fn get_num_states(&self) -> usize {
        return self.states.len();
    }

    /// Get the initial state of the propagator.
    ///
    /// # Returns
    ///
    /// The initial state of the propagator.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{NumericalOrbitPropagator, StatePropagator, NumericalOrbitPropagatorParams};
    /// use brahe::time::Epoch;
    /// use brahe::{R_EARTH, RAD2DEG, TimeSystem};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let params = NumericalOrbitPropagatorParams {
    ///    mass: 100.0,
    ///    n_gravity: 20,
    ///    m_gravity: 20,
    ///    enable_drag: true,
    ///    drag_coefficient: 2.3,
    ///    drag_area: 1.0,
    ///    enable_srp: true,
    ///    srp_coefficient: 1.8,
    ///    srp_area: 1.0,
    ///    enable_third_body_sun: true,
    ///    enable_third_body_moon: true,
    ///    enable_relativity: true,
    /// };
    ///
    /// let propagator = NumericalOrbitPropagator::new(epoch, state, params);
    ///
    /// assert_eq!(*propagator.get_initial_state(), state);
    /// ```
    fn get_initial_state(&self) -> &SVector<f64, 6> {
        &self.initial_state
    }

    /// Get the initial epoch of the propagator.
    ///
    /// # Returns
    ///
    /// The initial epoch of the propagator.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{NumericalOrbitPropagator, StatePropagator, NumericalOrbitPropagatorParams};
    /// use brahe::time::Epoch;
    /// use brahe::{R_EARTH, RAD2DEG, TimeSystem};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let params = NumericalOrbitPropagatorParams {
    ///    mass: 100.0,
    ///    n_gravity: 20,
    ///    m_gravity: 20,
    ///    enable_drag: true,
    ///    drag_coefficient: 2.3,
    ///    drag_area: 1.0,
    ///    enable_srp: true,
    ///    srp_coefficient: 1.8,
    ///    srp_area: 1.0,
    ///    enable_third_body_sun: true,
    ///    enable_third_body_moon: true,
    ///    enable_relativity: true,
    /// };
    ///
    /// let propagator = NumericalOrbitPropagator::new(epoch, state, params);
    ///
    /// assert_eq!(propagator.get_initial_epoch(), epoch);
    /// ```
    fn get_initial_epoch(&self) -> Epoch {
        self.initial_epoch
    }

    /// Get the last step size of the propagator. This is the last time step used to propagate the
    /// state forward in time. If the state has not been propagated, this will return `None`.
    ///
    /// # Returns
    ///
    /// The last step size of the propagator, or `None` if the state has not been propagated.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{NumericalOrbitPropagator, StatePropagator, NumericalOrbitPropagatorParams};
    /// use brahe::time::Epoch;
    /// use brahe::{R_EARTH, RAD2DEG, TimeSystem};
    /// use brahe::eop::{FileEOPProvider, EarthOrientationProvider, EOPExtrapolation, set_global_eop_provider};
    /// use brahe::orbit_dynamics::gravity::{DefaultGravityModel, GravityModel, set_global_gravity_model};
    ///
    /// // Initialize required global data providers
    /// set_global_eop_provider(FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap());
    /// set_global_gravity_model(GravityModel::from_default(DefaultGravityModel::EGM2008_360));
    ///
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let params = NumericalOrbitPropagatorParams {
    ///    mass: 100.0,
    ///    n_gravity: 20,
    ///    m_gravity: 20,
    ///    enable_drag: true,
    ///    drag_coefficient: 2.3,
    ///    drag_area: 1.0,
    ///    enable_srp: true,
    ///    srp_coefficient: 1.8,
    ///    srp_area: 1.0,
    ///    enable_third_body_sun: true,
    ///    enable_third_body_moon: true,
    ///    enable_relativity: true,
    /// };
    ///
    /// let mut propagator = NumericalOrbitPropagator::new(epoch, state, params);
    ///
    /// assert_eq!(propagator.get_last_step_size(), None);
    ///
    /// propagator.set_step_size(60.0).unwrap();
    /// propagator.step().unwrap();
    /// assert_eq!(propagator.get_last_step_size(), Some(60.0));
    /// ```
    fn get_last_step_size(&self) -> Option<f64> {
        self.last_step
    }

    /// Get the last epoch of the propagator. This is the epoch of the last state that was propagated.
    /// For a newly created propagator, this will be the same as the initial epoch.
    ///
    /// # Returns
    ///
    /// The last epoch of the propagator.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{NumericalOrbitPropagator, StatePropagator, NumericalOrbitPropagatorParams};
    /// use brahe::time::Epoch;
    /// use brahe::{R_EARTH, RAD2DEG, TimeSystem};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let params = NumericalOrbitPropagatorParams {
    ///    mass: 100.0,
    ///    n_gravity: 20,
    ///    m_gravity: 20,
    ///    enable_drag: true,
    ///    drag_coefficient: 2.3,
    ///    drag_area: 1.0,
    ///    enable_srp: true,
    ///    srp_coefficient: 1.8,
    ///    srp_area: 1.0,
    ///    enable_third_body_sun: true,
    ///    enable_third_body_moon: true,
    ///    enable_relativity: true,
    /// };
    ///
    /// let propagator = NumericalOrbitPropagator::new(epoch, state, params);
    ///
    /// assert_eq!(propagator.get_last_epoch(), epoch);
    /// ```
    fn get_last_epoch(&self) -> Epoch {
        self.last_epoch
    }

    /// Get the last state of the propagator. This is the state of the system at the last epoch that was
    /// propagated. For a newly created propagator, this will be the same as the initial state.
    ///
    /// # Returns
    ///
    /// The last state of the propagator.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{NumericalOrbitPropagator, StatePropagator, NumericalOrbitPropagatorParams};
    /// use brahe::time::Epoch;
    /// use brahe::{R_EARTH, RAD2DEG, TimeSystem};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let params = NumericalOrbitPropagatorParams {
    ///    mass: 100.0,
    ///    n_gravity: 20,
    ///    m_gravity: 20,
    ///    enable_drag: true,
    ///    drag_coefficient: 2.3,
    ///    drag_area: 1.0,
    ///    enable_srp: true,
    ///    srp_coefficient: 1.8,
    ///    srp_area: 1.0,
    ///    enable_third_body_sun: true,
    ///    enable_third_body_moon: true,
    ///    enable_relativity: true,
    /// };
    ///
    /// let propagator = NumericalOrbitPropagator::new(epoch, state, params);
    ///
    /// assert_eq!(*propagator.get_last_state(), state);
    /// ```
    fn get_last_state(&self) -> &SVector<f64, 6> {
        // This is safe because we always have at least one state in the vector, the initial state
        self.states.last().unwrap()
    }

    /// Get the final epoch of the propagator. This is the epoch to which the propagator will propagate
    /// the state. If the final epoch has not been set, this will return `None`.
    ///
    /// # Returns
    ///
    /// The final epoch of the propagator, or `None` if the final epoch has not been set.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{NumericalOrbitPropagator, StatePropagator, NumericalOrbitPropagatorParams};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::{R_EARTH, RAD2DEG};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let params = NumericalOrbitPropagatorParams {
    ///    mass: 100.0,
    ///    n_gravity: 20,
    ///    m_gravity: 20,
    ///    enable_drag: true,
    ///    drag_coefficient: 2.3,
    ///    drag_area: 1.0,
    ///    enable_srp: true,
    ///    srp_coefficient: 1.8,
    ///    srp_area: 1.0,
    ///    enable_third_body_sun: true,
    ///    enable_third_body_moon: true,
    ///    enable_relativity: true,
    /// };
    ///
    /// let mut propagator = NumericalOrbitPropagator::new(epoch, state, params);
    ///
    /// assert_eq!(propagator.get_final_epoch(), None);
    ///
    /// propagator.set_final_epoch(Epoch::from_datetime(2024, 1, 2, 12, 0, 0.0, 0.0, TimeSystem::UTC));
    /// assert_eq!(propagator.get_final_epoch(), Some(Epoch::from_datetime(2024, 1, 2, 12, 0, 0.0, 0.0, TimeSystem::UTC)));
    /// ```
    fn get_final_epoch(&self) -> Option<Epoch> {
        self.final_epoch
    }

    /// Get the step size of the propagator. The step size is the time step used to propagate the state
    /// forward in time. The step size is in seconds. If the step size has not been set, this will
    /// return `None`.
    ///
    /// # Returns
    ///
    /// The step size of the propagator, or `None` if the step size has not been set.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{NumericalOrbitPropagator, StatePropagator, NumericalOrbitPropagatorParams};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::{R_EARTH, RAD2DEG};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let params = NumericalOrbitPropagatorParams {
    ///    mass: 100.0,
    ///    n_gravity: 20,
    ///    m_gravity: 20,
    ///    enable_drag: true,
    ///    drag_coefficient: 2.3,
    ///    drag_area: 1.0,
    ///    enable_srp: true,
    ///    srp_coefficient: 1.8,
    ///    srp_area: 1.0,
    ///    enable_third_body_sun: true,
    ///    enable_third_body_moon: true,
    ///    enable_relativity: true,
    /// };
    ///
    /// let mut propagator = NumericalOrbitPropagator::new(epoch, state, params);
    ///
    /// assert_eq!(propagator.get_step_size(), None);
    ///
    /// propagator.set_step_size(60.0).unwrap();
    /// assert_eq!(propagator.get_step_size(), Some(60.0));
    /// ```
    fn get_step_size(&self) -> Option<f64> {
        self.step_size
    }

    /// Get the state at the given index. If the index is out of range, this will return `None`.
    ///
    /// # Arguments
    ///
    /// - `index`: The index of the state to retrieve.
    ///
    /// # Returns
    ///
    /// The state at the given index, or `None` if the index is out of range.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{NumericalOrbitPropagator, StatePropagator, NumericalOrbitPropagatorParams};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::{R_EARTH, RAD2DEG};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let params = NumericalOrbitPropagatorParams {
    ///    mass: 100.0,
    ///    n_gravity: 20,
    ///    m_gravity: 20,
    ///    enable_drag: true,
    ///    drag_coefficient: 2.3,
    ///    drag_area: 1.0,
    ///    enable_srp: true,
    ///    srp_coefficient: 1.8,
    ///    srp_area: 1.0,
    ///    enable_third_body_sun: true,
    ///    enable_third_body_moon: true,
    ///    enable_relativity: true,
    /// };
    ///
    /// let mut propagator = NumericalOrbitPropagator::new(epoch, state, params);
    ///
    /// assert_eq!(propagator.get_state_by_index(0), Some(&state));
    /// assert_eq!(propagator.get_state_by_index(1), None);
    /// ```
    fn get_state_by_index(&self, index: usize) -> Option<&SVector<f64, 6>> {
        if index < self.states.len() {
            Some(&self.states[index])
        } else {
            None
        }
    }

    /// Get the state at the given epoch. If the epoch is not in the propagator, this will return `None`.
    ///
    /// # Arguments
    ///
    /// - `epoch`: The epoch of the state to retrieve.
    ///
    /// # Returns
    ///
    /// The state at the given epoch, or `None` if the epoch is not in the propagator.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{NumericalOrbitPropagator, StatePropagator, NumericalOrbitPropagatorParams};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::{R_EARTH, RAD2DEG};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let params = NumericalOrbitPropagatorParams {
    ///    mass: 100.0,
    ///    n_gravity: 20,
    ///    m_gravity: 20,
    ///    enable_drag: true,
    ///    drag_coefficient: 2.3,
    ///    drag_area: 1.0,
    ///    enable_srp: true,
    ///    srp_coefficient: 1.8,
    ///    srp_area: 1.0,
    ///    enable_third_body_sun: true,
    ///    enable_third_body_moon: true,
    ///    enable_relativity: true,
    /// };
    ///
    /// let mut propagator = NumericalOrbitPropagator::new(epoch, state, params);
    ///
    /// assert_eq!(propagator.get_state_by_epoch(epoch), Some(&state));
    /// assert_eq!(propagator.get_state_by_epoch(Epoch::from_datetime(2024, 1, 2, 12, 0, 0.0, 0.0, TimeSystem::UTC)), None);
    /// ```
    fn get_state_by_epoch(&self, epoch: Epoch) -> Option<&SVector<f64, 6>> {
        if let Some(index) = self.epoch_index.get(&epoch) {
            self.get_state_by_index(*index)
        } else {
            // TODO: This could be improved by implementing interpolation and returning the interpolated state
            // instead of None
            None
        }
    }

    /// Set the final epoch of the propagator. This is the epoch to which the propagator will propagate
    /// the state.
    ///
    /// # Arguments
    ///
    /// - `epoch`: The final epoch of the propagator.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success (`Ok`) or an error (`Err`).
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{NumericalOrbitPropagator, StatePropagator, NumericalOrbitPropagatorParams};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::{R_EARTH, RAD2DEG};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let params = NumericalOrbitPropagatorParams {
    ///    mass: 100.0,
    ///    n_gravity: 20,
    ///    m_gravity: 20,
    ///    enable_drag: true,
    ///    drag_coefficient: 2.3,
    ///    drag_area: 1.0,
    ///    enable_srp: true,
    ///    srp_coefficient: 1.8,
    ///    srp_area: 1.0,
    ///    enable_third_body_sun: true,
    ///    enable_third_body_moon: true,
    ///    enable_relativity: true,
    /// };
    ///
    /// let mut propagator = NumericalOrbitPropagator::new(epoch, state, params);
    ///
    /// assert_eq!(propagator.get_final_epoch(), None);
    ///
    /// propagator.set_final_epoch(Epoch::from_datetime(2024, 1, 2, 12, 0, 0.0, 0.0, TimeSystem::UTC));
    /// assert_eq!(propagator.get_final_epoch(), Some(Epoch::from_datetime(2024, 1, 2, 12, 0, 0.0, 0.0, TimeSystem::UTC)));
    /// ```
    fn set_final_epoch(&mut self, epoch: Epoch) -> Result<(), BraheError> {
        if let Some(step_size) = self.step_size {
            // Confirm that the step size is in the same direction as the final epoch
            if (epoch - self.initial_epoch).signum() == step_size.signum() {
                self.final_epoch = Some(epoch);
            } else {
                return Err(BraheError::InitializationError("The final epoch is in the opposite direction of the step size".to_string()));
            }
        } else {
            // If the step size has not been set, simply set the final epoch
            self.final_epoch = Some(epoch);
        }

        Ok(())
    }

    /// Set the step size of the propagator. The step size is the time step used to propagate the state
    /// forward in time. The step size is in seconds.
    ///
    /// If the step size is set to a positive value, the propagator will propagate the state forward in
    /// time. If the step size is set to a negative value, the propagator will propagate the state
    /// backward in time.
    ///
    /// The step size cannot be set to zero.
    ///
    /// If the final epoch has been set, the step size must be in the same direction as the final epoch.
    ///
    /// # Arguments
    ///
    /// - `step_size`: The step size of the propagator, in seconds.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success (`Ok`) or an error (`Err`).
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{NumericalOrbitPropagator, StatePropagator, NumericalOrbitPropagatorParams};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::{R_EARTH, RAD2DEG};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let params = NumericalOrbitPropagatorParams {
    ///    mass: 100.0,
    ///    n_gravity: 20,
    ///    m_gravity: 20,
    ///    enable_drag: true,
    ///    drag_coefficient: 2.3,
    ///    drag_area: 1.0,
    ///    enable_srp: true,
    ///    srp_coefficient: 1.8,
    ///    srp_area: 1.0,
    ///    enable_third_body_sun: true,
    ///    enable_third_body_moon: true,
    ///    enable_relativity: true,
    /// };
    ///
    /// let mut propagator = NumericalOrbitPropagator::new(epoch, state, params);
    ///
    /// assert_eq!(propagator.get_step_size(), None);
    ///
    /// propagator.set_step_size(60.0).unwrap();
    /// assert_eq!(propagator.get_step_size(), Some(60.0));
    /// ```
    fn set_step_size(&mut self, step_size: f64) -> Result<(), BraheError> {
        if step_size == 0.0 {
            return Err(BraheError::InitializationError("The step size cannot be zero".to_string()));
        }

        if let Some(final_epoch) = self.final_epoch {
            // Confirm that the step size is in the same direction as the final epoch
            if (final_epoch - self.initial_epoch).signum() == step_size.signum() {
                self.step_size = Some(step_size);
            } else {
                return Err(BraheError::InitializationError("The step size is in the opposite direction of the final epoch".to_string()));
            }
        } else {
            // If the final epoch has not been set, simply set the step size
            self.step_size = Some(step_size);
        }

        Ok(())
    }

    /// Reinitialize the propagator. This will reset the propagator to its initial state and epoch
    /// and clear all stored states.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{NumericalOrbitPropagator, StatePropagator, NumericalOrbitPropagatorParams};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::{R_EARTH, RAD2DEG};
    /// use brahe::eop::{FileEOPProvider, EarthOrientationProvider, EOPExtrapolation, set_global_eop_provider};
    /// use brahe::orbit_dynamics::gravity::{DefaultGravityModel, GravityModel, set_global_gravity_model};
    ///
    /// // Initialize required global data providers
    /// set_global_eop_provider(FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap());
    /// set_global_gravity_model(GravityModel::from_default(DefaultGravityModel::EGM2008_360));
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let params = NumericalOrbitPropagatorParams {
    ///    mass: 100.0,
    ///    n_gravity: 20,
    ///    m_gravity: 20,
    ///    enable_drag: true,
    ///    drag_coefficient: 2.3,
    ///    drag_area: 1.0,
    ///    enable_srp: true,
    ///    srp_coefficient: 1.8,
    ///    srp_area: 1.0,
    ///    enable_third_body_sun: true,
    ///    enable_third_body_moon: true,
    ///    enable_relativity: true,
    /// };
    ///
    /// let mut propagator = NumericalOrbitPropagator::new(epoch, state, params);
    ///
    /// propagator.step_by(60.0);
    /// assert_eq!(propagator.get_num_states(), 2);
    ///
    /// propagator.reinitialize();
    /// assert_eq!(propagator.get_num_states(), 1);
    /// ```
    fn reinitialize(&mut self) {
        self.last_epoch = self.initial_epoch.clone();
        self.states.truncate(1);
        self.epoch_index.clear();
        self.epoch_index.insert(self.initial_epoch.clone(), 0);
    }

    /// Propagate the state forward in time by one step size. If the step size has not been set, this
    /// will return an error.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success (`Ok`) or an error (`Err`).
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{NumericalOrbitPropagator, StatePropagator, NumericalOrbitPropagatorParams};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::{R_EARTH, RAD2DEG};
    /// use brahe::eop::{FileEOPProvider, EarthOrientationProvider, EOPExtrapolation, set_global_eop_provider};
    /// use brahe::orbit_dynamics::gravity::{DefaultGravityModel, GravityModel, set_global_gravity_model};
    ///
    /// // Initialize required global data providers
    /// set_global_eop_provider(FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap());
    /// set_global_gravity_model(GravityModel::from_default(DefaultGravityModel::EGM2008_360));
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let params = NumericalOrbitPropagatorParams {
    ///    mass: 100.0,
    ///    n_gravity: 20,
    ///    m_gravity: 20,
    ///    enable_drag: true,
    ///    drag_coefficient: 2.3,
    ///    drag_area: 1.0,
    ///    enable_srp: true,
    ///    srp_coefficient: 1.8,
    ///    srp_area: 1.0,
    ///    enable_third_body_sun: true,
    ///    enable_third_body_moon: true,
    ///    enable_relativity: true,
    /// };
    ///
    /// let mut propagator = NumericalOrbitPropagator::new(epoch, state, params);
    ///
    /// propagator.set_step_size(60.0);
    /// propagator.step().unwrap();
    /// ```
    fn step(&mut self) -> Result<(), BraheError> {
        if let Some(step_size) = self.step_size {
            let new_epoch = self.last_epoch + step_size;
            let t = self.last_epoch - self.initial_epoch;
            let x = self.integrator.step(t, self.state, step_size);
            self.state = x;
            self.states.push(x);
            self.epoch_index.insert(new_epoch, self.states.len() - 1);
            self.last_epoch = new_epoch;
            self.last_step = Some(step_size);
            Ok(())
        } else {
            Err(BraheError::InitializationError("The step size has not been set".to_string()))
        }
    }

    /// Propagate the state forward in time by the given time step. If the final epoch has been set,
    /// the step must be in the same direction as the final epoch, otherwise this will return an error.
    ///
    /// # Arguments
    ///
    /// - `dt`: The time step by which to propagate the state forward in time.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success (`Ok`) or an error (`Err`).
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{NumericalOrbitPropagator, StatePropagator, NumericalOrbitPropagatorParams};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::{R_EARTH, RAD2DEG};
    /// use brahe::eop::{FileEOPProvider, EarthOrientationProvider, EOPExtrapolation, set_global_eop_provider};
    /// use brahe::orbit_dynamics::gravity::{DefaultGravityModel, GravityModel, set_global_gravity_model};
    ///
    /// // Initialize required global data providers
    /// set_global_eop_provider(FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap());
    /// set_global_gravity_model(GravityModel::from_default(DefaultGravityModel::EGM2008_360));
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let params = NumericalOrbitPropagatorParams {
    ///    mass: 100.0,
    ///    n_gravity: 20,
    ///    m_gravity: 20,
    ///    enable_drag: true,
    ///    drag_coefficient: 2.3,
    ///    drag_area: 1.0,
    ///    enable_srp: true,
    ///    srp_coefficient: 1.8,
    ///    srp_area: 1.0,
    ///    enable_third_body_sun: true,
    ///    enable_third_body_moon: true,
    ///    enable_relativity: true,
    /// };
    ///
    /// let mut propagator = NumericalOrbitPropagator::new(epoch, state, params);
    ///
    /// propagator.step_by(60.0).unwrap();
    /// ```
    fn step_by(&mut self, dt: f64) -> Result<(), BraheError> {
        if let Some(final_epoch) = self.final_epoch {
            // Confirm that the step size is in the same direction as the final epoch
            if (final_epoch - self.initial_epoch).signum() == dt.signum() {
                let epc = self.last_epoch + dt;
                let t = self.last_epoch - self.initial_epoch;
                let x = self.integrator.step(t, self.state, dt);
                self.state = x;
                self.states.push(x);
                self.epoch_index.insert(epc, self.states.len() - 1);
                self.last_epoch = epc;
                self.last_step = Some(dt);
                Ok(())
            } else {
                Err(BraheError::PropagatorError("The provided step is in the opposite direction of the final epoch".to_string()))
            }
        } else {
            let epc = self.last_epoch + dt;
            let t = self.last_epoch - self.initial_epoch;
            let x = self.integrator.step(t, self.state, dt);
            self.state = x;
            self.states.push(x);
            self.epoch_index.insert(epc, self.states.len() - 1);
            self.last_epoch = epc;
            self.last_step = Some(dt);
            Ok(())
        }
    }

    /// Propagate the state forward in time to the given epoch. If the final epoch has been set, the
    /// epoch must be in the same direction as the final epoch, otherwise this will return an error.
    /// Requires that the step size has been set. If the final epoch is not an integer multiple of the
    /// step size, the final step will be less than the step size.
    ///
    /// # Arguments
    ///
    /// - `epoch`: The epoch to which to propagate the state forward in time.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success (`Ok`) or an error (`Err`).
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{NumericalOrbitPropagator, StatePropagator, NumericalOrbitPropagatorParams};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::{R_EARTH, RAD2DEG};
    /// use brahe::eop::{FileEOPProvider, EarthOrientationProvider, EOPExtrapolation, set_global_eop_provider};
    /// use brahe::orbit_dynamics::gravity::{DefaultGravityModel, GravityModel, set_global_gravity_model};
    ///
    /// // Initialize required global data providers
    /// set_global_eop_provider(FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap());
    /// set_global_gravity_model(GravityModel::from_default(DefaultGravityModel::EGM2008_360));
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let params = NumericalOrbitPropagatorParams {
    ///    mass: 100.0,
    ///    n_gravity: 20,
    ///    m_gravity: 20,
    ///    enable_drag: true,
    ///    drag_coefficient: 2.3,
    ///    drag_area: 1.0,
    ///    enable_srp: true,
    ///    srp_coefficient: 1.8,
    ///    srp_area: 1.0,
    ///    enable_third_body_sun: true,
    ///    enable_third_body_moon: true,
    ///    enable_relativity: true,
    /// };
    ///
    /// let mut propagator = NumericalOrbitPropagator::new(epoch, state, params);
    ///
    /// propagator.set_step_size(60.0);
    /// propagator.step_to_epoch(Epoch::from_datetime(2024, 1, 1, 12, 3, 30.0, 0.0, TimeSystem::UTC)).unwrap();
    ///
    /// assert_eq!(propagator.get_last_epoch(), Epoch::from_datetime(2024, 1, 1, 12, 3, 30.0, 0.0, TimeSystem::UTC));
    /// assert_eq!(propagator.get_num_states(), 6);
    /// ```
    fn step_to_epoch(&mut self, epoch: Epoch) -> Result<(), BraheError> {
        if let Some(step_size) = self.step_size {
            // Confirm that the step size is in the same direction as the final epoch
            if (epoch - self.last_epoch).signum() == step_size.signum() {
                while self.last_epoch < epoch {
                    self.step_by(step_size.min(epoch - self.last_epoch))?
                }
                Ok(())
            } else {
                Err(BraheError::PropagatorError("The provided epoch is in the opposite direction of the step size".to_string()))
            }
        } else {
            Err(BraheError::InitializationError("The step size has not been set".to_string()))
        }
    }

    /// Propagate the state forward in time to the final epoch. Will return an error if the final epoch
    /// has not been set. Requires that the step size has been set. If the final epoch is not an integer
    /// multiple of the step size, the final step will be less than the step size.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success (`Ok`) or an error (`Err`).
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{NumericalOrbitPropagator, StatePropagator, NumericalOrbitPropagatorParams};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::{R_EARTH, RAD2DEG};
    /// use brahe::eop::{FileEOPProvider, EarthOrientationProvider, EOPExtrapolation, set_global_eop_provider};
    /// use brahe::orbit_dynamics::gravity::{DefaultGravityModel, GravityModel, set_global_gravity_model};
    ///
    /// // Initialize required global data providers
    /// set_global_eop_provider(FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap());
    /// set_global_gravity_model(GravityModel::from_default(DefaultGravityModel::EGM2008_360));
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let params = NumericalOrbitPropagatorParams {
    ///    mass: 100.0,
    ///    n_gravity: 20,
    ///    m_gravity: 20,
    ///    enable_drag: true,
    ///    drag_coefficient: 2.3,
    ///    drag_area: 1.0,
    ///    enable_srp: true,
    ///    srp_coefficient: 1.8,
    ///    srp_area: 1.0,
    ///    enable_third_body_sun: true,
    ///    enable_third_body_moon: true,
    ///    enable_relativity: true,
    /// };
    ///
    /// let mut propagator = NumericalOrbitPropagator::new(epoch, state, params);
    ///
    /// propagator.set_step_size(60.0);
    /// propagator.set_final_epoch(Epoch::from_datetime(2024, 1, 1, 12, 3, 30.0, 0.0, TimeSystem::UTC)).unwrap();
    /// propagator.step_to_final_epoch().unwrap();
    ///
    /// assert_eq!(propagator.get_last_epoch(), Epoch::from_datetime(2024, 1, 1, 12, 3, 30.0, 0.0, TimeSystem::UTC));
    /// assert_eq!(propagator.get_num_states(), 6);
    /// ```
    fn step_to_final_epoch(&mut self) -> Result<(), BraheError> {
        if let Some(final_epoch) = self.final_epoch {
            self.step_to_epoch(final_epoch)?;
            Ok(())
        } else {
            Err(BraheError::InitializationError("The final epoch has not been set".to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use approx::assert_abs_diff_eq;

    use crate::{NumericalOrbitPropagator, orbital_period, R_EARTH, RAD2DEG, state_osculating_to_cartesian};
    use crate::time::TimeSystem;
    use crate::utils::testing::{setup_global_test_eop, setup_global_test_gravity_model};

    use super::*;

    #[test]
    fn test_numerical_propagator_new() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0, 15.0, 30.0, 0.0);
        let params = NumericalOrbitPropagatorParams {
            mass: 100.0,
            n_gravity: 20,
            m_gravity: 20,
            enable_drag: true,
            drag_coefficient: 2.3,
            drag_area: 1.0,
            enable_srp: true,
            srp_coefficient: 1.8,
            srp_area: 1.0,
            enable_third_body_sun: true,
            enable_third_body_moon: true,
            enable_relativity: true,
        };
        let propagator = NumericalOrbitPropagator::new(epoch, state, params);

        assert_eq!(propagator.get_initial_epoch(), epoch);
        assert_eq!(propagator.get_initial_state(), &state);
        assert_eq!(propagator.get_last_epoch(), epoch);
        assert_eq!(propagator.get_final_epoch(), None);
        assert_eq!(propagator.get_num_states(), 1);
    }

    #[test]
    fn test_numerical_propagator_get_state_size() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0, 15.0, 30.0, 0.0);
        let params = NumericalOrbitPropagatorParams {
            mass: 100.0,
            n_gravity: 20,
            m_gravity: 20,
            enable_drag: true,
            drag_coefficient: 2.3,
            drag_area: 1.0,
            enable_srp: true,
            srp_coefficient: 1.8,
            srp_area: 1.0,
            enable_third_body_sun: true,
            enable_third_body_moon: true,
            enable_relativity: true,
        };
        let propagator = NumericalOrbitPropagator::new(epoch, state, params);

        assert_eq!(propagator.get_state_size(), 6);
    }

    #[test]
    fn test_numerical_propagator_get_num_states() {
        setup_global_test_eop();
        setup_global_test_gravity_model();

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0, 15.0, 30.0, 0.0);
        let params = NumericalOrbitPropagatorParams {
            mass: 100.0,
            n_gravity: 20,
            m_gravity: 20,
            enable_drag: true,
            drag_coefficient: 2.3,
            drag_area: 1.0,
            enable_srp: true,
            srp_coefficient: 1.8,
            srp_area: 1.0,
            enable_third_body_sun: true,
            enable_third_body_moon: true,
            enable_relativity: true,
        };
        let mut propagator = NumericalOrbitPropagator::new(epoch, state, params);

        assert_eq!(propagator.get_num_states(), 1);

        propagator.step_by(60.0).unwrap();
        assert_eq!(propagator.get_num_states(), 2);
    }

    #[test]
    fn test_numerical_propagator_get_initial_state() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0, 15.0, 30.0, 0.0);
        let params = NumericalOrbitPropagatorParams {
            mass: 100.0,
            n_gravity: 20,
            m_gravity: 20,
            enable_drag: true,
            drag_coefficient: 2.3,
            drag_area: 1.0,
            enable_srp: true,
            srp_coefficient: 1.8,
            srp_area: 1.0,
            enable_third_body_sun: true,
            enable_third_body_moon: true,
            enable_relativity: true,
        };
        let propagator = NumericalOrbitPropagator::new(epoch, state, params);

        assert_eq!(propagator.get_initial_state(), &Vector6::new(R_EARTH + 500e3, 0.01, 90.0, 15.0, 30.0, 0.0));
    }

    #[test]
    fn test_numerical_propagator_get_initial_epoch() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0, 15.0, 30.0, 0.0);
        let params = NumericalOrbitPropagatorParams {
            mass: 100.0,
            n_gravity: 20,
            m_gravity: 20,
            enable_drag: true,
            drag_coefficient: 2.3,
            drag_area: 1.0,
            enable_srp: true,
            srp_coefficient: 1.8,
            srp_area: 1.0,
            enable_third_body_sun: true,
            enable_third_body_moon: true,
            enable_relativity: true,
        };
        let propagator = NumericalOrbitPropagator::new(epoch, state, params);

        assert_eq!(propagator.get_initial_epoch(), Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC));
    }

    #[test]
    fn test_numerical_propagator_get_last_step_size() {
        setup_global_test_eop();
        setup_global_test_gravity_model();

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0, 15.0, 30.0, 0.0);
        let params = NumericalOrbitPropagatorParams {
            mass: 100.0,
            n_gravity: 20,
            m_gravity: 20,
            enable_drag: true,
            drag_coefficient: 2.3,
            drag_area: 1.0,
            enable_srp: true,
            srp_coefficient: 1.8,
            srp_area: 1.0,
            enable_third_body_sun: true,
            enable_third_body_moon: true,
            enable_relativity: true,
        };
        let mut propagator = NumericalOrbitPropagator::new(epoch, state, params);

        assert_eq!(propagator.get_last_step_size(), None);

        propagator.set_step_size(60.0).unwrap();
        propagator.step().unwrap();
        assert_eq!(propagator.get_last_step_size(), Some(60.0));
    }

    #[test]
    fn test_numerical_propagator_get_last_epoch() {
        setup_global_test_eop();
        setup_global_test_gravity_model();

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0, 15.0, 30.0, 0.0);
        let params = NumericalOrbitPropagatorParams {
            mass: 100.0,
            n_gravity: 20,
            m_gravity: 20,
            enable_drag: true,
            drag_coefficient: 2.3,
            drag_area: 1.0,
            enable_srp: true,
            srp_coefficient: 1.8,
            srp_area: 1.0,
            enable_third_body_sun: true,
            enable_third_body_moon: true,
            enable_relativity: true,
        };
        let mut propagator = NumericalOrbitPropagator::new(epoch, state, params);

        assert_eq!(propagator.get_last_epoch(), epoch);

        propagator.step_by(60.0).unwrap();
        assert_eq!(propagator.get_last_epoch(), epoch + 60);
    }

    #[test]
    fn test_numerical_propagator_get_final_epoch() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0, 15.0, 30.0, 0.0);
        let params = NumericalOrbitPropagatorParams {
            mass: 100.0,
            n_gravity: 20,
            m_gravity: 20,
            enable_drag: true,
            drag_coefficient: 2.3,
            drag_area: 1.0,
            enable_srp: true,
            srp_coefficient: 1.8,
            srp_area: 1.0,
            enable_third_body_sun: true,
            enable_third_body_moon: true,
            enable_relativity: true,
        };
        let mut propagator = NumericalOrbitPropagator::new(epoch, state, params);

        assert_eq!(propagator.get_final_epoch(), None);

        propagator.set_final_epoch(Epoch::from_datetime(2024, 1, 2, 12, 0, 0.0, 0.0, TimeSystem::UTC)).unwrap();
        assert_eq!(propagator.get_final_epoch(), Some(Epoch::from_datetime(2024, 1, 2, 12, 0, 0.0, 0.0, TimeSystem::UTC)));
    }

    #[test]
    fn test_numerical_propagator_get_step_size() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0, 15.0, 30.0, 0.0);
        let params = NumericalOrbitPropagatorParams {
            mass: 100.0,
            n_gravity: 20,
            m_gravity: 20,
            enable_drag: true,
            drag_coefficient: 2.3,
            drag_area: 1.0,
            enable_srp: true,
            srp_coefficient: 1.8,
            srp_area: 1.0,
            enable_third_body_sun: true,
            enable_third_body_moon: true,
            enable_relativity: true,
        };
        let mut propagator = NumericalOrbitPropagator::new(epoch, state, params);

        assert_eq!(propagator.get_step_size(), None);

        propagator.set_step_size(60.0).unwrap();
        assert_eq!(propagator.get_step_size(), Some(60.0));
    }

    #[test]
    fn test_numerical_propagator_get_state_by_index() {
        setup_global_test_eop();
        setup_global_test_gravity_model();

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let sma = R_EARTH + 500e3;
        let state_oe = Vector6::new(sma, 0.01, 90.0, 15.0, 30.0, 0.0);
        let state = state_osculating_to_cartesian(state_oe, true);
        let params = NumericalOrbitPropagatorParams {
            mass: 100.0,
            n_gravity: 20,
            m_gravity: 20,
            enable_drag: true,
            drag_coefficient: 2.3,
            drag_area: 1.0,
            enable_srp: true,
            srp_coefficient: 1.8,
            srp_area: 1.0,
            enable_third_body_sun: true,
            enable_third_body_moon: true,
            enable_relativity: true,
        };
        let mut propagator = NumericalOrbitPropagator::new(epoch, state, params);

        assert_eq!(propagator.get_state_by_index(0), Some(&state));
        assert_eq!(propagator.get_state_by_index(1), None);

        propagator.step_by(orbital_period(5.0)).unwrap();
        let state2 = propagator.get_state_by_index(0).unwrap();
        assert_eq!(state2[0], state[0]);
        assert_eq!(state2[1], state[1]);
        assert_eq!(state2[2], state[2]);
        assert_eq!(state2[3], state[3]);
        assert_eq!(state2[4], state[4]);
        assert_eq!(state2[5], state[5]);
    }

    #[test]
    fn test_numerical_propagator_get_state_by_epoch() {
        setup_global_test_eop();
        setup_global_test_gravity_model();

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let sma = R_EARTH + 500e3;
        let state_oe = Vector6::new(sma, 0.01, 90.0, 15.0, 30.0, 0.0);
        let state = state_osculating_to_cartesian(state_oe, true);
        let params = NumericalOrbitPropagatorParams {
            mass: 100.0,
            n_gravity: 20,
            m_gravity: 20,
            enable_drag: true,
            drag_coefficient: 2.3,
            drag_area: 1.0,
            enable_srp: true,
            srp_coefficient: 1.8,
            srp_area: 1.0,
            enable_third_body_sun: true,
            enable_third_body_moon: true,
            enable_relativity: true,
        };
        let mut propagator = NumericalOrbitPropagator::new(epoch, state, params);

        assert_eq!(propagator.get_state_by_epoch(epoch), Some(&state));
        assert_eq!(propagator.get_state_by_epoch(Epoch::from_datetime(2024, 1, 2, 12, 0, 0.0, 0.0, TimeSystem::UTC)), None);

        let period = orbital_period(sma);
        propagator.step_by(1.0).unwrap();
        let state2 = propagator.get_state_by_epoch(epoch).unwrap();
        assert_eq!(state2[0], state[0]);
        assert_eq!(state2[1], state[1]);
        assert_eq!(state2[2], state[2]);
        assert_eq!(state2[3], state[3]);
        assert_eq!(state2[4], state[4]);
        assert_eq!(state2[5], state[5]);
    }

    #[test]
    fn test_numerical_propagator_set_final_epoch() {
        setup_global_test_eop();
        setup_global_test_gravity_model();

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let params = NumericalOrbitPropagatorParams {
            mass: 100.0,
            n_gravity: 20,
            m_gravity: 20,
            enable_drag: true,
            drag_coefficient: 2.3,
            drag_area: 1.0,
            enable_srp: true,
            srp_coefficient: 1.8,
            srp_area: 1.0,
            enable_third_body_sun: true,
            enable_third_body_moon: true,
            enable_relativity: true,
        };
        let mut propagator = NumericalOrbitPropagator::new(epoch, Vector6::zeros(), params);

        assert_eq!(propagator.get_final_epoch(), None);

        propagator.set_final_epoch(Epoch::from_datetime(2024, 1, 2, 12, 0, 0.0, 0.0, TimeSystem::UTC)).unwrap();
        assert_eq!(propagator.get_final_epoch(), Some(Epoch::from_datetime(2024, 1, 2, 12, 0, 0.0, 0.0, TimeSystem::UTC)));
    }

    #[test]
    fn test_numerical_propagator_set_step_size() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let params = NumericalOrbitPropagatorParams {
            mass: 100.0,
            n_gravity: 20,
            m_gravity: 20,
            enable_drag: true,
            drag_coefficient: 2.3,
            drag_area: 1.0,
            enable_srp: true,
            srp_coefficient: 1.8,
            srp_area: 1.0,
            enable_third_body_sun: true,
            enable_third_body_moon: true,
            enable_relativity: true,
        };
        let mut propagator = NumericalOrbitPropagator::new(epoch, Vector6::zeros(), params);

        assert_eq!(propagator.get_step_size(), None);

        propagator.set_step_size(60.0).unwrap();
        assert_eq!(propagator.get_step_size(), Some(60.0));
    }

    #[test]
    fn test_numerical_propagator_reinitialize() {
        setup_global_test_eop();
        setup_global_test_gravity_model();

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state_oe = Vector6::new(R_EARTH + 500e3, 0.01, 90.0, 15.0, 30.0, 0.0);
        let state = state_osculating_to_cartesian(state_oe, true);
        let params = NumericalOrbitPropagatorParams {
            mass: 100.0,
            n_gravity: 20,
            m_gravity: 20,
            enable_drag: true,
            drag_coefficient: 2.3,
            drag_area: 1.0,
            enable_srp: true,
            srp_coefficient: 1.8,
            srp_area: 1.0,
            enable_third_body_sun: true,
            enable_third_body_moon: true,
            enable_relativity: true,
        };
        let mut propagator = NumericalOrbitPropagator::new(epoch, state, params);

        propagator.step_by(60.0).unwrap();
        assert_eq!(propagator.get_num_states(), 2);

        propagator.reinitialize();
        assert_eq!(propagator.get_num_states(), 1);
    }

    #[test]
    fn test_numerical_propagator_step() {
        setup_global_test_eop();
        setup_global_test_gravity_model();

        let sma = R_EARTH + 500e3;
        let period = orbital_period(sma);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state_oe = Vector6::new(R_EARTH + 500e3, 0.01, 90.0, 15.0, 30.0, 0.0);
        let state = state_osculating_to_cartesian(state_oe, true);

        let params = NumericalOrbitPropagatorParams {
            mass: 100.0,
            n_gravity: 20,
            m_gravity: 20,
            enable_drag: true,
            drag_coefficient: 2.3,
            drag_area: 1.0,
            enable_srp: true,
            srp_coefficient: 1.8,
            srp_area: 1.0,
            enable_third_body_sun: true,
            enable_third_body_moon: true,
            enable_relativity: true,
        };
        let mut propagator = NumericalOrbitPropagator::new(epoch, state, params);
        propagator.set_step_size(1.0).unwrap();

        for _ in 0..10 {
            propagator.step().unwrap();
        }

        assert_eq!(propagator.get_num_states(), 11);
        assert_eq!(propagator.get_last_epoch(), epoch + 10.0);
        assert_eq!(propagator.get_last_step_size(), Some(1.0));
    }

    #[test]
    fn test_numerical_propagator_step_by() {
        setup_global_test_eop();
        setup_global_test_gravity_model();

        let sma = R_EARTH + 500e3;
        let period = orbital_period(sma);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state_oe = Vector6::new(sma, 0.01, 90.0, 15.0, 30.0, 0.0);
        let state = state_osculating_to_cartesian(state_oe, true);

        // Parameters for two-body motion
        let params = NumericalOrbitPropagatorParams {
            mass: 100.0,
            n_gravity: 20,
            m_gravity: 20,
            enable_drag: true,
            drag_coefficient: 2.3,
            drag_area: 1.0,
            enable_srp: true,
            srp_coefficient: 1.8,
            srp_area: 1.0,
            enable_third_body_sun: true,
            enable_third_body_moon: true,
            enable_relativity: true,
        };
        let mut propagator = NumericalOrbitPropagator::new(epoch, state, params);
        propagator.set_step_size(1.0).unwrap();

        // Step by something other than the set step
        propagator.step_by(1.0).unwrap();

        assert_eq!(propagator.get_num_states(), 2);
        assert_eq!(propagator.get_last_epoch(), epoch + 1.0);
        assert_eq!(propagator.get_last_step_size(), Some(1.0));
    }

    #[test]
    fn test_numerical_propagator_step_to_epoch() {
        setup_global_test_eop();
        setup_global_test_gravity_model();

        let sma = R_EARTH + 500e3;
        let period = orbital_period(sma);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state_oe = Vector6::new(sma, 0.01, 90.0, 15.0, 30.0, 0.0);
        let state = state_osculating_to_cartesian(state_oe, true);

        // Parameters for two-body motion
        let params = NumericalOrbitPropagatorParams {
            mass: 100.0,
            n_gravity: 0,
            m_gravity: 0,
            enable_drag: false,
            drag_coefficient: 2.3,
            drag_area: 1.0,
            enable_srp: false,
            srp_coefficient: 1.8,
            srp_area: 1.0,
            enable_third_body_sun: false,
            enable_third_body_moon: false,
            enable_relativity: false,
        };
        let mut propagator = NumericalOrbitPropagator::new(epoch, state, params);
        propagator.set_step_size(1.0).unwrap();

        propagator.step_to_epoch(epoch + period).unwrap();

        assert_eq!(propagator.get_num_states(), period.floor() as usize + 2);
        assert_eq!(propagator.get_last_epoch(), epoch + period);
        let state2 = propagator.get_last_state();
        let tol = 1e-6;
        assert_abs_diff_eq!(state2[0], state[0], epsilon = tol);
        assert_abs_diff_eq!(state2[1], state[1], epsilon = tol);
        assert_abs_diff_eq!(state2[2], state[2], epsilon = tol);
        assert_abs_diff_eq!(state2[3], state[3], epsilon = tol);
        assert_abs_diff_eq!(state2[4], state[4], epsilon = tol);
        assert_abs_diff_eq!(state2[5], state[5], epsilon = tol);
    }

    #[test]
    fn test_numerical_propagator_step_to_final_epoch() {
        setup_global_test_eop();
        setup_global_test_gravity_model();

        let sma = R_EARTH + 500e3;
        let period = orbital_period(sma);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state_oe = Vector6::new(sma, 0.01, 90.0, 15.0, 30.0, 0.0);
        let state = state_osculating_to_cartesian(state_oe, true);

        // Parameters for two-body motion
        let params = NumericalOrbitPropagatorParams {
            mass: 100.0,
            n_gravity: 0,
            m_gravity: 0,
            enable_drag: false,
            drag_coefficient: 2.3,
            drag_area: 1.0,
            enable_srp: false,
            srp_coefficient: 1.8,
            srp_area: 1.0,
            enable_third_body_sun: false,
            enable_third_body_moon: false,
            enable_relativity: false,
        };
        let mut propagator = NumericalOrbitPropagator::new(epoch, state, params);
        propagator.set_step_size(1.0).unwrap();
        propagator.set_final_epoch(epoch + period).unwrap();

        propagator.step_to_final_epoch().unwrap();

        assert_eq!(propagator.get_num_states(), period.floor() as usize + 2);
        assert_eq!(propagator.get_last_epoch(), epoch + period);
        let state2 = propagator.get_last_state();
        let tol = 1e-6;
        assert_abs_diff_eq!(state2[0], state[0], epsilon = tol);
        assert_abs_diff_eq!(state2[1], state[1], epsilon = tol);
        assert_abs_diff_eq!(state2[2], state[2], epsilon = tol);
        assert_abs_diff_eq!(state2[3], state[3], epsilon = tol);
        assert_abs_diff_eq!(state2[4], state[4], epsilon = tol);
        assert_abs_diff_eq!(state2[5], state[5], epsilon = tol);
    }

    #[test]
    fn test_numerical_propagator_step_to_final_epoch_full_force() {
        setup_global_test_eop();
        setup_global_test_gravity_model();

        let sma = R_EARTH + 650e3;
        let period = orbital_period(sma);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state_oe = Vector6::new(sma, 0.01, 90.0, 15.0, 30.0, 0.0);
        let state = state_osculating_to_cartesian(state_oe, true);

        // Parameters for two-body motion
        let params = NumericalOrbitPropagatorParams {
            mass: 1000.0,
            n_gravity: 20,
            m_gravity: 20,
            enable_drag: true,
            drag_coefficient: 2.3,
            drag_area: 0.5,
            enable_srp: true,
            srp_coefficient: 1.8,
            srp_area: 0.5,
            enable_third_body_sun: true,
            enable_third_body_moon: true,
            enable_relativity: true,
        };
        let mut propagator = NumericalOrbitPropagator::new(epoch, state, params);
        propagator.set_step_size(1.0).unwrap();
        propagator.set_final_epoch(epoch + period).unwrap();

        propagator.step_to_final_epoch().unwrap();

        assert_eq!(propagator.get_num_states(), period.floor() as usize + 2);
        assert_eq!(propagator.get_last_epoch(), epoch + period);
    }
}