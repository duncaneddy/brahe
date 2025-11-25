/*!
 * Numerical orbit propagator with configurable force models
 *
 * This module provides a high-fidelity numerical orbit propagator that:
 * - Builds dynamics from configurable force models
 * - Supports extended state dimensions (6D + additional states)
 * - Provides STM and sensitivity matrix propagation
 * - Integrates with event detection framework
 * - Handles frame and representation conversions
 */

use std::sync::Arc;

use nalgebra::{DMatrix, DVector, Vector3, Vector6};

use crate::constants::{GM_EARTH, R_EARTH};
use crate::earth_models::{density_harris_priester, density_nrlmsise00};
use crate::frames::rotation_eci_to_ecef;
use crate::integrators::traits::DIntegrator;
use crate::math::interpolation::{
    CovarianceInterpolationConfig, CovarianceInterpolationMethod, InterpolationConfig,
    InterpolationMethod,
};
#[allow(unused_imports)]
use crate::math::jacobian::DNumericalJacobian;
use crate::math::jacobian::DifferenceMethod;
#[allow(unused_imports)]
use crate::math::sensitivity::DNumericalSensitivity;
use crate::orbit_dynamics::{
    GravityModel, accel_drag, accel_gravity_spherical_harmonics, accel_point_mass_gravity,
    accel_relativity, accel_solar_radiation_pressure, accel_third_body_jupiter_de440s,
    accel_third_body_mars_de440s, accel_third_body_mercury_de440s, accel_third_body_moon,
    accel_third_body_moon_de440s, accel_third_body_neptune_de440s, accel_third_body_saturn_de440s,
    accel_third_body_sun, accel_third_body_sun_de440s, accel_third_body_uranus_de440s,
    accel_third_body_venus_de440s, eclipse_conical, eclipse_cylindrical, get_global_gravity_model,
    sun_position,
};
use crate::propagators::{
    AtmosphericModel, EclipseModel, EphemerisSource, ForceModelConfiguration, GravityConfiguration,
    GravityModelSource, ThirdBody,
};
use crate::relative_motion::rotation_eci_to_rtn;
use crate::time::Epoch;
use crate::traits::{OrbitFrame, OrbitRepresentation};
use crate::trajectories::DOrbitTrajectory;
use crate::trajectories::traits::{InterpolatableTrajectory, Trajectory};
use crate::utils::errors::BraheError;
use crate::utils::identifiable::Identifiable;
use crate::utils::state_providers::{
    DCovarianceProvider, DOrbitCovarianceProvider, DOrbitStateProvider, DStateProvider,
};
use crate::{AngleFormat, state_cartesian_to_osculating};

use super::TrajectoryMode;
use super::traits::DStatePropagator;

// Event detection imports
use crate::events::{DDetectedEvent, DEventDetector, EventAction, dscan_for_event};

// Import dynamics type from integrator traits
use crate::integrators::traits::DStateDynamics;

// =============================================================================
// Propagation Mode
// =============================================================================

/// Propagation mode determining which matrices are propagated
///
/// This is configured at construction time and cannot be changed during propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum PropagationMode {
    /// Propagate state only (basic orbit propagation)
    StateOnly,
    /// Propagate state and STM (for covariance propagation)
    WithSTM,
    /// Propagate state and sensitivity matrix (for parameter sensitivity analysis)
    WithSensitivity,
    /// Propagate state, STM, and sensitivity matrix (full uncertainty quantification)
    WithSTMAndSensitivity,
}

// =============================================================================
// Event Processing Result
// =============================================================================

/// Result of processing detected events in an integration step
#[derive(Debug)]
enum EventProcessingResult {
    /// No events detected, or all events processed without callbacks
    NoEvents,
    /// Event callback requested state/param update - restart integration from event time
    Restart { epoch: Epoch, state: DVector<f64> },
    /// Terminal event detected - stop propagation
    Terminal,
}

// =============================================================================
// Shared Dynamics Type
// =============================================================================

/// Shared dynamics function that can be used by multiple consumers
///
/// This type wraps the dynamics function in an Arc to allow sharing between:
/// - The main integrator
/// - The Jacobian provider (for STM computation)
/// - The sensitivity provider (for parameter sensitivity computation)
///
/// This ensures consistency - all three use the exact same dynamics function,
/// including any `additional_dynamics` that were provided.
type SharedDynamics =
    Arc<dyn Fn(f64, &DVector<f64>, Option<&DVector<f64>>) -> DVector<f64> + Send + Sync>;

// =============================================================================
// Numerical Orbit Propagator
// =============================================================================

/// High-fidelity numerical orbit propagator with built-in force models
///
/// This propagator wraps a dynamic-sized adaptive integrator and automatically builds
/// the dynamics function from a `ForceModelConfiguration`. It handles:
/// - Multiple gravity models (point mass or spherical harmonic)
/// - Atmospheric drag (Harris-Priester, NRLMSISE-00, or exponential)
/// - Solar radiation pressure with eclipse modeling
/// - Third-body perturbations (Sun, Moon, planets)
/// - Relativistic corrections
/// - User-defined additional dynamics for extended state
///
/// # State Dimensions
/// The propagator supports flexible state dimensions:
/// - Base: 6D (position + velocity)
/// - Extended: 6 + N where N is the number of additional state elements
///
/// # Parameter Vector Layout
/// Default layout: `[mass, drag_area, Cd, srp_area, Cr, ...]`
/// - Index 0: mass [kg]
/// - Index 1: drag cross-sectional area [m²]
/// - Index 2: drag coefficient (dimensionless)
/// - Index 3: SRP cross-sectional area [m²]
/// - Index 4: coefficient of reflectivity (dimensionless)
///
/// Users can customize indices in the force configuration.
///
/// # Example
///
/// ```rust,ignore
/// use brahe::prelude::*;
/// use nalgebra::DVector;
///
/// // Create configurations
/// let prop_config = NumericalPropagationConfig::default();
/// let force_config = ForceModelConfiguration::default();
///
/// // Create initial state (ECI Cartesian)
/// let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let state = DVector::from_vec(vec![
///     R_EARTH + 500e3, 0.0, 0.0,  // position
///     0.0, 7500.0, 0.0,            // velocity
/// ]);
///
/// // Parameters: [mass, drag_area, Cd, srp_area, Cr]
/// let params = DVector::from_vec(vec![1000.0, 10.0, 2.2, 10.0, 1.3]);
///
/// // Create propagator
/// let mut prop = DNumericalOrbitPropagator::new(
///     epoch,
///     state,
///     prop_config,
///     force_config,
///     Some(params),
///     None,  // additional_dynamics
///     None,  // initial_covariance
/// ).unwrap();
///
/// // Propagate
/// prop.propagate_to(epoch + 86400.0);  // 1 day
/// ```
pub struct DNumericalOrbitPropagator {
    // ===== Time Management =====
    /// Initial absolute time (Epoch)
    epoch_initial: Epoch,
    /// Current absolute time (Epoch) - updated on each step
    epoch_current: Epoch,
    /// Current relative time (seconds since start) - used by integrator for numerical stability
    t_rel: f64,

    // ===== Integration =====
    /// Numerical integrator (type-erased for runtime flexibility)
    integrator: Box<dyn DIntegrator>,
    /// Force model configuration
    #[allow(dead_code)]
    force_config: ForceModelConfiguration,
    /// Gravity model (loaded at construction if source is ModelType)
    /// Note: The model is captured by the dynamics closure, this field stores it for Clone support
    #[allow(dead_code)]
    gravity_model: Option<Arc<GravityModel>>,
    /// Current integration step size
    dt: f64,
    /// Suggested next step size (from adaptive integrator)
    dt_next: f64,

    // ===== State Management =====
    /// Initial state vector (for reset) - in ECI Cartesian
    x_initial: DVector<f64>,
    /// Current state vector - in ECI Cartesian
    x_curr: DVector<f64>,
    /// Mutable parameter vector
    params: DVector<f64>,
    /// State dimension
    state_dim: usize,

    // ===== Frame/Representation Conversion =====
    /// Input/output frame
    input_frame: OrbitFrame,
    /// Input/output representation
    input_representation: OrbitRepresentation,
    /// Angle format for conversions
    angle_format: AngleFormat,

    // ===== STM and Sensitivity =====
    /// Propagation mode (configured at construction, immutable)
    propagation_mode: PropagationMode,
    /// State transition matrix Φ(t, t₀)
    stm: Option<DMatrix<f64>>,
    /// Sensitivity matrix S(t, t₀) = ∂x/∂p
    sensitivity: Option<DMatrix<f64>>,
    /// Whether to store STM history in trajectory
    store_stm_history: bool,
    /// Whether to store sensitivity history in trajectory
    store_sensitivity_history: bool,

    // ===== Covariance =====
    /// Initial covariance matrix P₀ (if provided)
    initial_covariance: Option<DMatrix<f64>>,
    /// Current covariance matrix P(t)
    current_covariance: Option<DMatrix<f64>>,

    // ===== Trajectory Storage =====
    /// Storage for state history
    trajectory: DOrbitTrajectory,
    /// Trajectory storage mode
    trajectory_mode: TrajectoryMode,
    /// Interpolation method for state retrieval
    interpolation_method: InterpolationMethod,
    /// Covariance interpolation method
    covariance_interpolation_method: CovarianceInterpolationMethod,

    // ===== Event Detection =====
    /// Event detectors for monitoring propagation
    event_detectors: Vec<Box<dyn DEventDetector>>,
    /// Log of detected events
    event_log: Vec<DDetectedEvent>,
    /// Termination flag (set by terminal events)
    terminated: bool,

    // ===== Metadata =====
    /// Propagator name (optional)
    pub name: Option<String>,
    /// Propagator ID (optional)
    pub id: Option<u64>,
    /// Propagator UUID (optional)
    pub uuid: Option<uuid::Uuid>,
}

impl DNumericalOrbitPropagator {
    // =========================================================================
    // Construction
    // =========================================================================

    /// Create a new numerical orbit propagator
    ///
    /// This is the primary constructor that builds the integrator based on the
    /// propagation configuration. State is assumed to be in ECI Cartesian format.
    ///
    /// # Arguments
    /// * `epoch` - Initial epoch
    /// * `state` - Initial state vector in ECI Cartesian format (6D or 6+N dimensional)
    /// * `propagation_config` - Numerical propagation configuration (integrator method + settings)
    /// * `force_config` - Force model configuration (gravity, drag, SRP, third-body, etc.)
    /// * `params` - Optional parameter vector `[mass, drag_area, Cd, srp_area, Cr, ...]`.
    ///   Required if force_config references parameter indices.
    /// * `additional_dynamics` - Optional function for extended state dynamics (beyond 6D)
    /// * `initial_covariance` - Optional initial covariance matrix P₀ (enables STM propagation)
    ///
    /// # Returns
    /// New propagator ready for propagation, or error if configuration is invalid
    ///
    /// # Errors
    /// Returns `BraheError` if:
    /// - Force model references parameter indices but no parameter vector is provided
    /// - Parameter vector is too short for the force model configuration
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use brahe::propagators::{DNumericalOrbitPropagator, NumericalPropagationConfig, ForceModelConfiguration};
    /// use nalgebra::DVector;
    ///
    /// let prop = DNumericalOrbitPropagator::new(
    ///     epoch,
    ///     state,
    ///     NumericalPropagationConfig::default(),
    ///     ForceModelConfiguration::default(),
    ///     Some(DVector::from_vec(vec![1000.0, 10.0, 2.2, 10.0, 1.3])),
    ///     None,
    ///     None,
    /// )?;
    /// ```
    pub fn new(
        epoch: Epoch,
        state: DVector<f64>,
        propagation_config: super::NumericalPropagationConfig,
        force_config: ForceModelConfiguration,
        params: Option<DVector<f64>>,
        additional_dynamics: Option<DStateDynamics>,
        initial_covariance: Option<DMatrix<f64>>,
    ) -> Result<Self, BraheError> {
        // Validate parameters against force model requirements
        force_config.validate_params(params.as_ref())?;

        // Use provided params or create empty vector (only valid if force config doesn't need params)
        let params = params.unwrap_or_else(|| DVector::zeros(0));

        // State is assumed to be in ECI Cartesian format
        let state_eci = state;

        // Get state dimension
        let state_dim = state_eci.len();

        // Determine what to propagate based on config and provided data
        // STM is auto-enabled if initial_covariance is provided, or can be explicitly enabled
        let enable_stm = propagation_config.variational.enable_stm || initial_covariance.is_some();
        let enable_sensitivity = propagation_config.variational.enable_sensitivity;

        // Validate: sensitivity requires params
        if enable_sensitivity && params.is_empty() {
            return Err(BraheError::PropagatorError(
                "Sensitivity propagation requires params to be provided".to_string(),
            ));
        }

        // Load gravity model if using ModelType source, truncating to requested degree/order
        let gravity_model = match &force_config.gravity {
            GravityConfiguration::SphericalHarmonic {
                source: GravityModelSource::ModelType(model_type),
                degree,
                order,
            } => {
                let mut model = GravityModel::from_model_type(model_type)?;
                // Truncate model to save memory (only keep coefficients we'll use)
                model.set_max_degree_order(*degree, *order)?;
                Some(Arc::new(model))
            }
            _ => None,
        };

        // Build shared dynamics function (includes additional_dynamics)
        let shared_dynamics = Self::build_shared_dynamics(
            epoch,
            force_config.clone(),
            params.clone(),
            additional_dynamics,
            gravity_model.clone(),
        );

        // Wrap for main integrator
        let dynamics = Self::wrap_for_integrator(Arc::clone(&shared_dynamics));

        // Get initial step size from config
        let initial_dt = propagation_config.integrator.initial_step.unwrap_or(60.0);

        // Build Jacobian provider if STM or sensitivity enabled
        // (sensitivity propagation requires the Jacobian: dS/dt = A*S + B where A is ∂f/∂x)
        let jacobian_provider = if enable_stm || enable_sensitivity {
            Some(Self::build_jacobian_provider(
                Arc::clone(&shared_dynamics),
                propagation_config.variational.jacobian_method,
            ))
        } else {
            None
        };

        // Build Sensitivity provider if enabled
        let sensitivity_provider = if enable_sensitivity {
            Some(Self::build_sensitivity_provider(
                Arc::clone(&shared_dynamics),
                propagation_config.variational.sensitivity_method,
            ))
        } else {
            None
        };

        // Create integrator using factory function
        let integrator = crate::integrators::create_dintegrator(
            propagation_config.method,
            state_dim,
            dynamics,
            jacobian_provider,
            sensitivity_provider,
            None, // No control input by default
            propagation_config.integrator,
        );

        // Create trajectory storage (internally always ECI Cartesian)
        let mut trajectory =
            DOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);

        // Enable STM/sensitivity storage in trajectory if configured
        if propagation_config.variational.store_stm_history {
            trajectory.enable_stm_storage();
        }
        if propagation_config.variational.store_sensitivity_history && !params.is_empty() {
            trajectory.enable_sensitivity_storage(params.len());
        }

        // Determine propagation mode based on what's enabled
        let propagation_mode = match (enable_stm, enable_sensitivity) {
            (false, false) => PropagationMode::StateOnly,
            (true, false) => PropagationMode::WithSTM,
            (false, true) => PropagationMode::WithSensitivity,
            (true, true) => PropagationMode::WithSTMAndSensitivity,
        };

        // Initialize matrices
        let stm = if enable_stm {
            Some(DMatrix::identity(state_dim, state_dim))
        } else {
            None
        };

        let sensitivity = if enable_sensitivity {
            Some(DMatrix::zeros(state_dim, params.len()))
        } else {
            None
        };

        // Set up covariance if initial covariance provided
        let current_covariance = initial_covariance.clone();

        Ok(Self {
            epoch_initial: epoch,
            epoch_current: epoch,
            t_rel: 0.0,
            integrator,
            force_config,
            gravity_model,
            dt: initial_dt,
            dt_next: initial_dt,
            x_initial: state_eci.clone(),
            x_curr: state_eci,
            params,
            state_dim,
            input_frame: OrbitFrame::ECI,
            input_representation: OrbitRepresentation::Cartesian,
            angle_format: AngleFormat::Radians,
            propagation_mode,
            stm,
            sensitivity,
            store_stm_history: propagation_config.variational.store_stm_history,
            store_sensitivity_history: propagation_config.variational.store_sensitivity_history,
            initial_covariance,
            current_covariance,
            trajectory,
            trajectory_mode: TrajectoryMode::AllSteps,
            interpolation_method: InterpolationMethod::Linear,
            covariance_interpolation_method: CovarianceInterpolationMethod::TwoWasserstein,
            event_detectors: Vec::new(),
            event_log: Vec::new(),
            terminated: false,
            name: None,
            id: None,
            uuid: None,
        })
    }

    /// Convert state from ECI Cartesian (internal format) to user format
    fn convert_state_from_eci(
        &self,
        state: &DVector<f64>,
        epoch: Epoch,
    ) -> Result<DVector<f64>, BraheError> {
        // Extract orbital state (first 6 elements)
        let eci_state = state.fixed_rows::<6>(0).into_owned();

        // Convert frame if needed
        let frame_state = match self.input_frame {
            OrbitFrame::ECI => eci_state,
            OrbitFrame::ECEF => crate::frames::state_eci_to_ecef(epoch, eci_state),
            _ => {
                return Err(BraheError::Error(
                    "Unsupported orbit frame for numerical propagation".to_string(),
                ));
            }
        };

        // Convert representation if needed
        let output_state = match self.input_representation {
            OrbitRepresentation::Cartesian => frame_state,
            OrbitRepresentation::Keplerian => {
                state_cartesian_to_osculating(frame_state, self.angle_format)
            }
        };

        // If state has additional elements, preserve them
        if state.len() > 6 {
            let mut full_state = DVector::zeros(state.len());
            full_state.fixed_rows_mut::<6>(0).copy_from(&output_state);
            full_state
                .rows_mut(6, state.len() - 6)
                .copy_from(&state.rows(6, state.len() - 6));
            Ok(full_state)
        } else {
            Ok(DVector::from_column_slice(output_state.as_slice()))
        }
    }

    // =========================================================================
    // Event Detection
    // =========================================================================

    /// Scan all event detectors for events in the interval [epoch_prev, epoch_new]
    ///
    /// Returns a vector of (detector_index, detected_event) pairs
    fn scan_all_events(
        &self,
        epoch_prev: Epoch,
        epoch_new: Epoch,
        x_prev: &DVector<f64>,
        x_new: &DVector<f64>,
    ) -> Vec<(usize, DDetectedEvent)> {
        let mut events = Vec::new();

        // Create state function for bisection search: Epoch → State
        // Uses linear interpolation between x_prev and x_new
        // TODO: Use integrator's dense output if available for better accuracy
        let state_fn = |epoch: Epoch| -> DVector<f64> {
            let t_rel = epoch - self.epoch_initial;
            let t_prev = epoch_prev - self.epoch_initial;
            let t_new = epoch_new - self.epoch_initial;

            // Linear interpolation parameter
            let alpha = (t_rel - t_prev) / (t_new - t_prev);

            // Interpolate state
            x_prev + (x_new - x_prev) * alpha
        };

        // Scan each detector
        for (idx, detector) in self.event_detectors.iter().enumerate() {
            if let Some(event) = dscan_for_event(
                detector.as_ref(),
                &state_fn,
                epoch_prev,
                epoch_new,
                x_prev,
                x_new,
                Some(&self.params),
            ) {
                events.push((idx, event));
            }
        }

        events
    }

    /// Process detected events using smart sequential algorithm
    ///
    /// This implements the smart event processing strategy:
    /// 1. Sort events chronologically
    /// 2. Find first event with callback
    /// 3. Process all no-callback events before it (they're safe - don't modify state/params)
    /// 4. Process first callback event, apply mutations, return Restart
    /// 5. Discard events after callback (invalid with new state/params)
    fn process_events_smart(
        &mut self,
        detected_events: Vec<(usize, DDetectedEvent)>,
    ) -> EventProcessingResult {
        if detected_events.is_empty() {
            return EventProcessingResult::NoEvents;
        }

        // 1. Sort events chronologically by window_open time
        let mut sorted_events = detected_events;
        sorted_events.sort_by(|(_, a), (_, b)| a.window_open.partial_cmp(&b.window_open).unwrap());

        // 2. Find first event with callback
        let first_callback_idx = sorted_events
            .iter()
            .position(|(det_idx, _)| self.event_detectors[*det_idx].callback().is_some());

        match first_callback_idx {
            None => {
                // 3a. No callbacks - process all events (just log them)
                // But check if any are terminal
                let mut terminal_event = None;

                for (det_idx, event) in sorted_events {
                    self.event_log.push(event.clone());

                    // Add to trajectory at exact event time if configured
                    if !matches!(self.trajectory_mode, TrajectoryMode::Disabled) {
                        self.trajectory
                            .add(event.window_open, event.entry_state.clone());
                    }

                    // Check if this event is terminal (no callback but has terminal action)
                    let detector = &self.event_detectors[det_idx];
                    if detector.action() == EventAction::Stop {
                        terminal_event = Some(event);
                        break; // Stop processing further events
                    }
                }

                if let Some(_term_event) = terminal_event {
                    self.terminated = true;
                    return EventProcessingResult::Terminal;
                }

                EventProcessingResult::NoEvents // Continue with step
            }

            Some(callback_idx) => {
                // 3b. Process all no-callback events before the callback event
                for (_, event) in sorted_events.iter().take(callback_idx) {
                    self.event_log.push(event.clone());

                    if !matches!(self.trajectory_mode, TrajectoryMode::Disabled) {
                        self.trajectory
                            .add(event.window_open, event.entry_state.clone());
                    }
                }

                // 4. Process the first callback event
                let (det_idx, callback_event) = &sorted_events[callback_idx];
                let detector = &self.event_detectors[*det_idx];

                // Log pre-callback event
                self.event_log.push(callback_event.clone());

                // Add pre-callback state to trajectory
                if !matches!(self.trajectory_mode, TrajectoryMode::Disabled) {
                    self.trajectory.add(
                        callback_event.window_open,
                        callback_event.entry_state.clone(),
                    );
                }

                // Execute callback
                if let Some(callback) = detector.callback() {
                    let (new_state, new_params, action) = callback(
                        callback_event.window_open,
                        &callback_event.entry_state,
                        Some(&self.params),
                    );

                    // Apply state mutation
                    let y_after = if let Some(y_new) = new_state {
                        // Add post-callback state to trajectory (same time, different state)
                        if !matches!(self.trajectory_mode, TrajectoryMode::Disabled) {
                            self.trajectory
                                .add(callback_event.window_open, y_new.clone());
                        }
                        y_new
                    } else {
                        callback_event.entry_state.clone()
                    };

                    // Apply parameter mutation
                    if let Some(p_new) = new_params {
                        self.params = p_new;
                    }

                    // Check terminal
                    if action == EventAction::Stop {
                        self.terminated = true;
                        return EventProcessingResult::Terminal;
                    }

                    // Restart integration from event time with new state/params
                    return EventProcessingResult::Restart {
                        epoch: callback_event.window_open,
                        state: y_after,
                    };
                }

                unreachable!("Callback event must have callback");
            }
        }
    }

    // =========================================================================
    // Unified Dynamics Builder
    // =========================================================================

    /// Build shared dynamics function from force model configuration
    ///
    /// Creates a shared dynamics function that is used consistently by:
    /// - The main integrator
    /// - The Jacobian provider (for STM computation)
    /// - The Sensitivity provider (for parameter sensitivity computation)
    ///
    /// This ensures that all three use the exact same dynamics, including
    /// any `additional_dynamics` that were provided.
    fn build_shared_dynamics(
        epoch_initial: Epoch,
        force_config: ForceModelConfiguration,
        params: DVector<f64>,
        additional_dynamics: Option<DStateDynamics>,
        gravity_model: Option<Arc<GravityModel>>,
    ) -> SharedDynamics {
        Arc::new(
            move |t: f64,
                  state: &DVector<f64>,
                  params_opt: Option<&DVector<f64>>|
                  -> DVector<f64> {
                // Compute orbital dynamics (first 6 elements)
                let mut dx = Self::compute_dynamics(
                    t,
                    state.clone(),
                    epoch_initial,
                    &force_config,
                    params_opt.or(Some(&params)),
                    gravity_model.as_ref(),
                );

                // If additional dynamics provided and state dimension > 6, compute extended state derivatives
                if let Some(ref add_dyn) = additional_dynamics
                    && state.len() > 6
                {
                    dx += add_dyn(t, state, params_opt);
                }

                dx
            },
        )
    }

    /// Convert shared dynamics (Arc) to integrator dynamics (Box)
    ///
    /// Both SharedDynamics and DStateDynamics now use references for the state parameter,
    /// so this is a simple Arc->Box conversion that moves the Arc into a boxed closure.
    fn wrap_for_integrator(shared: SharedDynamics) -> DStateDynamics {
        Box::new(
            move |t: f64, state: &DVector<f64>, params: Option<&DVector<f64>>| -> DVector<f64> {
                shared(t, state, params)
            },
        )
    }

    // =========================================================================
    // Jacobian and Sensitivity Provider Builders
    // =========================================================================

    /// Build Jacobian provider for STM propagation
    ///
    /// Creates a numerical Jacobian provider that computes ∂f/∂x using
    /// finite differences on the shared dynamics function.
    ///
    /// # Arguments
    /// * `shared_dynamics` - The shared dynamics function
    /// * `method` - Finite difference method (Forward, Central, or Backward)
    fn build_jacobian_provider(
        shared_dynamics: SharedDynamics,
        method: DifferenceMethod,
    ) -> Box<dyn crate::math::jacobian::DJacobianProvider> {
        // Wrap shared dynamics for the Jacobian provider signature
        // Both SharedDynamics and DStateDynamics use references, so this is a simple conversion
        let dynamics_for_jacobian = Box::new(
            move |t: f64, state: &DVector<f64>, params: Option<&DVector<f64>>| -> DVector<f64> {
                shared_dynamics(t, state, params)
            },
        );

        match method {
            DifferenceMethod::Forward => {
                Box::new(DNumericalJacobian::forward(dynamics_for_jacobian))
            }
            DifferenceMethod::Central => {
                Box::new(DNumericalJacobian::central(dynamics_for_jacobian))
            }
            DifferenceMethod::Backward => {
                Box::new(DNumericalJacobian::backward(dynamics_for_jacobian))
            }
        }
    }

    /// Build Sensitivity provider for parameter sensitivity propagation
    ///
    /// Creates a numerical sensitivity provider that computes ∂f/∂p using
    /// finite differences on the shared dynamics function.
    ///
    /// # Arguments
    /// * `shared_dynamics` - The shared dynamics function
    /// * `method` - Finite difference method (Forward, Central, or Backward)
    fn build_sensitivity_provider(
        shared_dynamics: SharedDynamics,
        method: DifferenceMethod,
    ) -> Box<dyn crate::math::sensitivity::DSensitivityProvider> {
        // Wrap shared dynamics for the Sensitivity provider signature
        let dynamics_with_params = Box::new(
            move |t: f64, state: &DVector<f64>, params: &DVector<f64>| -> DVector<f64> {
                shared_dynamics(t, state, Some(params))
            },
        );

        match method {
            DifferenceMethod::Forward => {
                Box::new(DNumericalSensitivity::forward(dynamics_with_params))
            }
            DifferenceMethod::Central => {
                Box::new(DNumericalSensitivity::central(dynamics_with_params))
            }
            DifferenceMethod::Backward => {
                Box::new(DNumericalSensitivity::backward(dynamics_with_params))
            }
        }
    }

    /// Core dynamics computation function
    ///
    /// Computes the state derivative for given time, state, and parameters.
    /// This is the actual force model evaluation logic.
    fn compute_dynamics(
        t: f64,
        state: DVector<f64>,
        epoch_initial: Epoch,
        force_config: &ForceModelConfiguration,
        params_opt: Option<&DVector<f64>>,
        gravity_model: Option<&Arc<GravityModel>>,
    ) -> DVector<f64> {
        // Convert relative time to absolute Epoch
        let epoch = epoch_initial + t;

        // Extract position and velocity (first 6 elements always orbital state)
        let r = Vector3::new(state[0], state[1], state[2]);
        let v = Vector3::new(state[3], state[4], state[5]);
        let x_eci: nalgebra::Matrix<
            f64,
            nalgebra::Const<6>,
            nalgebra::Const<1>,
            nalgebra::ArrayStorage<f64, 6, 1>,
        > = Vector6::new(state[0], state[1], state[2], state[3], state[4], state[5]);

        // Accumulate total acceleration
        let mut a_total = Vector3::zeros();

        // ===== GRAVITY =====
        match &force_config.gravity {
            GravityConfiguration::PointMass => {
                a_total += accel_point_mass_gravity(r, Vector3::zeros(), GM_EARTH);
            }
            GravityConfiguration::SphericalHarmonic {
                source,
                degree,
                order,
            } => {
                // Get rotation matrix from ECI to ECEF
                let r_i2b = rotation_eci_to_ecef(epoch);

                // Use gravity model based on source
                match source {
                    GravityModelSource::Global => {
                        // Use global gravity model
                        let global_model: std::sync::RwLockReadGuard<'_, Box<GravityModel>> =
                            get_global_gravity_model();
                        a_total += accel_gravity_spherical_harmonics(
                            r,
                            r_i2b,
                            &global_model,
                            *degree,
                            *order,
                        );
                    }
                    GravityModelSource::ModelType(_) => {
                        // Use the model loaded at construction (passed in)
                        if let Some(model) = gravity_model {
                            a_total += accel_gravity_spherical_harmonics(
                                r,
                                r_i2b,
                                model.as_ref(),
                                *degree,
                                *order,
                            );
                        }
                    }
                }
            }
        }

        // ===== DRAG =====
        if let Some(drag_config) = &force_config.drag {
            // Get mass from configuration (respects priority: params > config value)
            let mass = force_config
                .get_mass(params_opt)
                .expect("Mass required for drag calculation. Either provide a parameter vector or set ForceModelConfiguration.mass to ParameterSource::Value(mass_kg)");

            // Get drag parameters (area, Cd)
            let drag_area = drag_config.area.get_value(params_opt, 10.0);
            let cd = drag_config.cd.get_value(params_opt, 2.2);

            // Compute atmospheric density
            let density = match &drag_config.model {
                AtmosphericModel::HarrisPriester => {
                    let r_sun = sun_position(epoch);
                    density_harris_priester(r, r_sun)
                }
                AtmosphericModel::NRLMSISE00 => {
                    // NRLMSISE00 requires ECEF position
                    let r_i2b = rotation_eci_to_ecef(epoch);
                    let r_ecef = r_i2b * r;
                    density_nrlmsise00(&epoch, r_ecef).unwrap_or(0.0)
                }
                AtmosphericModel::Exponential {
                    scale_height,
                    rho0,
                    h0,
                } => {
                    let altitude = r.norm() - R_EARTH;
                    let h_diff = altitude - h0;
                    rho0 * (-h_diff / scale_height).exp()
                }
            };

            // Compute drag acceleration
            let r_i2b = rotation_eci_to_ecef(epoch);
            a_total += accel_drag(x_eci, density, mass, drag_area, cd, r_i2b);
        }

        // ===== SOLAR RADIATION PRESSURE =====
        if let Some(srp_config) = &force_config.srp {
            // Get mass from configuration (respects priority: params > config value)
            let mass = force_config
                .get_mass(params_opt)
                .expect("Mass required for SRP calculation. Either provide a parameter vector or set ForceModelConfiguration.mass to ParameterSource::Value(mass_kg)");

            // Get SRP parameters (area, Cr)
            let srp_area = srp_config.area.get_value(params_opt, 10.0);
            let cr = srp_config.cr.get_value(params_opt, 1.3);

            // Get sun position
            let r_sun = sun_position(epoch);

            // Compute SRP acceleration (P0 = 4.56e-6 N/m² at 1 AU)
            let mut a_srp = accel_solar_radiation_pressure(r, r_sun, mass, cr, srp_area, 4.56e-6);

            // Apply eclipse factor
            let eclipse_factor = match srp_config.eclipse_model {
                EclipseModel::None => 1.0,
                EclipseModel::Cylindrical => eclipse_cylindrical(r, r_sun),
                EclipseModel::Conical => eclipse_conical(r, r_sun),
            };

            a_srp *= eclipse_factor;
            a_total += a_srp;
        }

        // ===== THIRD BODY =====
        if let Some(tb_config) = &force_config.third_body {
            for body in &tb_config.bodies {
                let a_tb = match tb_config.ephemeris_source {
                    EphemerisSource::LowPrecision => match body {
                        ThirdBody::Sun => accel_third_body_sun(epoch, r),
                        ThirdBody::Moon => accel_third_body_moon(epoch, r),
                        _ => Vector3::zeros(), // Only Sun/Moon available for low-precision
                    },
                    EphemerisSource::DE440s => match body {
                        ThirdBody::Sun => accel_third_body_sun_de440s(epoch, r),
                        ThirdBody::Moon => accel_third_body_moon_de440s(epoch, r),
                        ThirdBody::Mercury => accel_third_body_mercury_de440s(epoch, r),
                        ThirdBody::Venus => accel_third_body_venus_de440s(epoch, r),
                        ThirdBody::Mars => accel_third_body_mars_de440s(epoch, r),
                        ThirdBody::Jupiter => accel_third_body_jupiter_de440s(epoch, r),
                        ThirdBody::Saturn => accel_third_body_saturn_de440s(epoch, r),
                        ThirdBody::Uranus => accel_third_body_uranus_de440s(epoch, r),
                        ThirdBody::Neptune => accel_third_body_neptune_de440s(epoch, r),
                    },
                };
                a_total += a_tb;
            }
        }

        // ===== RELATIVITY =====
        if force_config.relativity {
            a_total += accel_relativity(x_eci);
        }

        // Build state derivative: [vx, vy, vz, ax, ay, az, ...]
        let mut dx = DVector::zeros(state.len());
        dx[0] = v[0];
        dx[1] = v[1];
        dx[2] = v[2];
        dx[3] = a_total[0];
        dx[4] = a_total[1];
        dx[5] = a_total[2];

        // Additional state elements (if any) have zero derivative by default
        // These will be overridden if additional_dynamics is provided

        dx
    }

    // =========================================================================
    // Configuration
    // =========================================================================

    /// Set trajectory storage mode
    pub fn set_trajectory_mode(&mut self, mode: TrajectoryMode) {
        self.trajectory_mode = mode;
    }

    /// Get trajectory storage mode
    pub fn trajectory_mode(&self) -> TrajectoryMode {
        self.trajectory_mode
    }

    /// Enable output step (dense output mode)
    pub fn set_output_step(&mut self, _step: f64) {
        // Future: implement dense output
    }

    /// Disable output step
    pub fn disable_output_step(&mut self) {
        // Future: implement dense output
    }

    // =========================================================================
    // State Access
    // =========================================================================

    /// Get current state in ECI Cartesian format (reference)
    ///
    /// This returns a reference to the internal state vector, which is more
    /// efficient than the trait's `current_state()` method that returns a clone.
    ///
    /// # Returns
    /// Reference to current state vector in ECI Cartesian format
    pub fn current_state_ref(&self) -> &DVector<f64> {
        &self.x_curr
    }

    /// Get current parameters
    pub fn current_params(&self) -> &DVector<f64> {
        &self.params
    }

    /// Get trajectory (in ECI Cartesian format)
    pub fn trajectory(&self) -> &DOrbitTrajectory {
        &self.trajectory
    }

    /// Get current STM
    pub fn stm(&self) -> Option<&DMatrix<f64>> {
        self.stm.as_ref()
    }

    /// Get current sensitivity matrix
    pub fn sensitivity(&self) -> Option<&DMatrix<f64>> {
        self.sensitivity.as_ref()
    }

    /// Get STM at a specific index in the trajectory
    ///
    /// Returns the STM stored at the specified trajectory index, if STM
    /// history storage was enabled and the index is valid.
    ///
    /// # Arguments
    /// * `index` - Index into the trajectory storage
    ///
    /// # Returns
    /// Reference to the STM matrix if available, None otherwise
    pub fn stm_at_idx(&self, index: usize) -> Option<&DMatrix<f64>> {
        self.trajectory.stm_at_idx(index)
    }

    /// Get sensitivity matrix at a specific index in the trajectory
    ///
    /// Returns the sensitivity matrix stored at the specified trajectory index,
    /// if sensitivity history storage was enabled and the index is valid.
    ///
    /// # Arguments
    /// * `index` - Index into the trajectory storage
    ///
    /// # Returns
    /// Reference to the sensitivity matrix if available, None otherwise
    pub fn sensitivity_at_idx(&self, index: usize) -> Option<&DMatrix<f64>> {
        self.trajectory.sensitivity_at_idx(index)
    }

    /// Get STM at a specific epoch (with interpolation)
    ///
    /// Returns the STM at the specified epoch by interpolating between stored
    /// trajectory points. Requires STM history storage to be enabled.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to retrieve the STM
    ///
    /// # Returns
    /// Interpolated STM matrix if available, None otherwise
    pub fn stm_at(&self, epoch: Epoch) -> Option<DMatrix<f64>> {
        self.trajectory.stm_at(epoch)
    }

    /// Get sensitivity matrix at a specific epoch (with interpolation)
    ///
    /// Returns the sensitivity matrix at the specified epoch by interpolating
    /// between stored trajectory points. Requires sensitivity history storage
    /// to be enabled.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to retrieve the sensitivity matrix
    ///
    /// # Returns
    /// Interpolated sensitivity matrix if available, None otherwise
    pub fn sensitivity_at(&self, epoch: Epoch) -> Option<DMatrix<f64>> {
        self.trajectory.sensitivity_at(epoch)
    }

    /// Check if propagation was terminated
    pub fn terminated(&self) -> bool {
        self.terminated
    }

    // =========================================================================
    // Event Detection API
    // =========================================================================

    /// Add an event detector to monitor during propagation
    ///
    /// Events are detected during each integration step and processed according
    /// to the smart sequential algorithm.
    ///
    /// # Arguments
    /// * `detector` - Event detector implementing the `DEventDetector` trait
    pub fn add_event_detector(&mut self, detector: Box<dyn DEventDetector>) {
        self.event_detectors.push(detector);
    }

    /// Get all detected events
    ///
    /// Returns a slice of all events that have been detected during propagation.
    pub fn event_log(&self) -> &[DDetectedEvent] {
        &self.event_log
    }

    /// Get events by name (substring match)
    ///
    /// Returns all events whose name contains the specified substring.
    ///
    /// # Arguments
    /// * `name` - Substring to search for in event names
    pub fn events_by_name(&self, name: &str) -> Vec<&DDetectedEvent> {
        self.event_log
            .iter()
            .filter(|e| e.name.contains(name))
            .collect()
    }

    /// Get the most recently detected event
    ///
    /// Returns `None` if no events have been detected.
    pub fn latest_event(&self) -> Option<&DDetectedEvent> {
        self.event_log.last()
    }

    /// Get events in a time range
    ///
    /// Returns all events that occurred between the start and end epochs (inclusive).
    ///
    /// # Arguments
    /// * `start` - Start of time range
    /// * `end` - End of time range
    pub fn events_in_range(&self, start: Epoch, end: Epoch) -> Vec<&DDetectedEvent> {
        self.event_log
            .iter()
            .filter(|e| e.window_open >= start && e.window_open <= end)
            .collect()
    }

    /// Clear the event log
    ///
    /// Removes all detected events from the log. Event detectors are not removed.
    pub fn clear_events(&mut self) {
        self.event_log.clear();
    }

    /// Reset the termination flag
    ///
    /// Allows propagation to continue after a terminal event. The event log
    /// is not cleared.
    pub fn reset_termination(&mut self) {
        self.terminated = false;
    }

    // =========================================================================
    // Propagation
    // =========================================================================

    /// Take a single adaptive integration step (internal helper)
    ///
    /// Propagates the state forward by one adaptive timestep. Depending on the propagation
    /// mode configured at construction, also propagates STM and/or sensitivity matrices.
    ///
    /// Includes event detection loop: if events with callbacks are detected, integration
    /// restarts from the event time with the modified state/parameters.
    fn step_once(&mut self) {
        loop {
            // Save state before integration step (for event detection)
            let epoch_prev = self.current_epoch();
            let x_prev = self.x_curr.clone();

            // Use adaptive step size (dt_next from previous step)
            let dt_requested = self.dt_next;

            // Dispatch to appropriate integrator method based on propagation mode
            match self.propagation_mode {
                PropagationMode::StateOnly => {
                    // Basic state propagation only
                    let result =
                        self.integrator
                            .step(self.t_rel, self.x_curr.clone(), Some(dt_requested));

                    self.x_curr = result.state;
                    self.dt = result.dt_used;
                    self.dt_next = result.dt_next;
                }
                PropagationMode::WithSTM => {
                    // Propagate state and STM (variational matrix)
                    let phi = self.stm.take().unwrap();

                    let result = self.integrator.step_with_varmat(
                        self.t_rel,
                        self.x_curr.clone(),
                        phi,
                        Some(dt_requested),
                    );

                    self.x_curr = result.state;
                    self.stm = result.phi;
                    self.dt = result.dt_used;
                    self.dt_next = result.dt_next;

                    // Propagate covariance if initial covariance was provided
                    // P(t) = Φ(t, t₀) * P(t₀) * Φ(t, t₀)^T
                    if let Some(ref p0) = self.initial_covariance
                        && let Some(ref phi_result) = self.stm
                    {
                        let p_new = phi_result * p0 * phi_result.transpose();
                        self.current_covariance = Some(p_new);
                    }
                }
                PropagationMode::WithSensitivity => {
                    // Propagate state and sensitivity
                    let sens = self.sensitivity.take().unwrap();

                    let result = self.integrator.step_with_sensmat(
                        self.t_rel,
                        self.x_curr.clone(),
                        sens,
                        &self.params,
                        Some(dt_requested),
                    );

                    self.x_curr = result.state;
                    self.sensitivity = result.sens;
                    self.dt = result.dt_used;
                    self.dt_next = result.dt_next;
                }
                PropagationMode::WithSTMAndSensitivity => {
                    // Propagate state, STM, and sensitivity
                    let phi = self.stm.take().unwrap();
                    let sens = self.sensitivity.take().unwrap();

                    let result = self.integrator.step_with_varmat_sensmat(
                        self.t_rel,
                        self.x_curr.clone(),
                        phi,
                        sens,
                        &self.params,
                        Some(dt_requested),
                    );

                    self.x_curr = result.state;
                    self.stm = result.phi.clone();
                    self.sensitivity = result.sens;
                    self.dt = result.dt_used;
                    self.dt_next = result.dt_next;

                    // Propagate covariance if initial covariance was provided
                    // P(t) = Φ(t, t₀) * P(t₀) * Φ(t, t₀)^T
                    if let Some(ref p0) = self.initial_covariance
                        && let Some(ref phi_result) = self.stm
                    {
                        let p_new = phi_result * p0 * phi_result.transpose();
                        self.current_covariance = Some(p_new);
                    }
                }
            }

            // Update time (use actual dt_used)
            self.t_rel += self.dt;
            self.epoch_current = self.epoch_initial + self.t_rel;
            let epoch_new = self.epoch_current;

            // Scan for events in [epoch_prev, epoch_new]
            let detected_events =
                self.scan_all_events(epoch_prev, epoch_new, &x_prev, &self.x_curr);

            // Process events using smart sequential algorithm
            match self.process_events_smart(detected_events) {
                EventProcessingResult::NoEvents => {
                    // No events or all events processed without callbacks
                    // Accept step and store in trajectory if needed
                    if self.should_store_state() {
                        // Prepare optional matrices for storage
                        let cov = self.current_covariance.clone();
                        let stm_to_store = if self.store_stm_history {
                            self.stm.clone()
                        } else {
                            None
                        };
                        let sens_to_store = if self.store_sensitivity_history {
                            self.sensitivity.clone()
                        } else {
                            None
                        };

                        // Use add_full if any optional data is present, otherwise use add
                        if cov.is_some() || stm_to_store.is_some() || sens_to_store.is_some() {
                            self.trajectory.add_full(
                                epoch_new,
                                self.x_curr.clone(),
                                cov,
                                stm_to_store,
                                sens_to_store,
                            );
                        } else {
                            self.trajectory.add(epoch_new, self.x_curr.clone());
                        }
                    }
                    break; // Exit event loop
                }

                EventProcessingResult::Restart { epoch, state } => {
                    // Event callback modified state/params
                    // Reset to event time and state, then continue loop
                    self.t_rel = epoch - self.epoch_initial;
                    self.epoch_current = epoch;
                    self.x_curr = state;
                    // Note: process_events_smart() already updated self.params if needed
                    continue; // Take new step from event time
                }

                EventProcessingResult::Terminal => {
                    // Terminal event detected
                    // Store final state and exit (terminated flag already set)
                    if self.should_store_state() {
                        // Prepare optional matrices for storage
                        let cov = self.current_covariance.clone();
                        let stm_to_store = if self.store_stm_history {
                            self.stm.clone()
                        } else {
                            None
                        };
                        let sens_to_store = if self.store_sensitivity_history {
                            self.sensitivity.clone()
                        } else {
                            None
                        };

                        // Use add_full if any optional data is present, otherwise use add
                        if cov.is_some() || stm_to_store.is_some() || sens_to_store.is_some() {
                            self.trajectory.add_full(
                                epoch_new,
                                self.x_curr.clone(),
                                cov,
                                stm_to_store,
                                sens_to_store,
                            );
                        } else {
                            self.trajectory.add(epoch_new, self.x_curr.clone());
                        }
                    }
                    break; // Exit event loop
                }
            }
        }
    }

    /// Helper to determine if current state should be stored
    fn should_store_state(&self) -> bool {
        match self.trajectory_mode {
            TrajectoryMode::OutputStepsOnly => true,
            TrajectoryMode::AllSteps => true,
            TrajectoryMode::Disabled => false,
        }
    }

    // =========================================================================
    // State/Covariance Access with On-Demand Propagation
    // =========================================================================

    /// Ensure trajectory has data at or beyond the requested epoch.
    ///
    /// This method propagates forward or backward as needed to ensure the
    /// trajectory covers the requested epoch. The propagation direction is
    /// determined by the sign of `step_size()`.
    ///
    /// # Arguments
    /// * `epoch` - The epoch to ensure coverage for
    ///
    /// # Panics
    /// Panics if the requested epoch is in the opposite direction of propagation
    /// (i.e., past epoch requested in forward mode, or future epoch in backward mode).
    pub fn ensure_trajectory_coverage(&mut self, epoch: Epoch) {
        use crate::trajectories::traits::Trajectory;

        // Get trajectory bounds
        let traj_start = self.trajectory.start_epoch();
        let traj_end = self.trajectory.end_epoch();

        if let (Some(t0), Some(t1)) = (traj_start, traj_end)
            && epoch >= t0
            && epoch <= t1
        {
            // Epoch is within trajectory bounds, no propagation needed
            return;
        }

        // Need to propagate - determine direction based on step_size sign
        let current = self.current_epoch();
        let step = self.step_size();

        if step >= 0.0 {
            // Forward propagation mode
            if epoch > current {
                self.step_past(epoch);
            } else if epoch < current {
                panic!(
                    "Cannot propagate backwards to epoch {} when in forward propagation mode (step_size={}, current={})",
                    epoch, step, current
                );
            }
        } else {
            // Backward propagation mode
            if epoch < current {
                self.step_past(epoch);
            } else if epoch > current {
                panic!(
                    "Cannot propagate forwards to epoch {} when in backward propagation mode (step_size={}, current={})",
                    epoch, step, current
                );
            }
        }
    }
}

// =============================================================================
// DStatePropagator Trait Implementation
// =============================================================================

impl super::traits::DStatePropagator for DNumericalOrbitPropagator {
    fn step_by(&mut self, step_size: f64) {
        let target_t = self.t_rel + step_size;

        // Take adaptive steps until we've advanced by at least step_size
        while self.t_rel < target_t && !self.terminated {
            // Calculate remaining time
            let remaining = target_t - self.t_rel;

            // Guard against very small steps
            if remaining <= 1e-12 {
                break;
            }

            // Limit next step to not overshoot target
            // But allow adaptive integrator to suggest smaller steps
            let dt_max = remaining.min(self.dt_next);

            // Temporarily set the suggested next step to not overshoot
            let saved_dt_next = self.dt_next;
            self.dt_next = dt_max;

            // Take one adaptive step (includes event detection)
            self.step_once();

            // If we overshot, we're done (shouldn't happen with limiting above)
            if self.t_rel >= target_t {
                break;
            }

            // Restore suggested dt_next for subsequent steps
            // (unless step_once updated it to something smaller)
            if self.dt_next > saved_dt_next {
                self.dt_next = saved_dt_next;
            }
        }
    }

    fn propagate_to(&mut self, target_epoch: Epoch) {
        let target_rel = target_epoch - self.epoch_initial;

        while self.t_rel < target_rel && !self.terminated {
            // Calculate remaining time
            let remaining = target_rel - self.t_rel;

            // Guard against very small steps
            if remaining <= 1e-12 {
                break;
            }

            // Limit next step to not overshoot target
            let dt_max = remaining.min(self.dt_next);
            let saved_dt_next = self.dt_next;
            self.dt_next = dt_max;

            // Take one adaptive step (includes event detection)
            self.step_once();

            // Restore suggested dt_next for subsequent steps
            if self.dt_next > saved_dt_next {
                self.dt_next = saved_dt_next;
            }
        }
    }

    fn current_epoch(&self) -> Epoch {
        self.epoch_current
    }

    fn current_state(&self) -> DVector<f64> {
        self.x_curr.clone()
    }

    fn initial_epoch(&self) -> Epoch {
        self.epoch_initial
    }

    fn initial_state(&self) -> DVector<f64> {
        self.convert_state_from_eci(&self.x_initial, self.epoch_initial)
            .expect("State conversion from ECI to user format failed")
    }

    fn state_dim(&self) -> usize {
        self.state_dim
    }

    fn step_size(&self) -> f64 {
        self.dt
    }

    fn set_step_size(&mut self, step_size: f64) {
        self.dt = step_size;
        self.dt_next = step_size;
    }

    fn reset(&mut self) {
        // Reset time
        self.t_rel = 0.0;
        self.epoch_current = self.epoch_initial;

        // Reset state
        self.x_curr = self.x_initial.clone();

        // Reset step size to initial value
        let initial_dt = self.dt; // Keep the user-set step size
        self.dt_next = initial_dt;

        // Reset STM and/or sensitivity based on propagation mode
        match self.propagation_mode {
            PropagationMode::StateOnly => {
                self.stm = None;
                self.sensitivity = None;
            }
            PropagationMode::WithSTM => {
                self.stm = Some(DMatrix::identity(self.state_dim, self.state_dim));
                self.sensitivity = None;
            }
            PropagationMode::WithSensitivity => {
                self.stm = None;
                let param_dim = self.params.len();
                self.sensitivity = Some(DMatrix::zeros(self.state_dim, param_dim));
            }
            PropagationMode::WithSTMAndSensitivity => {
                self.stm = Some(DMatrix::identity(self.state_dim, self.state_dim));
                let param_dim = self.params.len();
                self.sensitivity = Some(DMatrix::zeros(self.state_dim, param_dim));
            }
        }

        // Clear trajectory
        self.trajectory =
            DOrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);

        // Clear event state
        self.event_log.clear();
        self.terminated = false;
    }

    fn set_eviction_policy_max_size(&mut self, max_size: usize) -> Result<(), BraheError> {
        self.trajectory.set_eviction_policy_max_size(max_size)
    }

    fn set_eviction_policy_max_age(&mut self, max_age: f64) -> Result<(), BraheError> {
        self.trajectory.set_eviction_policy_max_age(max_age)
    }
}

// =============================================================================
// InterpolationConfig Trait
// =============================================================================

impl InterpolationConfig for DNumericalOrbitPropagator {
    fn with_interpolation_method(mut self, method: InterpolationMethod) -> Self {
        self.interpolation_method = method;
        self.trajectory.set_interpolation_method(method);
        self
    }

    fn set_interpolation_method(&mut self, method: InterpolationMethod) {
        self.interpolation_method = method;
        self.trajectory.set_interpolation_method(method);
    }

    fn get_interpolation_method(&self) -> InterpolationMethod {
        self.interpolation_method
    }
}

// =============================================================================
// CovarianceInterpolationConfig Trait
// =============================================================================

impl CovarianceInterpolationConfig for DNumericalOrbitPropagator {
    fn with_covariance_interpolation_method(
        mut self,
        method: CovarianceInterpolationMethod,
    ) -> Self {
        self.covariance_interpolation_method = method;
        self.trajectory.set_covariance_interpolation_method(method);
        self
    }

    fn set_covariance_interpolation_method(&mut self, method: CovarianceInterpolationMethod) {
        self.covariance_interpolation_method = method;
        self.trajectory.set_covariance_interpolation_method(method);
    }

    fn get_covariance_interpolation_method(&self) -> CovarianceInterpolationMethod {
        self.covariance_interpolation_method
    }
}

// =============================================================================
// DStateProvider Trait
// =============================================================================

impl DStateProvider for DNumericalOrbitPropagator {
    fn state(&self, epoch: Epoch) -> Result<DVector<f64>, BraheError> {
        // Try to interpolate from trajectory
        if let Ok(state) = self.trajectory.interpolate(&epoch) {
            return Ok(state);
        }

        // If epoch matches current, return current state (always allowed)
        if (self.current_epoch() - epoch).abs() < 1e-9 {
            return Ok(self.x_curr.clone());
        }

        // Return error - epoch is neither in trajectory nor at current time
        let start = self.trajectory.start_epoch().unwrap_or(self.epoch_initial);
        let end = self.current_epoch();
        Err(BraheError::OutOfBoundsError(format!(
            "Cannot get state at epoch {}: outside propagator time range [{}, {}]. \
             Call step_by() or propagate_to() to advance the propagator first.",
            epoch, start, end
        )))
    }

    fn state_dim(&self) -> usize {
        self.state_dim
    }
}

// =============================================================================
// DOrbitStateProvider Trait
// =============================================================================

impl DOrbitStateProvider for DNumericalOrbitPropagator {
    fn state_eci(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        // Try to interpolate from trajectory
        if let Ok(state) = self.trajectory.interpolate(&epoch) {
            return Ok(state.fixed_rows::<6>(0).into());
        }

        // If at current epoch, return current state (always allowed)
        if (self.current_epoch() - epoch).abs() < 1e-9 {
            return Ok(self.x_curr.fixed_rows::<6>(0).into());
        }

        // Return error - epoch is neither in trajectory nor at current time
        let start = self.trajectory.start_epoch().unwrap_or(self.epoch_initial);
        let end = self.current_epoch();
        Err(BraheError::OutOfBoundsError(format!(
            "Cannot get state at epoch {}: outside propagator time range [{}, {}]. \
             Call step_by() or propagate_to() to advance the propagator first.",
            epoch, start, end
        )))
    }

    fn state_ecef(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        let eci_state = self.state_eci(epoch)?;
        Ok(crate::frames::state_eci_to_ecef(epoch, eci_state))
    }

    fn state_gcrf(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        // For now, GCRF ≈ ECI (very close for most applications)
        self.state_eci(epoch)
    }

    fn state_itrf(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        let gcrf_state = self.state_gcrf(epoch)?;
        Ok(crate::frames::state_gcrf_to_itrf(epoch, gcrf_state))
    }

    fn state_eme2000(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        let gcrf_state = self.state_gcrf(epoch)?;
        Ok(crate::frames::state_gcrf_to_eme2000(gcrf_state))
    }

    fn state_as_osculating_elements(
        &self,
        epoch: Epoch,
        angle_format: AngleFormat,
    ) -> Result<Vector6<f64>, BraheError> {
        let eci_state = self.state_eci(epoch)?;
        Ok(state_cartesian_to_osculating(eci_state, angle_format))
    }
}

// =============================================================================
// DCovarianceProvider Trait
// =============================================================================

impl DCovarianceProvider for DNumericalOrbitPropagator {
    fn covariance(&self, epoch: Epoch) -> Result<DMatrix<f64>, BraheError> {
        // Check if covariance tracking is enabled
        if self.current_covariance.is_none() {
            return Err(BraheError::InitializationError(
                "Covariance not available: covariance tracking was not enabled for this propagator"
                    .to_string(),
            ));
        }

        // Check bounds
        if let Some(start) = self.trajectory.start_epoch()
            && epoch < start
        {
            return Err(BraheError::OutOfBoundsError(format!(
                "Cannot get covariance at epoch {}: before trajectory start {}. \
                 Call step_by() or propagate_to() to advance the propagator first.",
                epoch, start
            )));
        }
        if let Some(end) = self.trajectory.end_epoch()
            && epoch > end
        {
            return Err(BraheError::OutOfBoundsError(format!(
                "Cannot get covariance at epoch {}: after trajectory end {}. \
                 Call step_by() or propagate_to() to advance the propagator first.",
                epoch, end
            )));
        }

        // Try to get from trajectory
        self.trajectory.covariance_at(epoch).ok_or_else(|| {
            BraheError::OutOfBoundsError(format!(
                "Cannot get covariance at epoch {}: no covariance data available at this epoch",
                epoch
            ))
        })
    }

    fn covariance_dim(&self) -> usize {
        self.state_dim
    }
}

// =============================================================================
// DOrbitCovarianceProvider Trait
// =============================================================================

impl DOrbitCovarianceProvider for DNumericalOrbitPropagator {
    fn covariance_eci(&self, epoch: Epoch) -> Result<DMatrix<f64>, BraheError> {
        // Native frame is ECI
        DCovarianceProvider::covariance(self, epoch)
    }

    fn covariance_gcrf(&self, epoch: Epoch) -> Result<DMatrix<f64>, BraheError> {
        // GCRF ≈ ECI for most applications
        DCovarianceProvider::covariance(self, epoch)
    }

    fn covariance_rtn(&self, epoch: Epoch) -> Result<DMatrix<f64>, BraheError> {
        let cov_eci = DCovarianceProvider::covariance(self, epoch)?;

        // Get state at this epoch for RTN rotation
        let state_eci = self.state_eci(epoch)?;

        // Compute RTN rotation matrix using the library function
        let rot_eci_to_rtn = rotation_eci_to_rtn(state_eci);

        // Extract position and velocity
        let r = state_eci.fixed_rows::<3>(0);
        let v = state_eci.fixed_rows::<3>(3);

        // Get angular velocity of RTN frame with respect to ECI frame (Alfriend equation 2.16)
        let f_dot = (r.cross(&v)).norm() / (r.norm().powi(2));
        let omega = nalgebra::Vector3::new(0.0, 0.0, f_dot);

        // Build skew-symmetric matrix of omega
        let omega_skew = nalgebra::SMatrix::<f64, 3, 3>::new(
            0.0, -omega[2], omega[1], omega[2], 0.0, -omega[0], -omega[1], omega[0], 0.0,
        );

        let j21 = -omega_skew * rot_eci_to_rtn;

        // Build full transformation Jacobian for dynamic-sized covariance
        let dim = cov_eci.nrows();
        let mut jacobian = DMatrix::<f64>::zeros(dim, dim);

        // Block diagonal rotation parts (6x6 core)
        for i in 0..3 {
            for j in 0..3 {
                jacobian[(i, j)] = rot_eci_to_rtn[(i, j)];
                jacobian[(3 + i, 3 + j)] = rot_eci_to_rtn[(i, j)];
            }
        }

        // Off-diagonal parts due to angular velocity
        for i in 3..6 {
            for j in 0..3 {
                jacobian[(i, j)] = j21[(i - 3, j)];
            }
        }

        // For extended state dimensions (beyond 6D), leave as identity
        for i in 6..dim {
            jacobian[(i, i)] = 1.0;
        }

        // Transform covariance: C_RTN = J * C_ECI * J^T
        let cov_rtn = &jacobian * &cov_eci * jacobian.transpose();

        Ok(cov_rtn)
    }
}

// =============================================================================
// Identifiable Trait
// =============================================================================

impl Identifiable for DNumericalOrbitPropagator {
    fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    fn with_uuid(mut self, uuid: uuid::Uuid) -> Self {
        self.uuid = Some(uuid);
        self
    }

    fn with_new_uuid(mut self) -> Self {
        self.uuid = Some(uuid::Uuid::new_v4());
        self
    }

    fn with_id(mut self, id: u64) -> Self {
        self.id = Some(id);
        self
    }

    fn with_identity(
        mut self,
        name: Option<&str>,
        uuid: Option<uuid::Uuid>,
        id: Option<u64>,
    ) -> Self {
        self.name = name.map(|s| s.to_string());
        self.uuid = uuid;
        self.id = id;
        self
    }

    fn set_identity(&mut self, name: Option<&str>, uuid: Option<uuid::Uuid>, id: Option<u64>) {
        self.name = name.map(|s| s.to_string());
        self.uuid = uuid;
        self.id = id;
    }

    fn set_id(&mut self, id: Option<u64>) {
        self.id = id;
    }

    fn set_name(&mut self, name: Option<&str>) {
        self.name = name.map(|s| s.to_string());
    }

    fn generate_uuid(&mut self) {
        self.uuid = Some(uuid::Uuid::new_v4());
    }

    fn get_id(&self) -> Option<u64> {
        self.id
    }

    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn get_uuid(&self) -> Option<uuid::Uuid> {
        self.uuid
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eop::{EOPExtrapolation, FileEOPProvider, set_global_eop_provider};
    use crate::events::{DAltitudeEvent, DTimeEvent, EventDirection};
    use crate::propagators::NumericalPropagationConfig;
    use crate::propagators::force_model_config::{
        AtmosphericModel, DragConfiguration, ParameterSource,
    };
    use crate::propagators::traits::DStatePropagator;
    use crate::time::TimeSystem;
    use crate::{orbital_period, state_osculating_to_cartesian};

    fn setup_global_test_eop() {
        let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
        set_global_eop_provider(eop);
    }

    /// Create default spacecraft parameters for tests
    fn default_test_params() -> DVector<f64> {
        DVector::from_vec(vec![
            1000.0, // mass [kg]
            10.0,   // drag area [m²]
            2.2,    // Cd
            10.0,   // SRP area [m²]
            1.3,    // Cr
        ])
    }

    /// Create a test-friendly force config that uses point mass gravity but has drag
    /// This allows testing parameter-dependent features without loading gravity model
    fn test_force_config_with_params() -> ForceModelConfiguration {
        ForceModelConfiguration {
            gravity: GravityConfiguration::PointMass,
            drag: Some(DragConfiguration {
                model: AtmosphericModel::HarrisPriester,
                area: ParameterSource::ParameterIndex(1),
                cd: ParameterSource::ParameterIndex(2),
            }),
            srp: None,
            third_body: None,
            relativity: false,
            mass: None, // Uses params[0] by default
        }
    }

    #[test]
    fn test_dnumericalorbitpropagator_construction_default() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![
            R_EARTH + 500e3,
            0.0,
            0.0, // position
            0.0,
            7500.0,
            0.0, // velocity
        ]);

        // Test construction with default integrator (DP54)
        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::default(),
            Some(default_test_params()),
            None,
            None,
        );

        assert!(prop.is_ok());
    }

    #[test]
    fn test_force_model_configuration_construction_variants() {
        let _ = ForceModelConfiguration::default();
        let _ = ForceModelConfiguration::high_fidelity();
        let _ = ForceModelConfiguration::gravity_only();
        let _ = ForceModelConfiguration::leo_default();
        let _ = ForceModelConfiguration::geo_default();
    }

    #[test]
    fn test_dnumericalorbitpropagator_dstate_propagator_step_by() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![
            R_EARTH + 500e3,
            0.0,
            0.0, // position
            0.0,
            7500.0,
            0.0, // velocity
        ]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        )
        .unwrap();

        // Test that the propagator implements DStatePropagator
        fn use_trait<P: DStatePropagator>(prop: &mut P) {
            let initial_epoch = prop.initial_epoch();
            let _initial_state = prop.initial_state();
            let _state_dim = prop.state_dim();

            // Take a step
            prop.step_by(60.0);

            let current_epoch = prop.current_epoch();
            let _current_state = prop.current_state();

            // Verify time advanced
            assert!((current_epoch - initial_epoch - 60.0).abs() < 1.0);
        }

        use_trait(&mut prop);
    }

    // =========================================================================
    // Event Detection Tests
    // =========================================================================

    #[test]
    fn test_dnumericalorbitpropagator_event_detection_api_methods() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        )
        .unwrap();

        // Test initial state
        assert_eq!(prop.event_log().len(), 0);
        assert!(!prop.terminated());
        assert!(prop.latest_event().is_none());

        // Add an event detector
        let time_event = DTimeEvent::new(epoch + 3600.0, "Test Event");
        prop.add_event_detector(Box::new(time_event));

        // Event log still empty (not detected yet)
        assert_eq!(prop.event_log().len(), 0);

        // Test reset_termination
        prop.terminated = true;
        assert!(prop.terminated());
        prop.reset_termination();
        assert!(!prop.terminated());

        // Test clear_events
        prop.clear_events();
        assert_eq!(prop.event_log().len(), 0);
    }

    #[test]
    fn test_dnumericalorbitpropagator_event_detection_time_event() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        )
        .unwrap();

        // Add time event at 30 minutes
        let event_time = epoch + 1800.0;
        let time_event = DTimeEvent::new(event_time, "30 Minute Mark");
        prop.add_event_detector(Box::new(time_event));

        // Propagate to 1 hour
        prop.propagate_to(epoch + 3600.0);

        // Event should be detected
        let events = prop.event_log();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "30 Minute Mark");

        // Event time should be close to 30 minutes
        let event_epoch = events[0].window_open;
        assert!((event_epoch - event_time).abs() < 0.1);
    }

    #[test]
    fn test_dnumericalorbitpropagator_event_detection_altitude_event() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        // Start with elliptical orbit that crosses 450 km altitude
        let a = R_EARTH + 500e3; // 500 km semi-major axis
        let e = 0.02; // Small eccentricity
        let i = 0.0;
        let raan = 0.0;
        let argp = 0.0;
        let ta = 0.0;

        let oe = DVector::from_vec(vec![a, e, i, raan, argp, ta]);
        let state = state_osculating_to_cartesian(
            Vector6::from_column_slice(oe.as_slice()),
            AngleFormat::Radians,
        );

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            DVector::from_column_slice(state.as_slice()),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        )
        .unwrap();

        // Add altitude event at 450 km (detect both increasing and decreasing)
        let alt_event = DAltitudeEvent::new(450e3, "Low Alt", EventDirection::Any);
        prop.add_event_detector(Box::new(alt_event));

        // Propagate for one orbit period
        let period = 2.0 * orbital_period(a);
        prop.propagate_to(epoch + period);

        // Should detect 2 events for elliptical orbit (ascending and descending)
        let events = prop.event_log();
        assert!(
            !events.is_empty(),
            "Expected at least 1 altitude crossing (should be 2), got {}. Period: {} s",
            events.len(),
            period
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_event_detection_no_altitude_events() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        // Start with elliptical orbit that crosses 450 km altitude
        let a = R_EARTH + 500e3; // 500 km semi-major axis
        let e = 0.00; // Small eccentricity
        let i = 0.0;
        let raan = 0.0;
        let argp = 0.0;
        let ta = 0.0;

        let oe = DVector::from_vec(vec![a, e, i, raan, argp, ta]);
        let state = state_osculating_to_cartesian(
            Vector6::from_column_slice(oe.as_slice()),
            AngleFormat::Radians,
        );

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            DVector::from_column_slice(state.as_slice()),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        )
        .unwrap();

        // Add altitude event at 450 km (detect both increasing and decreasing)
        let alt_event = DAltitudeEvent::new(450e3, "Low Alt", EventDirection::Any);
        prop.add_event_detector(Box::new(alt_event));

        // Propagate for one orbit period
        let period = 2.0 * orbital_period(a);
        prop.propagate_to(epoch + period);

        // Should detect 2 events for elliptical orbit (ascending and descending)
        let events = prop.event_log();
        assert!(
            events.is_empty(),
            "Expected at least 0 altitude crossings, got {}. Period: {} s",
            events.len(),
            period
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_event_detection_callback_state_mutation() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state.clone(),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        )
        .unwrap();

        // Add event with callback that applies delta-V
        let maneuver_time = epoch + 1800.0;
        let delta_v = 10.0; // 10 m/s

        let maneuver = DTimeEvent::new(maneuver_time, "Maneuver").with_callback(Box::new(
            move |_t, state, _params| {
                let mut new_state = state.clone();
                new_state[3] += delta_v; // Add to vx
                (Some(new_state), None, EventAction::Continue)
            },
        ));

        prop.add_event_detector(Box::new(maneuver));

        // Propagate past maneuver
        prop.propagate_to(epoch + 3600.0);

        // Event should be detected
        assert_eq!(prop.event_log().len(), 1);

        // Verify propagation continued successfully
        assert!(!prop.terminated());
    }

    #[test]
    fn test_dnumericalorbitpropagator_event_detection_callback_parameter_mutation() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        // Use test-friendly force config that has params but point mass gravity
        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            test_force_config_with_params(),
            Some(default_test_params()),
            None,
            None,
        )
        .unwrap();

        // Get initial mass parameter
        let initial_mass = prop.current_params()[0];

        // Add event that changes mass
        let event_time = epoch + 1800.0;
        let new_mass = 500.0;

        let mass_update = DTimeEvent::new(event_time, "Mass Update").with_callback(Box::new(
            move |_t, _state, params| {
                let mut new_params = params.unwrap().clone();
                new_params[0] = new_mass; // Update mass
                (None, Some(new_params), EventAction::Continue)
            },
        ));

        prop.add_event_detector(Box::new(mass_update));

        // Propagate past event
        prop.propagate_to(epoch + 3600.0);

        // Parameter should have changed
        let final_mass = prop.current_params()[0];
        assert_ne!(initial_mass, final_mass);
        assert_eq!(final_mass, new_mass);
    }

    #[test]
    fn test_dnumericalorbitpropagator_event_detection_terminal_event() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        )
        .unwrap();

        // Add terminal event at 30 minutes
        let terminal_time = epoch + 1800.0;
        let terminal_event = DTimeEvent::new(terminal_time, "Terminal").is_terminal();

        prop.add_event_detector(Box::new(terminal_event));

        // Try to propagate to 1 hour
        prop.propagate_to(epoch + 3600.0);

        // Check if event was detected
        let events = prop.event_log();
        println!("Events detected: {}", events.len());
        if !events.is_empty() {
            println!(
                "Event: {} at {}",
                events[0].name,
                events[0].window_open - epoch
            );
        }

        // Should have stopped at 30 minutes
        assert!(
            prop.terminated(),
            "Propagator should be terminated but isn't. Events detected: {}",
            events.len()
        );
        let current = prop.current_epoch();
        assert!(
            (current - terminal_time).abs() < 10.0,
            "Current epoch {} is not close to terminal time {}",
            current - epoch,
            terminal_time - epoch
        ); // Within 10 seconds
    }

    #[test]
    fn test_dnumericalorbitpropagator_event_detection_multiple_no_callbacks() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        )
        .unwrap();

        // Add multiple time events without callbacks
        prop.add_event_detector(Box::new(DTimeEvent::new(epoch + 1200.0, "Event 1")));
        prop.add_event_detector(Box::new(DTimeEvent::new(epoch + 1800.0, "Event 2")));
        prop.add_event_detector(Box::new(DTimeEvent::new(epoch + 2400.0, "Event 3")));

        // Propagate
        prop.propagate_to(epoch + 3600.0);

        // All events should be detected
        let events = prop.event_log();
        assert_eq!(events.len(), 3);

        // Events should be in chronological order
        assert_eq!(events[0].name, "Event 1");
        assert_eq!(events[1].name, "Event 2");
        assert_eq!(events[2].name, "Event 3");
    }

    #[test]
    fn test_dnumericalorbitpropagator_event_detection_smart_processing() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        )
        .unwrap();

        // Add events: 2 without callbacks, then 1 with callback, then 1 without
        prop.add_event_detector(Box::new(DTimeEvent::new(epoch + 600.0, "Event 1 - No CB")));
        prop.add_event_detector(Box::new(DTimeEvent::new(epoch + 1200.0, "Event 2 - No CB")));

        let callback_event = DTimeEvent::new(epoch + 1800.0, "Event 3 - WITH CB").with_callback(
            Box::new(|_t, _state, _params| (None, None, EventAction::Continue)),
        );
        prop.add_event_detector(Box::new(callback_event));

        prop.add_event_detector(Box::new(DTimeEvent::new(epoch + 2400.0, "Event 4 - No CB")));

        // Propagate
        prop.propagate_to(epoch + 3600.0);

        // All events should be detected
        let events = prop.event_log();
        assert_eq!(events.len(), 4);

        // Verify they're in order
        assert_eq!(events[0].name, "Event 1 - No CB");
        assert_eq!(events[1].name, "Event 2 - No CB");
        assert_eq!(events[2].name, "Event 3 - WITH CB");
        assert_eq!(events[3].name, "Event 4 - No CB");
    }

    // =========================================================================
    // GROUP 1: Construction Tests
    // =========================================================================

    #[test]
    fn test_dnumericalorbitpropagator_construction_with_custom_params() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        // Create with custom parameters
        let force_config = ForceModelConfiguration::default();
        let params = DVector::from_vec(vec![500.0, 5.0, 2.0, 8.0, 1.5]); // Custom values

        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state.clone(),
            NumericalPropagationConfig::default(),
            force_config,
            Some(params),
            None,
            None,
        );

        assert!(prop.is_ok());
        let prop = prop.unwrap();
        assert_eq!(DStatePropagator::state_dim(&prop), 6);
        assert_eq!(prop.initial_epoch(), epoch);
    }

    #[test]
    fn test_dnumericalorbitpropagator_construction_keplerian_representation() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        // Input as Keplerian elements: [a, e, i, RAAN, argp, M]
        // Use circular orbit (e=0) for precise semi-major axis matching
        let keplerian_state = DVector::from_vec(vec![
            R_EARTH + 500e3,       // a [m]
            0.0,                   // e (circular)
            97.8_f64.to_radians(), // i [rad]
            0.0,                   // RAAN [rad]
            0.0,                   // argp [rad]
            0.0,                   // M [rad]
        ]);

        // Convert to ECI Cartesian first (constructor expects ECI Cartesian)
        let cartesian_state = state_osculating_to_cartesian(
            keplerian_state.fixed_rows::<6>(0).into_owned(),
            AngleFormat::Radians,
        );
        let cartesian_state_dvector = DVector::from_vec(cartesian_state.as_slice().to_vec());

        let prop = DNumericalOrbitPropagator::new(
            epoch,
            cartesian_state_dvector,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        );

        assert!(prop.is_ok());
        let prop = prop.unwrap();

        // Verify state is stored as ECI Cartesian internally
        let eci_state = prop.current_state();
        assert_eq!(eci_state.len(), 6);

        // Position magnitude should match semi-major axis (circular orbit)
        let r_mag = (eci_state[0].powi(2) + eci_state[1].powi(2) + eci_state[2].powi(2)).sqrt();
        assert!((r_mag - (R_EARTH + 500e3)).abs() < 10.0); // 10 m tolerance for numerical conversion
    }

    #[test]
    fn test_dnumericalorbitpropagator_construction_ecef_frame() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        // Create ECI state (constructor expects ECI Cartesian)
        let eci_state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let prop = DNumericalOrbitPropagator::new(
            epoch,
            eci_state.clone(),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        );

        assert!(prop.is_ok());
        let prop = prop.unwrap();

        // Internal state should match input ECI state
        let internal_eci = prop.current_state();
        assert!((internal_eci[0] - eci_state[0]).abs() < 1.0);
        assert!((internal_eci[1] - eci_state[1]).abs() < 1.0);
        assert!((internal_eci[2] - eci_state[2]).abs() < 1.0);
    }

    #[test]
    fn test_dnumericalorbitpropagator_construction_extended_state() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        // 6D orbital state + 2 additional states
        let extended_state = DVector::from_vec(vec![
            R_EARTH + 500e3,
            0.0,
            0.0, // position
            0.0,
            7500.0,
            0.0,   // velocity
            100.0, // additional state 1
            200.0, // additional state 2
        ]);

        let prop = DNumericalOrbitPropagator::new(
            epoch,
            extended_state.clone(),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        );

        assert!(prop.is_ok());
        let prop = prop.unwrap();
        assert_eq!(DStatePropagator::state_dim(&prop), 8);

        let current = prop.current_state();
        assert_eq!(current.len(), 8);
        // Without additional dynamics, extended states should maintain initial values
        assert_eq!(current[6], 100.0);
        assert_eq!(current[7], 200.0);
    }

    #[test]
    fn test_dnumericalorbitpropagator_construction_with_additional_dynamics() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        // 6D orbital state + 1 additional state (e.g., spacecraft mass)
        let extended_state = DVector::from_vec(vec![
            R_EARTH + 500e3,
            0.0,
            0.0,
            0.0,
            7500.0,
            0.0,
            1000.0, // mass [kg]
        ]);

        // Additional dynamics: mass depletion (e.g., -0.1 kg/s)
        // Returns full state-sized vector with additional contributions
        let additional_dynamics: DStateDynamics = Box::new(|_t, state, _params| {
            let mut dx = DVector::zeros(state.len());
            dx[6] = -0.1; // dm/dt = -0.1 kg/s
            dx
        });

        let prop = DNumericalOrbitPropagator::new(
            epoch,
            extended_state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            Some(additional_dynamics),
            None,
        );

        assert!(prop.is_ok());
        let mut prop = prop.unwrap();
        assert_eq!(DStatePropagator::state_dim(&prop), 7);

        let initial_mass = prop.current_state()[6];

        // Propagate for 10 seconds
        prop.step_by(10.0);

        let final_mass = prop.current_state()[6];

        // Mass should have decreased by approximately 1 kg (10s * 0.1 kg/s)
        assert!((final_mass - (initial_mass - 1.0)).abs() < 0.1);
    }

    #[test]
    fn test_dnumericalorbitpropagator_construction_multiple_integrators() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        use crate::propagators::IntegratorMethod;

        // Test with DormandPrince54 (default)
        let prop_dp54 = DNumericalOrbitPropagator::new(
            epoch,
            state.clone(),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        );
        assert!(prop_dp54.is_ok());

        // Test with RKF45
        let prop_rkf45 = DNumericalOrbitPropagator::new(
            epoch,
            state.clone(),
            NumericalPropagationConfig::with_method(IntegratorMethod::RKF45),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        );
        assert!(prop_rkf45.is_ok());

        // Test with RKN1210
        let prop_rkn = DNumericalOrbitPropagator::new(
            epoch,
            state.clone(),
            NumericalPropagationConfig::with_method(IntegratorMethod::RKN1210),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        );
        assert!(prop_rkn.is_ok());

        // Test with RK4
        let prop_rk4 = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::with_method(IntegratorMethod::RK4),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        );
        assert!(prop_rk4.is_ok());
    }

    // =========================================================================
    // GROUP 2: State Access Method Tests
    // =========================================================================

    #[test]
    fn test_dnumericalorbitpropagator_current_epoch() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        )
        .unwrap();

        // Initially should match input epoch
        assert_eq!(prop.current_epoch(), epoch);

        // After stepping, should advance
        prop.step_by(100.0);
        let new_epoch = prop.current_epoch();
        assert!((new_epoch - epoch - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_dnumericalorbitpropagator_current_state() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state.clone(),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        )
        .unwrap();

        let eci_state = prop.current_state();
        assert_eq!(eci_state.len(), 6);

        // Should match input state (already in ECI)
        for i in 0..6 {
            assert!((eci_state[i] - state[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_dnumericalorbitpropagator_current_params() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        // Test with gravity_only - no params needed
        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state.clone(),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        )
        .unwrap();

        let params = prop.current_params();
        // With gravity_only, no params are required so empty vec is stored
        assert_eq!(params.len(), 0);

        // Test with explicit params
        let custom_params = DVector::from_vec(vec![1000.0, 10.0, 2.2, 10.0, 1.3]);
        let prop_with_params = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::default(),
            Some(custom_params),
            None,
            None,
        )
        .unwrap();

        let params = prop_with_params.current_params();
        assert_eq!(params.len(), 5);
        assert_eq!(params[0], 1000.0); // mass
        assert_eq!(params[1], 10.0); // drag_area
        assert_eq!(params[2], 2.2); // Cd
        assert_eq!(params[3], 10.0); // srp_area
        assert_eq!(params[4], 1.3); // Cr
    }

    #[test]
    fn test_dnumericalorbitpropagator_initial_epoch() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 30, 45.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        )
        .unwrap();

        // Initial epoch should be immutable
        assert_eq!(prop.initial_epoch(), epoch);

        // Even after propagation
        prop.step_by(1000.0);
        assert_eq!(prop.initial_epoch(), epoch);
    }

    #[test]
    fn test_dnumericalorbitpropagator_initial_state() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state.clone(),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        )
        .unwrap();

        let initial = prop.initial_state();

        // Should match input state
        for i in 0..6 {
            assert!((initial[i] - state[i]).abs() < 1e-6);
        }

        // Should remain constant after propagation
        prop.step_by(1000.0);
        let initial_after = prop.initial_state();
        for i in 0..6 {
            assert!((initial_after[i] - state[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_dnumericalorbitpropagator_state_dim() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        // Test 6D state
        let state_6d = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let prop_6d = DNumericalOrbitPropagator::new(
            epoch,
            state_6d,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(DStatePropagator::state_dim(&prop_6d), 6);

        // Test extended state
        let state_8d = DVector::from_vec(vec![
            R_EARTH + 500e3,
            0.0,
            0.0,
            0.0,
            7500.0,
            0.0,
            100.0,
            200.0,
        ]);

        let prop_8d = DNumericalOrbitPropagator::new(
            epoch,
            state_8d,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(DStatePropagator::state_dim(&prop_8d), 8);
    }

    #[test]
    fn test_dnumericalorbitpropagator_trajectory_access() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        )
        .unwrap();

        // Initially trajectory should be empty
        let traj = prop.trajectory();
        assert_eq!(traj.len(), 0);

        // After propagation, trajectory should have states
        prop.step_by(100.0);
        prop.step_by(100.0);

        let traj_after = prop.trajectory();
        assert!(traj_after.len() > 0);
    }

    #[test]
    fn test_dnumericalorbitpropagator_stm_access() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        // Test 1: Without STM enabled (default config)
        let prop_no_stm = DNumericalOrbitPropagator::new(
            epoch,
            state.clone(),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        )
        .unwrap();

        // STM should be None (StateOnly mode)
        assert!(prop_no_stm.stm().is_none());

        // Test 2: With STM enabled via config
        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_stm = true;

        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            config,
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        )
        .unwrap();

        // Now STM should be Some (initialized to identity)
        assert!(prop.stm().is_some());
        let stm = prop.stm().unwrap();
        assert_eq!(stm.nrows(), 6);
        assert_eq!(stm.ncols(), 6);
    }

    #[test]
    fn test_dnumericalorbitpropagator_sensitivity_access() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        // Test 1: Without sensitivity enabled (default config)
        let prop_no_sens = DNumericalOrbitPropagator::new(
            epoch,
            state.clone(),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        )
        .unwrap();

        // Sensitivity should be None
        assert!(prop_no_sens.sensitivity().is_none());

        // Test 2: With sensitivity enabled via config (requires params)
        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_sensitivity = true;
        let params = DVector::from_vec(vec![1000.0, 10.0, 2.2, 10.0, 1.3]); // 5 params

        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            config,
            ForceModelConfiguration::gravity_only(),
            Some(params),
            None,
            None,
        )
        .unwrap();

        // Now sensitivity should be Some (initialized to zeros)
        assert!(prop.sensitivity().is_some());
        let sens = prop.sensitivity().unwrap();
        assert_eq!(sens.nrows(), 6);
        assert_eq!(sens.ncols(), 5);
    }

    #[test]
    fn test_dnumericalorbitpropagator_terminated_flag() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        )
        .unwrap();

        // Initially should not be terminated
        assert!(!prop.terminated());

        // Add terminal event
        let terminal_event = DTimeEvent::new(epoch + 100.0, "Terminal").is_terminal();
        prop.add_event_detector(Box::new(terminal_event));

        // Propagate past event
        prop.propagate_to(epoch + 200.0);

        // Should be terminated
        assert!(prop.terminated());
    }

    // =========================================================================
    // GROUP 3: Configuration Method Tests
    // =========================================================================

    #[test]
    fn test_dnumericalorbitpropagator_set_trajectory_mode() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        )
        .unwrap();

        // Test AllSteps mode (default)
        prop.set_trajectory_mode(TrajectoryMode::AllSteps);
        assert_eq!(prop.trajectory_mode(), TrajectoryMode::AllSteps);

        prop.step_by(100.0);
        prop.step_by(100.0);
        assert!(prop.trajectory().len() > 0);

        // Test Disabled mode
        prop.reset();
        prop.set_trajectory_mode(TrajectoryMode::Disabled);
        assert_eq!(prop.trajectory_mode(), TrajectoryMode::Disabled);

        prop.step_by(100.0);
        prop.step_by(100.0);
        assert_eq!(prop.trajectory().len(), 0); // No trajectory storage

        // Test OutputStepsOnly mode
        prop.reset();
        prop.set_trajectory_mode(TrajectoryMode::OutputStepsOnly);
        assert_eq!(prop.trajectory_mode(), TrajectoryMode::OutputStepsOnly);
    }

    #[test]
    fn test_dnumericalorbitpropagator_trajectory_mode_getter() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        )
        .unwrap();

        // Default should be AllSteps
        assert_eq!(prop.trajectory_mode(), TrajectoryMode::AllSteps);
    }

    #[test]
    fn test_dnumericalorbitpropagator_set_step_size() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        )
        .unwrap();

        // Set new step size
        prop.set_step_size(30.0);
        assert_eq!(prop.step_size(), 30.0);

        // Set different step size
        prop.set_step_size(120.0);
        assert_eq!(prop.step_size(), 120.0);
    }

    #[test]
    fn test_dnumericalorbitpropagator_step_size_getter() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        )
        .unwrap();

        // Default step size from IntegratorConfig::default()
        let step_size = prop.step_size();
        assert!(step_size > 0.0);
    }

    #[test]
    fn test_dnumericalorbitpropagator_set_eviction_policy_max_size() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        )
        .unwrap();

        // Set max size policy
        let result = prop.set_eviction_policy_max_size(10);
        assert!(result.is_ok());

        // Propagate and verify trajectory doesn't exceed max size
        for _ in 0..20 {
            prop.step_by(10.0);
        }

        let traj_len = prop.trajectory().len();
        assert!(
            traj_len <= 10,
            "Trajectory length {} exceeds max size 10",
            traj_len
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_set_eviction_policy_max_age() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        )
        .unwrap();

        // Set max age policy (100 seconds)
        let result = prop.set_eviction_policy_max_age(100.0);
        assert!(result.is_ok());

        // Propagate for 200 seconds
        for _ in 0..20 {
            prop.step_by(10.0);
        }

        // Trajectory should only contain recent states (within 100s)
        let traj = prop.trajectory();
        if traj.len() > 0 {
            let epochs = &traj.epochs;
            let current_time = prop.current_epoch() - prop.initial_epoch();
            let oldest_time = epochs[0] - prop.initial_epoch();

            // Oldest state should not be more than 100s old
            assert!(
                current_time - oldest_time <= 100.0 + 10.0,
                "Oldest state age {} exceeds max age 100s",
                current_time - oldest_time
            );
        }
    }

    // =========================================================================
    // GROUP 4: Force Model Tests (Comprehensive)
    // =========================================================================

    #[test]
    fn test_dnumericalorbitpropagator_force_gravity_point_mass() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let force_config = ForceModelConfiguration {
            mass: None,
            gravity: GravityConfiguration::PointMass,
            drag: None,
            srp: None,
            third_body: None,
            relativity: false,
        };

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state.clone(),
            NumericalPropagationConfig::default(),
            force_config,
            Some(default_test_params()),
            None,
            None,
        )
        .unwrap();

        let initial_energy = compute_orbital_energy(&state);

        // Propagate for one orbit
        prop.step_by(5400.0); // ~90 minutes

        let final_state = prop.current_state();
        let final_energy = compute_orbital_energy(&final_state);

        // Energy should be conserved with point mass gravity only
        assert!(
            (final_energy - initial_energy).abs() / initial_energy.abs() < 1e-6,
            "Energy not conserved: {} vs {}",
            initial_energy,
            final_energy
        );
    }

    fn compute_orbital_energy(state: &DVector<f64>) -> f64 {
        let r = (state[0].powi(2) + state[1].powi(2) + state[2].powi(2)).sqrt();
        let v = (state[3].powi(2) + state[4].powi(2) + state[5].powi(2)).sqrt();
        0.5 * v.powi(2) - GM_EARTH / r
    }

    #[test]
    fn test_dnumericalorbitpropagator_force_gravity_spherical_harmonic() {
        use crate::orbit_dynamics::gravity::GravityModelType;

        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        // Test with 4x4 spherical harmonic
        let force_config = ForceModelConfiguration {
            mass: None,
            gravity: GravityConfiguration::SphericalHarmonic {
                source: GravityModelSource::ModelType(GravityModelType::EGM2008_360),
                degree: 4,
                order: 4,
            },
            drag: None,
            srp: None,
            third_body: None,
            relativity: false,
        };

        // Note: Spherical harmonic gravity requires loading model data files
        // This test verifies construction succeeds; actual propagation would
        // require gravity model data to be loaded
        let prop_result = DNumericalOrbitPropagator::new(
            epoch,
            state.clone(),
            NumericalPropagationConfig::default(),
            force_config,
            Some(default_test_params()),
            None,
            None,
        );

        // Construction should succeed even without gravity data loaded
        assert!(
            prop_result.is_ok(),
            "Spherical harmonic config should construct successfully"
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_force_drag_harris_priester() {
        use crate::propagators::force_model_config::{DragConfiguration, ParameterSource};

        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![
            R_EARTH + 300e3,
            0.0,
            0.0, // Lower altitude for drag
            0.0,
            7700.0,
            0.0,
        ]);

        let force_config = ForceModelConfiguration {
            mass: None,
            gravity: GravityConfiguration::PointMass,
            drag: Some(DragConfiguration {
                model: AtmosphericModel::HarrisPriester,
                area: ParameterSource::Value(10.0),
                cd: ParameterSource::Value(2.2),
            }),
            srp: None,
            third_body: None,
            relativity: false,
        };

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state.clone(),
            NumericalPropagationConfig::default(),
            force_config,
            Some(default_test_params()),
            None,
            None,
        )
        .unwrap();

        let initial_energy = compute_orbital_energy(&state);

        // Propagate for 10 minutes
        prop.step_by(600.0);

        let final_state = prop.current_state();
        let final_energy = compute_orbital_energy(&final_state);

        // Drag should decrease energy
        assert!(
            final_energy < initial_energy,
            "Drag should decrease orbital energy"
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_force_drag_exponential() {
        use crate::propagators::force_model_config::{DragConfiguration, ParameterSource};

        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 300e3, 0.0, 0.0, 0.0, 7700.0, 0.0]);

        let force_config = ForceModelConfiguration {
            mass: None,
            gravity: GravityConfiguration::PointMass,
            drag: Some(DragConfiguration {
                model: AtmosphericModel::Exponential {
                    scale_height: 60000.0, // 60 km
                    rho0: 3.614e-13,       // kg/m³ at h0
                    h0: 300000.0,          // Reference altitude 300 km
                },
                area: ParameterSource::Value(10.0),
                cd: ParameterSource::Value(2.2),
            }),
            srp: None,
            third_body: None,
            relativity: false,
        };

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state.clone(),
            NumericalPropagationConfig::default(),
            force_config,
            Some(default_test_params()),
            None,
            None,
        )
        .unwrap();

        let initial_energy = compute_orbital_energy(&state);

        // Propagate
        prop.step_by(600.0);

        let final_energy = compute_orbital_energy(&prop.current_state());

        // Drag should decrease energy
        assert!(final_energy < initial_energy);
    }

    #[test]
    fn test_dnumericalorbitpropagator_force_srp_no_eclipse() {
        use crate::propagators::force_model_config::{
            ParameterSource, SolarRadiationPressureConfiguration,
        };

        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![
            R_EARTH + 20000e3,
            0.0,
            0.0, // High orbit for SRP
            0.0,
            4000.0,
            0.0,
        ]);

        let force_config = ForceModelConfiguration {
            mass: None,
            gravity: GravityConfiguration::PointMass,
            drag: None,
            srp: Some(SolarRadiationPressureConfiguration {
                area: ParameterSource::Value(10.0),
                cr: ParameterSource::Value(1.3),
                eclipse_model: EclipseModel::None,
            }),
            third_body: None,
            relativity: false,
        };

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            force_config,
            Some(default_test_params()),
            None,
            None,
        )
        .unwrap();

        // Propagate
        prop.step_by(3600.0);

        // Should complete without error
        assert!(prop.current_epoch() > epoch);
    }

    #[test]
    fn test_dnumericalorbitpropagator_force_srp_cylindrical_eclipse() {
        use crate::propagators::force_model_config::{
            ParameterSource, SolarRadiationPressureConfiguration,
        };

        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let force_config = ForceModelConfiguration {
            mass: None,
            gravity: GravityConfiguration::PointMass,
            drag: None,
            srp: Some(SolarRadiationPressureConfiguration {
                area: ParameterSource::Value(10.0),
                cr: ParameterSource::Value(1.3),
                eclipse_model: EclipseModel::Cylindrical,
            }),
            third_body: None,
            relativity: false,
        };

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            force_config,
            Some(default_test_params()),
            None,
            None,
        )
        .unwrap();

        prop.step_by(5400.0);
        assert!(prop.current_epoch() > epoch);
    }

    #[test]
    fn test_dnumericalorbitpropagator_force_srp_conical_eclipse() {
        use crate::propagators::force_model_config::{
            ParameterSource, SolarRadiationPressureConfiguration,
        };

        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let force_config = ForceModelConfiguration {
            mass: None,
            gravity: GravityConfiguration::PointMass,
            drag: None,
            srp: Some(SolarRadiationPressureConfiguration {
                area: ParameterSource::Value(10.0),
                cr: ParameterSource::Value(1.3),
                eclipse_model: EclipseModel::Conical,
            }),
            third_body: None,
            relativity: false,
        };

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            force_config,
            Some(default_test_params()),
            None,
            None,
        )
        .unwrap();

        prop.step_by(5400.0);
        assert!(prop.current_epoch() > epoch);
    }

    #[test]
    fn test_dnumericalorbitpropagator_force_third_body_sun() {
        use crate::propagators::force_model_config::ThirdBodyConfiguration;

        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![
            R_EARTH + 35786e3,
            0.0,
            0.0, // GEO altitude
            0.0,
            3075.0,
            0.0,
        ]);

        let force_config = ForceModelConfiguration {
            mass: None,
            gravity: GravityConfiguration::PointMass,
            drag: None,
            srp: None,
            third_body: Some(ThirdBodyConfiguration {
                ephemeris_source: EphemerisSource::LowPrecision,
                bodies: vec![ThirdBody::Sun],
            }),
            relativity: false,
        };

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            force_config,
            Some(default_test_params()),
            None,
            None,
        )
        .unwrap();

        prop.step_by(86400.0); // 1 day
        assert!(prop.current_epoch() > epoch);
    }

    #[test]
    fn test_dnumericalorbitpropagator_force_third_body_moon() {
        use crate::propagators::force_model_config::ThirdBodyConfiguration;

        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 35786e3, 0.0, 0.0, 0.0, 3075.0, 0.0]);

        let force_config = ForceModelConfiguration {
            mass: None,
            gravity: GravityConfiguration::PointMass,
            drag: None,
            srp: None,
            third_body: Some(ThirdBodyConfiguration {
                ephemeris_source: EphemerisSource::LowPrecision,
                bodies: vec![ThirdBody::Moon],
            }),
            relativity: false,
        };

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            force_config,
            Some(default_test_params()),
            None,
            None,
        )
        .unwrap();

        prop.step_by(86400.0);
        assert!(prop.current_epoch() > epoch);
    }

    #[test]
    fn test_dnumericalorbitpropagator_force_third_body_planets_de440s() {
        use crate::propagators::force_model_config::ThirdBodyConfiguration;

        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 35786e3, 0.0, 0.0, 0.0, 3075.0, 0.0]);

        let force_config = ForceModelConfiguration {
            mass: None,
            gravity: GravityConfiguration::PointMass,
            drag: None,
            srp: None,
            third_body: Some(ThirdBodyConfiguration {
                ephemeris_source: EphemerisSource::DE440s,
                bodies: vec![ThirdBody::Sun, ThirdBody::Moon, ThirdBody::Jupiter],
            }),
            relativity: false,
        };

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            force_config,
            Some(default_test_params()),
            None,
            None,
        )
        .unwrap();

        prop.step_by(86400.0);
        assert!(prop.current_epoch() > epoch);
    }

    #[test]
    fn test_dnumericalorbitpropagator_force_relativity() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let force_config = ForceModelConfiguration {
            mass: None,
            gravity: GravityConfiguration::PointMass,
            drag: None,
            srp: None,
            third_body: None,
            relativity: true, // Enable relativity
        };

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            force_config,
            Some(default_test_params()),
            None,
            None,
        )
        .unwrap();

        prop.step_by(5400.0);
        assert!(prop.current_epoch() > epoch);
    }

    #[test]
    fn test_dnumericalorbitpropagator_force_combined_leo() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        // Use LEO default configuration
        let force_config = ForceModelConfiguration::leo_default();

        // Note: LEO default includes spherical harmonic gravity which requires model data
        // This test verifies construction succeeds
        let prop_result = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            force_config,
            Some(default_test_params()),
            None,
            None,
        );

        // Construction should succeed
        assert!(
            prop_result.is_ok(),
            "LEO default config should construct successfully"
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_force_combined_geo() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 35786e3, 0.0, 0.0, 0.0, 3075.0, 0.0]);

        // Use GEO default configuration
        let force_config = ForceModelConfiguration::geo_default();

        // Note: GEO default includes spherical harmonic gravity which requires model data
        // This test verifies construction succeeds
        let prop_result = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            force_config,
            Some(default_test_params()),
            None,
            None,
        );

        // Construction should succeed
        assert!(
            prop_result.is_ok(),
            "GEO default config should construct successfully"
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_force_parameter_sensitivity() {
        use crate::propagators::force_model_config::{DragConfiguration, ParameterSource};

        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 300e3, 0.0, 0.0, 0.0, 7700.0, 0.0]);

        // Configure with Cd from parameter vector
        let force_config = ForceModelConfiguration {
            mass: None,
            gravity: GravityConfiguration::PointMass,
            drag: Some(DragConfiguration {
                model: AtmosphericModel::HarrisPriester,
                area: ParameterSource::Value(10.0),
                cd: ParameterSource::ParameterIndex(2), // From params[2]
            }),
            srp: None,
            third_body: None,
            relativity: false,
        };

        let mut prop1 = DNumericalOrbitPropagator::new(
            epoch,
            state.clone(),
            NumericalPropagationConfig::default(),
            force_config.clone(),
            Some(default_test_params()),
            None,
            None,
        )
        .unwrap();

        // Propagate with default Cd=2.2
        prop1.step_by(600.0);
        let final_state1 = prop1.current_state().clone();

        // Create second propagator and modify Cd parameter
        let mut prop2 = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            force_config,
            Some(default_test_params()),
            None,
            None,
        )
        .unwrap();

        // Manually modify parameter (this would normally be done via event callback)
        // For now just verify that different Cd values can be set
        let params = prop2.current_params();
        assert_eq!(params[2], 2.2); // Default Cd

        prop2.step_by(600.0);
        let final_state2 = prop2.current_state();

        // States should be very similar since we used same Cd
        let position_diff = (final_state1[0] - final_state2[0]).abs()
            + (final_state1[1] - final_state2[1]).abs()
            + (final_state1[2] - final_state2[2]).abs();
        assert!(
            position_diff < 100.0,
            "Position difference too large: {}",
            position_diff
        );
    }

    // =========================================================================
    // GROUP 5: Propagation Mode Tests (STM, Sensitivity)
    // =========================================================================

    #[test]
    fn test_dnumericalorbitpropagator_propagation_mode_state_only() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        )
        .unwrap();

        // In StateOnly mode, STM and sensitivity should be None
        assert!(prop.stm().is_none());
        assert!(prop.sensitivity().is_none());
    }

    #[test]
    fn test_dnumericalorbitpropagator_propagation_mode_with_stm() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        // Enable STM via config
        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_stm = true;

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            config,
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        )
        .unwrap();

        assert!(prop.stm().is_some());
        assert!(prop.sensitivity().is_none());

        // Propagate
        prop.step_by(100.0);

        // STM should have evolved
        let stm = prop.stm().unwrap();
        assert_eq!(stm.nrows(), 6);
        assert_eq!(stm.ncols(), 6);

        // STM should not be identity anymore
        let identity = DMatrix::<f64>::identity(6, 6);
        let mut differs = false;
        for i in 0..6 {
            for j in 0..6 {
                let diff: f64 = stm[(i, j)] - identity[(i, j)];
                if diff.abs() > 1e-6 {
                    differs = true;
                    break;
                }
            }
        }
        assert!(differs, "STM should have evolved from identity");
    }

    #[test]
    fn test_dnumericalorbitpropagator_propagation_mode_with_sensitivity() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        // Enable sensitivity via config
        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_sensitivity = true;

        // Must provide params for sensitivity propagation (5 params -> 6x5 sensitivity matrix)
        // Use test-friendly config with point mass gravity to avoid gravity model dependency
        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            config,
            test_force_config_with_params(),
            Some(default_test_params()),
            None,
            None,
        )
        .unwrap();

        assert!(prop.stm().is_none());
        assert!(prop.sensitivity().is_some());

        // Propagate
        prop.step_by(100.0);

        // Sensitivity should exist
        let sens = prop.sensitivity().unwrap();
        assert_eq!(sens.nrows(), 6);
        assert_eq!(sens.ncols(), 5);
    }

    #[test]
    fn test_dnumericalorbitpropagator_propagation_mode_with_stm_and_sensitivity() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        // Enable both STM and sensitivity via config
        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_stm = true;
        config.variational.enable_sensitivity = true;

        // Must provide params for sensitivity propagation (5 params -> 6x5 sensitivity matrix)
        // Use test-friendly config with point mass gravity to avoid gravity model dependency
        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            config,
            test_force_config_with_params(),
            Some(default_test_params()),
            None,
            None,
        )
        .unwrap();

        assert!(prop.stm().is_some());
        assert!(prop.sensitivity().is_some());

        // Propagate
        prop.step_by(100.0);

        // Both should exist and have correct dimensions
        assert_eq!(prop.stm().unwrap().nrows(), 6);
        assert_eq!(prop.sensitivity().unwrap().nrows(), 6);
    }

    #[test]
    fn test_dnumericalorbitpropagator_config_stm() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        // Test: STM enabled via config
        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_stm = true;

        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            config,
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            None,
        )
        .unwrap();

        assert!(prop.stm().is_some());
    }

    #[test]
    fn test_dnumericalorbitpropagator_config_sensitivity() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        // Test: sensitivity enabled via config (requires params)
        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_sensitivity = true;
        let params = DVector::from_vec(vec![1000.0, 10.0, 2.2, 10.0, 1.3]); // 5 params

        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            config,
            ForceModelConfiguration::gravity_only(),
            Some(params),
            None,
            None,
        )
        .unwrap();

        assert!(prop.sensitivity().is_some());
    }

    #[test]
    fn test_dnumericalorbitpropagator_config_stm_and_sensitivity() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        // Test: both STM and sensitivity enabled via config (requires params)
        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_stm = true;
        config.variational.enable_sensitivity = true;
        let params = DVector::from_vec(vec![1000.0, 10.0, 2.2, 10.0, 1.3]); // 5 params

        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            config,
            ForceModelConfiguration::gravity_only(),
            Some(params),
            None,
            None,
        )
        .unwrap();

        assert!(prop.stm().is_some());
        assert!(prop.sensitivity().is_some());
    }

    // =========================================================================================
    // COVARIANCE PROPAGATION TESTS
    // =========================================================================================

    #[test]
    fn test_covariance_propagation_initialization() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        // Create initial covariance (100 m² position uncertainty, 1 m²/s² velocity uncertainty)
        let mut initial_cov = DMatrix::zeros(6, 6);
        for i in 0..3 {
            initial_cov[(i, i)] = 100.0; // Position variance [m²]
        }
        for i in 3..6 {
            initial_cov[(i, i)] = 1.0; // Velocity variance [m²/s²]
        }

        // Constructor with initial covariance should automatically enable STM
        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            Some(initial_cov.clone()),
        )
        .unwrap();

        // Verify STM was enabled
        assert!(prop.stm().is_some());

        // Verify initial covariance was stored
        assert!(prop.initial_covariance.is_some());
        assert!(prop.current_covariance.is_some());

        let stored_cov = prop.current_covariance.as_ref().unwrap();
        for i in 0..6 {
            for j in 0..6 {
                assert!((stored_cov[(i, j)] - initial_cov[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_covariance_propagates_with_stm() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        // Small initial covariance
        let initial_cov = DMatrix::identity(6, 6) * 10.0; // 10 m²/m²/s²

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            Some(initial_cov.clone()),
        )
        .unwrap();

        // Propagate forward in time
        prop.propagate_to(epoch + 600.0); // 10 minutes

        // Covariance should have changed
        let final_cov = prop.current_covariance.as_ref().unwrap();

        // Check that covariance grew (uncertainty increases over time)
        let initial_pos_variance = initial_cov[(0, 0)];
        let final_pos_variance = final_cov[(0, 0)];
        assert!(
            final_pos_variance > initial_pos_variance,
            "Position variance should grow: initial={}, final={}",
            initial_pos_variance,
            final_pos_variance
        );
    }

    #[test]
    fn test_covariance_stored_in_trajectory() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let initial_cov = DMatrix::identity(6, 6) * 100.0;

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::gravity_only(),
            None,
            None,
            Some(initial_cov),
        )
        .unwrap();

        // Propagate with multiple steps
        prop.propagate_steps(5);

        // Check trajectory has covariance data
        assert!(prop.trajectory().covariances.is_some());
        let covs = prop.trajectory().covariances.as_ref().unwrap();
        assert!(
            !covs.is_empty(),
            "Trajectory should contain covariance matrices"
        );
    }
}
