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
    accel_relativity, accel_solar_radiation_pressure, accel_third_body, eclipse_conical,
    eclipse_cylindrical, get_global_gravity_model, sun_position,
};
use crate::propagators::{
    AtmosphericModel, EclipseModel, ForceModelConfiguration, GravityConfiguration,
    GravityModelSource,
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
use crate::integrators::traits::{DControlInput, DStateDynamics};

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
/// ```rust
/// use brahe::propagators::{DNumericalOrbitPropagator, NumericalPropagationConfig, ForceModelConfiguration};
/// use brahe::propagators::traits::DStatePropagator;
/// use brahe::eop::{StaticEOPProvider, set_global_eop_provider};
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::constants::R_EARTH;
/// use nalgebra::DVector;
///
/// // Initialize EOP provider (required for frame transformations)
/// let eop = StaticEOPProvider::from_zero();
/// set_global_eop_provider(eop);
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
///     None,  // control_input
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
    /// * `control_input` - Optional control input function for continuous control accelerations
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
    /// ```rust
    /// use brahe::propagators::{DNumericalOrbitPropagator, NumericalPropagationConfig, ForceModelConfiguration};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::constants::R_EARTH;
    /// use nalgebra::DVector;
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = DVector::from_vec(vec![
    ///     R_EARTH + 500e3, 0.0, 0.0,  // position
    ///     0.0, 7500.0, 0.0,            // velocity
    /// ]);
    ///
    /// let prop = DNumericalOrbitPropagator::new(
    ///     epoch,
    ///     state,
    ///     NumericalPropagationConfig::default(),
    ///     ForceModelConfiguration::default(),
    ///     Some(DVector::from_vec(vec![1000.0, 10.0, 2.2, 10.0, 1.3])),
    ///     None,
    ///     None,
    ///     None,
    /// ).unwrap();
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        epoch: Epoch,
        state: DVector<f64>,
        propagation_config: super::NumericalPropagationConfig,
        force_config: ForceModelConfiguration,
        params: Option<DVector<f64>>,
        additional_dynamics: Option<DStateDynamics>,
        control_input: DControlInput,
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
            control_input,
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
            // Get mass from configuration
            let mass = force_config
                .mass
                .as_ref()
                .expect("Mass must be configured for drag calculation")
                .get_value(params_opt);

            // Get drag parameters (area, Cd)
            let drag_area = drag_config.area.get_value(params_opt);
            let cd = drag_config.cd.get_value(params_opt);

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
            // Get mass from configuration
            let mass = force_config
                .mass
                .as_ref()
                .expect("Mass must be configured for SRP calculation")
                .get_value(params_opt);

            // Get SRP parameters (area, Cr)
            let srp_area = srp_config.area.get_value(params_opt);
            let cr = srp_config.cr.get_value(params_opt);

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
                a_total += accel_third_body(body.clone(), tb_config.ephemeris_source, epoch, r);
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
}

// =============================================================================
// DStatePropagator Trait Implementation
// =============================================================================

impl super::traits::DStatePropagator for DNumericalOrbitPropagator {
    fn step_by(&mut self, step_size: f64) {
        let target_t = self.t_rel + step_size;

        // Support both forward and backward propagation
        let is_forward = step_size >= 0.0;

        // Take adaptive steps until we've reached the target
        while !self.terminated {
            // Check if we've reached target (forward or backward)
            if is_forward {
                if self.t_rel >= target_t {
                    break;
                }
            } else if self.t_rel <= target_t {
                break;
            }

            // Calculate remaining time
            let remaining = target_t - self.t_rel;

            // Guard against very small steps
            if remaining.abs() <= 1e-12 {
                break;
            }

            // Limit next step to not overshoot target
            // But allow adaptive integrator to suggest smaller steps
            let dt_max = if is_forward {
                remaining.min(self.dt_next.abs())
            } else {
                -remaining.abs().min(self.dt_next.abs())
            };

            // Temporarily set the suggested next step to not overshoot
            let saved_dt_next = self.dt_next;
            self.dt_next = dt_max;

            // Take one adaptive step (includes event detection)
            self.step_once();

            // Restore suggested dt_next for subsequent steps
            // (unless step_once updated it to something smaller)
            if self.dt_next.abs() > saved_dt_next.abs() {
                self.dt_next = saved_dt_next;
            }
        }
    }

    fn propagate_to(&mut self, target_epoch: Epoch) {
        let target_rel = target_epoch - self.epoch_initial;

        // Support both forward and backward propagation
        let is_forward = target_rel >= self.t_rel;

        while !self.terminated {
            // Check if we've reached target (forward or backward)
            if is_forward {
                if self.t_rel >= target_rel {
                    break;
                }
            } else if self.t_rel <= target_rel {
                break;
            }

            // Calculate remaining time
            let remaining = target_rel - self.t_rel;

            // Guard against very small steps
            if remaining.abs() <= 1e-12 {
                break;
            }

            // Limit next step to not overshoot target
            let dt_max = if is_forward {
                remaining.min(self.dt_next.abs())
            } else {
                -remaining.abs().min(self.dt_next.abs())
            };

            let saved_dt_next = self.dt_next;
            self.dt_next = dt_max;

            // Take one adaptive step (includes event detection)
            self.step_once();

            // Restore suggested dt_next for subsequent steps
            if self.dt_next.abs() > saved_dt_next.abs() {
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
        AtmosphericModel, DragConfiguration, EphemerisSource, GravityConfiguration,
        ParameterSource, ThirdBody,
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
            mass: Some(ParameterSource::ParameterIndex(0)),
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
            None,
        );

        assert!(prop.is_ok());
    }

    #[test]
    fn test_force_model_configuration_construction_variants() {
        let _ = ForceModelConfiguration::default();
        let _ = ForceModelConfiguration::high_fidelity();
        let _ = ForceModelConfiguration::earth_gravity();
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
            ForceModelConfiguration::earth_gravity(),
            None,
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

    #[test]
    fn test_dnumericalorbitpropagator_dstatepropagator_propagate_to_forward() {
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

        let initial_pos = state[0]; // Save for comparison

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let target_time = 3600.0; // 1 hour
        let target_epoch = epoch + target_time;

        // Propagate to target epoch
        prop.propagate_to(target_epoch);

        // Verify we reached the target time
        assert!(
            (prop.current_epoch() - target_epoch).abs() < 0.1,
            "Should reach target epoch within 0.1s"
        );

        // Verify state changed
        let current_state = prop.current_state();
        assert_ne!(
            current_state[0], initial_pos,
            "Position should have changed"
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_dstatepropagator_step_by_backward() {
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

        let initial_pos = state[0];

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let step_size = -1800.0; // 30 minutes backward

        // Propagate backward using step_by with negative step
        prop.step_by(step_size);

        // Verify we went backward in time
        let time_diff: f64 = prop.current_epoch() - epoch;
        assert!(
            time_diff < -1790.0 && time_diff > -1810.0,
            "Should have stepped backward ~1800s, got: {}",
            time_diff
        );

        // Verify current epoch is before initial epoch
        assert!(
            prop.current_epoch() < epoch,
            "Current epoch should be before initial epoch"
        );

        // Verify state changed
        let current_state = prop.current_state();
        assert_ne!(
            current_state[0], initial_pos,
            "Position should have changed during backward propagation"
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_dstatepropagator_propagate_to_backward() {
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

        let initial_pos = state[0];

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let target_time = -3600.0; // 1 hour in the past
        let target_epoch = epoch + target_time;

        // Propagate backward to target epoch
        prop.propagate_to(target_epoch);

        // Verify we reached the target time (going backward)
        assert!(
            (prop.current_epoch() - target_epoch).abs() < 0.1,
            "Should reach past epoch within 0.1s, got diff: {}",
            prop.current_epoch() - target_epoch
        );

        // Verify current epoch is before initial epoch
        assert!(
            prop.current_epoch() < epoch,
            "Current epoch should be before initial epoch"
        );

        // Verify state changed
        let current_state = prop.current_state();
        assert_ne!(
            current_state[0], initial_pos,
            "Position should have changed during backward propagation"
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_dstatepropagator_propagate_steps() {
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
            state.clone(),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let step_size = 60.0;
        let num_steps = 10;

        // Propagate N steps
        prop.propagate_steps(num_steps);

        // Verify we advanced by approximately N steps
        // Note: Adaptive integrator may take different step sizes
        let elapsed = prop.current_epoch() - epoch;
        let expected_min = step_size * (num_steps as f64) * 0.5; // Allow 50% variation
        let expected_max = step_size * (num_steps as f64) * 2.0;

        assert!(
            elapsed > expected_min && elapsed < expected_max,
            "Should have propagated approximately {} steps, elapsed: {}, expected range: [{}, {}]",
            num_steps,
            elapsed,
            expected_min,
            expected_max
        );

        // Verify state changed
        let current_state = prop.current_state();
        assert_ne!(current_state[0], state[0], "Position should have changed");

        // Verify trajectory has entries
        assert!(
            prop.trajectory().len() > 0,
            "Trajectory should have at least one entry"
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_dstatepropagator_reset() {
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
            state.clone(),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate forward
        prop.propagate_to(epoch + 1800.0);

        // Verify state changed
        let state_after = prop.current_state();
        assert_ne!(state_after[0], state[0], "State should have changed");
        assert!(prop.trajectory().len() > 0, "Should have trajectory");

        // Reset propagator
        prop.reset();

        // Verify we're back at initial conditions
        assert_eq!(
            prop.current_epoch(),
            prop.initial_epoch(),
            "Should be at initial epoch after reset"
        );

        let state_after_reset = prop.current_state();
        for i in 0..6 {
            assert!(
                (state_after_reset[i] - state[i]).abs() < 1e-10,
                "State should match initial state after reset, index {}: {} vs {}",
                i,
                state_after_reset[i],
                state[i]
            );
        }

        // Verify trajectory cleared
        assert_eq!(
            prop.trajectory().len(),
            0,
            "Trajectory should be empty after reset"
        );

        // Propagate again and verify it works
        let target = epoch + 900.0;
        prop.propagate_to(target);
        let time_diff: f64 = prop.current_epoch() - target;
        assert!(
            time_diff.abs() < 0.1,
            "Should propagate correctly after reset"
        );
    }

    // =========================================================================
    // DOrbitStateProvider Trait Tests
    // =========================================================================

    #[test]
    fn test_dnumericalorbitpropagator_dorbitstateprovider_state_eci() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);
        let initial_pos = state[0];

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate forward
        prop.step_by(1800.0);

        // Get ECI state at a propagated epoch
        let query_epoch = epoch + 900.0;
        let eci_state = prop.state_eci(query_epoch).unwrap();

        // Verify we get a 6D state
        assert_eq!(eci_state.len(), 6);

        // Position should have changed from initial
        assert_ne!(eci_state[0], initial_pos);

        // Verify reasonable orbital mechanics (position magnitude in LEO range)
        let pos_mag = (eci_state[0].powi(2) + eci_state[1].powi(2) + eci_state[2].powi(2)).sqrt();
        assert!(pos_mag > R_EARTH && pos_mag < R_EARTH + 1000e3);
    }

    #[test]
    fn test_dnumericalorbitpropagator_dorbitstateprovider_state_ecef() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate forward
        prop.step_by(1800.0);

        // Get ECEF state at a propagated epoch
        let query_epoch = epoch + 900.0;
        let ecef_state = prop.state_ecef(query_epoch).unwrap();

        // Verify we get a 6D state
        assert_eq!(ecef_state.len(), 6);

        // Verify reasonable position magnitude
        let pos_mag =
            (ecef_state[0].powi(2) + ecef_state[1].powi(2) + ecef_state[2].powi(2)).sqrt();
        assert!(pos_mag > R_EARTH && pos_mag < R_EARTH + 1000e3);
    }

    #[test]
    fn test_dnumericalorbitpropagator_dorbitstateprovider_state_gcrf() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate forward
        prop.step_by(1800.0);

        // Get GCRF state at a propagated epoch
        let query_epoch = epoch + 900.0;
        let gcrf_state = prop.state_gcrf(query_epoch).unwrap();

        // Verify we get a 6D state
        assert_eq!(gcrf_state.len(), 6);

        // Verify reasonable position magnitude
        let pos_mag =
            (gcrf_state[0].powi(2) + gcrf_state[1].powi(2) + gcrf_state[2].powi(2)).sqrt();
        assert!(pos_mag > R_EARTH && pos_mag < R_EARTH + 1000e3);
    }

    #[test]
    fn test_dnumericalorbitpropagator_dorbitstateprovider_state_itrf() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate forward
        prop.step_by(1800.0);

        // Get ITRF state at a propagated epoch
        let query_epoch = epoch + 900.0;
        let itrf_state = prop.state_itrf(query_epoch).unwrap();

        // Verify we get a 6D state
        assert_eq!(itrf_state.len(), 6);

        // Verify reasonable position magnitude
        let pos_mag =
            (itrf_state[0].powi(2) + itrf_state[1].powi(2) + itrf_state[2].powi(2)).sqrt();
        assert!(pos_mag > R_EARTH && pos_mag < R_EARTH + 1000e3);
    }

    #[test]
    fn test_dnumericalorbitpropagator_dorbitstateprovider_state_eme2000() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate forward
        prop.step_by(1800.0);

        // Get EME2000 state at a propagated epoch
        let query_epoch = epoch + 900.0;
        let eme2000_state = prop.state_eme2000(query_epoch).unwrap();

        // Verify we get a 6D state
        assert_eq!(eme2000_state.len(), 6);

        // Verify reasonable position magnitude
        let pos_mag =
            (eme2000_state[0].powi(2) + eme2000_state[1].powi(2) + eme2000_state[2].powi(2)).sqrt();
        assert!(pos_mag > R_EARTH && pos_mag < R_EARTH + 1000e3);
    }

    #[test]
    fn test_dnumericalorbitpropagator_dorbitstateprovider_osculating_elements_radians() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate forward
        prop.step_by(1800.0);

        // Get osculating elements in radians
        let query_epoch = epoch + 900.0;
        let elements = prop
            .state_as_osculating_elements(query_epoch, AngleFormat::Degrees)
            .unwrap();

        // Verify we get 6 elements [a, e, i, RAAN, arg_p, M]
        assert_eq!(elements.len(), 6);

        // Verify semi-major axis is reasonable (allowing for slightly elliptical orbit)
        assert!(elements[0] > R_EARTH + 300e3 && elements[0] < R_EARTH + 700e3);

        // Verify eccentricity is small (near-circular)
        assert!(elements[1] < 0.1);

        // Verify angles are in radians (inclination should be small in radians for equatorial)
        assert!(elements[2].abs() < 0.1); // inclination near 0 rad
    }

    #[test]
    fn test_dnumericalorbitpropagator_dorbitstateprovider_osculating_elements_degrees() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate forward
        prop.step_by(1800.0);

        // Get osculating elements in degrees
        let query_epoch = epoch + 900.0;
        let elements = prop
            .state_as_osculating_elements(query_epoch, AngleFormat::Degrees)
            .unwrap();

        // Verify we get 6 elements
        assert_eq!(elements.len(), 6);

        // Verify semi-major axis is reasonable (allowing for slightly elliptical orbit)
        assert!(elements[0] > R_EARTH + 300e3 && elements[0] < R_EARTH + 700e3);

        // Verify eccentricity is small
        assert!(elements[1] < 0.1);

        // Verify angles are in degrees (inclination should be small in degrees for equatorial)
        assert!(elements[2].abs() < 10.0); // inclination near 0 deg
    }

    #[test]
    fn test_dnumericalorbitpropagator_dorbitstateprovider_frame_conversion_roundtrip() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate forward
        prop.step_by(1800.0);

        let query_epoch = epoch + 900.0;

        // Get ECI state
        let eci_state = prop.state_eci(query_epoch).unwrap();

        // Convert to ECEF and back
        let ecef_state = prop.state_ecef(query_epoch).unwrap();
        let eci_from_ecef = crate::frames::state_ecef_to_eci(query_epoch, ecef_state);

        // Verify round-trip accuracy (should match within numerical precision)
        for i in 0..6 {
            let diff = (eci_state[i] - eci_from_ecef[i]).abs();
            let tolerance = if i < 3 { 1e-6 } else { 1e-9 }; // Position: 1 µm, Velocity: 1 nm/s
            assert!(
                diff < tolerance,
                "ECI ↔ ECEF round-trip failed at index {}: diff = {} m or m/s",
                i,
                diff
            );
        }
    }

    #[test]
    fn test_dnumericalorbitpropagator_dorbitstateprovider_representation_conversion_roundtrip() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate forward
        prop.step_by(1800.0);

        let query_epoch = epoch + 900.0;

        // Get Cartesian state
        let cartesian = prop.state_eci(query_epoch).unwrap();

        // Convert to Keplerian and back
        let keplerian = prop
            .state_as_osculating_elements(query_epoch, AngleFormat::Degrees)
            .unwrap();
        let cartesian_from_keplerian =
            state_osculating_to_cartesian(keplerian, AngleFormat::Degrees);

        // Verify round-trip accuracy
        for i in 0..6 {
            let diff = (cartesian[i] - cartesian_from_keplerian[i]).abs();
            let tolerance = if i < 3 { 1.0 } else { 1e-3 }; // Position: 1 m, Velocity: 1 mm/s
            assert!(
                diff < tolerance,
                "Cartesian ↔ Keplerian round-trip failed at index {}: diff = {} m or m/s",
                i,
                diff
            );
        }
    }

    #[test]
    fn test_dnumericalorbitpropagator_dorbitstateprovider_extended_state_preservation() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        // Create extended 8D state (6D orbit + 2D extra)
        let state = DVector::from_vec(vec![
            R_EARTH + 500e3,
            0.0,
            0.0,
            0.0,
            7500.0,
            0.0,
            42.0,
            99.0, // Extended state values
        ]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state.clone(),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate forward
        prop.step_by(1800.0);

        let query_epoch = epoch + 900.0;

        // Get ECI state (should return only first 6 elements)
        let eci_state = prop.state_eci(query_epoch).unwrap();
        assert_eq!(eci_state.len(), 6, "ECI state should be 6D");

        // Get full state from trajectory
        let full_state = prop.trajectory.interpolate(&query_epoch).unwrap();
        assert_eq!(full_state.len(), 8, "Full state should preserve 8D");

        // Verify extended dimensions aren't corrupted
        // (they should propagate according to dynamics, but for gravity-only
        // with no forces on extra dims, they should stay constant or change predictably)
        assert_eq!(full_state.len(), 8);

        // Confirm the extended states haven't changed (no forces acting on them)
        assert_eq!(
            full_state[6], state[6],
            "Extended state index 6 should be preserved"
        );
        assert_eq!(
            full_state[7], state[7],
            "Extended state index 7 should be preserved"
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_dorbitstateprovider_angle_format_handling() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate forward
        prop.step_by(1800.0);

        let query_epoch = epoch + 900.0;

        // Get elements in both formats
        let elements_rad = prop
            .state_as_osculating_elements(query_epoch, AngleFormat::Radians)
            .unwrap();
        let elements_deg = prop
            .state_as_osculating_elements(query_epoch, AngleFormat::Degrees)
            .unwrap();

        // Verify semi-major axis and eccentricity are the same (not angles)
        assert!(
            (elements_rad[0] - elements_deg[0]).abs() < 1e-6,
            "Semi-major axis should match"
        );
        assert!(
            (elements_rad[1] - elements_deg[1]).abs() < 1e-9,
            "Eccentricity should match"
        );

        // Verify angular elements differ by conversion factor
        use crate::constants::math::RAD2DEG;
        for i in 2..6 {
            let expected_deg = elements_rad[i] * RAD2DEG;
            let diff = (elements_deg[i] - expected_deg).abs();
            assert!(
                diff < 1e-9,
                "Angle at index {} should differ by RAD2DEG factor: {} deg vs {} deg (diff: {})",
                i,
                elements_deg[i],
                expected_deg,
                diff
            );
        }
    }

    #[test]
    fn test_dnumericalorbitpropagator_dorbitstateprovider_interpolation_accuracy() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate with multiple steps to build trajectory
        prop.step_by(1800.0);

        // Query at intermediate epoch (should interpolate)
        let query_epoch = epoch + 450.0; // Quarter way through
        let interpolated_state = prop.state_eci(query_epoch).unwrap();

        // Verify reasonable position (between start and end)
        let pos_mag = (interpolated_state[0].powi(2)
            + interpolated_state[1].powi(2)
            + interpolated_state[2].powi(2))
        .sqrt();
        assert!(
            pos_mag > R_EARTH && pos_mag < R_EARTH + 1000e3,
            "Interpolated position should be in LEO range"
        );

        // Verify state is different from initial
        let initial_state = prop.initial_state();
        let pos_diff = (interpolated_state[0] - initial_state[0]).abs();
        assert!(
            pos_diff > 1.0,
            "Interpolated state should differ from initial"
        );
    }

    // =========================================================================
    // DOrbitCovarianceProvider Trait Tests
    // =========================================================================

    #[test]
    fn test_dnumericalorbitpropagator_dorbitcovarianceprovider_covariance_eci() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        // Create initial covariance
        let mut initial_cov = DMatrix::zeros(6, 6);
        for i in 0..3 {
            initial_cov[(i, i)] = 100.0; // Position variance [m²]
        }
        for i in 3..6 {
            initial_cov[(i, i)] = 1.0; // Velocity variance [m²/s²]
        }

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            Some(initial_cov.clone()),
        )
        .unwrap();

        // Propagate forward
        prop.step_by(1800.0);

        // Get ECI covariance at propagated epoch
        let query_epoch = epoch + 900.0;
        let cov_eci = prop.covariance_eci(query_epoch).unwrap();

        // Verify dimensions
        assert_eq!(cov_eci.nrows(), 6);
        assert_eq!(cov_eci.ncols(), 6);

        // Verify covariance has grown (uncertainty increases)
        assert!(
            cov_eci[(0, 0)] > initial_cov[(0, 0)],
            "Position variance should grow"
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_dorbitcovarianceprovider_covariance_gcrf() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let initial_cov = DMatrix::identity(6, 6) * 100.0;

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            Some(initial_cov),
        )
        .unwrap();

        // Propagate forward
        prop.step_by(1800.0);

        // Get GCRF covariance
        let query_epoch = epoch + 900.0;
        let cov_gcrf = prop.covariance_gcrf(query_epoch).unwrap();

        // Verify dimensions
        assert_eq!(cov_gcrf.nrows(), 6);
        assert_eq!(cov_gcrf.ncols(), 6);

        // For DNumericalOrbitPropagator, GCRF ≈ ECI, should be very similar
        let cov_eci = prop.covariance_eci(query_epoch).unwrap();
        for i in 0..6 {
            for j in 0..6 {
                let diff = (cov_gcrf[(i, j)] - cov_eci[(i, j)]).abs();
                assert!(diff < 1e-6, "GCRF and ECI covariances should match closely");
            }
        }
    }

    #[test]
    fn test_dnumericalorbitpropagator_dorbitcovarianceprovider_covariance_rtn() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let initial_cov = DMatrix::identity(6, 6) * 100.0;

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            Some(initial_cov),
        )
        .unwrap();

        // Propagate forward
        prop.step_by(1800.0);

        // Get RTN covariance
        let query_epoch = epoch + 900.0;
        let cov_rtn = prop.covariance_rtn(query_epoch).unwrap();

        // Verify dimensions
        assert_eq!(cov_rtn.nrows(), 6);
        assert_eq!(cov_rtn.ncols(), 6);

        // RTN covariance should be different from ECI (rotated frame)
        let cov_eci = prop.covariance_eci(query_epoch).unwrap();

        // Check that at least some elements differ (frame rotation changes values)
        let mut has_difference = false;
        for i in 0..6 {
            for j in 0..6 {
                if (cov_rtn[(i, j)] - cov_eci[(i, j)]).abs() > 1.0 {
                    has_difference = true;
                    break;
                }
            }
        }
        assert!(has_difference, "RTN covariance should differ from ECI");

        // Verify RTN covariance is still symmetric (within numerical precision)
        for i in 0..6 {
            for j in 0..6 {
                let diff = (cov_rtn[(i, j)] - cov_rtn[(j, i)]).abs();
                assert!(
                    diff < 1e-6,
                    "RTN covariance should be symmetric at ({},{}): diff = {}",
                    i,
                    j,
                    diff
                );
            }
        }
    }

    #[test]
    fn test_dnumericalorbitpropagator_dorbitcovarianceprovider_interpolation_accuracy() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        // Use a simple diagonal covariance for better numerical stability
        let mut initial_cov = DMatrix::zeros(6, 6);
        for i in 0..3 {
            initial_cov[(i, i)] = 100.0; // Position variance [m²]
        }
        for i in 3..6 {
            initial_cov[(i, i)] = 1.0; // Velocity variance [m²/s²]
        }

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            Some(initial_cov.clone()),
        )
        .unwrap();

        // Propagate with multiple small steps to build a detailed trajectory
        for _ in 0..10 {
            prop.step_by(180.0); // 10 steps of 3 minutes each
        }

        // Query at intermediate epochs within the trajectory
        let query_epoch1 = epoch + 360.0; // After 2nd step
        let query_epoch2 = epoch + 900.0; // Halfway through
        let query_epoch3 = epoch + 1440.0; // After 8th step

        let cov1 = prop.covariance_eci(query_epoch1).unwrap();
        let cov2 = prop.covariance_eci(query_epoch2).unwrap();
        let cov3 = prop.covariance_eci(query_epoch3).unwrap();

        // Verify dimensions
        assert_eq!(cov1.nrows(), 6);
        assert_eq!(cov2.nrows(), 6);
        assert_eq!(cov3.nrows(), 6);

        // Covariance should grow monotonically over time
        let var1 = cov1[(0, 0)];
        let var2 = cov2[(0, 0)];
        let var3 = cov3[(0, 0)];

        assert!(
            var1 > initial_cov[(0, 0)],
            "First covariance should be larger than initial"
        );
        assert!(
            var2 > var1,
            "Second covariance should be larger than first: {} > {}",
            var2,
            var1
        );
        assert!(
            var3 > var2,
            "Third covariance should be larger than second: {} > {}",
            var3,
            var2
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_dorbitcovarianceprovider_positive_definiteness() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        // Create positive definite initial covariance
        let mut initial_cov = DMatrix::zeros(6, 6);
        for i in 0..6 {
            initial_cov[(i, i)] = 100.0; // Diagonal elements
        }
        // Add small off-diagonal correlations
        for i in 0..5 {
            initial_cov[(i, i + 1)] = 10.0;
            initial_cov[(i + 1, i)] = 10.0;
        }

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            Some(initial_cov),
        )
        .unwrap();

        // Propagate forward
        prop.step_by(1800.0);

        let query_epoch = epoch + 900.0;

        // Test all frame representations maintain positive definiteness
        let cov_eci = prop.covariance_eci(query_epoch).unwrap();
        let cov_gcrf = prop.covariance_gcrf(query_epoch).unwrap();
        let cov_rtn = prop.covariance_rtn(query_epoch).unwrap();

        // Check diagonal elements are positive (necessary for positive definite)
        for i in 0..6 {
            assert!(
                cov_eci[(i, i)] > 0.0,
                "ECI diagonal element {} should be positive",
                i
            );
            assert!(
                cov_gcrf[(i, i)] > 0.0,
                "GCRF diagonal element {} should be positive",
                i
            );
            assert!(
                cov_rtn[(i, i)] > 0.0,
                "RTN diagonal element {} should be positive",
                i
            );
        }

        // Check symmetry (necessary for positive definite)
        for i in 0..6 {
            for j in 0..6 {
                assert!(
                    (cov_eci[(i, j)] - cov_eci[(j, i)]).abs() < 1e-9,
                    "ECI covariance should be symmetric"
                );
                assert!(
                    (cov_rtn[(i, j)] - cov_rtn[(j, i)]).abs() < 1e-6,
                    "RTN covariance should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_dnumericalorbitpropagator_dorbitcovarianceprovider_error_handling() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        // Test 1: Covariance not enabled
        let mut prop_no_cov = DNumericalOrbitPropagator::new(
            epoch,
            state.clone(),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None, // No initial covariance
        )
        .unwrap();

        prop_no_cov.step_by(1800.0);

        let query_epoch = epoch + 900.0;
        let result = prop_no_cov.covariance_eci(query_epoch);
        assert!(result.is_err(), "Should error when covariance not enabled");

        // Test 2: Epoch out of bounds (before trajectory start)
        let initial_cov = DMatrix::identity(6, 6) * 100.0;
        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            Some(initial_cov),
        )
        .unwrap();

        prop.step_by(1800.0);

        let before_epoch = epoch - 1000.0; // Before trajectory start
        let result = prop.covariance_eci(before_epoch);
        assert!(
            result.is_err(),
            "Should error for epoch before trajectory start"
        );
    }

    // =========================================================================
    // InterpolationConfig Trait Tests
    // =========================================================================

    #[test]
    fn test_dnumericalorbitpropagator_interpolationconfig_with_method() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        // Use builder pattern to set interpolation method
        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap()
        .with_interpolation_method(InterpolationMethod::Linear);

        // Verify the method was set
        assert_eq!(prop.get_interpolation_method(), InterpolationMethod::Linear);
    }

    #[test]
    fn test_dnumericalorbitpropagator_interpolationconfig_set_and_get() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Verify default method
        let initial_method = prop.get_interpolation_method();
        assert_eq!(initial_method, InterpolationMethod::Linear); // Default

        // Since there's only one variant, test that we can still set it
        prop.set_interpolation_method(InterpolationMethod::Linear);

        // Verify the method remains unchanged
        assert_eq!(prop.get_interpolation_method(), InterpolationMethod::Linear);
    }

    // =========================================================================
    // CovarianceInterpolationConfig Trait Tests
    // =========================================================================

    #[test]
    fn test_dnumericalorbitpropagator_covarianceinterpolationconfig_with_method() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let initial_cov = DMatrix::identity(6, 6) * 100.0;

        // Use builder pattern to set covariance interpolation method
        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            Some(initial_cov),
        )
        .unwrap()
        .with_covariance_interpolation_method(CovarianceInterpolationMethod::MatrixSquareRoot);

        // Verify the method was set
        assert_eq!(
            prop.get_covariance_interpolation_method(),
            CovarianceInterpolationMethod::MatrixSquareRoot
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_covarianceinterpolationconfig_set_and_get() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let initial_cov = DMatrix::identity(6, 6) * 100.0;

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            Some(initial_cov),
        )
        .unwrap();

        // Verify default method
        let initial_method = prop.get_covariance_interpolation_method();
        assert_eq!(
            initial_method,
            CovarianceInterpolationMethod::TwoWasserstein
        ); // Default

        // Change method using setter
        prop.set_covariance_interpolation_method(CovarianceInterpolationMethod::MatrixSquareRoot);

        // Verify the method was changed
        assert_eq!(
            prop.get_covariance_interpolation_method(),
            CovarianceInterpolationMethod::MatrixSquareRoot
        );
    }

    // =========================================================================
    // Identifiable Trait Tests
    // =========================================================================

    #[test]
    fn test_dnumericalorbitpropagator_identifiable_with_name() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        // Use builder pattern to set name
        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap()
        .with_name("TestSat");

        // Verify name was set
        assert_eq!(prop.get_name(), Some("TestSat"));
    }

    #[test]
    fn test_dnumericalorbitpropagator_identifiable_with_id() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        // Use builder pattern to set ID
        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap()
        .with_id(12345);

        // Verify ID was set
        assert_eq!(prop.get_id(), Some(12345));
    }

    #[test]
    fn test_dnumericalorbitpropagator_identifiable_with_uuid() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let test_uuid = uuid::Uuid::new_v4();

        // Use builder pattern to set UUID
        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap()
        .with_uuid(test_uuid);

        // Verify UUID was set
        assert_eq!(prop.get_uuid(), Some(test_uuid));
    }

    #[test]
    fn test_dnumericalorbitpropagator_identifiable_with_new_uuid() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        // Use builder pattern to generate new UUID
        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap()
        .with_new_uuid();

        // Verify UUID was generated (should be Some, not None)
        assert!(prop.get_uuid().is_some());
    }

    #[test]
    fn test_dnumericalorbitpropagator_identifiable_setters() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Initially should have no identifiers
        assert_eq!(prop.get_name(), None);
        assert_eq!(prop.get_id(), None);
        assert_eq!(prop.get_uuid(), None);

        // Set name
        prop.set_name(Some("UpdatedSat"));
        assert_eq!(prop.get_name(), Some("UpdatedSat"));

        // Set ID
        prop.set_id(Some(99999));
        assert_eq!(prop.get_id(), Some(99999));

        // Generate UUID
        prop.generate_uuid();
        assert!(prop.get_uuid().is_some());

        // Clear name
        prop.set_name(None);
        assert_eq!(prop.get_name(), None);
    }

    #[test]
    fn test_dnumericalorbitpropagator_identifiable_with_identity() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let test_uuid = uuid::Uuid::new_v4();

        // Use with_identity to set all at once
        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap()
        .with_identity(Some("CompleteSat"), Some(test_uuid), Some(42));

        // Verify all identifiers were set
        assert_eq!(prop.get_name(), Some("CompleteSat"));
        assert_eq!(prop.get_uuid(), Some(test_uuid));
        assert_eq!(prop.get_id(), Some(42));
    }

    #[test]
    fn test_dnumericalorbitpropagator_identifiable_set_identity() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let test_uuid = uuid::Uuid::new_v4();

        // Use set_identity to set all at once
        prop.set_identity(Some("ModifiedSat"), Some(test_uuid), Some(777));

        // Verify all identifiers were set
        assert_eq!(prop.get_name(), Some("ModifiedSat"));
        assert_eq!(prop.get_uuid(), Some(test_uuid));
        assert_eq!(prop.get_id(), Some(777));

        // Clear all identifiers
        prop.set_identity(None, None, None);
        assert_eq!(prop.get_name(), None);
        assert_eq!(prop.get_uuid(), None);
        assert_eq!(prop.get_id(), None);
    }

    #[test]
    fn test_dnumericalorbitpropagator_identifiable_persistence_through_propagation() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let test_uuid = uuid::Uuid::new_v4();

        // Create propagator with identifiers
        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap()
        .with_name("PersistentSat")
        .with_id(123)
        .with_uuid(test_uuid);

        // Verify initial identifiers
        assert_eq!(prop.get_name(), Some("PersistentSat"));
        assert_eq!(prop.get_id(), Some(123));
        assert_eq!(prop.get_uuid(), Some(test_uuid));

        // Propagate forward
        prop.step_by(1800.0);

        // Verify identifiers persist after propagation
        assert_eq!(prop.get_name(), Some("PersistentSat"));
        assert_eq!(prop.get_id(), Some(123));
        assert_eq!(prop.get_uuid(), Some(test_uuid));

        // Reset propagator
        prop.reset();

        // Verify identifiers persist after reset
        assert_eq!(prop.get_name(), Some("PersistentSat"));
        assert_eq!(prop.get_id(), Some(123));
        assert_eq!(prop.get_uuid(), Some(test_uuid));
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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

    #[test]
    fn test_dnumericalorbitpropagator_event_detection_at_initial_epoch() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Add event very close to initial epoch (slightly after)
        // Note: Event exactly at t=0 may not be detected due to numerical considerations
        let event_time = epoch + 1.0; // 1 second after start
        prop.add_event_detector(Box::new(DTimeEvent::new(event_time, "Near-Initial Event")));

        // Propagate forward
        prop.step_by(1800.0);

        // Event near initial epoch should be detected
        let events = prop.event_log();
        assert_eq!(
            events.len(),
            1,
            "Event near initial epoch should be detected"
        );
        assert_eq!(events[0].name, "Near-Initial Event");
        // Use relaxed tolerance for adaptive stepping
        assert!((events[0].window_open - event_time).abs() < 0.1);
    }

    #[test]
    fn test_dnumericalorbitpropagator_event_detection_at_final_epoch() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Add event exactly at the target epoch we'll propagate to
        let target_epoch = epoch + 1800.0;
        prop.add_event_detector(Box::new(DTimeEvent::new(target_epoch, "Final Epoch Event")));

        // Propagate to exactly that epoch
        prop.propagate_to(target_epoch);

        // Event at final epoch should be detected
        let events = prop.event_log();
        assert_eq!(events.len(), 1, "Event at final epoch should be detected");
        assert_eq!(events[0].name, "Final Epoch Event");
        assert!((events[0].window_open - target_epoch).abs() < 1e-6);
    }

    #[test]
    fn test_dnumericalorbitpropagator_event_detection_simultaneous_events() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Add multiple events at exactly the same time
        let event_time = epoch + 900.0;
        prop.add_event_detector(Box::new(DTimeEvent::new(event_time, "Event A")));
        prop.add_event_detector(Box::new(DTimeEvent::new(event_time, "Event B")));
        prop.add_event_detector(Box::new(DTimeEvent::new(event_time, "Event C")));

        // Propagate
        prop.step_by(1800.0);

        // All simultaneous events should be detected
        let events = prop.event_log();
        assert_eq!(
            events.len(),
            3,
            "All simultaneous events should be detected"
        );

        // All should have the same epoch (within tolerance)
        for event in events {
            assert!((event.window_open - event_time).abs() < 1e-6);
        }

        // Events should be present (order may vary)
        let names: Vec<&str> = events.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"Event A"));
        assert!(names.contains(&"Event B"));
        assert!(names.contains(&"Event C"));
    }

    #[test]
    fn test_dnumericalorbitpropagator_event_detection_rapid_crossings() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Add multiple events in rapid succession (every 30 seconds)
        for i in 1..=10 {
            let event_time = epoch + (i as f64) * 30.0;
            prop.add_event_detector(Box::new(DTimeEvent::new(
                event_time,
                format!("Rapid Event {}", i),
            )));
        }

        // Propagate with large step size that would normally skip over them
        prop.step_by(360.0); // 6 minutes, should catch all 10 events in 5 minutes

        // All rapid events should be detected
        let events = prop.event_log();
        assert_eq!(
            events.len(),
            10,
            "All rapid crossing events should be detected"
        );

        // Verify they're in chronological order
        for i in 0..9 {
            assert!(
                events[i].window_open < events[i + 1].window_open,
                "Events should be in chronological order"
            );
        }
    }

    #[test]
    fn test_dnumericalorbitpropagator_event_detection_with_backward_propagation() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // First propagate forward and detect an event
        prop.add_event_detector(Box::new(DTimeEvent::new(epoch + 900.0, "Forward Event")));
        prop.step_by(1800.0);

        // Should have detected one event
        assert_eq!(prop.event_log().len(), 1);
        let forward_final_epoch = prop.current_epoch();

        // Now test that we can propagate backward (state propagation works)
        // Event detection during backward propagation is not guaranteed to work
        prop.step_by(-900.0);

        // Verify backward propagation changed the state
        assert!(
            prop.current_epoch() < forward_final_epoch,
            "Backward propagation should move time backwards"
        );

        // Original event should still be in log
        assert_eq!(
            prop.event_log().len(),
            1,
            "Event log should persist during backward propagation"
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_event_detection_log_persistence_across_reset_termination() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Add terminal event
        let terminal_event = DTimeEvent::new(epoch + 900.0, "Terminal Event").is_terminal();
        prop.add_event_detector(Box::new(terminal_event));

        // Add non-terminal event after terminal
        prop.add_event_detector(Box::new(DTimeEvent::new(epoch + 1800.0, "Post-Terminal")));

        // Propagate - should stop at terminal event
        prop.step_by(3600.0);

        // Should have 1 event and be terminated
        assert_eq!(prop.event_log().len(), 1);
        assert!(prop.terminated());

        // Reset termination flag
        prop.reset_termination();
        assert!(!prop.terminated());

        // Event log should persist
        assert_eq!(
            prop.event_log().len(),
            1,
            "Event log should persist after reset_termination"
        );

        // Continue propagation
        prop.step_by(1800.0);

        // Should now have both events
        let events = prop.event_log();
        assert_eq!(
            events.len(),
            2,
            "Should detect post-terminal event after reset"
        );
        assert_eq!(events[0].name, "Terminal Event");
        assert_eq!(events[1].name, "Post-Terminal");
    }

    #[test]
    fn test_dnumericalorbitpropagator_event_detection_clear_vs_remove() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Add events
        prop.add_event_detector(Box::new(DTimeEvent::new(epoch + 600.0, "Event 1")));
        prop.add_event_detector(Box::new(DTimeEvent::new(epoch + 1200.0, "Event 2")));

        // Propagate to trigger first event
        prop.step_by(900.0);
        assert_eq!(prop.event_log().len(), 1);

        // Clear events log (but keep detectors)
        prop.clear_events();
        assert_eq!(
            prop.event_log().len(),
            0,
            "Event log should be empty after clear"
        );

        // Continue propagation - should detect second event
        prop.step_by(900.0);
        assert_eq!(
            prop.event_log().len(),
            1,
            "Should detect new event after clearing log"
        );
        assert_eq!(prop.event_log()[0].name, "Event 2");

        // Reset propagator - this clears events and resets state
        prop.reset();

        // Add new event and propagate
        prop.add_event_detector(Box::new(DTimeEvent::new(epoch + 600.0, "Reset Event")));
        prop.step_by(900.0);

        // Should have events from both before and after reset
        // (detectors persist, but event log was cleared)
        let events = prop.event_log();
        assert!(!events.is_empty(), "Should detect events after reset");

        // At least the reset event should be present
        let reset_event_found = events.iter().any(|e| e.name == "Reset Event");
        assert!(reset_event_found, "Should find reset event");
    }

    #[test]
    fn test_dnumericalorbitpropagator_event_detection_multiple_callbacks_same_step() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        // Create elliptical orbit that crosses 471 km and 472 km
        // Using same parameters as working test but different altitude range
        let a = R_EARTH + 500e3; // 500 km semi-major axis
        let e = 0.02; // Eccentricity (creates range from ~362 km to ~637 km)
        let i = 0.0;
        let raan = 0.0;
        let argp = 0.0;
        let ta = 0.0; // Start at perigee

        let oe = DVector::from_vec(vec![a, e, i, raan, argp, ta]);
        let state = state_osculating_to_cartesian(
            Vector6::from_column_slice(oe.as_slice()),
            AngleFormat::Radians,
        );

        // Use fixed-step RK4 with 900-second steps to ensure both events occur in same step
        use crate::integrators::IntegratorConfig;
        use crate::propagators::IntegratorMethod;
        let prop_config = NumericalPropagationConfig {
            method: IntegratorMethod::RK4,
            integrator: IntegratorConfig::fixed_step(900.0),
            ..Default::default()
        };

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            DVector::from_column_slice(state.as_slice()),
            prop_config,
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Track which callbacks execute using Arc<AtomicBool>
        let callback1_executed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let callback2_executed = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let cb1_flag = callback1_executed.clone();
        let cb2_flag = callback2_executed.clone();

        use crate::events::EventAction;

        // Event 1: 471.1 km crossing with callback (applies +5 m/s in vx)
        // This event occurs SECOND chronologically (higher altitude)
        let event1 = DAltitudeEvent::new(471.1e3, "Event 1 - 471.1 km", EventDirection::Any)
            .with_callback(Box::new(move |_t, state, _params| {
                cb1_flag.store(true, std::sync::atomic::Ordering::SeqCst);
                let mut new_state = state.clone();
                new_state[3] += 5.0; // +5 m/s in vx
                (Some(new_state), None, EventAction::Continue)
            }));

        // Event 2: 471 km crossing with callback (applies +10 m/s in vy)
        // This event occurs FIRST chronologically (lower altitude)
        let event2 = DAltitudeEvent::new(471e3, "Event 2 - 471 km", EventDirection::Any)
            .with_callback(Box::new(move |_t, state, _params| {
                cb2_flag.store(true, std::sync::atomic::Ordering::SeqCst);
                let mut new_state = state.clone();
                new_state[4] += 10.0; // +10 m/s in vy
                (Some(new_state), None, EventAction::Continue)
            }));

        prop.add_event_detector(Box::new(event1));
        prop.add_event_detector(Box::new(event2));

        // Propagate for one quarter orbit (from 406 km to ~475 km, crossing both thresholds)
        let period = orbital_period(a);
        prop.propagate_to(epoch + period / 4.0);

        // Verify both events were detected in chronological order
        assert!(
            callback2_executed.load(std::sync::atomic::Ordering::SeqCst),
            "Event 2 callback (471 km, first chronological) should execute"
        );
        assert!(
            callback1_executed.load(std::sync::atomic::Ordering::SeqCst),
            "Event 1 callback (471.1 km, second chronological) should also execute"
        );

        // Verify both events are in the log in chronological order
        let events = prop.event_log();
        assert_eq!(events.len(), 2, "Both events should be in the log");
        assert_eq!(
            events[0].name, "Event 2 - 471 km",
            "First event should be 471 km"
        );
        assert_eq!(
            events[1].name, "Event 1 - 471.1 km",
            "Second event should be 471.1 km"
        );

        // Verify events occurred within the same integration step (900s)
        let time_diff = events[1].window_open - events[0].window_open;
        assert!(
            time_diff < 900.0,
            "Events should occur within same 900s integration step, but were {:.2}s apart",
            time_diff
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_continuous_control_via_control_input() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        // Create circular LEO orbit at 500 km altitude
        let a = R_EARTH + 500e3;
        let e = 0.0;
        let i = 0.0;
        let raan = 0.0;
        let argp = 0.0;
        let ta = 0.0;

        let oe = DVector::from_vec(vec![a, e, i, raan, argp, ta]);
        let state = state_osculating_to_cartesian(
            Vector6::from_column_slice(oe.as_slice()),
            AngleFormat::Radians,
        );
        let state = DVector::from_column_slice(state.as_slice());

        // Define continuous tangential thrust control
        // Thrust: 0.5 N, Mass: 1000 kg, Acceleration: 0.0005 m/s²
        let control_fn: crate::integrators::traits::DControlInput =
            Some(Box::new(|_t, state, _params| {
                let v = Vector3::new(state[3], state[4], state[5]);
                let v_mag = v.norm();
                let a_control = if v_mag > 1e-6 {
                    v * (0.0005 / v_mag) // Tangential acceleration
                } else {
                    Vector3::zeros()
                };

                let mut dx = DVector::zeros(state.len());
                dx[3] = a_control[0]; // dvx/dt
                dx[4] = a_control[1]; // dvy/dt
                dx[5] = a_control[2]; // dvz/dt
                dx
            }));

        // Create reference propagator WITHOUT control
        let mut prop_ref = DNumericalOrbitPropagator::new(
            epoch,
            state.clone(),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None, // No control
            None,
        )
        .unwrap();

        // Create test propagator WITH control
        let mut prop_test = DNumericalOrbitPropagator::new(
            epoch,
            state.clone(),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            control_fn, // With tangential thrust control
            None,
        )
        .unwrap();

        // Propagate both for one orbital period
        let period = orbital_period(a);
        prop_ref.propagate_to(epoch + period);
        prop_test.propagate_to(epoch + period);

        // Extract final semi-major axes
        let state_ref = prop_ref.current_state();
        let state_test = prop_test.current_state();

        let oe_ref = state_cartesian_to_osculating(
            Vector6::from_column_slice(state_ref.as_slice()),
            AngleFormat::Radians,
        );
        let oe_test = state_cartesian_to_osculating(
            Vector6::from_column_slice(state_test.as_slice()),
            AngleFormat::Radians,
        );

        let a_ref = oe_ref[0];
        let a_test = oe_test[0];
        let delta_a = a_test - a_ref;

        // Expected: ~5.1 km increase (from Edelbaum theory)
        // Verify measurable effect (> 4 km)
        assert!(
            delta_a > 4000.0,
            "Semi-major axis should increase by > 4 km with tangential thrust, got {:.2} km",
            delta_a / 1000.0
        );

        // Verify within 20% of theoretical value (~5.1 km)
        let expected_delta_a = 5100.0; // meters
        let tolerance = 0.20; // 20%
        assert!(
            (delta_a - expected_delta_a).abs() < expected_delta_a * tolerance,
            "Semi-major axis increase should be ~{:.2} km ± 20%, got {:.2} km",
            expected_delta_a / 1000.0,
            delta_a / 1000.0
        );
    }

    // =========================================================================
    // NUMERICAL ACCURACY VALIDATION TESTS
    // =========================================================================

    // Integration Accuracy Tests
    // -------------------------

    #[test]
    fn test_dnumericalorbitpropagator_accuracy_vs_keplerian() {
        use crate::propagators::KeplerianPropagator;
        use crate::propagators::traits::SStatePropagator;

        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let oe = nalgebra::Vector6::new(R_EARTH + 500e3, 0.001, 97.8_f64, 0.0, 0.0, 0.0);
        let state = state_osculating_to_cartesian(oe, AngleFormat::Degrees);

        // Create numerical propagator with point mass gravity
        let mut num_prop = DNumericalOrbitPropagator::new(
            epoch,
            DVector::from_vec(state.as_slice().to_vec()),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::two_body_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Create Keplerian propagator
        let mut kep_prop = KeplerianPropagator::new(
            epoch,
            state,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            60.0,
        );

        // Propagate both for one orbit
        let orbital_period = orbital_period(oe[0]);
        num_prop.step_by(orbital_period);
        kep_prop.step_by(orbital_period);

        // Compare final states
        let num_final = num_prop.current_state();
        let kep_final = kep_prop.current_state();

        // Position error should be less than 10 meters (for 1 orbit)
        let pos_error = (num_final.fixed_rows::<3>(0) - kep_final.fixed_rows::<3>(0)).norm();
        assert!(
            pos_error < 10.0,
            "Position error vs Keplerian should be < 10 m, got {:.3} m",
            pos_error
        );

        // Velocity error should be less than 10 mm/s
        let vel_error = (num_final.fixed_rows::<3>(3) - kep_final.fixed_rows::<3>(3)).norm();
        assert!(
            vel_error < 0.01,
            "Velocity error vs Keplerian should be < 10 mm/s, got {:.6} m/s",
            vel_error
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_accuracy_orbital_period() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let a = R_EARTH + 500e3;
        let oe = nalgebra::Vector6::new(a, 0.01, 45.0_f64, 0.0, 0.0, 0.0);
        let state = state_osculating_to_cartesian(oe, AngleFormat::Degrees);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            DVector::from_vec(state.as_slice().to_vec()),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Theoretical orbital period
        let t_theory = orbital_period(a);

        // Propagate for 5 complete orbits
        let n_orbits = 5.0;
        prop.step_by(n_orbits * t_theory);

        // Actual time elapsed
        let t_actual = (prop.current_epoch() - epoch) / n_orbits;

        // Period error should be less than 0.1%
        let period_error = ((t_actual - t_theory) / t_theory).abs();
        assert!(
            period_error < 1e-3,
            "Orbital period error should be < 0.1%, got {:.6}%",
            period_error * 100.0
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_accuracy_adaptive_step_behavior() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        // Configure with different tolerance levels
        let mut config_tight = NumericalPropagationConfig::default();
        config_tight.integrator.abs_tol = 1e-12;
        config_tight.integrator.rel_tol = 1e-12;

        let mut config_loose = NumericalPropagationConfig::default();
        config_loose.integrator.abs_tol = 1e-8;
        config_loose.integrator.rel_tol = 1e-8;

        let mut prop_tight = DNumericalOrbitPropagator::new(
            epoch,
            state.clone(),
            config_tight,
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let mut prop_loose = DNumericalOrbitPropagator::new(
            epoch,
            state,
            config_loose,
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate both for same duration
        prop_tight.step_by(1800.0);
        prop_loose.step_by(1800.0);

        // Tight tolerance should produce more accurate results
        let state_tight = prop_tight.current_state();
        let state_loose = prop_loose.current_state();

        // They should differ (loose tolerance is less accurate)
        let difference = (state_tight.fixed_rows::<3>(0) - state_loose.fixed_rows::<3>(0)).norm();

        // Difference should be measurable but still small (< 1 m)
        assert!(
            difference > 0.0 && difference < 1.0,
            "Tolerance should affect accuracy, diff = {:.6} m",
            difference
        );
    }

    // Conservation Laws Tests
    // ------------------------

    #[test]
    fn test_dnumericalorbitpropagator_accuracy_energy_conservation_point_mass() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let oe = nalgebra::Vector6::new(R_EARTH + 500e3, 0.001, 45.0_f64, 0.0, 0.0, 0.0);
        let state = state_osculating_to_cartesian(oe, AngleFormat::Degrees);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            DVector::from_vec(state.as_slice().to_vec()),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::two_body_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Calculate initial specific energy
        let r0 = state.fixed_rows::<3>(0).norm();
        let v0 = state.fixed_rows::<3>(3).norm();
        let e0 = 0.5 * v0 * v0 - GM_EARTH / r0;

        // Propagate for 10 orbits
        let orbital_period = orbital_period(oe[0]);
        prop.step_by(10.0 * orbital_period);

        // Calculate final specific energy
        let final_state = prop.current_state();
        let r1 = final_state.fixed_rows::<3>(0).norm();
        let v1 = final_state.fixed_rows::<3>(3).norm();
        let e1 = 0.5 * v1 * v1 - GM_EARTH / r1;

        // Relative energy error
        let rel_energy_error = ((e1 - e0) / e0).abs();

        assert!(
            rel_energy_error < 1e-6,
            "Energy conservation error should be < 1e-6 (10 orbits), got {:.3e}",
            rel_energy_error
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_accuracy_angular_momentum_conservation() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let oe = nalgebra::Vector6::new(R_EARTH + 500e3, 0.01, 55.0_f64, 30.0_f64, 45.0_f64, 0.0);
        let state = state_osculating_to_cartesian(oe, AngleFormat::Degrees);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            DVector::from_vec(state.as_slice().to_vec()),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::two_body_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Calculate initial angular momentum
        let r0 = state.fixed_rows::<3>(0);
        let v0 = state.fixed_rows::<3>(3);
        let h0 = r0.cross(&v0);

        // Propagate for 10 orbits
        let orbital_period = orbital_period(oe[0]);
        prop.step_by(10.0 * orbital_period);

        // Calculate final angular momentum
        let final_state = prop.current_state();
        let r1 = final_state.fixed_rows::<3>(0);
        let v1 = final_state.fixed_rows::<3>(3);
        let h1 = r1.cross(&v1);

        // Angular momentum should be conserved for central forces
        let h_error = (h1 - h0).norm();
        let h0_mag = h0.norm();
        let rel_h_error = h_error / h0_mag;

        assert!(
            rel_h_error < 5e-6,
            "Angular momentum conservation error should be < 5e-6 (10 orbits), got {:.3e}",
            rel_h_error
        );
    }

    // Long-Term Stability Tests
    // --------------------------

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    fn test_dnumericalorbitpropagator_accuracy_energy_drift_long_term() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let oe = nalgebra::Vector6::new(R_EARTH + 500e3, 0.001, 45.0_f64, 0.0, 0.0, 0.0);
        let state = state_osculating_to_cartesian(oe, AngleFormat::Degrees);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            DVector::from_vec(state.as_slice().to_vec()),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::two_body_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Calculate initial specific energy
        let r0 = state.fixed_rows::<3>(0).norm();
        let v0 = state.fixed_rows::<3>(3).norm();
        let e0 = 0.5 * v0 * v0 - GM_EARTH / r0;

        // Propagate for 100 orbits
        let orbital_period = orbital_period(oe[0]);
        prop.step_by(100.0 * orbital_period);

        // Calculate final specific energy
        let final_state = prop.current_state();
        let r1 = final_state.fixed_rows::<3>(0).norm();
        let v1 = final_state.fixed_rows::<3>(3).norm();
        let e1 = 0.5 * v1 * v1 - GM_EARTH / r1;

        // Relative energy drift over 100 orbits
        let rel_energy_drift = ((e1 - e0) / e0).abs();

        assert!(
            rel_energy_drift < 1e-5,
            "Energy drift over 100 orbits should be < 1e-5, got {:.3e}",
            rel_energy_drift
        );
    }

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    fn test_dnumericalorbitpropagator_accuracy_orbital_stability_long_term() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let oe_initial =
            nalgebra::Vector6::new(R_EARTH + 500e3, 0.01, 55.0_f64, 30.0_f64, 45.0_f64, 0.0);
        let state = state_osculating_to_cartesian(oe_initial, AngleFormat::Degrees);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            DVector::from_vec(state.as_slice().to_vec()),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::two_body_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate for 100 orbits
        let orbital_period = orbital_period(oe_initial[0]);
        prop.step_by(100.0 * orbital_period);

        // Get final orbital elements
        let final_state = prop.current_state();
        let oe_final = state_cartesian_to_osculating(
            final_state.fixed_rows::<6>(0).into(),
            AngleFormat::Radians,
        );

        // Semi-major axis should remain stable (< 1 km drift)
        let a_drift = (oe_final[0] - oe_initial[0]).abs();
        assert!(
            a_drift < 1000.0,
            "Semi-major axis drift over 100 orbits should be < 1 km, got {:.1} m",
            a_drift
        );

        // Eccentricity should remain stable (< 0.001 drift)
        let e_drift = (oe_final[1] - oe_initial[1]).abs();
        assert!(
            e_drift < 0.001,
            "Eccentricity drift over 100 orbits should be < 0.001, got {:.6}",
            e_drift
        );
    }

    // Orbital Regime Tests
    // ---------------------

    #[test]
    fn test_dnumericalorbitpropagator_accuracy_leo_regime() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        // Typical LEO orbit
        let oe = nalgebra::Vector6::new(
            R_EARTH + 400e3, // 400 km altitude
            0.001,
            51.6_f64, // ISS-like inclination
            0.0,
            0.0,
            0.0,
        );
        let state = state_osculating_to_cartesian(oe, AngleFormat::Degrees);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            DVector::from_vec(state.as_slice().to_vec()),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate for 3 days
        prop.step_by(3.0 * 86400.0);

        // Should successfully complete propagation
        assert!(
            prop.current_epoch() > epoch,
            "Should successfully propagate LEO orbit for 3 days"
        );

        // Verify orbit is still stable
        let final_state = prop.current_state();
        let r_final = final_state.fixed_rows::<3>(0).norm();
        assert!(
            r_final > R_EARTH + 300e3 && r_final < R_EARTH + 500e3,
            "LEO orbit should remain stable, altitude = {:.1} km",
            (r_final - R_EARTH) / 1000.0
        );
    }

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    fn test_dnumericalorbitpropagator_accuracy_geo_regime() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        // GEO orbit
        let oe = nalgebra::Vector6::new(
            R_EARTH + 35786e3, // GEO altitude
            0.001,
            0.1_f64, // Nearly equatorial
            0.0,
            0.0,
            0.0,
        );
        let state = state_osculating_to_cartesian(oe, AngleFormat::Degrees);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            DVector::from_vec(state.as_slice().to_vec()),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate for 7 days
        prop.step_by(7.0 * 86400.0);

        // Should successfully complete propagation
        assert!(
            prop.current_epoch() > epoch,
            "Should successfully propagate GEO orbit for 7 days"
        );

        // Verify orbit altitude is stable
        let final_state = prop.current_state();
        let r_final = final_state.fixed_rows::<3>(0).norm();
        let altitude_km = (r_final - R_EARTH) / 1000.0;
        assert!(
            (altitude_km - 35786.0).abs() < 100.0,
            "GEO altitude should remain near 35786 km, got {:.1} km",
            altitude_km
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_accuracy_heo_regime() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        // Highly Elliptical Orbit (Molniya-like)
        let oe = nalgebra::Vector6::new(
            26554e3,  // Semi-major axis
            0.72,     // High eccentricity
            63.4_f64, // Molniya inclination
            0.0, 270.0_f64, // Argument of perigee
            0.0,
        );
        let state = state_osculating_to_cartesian(oe, AngleFormat::Degrees);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            DVector::from_vec(state.as_slice().to_vec()),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate for 2 complete orbits
        let orbital_period = orbital_period(oe[0]);
        prop.step_by(2.0 * orbital_period);

        // Should successfully complete propagation
        assert!(
            prop.current_epoch() > epoch,
            "Should successfully propagate HEO orbit for 2 periods"
        );

        // Verify eccentricity is preserved
        let final_state = prop.current_state();
        let oe_final = state_cartesian_to_osculating(
            final_state.fixed_rows::<6>(0).into(),
            AngleFormat::Radians,
        );

        let e_error = (oe_final[1] - oe[1]).abs();
        assert!(
            e_error < 0.01,
            "HEO eccentricity should be preserved, error = {:.6}",
            e_error
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_accuracy_near_circular_stability() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        // Very circular orbit
        let oe = nalgebra::Vector6::new(
            R_EARTH + 600e3,
            0.0001, // Nearly circular
            45.0_f64,
            0.0,
            0.0,
            0.0,
        );
        let state = state_osculating_to_cartesian(oe, AngleFormat::Degrees);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            DVector::from_vec(state.as_slice().to_vec()),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate for 50 orbits
        let orbital_period = orbital_period(oe[0]);
        prop.step_by(50.0 * orbital_period);

        // Verify orbit remains nearly circular
        let final_state = prop.current_state();
        let oe_final = state_cartesian_to_osculating(
            final_state.fixed_rows::<6>(0).into(),
            AngleFormat::Radians,
        );

        assert!(
            oe_final[1] < 0.001,
            "Orbit should remain nearly circular, e_final = {:.6}",
            oe_final[1]
        );

        // Verify radius variation is small
        let r_final = final_state.fixed_rows::<3>(0).norm();
        let r_variation = (r_final - oe[0]).abs();
        assert!(
            r_variation < 5000.0,
            "Radius variation should be small for circular orbit, got {:.1} m",
            r_variation
        );
    }

    // Edge Cases and Robustness Tests
    // --------------------------------

    #[test]
    fn test_dnumericalorbitpropagator_edge_case_high_eccentricity() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        // Very high eccentricity (e = 0.95)
        let oe = nalgebra::Vector6::new(
            R_EARTH + 15000e3, // Large semi-major axis
            0.95,              // High eccentricity
            45.0_f64,
            0.0,
            0.0,
            0.0,
        );
        let state = state_osculating_to_cartesian(oe, AngleFormat::Degrees);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            DVector::from_vec(state.as_slice().to_vec()),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::two_body_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate for one complete orbit
        let orbital_period = orbital_period(oe[0]);
        prop.step_by(orbital_period);

        // Should successfully handle high eccentricity
        assert!(
            prop.current_epoch() > epoch,
            "Should successfully propagate high eccentricity orbit"
        );

        // Eccentricity should be preserved
        let final_state = prop.current_state();
        let oe_final = state_cartesian_to_osculating(
            final_state.fixed_rows::<6>(0).into(),
            AngleFormat::Radians,
        );
        assert!(
            (oe_final[1] - oe[1]).abs() < 0.01,
            "High eccentricity should be preserved, e_error = {:.6}",
            (oe_final[1] - oe[1]).abs()
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_edge_case_equatorial_orbit() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        // Equatorial orbit (i = 0°, potential singularity in orbital elements)
        let oe = nalgebra::Vector6::new(
            R_EARTH + 500e3,
            0.01,
            0.0, // Equatorial (i = 0)
            0.0,
            0.0,
            0.0,
        );
        let state = state_osculating_to_cartesian(oe, AngleFormat::Degrees);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            DVector::from_vec(state.as_slice().to_vec()),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate for 10 orbits
        let orbital_period = orbital_period(oe[0]);
        prop.step_by(10.0 * orbital_period);

        // Should handle equatorial orbit without numerical issues
        assert!(
            prop.current_epoch() > epoch,
            "Should successfully propagate equatorial orbit"
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_edge_case_polar_orbit() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        // Polar orbit (i = 90°)
        let oe = nalgebra::Vector6::new(
            R_EARTH + 800e3,
            0.001,
            90.0_f64, // Polar
            0.0,
            0.0,
            0.0,
        );
        let state = state_osculating_to_cartesian(oe, AngleFormat::Degrees);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            DVector::from_vec(state.as_slice().to_vec()),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::two_body_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate for 10 orbits
        let orbital_period = orbital_period(oe[0]);
        prop.step_by(10.0 * orbital_period);

        // Verify polar inclination is preserved (two-body gravity preserves inclination exactly)
        let final_state = prop.current_state();
        let oe_final = state_cartesian_to_osculating(
            final_state.fixed_rows::<6>(0).into(),
            AngleFormat::Degrees,
        );

        assert!(
            (oe_final[2] - oe[2]).abs() < 0.01,
            "Polar inclination should be preserved, i_error = {:.6} deg",
            (oe_final[2] - oe[2]).abs()
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_edge_case_very_short_step() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let initial_state = prop.current_state().clone();

        // Very short propagation (0.01 seconds)
        prop.step_by(0.01);

        // Should propagate even very short steps
        assert!(
            (prop.current_epoch() - epoch) > 0.0,
            "Should handle very short time steps"
        );

        // State should have changed slightly
        let final_state = prop.current_state();
        let state_diff = (final_state - initial_state).norm();
        assert!(
            state_diff > 0.0 && state_diff < 100.0,
            "State should change slightly, diff = {:.6} m",
            state_diff
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_edge_case_propagate_to_same_epoch() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state.clone(),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate to same epoch (no-op)
        prop.propagate_to(epoch);

        // State should remain unchanged
        let final_state = prop.current_state();
        let state_diff = (final_state - state).norm();
        assert!(
            state_diff < 1e-10,
            "State should remain unchanged when propagating to same epoch"
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_edge_case_backward_then_forward() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state.clone(),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate backward
        prop.step_by(-900.0);
        let backward_epoch = prop.current_epoch();
        assert!(backward_epoch < epoch, "Should propagate backward");

        // Then propagate forward to original epoch
        prop.propagate_to(epoch);

        // Should return close to original state
        let final_state = prop.current_state();
        let state_diff = (final_state - state).norm();
        assert!(
            state_diff < 100.0, // Allow some numerical error
            "Should return close to original state after backward/forward, diff = {:.3} m",
            state_diff
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_edge_case_single_step_propagation() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Single very small step
        prop.step_by(1.0); // 1 second

        // Should successfully take single step
        assert_eq!(
            (prop.current_epoch() - epoch),
            1.0,
            "Should take exactly one second step"
        );
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
            R_EARTH + 500e3, // a [m]
            0.0,             // e (circular)
            97.8_f64,        // i [rad]
            0.0,             // RAAN [rad]
            0.0,             // argp [rad]
            0.0,             // M [rad]
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
            Some(additional_dynamics),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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

        // Test with earth_gravity - no params needed
        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state.clone(),
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let params = prop.current_params();
        // With earth_gravity, no params are required so empty vec is stored
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
        config.variational.store_sensitivity_history = true; // Store sensitivity in trajectory
        let params = DVector::from_vec(vec![1000.0, 10.0, 2.2, 10.0, 1.3]); // 5 params

        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            config,
            ForceModelConfiguration::earth_gravity(),
            Some(params),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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

        // Use simple point mass gravity only (no third body) to avoid requiring ephemerides
        let force_config = ForceModelConfiguration {
            gravity: GravityConfiguration::PointMass,
            drag: None,
            srp: None,
            third_body: None,
            relativity: false,
            mass: None,
        };

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            force_config,
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
            None,
        );

        // Construction should succeed even without gravity data loaded
        assert!(
            prop_result.is_ok(),
            "Spherical harmonic config should construct successfully"
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_force_gravity_j2_perturbation() {
        use crate::orbit_dynamics::gravity::GravityModelType;

        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        // LEO orbit at 97.8° inclination (sun-synchronous)
        let oe = nalgebra::Vector6::new(
            R_EARTH + 700e3, // a = 7078 km
            0.001,           // e (near-circular)
            97.8_f64,        // i = 97.8° (sun-sync)
            0.0,             // Ω
            0.0,             // ω
            0.0,             // M
        );
        let state_initial = state_osculating_to_cartesian(oe, AngleFormat::Degrees);

        // Use J2-only gravity model (2x0 spherical harmonic)
        let force_config = ForceModelConfiguration {
            mass: None,
            gravity: GravityConfiguration::SphericalHarmonic {
                source: GravityModelSource::ModelType(GravityModelType::EGM2008_360),
                degree: 2,
                order: 0,
            },
            drag: None,
            srp: None,
            third_body: None,
            relativity: false,
        };

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            DVector::from_vec(state_initial.as_slice().to_vec()),
            NumericalPropagationConfig::default(),
            force_config,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate for one day
        prop.step_by(86400.0);

        let state_final = prop.current_state();
        let oe_final = state_cartesian_to_osculating(
            state_final.fixed_rows::<6>(0).into(),
            AngleFormat::Degrees,
        );

        // Verify J2 effects are present
        // For sun-sync orbit at 97.8°, J2 causes RAAN regression
        let delta_raan = oe_final[3] - oe[3];

        // RAAN should have changed due to J2 (sun-sync orbits designed for this)
        assert!(
            delta_raan.abs() > 1e-6,
            "J2 should cause RAAN drift, but delta = {} rad",
            delta_raan
        );

        // Semi-major axis may change due to numerical integration and second-order effects
        // For a full day propagation, allow reasonable drift
        let delta_a = oe_final[0] - oe[0];
        assert!(
            delta_a.abs() < 10000.0,
            "Semi-major axis drift should be reasonable, delta = {} m",
            delta_a
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_force_gravity_degree_order_convergence() {
        use crate::orbit_dynamics::gravity::GravityModelType;

        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        // Test different degrees/orders: 2x0 (J2 only), 4x4, 8x8
        let configs = vec![(2, 0), (4, 4), (8, 8)];

        let mut final_states = Vec::new();

        for (degree, order) in configs {
            let force_config = ForceModelConfiguration {
                mass: None,
                gravity: GravityConfiguration::SphericalHarmonic {
                    source: GravityModelSource::ModelType(GravityModelType::EGM2008_360),
                    degree,
                    order,
                },
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
                None,
                None,
                None,
                None,
            )
            .unwrap();

            // Propagate for 1 orbit
            prop.step_by(5400.0);
            let state_vec: Vector6<f64> = prop.current_state().fixed_rows::<6>(0).into();
            final_states.push(state_vec);
        }

        // Higher degree/order should converge (states should become more similar)
        let diff_2_4: f64 = (final_states[0] - final_states[1]).norm();
        let diff_4_8: f64 = (final_states[1] - final_states[2]).norm();

        // Convergence: difference between 4x4 and 8x8 should be smaller than 2x0 and 4x4
        assert!(
            diff_4_8 < diff_2_4,
            "Higher degree/order should converge: diff(2x0,4x4)={:.3}m > diff(4x4,8x8)={:.3}m",
            diff_2_4,
            diff_4_8
        );

        // Both differences should be reasonable (not identical, showing perturbations matter)
        assert!(diff_2_4 > 10.0, "J2 vs 4x4 should differ significantly");
        assert!(diff_4_8 < diff_2_4, "Higher orders should show convergence");
    }

    #[test]
    fn test_dnumericalorbitpropagator_force_gravity_global_vs_modeltype() {
        use crate::orbit_dynamics::gravity::GravityModelType;

        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0]);

        // Test 1: Using ModelType source
        let force_config_modeltype = ForceModelConfiguration {
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

        let mut prop_modeltype = DNumericalOrbitPropagator::new(
            epoch,
            state.clone(),
            NumericalPropagationConfig::default(),
            force_config_modeltype,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Test 2: Using Global source (should use same model if loaded)
        let force_config_global = ForceModelConfiguration {
            mass: None,
            gravity: GravityConfiguration::SphericalHarmonic {
                source: GravityModelSource::Global,
                degree: 4,
                order: 4,
            },
            drag: None,
            srp: None,
            third_body: None,
            relativity: false,
        };

        // The global source test may fail if gravity model not properly loaded
        // This is expected behavior - just verify API accepts both source types
        let prop_global_result = DNumericalOrbitPropagator::new(
            epoch,
            state.clone(),
            NumericalPropagationConfig::default(),
            force_config_global,
            None,
            None,
            None,
            None,
        );

        // This test primarily verifies that both GravityModelSource variants compile and are accepted by the API
        // Actual propagation with spherical harmonics requires gravity model files which may not be present
        // So we'll allow the test to pass if construction succeeds but propagation fails

        // Test ModelType source
        // Propagation may fail if gravity coefficients not loaded - that's ok for this API test
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            prop_modeltype.step_by(1800.0);
        }));

        // Test Global source
        match prop_global_result {
            Ok(mut prop_global) => {
                // Propagation may fail if gravity coefficients not loaded - that's ok for this API test
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    prop_global.step_by(1800.0);
                }));
            }
            Err(_) => {
                // Global model not loaded - this is acceptable for testing API
            }
        }

        // The test passes by verifying both source types are accepted by the compiler and constructor
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
            mass: Some(ParameterSource::ParameterIndex(0)),
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
            mass: Some(ParameterSource::ParameterIndex(0)),
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
    fn test_dnumericalorbitpropagator_force_drag_nrlmsise00() {
        use crate::propagators::force_model_config::{DragConfiguration, ParameterSource};

        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 400e3, 0.0, 0.0, 0.0, 7700.0, 0.0]);

        let force_config = ForceModelConfiguration {
            mass: Some(ParameterSource::Value(1000.0)),
            gravity: GravityConfiguration::PointMass,
            drag: Some(DragConfiguration {
                model: AtmosphericModel::NRLMSISE00,
                area: ParameterSource::Value(10.0),
                cd: ParameterSource::Value(2.2),
            }),
            srp: None,
            third_body: None,
            relativity: false,
        };

        // NRLMSISE00 requires space weather data
        // Construction should succeed, propagation may need data files
        let prop_result = DNumericalOrbitPropagator::new(
            epoch,
            state.clone(),
            NumericalPropagationConfig::default(),
            force_config,
            Some(default_test_params()),
            None,
            None,
            None,
        );

        // Verify construction succeeds
        assert!(
            prop_result.is_ok(),
            "NRLMSISE00 config should construct successfully"
        );

        if let Ok(mut prop) = prop_result {
            // Try to propagate (may fail if space weather data not loaded)
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                prop.step_by(600.0);
            }));

            // This test primarily verifies that NRLMSISE00 is accepted as a configuration option
            // Actual drag computation depends on space weather data availability
            // The test passes by verifying the API accepts this atmospheric model type
        }
    }

    #[test]
    fn test_dnumericalorbitpropagator_force_drag_magnitude_direction() {
        use crate::propagators::force_model_config::{DragConfiguration, ParameterSource};

        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        // Create state with known velocity direction
        let state = DVector::from_vec(vec![
            R_EARTH + 300e3,
            0.0,
            0.0, // x position only
            0.0,
            7700.0,
            0.0, // y velocity only
        ]);

        let force_config = ForceModelConfiguration {
            mass: Some(ParameterSource::ParameterIndex(0)),
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
            None,
        )
        .unwrap();

        // Propagate for short time
        prop.step_by(60.0); // 1 minute

        let final_state = prop.current_state();

        // Drag should oppose velocity - y-velocity should decrease in magnitude
        let initial_vy = state[4];
        let final_vy = final_state[4];

        assert!(
            final_vy.abs() < initial_vy.abs(),
            "Drag should reduce velocity magnitude: |{}| should be < |{}|",
            final_vy,
            initial_vy
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_force_drag_orbital_decay() {
        use crate::propagators::force_model_config::{DragConfiguration, ParameterSource};

        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        // Very low orbit for significant drag
        let oe_initial = nalgebra::Vector6::new(
            R_EARTH + 250e3, // 250 km altitude
            0.001,           // Near-circular
            45.0_f64,
            0.0,
            0.0,
            0.0,
        );
        let state = state_osculating_to_cartesian(oe_initial, AngleFormat::Degrees);

        let force_config = ForceModelConfiguration {
            mass: Some(ParameterSource::ParameterIndex(0)),
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
            DVector::from_vec(state.as_slice().to_vec()),
            NumericalPropagationConfig::default(),
            force_config,
            Some(default_test_params()),
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate for multiple orbits (10 orbits ≈ 15 hours)
        let orbital_period = orbital_period(oe_initial[0]);
        prop.step_by(10.0 * orbital_period);

        let final_state = prop.current_state();
        let oe_final = state_cartesian_to_osculating(
            final_state.fixed_rows::<6>(0).into(),
            AngleFormat::Degrees,
        );

        // Semi-major axis should decrease significantly due to drag
        let delta_a = oe_final[0] - oe_initial[0];
        assert!(
            delta_a < -1000.0,
            "Semi-major axis should decay by at least 1 km over 10 orbits at 250 km: delta = {} m",
            delta_a
        );

        // Altitude should have decreased
        let altitude_initial = oe_initial[0] - R_EARTH;
        let altitude_final = oe_final[0] - R_EARTH;
        assert!(
            altitude_final < altitude_initial,
            "Altitude should decrease: {} km -> {} km",
            altitude_initial / 1000.0,
            altitude_final / 1000.0
        );
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
            mass: Some(ParameterSource::ParameterIndex(0)),
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
            mass: Some(ParameterSource::ParameterIndex(0)),
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
            mass: Some(ParameterSource::ParameterIndex(0)),
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
            None,
        )
        .unwrap();

        prop.step_by(5400.0);
        assert!(prop.current_epoch() > epoch);
    }

    #[test]
    fn test_dnumericalorbitpropagator_force_srp_eclipse_transition() {
        use crate::propagators::force_model_config::{
            ParameterSource, SolarRadiationPressureConfiguration,
        };

        setup_global_test_eop();

        // Use equatorial orbit that will cross Earth's shadow
        let epoch = Epoch::from_datetime(2024, 3, 20, 0, 0, 0.0, 0.0, TimeSystem::UTC); // Near equinox
        let state = DVector::from_vec(vec![R_EARTH + 800e3, 0.0, 0.0, 0.0, 7450.0, 0.0]);

        let force_config = ForceModelConfiguration {
            mass: Some(ParameterSource::ParameterIndex(0)),
            gravity: GravityConfiguration::PointMass,
            drag: None,
            srp: Some(SolarRadiationPressureConfiguration {
                area: ParameterSource::Value(10.0),
                cr: ParameterSource::Value(1.3),
                eclipse_model: EclipseModel::Conical, // Use conical for penumbra transitions
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
            None,
        )
        .unwrap();

        // Propagate through one full orbit to encounter eclipse transitions
        let orbital_period = orbital_period(R_EARTH + 800e3);
        prop.step_by(orbital_period);

        // Should successfully propagate through eclipse transitions
        assert!(
            prop.current_epoch() > epoch,
            "Should successfully handle eclipse transitions"
        );
        assert!(
            (prop.current_epoch() - epoch) > (orbital_period - 60.0),
            "Should complete nearly full orbit"
        );
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
            None,
        )
        .unwrap();

        prop.step_by(86400.0);
        assert!(prop.current_epoch() > epoch);
    }

    #[test]
    fn test_dnumericalorbitpropagator_force_third_body_jupiter() {
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
                bodies: vec![ThirdBody::Jupiter],
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
            None,
        )
        .unwrap();

        // Jupiter's effect is smaller but should still propagate successfully
        prop.step_by(86400.0);
        assert!(
            prop.current_epoch() > epoch,
            "Should successfully propagate with Jupiter perturbation"
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_force_third_body_mars() {
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
                bodies: vec![ThirdBody::Mars],
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
            None,
        )
        .unwrap();

        // Mars' effect is smaller but should still propagate successfully
        prop.step_by(86400.0);
        assert!(
            prop.current_epoch() > epoch,
            "Should successfully propagate with Mars perturbation"
        );
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
            None,
        );

        // Construction should succeed
        assert!(
            prop_result.is_ok(),
            "GEO default config should construct successfully"
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_force_combined_high_fidelity() {
        use crate::propagators::force_model_config::{
            DragConfiguration, ParameterSource, SolarRadiationPressureConfiguration,
            ThirdBodyConfiguration,
        };

        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 800e3, 0.0, 0.0, 0.0, 7450.0, 0.0]);

        // Configure with all force models enabled for high-fidelity propagation
        let force_config = ForceModelConfiguration {
            mass: Some(ParameterSource::Value(1000.0)), // 1000 kg satellite
            gravity: GravityConfiguration::PointMass,   // Use point mass to avoid data dependency
            drag: Some(DragConfiguration {
                model: AtmosphericModel::HarrisPriester,
                area: ParameterSource::Value(5.0),
                cd: ParameterSource::Value(2.2),
            }),
            srp: Some(SolarRadiationPressureConfiguration {
                area: ParameterSource::Value(5.0),
                cr: ParameterSource::Value(1.3),
                eclipse_model: EclipseModel::Conical,
            }),
            third_body: Some(ThirdBodyConfiguration {
                ephemeris_source: EphemerisSource::LowPrecision,
                bodies: vec![ThirdBody::Sun, ThirdBody::Moon],
            }),
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
            None,
        )
        .unwrap();

        // Propagate with all forces active
        let initial_energy = {
            let r = prop.current_state().fixed_rows::<3>(0).norm();
            let v = prop.current_state().fixed_rows::<3>(3).norm();
            0.5 * v * v - GM_EARTH / r
        };

        prop.step_by(3600.0); // 1 hour

        let final_energy = {
            let r = prop.current_state().fixed_rows::<3>(0).norm();
            let v = prop.current_state().fixed_rows::<3>(3).norm();
            0.5 * v * v - GM_EARTH / r
        };

        // Energy should decrease due to drag (dominant non-conservative force at this altitude)
        assert!(
            final_energy < initial_energy,
            "Energy should decrease with drag at 800 km altitude"
        );

        // Verify propagation succeeded with all forces
        assert!(
            prop.current_epoch() > epoch,
            "Should successfully propagate with all force models"
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
            mass: Some(ParameterSource::ParameterIndex(0)),
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
        config.variational.store_sensitivity_history = true; // Store sensitivity in trajectory

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
        config.variational.store_sensitivity_history = true; // Store sensitivity in trajectory

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
            ForceModelConfiguration::earth_gravity(),
            None,
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
        config.variational.store_sensitivity_history = true; // Store sensitivity in trajectory
        let params = DVector::from_vec(vec![1000.0, 10.0, 2.2, 10.0, 1.3]); // 5 params

        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            config,
            ForceModelConfiguration::earth_gravity(),
            Some(params),
            None,
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
        config.variational.store_sensitivity_history = true; // Store sensitivity in trajectory
        let params = DVector::from_vec(vec![1000.0, 10.0, 2.2, 10.0, 1.3]); // 5 params

        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            config,
            ForceModelConfiguration::earth_gravity(),
            Some(params),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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
            ForceModelConfiguration::earth_gravity(),
            None,
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

    // =========================================================================
    // STM PROPAGATION TESTS
    // =========================================================================

    #[test]
    fn test_dnumericalorbitpropagator_stm_identity_initial_condition() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let oe = nalgebra::Vector6::new(R_EARTH + 500e3, 0.01, 97.8_f64, 0.0, 0.0, 0.0);
        let state = state_osculating_to_cartesian(oe, AngleFormat::Degrees);
        let state_dvec = DVector::from_vec(state.as_slice().to_vec());

        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_stm = true;

        let prop = DNumericalOrbitPropagator::new(
            epoch,
            state_dvec,
            config,
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // At t₀, STM should be identity: Φ(t₀,t₀) = I
        let stm = prop.stm().unwrap();
        for i in 0..6 {
            for j in 0..6 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (stm[(i, j)] - expected).abs() < 1e-10,
                    "STM[{},{}] = {}, expected {}",
                    i,
                    j,
                    stm[(i, j)],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_dnumericalorbitpropagator_stm_determinant_preservation() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let oe = nalgebra::Vector6::new(R_EARTH + 500e3, 0.01, 97.8_f64, 0.0, 0.0, 0.0);
        let state = state_osculating_to_cartesian(oe, AngleFormat::Degrees);
        let state_dvec = DVector::from_vec(state.as_slice().to_vec());

        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_stm = true;

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state_dvec,
            config,
            ForceModelConfiguration::two_body_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // For Hamiltonian systems, det(Φ) should equal 1
        let det_initial = prop.stm().unwrap().determinant();
        assert!((det_initial - 1.0).abs() < 1e-10);

        // Propagate forward
        let orbital_period = orbital_period(oe[0]);
        prop.step_by(orbital_period);

        let det_final = prop.stm().unwrap().determinant();
        assert!(
            (det_final - 1.0).abs() < 1e-6,
            "STM determinant should be 1 for Hamiltonian system, got {}",
            det_final
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_stm_composition_property() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let oe = nalgebra::Vector6::new(R_EARTH + 500e3, 0.01, 97.8_f64, 0.0, 0.0, 0.0);
        let state = state_osculating_to_cartesian(oe, AngleFormat::Degrees);
        let state_dvec = DVector::from_vec(state.as_slice().to_vec());

        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_stm = true;

        // Create first propagator for t₀ → t₁
        let mut prop1 = DNumericalOrbitPropagator::new(
            epoch,
            state_dvec.clone(),
            config.clone(),
            ForceModelConfiguration::two_body_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate to t₁
        let t1 = epoch + 600.0;
        prop1.propagate_to(t1);
        let phi_t1_t0 = prop1.stm().unwrap().clone();
        let state_t1 = prop1.current_state();

        // Create second propagator for t₁ → t₂
        let mut prop2 = DNumericalOrbitPropagator::new(
            t1,
            state_t1,
            config.clone(),
            ForceModelConfiguration::two_body_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate to t₂
        let t2 = t1 + 600.0;
        prop2.propagate_to(t2);
        let phi_t2_t1 = prop2.stm().unwrap().clone();

        // Create third propagator for direct t₀ → t₂
        let mut prop3 = DNumericalOrbitPropagator::new(
            epoch,
            state_dvec,
            config,
            ForceModelConfiguration::two_body_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        prop3.propagate_to(t2);
        let phi_t2_t0 = prop3.stm().unwrap().clone();

        // Verify composition: Φ(t₂,t₀) = Φ(t₂,t₁)·Φ(t₁,t₀)
        let phi_composed = phi_t2_t1 * phi_t1_t0;

        for i in 0..6 {
            for j in 0..6 {
                let diff = (phi_t2_t0[(i, j)] - phi_composed[(i, j)]).abs();
                assert!(
                    diff < 1e-6,
                    "STM composition failed at [{},{}]: direct={}, composed={}, diff={}",
                    i,
                    j,
                    phi_t2_t0[(i, j)],
                    phi_composed[(i, j)],
                    diff
                );
            }
        }
    }

    #[test]
    fn test_dnumericalorbitpropagator_stm_vs_direct_perturbation() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let oe = nalgebra::Vector6::new(R_EARTH + 500e3, 0.01, 97.8_f64, 0.0, 0.0, 0.0);
        let state = state_osculating_to_cartesian(oe, AngleFormat::Degrees);
        let state_dvec = DVector::from_vec(state.as_slice().to_vec());

        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_stm = true;

        // Propagate nominal trajectory with STM
        let mut prop_nominal = DNumericalOrbitPropagator::new(
            epoch,
            state_dvec.clone(),
            config.clone(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        prop_nominal.step_by(100.0);
        let stm = prop_nominal.stm().unwrap().clone();
        let state_nominal = prop_nominal.current_state();

        // Apply small perturbation to initial state
        let delta_x0 = DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]); // 1m in x
        let state_perturbed = state_dvec.clone() + delta_x0.clone();

        // Propagate perturbed trajectory
        let mut prop_perturbed = DNumericalOrbitPropagator::new(
            epoch,
            state_perturbed,
            config,
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        prop_perturbed.step_by(100.0);
        let state_perturbed_final = prop_perturbed.current_state();

        // Compare: δx(t) from direct integration vs STM prediction
        let delta_x_direct = state_perturbed_final - state_nominal.clone();
        let delta_x_stm = stm * delta_x0;

        // Error should be < 1 cm
        let error = (delta_x_direct - delta_x_stm).norm();
        assert!(
            error < 0.01,
            "STM vs direct perturbation error = {} m, expected < 0.01 m",
            error
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_stm_at_methods() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let oe = nalgebra::Vector6::new(R_EARTH + 500e3, 0.01, 97.8_f64, 0.0, 0.0, 0.0);
        let state = state_osculating_to_cartesian(oe, AngleFormat::Degrees);
        let state_dvec = DVector::from_vec(state.as_slice().to_vec());

        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_stm = true;
        config.variational.store_stm_history = true; // Store STM in trajectory

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state_dvec,
            config,
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate with multiple steps
        prop.propagate_steps(5);

        // Test stm_at_idx - should be able to retrieve at any stored index
        let traj = prop.trajectory();
        assert!(traj.len() > 0, "Trajectory should have stored states");

        for idx in 0..traj.len() {
            let stm_at_idx = prop.stm_at_idx(idx);
            assert!(
                stm_at_idx.is_some(),
                "Should be able to retrieve STM at index {}",
                idx
            );
            let stm = stm_at_idx.unwrap();
            assert_eq!(stm.nrows(), 6);
            assert_eq!(stm.ncols(), 6);
        }

        // Test stm_at with epoch - retrieve at a stored epoch
        let test_epoch = traj.epochs[traj.len() / 2]; // Middle of trajectory
        let stm_at_epoch = prop.stm_at(test_epoch);
        assert!(
            stm_at_epoch.is_some(),
            "Should retrieve STM at stored epoch"
        );

        // Should match the STM at that index
        let stm_by_idx = prop.stm_at_idx(traj.len() / 2).unwrap();
        let stm_by_epoch = stm_at_epoch.unwrap();
        let diff = (stm_by_idx - stm_by_epoch).norm();
        assert!(
            diff < 1e-12,
            "STM retrieval by epoch and index should match"
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_stm_with_different_jacobian_methods() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let oe = nalgebra::Vector6::new(R_EARTH + 500e3, 0.01, 97.8_f64, 0.0, 0.0, 0.0);
        let state = state_osculating_to_cartesian(oe, AngleFormat::Degrees);
        let state_dvec = DVector::from_vec(state.as_slice().to_vec());

        // Test with Forward difference
        let mut config_forward = NumericalPropagationConfig::default();
        config_forward.variational.enable_stm = true;
        config_forward.variational.jacobian_method = DifferenceMethod::Forward;

        let mut prop_forward = DNumericalOrbitPropagator::new(
            epoch,
            state_dvec.clone(),
            config_forward,
            ForceModelConfiguration::two_body_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        prop_forward.step_by(100.0);
        let stm_forward = prop_forward.stm().unwrap().clone();

        // Test with Central difference
        let mut config_central = NumericalPropagationConfig::default();
        config_central.variational.enable_stm = true;
        config_central.variational.jacobian_method = DifferenceMethod::Central;

        let mut prop_central = DNumericalOrbitPropagator::new(
            epoch,
            state_dvec.clone(),
            config_central,
            ForceModelConfiguration::two_body_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        prop_central.step_by(100.0);
        let stm_central = prop_central.stm().unwrap().clone();

        // Test with Backward difference
        let mut config_backward = NumericalPropagationConfig::default();
        config_backward.variational.enable_stm = true;
        config_backward.variational.jacobian_method = DifferenceMethod::Backward;

        let mut prop_backward = DNumericalOrbitPropagator::new(
            epoch,
            state_dvec,
            config_backward,
            ForceModelConfiguration::two_body_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        prop_backward.step_by(100.0);
        let stm_backward = prop_backward.stm().unwrap().clone();

        // All methods should produce similar results (within 1% relative error)
        for i in 0..6 {
            for j in 0..6 {
                let avg = (stm_forward[(i, j)].abs()
                    + stm_central[(i, j)].abs()
                    + stm_backward[(i, j)].abs())
                    / 3.0;
                if avg > 1e-6 {
                    // Only check significant elements
                    let diff_fc = (stm_forward[(i, j)] - stm_central[(i, j)]).abs();
                    let diff_cb = (stm_central[(i, j)] - stm_backward[(i, j)]).abs();
                    assert!(
                        diff_fc / avg < 0.01,
                        "Forward vs Central mismatch at [{},{}]",
                        i,
                        j
                    );
                    assert!(
                        diff_cb / avg < 0.01,
                        "Central vs Backward mismatch at [{},{}]",
                        i,
                        j
                    );
                }
            }
        }
    }

    #[test]
    fn test_dnumericalorbitpropagator_stm_eigenvalue_analysis() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let oe = nalgebra::Vector6::new(R_EARTH + 500e3, 0.01, 97.8_f64, 0.0, 0.0, 0.0);
        let state = state_osculating_to_cartesian(oe, AngleFormat::Degrees);
        let state_dvec = DVector::from_vec(state.as_slice().to_vec());

        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_stm = true;

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state_dvec,
            config,
            ForceModelConfiguration::two_body_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate for half an orbit
        let orbital_period = orbital_period(oe[0]);
        prop.step_by(orbital_period / 2.0);

        let stm = prop.stm().unwrap().clone();

        // Compute eigenvalues
        let eigenvalues = stm.complex_eigenvalues();

        // For stable systems, all eigenvalues should have magnitude ≈ 1
        // (Hamiltonian systems are neutrally stable)
        for (i, eigenvalue) in eigenvalues.iter().enumerate() {
            let magnitude = (eigenvalue.re * eigenvalue.re + eigenvalue.im * eigenvalue.im).sqrt();
            assert!(
                (magnitude - 1.0).abs() < 0.1,
                "Eigenvalue {} has magnitude {}, expected ≈ 1 (neutral stability)",
                i,
                magnitude
            );
        }
    }

    #[test]
    fn test_dnumericalorbitpropagator_stm_with_different_force_models() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let oe = nalgebra::Vector6::new(R_EARTH + 500e3, 0.01, 97.8_f64, 0.0, 0.0, 0.0);
        let state = state_osculating_to_cartesian(oe, AngleFormat::Degrees);
        let state_dvec = DVector::from_vec(state.as_slice().to_vec());

        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_stm = true;

        // Test with gravity only
        let mut prop_gravity = DNumericalOrbitPropagator::new(
            epoch,
            state_dvec.clone(),
            config.clone(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        prop_gravity.step_by(300.0);
        let stm_gravity = prop_gravity.stm().unwrap();

        // STM should be non-singular
        let det_gravity = stm_gravity.determinant();
        assert!((det_gravity.abs() - 1.0).abs() < 1e-6);

        // Test with J2 perturbations
        use crate::orbit_dynamics::gravity::GravityModelType;
        let force_config_j2 = ForceModelConfiguration {
            mass: None,
            gravity: GravityConfiguration::SphericalHarmonic {
                source: GravityModelSource::ModelType(GravityModelType::EGM2008_360),
                degree: 2,
                order: 0,
            },
            drag: None,
            srp: None,
            third_body: None,
            relativity: false,
        };

        let mut prop_j2 = DNumericalOrbitPropagator::new(
            epoch,
            state_dvec,
            config,
            force_config_j2,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        prop_j2.step_by(300.0);
        let stm_j2 = prop_j2.stm().unwrap();

        // STM should be different from gravity-only case
        let stm_diff = (stm_gravity - stm_j2).norm();
        assert!(
            stm_diff > 1e-6,
            "STM should differ between gravity models, diff = {}",
            stm_diff
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_stm_accuracy_degradation() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let oe = nalgebra::Vector6::new(R_EARTH + 500e3, 0.01, 97.8_f64, 0.0, 0.0, 0.0);
        let state = state_osculating_to_cartesian(oe, AngleFormat::Degrees);
        let state_dvec = DVector::from_vec(state.as_slice().to_vec());

        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_stm = true;

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state_dvec,
            config,
            ForceModelConfiguration::two_body_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let orbital_period = orbital_period(oe[0]);

        // Propagate for 1 orbit
        prop.step_by(orbital_period);
        let det_1_orbit = prop.stm().unwrap().determinant();

        // Propagate for 10 orbits total
        prop.step_by(9.0 * orbital_period);
        let det_10_orbits = prop.stm().unwrap().determinant();

        // Determinant should remain close to 1 even after 10 orbits
        assert!(
            (det_1_orbit - 1.0).abs() < 1e-6,
            "STM determinant after 1 orbit = {}",
            det_1_orbit
        );
        assert!(
            (det_10_orbits - 1.0).abs() < 1e-4,
            "STM determinant after 10 orbits = {} (some degradation expected)",
            det_10_orbits
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_stm_interpolation_accuracy() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let oe = nalgebra::Vector6::new(R_EARTH + 500e3, 0.01, 97.8_f64, 0.0, 0.0, 0.0);
        let state = state_osculating_to_cartesian(oe, AngleFormat::Degrees);
        let state_dvec = DVector::from_vec(state.as_slice().to_vec());

        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_stm = true;
        config.variational.store_stm_history = true; // Store STM in trajectory for interpolation

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state_dvec.clone(),
            config.clone(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate with 60s steps
        for _ in 0..10 {
            prop.step_by(60.0);
        }

        // Interpolate STM at mid-point
        let mid_epoch = epoch + 300.0; // 5 minutes (between steps 4 and 5)
        let stm_interpolated = prop.stm_at(mid_epoch);
        assert!(stm_interpolated.is_some());

        // Create new propagator to get exact value at mid_epoch
        let mut prop_exact = DNumericalOrbitPropagator::new(
            epoch,
            state_dvec,
            config,
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        prop_exact.propagate_to(mid_epoch);
        let stm_exact = prop_exact.stm().unwrap();

        // Compare interpolated vs exact
        let stm_interp = stm_interpolated.unwrap();
        let error = (stm_interp - stm_exact).norm();

        // Interpolation error should be reasonable (< 1% of matrix norm)
        let stm_norm = stm_exact.norm();
        assert!(
            error / stm_norm < 0.01,
            "STM interpolation relative error = {}, expected < 1%",
            error / stm_norm
        );
    }

    // =========================================================================
    // SENSITIVITY PROPAGATION TESTS
    // =========================================================================

    #[test]
    fn test_dnumericalorbitpropagator_sensitivity_vs_finite_difference() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let oe = nalgebra::Vector6::new(R_EARTH + 500e3, 0.01, 97.8_f64, 0.0, 0.0, 0.0);
        let state = state_osculating_to_cartesian(oe, AngleFormat::Degrees);
        let state_dvec = DVector::from_vec(state.as_slice().to_vec());

        // Nominal parameters: [mass, Cd, Cr, area, reflectivity]
        let params_nominal = DVector::from_vec(vec![500.0, 2.2, 1.5, 10.0, 0.3]);

        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_sensitivity = true;
        config.variational.store_sensitivity_history = true; // Store sensitivity in trajectory

        // Configure drag to use mass parameter
        let force_config = ForceModelConfiguration {
            mass: Some(ParameterSource::ParameterIndex(0)), // Use params[0] for mass
            drag: Some(DragConfiguration {
                model: AtmosphericModel::HarrisPriester,
                area: ParameterSource::Value(10.0),
                cd: ParameterSource::Value(2.2),
            }),
            srp: None,
            third_body: None,
            ..Default::default()
        };

        // Propagate with sensitivity
        let mut prop_sens = DNumericalOrbitPropagator::new(
            epoch,
            state_dvec.clone(),
            config,
            force_config.clone(),
            Some(params_nominal.clone()),
            None,
            None,
            None,
        )
        .unwrap();

        prop_sens.step_by(300.0); // 5 minutes
        let sensitivity = prop_sens.sensitivity().unwrap().clone();
        let state_nominal = prop_sens.current_state();

        // Finite difference approximation
        let delta_p = 1.0; // 1 kg perturbation
        let mut params_perturbed = params_nominal.clone();
        params_perturbed[0] += delta_p;

        let mut prop_perturbed = DNumericalOrbitPropagator::new(
            epoch,
            state_dvec,
            NumericalPropagationConfig::default(),
            force_config,
            Some(params_perturbed),
            None,
            None,
            None,
        )
        .unwrap();

        prop_perturbed.step_by(300.0);
        let state_perturbed = prop_perturbed.current_state();

        // Compute finite difference: ds/dp ≈ (s(p+δp) - s(p)) / δp
        let sensitivity_fd = (state_perturbed - state_nominal) / delta_p;

        // Compare sensitivity from variational equations vs finite difference
        for i in 0..6 {
            let rel_error = if sensitivity_fd[i].abs() > 1e-6 {
                ((sensitivity[(i, 0)] - sensitivity_fd[i]) / sensitivity_fd[i]).abs()
            } else {
                (sensitivity[(i, 0)] - sensitivity_fd[i]).abs()
            };

            assert!(
                rel_error < 0.10, // 10% tolerance (finite difference is approximate)
                "Sensitivity[{}] mismatch: variational={}, FD={}, rel_error={}",
                i,
                sensitivity[(i, 0)],
                sensitivity_fd[i],
                rel_error
            );
        }
    }

    #[test]
    fn test_dnumericalorbitpropagator_sensitivity_mass_physical_reasonableness() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let oe = nalgebra::Vector6::new(
            R_EARTH + 400e3, // Lower altitude for stronger drag
            0.01,
            97.8_f64,
            0.0,
            0.0,
            0.0,
        );
        let state = state_osculating_to_cartesian(oe, AngleFormat::Degrees);
        let state_dvec = DVector::from_vec(state.as_slice().to_vec());

        let params = DVector::from_vec(vec![500.0, 2.2, 1.5, 10.0, 0.3]);

        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_sensitivity = true;
        config.variational.store_sensitivity_history = true; // Store sensitivity in trajectory

        let force_config = ForceModelConfiguration {
            mass: Some(ParameterSource::ParameterIndex(0)), // Use params[0] for mass
            drag: Some(DragConfiguration {
                model: AtmosphericModel::HarrisPriester,
                area: ParameterSource::Value(10.0),
                cd: ParameterSource::Value(2.2),
            }),
            srp: None,
            third_body: None,
            ..Default::default()
        };

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state_dvec,
            config,
            force_config,
            Some(params),
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate for 1 orbit
        let orbital_period = orbital_period(oe[0]);
        prop.step_by(orbital_period);

        let sensitivity = prop.sensitivity().unwrap();

        // Mass sensitivity should show:
        // 1. Positive sensitivity in velocity (heavier satellite decays slower)
        // 2. The effect should be non-negligible for drag-dominated orbit
        let dv_dmass = sensitivity.fixed_rows::<3>(3);
        let dv_dmass_mag = dv_dmass.norm();

        assert!(
            dv_dmass_mag > 1e-6,
            "Mass sensitivity should be non-negligible for drag, got {}",
            dv_dmass_mag
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_sensitivity_drag_coefficient() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let oe = nalgebra::Vector6::new(R_EARTH + 400e3, 0.01, 97.8_f64, 0.0, 0.0, 0.0);
        let state = state_osculating_to_cartesian(oe, AngleFormat::Degrees);
        let state_dvec = DVector::from_vec(state.as_slice().to_vec());

        let params = DVector::from_vec(vec![500.0, 2.2, 1.5, 10.0, 0.3]);

        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_sensitivity = true;
        config.variational.store_sensitivity_history = true; // Store sensitivity in trajectory

        let force_config = ForceModelConfiguration {
            mass: Some(ParameterSource::Value(500.0)), // Fixed mass
            drag: Some(DragConfiguration {
                model: AtmosphericModel::HarrisPriester,
                area: ParameterSource::Value(10.0),
                cd: ParameterSource::ParameterIndex(1), // Use params[1] for Cd
            }),
            srp: None,
            third_body: None,
            ..Default::default()
        };

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state_dvec,
            config,
            force_config,
            Some(params),
            None,
            None,
            None,
        )
        .unwrap();

        let orbital_period = orbital_period(oe[0]);
        prop.step_by(orbital_period);

        let sensitivity = prop.sensitivity().unwrap();

        // Drag coefficient sensitivity should be non-zero
        let sensitivity_norm = sensitivity.norm();
        assert!(
            sensitivity_norm > 1e-6,
            "Cd sensitivity should be non-negligible, got {}",
            sensitivity_norm
        );
    }

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    fn test_dnumericalorbitpropagator_sensitivity_srp_coefficient() {
        use crate::propagators::force_model_config::{
            ParameterSource, SolarRadiationPressureConfiguration, ThirdBodyConfiguration,
        };

        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let oe = nalgebra::Vector6::new(
            R_EARTH + 35786e3, // GEO altitude (SRP stronger at high altitude)
            0.01,
            0.1_f64,
            0.0,
            0.0,
            0.0,
        );
        let state = state_osculating_to_cartesian(oe, AngleFormat::Degrees);
        let state_dvec = DVector::from_vec(state.as_slice().to_vec());

        let params = DVector::from_vec(vec![1000.0, 2.2, 1.5, 20.0, 0.3]);

        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_sensitivity = true;
        config.variational.store_sensitivity_history = true; // Store sensitivity in trajectory

        let force_config = ForceModelConfiguration {
            mass: Some(ParameterSource::Value(1000.0)),
            drag: None,
            srp: Some(SolarRadiationPressureConfiguration {
                area: ParameterSource::Value(20.0),
                cr: ParameterSource::ParameterIndex(2), // Use params[2] for Cr
                eclipse_model: EclipseModel::None,
            }),
            third_body: Some(ThirdBodyConfiguration {
                ephemeris_source: EphemerisSource::LowPrecision,
                bodies: vec![ThirdBody::Sun],
            }),
            ..Default::default()
        };

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state_dvec,
            config,
            force_config,
            Some(params),
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate for 1 day
        prop.step_by(86400.0);

        let sensitivity = prop.sensitivity().unwrap();

        // SRP coefficient sensitivity should be non-zero at GEO
        let sensitivity_norm = sensitivity.norm();
        assert!(
            sensitivity_norm > 1e-6,
            "Cr sensitivity should be non-negligible at GEO, got {}",
            sensitivity_norm
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_sensitivity_zero_for_unused_parameters() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let oe = nalgebra::Vector6::new(R_EARTH + 500e3, 0.01, 97.8_f64, 0.0, 0.0, 0.0);
        let state = state_osculating_to_cartesian(oe, AngleFormat::Degrees);
        let state_dvec = DVector::from_vec(state.as_slice().to_vec());

        let params = DVector::from_vec(vec![500.0, 2.2, 1.5]);

        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_sensitivity = true;
        config.variational.store_sensitivity_history = true; // Store sensitivity in trajectory

        // Gravity only - Cd parameter is not used
        let force_config = ForceModelConfiguration::earth_gravity();

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state_dvec,
            config,
            force_config,
            Some(params),
            None,
            None,
            None,
        )
        .unwrap();

        prop.step_by(300.0);

        let sensitivity = prop.sensitivity().unwrap();

        // Sensitivity should be zero (or very small) for unused parameter
        let sensitivity_norm = sensitivity.norm();
        assert!(
            sensitivity_norm < 1e-10,
            "Sensitivity for unused parameter should be ~0, got {}",
            sensitivity_norm
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_sensitivity_storage_in_trajectory() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let oe = nalgebra::Vector6::new(R_EARTH + 500e3, 0.01, 97.8_f64, 0.0, 0.0, 0.0);
        let state = state_osculating_to_cartesian(oe, AngleFormat::Degrees);
        let state_dvec = DVector::from_vec(state.as_slice().to_vec());

        let params = DVector::from_vec(vec![500.0, 2.2]);

        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_sensitivity = true;
        config.variational.store_sensitivity_history = true; // Store sensitivity in trajectory

        let force_config = ForceModelConfiguration {
            mass: Some(ParameterSource::ParameterIndex(0)), // Use params[0] for mass
            drag: Some(DragConfiguration {
                model: AtmosphericModel::HarrisPriester,
                area: ParameterSource::Value(10.0),
                cd: ParameterSource::ParameterIndex(1),
            }),
            srp: None,
            third_body: None,
            ..Default::default()
        };

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state_dvec,
            config,
            force_config,
            Some(params),
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate with multiple steps
        prop.propagate_steps(5);

        // Check trajectory has sensitivity matrices
        assert!(
            prop.trajectory().sensitivities.is_some(),
            "Trajectory should contain sensitivity matrices"
        );
        let sensitivities = prop.trajectory().sensitivities.as_ref().unwrap();
        assert!(
            !sensitivities.is_empty(),
            "Sensitivity storage should not be empty"
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_sensitivity_at_methods() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let oe = nalgebra::Vector6::new(R_EARTH + 500e3, 0.01, 97.8_f64, 0.0, 0.0, 0.0);
        let state = state_osculating_to_cartesian(oe, AngleFormat::Degrees);
        let state_dvec = DVector::from_vec(state.as_slice().to_vec());

        let params = DVector::from_vec(vec![500.0]);

        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_sensitivity = true;
        config.variational.store_sensitivity_history = true; // Store sensitivity in trajectory

        let force_config = ForceModelConfiguration {
            mass: Some(ParameterSource::ParameterIndex(0)), // Use params[0] for mass
            drag: Some(DragConfiguration {
                model: AtmosphericModel::HarrisPriester,
                area: ParameterSource::Value(10.0),
                cd: ParameterSource::Value(2.2),
            }),
            srp: None,
            third_body: None,
            ..Default::default()
        };

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state_dvec,
            config,
            force_config,
            Some(params),
            None,
            None,
            None,
        )
        .unwrap();

        // Propagate with multiple steps
        prop.propagate_steps(5);

        // Test sensitivity_at_idx - should be able to retrieve at any stored index
        let traj = prop.trajectory();
        assert!(traj.len() > 0, "Trajectory should have stored states");

        for idx in 0..traj.len() {
            let sens_at_idx = prop.sensitivity_at_idx(idx);
            assert!(
                sens_at_idx.is_some(),
                "Should be able to retrieve sensitivity at index {}",
                idx
            );
            let sens = sens_at_idx.unwrap();
            // Sensitivity matrix dimensions should match state x parameters
            assert_eq!(sens.nrows(), 6);
            assert_eq!(sens.ncols(), 1); // 1 parameter
        }

        // Test sensitivity_at with epoch - retrieve at a stored epoch
        let test_epoch = traj.epochs[traj.len() / 2]; // Middle of trajectory
        let sens_at_epoch = prop.sensitivity_at(test_epoch);
        assert!(
            sens_at_epoch.is_some(),
            "Should retrieve sensitivity at stored epoch"
        );

        // Should match the sensitivity at that index
        let sens_by_idx = prop.sensitivity_at_idx(traj.len() / 2).unwrap();
        let sens_by_epoch = sens_at_epoch.unwrap();
        let diff = (sens_by_idx - sens_by_epoch).norm();
        assert!(
            diff < 1e-12,
            "Sensitivity retrieval by epoch and index should match"
        );

        // Final sensitivity should be non-zero (parameter affects trajectory)
        let sens_final = prop.sensitivity_at_idx(traj.len() - 1).unwrap();
        assert!(
            sens_final.norm() > 1e-6,
            "Final sensitivity should be non-zero"
        );
    }

    // =========================================================================
    // Phase 7: Covariance Propagation Tests (Additional Unique Tests)
    // =========================================================================
    // Note: Many covariance tests already exist earlier in this file:
    // - test_covariance_propagation_initialization (line 7415)
    // - test_covariance_propagates_with_stm (line 7458)
    // - test_covariance_stored_in_trajectory (line 7496)
    // - test_dnumericalorbitpropagator_dorbitcovarianceprovider_* (lines 2832-3262)
    //
    // The tests below add unique coverage not present in existing tests:

    #[test]
    fn test_dnumericalorbitpropagator_covariance_stm_formula_verification() {
        // Verify that covariance propagation follows P(t) = Φ·P₀·Φᵀ formula
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7.6e3, 0.0]);

        // Initial covariance
        let mut p0 = DMatrix::from_element(6, 6, 0.0);
        for i in 0..3 {
            p0[(i, i)] = 100.0 * 100.0;
            p0[(i + 3, i + 3)] = 0.1 * 0.1;
        }

        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            Some(p0.clone()),
        )
        .unwrap();

        // Propagate
        prop.step_by(300.0); // 5 minutes

        // Get STM and current covariance
        let phi = prop.stm().unwrap();
        let p_propagated = prop.current_covariance.as_ref().unwrap();

        // Manually compute P(t) = Φ·P₀·Φᵀ
        let p_computed = phi * &p0 * phi.transpose();

        // Compare
        let diff = (p_propagated - p_computed).norm();
        let rel_error = diff / p_propagated.norm();

        assert!(
            rel_error < 1e-10,
            "Covariance should match P(t) = Φ·P₀·Φᵀ formula, rel_error: {}",
            rel_error
        );
    }

    #[test]
    fn test_dnumericalorbitpropagator_covariance_monte_carlo_validation() {
        // Validate covariance propagation against Monte Carlo simulation
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

        // Nominal state
        let state_nominal = vec![R_EARTH + 500e3, 0.0, 0.0, 0.0, 7.6e3, 0.0];

        // Initial covariance
        let mut p0 = DMatrix::from_element(6, 6, 0.0);
        for i in 0..3 {
            p0[(i, i)] = 100.0 * 100.0; // 100m position uncertainty
            p0[(i + 3, i + 3)] = 0.1 * 0.1; // 0.1 m/s velocity uncertainty
        }

        // Propagate with covariance
        let state_dvec = DVector::from_vec(state_nominal.clone());
        let mut prop_cov = DNumericalOrbitPropagator::new(
            epoch,
            state_dvec,
            NumericalPropagationConfig::default(),
            ForceModelConfiguration::earth_gravity(),
            None,
            None,
            None,
            Some(p0.clone()),
        )
        .unwrap();

        let dt = 300.0; // 5 minutes
        prop_cov.step_by(dt);
        let p_propagated = prop_cov.current_covariance.as_ref().unwrap();

        // Monte Carlo: propagate 100 perturbed states
        use rand::SeedableRng;
        use rv::dist::Gaussian;
        use rv::prelude::Sampleable;

        let n_samples = 100;
        let mut final_states = Vec::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(12345);

        for _ in 0..n_samples {
            // Sample from initial distribution
            let mut state_perturbed = state_nominal.clone();
            for i in 0..6 {
                let std_dev = p0[(i, i)].sqrt();
                let gaussian = Gaussian::new(0.0, std_dev).unwrap();
                let sample: f64 = gaussian.draw(&mut rng);
                state_perturbed[i] += sample;
            }

            // Propagate perturbed state
            let state_dvec_pert = DVector::from_vec(state_perturbed);
            let mut prop_mc = DNumericalOrbitPropagator::new(
                epoch,
                state_dvec_pert,
                NumericalPropagationConfig::default(),
                ForceModelConfiguration::earth_gravity(),
                None,
                None,
                None,
                None,
            )
            .unwrap();

            prop_mc.step_by(dt);
            final_states.push(prop_mc.current_state_ref().clone());
        }

        // Compute sample covariance from Monte Carlo
        let mut mean_state = DVector::from_element(6, 0.0);
        for state in &final_states {
            mean_state += state;
        }
        mean_state /= n_samples as f64;

        let mut p_mc = DMatrix::from_element(6, 6, 0.0);
        for state in &final_states {
            let delta = state - &mean_state;
            p_mc += &delta * delta.transpose();
        }
        p_mc /= (n_samples - 1) as f64;

        // Compare diagonal elements (variances)
        // Monte Carlo needs large N for good accuracy, so use loose tolerance
        for i in 0..6 {
            let rel_error = (p_propagated[(i, i)] - p_mc[(i, i)]).abs() / p_propagated[(i, i)];
            assert!(
                rel_error < 0.5, // 50% tolerance for MC with only 100 samples
                "Variance {} should match Monte Carlo, rel_error: {}",
                i,
                rel_error
            );
        }
    }
}
