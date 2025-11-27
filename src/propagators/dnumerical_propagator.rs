/*!
 * Generic numerical propagator for arbitrary dynamical systems
 *
 * This module provides a high-fidelity numerical propagator that:
 * - Accepts user-defined dynamics functions for any system
 * - Supports arbitrary state dimensions (N-dimensional)
 * - Provides STM and sensitivity matrix propagation
 * - Integrates with event detection framework
 * - Supports trajectory storage and interpolation
 *
 * Unlike [`DNumericalOrbitPropagator`] which includes built-in orbital
 * force models, this propagator is completely generic and does not make
 * assumptions about the problem domain. It can be applied to attitude
 * dynamics, chemical kinetics, population models, or any other ODE system.
 *
 * [`DNumericalOrbitPropagator`]: crate::propagators::DNumericalOrbitPropagator
 */

use std::sync::Arc;

use nalgebra::{DMatrix, DVector};

use crate::integrators::traits::DIntegrator;
use crate::math::interpolation::{
    CovarianceInterpolationConfig, CovarianceInterpolationMethod, InterpolationConfig,
    InterpolationMethod,
};
use crate::math::jacobian::DNumericalJacobian;
use crate::math::jacobian::DifferenceMethod;
use crate::math::sensitivity::DNumericalSensitivity;
use crate::time::Epoch;
use crate::trajectories::DTrajectory;
use crate::trajectories::traits::{
    InterpolatableTrajectory, STMStorage, SensitivityStorage, Trajectory,
};
use crate::utils::errors::BraheError;
use crate::utils::identifiable::Identifiable;
use crate::utils::state_providers::{DCovarianceProvider, DStateProvider};

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

/// Generic numerical propagator for arbitrary dynamical systems
///
/// This propagator wraps a dynamic-sized adaptive integrator and accepts
/// user-defined dynamics functions. It is completely generic and can be
/// applied to any system of ordinary differential equations, including:
/// - Orbital mechanics (position and velocity dynamics)
/// - Attitude dynamics (quaternion or Euler angle kinematics)
/// - Chemical kinetics (concentration evolution)
/// - Population models (ecological or epidemiological systems)
/// - Control systems (state-space models)
/// - Any other N-dimensional ODE system
///
/// # Features
/// - **STM Propagation**: State transition matrix Φ(t, t₀) for covariance propagation
/// - **Sensitivity Analysis**: Parameter sensitivity matrix S = ∂x/∂p
/// - **Event Detection**: Monitor and react to user-defined events during propagation
/// - **Trajectory Storage**: Configurable history with interpolation support
/// - **Adaptive Integration**: Multiple adaptive integrator methods (RK4, RKF45, DP54, etc.)
///
/// # State Dimensions
/// The propagator supports arbitrary state dimensions (N-D systems).
/// The dynamics function defines the system dimension implicitly through
/// the returned derivative vector.
///
/// # Example
///
/// ```rust
/// use brahe::propagators::{DNumericalPropagator, NumericalPropagationConfig};
/// use brahe::propagators::traits::DStatePropagator;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::DVector;
///
/// // Define dynamics: simple harmonic oscillator
/// // dx/dt = v, dv/dt = -ω²x
/// let omega = 1.0;
/// let dynamics = Box::new(move |_t: f64, x: &DVector<f64>, _p: Option<&DVector<f64>>| {
///     DVector::from_vec(vec![x[1], -omega * omega * x[0]])
/// });
///
/// // Create initial state
/// let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let state = DVector::from_vec(vec![1.0, 0.0]);  // [position, velocity]
///
/// // Create propagator
/// let mut prop = DNumericalPropagator::new(
///     epoch,
///     state,
///     dynamics,
///     NumericalPropagationConfig::default(),
///     None,  // parameters
///     None,  // control_input
///     None,  // initial_covariance
/// ).unwrap();
///
/// // Propagate one period
/// prop.propagate_to(epoch + 2.0 * std::f64::consts::PI);
/// ```
pub struct DNumericalPropagator {
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
    trajectory: DTrajectory,
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

impl DNumericalPropagator {
    // =========================================================================
    // Construction
    // =========================================================================

    /// Create a new generic numerical propagator
    ///
    /// This is the primary constructor that builds the integrator based on the
    /// propagation configuration and accepts a user-defined dynamics function.
    ///
    /// # Arguments
    /// * `epoch` - Initial epoch
    /// * `state` - Initial state vector (N-dimensional)
    /// * `dynamics_fn` - Dynamics function: `f(t, x, params) -> dx/dt`
    /// * `propagation_config` - Numerical propagation configuration (integrator method + settings)
    /// * `params` - Optional parameter vector for the dynamics function
    /// * `control_input` - Optional control input function
    /// * `initial_covariance` - Optional initial covariance matrix P₀ (enables STM propagation)
    ///
    /// # Returns
    /// New propagator ready for propagation, or error if configuration is invalid
    ///
    /// # Errors
    /// Returns `BraheError` if:
    /// - Sensitivity propagation is enabled but no parameters are provided
    ///
    /// # Example
    ///
    /// ```rust
    /// use brahe::propagators::{DNumericalPropagator, NumericalPropagationConfig};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use nalgebra::DVector;
    ///
    /// // Define simple harmonic oscillator dynamics
    /// let omega = 1.0;
    /// let dynamics = Box::new(move |_t: f64, x: &DVector<f64>, _p: Option<&DVector<f64>>| {
    ///     DVector::from_vec(vec![x[1], -omega * omega * x[0]])
    /// });
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = DVector::from_vec(vec![1.0, 0.0]);
    ///
    /// let prop = DNumericalPropagator::new(
    ///     epoch,
    ///     state,
    ///     dynamics,
    ///     NumericalPropagationConfig::default(),
    ///     None,
    ///     None,
    ///     None,
    /// ).unwrap();
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        epoch: Epoch,
        state: DVector<f64>,
        dynamics_fn: DStateDynamics,
        propagation_config: super::NumericalPropagationConfig,
        params: Option<DVector<f64>>,
        control_input: DControlInput,
        initial_covariance: Option<DMatrix<f64>>,
    ) -> Result<Self, BraheError> {
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

        // Wrap for main integrator
        let dynamics_fn = Arc::from(dynamics_fn);
        let dynamics = Self::wrap_for_integrator(Arc::clone(&dynamics_fn));

        // Get initial step size from config
        let initial_dt = propagation_config.integrator.initial_step.unwrap_or(60.0);

        // Build Jacobian provider if STM or sensitivity enabled
        // (sensitivity propagation requires the Jacobian: dS/dt = A*S + B where A is ∂f/∂x)
        let jacobian_provider = if enable_stm || enable_sensitivity {
            Some(Self::build_jacobian_provider(
                Arc::clone(&dynamics_fn),
                propagation_config.variational.jacobian_method,
            ))
        } else {
            None
        };

        // Build Sensitivity provider if enabled
        let sensitivity_provider = if enable_sensitivity {
            Some(Self::build_sensitivity_provider(
                Arc::clone(&dynamics_fn),
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
        let mut trajectory = DTrajectory::new(state_dim);

        // Enable STM/sensitivity storage in trajectory if configured
        if propagation_config.variational.store_stm_history {
            trajectory.enable_stm_storage();
        }
        if propagation_config.variational.store_sensitivity_history && !params.is_empty() {
            trajectory.enable_sensitivity_storage(params.len());
        }

        // Store initial state in trajectory with identity STM and zero sensitivity if needed
        let initial_stm = if propagation_config.variational.store_stm_history {
            Some(DMatrix::identity(state_dim, state_dim))
        } else {
            None
        };
        let initial_sensitivity =
            if propagation_config.variational.store_sensitivity_history && !params.is_empty() {
                Some(DMatrix::zeros(state_dim, params.len()))
            } else {
                None
            };
        trajectory.add_full(
            epoch,
            state_eci.clone(),
            initial_covariance.clone(),
            initial_stm,
            initial_sensitivity,
        );

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
            dt: initial_dt,
            dt_next: initial_dt,
            x_initial: state_eci.clone(),
            x_curr: state_eci,
            params,
            state_dim,
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
            // Pass params only if non-empty
            let params_opt = if !self.params.is_empty() {
                Some(&self.params)
            } else {
                None
            };

            if let Some(event) = dscan_for_event(
                detector.as_ref(),
                idx,
                &state_fn,
                epoch_prev,
                epoch_new,
                x_prev,
                x_new,
                params_opt,
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

                    // Mark this event as processed so it won't trigger again on restart
                    detector.mark_processed();

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
    pub fn trajectory(&self) -> &DTrajectory {
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
        self.query_events().by_name_contains(name).collect()
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
        self.query_events().in_time_range(start, end).collect()
    }

    /// Query events with flexible filtering
    ///
    /// Returns an EventQuery that supports chainable filters and
    /// standard iterator methods.
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::propagators::DNumericalPropagator;
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # use brahe::propagators::NumericalPropagationConfig;
    /// # use nalgebra::DVector;
    /// # let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    /// # let state = DVector::from_vec(vec![1.0, 0.0]);
    /// # let omega = 1.0;
    /// # let dynamics = Box::new(move |_t: f64, x: &DVector<f64>, _p: Option<&DVector<f64>>| {
    /// #     DVector::from_vec(vec![x[1], -omega * omega * x[0]])
    /// # });
    /// # let mut prop = DNumericalPropagator::new(
    /// #     epoch, state, dynamics, NumericalPropagationConfig::default(),
    /// #     None, None, None
    /// # ).unwrap();
    /// // Get events from detector 1 in time range
    /// let events: Vec<_> = prop.query_events()
    ///     .by_detector_index(1)
    ///     .in_time_range(epoch, epoch + 3600.0)
    ///     .collect();
    ///
    /// // Count events by name
    /// let count = prop.query_events()
    ///     .by_name_contains("threshold")
    ///     .count();
    /// ```
    pub fn query_events(
        &self,
    ) -> crate::events::EventQuery<'_, std::slice::Iter<'_, DDetectedEvent>> {
        crate::events::EventQuery::new(self.event_log.iter())
    }

    /// Get events by detector index
    ///
    /// Returns all events detected by the specified detector.
    ///
    /// # Arguments
    /// * `index` - Detector index (0-based, order detectors were added)
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::propagators::DNumericalPropagator;
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # use brahe::propagators::NumericalPropagationConfig;
    /// # use nalgebra::DVector;
    /// # let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    /// # let state = DVector::from_vec(vec![1.0, 0.0]);
    /// # let omega = 1.0;
    /// # let dynamics = Box::new(move |_t: f64, x: &DVector<f64>, _p: Option<&DVector<f64>>| {
    /// #     DVector::from_vec(vec![x[1], -omega * omega * x[0]])
    /// # });
    /// # let mut prop = DNumericalPropagator::new(
    /// #     epoch, state, dynamics, NumericalPropagationConfig::default(),
    /// #     None, None, None
    /// # ).unwrap();
    /// // Get all events from detector 0
    /// let events = prop.events_by_detector_index(0);
    /// ```
    pub fn events_by_detector_index(&self, index: usize) -> Vec<&DDetectedEvent> {
        self.query_events().by_detector_index(index).collect()
    }

    /// Get events by detector index within time range
    ///
    /// Returns events from the specified detector that occurred in the time range.
    ///
    /// # Arguments
    /// * `index` - Detector index (0-based)
    /// * `start` - Start of time range (inclusive)
    /// * `end` - End of time range (inclusive)
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::propagators::DNumericalPropagator;
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # use brahe::propagators::NumericalPropagationConfig;
    /// # use nalgebra::DVector;
    /// # let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    /// # let state = DVector::from_vec(vec![1.0, 0.0]);
    /// # let omega = 1.0;
    /// # let dynamics = Box::new(move |_t: f64, x: &DVector<f64>, _p: Option<&DVector<f64>>| {
    /// #     DVector::from_vec(vec![x[1], -omega * omega * x[0]])
    /// # });
    /// # let mut prop = DNumericalPropagator::new(
    /// #     epoch, state, dynamics, NumericalPropagationConfig::default(),
    /// #     None, None, None
    /// # ).unwrap();
    /// // Get detector 1 events in time range
    /// let events = prop.events_by_detector_index_in_range(1, epoch, epoch + 3600.0);
    /// ```
    pub fn events_by_detector_index_in_range(
        &self,
        index: usize,
        start: Epoch,
        end: Epoch,
    ) -> Vec<&DDetectedEvent> {
        self.query_events()
            .by_detector_index(index)
            .in_time_range(start, end)
            .collect()
    }

    /// Get events by name within time range
    ///
    /// Returns events matching name (substring) that occurred in the time range.
    ///
    /// # Arguments
    /// * `name` - Substring to search for in event names
    /// * `start` - Start of time range (inclusive)
    /// * `end` - End of time range (inclusive)
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::propagators::DNumericalPropagator;
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # use brahe::propagators::NumericalPropagationConfig;
    /// # use nalgebra::DVector;
    /// # let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    /// # let state = DVector::from_vec(vec![1.0, 0.0]);
    /// # let omega = 1.0;
    /// # let dynamics = Box::new(move |_t: f64, x: &DVector<f64>, _p: Option<&DVector<f64>>| {
    /// #     DVector::from_vec(vec![x[1], -omega * omega * x[0]])
    /// # });
    /// # let mut prop = DNumericalPropagator::new(
    /// #     epoch, state, dynamics, NumericalPropagationConfig::default(),
    /// #     None, None, None
    /// # ).unwrap();
    /// // Get threshold events in time range
    /// let events = prop.events_by_name_in_range("threshold", epoch, epoch + 3600.0);
    /// ```
    pub fn events_by_name_in_range(
        &self,
        name: &str,
        start: Epoch,
        end: Epoch,
    ) -> Vec<&DDetectedEvent> {
        self.query_events()
            .by_name_contains(name)
            .in_time_range(start, end)
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
            // Create params reference if params is not empty
            let params_ref = if self.params.is_empty() {
                None
            } else {
                Some(&self.params)
            };

            match self.propagation_mode {
                PropagationMode::StateOnly => {
                    // Basic state propagation only
                    let result = self.integrator.step(
                        self.t_rel,
                        self.x_curr.clone(),
                        params_ref,
                        Some(dt_requested),
                    );

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
                        params_ref,
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

impl super::traits::DStatePropagator for DNumericalPropagator {
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
        self.x_initial.clone()
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
        self.trajectory = DTrajectory::new(self.state_dim);

        // Clear event state
        self.event_log.clear();
        self.terminated = false;

        // Reset processed state on all event detectors so they can trigger again
        for detector in &self.event_detectors {
            detector.reset_processed();
        }
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

impl InterpolationConfig for DNumericalPropagator {
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

impl CovarianceInterpolationConfig for DNumericalPropagator {
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

impl DStateProvider for DNumericalPropagator {
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
// DCovarianceProvider Trait
// =============================================================================

impl DCovarianceProvider for DNumericalPropagator {
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
// Identifiable Trait
// =============================================================================

impl Identifiable for DNumericalPropagator {
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
    use crate::events::{DEventCallback, DTimeEvent, DValueEvent, EventDirection};
    use crate::propagators::NumericalPropagationConfig;
    use crate::propagators::traits::DStatePropagator as DStatePropagatorTrait;
    use crate::time::TimeSystem;
    use crate::utils::state_providers::DStateProvider;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    // =============================================================================
    // Test Helpers
    // =============================================================================

    /// Simple harmonic oscillator dynamics: dx/dt = v, dv/dt = -ω²x
    ///
    /// This is a linear 2D system with analytical solution. Perfect for testing
    /// all propagator features with known expected behavior.
    ///
    /// Energy is conserved: E = 0.5(v² + ω²x²) = constant
    fn sho_dynamics(omega: f64) -> DStateDynamics {
        Box::new(
            move |_t: f64, x: &DVector<f64>, _p: Option<&DVector<f64>>| {
                let pos = x[0];
                let vel = x[1];
                DVector::from_vec(vec![vel, -omega * omega * pos])
            },
        )
    }

    /// Damped harmonic oscillator with parameters [omega, zeta]
    ///
    /// Dynamics: dx/dt = v, dv/dt = -ω²x - 2ζωv
    /// where params[0] = ω (natural frequency), params[1] = ζ (damping ratio)
    ///
    /// Used to test parameter sensitivity propagation.
    fn damped_sho_dynamics() -> DStateDynamics {
        Box::new(|_t: f64, x: &DVector<f64>, p: Option<&DVector<f64>>| {
            let params = p.expect("Damped SHO requires parameters [omega, zeta]");
            let omega = params[0];
            let zeta = params[1];
            let pos = x[0];
            let vel = x[1];
            DVector::from_vec(vec![vel, -omega * omega * pos - 2.0 * zeta * omega * vel])
        })
    }

    /// Analytical solution for simple harmonic oscillator
    ///
    /// Given initial conditions (x0, v0) and time t, returns (x(t), v(t))
    #[allow(dead_code)]
    fn sho_analytical_solution(x0: f64, v0: f64, omega: f64, t: f64) -> (f64, f64) {
        let x = x0 * (omega * t).cos() + (v0 / omega) * (omega * t).sin();
        let v = -x0 * omega * (omega * t).sin() + v0 * (omega * t).cos();
        (x, v)
    }

    /// Compute energy of simple harmonic oscillator
    ///
    /// E = 0.5(v² + ω²x²) should be conserved
    #[allow(dead_code)]
    fn sho_energy(x: f64, v: f64, omega: f64) -> f64 {
        0.5 * (v * v + omega * omega * x * x)
    }

    /// Create a simple 2D SHO propagator for testing
    ///
    /// Default parameters: ω = 1.0, x0 = 1.0, v0 = 0.0
    fn create_test_sho_propagator() -> DNumericalPropagator {
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![1.0, 0.0]); // [pos, vel]
        let omega = 1.0;
        let dynamics = sho_dynamics(omega);

        // Configure integrator with appropriate max step for SHO dynamics
        // SHO period = 2π ≈ 6.28s, use max_step = 0.5s for adequate sampling
        let mut config = NumericalPropagationConfig::default();
        config.integrator.max_step = Some(0.5);

        DNumericalPropagator::new(epoch, state, dynamics, config, None, None, None).unwrap()
    }

    /// Create a damped SHO propagator with parameter sensitivity enabled
    fn create_test_damped_sho_with_sensitivity() -> DNumericalPropagator {
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![1.0, 0.0]);
        let params = DVector::from_vec(vec![1.0, 0.1]); // [omega, zeta]
        let dynamics = damped_sho_dynamics();

        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_sensitivity = true;

        DNumericalPropagator::new(epoch, state, dynamics, config, Some(params), None, None).unwrap()
    }

    /// Create a SHO propagator with STM enabled
    fn create_test_sho_with_stm() -> DNumericalPropagator {
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![1.0, 0.0]);
        let omega = 1.0;
        let dynamics = sho_dynamics(omega);

        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_stm = true;
        config.variational.store_stm_history = true;

        DNumericalPropagator::new(epoch, state, dynamics, config, None, None, None).unwrap()
    }

    // =============================================================================
    // Construction & Configuration Tests
    // =============================================================================

    #[test]
    fn test_dnumericalpropagator_construction_default() {
        // Test basic construction with default configuration
        let prop = create_test_sho_propagator();

        assert_eq!(DStatePropagatorTrait::state_dim(&prop), 2);
        assert_eq!(prop.current_state(), DVector::from_vec(vec![1.0, 0.0]));
        assert!(prop.stm().is_none()); // STM not enabled by default
        assert!(prop.sensitivity().is_none()); // Sensitivity not enabled by default
    }

    #[test]
    fn test_dnumericalpropagator_construction_with_stm() {
        let prop = create_test_sho_with_stm();

        assert_eq!(DStatePropagatorTrait::state_dim(&prop), 2);

        // STM should be identity initially
        let stm = prop.stm().expect("STM should be enabled");
        assert_eq!(stm.nrows(), 2);
        assert_eq!(stm.ncols(), 2);
        assert_abs_diff_eq!(stm[(0, 0)], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(stm[(1, 1)], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(stm[(0, 1)], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(stm[(1, 0)], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_dnumericalpropagator_construction_with_sensitivity() {
        let prop = create_test_damped_sho_with_sensitivity();

        assert_eq!(DStatePropagatorTrait::state_dim(&prop), 2);

        // Sensitivity should be zeros initially (2x2 for 2 params)
        let sens = prop.sensitivity().expect("Sensitivity should be enabled");
        assert_eq!(sens.nrows(), 2); // state dim
        assert_eq!(sens.ncols(), 2); // param dim

        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(sens[(i, j)], 0.0, epsilon = 1e-12);
            }
        }
    }

    // =============================================================================
    // DStatePropagator Trait Tests
    // =============================================================================

    #[test]
    fn test_dstatepropagator_step_by_forward() {
        let mut prop = create_test_sho_propagator();
        let initial_epoch = prop.current_epoch();

        // Propagate forward by one period (2π seconds for ω=1)
        let period = 2.0 * PI;
        prop.step_by(period);

        // Should be back at initial position (approximately)
        let state = prop.current_state();
        assert_abs_diff_eq!(state[0], 1.0, epsilon = 1e-3);
        assert_abs_diff_eq!(state[1], 0.0, epsilon = 2e-3);

        // Time should have advanced
        assert_abs_diff_eq!(
            (prop.current_epoch() - initial_epoch).abs(),
            period,
            epsilon = 0.1
        );
    }

    #[test]
    fn test_dstatepropagator_propagate_to_forward() {
        let mut prop = create_test_sho_propagator();
        let initial_epoch = prop.current_epoch();

        // Propagate to half period
        let target_epoch = initial_epoch + PI;
        prop.propagate_to(target_epoch);

        // Should be at opposite position
        let state = prop.current_state();
        assert_abs_diff_eq!(state[0], -1.0, epsilon = 1e-3);
        assert_abs_diff_eq!(state[1], 0.0, epsilon = 2e-3);

        assert_abs_diff_eq!(
            (prop.current_epoch() - target_epoch).abs(),
            0.0,
            epsilon = 0.1
        );
    }

    #[test]
    fn test_dstatepropagator_step_by_backward() {
        let mut prop = create_test_sho_propagator();

        // First propagate forward
        prop.step_by(PI);
        let _mid_state = prop.current_state();

        // Then propagate backward
        prop.step_by(-PI);

        // Should be back at initial state
        let state = prop.current_state();
        assert_abs_diff_eq!(state[0], 1.0, epsilon = 1e-3);
        assert_abs_diff_eq!(state[1], 0.0, epsilon = 2e-3);
    }

    #[test]
    fn test_dstatepropagator_propagate_to_backward() {
        let mut prop = create_test_sho_propagator();
        let initial_epoch = prop.current_epoch();

        // Propagate forward then back
        prop.propagate_to(initial_epoch + PI);
        prop.propagate_to(initial_epoch);

        // Should be back at initial state
        let state = prop.current_state();
        assert_abs_diff_eq!(state[0], 1.0, epsilon = 1e-3);
        assert_abs_diff_eq!(state[1], 0.0, epsilon = 2e-3);
    }

    #[test]
    fn test_dstatepropagator_propagate_steps() {
        let mut prop = create_test_sho_propagator();

        // Propagate 10 steps
        prop.propagate_steps(10);

        // Should have taken multiple steps (trajectory should have entries)
        assert!(prop.trajectory().len() >= 10);

        // State should have changed
        let state = prop.current_state();
        assert!((state[0] - 1.0).abs() > 1e-6 || (state[1] - 0.0).abs() > 1e-6);
    }

    #[test]
    fn test_dstatepropagator_reset() {
        let mut prop = create_test_sho_propagator();
        let initial_epoch = prop.initial_epoch();
        let initial_state = prop.initial_state();

        // Propagate forward
        prop.propagate_to(initial_epoch + 10.0);

        // State should have changed
        assert!((prop.current_state()[0] - initial_state[0]).abs() > 1e-6);

        // Reset
        prop.reset();

        // Should be back at initial conditions
        assert_eq!(prop.current_epoch(), initial_epoch);
        assert_eq!(prop.current_state(), initial_state);
        assert_eq!(prop.trajectory().len(), 0); // Trajectory cleared
    }

    #[test]
    fn test_dstatepropagator_getters() {
        let mut prop = create_test_sho_propagator();
        let initial_epoch = prop.initial_epoch();

        assert_eq!(DStatePropagatorTrait::state_dim(&prop), 2);
        assert_eq!(prop.current_epoch(), initial_epoch);
        assert_eq!(prop.initial_epoch(), initial_epoch);
        assert_eq!(prop.initial_state(), DVector::from_vec(vec![1.0, 0.0]));

        // Propagate and check getters update
        prop.step_by(1.0);
        assert!(prop.current_epoch() > initial_epoch);
        assert_ne!(prop.current_state(), prop.initial_state());
    }

    // =============================================================================
    // DStateProvider Trait Tests
    // =============================================================================

    #[test]
    fn test_dstateprovider_state_at_current() {
        let mut prop = create_test_sho_propagator();
        prop.propagate_to(prop.initial_epoch() + 1.0);

        let current_epoch = prop.current_epoch();
        let state = prop.state(current_epoch).unwrap();

        assert_abs_diff_eq!(state[0], prop.current_state()[0], epsilon = 1e-12);
        assert_abs_diff_eq!(state[1], prop.current_state()[1], epsilon = 1e-12);
    }

    #[test]
    fn test_dstateprovider_state_interpolation() {
        let mut prop = create_test_sho_propagator();
        let initial_epoch = prop.initial_epoch();

        // Propagate to build trajectory
        prop.propagate_to(initial_epoch + 10.0);

        // Get state at intermediate epoch
        let mid_epoch = initial_epoch + 5.0;
        let state = prop.state(mid_epoch).unwrap();

        assert_eq!(state.len(), 2);
        // Should be interpolated from trajectory
        assert!(state[0].abs() <= 1.5); // Reasonable bounds for SHO
    }

    #[test]
    fn test_dstateprovider_state_out_of_bounds() {
        let prop = create_test_sho_propagator();
        let initial_epoch = prop.initial_epoch();

        // Try to get state before propagation started
        let result = prop.state(initial_epoch - 10.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_dstateprovider_state_dim() {
        let prop = create_test_sho_propagator();
        assert_eq!(DStateProvider::state_dim(&prop), 2);
    }

    #[test]
    fn test_dstateprovider_state_dimension_preservation() {
        let mut prop = create_test_sho_propagator();
        prop.propagate_to(prop.initial_epoch() + 5.0);

        let mid_epoch = prop.initial_epoch() + 2.5;
        let state = prop.state(mid_epoch).unwrap();

        assert_eq!(state.len(), 2); // Dimension preserved
    }

    // =============================================================================
    // DCovarianceProvider Trait Tests
    // =============================================================================

    #[test]
    fn test_dcovarianceprovider_no_covariance() {
        let prop = create_test_sho_propagator();
        let result = prop.covariance(prop.current_epoch());

        // Should error - covariance not enabled
        assert!(result.is_err());
    }

    #[test]
    fn test_dcovarianceprovider_with_initial_covariance() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![1.0, 0.0]);
        let dynamics = sho_dynamics(1.0);

        let initial_cov = DMatrix::from_diagonal(&DVector::from_vec(vec![0.01, 0.01]));

        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_stm = true;

        let mut prop = DNumericalPropagator::new(
            epoch,
            state,
            dynamics,
            config,
            None,
            None,
            Some(initial_cov.clone()),
        )
        .unwrap();

        // Propagate
        prop.propagate_to(epoch + 5.0);

        // Should be able to get covariance
        let cov = prop.covariance(prop.current_epoch()).unwrap();
        assert_eq!(cov.nrows(), 2);
        assert_eq!(cov.ncols(), 2);
    }

    #[test]
    fn test_dcovarianceprovider_positive_definiteness() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![1.0, 0.0]);
        let dynamics = sho_dynamics(1.0);

        let initial_cov = DMatrix::from_diagonal(&DVector::from_vec(vec![0.01, 0.01]));

        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_stm = true;

        let mut prop = DNumericalPropagator::new(
            epoch,
            state,
            dynamics,
            config,
            None,
            None,
            Some(initial_cov),
        )
        .unwrap();

        prop.propagate_to(epoch + 5.0);

        let cov = prop.covariance(prop.current_epoch()).unwrap();

        // Check diagonal elements are positive
        assert!(cov[(0, 0)] > 0.0);
        assert!(cov[(1, 1)] > 0.0);

        // Check symmetry
        assert_abs_diff_eq!(cov[(0, 1)], cov[(1, 0)], epsilon = 1e-12);
    }

    #[test]
    fn test_dcovarianceprovider_interpolation() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![1.0, 0.0]);
        let dynamics = sho_dynamics(1.0);

        let initial_cov = DMatrix::from_diagonal(&DVector::from_vec(vec![0.01, 0.01]));

        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_stm = true;

        let mut prop = DNumericalPropagator::new(
            epoch,
            state,
            dynamics,
            config,
            None,
            None,
            Some(initial_cov),
        )
        .unwrap();

        prop.propagate_to(epoch + 10.0);

        // Get covariance at intermediate time
        let mid_epoch = epoch + 5.0;
        let cov = prop.covariance(mid_epoch).unwrap();

        assert_eq!(cov.nrows(), 2);
        assert_eq!(cov.ncols(), 2);
    }

    #[test]
    fn test_dcovarianceprovider_out_of_bounds() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![1.0, 0.0]);
        let dynamics = sho_dynamics(1.0);

        let initial_cov = DMatrix::from_diagonal(&DVector::from_vec(vec![0.01, 0.01]));

        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_stm = true;

        let mut prop = DNumericalPropagator::new(
            epoch,
            state,
            dynamics,
            config,
            None,
            None,
            Some(initial_cov),
        )
        .unwrap();

        prop.propagate_to(epoch + 10.0);

        // Try to get covariance outside trajectory
        let result = prop.covariance(epoch + 20.0);
        assert!(result.is_err());
    }

    // =============================================================================
    // STM Propagation Tests
    // =============================================================================

    #[test]
    fn test_stm_identity_initialization() {
        let prop = create_test_sho_with_stm();

        let stm = prop.stm().expect("STM should be enabled");

        // Should be identity matrix initially
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(stm[(i, j)], expected, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_stm_propagation() {
        let mut prop = create_test_sho_with_stm();

        // Propagate forward
        prop.propagate_to(prop.initial_epoch() + 5.0);

        let stm = prop.stm().expect("STM should be available");

        // STM should have evolved (not identity anymore)
        assert!((stm[(0, 0)] - 1.0).abs() > 1e-6 || (stm[(1, 1)] - 1.0).abs() > 1e-6);

        // STM dimensions should match state dimensions
        assert_eq!(stm.nrows(), 2);
        assert_eq!(stm.ncols(), 2);
    }

    #[test]
    fn test_stm_storage_in_trajectory() {
        let mut prop = create_test_sho_with_stm();

        // Propagate to build trajectory
        prop.propagate_to(prop.initial_epoch() + 10.0);

        // Check STM is stored at various indices
        assert!(prop.trajectory().len() > 0);

        let stm_at_0 = prop.stm_at_idx(0);
        assert!(stm_at_0.is_some());

        let last_idx = prop.trajectory().len() - 1;
        let stm_at_last = prop.stm_at_idx(last_idx);
        assert!(stm_at_last.is_some());
    }

    #[test]
    fn test_stm_interpolation() {
        let mut prop = create_test_sho_with_stm();
        let initial_epoch = prop.initial_epoch();

        // Propagate to build trajectory
        prop.propagate_to(initial_epoch + 10.0);

        // Get STM at intermediate time
        let mid_epoch = initial_epoch + 5.0;
        let stm = prop.stm_at(mid_epoch);

        assert!(stm.is_some());
        let stm_matrix = stm.unwrap();
        assert_eq!(stm_matrix.nrows(), 2);
        assert_eq!(stm_matrix.ncols(), 2);
    }

    #[test]
    fn test_stm_reset() {
        let mut prop = create_test_sho_with_stm();

        // Propagate forward
        prop.propagate_to(prop.initial_epoch() + 5.0);

        let stm_after_prop = prop.stm().unwrap().clone();
        assert!((stm_after_prop[(0, 0)] - 1.0).abs() > 1e-6); // STM has evolved

        // Reset
        prop.reset();

        // STM should be identity again
        let stm = prop.stm().expect("STM should still be enabled");
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(stm[(i, j)], expected, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_stm_energy_conservation_check() {
        // For SHO, the dynamics are linear so we can verify STM behavior
        let mut prop = create_test_sho_with_stm();

        // Propagate one full period
        let period = 2.0 * PI;
        prop.propagate_to(prop.initial_epoch() + period);

        // State should return to initial (approximately)
        let state = prop.current_state();
        assert_abs_diff_eq!(state[0], 1.0, epsilon = 1e-3);
        assert_abs_diff_eq!(state[1], 0.0, epsilon = 2e-3);

        // STM determinant should be close to 1 (symplectic property for Hamiltonian systems)
        let stm = prop.stm().unwrap();
        let det = stm[(0, 0)] * stm[(1, 1)] - stm[(0, 1)] * stm[(1, 0)];
        assert_abs_diff_eq!(det, 1.0, epsilon = 1e-3);
    }

    // =============================================================================
    // Sensitivity Matrix Tests
    // =============================================================================

    #[test]
    fn test_sensitivity_zero_initialization() {
        let prop = create_test_damped_sho_with_sensitivity();

        let sens = prop.sensitivity().expect("Sensitivity should be enabled");

        // Should be zero matrix initially
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(sens[(i, j)], 0.0, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_sensitivity_propagation() {
        let mut prop = create_test_damped_sho_with_sensitivity();

        // Propagate forward
        prop.propagate_to(prop.initial_epoch() + 5.0);

        let sens = prop.sensitivity().expect("Sensitivity should be available");

        // Sensitivity should have evolved (not zero anymore)
        let mut has_nonzero = false;
        for i in 0..2 {
            for j in 0..2 {
                if sens[(i, j)].abs() > 1e-6 {
                    has_nonzero = true;
                    break;
                }
            }
        }
        assert!(has_nonzero, "Sensitivity should have non-zero elements");

        // Sensitivity dimensions: state_dim x param_dim
        assert_eq!(sens.nrows(), 2); // state dim
        assert_eq!(sens.ncols(), 2); // param dim (omega, zeta)
    }

    #[test]
    fn test_sensitivity_parameter_dependence() {
        // Create two propagators with different damping ratios
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![1.0, 0.0]);
        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_sensitivity = true;

        let params1 = DVector::from_vec(vec![1.0, 0.1]); // Low damping
        let params2 = DVector::from_vec(vec![1.0, 0.5]); // High damping

        let mut prop1 = DNumericalPropagator::new(
            epoch,
            state.clone(),
            damped_sho_dynamics(),
            config.clone(),
            Some(params1),
            None,
            None,
        )
        .unwrap();

        let mut prop2 = DNumericalPropagator::new(
            epoch,
            state,
            damped_sho_dynamics(),
            config,
            Some(params2),
            None,
            None,
        )
        .unwrap();

        // Propagate both
        prop1.propagate_to(epoch + 5.0);
        prop2.propagate_to(epoch + 5.0);

        // States should differ due to different damping
        let state1 = prop1.current_state();
        let state2 = prop2.current_state();

        assert!((state1[0] - state2[0]).abs() > 1e-6 || (state1[1] - state2[1]).abs() > 1e-6);
    }

    #[test]
    fn test_sensitivity_storage_in_trajectory() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![1.0, 0.0]);
        let params = DVector::from_vec(vec![1.0, 0.1]);
        let dynamics = damped_sho_dynamics();

        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_sensitivity = true;
        config.variational.store_sensitivity_history = true;

        let mut prop =
            DNumericalPropagator::new(epoch, state, dynamics, config, Some(params), None, None)
                .unwrap();

        // Propagate to build trajectory
        prop.propagate_to(epoch + 10.0);

        // Check sensitivity is stored
        assert!(prop.trajectory().len() > 0);

        let sens_at_0 = prop.sensitivity_at_idx(0);
        assert!(sens_at_0.is_some());
    }

    #[test]
    fn test_sensitivity_interpolation() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![1.0, 0.0]);
        let params = DVector::from_vec(vec![1.0, 0.1]);
        let dynamics = damped_sho_dynamics();

        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_sensitivity = true;
        config.variational.store_sensitivity_history = true;

        let mut prop =
            DNumericalPropagator::new(epoch, state, dynamics, config, Some(params), None, None)
                .unwrap();

        // Propagate to build trajectory
        prop.propagate_to(epoch + 10.0);

        // Get sensitivity at intermediate time
        let mid_epoch = epoch + 5.0;
        let sens = prop.sensitivity_at(mid_epoch);

        assert!(sens.is_some());
        let sens_matrix = sens.unwrap();
        assert_eq!(sens_matrix.nrows(), 2);
        assert_eq!(sens_matrix.ncols(), 2);
    }

    #[test]
    fn test_sensitivity_reset() {
        let mut prop = create_test_damped_sho_with_sensitivity();

        // Propagate forward
        prop.propagate_to(prop.initial_epoch() + 5.0);

        let sens_after_prop = prop.sensitivity().unwrap().clone();

        // Verify it has evolved
        let mut has_nonzero = false;
        for i in 0..2 {
            for j in 0..2 {
                if sens_after_prop[(i, j)].abs() > 1e-6 {
                    has_nonzero = true;
                    break;
                }
            }
        }
        assert!(has_nonzero);

        // Reset
        prop.reset();

        // Sensitivity should be zeros again
        let sens = prop
            .sensitivity()
            .expect("Sensitivity should still be enabled");
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(sens[(i, j)], 0.0, epsilon = 1e-12);
            }
        }
    }

    // =============================================================================
    // InterpolationConfig Trait Tests
    // =============================================================================

    #[test]
    fn test_interpolationconfig_builder() {
        use crate::math::interpolation::InterpolationMethod;

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![1.0, 0.0]);
        let dynamics = sho_dynamics(1.0);

        let prop = DNumericalPropagator::new(
            epoch,
            state,
            dynamics,
            NumericalPropagationConfig::default(),
            None,
            None,
            None,
        )
        .unwrap()
        .with_interpolation_method(InterpolationMethod::Linear);

        assert_eq!(prop.get_interpolation_method(), InterpolationMethod::Linear);
    }

    #[test]
    fn test_interpolationconfig_setter_getter() {
        use crate::math::interpolation::InterpolationMethod;

        let mut prop = create_test_sho_propagator();

        // Default should be Linear
        assert_eq!(prop.get_interpolation_method(), InterpolationMethod::Linear);

        // Set to Linear explicitly
        prop.set_interpolation_method(InterpolationMethod::Linear);
        assert_eq!(prop.get_interpolation_method(), InterpolationMethod::Linear);
    }

    #[test]
    fn test_interpolationconfig_persistence() {
        use crate::math::interpolation::InterpolationMethod;

        let mut prop = create_test_sho_propagator();
        prop.set_interpolation_method(InterpolationMethod::Linear);

        // Propagate
        prop.propagate_to(prop.initial_epoch() + 5.0);

        // Setting should persist
        assert_eq!(prop.get_interpolation_method(), InterpolationMethod::Linear);
    }

    // =============================================================================
    // CovarianceInterpolationConfig Trait Tests
    // =============================================================================

    #[test]
    fn test_covarianceinterpolationconfig_builder() {
        use crate::math::interpolation::CovarianceInterpolationMethod;

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![1.0, 0.0]);
        let dynamics = sho_dynamics(1.0);

        let prop = DNumericalPropagator::new(
            epoch,
            state,
            dynamics,
            NumericalPropagationConfig::default(),
            None,
            None,
            None,
        )
        .unwrap()
        .with_covariance_interpolation_method(CovarianceInterpolationMethod::MatrixSquareRoot);

        assert_eq!(
            prop.get_covariance_interpolation_method(),
            CovarianceInterpolationMethod::MatrixSquareRoot
        );
    }

    #[test]
    fn test_covarianceinterpolationconfig_setter_getter() {
        use crate::math::interpolation::CovarianceInterpolationMethod;

        let mut prop = create_test_sho_propagator();

        // Default should be TwoWasserstein
        assert_eq!(
            prop.get_covariance_interpolation_method(),
            CovarianceInterpolationMethod::TwoWasserstein
        );

        // Set to MatrixSquareRoot
        prop.set_covariance_interpolation_method(CovarianceInterpolationMethod::MatrixSquareRoot);
        assert_eq!(
            prop.get_covariance_interpolation_method(),
            CovarianceInterpolationMethod::MatrixSquareRoot
        );
    }

    #[test]
    fn test_covarianceinterpolationconfig_persistence() {
        use crate::math::interpolation::CovarianceInterpolationMethod;

        let mut prop = create_test_sho_propagator();
        prop.set_covariance_interpolation_method(CovarianceInterpolationMethod::MatrixSquareRoot);

        // Propagate
        prop.propagate_to(prop.initial_epoch() + 5.0);

        // Setting should persist
        assert_eq!(
            prop.get_covariance_interpolation_method(),
            CovarianceInterpolationMethod::MatrixSquareRoot
        );
    }

    // =============================================================================
    // Identifiable Trait Tests
    // =============================================================================

    #[test]
    fn test_identifiable_with_name() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![1.0, 0.0]);
        let dynamics = sho_dynamics(1.0);

        let prop = DNumericalPropagator::new(
            epoch,
            state,
            dynamics,
            NumericalPropagationConfig::default(),
            None,
            None,
            None,
        )
        .unwrap()
        .with_name("TestProp");

        assert_eq!(prop.get_name(), Some("TestProp"));
    }

    #[test]
    fn test_identifiable_set_name() {
        let mut prop = create_test_sho_propagator();

        assert_eq!(prop.get_name(), None);

        prop.set_name(Some("MyPropagator"));
        assert_eq!(prop.get_name(), Some("MyPropagator"));

        prop.set_name(None);
        assert_eq!(prop.get_name(), None);
    }

    #[test]
    fn test_identifiable_with_id() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![1.0, 0.0]);
        let dynamics = sho_dynamics(1.0);

        let prop = DNumericalPropagator::new(
            epoch,
            state,
            dynamics,
            NumericalPropagationConfig::default(),
            None,
            None,
            None,
        )
        .unwrap()
        .with_id(12345);

        assert_eq!(prop.get_id(), Some(12345));
    }

    #[test]
    fn test_identifiable_set_id() {
        let mut prop = create_test_sho_propagator();

        assert_eq!(prop.get_id(), None);

        prop.set_id(Some(999));
        assert_eq!(prop.get_id(), Some(999));

        prop.set_id(None);
        assert_eq!(prop.get_id(), None);
    }

    #[test]
    fn test_identifiable_with_uuid() {
        use uuid::Uuid;

        let test_uuid = Uuid::new_v4();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![1.0, 0.0]);
        let dynamics = sho_dynamics(1.0);

        let prop = DNumericalPropagator::new(
            epoch,
            state,
            dynamics,
            NumericalPropagationConfig::default(),
            None,
            None,
            None,
        )
        .unwrap()
        .with_uuid(test_uuid);

        assert_eq!(prop.get_uuid(), Some(test_uuid));
    }

    #[test]
    fn test_identifiable_with_new_uuid() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![1.0, 0.0]);
        let dynamics = sho_dynamics(1.0);

        let prop = DNumericalPropagator::new(
            epoch,
            state,
            dynamics,
            NumericalPropagationConfig::default(),
            None,
            None,
            None,
        )
        .unwrap()
        .with_new_uuid();

        assert!(prop.get_uuid().is_some());
    }

    #[test]
    fn test_identifiable_with_identity() {
        use uuid::Uuid;

        let test_uuid = Uuid::new_v4();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![1.0, 0.0]);
        let dynamics = sho_dynamics(1.0);

        let prop = DNumericalPropagator::new(
            epoch,
            state,
            dynamics,
            NumericalPropagationConfig::default(),
            None,
            None,
            None,
        )
        .unwrap()
        .with_identity(Some("TestProp"), Some(test_uuid), Some(123));

        assert_eq!(prop.get_name(), Some("TestProp"));
        assert_eq!(prop.get_uuid(), Some(test_uuid));
        assert_eq!(prop.get_id(), Some(123));
    }

    #[test]
    fn test_identifiable_persistence_through_propagation() {
        let mut prop = create_test_sho_propagator();
        prop.set_name(Some("TestProp"));
        prop.set_id(Some(42));

        // Propagate
        prop.propagate_to(prop.initial_epoch() + 5.0);

        // Identity should persist
        assert_eq!(prop.get_name(), Some("TestProp"));
        assert_eq!(prop.get_id(), Some(42));
    }

    // =============================================================================
    // Event Detection Tests
    // =============================================================================

    #[test]
    fn test_event_time_event() {
        let mut prop = create_test_sho_propagator();
        let initial_epoch = prop.initial_epoch();

        // Add time event at 5 seconds
        let event = DTimeEvent::new(initial_epoch + 5.0, "TimeEvent".to_string());
        prop.add_event_detector(Box::new(event));

        // Propagate past event
        prop.propagate_to(initial_epoch + 10.0);

        // Event should be detected
        let events = prop.event_log();
        assert!(!events.is_empty());

        let detected = events.iter().any(|e| e.name.contains("TimeEvent"));
        assert!(detected);
    }

    #[test]
    fn test_event_threshold_event() {
        let mut prop = create_test_sho_propagator();
        let initial_epoch = prop.initial_epoch();

        // Add value event: position crosses zero
        let value_fn =
            |_epoch: Epoch, state: &DVector<f64>, _params: Option<&DVector<f64>>| state[0];
        let event = DValueEvent::new(
            "PositionCrossing",
            value_fn,
            0.0,                        // target value
            EventDirection::Decreasing, // detect when crossing from positive to negative (first crossing for SHO)
        );
        prop.add_event_detector(Box::new(event));

        // Propagate through multiple crossings
        prop.propagate_to(initial_epoch + 10.0);

        // Should have detected crossings
        let events = prop.event_log();
        assert!(!events.is_empty(), "No events detected in event log");

        let detected = events.iter().any(|e| e.name.contains("PositionCrossing"));
        assert!(detected, "PositionCrossing event not found in event log");
    }

    #[test]
    fn test_event_callback_state_modification() {
        use crate::events::EventAction;

        let mut prop = create_test_sho_propagator();
        let initial_epoch = prop.initial_epoch();

        // Create callback that adds a small impulse when position crosses zero
        // This modifies velocity but keeps the system evolving (unlike zeroing velocity
        // which would cause repeated event detection at position ≈ 0)
        let callback: DEventCallback = Box::new(|_epoch, state, _params| {
            let mut new_state = state.clone();
            new_state[1] += 0.1; // Add small velocity impulse
            (Some(new_state), None, EventAction::Continue)
        });

        let value_fn =
            |_epoch: Epoch, state: &DVector<f64>, _params: Option<&DVector<f64>>| state[0];
        let event = DValueEvent::new("ImpulseAtZero", value_fn, 0.0, EventDirection::Decreasing)
            .with_callback(callback);

        prop.add_event_detector(Box::new(event));

        // Propagate
        prop.propagate_to(initial_epoch + 3.0);

        // Event should have been detected and velocity modified
        let events = prop.event_log();
        assert!(!events.is_empty(), "No events detected in event log");

        let detected = events.iter().any(|e| e.name.contains("ImpulseAtZero"));
        assert!(detected, "ImpulseAtZero event not found in event log");
    }

    #[test]
    fn test_event_terminal() {
        use crate::events::EventAction;

        let mut prop = create_test_sho_propagator();
        let initial_epoch = prop.initial_epoch();

        // Create terminal event at time 5
        let callback: DEventCallback =
            Box::new(|_epoch, _state, _params| (None, None, EventAction::Stop));

        let event =
            DTimeEvent::new(initial_epoch + 5.0, "Terminal".to_string()).with_callback(callback);

        prop.add_event_detector(Box::new(event));

        // Try to propagate to time 10
        prop.propagate_to(initial_epoch + 10.0);

        // Should have stopped at event (around time 5)
        assert!(prop.terminated());
        let time_diff: f64 = prop.current_epoch() - (initial_epoch + 5.0);
        assert!(time_diff.abs() < 1.0);
    }

    #[test]
    fn test_event_query_by_detector_index() {
        let mut prop = create_test_sho_propagator();
        let initial_epoch = prop.initial_epoch();

        // Add multiple detectors
        prop.add_event_detector(Box::new(DTimeEvent::new(
            initial_epoch + 2.0,
            "Event1".to_string(),
        )));
        prop.add_event_detector(Box::new(DTimeEvent::new(
            initial_epoch + 4.0,
            "Event2".to_string(),
        )));

        // Propagate
        prop.propagate_to(initial_epoch + 5.0);

        // Query events by detector index
        let events_0 = prop.events_by_detector_index(0);
        let events_1 = prop.events_by_detector_index(1);

        assert!(!events_0.is_empty());
        assert!(!events_1.is_empty());
    }

    #[test]
    fn test_event_query_in_range() {
        let mut prop = create_test_sho_propagator();
        let initial_epoch = prop.initial_epoch();

        // Add events at different times
        prop.add_event_detector(Box::new(DTimeEvent::new(
            initial_epoch + 2.0,
            "Early".to_string(),
        )));
        prop.add_event_detector(Box::new(DTimeEvent::new(
            initial_epoch + 8.0,
            "Late".to_string(),
        )));

        // Propagate
        prop.propagate_to(initial_epoch + 10.0);

        // Query events in range
        let events_early = prop.events_in_range(initial_epoch, initial_epoch + 5.0);
        let events_late = prop.events_in_range(initial_epoch + 5.0, initial_epoch + 10.0);

        // Early event should be in first range
        let has_early = events_early.iter().any(|e| e.name.contains("Early"));
        assert!(has_early);

        // Late event should be in second range
        let has_late = events_late.iter().any(|e| e.name.contains("Late"));
        assert!(has_late);
    }

    #[test]
    fn test_event_clear_and_reset_termination() {
        use crate::events::EventAction;

        let mut prop = create_test_sho_propagator();
        let initial_epoch = prop.initial_epoch();

        // Create terminal event
        let callback: DEventCallback =
            Box::new(|_epoch, _state, _params| (None, None, EventAction::Stop));

        let event =
            DTimeEvent::new(initial_epoch + 5.0, "Terminal".to_string()).with_callback(callback);

        prop.add_event_detector(Box::new(event));

        // Propagate and hit terminal
        prop.propagate_to(initial_epoch + 10.0);
        assert!(prop.terminated());

        // Clear events and reset termination
        prop.clear_events();
        prop.reset_termination();

        assert!(!prop.terminated());
        assert_eq!(prop.event_log().len(), 0);

        // Can continue propagating
        prop.propagate_to(initial_epoch + 15.0);
    }

    #[test]
    fn test_dnumericalpropagator_events_combined_filters() {
        let mut prop = create_test_sho_propagator();
        let initial_epoch = prop.initial_epoch();

        // Add multiple detectors at different times
        let event1 = DTimeEvent::new(initial_epoch + 2.0, "Early".to_string());
        let event2 = DTimeEvent::new(initial_epoch + 4.0, "Middle".to_string());
        let event3 = DTimeEvent::new(initial_epoch + 6.0, "Late".to_string());
        let event4 = DTimeEvent::new(initial_epoch + 8.0, "VeryLate".to_string());

        prop.add_event_detector(Box::new(event1));
        prop.add_event_detector(Box::new(event2));
        prop.add_event_detector(Box::new(event3));
        prop.add_event_detector(Box::new(event4));

        // Propagate
        prop.propagate_to(initial_epoch + 10.0);

        // Test combining filters:
        // 1. Get events by detector index
        let events_0 = prop.events_by_detector_index(0);
        let events_1 = prop.events_by_detector_index(1);

        assert_eq!(events_0.len(), 1);
        assert_eq!(events_1.len(), 1);

        // 2. Get events in time range
        let early_events = prop.events_in_range(initial_epoch, initial_epoch + 3.0);
        let middle_events = prop.events_in_range(initial_epoch + 3.0, initial_epoch + 5.0);
        let late_events = prop.events_in_range(initial_epoch + 5.0, initial_epoch + 10.0);

        assert_eq!(early_events.len(), 1);
        assert!(early_events[0].name.contains("Early"));

        assert_eq!(middle_events.len(), 1);
        assert!(middle_events[0].name.contains("Middle"));

        assert_eq!(late_events.len(), 2);
        assert!(late_events.iter().any(|e| e.name.contains("Late")));
        assert!(late_events.iter().any(|e| e.name.contains("VeryLate")));

        // 3. Get events by name
        let early_by_name = prop.events_by_name("Early");
        assert_eq!(early_by_name.len(), 1);

        // 4. Combined: filter by name AND verify it's in the correct time range
        let middle_by_name = prop.events_by_name("Middle");
        assert_eq!(middle_by_name.len(), 1);
        let middle_event = &middle_by_name[0];
        // Verify the event time is approximately initial_epoch + 4.0
        let time_diff: f64 = middle_event.window_open - (initial_epoch + 4.0);
        assert!(time_diff.abs() < 0.1);
    }

    // =============================================================================
    // Trajectory Storage Tests
    // =============================================================================

    #[test]
    fn test_trajectory_allsteps_mode() {
        let mut prop = create_test_sho_propagator();
        prop.set_trajectory_mode(TrajectoryMode::AllSteps);

        // Propagate
        prop.propagate_to(prop.initial_epoch() + 5.0);

        // Trajectory should have multiple entries
        assert!(prop.trajectory().len() > 1);
    }

    #[test]
    fn test_trajectory_disabled_mode() {
        let mut prop = create_test_sho_propagator();
        // Reset clears the trajectory (including initial state added at construction)
        prop.reset();
        prop.set_trajectory_mode(TrajectoryMode::Disabled);

        // Propagate
        prop.propagate_to(prop.initial_epoch() + 5.0);

        // Trajectory should be empty since disabled mode prevents storage
        assert_eq!(prop.trajectory().len(), 0);
    }

    #[test]
    fn test_trajectory_eviction_max_size() {
        let mut prop = create_test_sho_propagator();

        // Set max size to 10 entries
        prop.set_eviction_policy_max_size(10).unwrap();

        // Propagate to generate many steps
        prop.propagate_to(prop.initial_epoch() + 50.0);

        // Trajectory should be limited
        assert!(prop.trajectory().len() <= 10);
    }

    #[test]
    fn test_trajectory_stm_sensitivity_storage() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![1.0, 0.0]);
        let params = DVector::from_vec(vec![1.0, 0.1]);
        let dynamics = damped_sho_dynamics();

        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_stm = true;
        config.variational.store_stm_history = true;
        config.variational.enable_sensitivity = true;
        config.variational.store_sensitivity_history = true;

        let mut prop =
            DNumericalPropagator::new(epoch, state, dynamics, config, Some(params), None, None)
                .unwrap();

        // Propagate
        prop.propagate_to(epoch + 5.0);

        // Both STM and sensitivity should be stored
        assert!(prop.stm_at_idx(0).is_some());
        assert!(prop.sensitivity_at_idx(0).is_some());
    }

    // =============================================================================
    // Corner Cases & Error Handling Tests
    // =============================================================================

    #[test]
    fn test_corner_case_zero_parameters() {
        // SHO doesn't need parameters - should work fine
        let mut prop = create_test_sho_propagator();

        assert_eq!(prop.current_params().len(), 0);

        // Should propagate successfully
        prop.propagate_to(prop.initial_epoch() + 5.0);

        let state = prop.current_state();
        assert_eq!(state.len(), 2);
    }

    #[test]
    fn test_corner_case_single_parameter() {
        // Create dynamics with single parameter
        let single_param_dynamics: DStateDynamics = Box::new(|_t, x, p| {
            let k = p.map(|params| params[0]).unwrap_or(1.0);
            DVector::from_vec(vec![x[1], -k * k * x[0]])
        });

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![1.0, 0.0]);
        let params = DVector::from_vec(vec![1.5]); // Single parameter

        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_sensitivity = true;

        let mut prop = DNumericalPropagator::new(
            epoch,
            state,
            single_param_dynamics,
            config,
            Some(params),
            None,
            None,
        )
        .unwrap();

        // Propagate
        prop.propagate_to(epoch + 5.0);

        // Sensitivity should be 2x1 (state_dim x param_dim)
        let sens = prop.sensitivity().unwrap();
        assert_eq!(sens.nrows(), 2);
        assert_eq!(sens.ncols(), 1);
    }

    #[test]
    fn test_corner_case_sensitivity_without_params() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![1.0, 0.0]);
        let dynamics = sho_dynamics(1.0);

        let mut config = NumericalPropagationConfig::default();
        config.variational.enable_sensitivity = true;

        // Should error - sensitivity needs parameters
        let result = DNumericalPropagator::new(epoch, state, dynamics, config, None, None, None);

        assert!(result.is_err());
    }

    #[test]
    fn test_corner_case_very_small_timestep() {
        let mut prop = create_test_sho_propagator();
        let initial_epoch = prop.initial_epoch();

        // Propagate by tiny amount
        prop.propagate_to(initial_epoch + 1e-6);

        // Should still work
        assert!(prop.current_epoch() > initial_epoch);
    }
}
