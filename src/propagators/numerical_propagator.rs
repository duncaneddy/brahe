/*!
 * Generic numerical propagator with event detection and control inputs
 *
 * This module provides a flexible numerical propagator that supports:
 * - User-defined dynamics functions
 * - Event detection with callbacks
 * - State and parameter mutations via event callbacks
 * - Configurable trajectory storage
 * - Time management (Epoch ↔ relative time conversion)
 */

use nalgebra::{DVector, SVector, Vector6};

use crate::events::{
    DDetectedEvent, DEventDetector, EventAction, SDetectedEvent, SEventDetector, dscan_for_event,
    sscan_for_event,
};
use crate::integrators::traits::{DIntegrator, SIntegrator};
use crate::math::interpolation::{interpolate_linear_dvector, interpolate_linear_svector};
use crate::propagators::traits::{DStatePropagator, SStatePropagator};
use crate::time::Epoch;
use crate::traits::Trajectory;
use crate::trajectories::{DTrajectory, STrajectory};
use crate::utils::errors::BraheError;

/// Trajectory storage mode
///
/// Controls when states are added to the trajectory during propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrajectoryMode {
    /// Store only at output intervals and event times
    OutputStepsOnly,
    /// Store every internal integrator step plus events
    AllSteps,
    /// No trajectory storage (only event log maintained)
    Disabled,
}

impl TrajectoryMode {
    /// Whether to store states at regular output steps
    pub fn store_output_steps(&self) -> bool {
        matches!(
            self,
            TrajectoryMode::OutputStepsOnly | TrajectoryMode::AllSteps
        )
    }

    /// Whether to store states at every internal integration step
    pub fn store_internal_steps(&self) -> bool {
        matches!(self, TrajectoryMode::AllSteps)
    }

    /// Whether to store states at event times
    pub fn store_event_points(&self) -> bool {
        !matches!(self, TrajectoryMode::Disabled)
    }
}

/// Result of processing events in an integration step
#[derive(Debug)]
enum EventProcessingResult<const S: usize> {
    /// No events detected, accept the integration step
    NoEvents,
    /// Event callback modified state/params, restart integration from event
    Restart {
        epoch: Epoch,
        state: SVector<f64, S>,
    },
}

/// Generic numerical propagator with event detection (static-sized)
///
/// Propagates user-defined dynamics with support for:
/// - Event detection and callbacks
/// - State and parameter mutations
/// - STM and sensitivity matrix propagation
/// - Configurable trajectory storage
///
/// # Type Parameters
/// - `I`: Integrator type (must implement SIntegrator<S, P>)
/// - `S`: State vector dimension
/// - `P`: Parameter vector dimension
///
/// # Time Management
/// The propagator manages conversion between:
/// - **Integrators**: Use `f64` relative time (seconds since start)
/// - **Event detectors**: Use `Epoch` absolute time
///
/// All conversions are handled internally via `to_epoch()` and `to_relative()`.
///
/// See also: [`DNumericalPropagator`] for dynamic-sized version
pub struct SNumericalPropagator<I, const S: usize, const P: usize>
where
    I: SIntegrator<S, P>,
{
    // ===== Time Management =====
    /// Initial absolute time (Epoch)
    epoch_initial: Epoch,
    /// Current relative time (seconds since start)
    t_rel: f64,

    // ===== Integration State =====
    /// Numerical integrator (works in relative time)
    integrator: I,
    /// Initial state vector (for reset)
    x_initial: SVector<f64, S>,
    /// Current state vector
    x_curr: SVector<f64, S>,
    /// Mutable parameter vector
    params: SVector<f64, P>,

    // ===== Event Handling =====
    /// Event detectors (work in absolute time)
    event_detectors: Vec<Box<dyn SEventDetector<S, P>>>,
    /// Chronological event log
    event_log: Vec<SDetectedEvent<S>>,

    // ===== Trajectory Storage =====
    /// State history storage
    trajectory: STrajectory<S>,
    /// Trajectory storage mode
    trajectory_mode: TrajectoryMode,

    // ===== Configuration =====
    /// Default step size for fixed-step propagation (seconds)
    step_size: f64,
    /// Optional fixed output step for dense output mode
    output_step: Option<f64>,

    // ===== Termination State =====
    /// Flag indicating if propagation was stopped by a terminal event
    terminated: bool,
}

impl<I, const S: usize, const P: usize> SNumericalPropagator<I, S, P>
where
    I: SIntegrator<S, P>,
{
    // =========================================================================
    // Construction
    // =========================================================================

    /// Create a new numerical propagator
    ///
    /// # Arguments
    /// * `epoch_initial` - Initial absolute time
    /// * `x_initial` - Initial state vector
    /// * `params` - Parameter vector (mutable during propagation)
    /// * `integrator` - Numerical integrator
    ///
    /// # Returns
    /// New propagator starting at t=0 (relative time)
    pub fn new(
        epoch_initial: Epoch,
        x_initial: SVector<f64, S>,
        params: SVector<f64, P>,
        integrator: I,
    ) -> Self {
        Self {
            epoch_initial,
            t_rel: 0.0,
            integrator,
            x_initial,
            x_curr: x_initial,
            params,
            event_detectors: Vec::new(),
            event_log: Vec::new(),
            trajectory: STrajectory::new(),
            trajectory_mode: TrajectoryMode::OutputStepsOnly,
            step_size: 10.0, // Default 10 seconds
            output_step: None,
            terminated: false,
        }
    }

    // =========================================================================
    // Time Conversion
    // =========================================================================

    /// Convert relative time to absolute Epoch
    ///
    /// # Arguments
    /// * `t_rel` - Relative time in seconds since start
    ///
    /// # Returns
    /// Absolute epoch
    #[inline]
    fn to_epoch(&self, t_rel: f64) -> Epoch {
        self.epoch_initial + t_rel
    }

    /// Convert absolute Epoch to relative time
    ///
    /// # Arguments
    /// * `epoch` - Absolute epoch
    ///
    /// # Returns
    /// Relative time in seconds since start
    #[inline]
    fn to_relative(&self, epoch: Epoch) -> f64 {
        epoch - self.epoch_initial
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

    /// Set default step size for fixed-step propagation
    pub fn set_step_size(&mut self, step_size: f64) {
        self.step_size = step_size;
    }

    /// Get default step size
    pub fn step_size(&self) -> f64 {
        self.step_size
    }

    /// Enable dense output mode with fixed output step
    ///
    /// When enabled, adaptive integrators will output states at regular
    /// intervals using dense output, rather than at natural step boundaries.
    pub fn set_output_step(&mut self, step: f64) {
        self.output_step = Some(step);
    }

    /// Disable dense output mode
    pub fn disable_output_step(&mut self) {
        self.output_step = None;
    }

    // =========================================================================
    // Event Handling
    // =========================================================================

    /// Add an event detector
    pub fn add_event_detector(&mut self, detector: Box<dyn SEventDetector<S, P>>) {
        self.event_detectors.push(detector);
    }

    /// Get all detected events
    pub fn event_log(&self) -> &[SDetectedEvent<S>] {
        &self.event_log
    }

    /// Get events by name (substring match)
    pub fn events_by_name(&self, name: &str) -> Vec<&SDetectedEvent<S>> {
        self.event_log
            .iter()
            .filter(|e| e.name.contains(name))
            .collect()
    }

    /// Get latest event
    pub fn latest_event(&self) -> Option<&SDetectedEvent<S>> {
        self.event_log.last()
    }

    /// Get events in time range
    pub fn events_in_range(&self, start: Epoch, end: Epoch) -> Vec<&SDetectedEvent<S>> {
        self.event_log
            .iter()
            .filter(|e| e.window_open >= start && e.window_open <= end)
            .collect()
    }

    /// Clear event log
    pub fn clear_events(&mut self) {
        self.event_log.clear();
    }

    // =========================================================================
    // State Access
    // =========================================================================

    /// Get current absolute epoch
    pub fn current_epoch(&self) -> Epoch {
        self.to_epoch(self.t_rel)
    }

    /// Get current state
    pub fn current_state(&self) -> &SVector<f64, S> {
        &self.x_curr
    }

    /// Get current parameters
    pub fn current_params(&self) -> &SVector<f64, P> {
        &self.params
    }

    /// Get initial epoch
    pub fn initial_epoch(&self) -> Epoch {
        self.epoch_initial
    }

    /// Get initial state
    pub fn initial_state(&self) -> &SVector<f64, S> {
        &self.x_initial
    }

    /// Get trajectory
    pub fn trajectory(&self) -> &STrajectory<S> {
        &self.trajectory
    }

    /// Check if propagation was terminated by a terminal event
    pub fn terminated(&self) -> bool {
        self.terminated
    }

    /// Reset termination flag
    ///
    /// Allows propagation to continue after a terminal event.
    /// Note: This does not change the state - the propagator will continue
    /// from wherever it stopped.
    pub fn reset_termination(&mut self) {
        self.terminated = false;
    }

    /// Reset propagator to initial conditions
    ///
    /// Resets:
    /// - Current state to initial state
    /// - Current time to initial epoch
    /// - Termination flag to false
    /// - Clears event log
    /// - Clears trajectory
    pub fn reset(&mut self) {
        self.t_rel = 0.0;
        self.x_curr = self.x_initial;
        self.terminated = false;
        self.event_log.clear();
        self.trajectory.clear();
    }

    /// Set eviction policy to keep a maximum number of states in trajectory
    pub fn set_eviction_policy_max_size(&mut self, max_size: usize) -> Result<(), BraheError> {
        self.trajectory.set_eviction_policy_max_size(max_size)
    }

    /// Set eviction policy to keep states within a maximum age in trajectory
    pub fn set_eviction_policy_max_age(&mut self, max_age: f64) -> Result<(), BraheError> {
        self.trajectory.set_eviction_policy_max_age(max_age)
    }

    // =========================================================================
    // Event Detection and Processing
    // =========================================================================

    /// Scan all event detectors for events in the interval [epoch_prev, epoch_new]
    ///
    /// Returns a list of (detector_index, detected_event) pairs.
    fn scan_all_events(
        &self,
        epoch_prev: Epoch,
        epoch_new: Epoch,
        x_prev: &SVector<f64, S>,
        x_new: &SVector<f64, S>,
    ) -> Vec<(usize, SDetectedEvent<S>)> {
        let mut events = Vec::new();

        // State function that bridges Epoch → f64 for event detection
        // Uses linear interpolation between integration steps
        // TODO: Could use dense output from integrator if available
        let state_fn = |epoch: Epoch| -> SVector<f64, S> {
            let t_rel = self.to_relative(epoch);
            let alpha = (t_rel - self.to_relative(epoch_prev))
                / (self.to_relative(epoch_new) - self.to_relative(epoch_prev));
            interpolate_linear_svector(*x_prev, *x_new, alpha)
        };

        // Scan each detector
        for (idx, detector) in self.event_detectors.iter().enumerate() {
            if let Some(event) = sscan_for_event(
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

    /// Process detected events with smart sequential handling
    ///
    /// Algorithm:
    /// 1. Sort events chronologically
    /// 2. Find first event with callback
    /// 3. Process all no-callback events before it (just log)
    /// 4. Process first callback event (apply state/param updates)
    /// 5. If callback exists, restart integration from that time
    /// 6. Discard events after callback (invalid with new state/params)
    fn process_events_smart(
        &mut self,
        detected_events: Vec<(usize, SDetectedEvent<S>)>,
    ) -> EventProcessingResult<S> {
        if detected_events.is_empty() {
            return EventProcessingResult::NoEvents;
        }

        // 1. Sort chronologically
        let mut sorted_events = detected_events;
        sorted_events.sort_by(|(_, a), (_, b)| a.window_open.partial_cmp(&b.window_open).unwrap());

        // 2. Find first event with callback
        let first_callback_idx = sorted_events
            .iter()
            .position(|(det_idx, _)| self.event_detectors[*det_idx].callback().is_some());

        match first_callback_idx {
            None => {
                // 3a. No callbacks - process all events (just log them)
                for (_, event) in sorted_events {
                    // Add to trajectory at exact event time if configured
                    if self.trajectory_mode.store_event_points() {
                        self.trajectory.add(event.window_open, event.entry_state);
                    }

                    // Log event
                    self.event_log.push(event);
                }
                EventProcessingResult::NoEvents
            }

            Some(callback_idx) => {
                // 3b. Process all no-callback events before the callback event
                for (_, event) in sorted_events.iter().take(callback_idx) {
                    if self.trajectory_mode.store_event_points() {
                        self.trajectory.add(event.window_open, event.entry_state);
                    }
                    self.event_log.push(event.clone());
                }

                // 4. Process the first callback event
                let (det_idx, callback_event) = &sorted_events[callback_idx];
                let detector = &self.event_detectors[*det_idx];

                // Add pre-callback state to trajectory
                if self.trajectory_mode.store_event_points() {
                    self.trajectory
                        .add(callback_event.window_open, callback_event.entry_state);
                }

                // Log pre-callback event
                self.event_log.push(callback_event.clone());

                // Execute callback
                if let Some(callback) = detector.callback() {
                    let (new_state, new_params, action) = callback(
                        callback_event.window_open,
                        &callback_event.entry_state,
                        Some(&self.params),
                    );

                    // Apply state mutation
                    let x_after = if let Some(x_new) = new_state {
                        // Add post-callback state to trajectory (same time, different state)
                        if self.trajectory_mode.store_event_points() {
                            self.trajectory.add(callback_event.window_open, x_new);
                        }
                        x_new
                    } else {
                        callback_event.entry_state
                    };

                    // Apply parameter mutation
                    if let Some(p_new) = new_params {
                        self.params = p_new;
                    }

                    // Check terminal
                    if action == EventAction::Stop {
                        self.terminated = true;
                        // Still return Restart to update state, but terminated flag is set
                        return EventProcessingResult::Restart {
                            epoch: callback_event.window_open,
                            state: x_after,
                        };
                    }

                    // Restart integration from event time with new state/params
                    return EventProcessingResult::Restart {
                        epoch: callback_event.window_open,
                        state: x_after,
                    };
                }

                unreachable!("Callback event must have callback");
            }
        }
    }

    // =========================================================================
    // Propagation
    // =========================================================================

    /// Single integration step with event handling
    ///
    /// This is the core propagation method that:
    /// 1. Takes an integration step
    /// 2. Scans for events
    /// 3. Processes events sequentially
    /// 4. Restarts if needed
    /// 5. Sets `terminated` flag if a terminal event occurs
    ///
    /// After calling this method, check `terminated()` to see if propagation
    /// was stopped by a terminal event.
    pub fn step(&mut self) {
        // Don't step if already terminated
        if self.terminated {
            return;
        }

        loop {
            // 1. Take integration step in relative time
            let t_prev_rel = self.t_rel;
            let x_prev = self.x_curr;

            // Use integrator's adaptive step method
            let dt = self.step_size;
            let result = self.integrator.step(t_prev_rel, x_prev, Some(dt));
            let x_new = result.state;
            let t_new_rel = t_prev_rel + result.dt_used;

            // Update step size for next iteration based on integrator's recommendation
            self.step_size = result.dt_next;

            // 2. Convert to Epoch for event detection
            let epoch_prev = self.to_epoch(t_prev_rel);
            let epoch_new = self.to_epoch(t_new_rel);

            // 3. Scan all event detectors
            let detected_events = self.scan_all_events(epoch_prev, epoch_new, &x_prev, &x_new);

            // 4. Smart event processing
            match self.process_events_smart(detected_events) {
                EventProcessingResult::NoEvents => {
                    // Accept step
                    if self.trajectory_mode.store_output_steps() {
                        self.trajectory.add(epoch_new, x_new);
                    }

                    self.t_rel = t_new_rel;
                    self.x_curr = x_new;

                    return; // Step complete
                }

                EventProcessingResult::Restart { epoch, state } => {
                    // Event callback modified state/params
                    // Restart integration from event time
                    self.t_rel = self.to_relative(epoch);
                    self.x_curr = state;

                    // If terminated flag was set during event processing, stop here
                    if self.terminated {
                        return;
                    }

                    continue; // Loop will take new step from event time
                }
            }
        }
    }

    /// Propagate to a target epoch
    ///
    /// Automatically stops if a terminal event occurs before reaching the target.
    /// Check `terminated()` after calling to see if propagation was stopped early.
    ///
    /// # Arguments
    /// * `target_epoch` - Absolute target time
    pub fn propagate_to(&mut self, target_epoch: Epoch) {
        let target_rel = self.to_relative(target_epoch);

        while self.t_rel < target_rel && !self.terminated {
            self.step();
        }
    }

    /// Propagate for a fixed number of steps
    ///
    /// Automatically stops if a terminal event occurs before completing all steps.
    /// Check `terminated()` after calling to see if propagation was stopped early.
    ///
    /// # Arguments
    /// * `num_steps` - Number of steps to take
    pub fn propagate_steps(&mut self, num_steps: usize) {
        for _ in 0..num_steps {
            if self.terminated {
                break;
            }
            self.step();
        }
    }
}

// =============================================================================
// SStatePropagator Trait Implementation (6D states only)
// =============================================================================

impl<I, const P: usize> SStatePropagator for SNumericalPropagator<I, 6, P>
where
    I: SIntegrator<6, P>,
{
    fn step_by(&mut self, step_size: f64) {
        // Temporarily set step size, take step, restore original
        let original_step = self.step_size;
        self.step_size = step_size;
        self.step();
        // Don't restore if terminated - final step size is from integrator
        if !self.terminated {
            self.step_size = original_step;
        }
    }

    fn current_epoch(&self) -> Epoch {
        SNumericalPropagator::current_epoch(self)
    }

    fn current_state(&self) -> Vector6<f64> {
        self.x_curr
    }

    fn initial_epoch(&self) -> Epoch {
        self.epoch_initial
    }

    fn initial_state(&self) -> Vector6<f64> {
        self.x_initial
    }

    fn step_size(&self) -> f64 {
        self.step_size
    }

    fn set_step_size(&mut self, step_size: f64) {
        self.step_size = step_size;
    }

    fn reset(&mut self) {
        SNumericalPropagator::reset(self);
    }

    fn set_eviction_policy_max_size(&mut self, max_size: usize) -> Result<(), BraheError> {
        SNumericalPropagator::set_eviction_policy_max_size(self, max_size)
    }

    fn set_eviction_policy_max_age(&mut self, max_age: f64) -> Result<(), BraheError> {
        SNumericalPropagator::set_eviction_policy_max_age(self, max_age)
    }

    // Override propagation methods to handle termination
    fn step_past(&mut self, target_epoch: Epoch) {
        while self.current_epoch() < target_epoch && !self.terminated {
            self.step();
        }
    }

    fn propagate_steps(&mut self, num_steps: usize) {
        for _ in 0..num_steps {
            if self.terminated {
                break;
            }
            self.step();
        }
    }

    fn propagate_to(&mut self, target_epoch: Epoch) {
        let target_rel = self.to_relative(target_epoch);

        while self.t_rel < target_rel && !self.terminated {
            self.step();
        }
    }
}

// =============================================================================
// Dynamic-Sized Numerical Propagator
// =============================================================================

/// Result of processing events in an integration step (dynamic-sized)
#[derive(Debug)]
enum DEventProcessingResult {
    /// No events detected, accept the integration step
    NoEvents,
    /// Event callback modified state/params, restart integration from event
    Restart { epoch: Epoch, state: DVector<f64> },
}

/// Generic numerical propagator with event detection (dynamic-sized)
///
/// Dynamic-sized version of [`SNumericalPropagator`]. Uses heap-allocated
/// vectors for state and parameters with runtime-determined dimensions.
///
/// # Type Parameters
/// - `I`: Integrator type (must implement DIntegrator)
///
/// # Time Management
/// The propagator manages conversion between:
/// - **Integrators**: Use `f64` relative time (seconds since start)
/// - **Event detectors**: Use `Epoch` absolute time
///
/// See also: [`SNumericalPropagator`] for static-sized version
pub struct DNumericalPropagator<I>
where
    I: DIntegrator,
{
    // ===== Time Management =====
    /// Initial absolute time (Epoch)
    epoch_initial: Epoch,
    /// Current relative time (seconds since start)
    t_rel: f64,

    // ===== Integration State =====
    /// Numerical integrator (works in relative time)
    integrator: I,
    /// Initial state vector (for reset)
    x_initial: DVector<f64>,
    /// Current state vector
    x_curr: DVector<f64>,
    /// Mutable parameter vector
    params: DVector<f64>,
    /// State dimension
    state_dim: usize,
    /// Parameter dimension
    param_dim: usize,

    // ===== Event Handling =====
    /// Event detectors (work in absolute time)
    event_detectors: Vec<Box<dyn DEventDetector>>,
    /// Chronological event log
    event_log: Vec<DDetectedEvent>,

    // ===== Trajectory Storage =====
    /// State history storage
    trajectory: DTrajectory,
    /// Trajectory storage mode
    trajectory_mode: TrajectoryMode,

    // ===== Configuration =====
    /// Default step size for fixed-step propagation (seconds)
    step_size: f64,
    /// Optional fixed output step for dense output mode
    output_step: Option<f64>,

    // ===== Termination State =====
    /// Flag indicating if propagation was stopped by a terminal event
    terminated: bool,
}

impl<I> DNumericalPropagator<I>
where
    I: DIntegrator,
{
    // =========================================================================
    // Construction
    // =========================================================================

    /// Create a new numerical propagator
    ///
    /// # Arguments
    /// * `epoch_initial` - Initial absolute time
    /// * `x_initial` - Initial state vector
    /// * `params` - Parameter vector (mutable during propagation)
    /// * `integrator` - Numerical integrator
    ///
    /// # Returns
    /// New propagator starting at t=0 (relative time)
    pub fn new(
        epoch_initial: Epoch,
        x_initial: DVector<f64>,
        params: DVector<f64>,
        integrator: I,
    ) -> Self {
        let state_dim = x_initial.len();
        let param_dim = params.len();

        Self {
            epoch_initial,
            t_rel: 0.0,
            integrator,
            x_initial: x_initial.clone(),
            x_curr: x_initial,
            params,
            state_dim,
            param_dim,
            event_detectors: Vec::new(),
            event_log: Vec::new(),
            trajectory: DTrajectory::new(state_dim),
            trajectory_mode: TrajectoryMode::OutputStepsOnly,
            step_size: 10.0,
            output_step: None,
            terminated: false,
        }
    }

    // =========================================================================
    // Time Conversion
    // =========================================================================

    /// Convert relative time to absolute Epoch
    #[inline]
    fn to_epoch(&self, t_rel: f64) -> Epoch {
        self.epoch_initial + t_rel
    }

    /// Convert absolute Epoch to relative time
    #[inline]
    fn to_relative(&self, epoch: Epoch) -> f64 {
        epoch - self.epoch_initial
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

    /// Set default step size for fixed-step propagation
    pub fn set_step_size(&mut self, step_size: f64) {
        self.step_size = step_size;
    }

    /// Get default step size
    pub fn step_size(&self) -> f64 {
        self.step_size
    }

    /// Enable dense output mode with fixed output step
    pub fn set_output_step(&mut self, step: f64) {
        self.output_step = Some(step);
    }

    /// Disable dense output mode
    pub fn disable_output_step(&mut self) {
        self.output_step = None;
    }

    // =========================================================================
    // Event Handling
    // =========================================================================

    /// Add an event detector
    pub fn add_event_detector(&mut self, detector: Box<dyn DEventDetector>) {
        self.event_detectors.push(detector);
    }

    /// Get all detected events
    pub fn event_log(&self) -> &[DDetectedEvent] {
        &self.event_log
    }

    /// Get events by name (substring match)
    pub fn events_by_name(&self, name: &str) -> Vec<&DDetectedEvent> {
        self.event_log
            .iter()
            .filter(|e| e.name.contains(name))
            .collect()
    }

    /// Get latest event
    pub fn latest_event(&self) -> Option<&DDetectedEvent> {
        self.event_log.last()
    }

    /// Get events in time range
    pub fn events_in_range(&self, start: Epoch, end: Epoch) -> Vec<&DDetectedEvent> {
        self.event_log
            .iter()
            .filter(|e| e.window_open >= start && e.window_open <= end)
            .collect()
    }

    /// Clear event log
    pub fn clear_events(&mut self) {
        self.event_log.clear();
    }

    // =========================================================================
    // State Access
    // =========================================================================

    /// Get current absolute epoch
    pub fn current_epoch(&self) -> Epoch {
        self.to_epoch(self.t_rel)
    }

    /// Get current state
    pub fn current_state(&self) -> &DVector<f64> {
        &self.x_curr
    }

    /// Get current parameters
    pub fn current_params(&self) -> &DVector<f64> {
        &self.params
    }

    /// Get initial epoch
    pub fn initial_epoch(&self) -> Epoch {
        self.epoch_initial
    }

    /// Get initial state
    pub fn initial_state(&self) -> &DVector<f64> {
        &self.x_initial
    }

    /// Get trajectory
    pub fn trajectory(&self) -> &DTrajectory {
        &self.trajectory
    }

    /// Get state dimension
    pub fn state_dim(&self) -> usize {
        self.state_dim
    }

    /// Get parameter dimension
    pub fn param_dim(&self) -> usize {
        self.param_dim
    }

    /// Check if propagation was terminated by a terminal event
    pub fn terminated(&self) -> bool {
        self.terminated
    }

    /// Reset termination flag
    ///
    /// Allows propagation to continue after a terminal event.
    /// Note: This does not change the state - the propagator will continue
    /// from wherever it stopped.
    pub fn reset_termination(&mut self) {
        self.terminated = false;
    }

    /// Reset propagator to initial conditions
    ///
    /// Resets:
    /// - Current state to initial state
    /// - Current time to initial epoch
    /// - Termination flag to false
    /// - Clears event log
    /// - Clears trajectory
    pub fn reset(&mut self) {
        self.t_rel = 0.0;
        self.x_curr = self.x_initial.clone();
        self.terminated = false;
        self.event_log.clear();
        self.trajectory.clear();
    }

    /// Set eviction policy to keep a maximum number of states in trajectory
    pub fn set_eviction_policy_max_size(&mut self, max_size: usize) -> Result<(), BraheError> {
        self.trajectory.set_eviction_policy_max_size(max_size)
    }

    /// Set eviction policy to keep states within a maximum age in trajectory
    pub fn set_eviction_policy_max_age(&mut self, max_age: f64) -> Result<(), BraheError> {
        self.trajectory.set_eviction_policy_max_age(max_age)
    }

    // =========================================================================
    // Event Detection and Processing
    // =========================================================================

    /// Scan all event detectors for events in the interval [epoch_prev, epoch_new]
    fn scan_all_events(
        &self,
        epoch_prev: Epoch,
        epoch_new: Epoch,
        x_prev: &DVector<f64>,
        x_new: &DVector<f64>,
    ) -> Vec<(usize, DDetectedEvent)> {
        let mut events = Vec::new();

        // State function that bridges Epoch → f64 for event detection
        // Uses linear interpolation between integration steps
        // TODO: Could use dense output from integrator if available
        let state_fn = |epoch: Epoch| -> DVector<f64> {
            let t_rel = self.to_relative(epoch);
            let alpha = (t_rel - self.to_relative(epoch_prev))
                / (self.to_relative(epoch_new) - self.to_relative(epoch_prev));
            interpolate_linear_dvector(x_prev, x_new, alpha)
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

    /// Process detected events with smart sequential handling
    fn process_events_smart(
        &mut self,
        detected_events: Vec<(usize, DDetectedEvent)>,
    ) -> DEventProcessingResult {
        if detected_events.is_empty() {
            return DEventProcessingResult::NoEvents;
        }

        // 1. Sort chronologically
        let mut sorted_events = detected_events;
        sorted_events.sort_by(|(_, a), (_, b)| a.window_open.partial_cmp(&b.window_open).unwrap());

        // 2. Find first event with callback
        let first_callback_idx = sorted_events
            .iter()
            .position(|(det_idx, _)| self.event_detectors[*det_idx].callback().is_some());

        match first_callback_idx {
            None => {
                // 3a. No callbacks - process all events (just log them)
                for (_, event) in sorted_events {
                    if self.trajectory_mode.store_event_points() {
                        self.trajectory
                            .add(event.window_open, event.entry_state.clone());
                    }
                    self.event_log.push(event);
                }
                DEventProcessingResult::NoEvents
            }

            Some(callback_idx) => {
                // 3b. Process all no-callback events before the callback event
                for (_, event) in sorted_events.iter().take(callback_idx) {
                    if self.trajectory_mode.store_event_points() {
                        self.trajectory
                            .add(event.window_open, event.entry_state.clone());
                    }
                    self.event_log.push(event.clone());
                }

                // 4. Process the first callback event
                let (det_idx, callback_event) = &sorted_events[callback_idx];
                let detector = &self.event_detectors[*det_idx];

                // Add pre-callback state to trajectory
                if self.trajectory_mode.store_event_points() {
                    self.trajectory.add(
                        callback_event.window_open,
                        callback_event.entry_state.clone(),
                    );
                }

                // Log pre-callback event
                self.event_log.push(callback_event.clone());

                // Execute callback
                if let Some(callback) = detector.callback() {
                    let (new_state, new_params, action) = callback(
                        callback_event.window_open,
                        &callback_event.entry_state,
                        Some(&self.params),
                    );

                    // Apply state mutation
                    let x_after = if let Some(x_new) = new_state {
                        // Add post-callback state to trajectory
                        if self.trajectory_mode.store_event_points() {
                            self.trajectory
                                .add(callback_event.window_open, x_new.clone());
                        }
                        x_new
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
                        // Still return Restart to update state, but terminated flag is set
                        return DEventProcessingResult::Restart {
                            epoch: callback_event.window_open,
                            state: x_after,
                        };
                    }

                    // Restart integration from event time with new state/params
                    return DEventProcessingResult::Restart {
                        epoch: callback_event.window_open,
                        state: x_after,
                    };
                }

                unreachable!("Callback event must have callback");
            }
        }
    }

    // =========================================================================
    // Propagation
    // =========================================================================

    /// Single integration step with event handling
    ///
    /// This is the core propagation method that:
    /// 1. Takes an integration step
    /// 2. Scans for events
    /// 3. Processes events sequentially
    /// 4. Restarts if needed
    /// 5. Sets `terminated` flag if a terminal event occurs
    ///
    /// After calling this method, check `terminated()` to see if propagation
    /// was stopped by a terminal event.
    pub fn step(&mut self) {
        // Don't step if already terminated
        if self.terminated {
            return;
        }

        loop {
            // 1. Take integration step in relative time
            let t_prev_rel = self.t_rel;
            let x_prev = self.x_curr.clone();

            // Use integrator's adaptive step method
            let dt = self.step_size;
            let result = self.integrator.step(t_prev_rel, x_prev.clone(), Some(dt));
            let x_new = result.state;
            let t_new_rel = t_prev_rel + result.dt_used;

            // Update step size for next iteration based on integrator's recommendation
            self.step_size = result.dt_next;

            // 2. Convert to Epoch for event detection
            let epoch_prev = self.to_epoch(t_prev_rel);
            let epoch_new = self.to_epoch(t_new_rel);

            // 3. Scan all event detectors
            let detected_events = self.scan_all_events(epoch_prev, epoch_new, &x_prev, &x_new);

            // 4. Smart event processing
            match self.process_events_smart(detected_events) {
                DEventProcessingResult::NoEvents => {
                    // Accept step
                    if self.trajectory_mode.store_output_steps() {
                        self.trajectory.add(epoch_new, x_new.clone());
                    }

                    self.t_rel = t_new_rel;
                    self.x_curr = x_new;

                    return; // Step complete
                }

                DEventProcessingResult::Restart { epoch, state } => {
                    // Event callback modified state/params
                    self.t_rel = self.to_relative(epoch);
                    self.x_curr = state;

                    // If terminated flag was set during event processing, stop here
                    if self.terminated {
                        return;
                    }

                    continue; // Loop will take new step from event time
                }
            }
        }
    }

    /// Propagate to a target epoch
    ///
    /// Automatically stops if a terminal event occurs before reaching the target.
    /// Check `terminated()` after calling to see if propagation was stopped early.
    ///
    /// # Arguments
    /// * `target_epoch` - Absolute target time
    pub fn propagate_to(&mut self, target_epoch: Epoch) {
        let target_rel = self.to_relative(target_epoch);

        while self.t_rel < target_rel && !self.terminated {
            self.step();
        }
    }

    /// Propagate for a fixed number of steps
    ///
    /// Automatically stops if a terminal event occurs before completing all steps.
    /// Check `terminated()` after calling to see if propagation was stopped early.
    ///
    /// # Arguments
    /// * `num_steps` - Number of steps to take
    pub fn propagate_steps(&mut self, num_steps: usize) {
        for _ in 0..num_steps {
            if self.terminated {
                break;
            }
            self.step();
        }
    }
}

// =============================================================================
// DStatePropagator Trait Implementation
// =============================================================================

impl<I> DStatePropagator for DNumericalPropagator<I>
where
    I: DIntegrator,
{
    fn step_by(&mut self, step_size: f64) {
        // Temporarily set step size, take step, restore original
        let original_step = self.step_size;
        self.step_size = step_size;
        self.step();
        // Don't restore if terminated - final step size is from integrator
        if !self.terminated {
            self.step_size = original_step;
        }
    }

    fn current_epoch(&self) -> Epoch {
        DNumericalPropagator::current_epoch(self)
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
        self.step_size
    }

    fn set_step_size(&mut self, step_size: f64) {
        self.step_size = step_size;
    }

    fn reset(&mut self) {
        DNumericalPropagator::reset(self);
    }

    fn set_eviction_policy_max_size(&mut self, max_size: usize) -> Result<(), BraheError> {
        DNumericalPropagator::set_eviction_policy_max_size(self, max_size)
    }

    fn set_eviction_policy_max_age(&mut self, max_age: f64) -> Result<(), BraheError> {
        DNumericalPropagator::set_eviction_policy_max_age(self, max_age)
    }

    // Override propagation methods to handle termination
    fn step_past(&mut self, target_epoch: Epoch) {
        while self.current_epoch() < target_epoch && !self.terminated {
            self.step();
        }
    }

    fn propagate_steps(&mut self, num_steps: usize) {
        for _ in 0..num_steps {
            if self.terminated {
                break;
            }
            self.step();
        }
    }

    fn propagate_to(&mut self, target_epoch: Epoch) {
        let target_rel = self.to_relative(target_epoch);

        while self.t_rel < target_rel && !self.terminated {
            self.step();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trajectory_mode() {
        assert!(TrajectoryMode::OutputStepsOnly.store_output_steps());
        assert!(!TrajectoryMode::OutputStepsOnly.store_internal_steps());
        assert!(TrajectoryMode::OutputStepsOnly.store_event_points());

        assert!(TrajectoryMode::AllSteps.store_output_steps());
        assert!(TrajectoryMode::AllSteps.store_internal_steps());
        assert!(TrajectoryMode::AllSteps.store_event_points());

        assert!(!TrajectoryMode::Disabled.store_output_steps());
        assert!(!TrajectoryMode::Disabled.store_internal_steps());
        assert!(!TrajectoryMode::Disabled.store_event_points());
    }

    #[test]
    fn test_time_conversion() {
        // Create a simple propagator to test time conversion
        // We'll use a placeholder since we don't have a real integrator yet
        // This test will be expanded when we integrate with actual integrators
    }
}
