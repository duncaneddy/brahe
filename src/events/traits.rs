/*!
 * Event detection traits and types for numerical propagation
 *
 * This module provides the core infrastructure for detecting events during
 * numerical propagation, including zero-crossing detection with bisection
 * refinement for accurate event timing.
 */

use crate::time::Epoch;
use nalgebra::{DVector, SVector};

/// Direction of event detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventDirection {
    /// Detect increasing zero-crossings (negative to positive)
    Increasing,
    /// Detect decreasing zero-crossings (positive to negative)
    Decreasing,
    /// Detect any zero-crossing
    Any,
}

/// Edge type for binary event detection
///
/// Used with [`BinaryEvent`] to specify which boolean transition to detect.
/// Internally converted to [`EventDirection`] for zero-crossing detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeType {
    /// Detect rising edge (false → true transition)
    ///
    /// Becomes `EventDirection::Increasing` internally
    RisingEdge,
    /// Detect falling edge (true → false transition)
    ///
    /// Becomes `EventDirection::Decreasing` internally
    FallingEdge,
    /// Detect any edge (any boolean transition)
    ///
    /// Becomes `EventDirection::Any` internally
    AnyEdge,
}

/// Type of event
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventType {
    /// Instantaneous event (zero-crossing at a point in time)
    Instantaneous,
    /// Period event (maintains condition over an interval)
    Period,
}

/// Action to take when event is detected
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventAction {
    /// Stop propagation
    Stop,
    /// Continue propagation
    Continue,
}

/// Information about a detected event (static-sized)
#[derive(Debug, Clone)]
pub struct SDetectedEvent<const S: usize> {
    /// Window open time (entry for periods, event time for instantaneous)
    pub window_open: Epoch,

    /// Window close time (exit for periods, same as window_open for instantaneous)
    pub window_close: Epoch,

    /// State at window open
    pub entry_state: SVector<f64, S>,

    /// State at window close (same as entry_state for instantaneous)
    pub exit_state: SVector<f64, S>,

    /// Event function value at detection
    pub value: f64,

    /// Event detector name
    pub name: String,

    /// Action to take
    pub action: EventAction,

    /// Event type
    pub event_type: EventType,

    /// Detector index (position in event_detectors Vec, 0-based)
    pub detector_index: usize,
}

impl<const S: usize> SDetectedEvent<S> {
    /// Alias for window_open
    pub fn t_start(&self) -> Epoch {
        self.window_open
    }

    /// Alias for window_close
    pub fn t_end(&self) -> Epoch {
        self.window_close
    }

    /// Alias for window_open
    pub fn start_time(&self) -> Epoch {
        self.window_open
    }

    /// Alias for window_close
    pub fn end_time(&self) -> Epoch {
        self.window_close
    }
}

/// Information about a detected event (dynamic-sized)
#[derive(Debug, Clone)]
pub struct DDetectedEvent {
    /// Window open time (entry for periods, event time for instantaneous)
    pub window_open: Epoch,

    /// Window close time (exit for periods, same as window_open for instantaneous)
    pub window_close: Epoch,

    /// State at window open
    pub entry_state: DVector<f64>,

    /// State at window close (same as entry_state for instantaneous)
    pub exit_state: DVector<f64>,

    /// Event function value at detection
    pub value: f64,

    /// Event detector name
    pub name: String,

    /// Action to take
    pub action: EventAction,

    /// Event type
    pub event_type: EventType,

    /// Detector index (position in event_detectors Vec, 0-based)
    pub detector_index: usize,
}

impl DDetectedEvent {
    /// Alias for window_open
    pub fn t_start(&self) -> Epoch {
        self.window_open
    }

    /// Alias for window_close
    pub fn t_end(&self) -> Epoch {
        self.window_close
    }

    /// Alias for window_open
    pub fn start_time(&self) -> Epoch {
        self.window_open
    }

    /// Alias for window_close
    pub fn end_time(&self) -> Epoch {
        self.window_close
    }
}

/// Event callback function signature (static-sized)
///
/// Takes current time, state, and parameters. Returns optional updates:
/// - Updated state (for impulsive maneuvers)
/// - Updated parameters (for parameter modifications)
/// - Action to take after event
///
/// # Callback vs Terminal Flag Priority
/// Callbacks can override the terminal flag set by `.is_terminal()`:
/// - If callback returns `EventAction::Stop`, propagation stops regardless of terminal flag
/// - If callback returns `EventAction::Continue`, propagation continues even if terminal flag is set
/// - This allows dynamic decisions based on event state
pub type SEventCallback<const S: usize, const P: usize> = Box<
    dyn Fn(
            Epoch,
            &SVector<f64, S>,
            Option<&SVector<f64, P>>,
        ) -> (
            Option<SVector<f64, S>>,
            Option<SVector<f64, P>>,
            EventAction,
        ) + Send
        + Sync,
>;

/// Event callback function signature (dynamic-sized)
///
/// See [`SEventCallback`] for details on callback vs terminal flag priority.
pub type DEventCallback = Box<
    dyn Fn(
            Epoch,
            &DVector<f64>,
            Option<&DVector<f64>>,
        ) -> (Option<DVector<f64>>, Option<DVector<f64>>, EventAction)
        + Send
        + Sync,
>;

/// Trait for event detection during numerical propagation (static-sized)
///
/// Event detectors monitor a value and detect when it crosses a target value.
/// The propagator uses bisection search to refine the event time to within
/// specified tolerances.
///
/// # Event Detection
/// Events are detected when `evaluate() - target_value()` crosses zero. The detector
/// should return the raw monitored value from `evaluate()`, and the target
/// value from `target_value()`. The detection algorithm computes the zero-crossing.
///
/// # Examples
/// - Altitude event: `evaluate()` returns altitude(y), `target_value()` returns target altitude
/// - Time event: `evaluate()` returns (t - target_time).abs(), `target_value()` returns 0.0
/// - Apogee: `evaluate()` returns velocity · position, `target_value()` returns 0.0
pub trait SEventDetector<const S: usize, const P: usize>: Send + Sync + std::any::Any {
    /// Evaluate the monitored value
    ///
    /// Returns the raw value being monitored (NOT a zero-crossing).
    /// The detection algorithm will compute `evaluate() - target_value()` to
    /// find zero-crossings.
    ///
    /// # Arguments
    /// * `t` - Current time
    /// * `state` - Current state vector
    /// * `params` - Optional parameter vector
    fn evaluate(&self, t: Epoch, state: &SVector<f64, S>, params: Option<&SVector<f64, P>>) -> f64;

    /// Get target value for comparison
    ///
    /// Event occurs when `evaluate() - target_value()` crosses zero according
    /// to the specified `direction()`.
    fn target_value(&self) -> f64;

    /// Get event name for identification
    fn name(&self) -> &str;

    /// Get event type
    fn event_type(&self) -> EventType {
        EventType::Instantaneous
    }

    /// Get detection direction
    fn direction(&self) -> EventDirection {
        EventDirection::Any
    }

    /// Get time tolerance for bisection (seconds)
    fn time_tolerance(&self) -> f64 {
        1e-3
    }

    /// Get value tolerance for zero-crossing
    fn value_tolerance(&self) -> f64 {
        1e-6
    }

    /// Get step reduction factor for bisection search
    ///
    /// When a zero-crossing is detected, the search bracket is narrowed and
    /// the new step size is set to this factor times the bracket width.
    /// Smaller values (e.g., 0.1) result in more conservative steps but
    /// potentially more iterations. Larger values (e.g., 0.5) are more
    /// aggressive but may overshoot.
    ///
    /// Default: 0.2 (1/5th of bracket width)
    fn step_reduction_factor(&self) -> f64 {
        0.2
    }

    /// Get optional callback to execute when event is detected
    fn callback(&self) -> Option<&SEventCallback<S, P>> {
        None
    }

    /// Get action to take when event is detected
    ///
    /// This is the default action if no callback is present, or serves as
    /// the initial action before the callback is executed. Callbacks can
    /// override this by returning a different `EventAction`.
    fn action(&self) -> EventAction {
        EventAction::Continue
    }
}

/// Trait for event detection with dynamic sizing
pub trait DEventDetector: Send + Sync + std::any::Any {
    /// Evaluate the monitored value
    ///
    /// Returns the raw value being monitored (NOT a zero-crossing).
    /// The detection algorithm will compute `evaluate() - target_value()` to
    /// find zero-crossings.
    fn evaluate(&self, t: Epoch, state: &DVector<f64>, params: Option<&DVector<f64>>) -> f64;

    /// Get target value for comparison
    ///
    /// Event occurs when `evaluate() - target_value()` crosses zero according
    /// to the specified `direction()`.
    fn target_value(&self) -> f64;

    /// Get event name
    fn name(&self) -> &str;

    /// Get event type
    fn event_type(&self) -> EventType {
        EventType::Instantaneous
    }

    /// Get detection direction
    fn direction(&self) -> EventDirection {
        EventDirection::Any
    }

    /// Get time tolerance (seconds)
    fn time_tolerance(&self) -> f64 {
        1e-3
    }

    /// Get value tolerance
    fn value_tolerance(&self) -> f64 {
        1e-6
    }

    /// Get step reduction factor for bisection search
    ///
    /// When a zero-crossing is detected, the search bracket is narrowed and
    /// the new step size is set to this factor times the bracket width.
    /// Default: 0.2 (1/5th of bracket width)
    fn step_reduction_factor(&self) -> f64 {
        0.2
    }

    /// Get optional callback
    fn callback(&self) -> Option<&DEventCallback> {
        None
    }

    /// Get action to take when event is detected
    ///
    /// This is the default action if no callback is present, or serves as
    /// the initial action before the callback is executed. Callbacks can
    /// override this by returning a different `EventAction`.
    fn action(&self) -> EventAction {
        EventAction::Continue
    }
}
