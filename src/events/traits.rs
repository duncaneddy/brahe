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
    Window,
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

/// Convert static-sized detected event to dynamic-sized
///
/// This enables SGPPropagator (which uses SDetectedEvent<6>) to return
/// events compatible with Python bindings (which use DDetectedEvent).
impl<const S: usize> From<SDetectedEvent<S>> for DDetectedEvent {
    fn from(event: SDetectedEvent<S>) -> Self {
        DDetectedEvent {
            window_open: event.window_open,
            window_close: event.window_close,
            entry_state: DVector::from_iterator(S, event.entry_state.iter().cloned()),
            exit_state: DVector::from_iterator(S, event.exit_state.iter().cloned()),
            value: event.value,
            name: event.name,
            action: event.action,
            event_type: event.event_type,
            detector_index: event.detector_index,
        }
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
/// Callbacks can override the terminal flag set by `.set_terminal()`:
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

    /// Mark this event as processed (callback has been executed)
    ///
    /// Called by the propagator after successfully executing the event's callback.
    /// After being marked processed, the event will not trigger again during
    /// the current propagation run.
    ///
    /// Default implementation is a no-op for backward compatibility with
    /// event detectors that should trigger repeatedly (e.g., altitude crossings).
    fn mark_processed(&self) {
        // Default no-op - allows repeating events
    }

    /// Check if this event has been processed
    ///
    /// Returns true if the event's callback has been executed and the event
    /// should not trigger again during the current propagation run.
    fn is_processed(&self) -> bool {
        false // Default: allows re-triggering
    }

    /// Reset the processed state
    ///
    /// Called when the event should be able to trigger again, such as after
    /// a propagator reset.
    fn reset_processed(&self) {
        // Default no-op
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

    /// Mark this event as processed (callback has been executed)
    ///
    /// Called by the propagator after successfully executing the event's callback.
    /// After being marked processed, the event will not trigger again during
    /// the current propagation run.
    ///
    /// Default implementation is a no-op for backward compatibility with
    /// event detectors that should trigger repeatedly (e.g., altitude crossings).
    fn mark_processed(&self) {
        // Default no-op - allows repeating events
    }

    /// Check if this event has been processed
    ///
    /// Returns true if the event's callback has been executed and the event
    /// should not trigger again during the current propagation run.
    fn is_processed(&self) -> bool {
        false // Default: allows re-triggering
    }

    /// Reset the processed state
    ///
    /// Called when the event should be able to trigger again, such as after
    /// a propagator reset.
    fn reset_processed(&self) {
        // Default no-op
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::time::TimeSystem;
    use nalgebra::{DVector, Vector6};

    // =========================================================================
    // Enum Tests
    // =========================================================================

    #[test]
    fn test_EventDirection_variants() {
        // Test that all variants exist and are distinct
        let increasing = EventDirection::Increasing;
        let decreasing = EventDirection::Decreasing;
        let any = EventDirection::Any;

        assert_eq!(increasing, EventDirection::Increasing);
        assert_eq!(decreasing, EventDirection::Decreasing);
        assert_eq!(any, EventDirection::Any);

        assert_ne!(increasing, decreasing);
        assert_ne!(increasing, any);
        assert_ne!(decreasing, any);
    }

    #[test]
    fn test_EventDirection_debug() {
        assert_eq!(format!("{:?}", EventDirection::Increasing), "Increasing");
        assert_eq!(format!("{:?}", EventDirection::Decreasing), "Decreasing");
        assert_eq!(format!("{:?}", EventDirection::Any), "Any");
    }

    #[test]
    fn test_EventDirection_clone() {
        let dir = EventDirection::Increasing;
        let cloned = Clone::clone(&dir); // Explicit trait call to test Clone impl
        assert_eq!(dir, cloned);
    }

    #[test]
    fn test_EventDirection_copy() {
        let dir = EventDirection::Decreasing;
        let copied: EventDirection = dir; // Copy semantics
        assert_eq!(dir, copied);
    }

    #[test]
    fn test_EdgeType_variants() {
        let rising = EdgeType::RisingEdge;
        let falling = EdgeType::FallingEdge;
        let any = EdgeType::AnyEdge;

        assert_eq!(rising, EdgeType::RisingEdge);
        assert_eq!(falling, EdgeType::FallingEdge);
        assert_eq!(any, EdgeType::AnyEdge);

        assert_ne!(rising, falling);
        assert_ne!(rising, any);
        assert_ne!(falling, any);
    }

    #[test]
    fn test_EdgeType_debug() {
        assert_eq!(format!("{:?}", EdgeType::RisingEdge), "RisingEdge");
        assert_eq!(format!("{:?}", EdgeType::FallingEdge), "FallingEdge");
        assert_eq!(format!("{:?}", EdgeType::AnyEdge), "AnyEdge");
    }

    #[test]
    fn test_EdgeType_clone() {
        let edge = EdgeType::FallingEdge;
        let cloned = Clone::clone(&edge); // Explicit trait call to test Clone impl
        assert_eq!(edge, cloned);
    }

    #[test]
    fn test_EdgeType_copy() {
        let edge = EdgeType::AnyEdge;
        let copied: EdgeType = edge;
        assert_eq!(edge, copied);
    }

    #[test]
    fn test_EventType_variants() {
        let instant = EventType::Instantaneous;
        let window = EventType::Window;

        assert_eq!(instant, EventType::Instantaneous);
        assert_eq!(window, EventType::Window);
        assert_ne!(instant, window);
    }

    #[test]
    fn test_EventType_debug() {
        assert_eq!(format!("{:?}", EventType::Instantaneous), "Instantaneous");
        assert_eq!(format!("{:?}", EventType::Window), "Window");
    }

    #[test]
    fn test_EventType_clone() {
        let event_type = EventType::Window;
        let cloned = Clone::clone(&event_type); // Explicit trait call to test Clone impl
        assert_eq!(event_type, cloned);
    }

    #[test]
    fn test_EventType_copy() {
        let event_type = EventType::Instantaneous;
        let copied: EventType = event_type;
        assert_eq!(event_type, copied);
    }

    #[test]
    fn test_EventAction_variants() {
        let stop = EventAction::Stop;
        let cont = EventAction::Continue;

        assert_eq!(stop, EventAction::Stop);
        assert_eq!(cont, EventAction::Continue);
        assert_ne!(stop, cont);
    }

    #[test]
    fn test_EventAction_debug() {
        assert_eq!(format!("{:?}", EventAction::Stop), "Stop");
        assert_eq!(format!("{:?}", EventAction::Continue), "Continue");
    }

    #[test]
    fn test_EventAction_clone() {
        let action = EventAction::Stop;
        let cloned = Clone::clone(&action); // Explicit trait call to test Clone impl
        assert_eq!(action, cloned);
    }

    #[test]
    fn test_EventAction_copy() {
        let action = EventAction::Continue;
        let copied: EventAction = action;
        assert_eq!(action, copied);
    }

    // =========================================================================
    // SDetectedEvent Tests
    // =========================================================================

    fn create_test_sdetected_event() -> SDetectedEvent<6> {
        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);

        SDetectedEvent {
            window_open: epoch,
            window_close: epoch + 100.0,
            entry_state: state,
            exit_state: state,
            value: 500e3,
            name: "Test Event".to_string(),
            action: EventAction::Continue,
            event_type: EventType::Instantaneous,
            detector_index: 0,
        }
    }

    #[test]
    fn test_SDetectedEvent_t_start() {
        let event = create_test_sdetected_event();
        assert_eq!(event.t_start(), event.window_open);
    }

    #[test]
    fn test_SDetectedEvent_t_end() {
        let event = create_test_sdetected_event();
        assert_eq!(event.t_end(), event.window_close);
    }

    #[test]
    fn test_SDetectedEvent_start_time() {
        let event = create_test_sdetected_event();
        assert_eq!(event.start_time(), event.window_open);
    }

    #[test]
    fn test_SDetectedEvent_end_time() {
        let event = create_test_sdetected_event();
        assert_eq!(event.end_time(), event.window_close);
    }

    #[test]
    fn test_SDetectedEvent_fields() {
        let event = create_test_sdetected_event();

        assert_eq!(event.name, "Test Event");
        assert_eq!(event.action, EventAction::Continue);
        assert_eq!(event.event_type, EventType::Instantaneous);
        assert_eq!(event.detector_index, 0);
        assert_eq!(event.value, 500e3);
    }

    #[test]
    fn test_SDetectedEvent_clone() {
        let event = create_test_sdetected_event();
        let cloned = event.clone();

        assert_eq!(event.window_open, cloned.window_open);
        assert_eq!(event.window_close, cloned.window_close);
        assert_eq!(event.name, cloned.name);
    }

    #[test]
    fn test_SDetectedEvent_debug() {
        let event = create_test_sdetected_event();
        let debug_str = format!("{:?}", event);
        assert!(debug_str.contains("SDetectedEvent"));
        assert!(debug_str.contains("Test Event"));
    }

    // =========================================================================
    // DDetectedEvent Tests
    // =========================================================================

    fn create_test_ddetected_event() -> DDetectedEvent {
        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);

        DDetectedEvent {
            window_open: epoch,
            window_close: epoch + 100.0,
            entry_state: state.clone(),
            exit_state: state,
            value: 500e3,
            name: "Test DEvent".to_string(),
            action: EventAction::Stop,
            event_type: EventType::Window,
            detector_index: 1,
        }
    }

    #[test]
    fn test_DDetectedEvent_t_start() {
        let event = create_test_ddetected_event();
        assert_eq!(event.t_start(), event.window_open);
    }

    #[test]
    fn test_DDetectedEvent_t_end() {
        let event = create_test_ddetected_event();
        assert_eq!(event.t_end(), event.window_close);
    }

    #[test]
    fn test_DDetectedEvent_start_time() {
        let event = create_test_ddetected_event();
        assert_eq!(event.start_time(), event.window_open);
    }

    #[test]
    fn test_DDetectedEvent_end_time() {
        let event = create_test_ddetected_event();
        assert_eq!(event.end_time(), event.window_close);
    }

    #[test]
    fn test_DDetectedEvent_fields() {
        let event = create_test_ddetected_event();

        assert_eq!(event.name, "Test DEvent");
        assert_eq!(event.action, EventAction::Stop);
        assert_eq!(event.event_type, EventType::Window);
        assert_eq!(event.detector_index, 1);
        assert_eq!(event.value, 500e3);
    }

    #[test]
    fn test_DDetectedEvent_clone() {
        let event = create_test_ddetected_event();
        let cloned = event.clone();

        assert_eq!(event.window_open, cloned.window_open);
        assert_eq!(event.window_close, cloned.window_close);
        assert_eq!(event.name, cloned.name);
    }

    #[test]
    fn test_DDetectedEvent_debug() {
        let event = create_test_ddetected_event();
        let debug_str = format!("{:?}", event);
        assert!(debug_str.contains("DDetectedEvent"));
        assert!(debug_str.contains("Test DEvent"));
    }

    // =========================================================================
    // SEventDetector Default Trait Implementation Tests
    // =========================================================================

    /// Minimal implementation for testing default trait methods
    struct MinimalSEventDetector {
        name: String,
    }

    impl SEventDetector<6, 0> for MinimalSEventDetector {
        fn evaluate(
            &self,
            _t: Epoch,
            _state: &SVector<f64, 6>,
            _params: Option<&SVector<f64, 0>>,
        ) -> f64 {
            0.0
        }

        fn target_value(&self) -> f64 {
            0.0
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    #[test]
    fn test_SEventDetector_default_event_type() {
        let detector = MinimalSEventDetector {
            name: "Test".to_string(),
        };
        assert_eq!(detector.event_type(), EventType::Instantaneous);
    }

    #[test]
    fn test_SEventDetector_default_direction() {
        let detector = MinimalSEventDetector {
            name: "Test".to_string(),
        };
        assert_eq!(detector.direction(), EventDirection::Any);
    }

    #[test]
    fn test_SEventDetector_default_time_tolerance() {
        let detector = MinimalSEventDetector {
            name: "Test".to_string(),
        };
        assert_eq!(detector.time_tolerance(), 1e-3);
    }

    #[test]
    fn test_SEventDetector_default_value_tolerance() {
        let detector = MinimalSEventDetector {
            name: "Test".to_string(),
        };
        assert_eq!(detector.value_tolerance(), 1e-6);
    }

    #[test]
    fn test_SEventDetector_default_step_reduction_factor() {
        let detector = MinimalSEventDetector {
            name: "Test".to_string(),
        };
        assert_eq!(detector.step_reduction_factor(), 0.2);
    }

    #[test]
    fn test_SEventDetector_default_callback() {
        let detector = MinimalSEventDetector {
            name: "Test".to_string(),
        };
        assert!(detector.callback().is_none());
    }

    #[test]
    fn test_SEventDetector_default_action() {
        let detector = MinimalSEventDetector {
            name: "Test".to_string(),
        };
        assert_eq!(detector.action(), EventAction::Continue);
    }

    #[test]
    fn test_SEventDetector_default_mark_processed() {
        let detector = MinimalSEventDetector {
            name: "Test".to_string(),
        };
        // Should not panic - is a no-op
        detector.mark_processed();
    }

    #[test]
    fn test_SEventDetector_default_is_processed() {
        let detector = MinimalSEventDetector {
            name: "Test".to_string(),
        };
        // Default returns false (allows re-triggering)
        assert!(!detector.is_processed());
    }

    #[test]
    fn test_SEventDetector_default_reset_processed() {
        let detector = MinimalSEventDetector {
            name: "Test".to_string(),
        };
        // Should not panic - is a no-op
        detector.reset_processed();
    }

    // =========================================================================
    // DEventDetector Default Trait Implementation Tests
    // =========================================================================

    /// Minimal implementation for testing default trait methods
    struct MinimalDEventDetector {
        name: String,
    }

    impl DEventDetector for MinimalDEventDetector {
        fn evaluate(
            &self,
            _t: Epoch,
            _state: &DVector<f64>,
            _params: Option<&DVector<f64>>,
        ) -> f64 {
            0.0
        }

        fn target_value(&self) -> f64 {
            0.0
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    #[test]
    fn test_DEventDetector_default_event_type() {
        let detector = MinimalDEventDetector {
            name: "Test".to_string(),
        };
        assert_eq!(detector.event_type(), EventType::Instantaneous);
    }

    #[test]
    fn test_DEventDetector_default_direction() {
        let detector = MinimalDEventDetector {
            name: "Test".to_string(),
        };
        assert_eq!(detector.direction(), EventDirection::Any);
    }

    #[test]
    fn test_DEventDetector_default_time_tolerance() {
        let detector = MinimalDEventDetector {
            name: "Test".to_string(),
        };
        assert_eq!(detector.time_tolerance(), 1e-3);
    }

    #[test]
    fn test_DEventDetector_default_value_tolerance() {
        let detector = MinimalDEventDetector {
            name: "Test".to_string(),
        };
        assert_eq!(detector.value_tolerance(), 1e-6);
    }

    #[test]
    fn test_DEventDetector_default_step_reduction_factor() {
        let detector = MinimalDEventDetector {
            name: "Test".to_string(),
        };
        assert_eq!(detector.step_reduction_factor(), 0.2);
    }

    #[test]
    fn test_DEventDetector_default_callback() {
        let detector = MinimalDEventDetector {
            name: "Test".to_string(),
        };
        assert!(detector.callback().is_none());
    }

    #[test]
    fn test_DEventDetector_default_action() {
        let detector = MinimalDEventDetector {
            name: "Test".to_string(),
        };
        assert_eq!(detector.action(), EventAction::Continue);
    }

    #[test]
    fn test_DEventDetector_default_mark_processed() {
        let detector = MinimalDEventDetector {
            name: "Test".to_string(),
        };
        // Should not panic - is a no-op
        detector.mark_processed();
    }

    #[test]
    fn test_DEventDetector_default_is_processed() {
        let detector = MinimalDEventDetector {
            name: "Test".to_string(),
        };
        // Default returns false (allows re-triggering)
        assert!(!detector.is_processed());
    }

    #[test]
    fn test_DEventDetector_default_reset_processed() {
        let detector = MinimalDEventDetector {
            name: "Test".to_string(),
        };
        // Should not panic - is a no-op
        detector.reset_processed();
    }
}
