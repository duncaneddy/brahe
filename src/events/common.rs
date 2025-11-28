/*!
 * Common event detectors
 *
 * Value and binary event detectors for custom conditions.
 */

use super::traits::{
    DEventCallback, DEventDetector, EdgeType, EventAction, EventDirection, SEventCallback,
    SEventDetector,
};
use crate::time::Epoch;
use nalgebra::{DVector, SVector};
use std::sync::atomic::{AtomicBool, Ordering};

/// Time-based event detector (static-sized)
///
/// Fires when simulation time reaches the target time. Useful for
/// pre-planned maneuvers or discrete events at known times.
///
/// # Examples
/// ```
/// use brahe::events::STimeEvent;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let target = Epoch::from_jd(2451545.5, TimeSystem::UTC);
/// let event = STimeEvent::<6, 0>::new(target, "Maneuver Start");
///
/// // Customize time tolerance for precise detection (default: 1e-6 s)
/// let event = STimeEvent::<6, 0>::new(target, "Precise Event")
///     .with_time_tolerance(1e-9);
/// ```
pub struct STimeEvent<const S: usize, const P: usize> {
    target_time: Epoch,
    base_name: String,
    formatted_name: String,
    callback: Option<SEventCallback<S, P>>,
    action: EventAction,
    time_tol: f64,
    step_reduction_factor: f64,
    /// Flag to track if this time event has already fired (prevents re-triggering)
    fired: AtomicBool,
}

impl<const S: usize, const P: usize> STimeEvent<S, P> {
    /// Create a new time event
    pub fn new(target_time: Epoch, name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            target_time,
            formatted_name: name.clone(),
            base_name: name,
            callback: None,
            action: EventAction::Continue,
            time_tol: 1e-6,
            step_reduction_factor: 0.2,
            fired: AtomicBool::new(false),
        }
    }

    /// Reset the fired flag (allows the event to trigger again)
    pub fn reset(&self) {
        self.fired.store(false, Ordering::SeqCst);
    }

    /// Set instance number for display name
    pub fn with_instance(mut self, instance: usize) -> Self {
        self.formatted_name = format!("{} {}", self.base_name, instance);
        self
    }

    /// Set event callback
    pub fn with_callback(mut self, callback: SEventCallback<S, P>) -> Self {
        self.callback = Some(callback);
        self
    }

    /// Mark as terminal event (stops propagation)
    pub fn set_terminal(mut self) -> Self {
        self.action = EventAction::Stop;
        self
    }

    /// Set custom time tolerance for event detection
    ///
    /// Controls the precision of the bisection search algorithm. Smaller values
    /// result in more precise event time detection at the cost of more iterations.
    ///
    /// # Arguments
    /// * `time_tol` - Time tolerance in seconds (default: 1e-6)
    pub fn with_time_tolerance(mut self, time_tol: f64) -> Self {
        self.time_tol = time_tol;
        self
    }

    /// Set custom step reduction factor for bisection search
    ///
    /// When a zero-crossing is detected, the new step size is set to this
    /// factor times the bracket width. Default: 0.2
    pub fn with_step_reduction_factor(mut self, factor: f64) -> Self {
        self.step_reduction_factor = factor;
        self
    }
}

impl<const S: usize, const P: usize> SEventDetector<S, P> for STimeEvent<S, P> {
    fn evaluate(
        &self,
        t: Epoch,
        _state: &SVector<f64, S>,
        _params: Option<&SVector<f64, P>>,
    ) -> f64 {
        // If already processed, return MAX to prevent re-detection
        if self.fired.load(Ordering::SeqCst) {
            return f64::MAX;
        }

        // Return time difference (no auto-marking - propagator calls mark_processed())
        t - self.target_time
    }

    fn target_value(&self) -> f64 {
        0.0 // Event occurs when time reaches target (crossing from negative to positive)
    }

    fn name(&self) -> &str {
        &self.formatted_name
    }

    fn callback(&self) -> Option<&SEventCallback<S, P>> {
        self.callback.as_ref()
    }

    fn action(&self) -> EventAction {
        self.action
    }

    fn time_tolerance(&self) -> f64 {
        self.time_tol
    }

    fn step_reduction_factor(&self) -> f64 {
        self.step_reduction_factor
    }

    fn mark_processed(&self) {
        self.fired.store(true, Ordering::SeqCst);
    }

    fn is_processed(&self) -> bool {
        self.fired.load(Ordering::SeqCst)
    }

    fn reset_processed(&self) {
        self.fired.store(false, Ordering::SeqCst);
    }
}

/// Dynamic-sized time event
///
/// See [`STimeEvent`] for details. This version works with dynamic-sized
/// state vectors.
pub struct DTimeEvent {
    target_time: Epoch,
    base_name: String,
    formatted_name: String,
    callback: Option<DEventCallback>,
    action: EventAction,
    time_tol: f64,
    step_reduction_factor: f64,
    /// Flag to track if this time event has already fired (prevents re-triggering)
    fired: AtomicBool,
}

impl DTimeEvent {
    /// Create a new time event
    pub fn new(target_time: Epoch, name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            target_time,
            formatted_name: name.clone(),
            base_name: name,
            callback: None,
            action: EventAction::Continue,
            time_tol: 1e-6,
            step_reduction_factor: 0.2,
            fired: AtomicBool::new(false),
        }
    }

    /// Reset the fired flag (allows the event to trigger again)
    pub fn reset(&self) {
        self.fired.store(false, Ordering::SeqCst);
    }

    /// Set instance number for display name
    pub fn with_instance(mut self, instance: usize) -> Self {
        self.formatted_name = format!("{} {}", self.base_name, instance);
        self
    }

    /// Set event callback
    pub fn with_callback(mut self, callback: DEventCallback) -> Self {
        self.callback = Some(callback);
        self
    }

    /// Mark as terminal event (stops propagation)
    pub fn set_terminal(mut self) -> Self {
        self.action = EventAction::Stop;
        self
    }

    /// Set custom time tolerance for event detection
    ///
    /// Controls the precision of the bisection search algorithm. Smaller values
    /// result in more precise event time detection at the cost of more iterations.
    ///
    /// # Arguments
    /// * `time_tol` - Time tolerance in seconds (default: 1e-6)
    pub fn with_time_tolerance(mut self, time_tol: f64) -> Self {
        self.time_tol = time_tol;
        self
    }

    /// Set custom step reduction factor for bisection search
    ///
    /// When a zero-crossing is detected, the new step size is set to this
    /// factor times the bracket width. Default: 0.2
    pub fn with_step_reduction_factor(mut self, factor: f64) -> Self {
        self.step_reduction_factor = factor;
        self
    }
}

impl DEventDetector for DTimeEvent {
    fn evaluate(&self, t: Epoch, _state: &DVector<f64>, _params: Option<&DVector<f64>>) -> f64 {
        // If already processed, return MAX to prevent re-detection
        if self.fired.load(Ordering::SeqCst) {
            return f64::MAX;
        }

        // Return time difference (no auto-marking - propagator calls mark_processed())
        t - self.target_time
    }

    fn target_value(&self) -> f64 {
        0.0 // Event occurs when time reaches target (crossing from negative to positive)
    }

    fn name(&self) -> &str {
        &self.formatted_name
    }

    fn callback(&self) -> Option<&DEventCallback> {
        self.callback.as_ref()
    }

    fn action(&self) -> EventAction {
        self.action
    }

    fn time_tolerance(&self) -> f64 {
        self.time_tol
    }

    fn step_reduction_factor(&self) -> f64 {
        self.step_reduction_factor
    }

    fn mark_processed(&self) {
        self.fired.store(true, Ordering::SeqCst);
    }

    fn is_processed(&self) -> bool {
        self.fired.load(Ordering::SeqCst)
    }

    fn reset_processed(&self) {
        self.fired.store(false, Ordering::SeqCst);
    }
}

/// Value event detector (static-sized)
///
/// Detects when a continuous value crosses a target value. The user provides
/// a function that computes the value, and the target value is specified separately.
/// Internally converts to zero-crossing for bisection detection.
///
/// # Examples
/// ```
/// use brahe::events::{SValueEvent, EventDirection};
/// use brahe::constants::R_EARTH;
/// use brahe::time::Epoch;
/// use nalgebra::SVector;
///
/// // Detect when altitude drops below 500 km
/// let event = SValueEvent::<7, 4>::new(
///     "Low Altitude",
///     |_t: Epoch, state: &SVector<f64, 7>, _params| {
///         state.fixed_rows::<3>(0).norm()  // Just compute radius
///     },
///     R_EARTH + 500e3,  // Target value specified separately
///     EventDirection::Decreasing
/// );
/// ```
pub struct SValueEvent<const S: usize, const P: usize> {
    base_name: String,
    formatted_name: String,
    #[allow(clippy::type_complexity)]
    value_fn: Box<dyn Fn(Epoch, &SVector<f64, S>, Option<&SVector<f64, P>>) -> f64 + Send + Sync>,
    target_value: f64,
    direction: EventDirection,
    callback: Option<SEventCallback<S, P>>,
    action: EventAction,
    time_tol: f64,
    value_tol: f64,
    step_reduction_factor: f64,
}

impl<const S: usize, const P: usize> SValueEvent<S, P> {
    /// Create new value event
    ///
    /// # Arguments
    /// * `name` - Event identifier
    /// * `value_fn` - Function that computes the value to monitor
    /// * `target_value` - Target value for comparison
    /// * `direction` - Detection direction (increasing/decreasing/any)
    pub fn new<F>(
        name: impl Into<String>,
        value_fn: F,
        target_value: f64,
        direction: EventDirection,
    ) -> Self
    where
        F: Fn(Epoch, &SVector<f64, S>, Option<&SVector<f64, P>>) -> f64 + Send + Sync + 'static,
    {
        let name = name.into();
        Self {
            formatted_name: name.clone(),
            base_name: name,
            value_fn: Box::new(value_fn),
            target_value,
            direction,
            callback: None,
            action: EventAction::Continue,
            time_tol: 1e-6,
            value_tol: 1e-9,
            step_reduction_factor: 0.2,
        }
    }

    /// Set instance number for display name
    pub fn with_instance(mut self, instance: usize) -> Self {
        self.formatted_name = format!("{} {}", self.base_name, instance);
        self
    }

    /// Set custom tolerances for event detection
    pub fn with_tolerances(mut self, time_tol: f64, value_tol: f64) -> Self {
        self.time_tol = time_tol;
        self.value_tol = value_tol;
        self
    }

    /// Set custom step reduction factor for bisection search
    ///
    /// When a zero-crossing is detected, the new step size is set to this
    /// factor times the bracket width. Default: 0.2
    pub fn with_step_reduction_factor(mut self, factor: f64) -> Self {
        self.step_reduction_factor = factor;
        self
    }

    /// Set event callback
    pub fn with_callback(mut self, callback: SEventCallback<S, P>) -> Self {
        self.callback = Some(callback);
        self
    }

    /// Mark as terminal event (stops propagation)
    pub fn set_terminal(mut self) -> Self {
        self.action = EventAction::Stop;
        self
    }
}

impl<const S: usize, const P: usize> SEventDetector<S, P> for SValueEvent<S, P> {
    fn evaluate(&self, t: Epoch, state: &SVector<f64, S>, params: Option<&SVector<f64, P>>) -> f64 {
        // Return raw value from user function
        (self.value_fn)(t, state, params)
    }

    fn target_value(&self) -> f64 {
        self.target_value
    }

    fn name(&self) -> &str {
        &self.formatted_name
    }

    fn direction(&self) -> EventDirection {
        self.direction
    }

    fn time_tolerance(&self) -> f64 {
        self.time_tol
    }

    fn value_tolerance(&self) -> f64 {
        self.value_tol
    }

    fn step_reduction_factor(&self) -> f64 {
        self.step_reduction_factor
    }

    fn callback(&self) -> Option<&SEventCallback<S, P>> {
        self.callback.as_ref()
    }

    fn action(&self) -> EventAction {
        self.action
    }
}

/// Dynamic-sized value event detector
///
/// See [`SValueEvent`] for details. This version works with dynamic-sized
/// state vectors.
pub struct DValueEvent {
    base_name: String,
    formatted_name: String,
    #[allow(clippy::type_complexity)]
    value_fn: Box<dyn Fn(Epoch, &DVector<f64>, Option<&DVector<f64>>) -> f64 + Send + Sync>,
    target_value: f64,
    direction: EventDirection,
    callback: Option<DEventCallback>,
    action: EventAction,
    time_tol: f64,
    value_tol: f64,
    step_reduction_factor: f64,
}

impl DValueEvent {
    /// Create new value event
    pub fn new<F>(
        name: impl Into<String>,
        value_fn: F,
        target_value: f64,
        direction: EventDirection,
    ) -> Self
    where
        F: Fn(Epoch, &DVector<f64>, Option<&DVector<f64>>) -> f64 + Send + Sync + 'static,
    {
        let name = name.into();
        Self {
            formatted_name: name.clone(),
            base_name: name,
            value_fn: Box::new(value_fn),
            target_value,
            direction,
            callback: None,
            action: EventAction::Continue,
            time_tol: 1e-6,
            value_tol: 1e-9,
            step_reduction_factor: 0.2,
        }
    }

    /// Set instance number for display name
    pub fn with_instance(mut self, instance: usize) -> Self {
        self.formatted_name = format!("{} {}", self.base_name, instance);
        self
    }

    /// Set custom tolerances for event detection
    pub fn with_tolerances(mut self, time_tol: f64, value_tol: f64) -> Self {
        self.time_tol = time_tol;
        self.value_tol = value_tol;
        self
    }

    /// Set custom step reduction factor for bisection search
    ///
    /// When a zero-crossing is detected, the new step size is set to this
    /// factor times the bracket width. Default: 0.2
    pub fn with_step_reduction_factor(mut self, factor: f64) -> Self {
        self.step_reduction_factor = factor;
        self
    }

    /// Set event callback
    pub fn with_callback(mut self, callback: DEventCallback) -> Self {
        self.callback = Some(callback);
        self
    }

    /// Mark as terminal event
    pub fn set_terminal(mut self) -> Self {
        self.action = EventAction::Stop;
        self
    }
}

impl DEventDetector for DValueEvent {
    fn evaluate(&self, t: Epoch, state: &DVector<f64>, params: Option<&DVector<f64>>) -> f64 {
        (self.value_fn)(t, state, params)
    }

    fn target_value(&self) -> f64 {
        self.target_value
    }

    fn name(&self) -> &str {
        &self.formatted_name
    }

    fn direction(&self) -> EventDirection {
        self.direction
    }

    fn time_tolerance(&self) -> f64 {
        self.time_tol
    }

    fn value_tolerance(&self) -> f64 {
        self.value_tol
    }

    fn step_reduction_factor(&self) -> f64 {
        self.step_reduction_factor
    }

    fn callback(&self) -> Option<&DEventCallback> {
        self.callback.as_ref()
    }

    fn action(&self) -> EventAction {
        self.action
    }
}

/// Binary event detector (static-sized)
///
/// Detects transitions in boolean conditions (e.g., entering/exiting eclipse,
/// visibility changes). The user provides a predicate function that returns
/// `true` or `false`, and specifies which edge to detect (rising/falling/any).
///
/// # Examples
/// ```
/// use brahe::events::{SBinaryEvent, EdgeType};
/// use brahe::time::Epoch;
/// use nalgebra::SVector;
///
/// # fn is_sunlit(_pos: nalgebra::Vector3<f64>) -> bool { true }
/// // Detect eclipse entry (sunlit → shadow)
/// let event = SBinaryEvent::<7, 4>::new(
///     "Enter Eclipse",
///     |_t: Epoch, state: &SVector<f64, 7>, _params| {
///         // Returns true if sunlit, false if in shadow
///         is_sunlit(state.fixed_rows::<3>(0).into())
///     },
///     EdgeType::FallingEdge  // Detect true → false
/// );
/// ```
pub struct SBinaryEvent<const S: usize, const P: usize> {
    base_name: String,
    formatted_name: String,
    #[allow(clippy::type_complexity)]
    condition_fn:
        Box<dyn Fn(Epoch, &SVector<f64, S>, Option<&SVector<f64, P>>) -> bool + Send + Sync>,
    edge: EdgeType,
    callback: Option<SEventCallback<S, P>>,
    action: EventAction,
    time_tol: f64,
    value_tol: f64,
    step_reduction_factor: f64,
}

impl<const S: usize, const P: usize> SBinaryEvent<S, P> {
    /// Create new binary event
    ///
    /// # Arguments
    /// * `name` - Event identifier
    /// * `condition_fn` - Function that returns true/false
    /// * `edge` - Which edge to detect (rising/falling/any)
    pub fn new<F>(name: impl Into<String>, condition_fn: F, edge: EdgeType) -> Self
    where
        F: Fn(Epoch, &SVector<f64, S>, Option<&SVector<f64, P>>) -> bool + Send + Sync + 'static,
    {
        let name = name.into();
        Self {
            formatted_name: name.clone(),
            base_name: name,
            condition_fn: Box::new(condition_fn),
            edge,
            callback: None,
            action: EventAction::Continue,
            time_tol: 1e-6,
            value_tol: 1e-9,
            step_reduction_factor: 0.2,
        }
    }

    /// Set instance number for display name
    pub fn with_instance(mut self, instance: usize) -> Self {
        self.formatted_name = format!("{} {}", self.base_name, instance);
        self
    }

    /// Set custom tolerances for event detection
    pub fn with_tolerances(mut self, time_tol: f64, value_tol: f64) -> Self {
        self.time_tol = time_tol;
        self.value_tol = value_tol;
        self
    }

    /// Set custom step reduction factor for bisection search
    ///
    /// When a zero-crossing is detected, the new step size is set to this
    /// factor times the bracket width. Default: 0.2
    pub fn with_step_reduction_factor(mut self, factor: f64) -> Self {
        self.step_reduction_factor = factor;
        self
    }

    /// Set event callback
    pub fn with_callback(mut self, callback: SEventCallback<S, P>) -> Self {
        self.callback = Some(callback);
        self
    }

    /// Mark as terminal event (stops propagation)
    pub fn set_terminal(mut self) -> Self {
        self.action = EventAction::Stop;
        self
    }
}

impl<const S: usize, const P: usize> SEventDetector<S, P> for SBinaryEvent<S, P> {
    fn evaluate(&self, t: Epoch, state: &SVector<f64, S>, params: Option<&SVector<f64, P>>) -> f64 {
        // Convert boolean to value: true = 1.0, false = -1.0
        if (self.condition_fn)(t, state, params) {
            1.0
        } else {
            -1.0
        }
    }

    fn target_value(&self) -> f64 {
        0.0 // Zero-crossing occurs at target value of 0.0
    }

    fn name(&self) -> &str {
        &self.formatted_name
    }

    fn direction(&self) -> EventDirection {
        // Convert EdgeType to EventDirection
        match self.edge {
            EdgeType::RisingEdge => EventDirection::Increasing,
            EdgeType::FallingEdge => EventDirection::Decreasing,
            EdgeType::AnyEdge => EventDirection::Any,
        }
    }

    fn time_tolerance(&self) -> f64 {
        self.time_tol
    }

    fn value_tolerance(&self) -> f64 {
        self.value_tol
    }

    fn step_reduction_factor(&self) -> f64 {
        self.step_reduction_factor
    }

    fn callback(&self) -> Option<&SEventCallback<S, P>> {
        self.callback.as_ref()
    }

    fn action(&self) -> EventAction {
        self.action
    }
}

/// Dynamic-sized binary event detector
///
/// See [`SBinaryEvent`] for details. This version works with dynamic-sized
/// state vectors.
pub struct DBinaryEvent {
    base_name: String,
    formatted_name: String,
    #[allow(clippy::type_complexity)]
    condition_fn: Box<dyn Fn(Epoch, &DVector<f64>, Option<&DVector<f64>>) -> bool + Send + Sync>,
    edge: EdgeType,
    callback: Option<DEventCallback>,
    action: EventAction,
    time_tol: f64,
    value_tol: f64,
    step_reduction_factor: f64,
}

impl DBinaryEvent {
    /// Create new binary event
    pub fn new<F>(name: impl Into<String>, condition_fn: F, edge: EdgeType) -> Self
    where
        F: Fn(Epoch, &DVector<f64>, Option<&DVector<f64>>) -> bool + Send + Sync + 'static,
    {
        let name = name.into();
        Self {
            formatted_name: name.clone(),
            base_name: name,
            condition_fn: Box::new(condition_fn),
            edge,
            callback: None,
            action: EventAction::Continue,
            time_tol: 1e-6,
            value_tol: 1e-9,
            step_reduction_factor: 0.2,
        }
    }

    /// Set instance number for display name
    pub fn with_instance(mut self, instance: usize) -> Self {
        self.formatted_name = format!("{} {}", self.base_name, instance);
        self
    }

    /// Set custom tolerances for event detection
    pub fn with_tolerances(mut self, time_tol: f64, value_tol: f64) -> Self {
        self.time_tol = time_tol;
        self.value_tol = value_tol;
        self
    }

    /// Set custom step reduction factor for bisection search
    ///
    /// When a zero-crossing is detected, the new step size is set to this
    /// factor times the bracket width. Default: 0.2
    pub fn with_step_reduction_factor(mut self, factor: f64) -> Self {
        self.step_reduction_factor = factor;
        self
    }

    /// Set event callback
    pub fn with_callback(mut self, callback: DEventCallback) -> Self {
        self.callback = Some(callback);
        self
    }

    /// Mark as terminal event
    pub fn set_terminal(mut self) -> Self {
        self.action = EventAction::Stop;
        self
    }
}

impl DEventDetector for DBinaryEvent {
    fn evaluate(&self, t: Epoch, state: &DVector<f64>, params: Option<&DVector<f64>>) -> f64 {
        if (self.condition_fn)(t, state, params) {
            1.0
        } else {
            -1.0
        }
    }

    fn target_value(&self) -> f64 {
        0.0 // Zero-crossing occurs at target value of 0.0
    }

    fn name(&self) -> &str {
        &self.formatted_name
    }

    fn direction(&self) -> EventDirection {
        match self.edge {
            EdgeType::RisingEdge => EventDirection::Increasing,
            EdgeType::FallingEdge => EventDirection::Decreasing,
            EdgeType::AnyEdge => EventDirection::Any,
        }
    }

    fn time_tolerance(&self) -> f64 {
        self.time_tol
    }

    fn value_tolerance(&self) -> f64 {
        self.value_tol
    }

    fn step_reduction_factor(&self) -> f64 {
        self.step_reduction_factor
    }

    fn callback(&self) -> Option<&DEventCallback> {
        self.callback.as_ref()
    }

    fn action(&self) -> EventAction {
        self.action
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use crate::time::TimeSystem;
    use nalgebra::{DVector, Vector6};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    // =========================================================================
    // STimeEvent Tests
    // =========================================================================

    #[test]
    fn test_STimeEvent_new() {
        let target = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let event = STimeEvent::<6, 0>::new(target, "Test Event");

        assert_eq!(event.name(), "Test Event");
        assert_eq!(event.target_value(), 0.0);
        assert_eq!(event.action(), EventAction::Continue);
        assert!(!event.is_processed());
    }

    #[test]
    fn test_STimeEvent_evaluate() {
        let target = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let event = STimeEvent::<6, 0>::new(target, "Test");

        let state = Vector6::zeros();

        // Before target - evaluate returns signed difference (negative)
        let before_val = event.evaluate(target - 10.0, &state, None);
        assert_eq!(before_val, -10.0); // 10 seconds before target
        assert!((before_val - event.target_value()) < 0.0); // -10.0 - 0.0 = -10.0 (negative)

        // At target
        let at_val = event.evaluate(target, &state, None);
        assert_eq!(at_val, 0.0); // At target
        assert_eq!(at_val - event.target_value(), 0.0); // Zero-crossing

        // After target - evaluate returns signed difference (positive)
        let after_val = event.evaluate(target + 10.0, &state, None);
        assert_eq!(after_val, 10.0); // 10 seconds after target
        assert!((after_val - event.target_value()) > 0.0); // 10.0 - 0.0 = 10.0 (positive)
    }

    #[test]
    fn test_STimeEvent_reset() {
        let target = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let event = STimeEvent::<6, 0>::new(target, "Test");

        // Initially not processed
        assert!(!event.is_processed());

        // Mark as processed
        event.mark_processed();
        assert!(event.is_processed());

        // Reset
        event.reset();
        assert!(!event.is_processed());
    }

    #[test]
    fn test_STimeEvent_with_instance() {
        let target = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let event = STimeEvent::<6, 0>::new(target, "Maneuver").with_instance(3);

        assert_eq!(event.name(), "Maneuver 3");
    }

    #[test]
    fn test_STimeEvent_with_callback() {
        let target = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let called = Arc::new(AtomicBool::new(false));
        let called_clone = called.clone();

        let callback: SEventCallback<6, 0> = Box::new(move |_t, _state, _params| {
            called_clone.store(true, Ordering::SeqCst);
            (None, None, EventAction::Continue)
        });

        let event = STimeEvent::<6, 0>::new(target, "Test").with_callback(callback);

        // Callback should exist
        assert!(event.callback().is_some());

        // Execute callback
        let state = Vector6::zeros();
        if let Some(cb) = event.callback() {
            cb(target, &state, None);
        }
        assert!(called.load(Ordering::SeqCst));
    }

    #[test]
    fn test_STimeEvent_set_terminal() {
        let target = Epoch::from_jd(2451545.0, TimeSystem::UTC);

        let event = STimeEvent::<6, 0>::new(target, "Test");
        assert_eq!(event.action(), EventAction::Continue);

        let event = STimeEvent::<6, 0>::new(target, "Test").set_terminal();
        assert_eq!(event.action(), EventAction::Stop);
    }

    #[test]
    fn test_STimeEvent_with_step_reduction_factor() {
        let target = Epoch::from_jd(2451545.0, TimeSystem::UTC);

        // Default
        let event = STimeEvent::<6, 0>::new(target, "Test");
        assert_eq!(event.step_reduction_factor(), 0.2);

        // Custom
        let event = STimeEvent::<6, 0>::new(target, "Test").with_step_reduction_factor(0.1);
        assert_eq!(event.step_reduction_factor(), 0.1);
    }

    #[test]
    fn test_STimeEvent_mark_processed() {
        let target = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let event = STimeEvent::<6, 0>::new(target, "Test");

        assert!(!event.is_processed());
        event.mark_processed();
        assert!(event.is_processed());
    }

    #[test]
    fn test_STimeEvent_reset_processed() {
        let target = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let event = STimeEvent::<6, 0>::new(target, "Test");

        event.mark_processed();
        assert!(event.is_processed());

        event.reset_processed();
        assert!(!event.is_processed());
    }

    #[test]
    fn test_STimeEvent_evaluate_after_processed() {
        let target = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let event = STimeEvent::<6, 0>::new(target, "Test");
        let state = Vector6::zeros();

        // Before marking processed, evaluate returns time difference
        let val = event.evaluate(target, &state, None);
        assert_eq!(val, 0.0);

        // After marking processed, evaluate returns MAX
        event.mark_processed();
        let val = event.evaluate(target, &state, None);
        assert_eq!(val, f64::MAX);
    }

    // =========================================================================
    // DTimeEvent Tests
    // =========================================================================

    #[test]
    fn test_DTimeEvent_new() {
        let target = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let event = DTimeEvent::new(target, "Test Event");

        assert_eq!(event.name(), "Test Event");
        assert_eq!(event.target_value(), 0.0);
        assert_eq!(event.action(), EventAction::Continue);
        assert!(!event.is_processed());
    }

    #[test]
    fn test_DTimeEvent_evaluate() {
        let target = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let event = DTimeEvent::new(target, "Test");

        let state = DVector::from_vec(vec![0.0; 6]);

        // Before target
        let before_val = event.evaluate(target - 10.0, &state, None);
        assert_eq!(before_val, -10.0);

        // At target
        let at_val = event.evaluate(target, &state, None);
        assert_eq!(at_val, 0.0);

        // After target
        let after_val = event.evaluate(target + 10.0, &state, None);
        assert_eq!(after_val, 10.0);
    }

    #[test]
    fn test_DTimeEvent_reset() {
        let target = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let event = DTimeEvent::new(target, "Test");

        event.mark_processed();
        assert!(event.is_processed());

        event.reset();
        assert!(!event.is_processed());
    }

    #[test]
    fn test_DTimeEvent_with_instance() {
        let target = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let event = DTimeEvent::new(target, "Maneuver").with_instance(5);

        assert_eq!(event.name(), "Maneuver 5");
    }

    #[test]
    fn test_DTimeEvent_with_callback() {
        let target = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let called = Arc::new(AtomicBool::new(false));
        let called_clone = called.clone();

        let callback: DEventCallback = Box::new(move |_t, _state, _params| {
            called_clone.store(true, Ordering::SeqCst);
            (None, None, EventAction::Continue)
        });

        let event = DTimeEvent::new(target, "Test").with_callback(callback);
        assert!(event.callback().is_some());
    }

    #[test]
    fn test_DTimeEvent_set_terminal() {
        let target = Epoch::from_jd(2451545.0, TimeSystem::UTC);

        let event = DTimeEvent::new(target, "Test");
        assert_eq!(event.action(), EventAction::Continue);

        let event = DTimeEvent::new(target, "Test").set_terminal();
        assert_eq!(event.action(), EventAction::Stop);
    }

    #[test]
    fn test_DTimeEvent_with_step_reduction_factor() {
        let target = Epoch::from_jd(2451545.0, TimeSystem::UTC);

        let event = DTimeEvent::new(target, "Test");
        assert_eq!(event.step_reduction_factor(), 0.2);

        let event = DTimeEvent::new(target, "Test").with_step_reduction_factor(0.3);
        assert_eq!(event.step_reduction_factor(), 0.3);
    }

    #[test]
    fn test_DTimeEvent_mark_processed() {
        let target = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let event = DTimeEvent::new(target, "Test");

        assert!(!event.is_processed());
        event.mark_processed();
        assert!(event.is_processed());
    }

    #[test]
    fn test_DTimeEvent_reset_processed() {
        let target = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let event = DTimeEvent::new(target, "Test");

        event.mark_processed();
        event.reset_processed();
        assert!(!event.is_processed());
    }

    #[test]
    fn test_DTimeEvent_evaluate_after_processed() {
        let target = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let event = DTimeEvent::new(target, "Test");
        let state = DVector::from_vec(vec![0.0; 6]);

        // Before marking processed
        let val = event.evaluate(target, &state, None);
        assert_eq!(val, 0.0);

        // After marking processed
        event.mark_processed();
        let val = event.evaluate(target, &state, None);
        assert_eq!(val, f64::MAX);
    }

    // =========================================================================
    // SValueEvent Tests
    // =========================================================================

    #[test]
    fn test_SValueEvent_new() {
        let event = SValueEvent::<6, 0>::new(
            "X-Crossing",
            |_t, state: &Vector6<f64>, _params| state[0],
            7000e3,
            EventDirection::Any,
        );

        assert_eq!(event.name(), "X-Crossing");
        assert_eq!(event.target_value(), 7000e3);
        assert_eq!(event.direction(), EventDirection::Any);
        assert_eq!(event.action(), EventAction::Continue);
    }

    #[test]
    fn test_SValueEvent_evaluate() {
        let event = SValueEvent::<6, 0>::new(
            "X-Crossing",
            |_t, state: &Vector6<f64>, _params| state[0],
            7000e3,
            EventDirection::Any,
        );

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);

        // Below target value
        let state_below = Vector6::new(6000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        let val_below = event.evaluate(epoch, &state_below, None);
        assert_eq!(val_below, 6000e3);
        assert!((val_below - event.target_value()) < 0.0);

        // At target value
        let state_at = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        let val_at = event.evaluate(epoch, &state_at, None);
        assert_eq!(val_at, 7000e3);

        // Above target value
        let state_above = Vector6::new(8000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        let val_above = event.evaluate(epoch, &state_above, None);
        assert_eq!(val_above, 8000e3);
        assert!((val_above - event.target_value()) > 0.0);
    }

    #[test]
    fn test_SValueEvent_with_instance() {
        let event = SValueEvent::<6, 0>::new(
            "Altitude",
            |_t, _state: &Vector6<f64>, _params| 0.0,
            500e3,
            EventDirection::Any,
        )
        .with_instance(2);

        assert_eq!(event.name(), "Altitude 2");
    }

    #[test]
    fn test_SValueEvent_with_tolerances() {
        let event = SValueEvent::<6, 0>::new(
            "Test",
            |_t, _state: &Vector6<f64>, _params| 0.0,
            0.0,
            EventDirection::Any,
        );

        // Default tolerances
        assert_eq!(event.time_tolerance(), 1e-6);
        assert_eq!(event.value_tolerance(), 1e-9);

        // Custom tolerances
        let event = SValueEvent::<6, 0>::new(
            "Test",
            |_t, _state: &Vector6<f64>, _params| 0.0,
            0.0,
            EventDirection::Any,
        )
        .with_tolerances(1e-3, 1e-6);

        assert_eq!(event.time_tolerance(), 1e-3);
        assert_eq!(event.value_tolerance(), 1e-6);
    }

    #[test]
    fn test_SValueEvent_with_step_reduction_factor() {
        let event = SValueEvent::<6, 0>::new(
            "Test",
            |_t, _state: &Vector6<f64>, _params| 0.0,
            0.0,
            EventDirection::Any,
        );

        assert_eq!(event.step_reduction_factor(), 0.2);

        let event = event.with_step_reduction_factor(0.1);
        assert_eq!(event.step_reduction_factor(), 0.1);
    }

    #[test]
    fn test_SValueEvent_with_callback() {
        let called = Arc::new(AtomicBool::new(false));
        let called_clone = called.clone();

        let callback: SEventCallback<6, 0> = Box::new(move |_t, _state, _params| {
            called_clone.store(true, Ordering::SeqCst);
            (None, None, EventAction::Continue)
        });

        let event = SValueEvent::<6, 0>::new(
            "Test",
            |_t, _state: &Vector6<f64>, _params| 0.0,
            0.0,
            EventDirection::Any,
        )
        .with_callback(callback);

        assert!(event.callback().is_some());
    }

    #[test]
    fn test_SValueEvent_set_terminal() {
        let event = SValueEvent::<6, 0>::new(
            "Test",
            |_t, _state: &Vector6<f64>, _params| 0.0,
            0.0,
            EventDirection::Any,
        );

        assert_eq!(event.action(), EventAction::Continue);

        let event = event.set_terminal();
        assert_eq!(event.action(), EventAction::Stop);
    }

    #[test]
    fn test_SValueEvent_direction_increasing() {
        let event = SValueEvent::<6, 0>::new(
            "Test",
            |_t, _state: &Vector6<f64>, _params| 0.0,
            0.0,
            EventDirection::Increasing,
        );

        assert_eq!(event.direction(), EventDirection::Increasing);
    }

    #[test]
    fn test_SValueEvent_direction_decreasing() {
        let event = SValueEvent::<6, 0>::new(
            "Test",
            |_t, _state: &Vector6<f64>, _params| 0.0,
            0.0,
            EventDirection::Decreasing,
        );

        assert_eq!(event.direction(), EventDirection::Decreasing);
    }

    // =========================================================================
    // DValueEvent Tests
    // =========================================================================

    #[test]
    fn test_DValueEvent_new() {
        let event = DValueEvent::new(
            "X-Crossing",
            |_t, state: &DVector<f64>, _params| state[0],
            7000e3,
            EventDirection::Any,
        );

        assert_eq!(event.name(), "X-Crossing");
        assert_eq!(event.target_value(), 7000e3);
        assert_eq!(event.direction(), EventDirection::Any);
    }

    #[test]
    fn test_DValueEvent_evaluate() {
        let event = DValueEvent::new(
            "X-Crossing",
            |_t, state: &DVector<f64>, _params| state[0],
            7000e3,
            EventDirection::Any,
        );

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![6000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);

        let val = event.evaluate(epoch, &state, None);
        assert_eq!(val, 6000e3);
    }

    #[test]
    fn test_DValueEvent_with_instance() {
        let event = DValueEvent::new(
            "Altitude",
            |_t, _state: &DVector<f64>, _params| 0.0,
            500e3,
            EventDirection::Any,
        )
        .with_instance(3);

        assert_eq!(event.name(), "Altitude 3");
    }

    #[test]
    fn test_DValueEvent_with_tolerances() {
        let event = DValueEvent::new(
            "Test",
            |_t, _state: &DVector<f64>, _params| 0.0,
            0.0,
            EventDirection::Any,
        )
        .with_tolerances(1e-4, 1e-7);

        assert_eq!(event.time_tolerance(), 1e-4);
        assert_eq!(event.value_tolerance(), 1e-7);
    }

    #[test]
    fn test_DValueEvent_with_step_reduction_factor() {
        let event = DValueEvent::new(
            "Test",
            |_t, _state: &DVector<f64>, _params| 0.0,
            0.0,
            EventDirection::Any,
        )
        .with_step_reduction_factor(0.15);

        assert_eq!(event.step_reduction_factor(), 0.15);
    }

    #[test]
    fn test_DValueEvent_with_callback() {
        let callback: DEventCallback =
            Box::new(|_t, _state, _params| (None, None, EventAction::Continue));

        let event = DValueEvent::new(
            "Test",
            |_t, _state: &DVector<f64>, _params| 0.0,
            0.0,
            EventDirection::Any,
        )
        .with_callback(callback);

        assert!(event.callback().is_some());
    }

    #[test]
    fn test_DValueEvent_set_terminal() {
        let event = DValueEvent::new(
            "Test",
            |_t, _state: &DVector<f64>, _params| 0.0,
            0.0,
            EventDirection::Any,
        )
        .set_terminal();

        assert_eq!(event.action(), EventAction::Stop);
    }

    // =========================================================================
    // SBinaryEvent Tests
    // =========================================================================

    #[test]
    fn test_SBinaryEvent_new() {
        let event = SBinaryEvent::<6, 0>::new(
            "X-Positive",
            |_t, state: &Vector6<f64>, _params| state[0] > 0.0,
            EdgeType::RisingEdge,
        );

        assert_eq!(event.name(), "X-Positive");
        assert_eq!(event.target_value(), 0.0);
        assert_eq!(event.direction(), EventDirection::Increasing);
    }

    #[test]
    fn test_SBinaryEvent_evaluate_true_returns_positive() {
        let event = SBinaryEvent::<6, 0>::new(
            "Test",
            |_t, state: &Vector6<f64>, _params| state[0] > 0.0,
            EdgeType::AnyEdge,
        );

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state = Vector6::new(1000.0, 0.0, 0.0, 0.0, 0.0, 0.0);

        assert_eq!(event.evaluate(epoch, &state, None), 1.0);
    }

    #[test]
    fn test_SBinaryEvent_evaluate_false_returns_negative() {
        let event = SBinaryEvent::<6, 0>::new(
            "Test",
            |_t, state: &Vector6<f64>, _params| state[0] > 0.0,
            EdgeType::AnyEdge,
        );

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state = Vector6::new(-1000.0, 0.0, 0.0, 0.0, 0.0, 0.0);

        assert_eq!(event.evaluate(epoch, &state, None), -1.0);
    }

    #[test]
    fn test_SBinaryEvent_edge_rising() {
        let event = SBinaryEvent::<6, 0>::new(
            "Test",
            |_t, _state: &Vector6<f64>, _params| true,
            EdgeType::RisingEdge,
        );

        assert_eq!(event.direction(), EventDirection::Increasing);
    }

    #[test]
    fn test_SBinaryEvent_edge_falling() {
        let event = SBinaryEvent::<6, 0>::new(
            "Test",
            |_t, _state: &Vector6<f64>, _params| true,
            EdgeType::FallingEdge,
        );

        assert_eq!(event.direction(), EventDirection::Decreasing);
    }

    #[test]
    fn test_SBinaryEvent_edge_any() {
        let event = SBinaryEvent::<6, 0>::new(
            "Test",
            |_t, _state: &Vector6<f64>, _params| true,
            EdgeType::AnyEdge,
        );

        assert_eq!(event.direction(), EventDirection::Any);
    }

    #[test]
    fn test_SBinaryEvent_with_instance() {
        let event = SBinaryEvent::<6, 0>::new(
            "Eclipse",
            |_t, _state: &Vector6<f64>, _params| true,
            EdgeType::AnyEdge,
        )
        .with_instance(1);

        assert_eq!(event.name(), "Eclipse 1");
    }

    #[test]
    fn test_SBinaryEvent_with_tolerances() {
        let event = SBinaryEvent::<6, 0>::new(
            "Test",
            |_t, _state: &Vector6<f64>, _params| true,
            EdgeType::AnyEdge,
        )
        .with_tolerances(1e-5, 1e-8);

        assert_eq!(event.time_tolerance(), 1e-5);
        assert_eq!(event.value_tolerance(), 1e-8);
    }

    #[test]
    fn test_SBinaryEvent_with_step_reduction_factor() {
        let event = SBinaryEvent::<6, 0>::new(
            "Test",
            |_t, _state: &Vector6<f64>, _params| true,
            EdgeType::AnyEdge,
        )
        .with_step_reduction_factor(0.25);

        assert_eq!(event.step_reduction_factor(), 0.25);
    }

    #[test]
    fn test_SBinaryEvent_with_callback() {
        let callback: SEventCallback<6, 0> =
            Box::new(|_t, _state, _params| (None, None, EventAction::Stop));

        let event = SBinaryEvent::<6, 0>::new(
            "Test",
            |_t, _state: &Vector6<f64>, _params| true,
            EdgeType::AnyEdge,
        )
        .with_callback(callback);

        assert!(event.callback().is_some());
    }

    #[test]
    fn test_SBinaryEvent_set_terminal() {
        let event = SBinaryEvent::<6, 0>::new(
            "Test",
            |_t, _state: &Vector6<f64>, _params| true,
            EdgeType::AnyEdge,
        )
        .set_terminal();

        assert_eq!(event.action(), EventAction::Stop);
    }

    // =========================================================================
    // DBinaryEvent Tests
    // =========================================================================

    #[test]
    fn test_DBinaryEvent_new() {
        let event = DBinaryEvent::new(
            "X-Positive",
            |_t, state: &DVector<f64>, _params| state[0] > 0.0,
            EdgeType::FallingEdge,
        );

        assert_eq!(event.name(), "X-Positive");
        assert_eq!(event.direction(), EventDirection::Decreasing);
    }

    #[test]
    fn test_DBinaryEvent_evaluate_true_returns_positive() {
        let event = DBinaryEvent::new(
            "Test",
            |_t, state: &DVector<f64>, _params| state[0] > 0.0,
            EdgeType::AnyEdge,
        );

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![1000.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        assert_eq!(event.evaluate(epoch, &state, None), 1.0);
    }

    #[test]
    fn test_DBinaryEvent_evaluate_false_returns_negative() {
        let event = DBinaryEvent::new(
            "Test",
            |_t, state: &DVector<f64>, _params| state[0] > 0.0,
            EdgeType::AnyEdge,
        );

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![-1000.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        assert_eq!(event.evaluate(epoch, &state, None), -1.0);
    }

    #[test]
    fn test_DBinaryEvent_edge_rising() {
        let event = DBinaryEvent::new(
            "Test",
            |_t, _state: &DVector<f64>, _params| true,
            EdgeType::RisingEdge,
        );

        assert_eq!(event.direction(), EventDirection::Increasing);
    }

    #[test]
    fn test_DBinaryEvent_edge_falling() {
        let event = DBinaryEvent::new(
            "Test",
            |_t, _state: &DVector<f64>, _params| true,
            EdgeType::FallingEdge,
        );

        assert_eq!(event.direction(), EventDirection::Decreasing);
    }

    #[test]
    fn test_DBinaryEvent_edge_any() {
        let event = DBinaryEvent::new(
            "Test",
            |_t, _state: &DVector<f64>, _params| true,
            EdgeType::AnyEdge,
        );

        assert_eq!(event.direction(), EventDirection::Any);
    }

    #[test]
    fn test_DBinaryEvent_with_instance() {
        let event = DBinaryEvent::new(
            "Eclipse",
            |_t, _state: &DVector<f64>, _params| true,
            EdgeType::AnyEdge,
        )
        .with_instance(2);

        assert_eq!(event.name(), "Eclipse 2");
    }

    #[test]
    fn test_DBinaryEvent_with_tolerances() {
        let event = DBinaryEvent::new(
            "Test",
            |_t, _state: &DVector<f64>, _params| true,
            EdgeType::AnyEdge,
        )
        .with_tolerances(1e-4, 1e-7);

        assert_eq!(event.time_tolerance(), 1e-4);
        assert_eq!(event.value_tolerance(), 1e-7);
    }

    #[test]
    fn test_DBinaryEvent_with_step_reduction_factor() {
        let event = DBinaryEvent::new(
            "Test",
            |_t, _state: &DVector<f64>, _params| true,
            EdgeType::AnyEdge,
        )
        .with_step_reduction_factor(0.3);

        assert_eq!(event.step_reduction_factor(), 0.3);
    }

    #[test]
    fn test_DBinaryEvent_with_callback() {
        let callback: DEventCallback =
            Box::new(|_t, _state, _params| (None, None, EventAction::Continue));

        let event = DBinaryEvent::new(
            "Test",
            |_t, _state: &DVector<f64>, _params| true,
            EdgeType::AnyEdge,
        )
        .with_callback(callback);

        assert!(event.callback().is_some());
    }

    #[test]
    fn test_DBinaryEvent_set_terminal() {
        let event = DBinaryEvent::new(
            "Test",
            |_t, _state: &DVector<f64>, _params| true,
            EdgeType::AnyEdge,
        )
        .set_terminal();

        assert_eq!(event.action(), EventAction::Stop);
    }
}
