//! Event query iterator adapter
//!
//! Provides chainable filtering for detected events with an idiomatic Rust iterator interface.

use crate::events::{DDetectedEvent, EventAction, EventType};
use crate::time::Epoch;

/// Iterator adapter for filtering detected events
///
/// Provides chainable filter methods that compose naturally with
/// standard iterator operations.
///
/// # Examples
///
/// ```ignore
/// use brahe::propagators::DNumericalOrbitPropagator;
/// use brahe::time::Epoch;
///
/// // Get events from detector 0
/// let events: Vec<_> = propagator.query_events()
///     .by_detector_index(0)
///     .collect();
///
/// // Combined filters
/// let events: Vec<_> = propagator.query_events()
///     .by_detector_index(1)
///     .in_time_range(start, end)
///     .collect();
///
/// // With iterator methods
/// let count = propagator.query_events()
///     .by_name_contains("Altitude")
///     .count();
/// ```
pub struct EventQuery<'a, I>
where
    I: Iterator<Item = &'a DDetectedEvent>,
{
    iter: I,
}

impl<'a, I> EventQuery<'a, I>
where
    I: Iterator<Item = &'a DDetectedEvent> + 'a,
{
    /// Create a new EventQuery from an iterator
    ///
    /// This is typically called by propagator methods, not directly by users.
    pub(crate) fn new(iter: I) -> Self {
        Self { iter }
    }

    /// Filter by detector index
    ///
    /// Returns events detected by the specified detector.
    ///
    /// # Arguments
    ///
    /// * `index` - Detector index (0-based, corresponding to add order)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let events: Vec<_> = propagator.query_events()
    ///     .by_detector_index(0)
    ///     .collect();
    /// ```
    pub fn by_detector_index(
        self,
        index: usize,
    ) -> EventQuery<'a, impl Iterator<Item = &'a DDetectedEvent> + 'a> {
        EventQuery::new(self.iter.filter(move |e| e.detector_index == index))
    }

    /// Filter by exact detector name
    ///
    /// Returns events where the detector name exactly matches the given string.
    ///
    /// # Arguments
    ///
    /// * `name` - Exact name to match
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let events: Vec<_> = propagator.query_events()
    ///     .by_name_exact("Altitude Event")
    ///     .collect();
    /// ```
    pub fn by_name_exact(
        self,
        name: &str,
    ) -> EventQuery<'a, impl Iterator<Item = &'a DDetectedEvent> + 'a> {
        let name = name.to_string();
        EventQuery::new(self.iter.filter(move |e| e.name == name))
    }

    /// Filter by detector name substring
    ///
    /// Returns events where the detector name contains the given substring.
    ///
    /// # Arguments
    ///
    /// * `substring` - Substring to search for in event names
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let events: Vec<_> = propagator.query_events()
    ///     .by_name_contains("Altitude")
    ///     .collect();
    /// ```
    pub fn by_name_contains(
        self,
        substring: &str,
    ) -> EventQuery<'a, impl Iterator<Item = &'a DDetectedEvent> + 'a> {
        let substring = substring.to_string();
        EventQuery::new(self.iter.filter(move |e| e.name.contains(&substring)))
    }

    /// Filter by time range (inclusive)
    ///
    /// Returns events that occurred within the specified time range.
    ///
    /// # Arguments
    ///
    /// * `start` - Start of time range (inclusive)
    /// * `end` - End of time range (inclusive)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let events: Vec<_> = propagator.query_events()
    ///     .in_time_range(start_epoch, end_epoch)
    ///     .collect();
    /// ```
    pub fn in_time_range(
        self,
        start: Epoch,
        end: Epoch,
    ) -> EventQuery<'a, impl Iterator<Item = &'a DDetectedEvent> + 'a> {
        EventQuery::new(
            self.iter
                .filter(move |e| e.window_open >= start && e.window_open <= end),
        )
    }

    /// Filter events after epoch (inclusive)
    ///
    /// Returns events that occurred at or after the specified epoch.
    ///
    /// # Arguments
    ///
    /// * `epoch` - Epoch value (inclusive)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let events: Vec<_> = propagator.query_events()
    ///     .after(cutoff_epoch)
    ///     .collect();
    /// ```
    pub fn after(
        self,
        epoch: Epoch,
    ) -> EventQuery<'a, impl Iterator<Item = &'a DDetectedEvent> + 'a> {
        EventQuery::new(self.iter.filter(move |e| e.window_open >= epoch))
    }

    /// Filter events before epoch (inclusive)
    ///
    /// Returns events that occurred at or before the specified epoch.
    ///
    /// # Arguments
    ///
    /// * `epoch` - Epoch value (inclusive)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let events: Vec<_> = propagator.query_events()
    ///     .before(cutoff_epoch)
    ///     .collect();
    /// ```
    pub fn before(
        self,
        epoch: Epoch,
    ) -> EventQuery<'a, impl Iterator<Item = &'a DDetectedEvent> + 'a> {
        EventQuery::new(self.iter.filter(move |e| e.window_open <= epoch))
    }

    /// Filter by event type
    ///
    /// Returns events of the specified type.
    ///
    /// # Arguments
    ///
    /// * `event_type` - Event type to filter by (Instantaneous or Period)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use brahe::events::EventType;
    ///
    /// let events: Vec<_> = propagator.query_events()
    ///     .by_event_type(EventType::Window)
    ///     .collect();
    /// ```
    pub fn by_event_type(
        self,
        event_type: EventType,
    ) -> EventQuery<'a, impl Iterator<Item = &'a DDetectedEvent> + 'a> {
        EventQuery::new(self.iter.filter(move |e| e.event_type == event_type))
    }

    /// Filter by event action
    ///
    /// Returns events with the specified action.
    ///
    /// # Arguments
    ///
    /// * `action` - Event action to filter by (Stop or Continue)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use brahe::events::EventAction;
    ///
    /// let events: Vec<_> = propagator.query_events()
    ///     .by_action(EventAction::Stop)
    ///     .collect();
    /// ```
    pub fn by_action(
        self,
        action: EventAction,
    ) -> EventQuery<'a, impl Iterator<Item = &'a DDetectedEvent> + 'a> {
        EventQuery::new(self.iter.filter(move |e| e.action == action))
    }
}

// Implement Iterator to enable .collect(), .count(), etc.
impl<'a, I> Iterator for EventQuery<'a, I>
where
    I: Iterator<Item = &'a DDetectedEvent>,
{
    type Item = &'a DDetectedEvent;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::time::TimeSystem;
    use nalgebra::DVector;

    /// Helper function to create test events
    fn create_test_event(
        name: &str,
        detector_index: usize,
        event_type: EventType,
        action: EventAction,
        time_offset_seconds: f64,
    ) -> DDetectedEvent {
        let base_epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let event_epoch = base_epoch + time_offset_seconds;
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);

        DDetectedEvent {
            window_open: event_epoch,
            window_close: event_epoch + 10.0,
            entry_state: state.clone(),
            exit_state: state,
            value: 0.0,
            name: name.to_string(),
            action,
            event_type,
            detector_index,
        }
    }

    // =========================================================================
    // by_name_exact Tests
    // =========================================================================

    #[test]
    fn test_EventQuery_by_name_exact_matches() {
        let events = [
            create_test_event(
                "Altitude Event",
                0,
                EventType::Instantaneous,
                EventAction::Continue,
                0.0,
            ),
            create_test_event(
                "Perigee Event",
                1,
                EventType::Instantaneous,
                EventAction::Continue,
                100.0,
            ),
        ];

        let result: Vec<_> = EventQuery::new(events.iter())
            .by_name_exact("Altitude Event")
            .collect();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "Altitude Event");
    }

    #[test]
    fn test_EventQuery_by_name_exact_no_match() {
        let events = [create_test_event(
            "Altitude Event",
            0,
            EventType::Instantaneous,
            EventAction::Continue,
            0.0,
        )];

        let result: Vec<_> = EventQuery::new(events.iter())
            .by_name_exact("Nonexistent Event")
            .collect();

        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_EventQuery_by_name_exact_partial_no_match() {
        let events = [create_test_event(
            "Altitude Event",
            0,
            EventType::Instantaneous,
            EventAction::Continue,
            0.0,
        )];

        // Partial match should NOT work with by_name_exact
        let result: Vec<_> = EventQuery::new(events.iter())
            .by_name_exact("Altitude")
            .collect();

        assert_eq!(result.len(), 0);
    }

    // =========================================================================
    // after Tests
    // =========================================================================

    #[test]
    fn test_EventQuery_after_includes_matching_epoch() {
        let base_epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let events = [create_test_event(
            "Event",
            0,
            EventType::Instantaneous,
            EventAction::Continue,
            0.0,
        )];

        // Filter at exactly the event epoch (should be inclusive)
        let result: Vec<_> = EventQuery::new(events.iter()).after(base_epoch).collect();

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_EventQuery_after_includes_later_events() {
        let base_epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let events = [
            create_test_event(
                "Early",
                0,
                EventType::Instantaneous,
                EventAction::Continue,
                0.0,
            ),
            create_test_event(
                "Late",
                1,
                EventType::Instantaneous,
                EventAction::Continue,
                200.0,
            ),
        ];

        let result: Vec<_> = EventQuery::new(events.iter())
            .after(base_epoch + 100.0)
            .collect();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "Late");
    }

    #[test]
    fn test_EventQuery_after_excludes_earlier_events() {
        let base_epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let events = [
            create_test_event(
                "Early",
                0,
                EventType::Instantaneous,
                EventAction::Continue,
                0.0,
            ),
            create_test_event(
                "Late",
                1,
                EventType::Instantaneous,
                EventAction::Continue,
                200.0,
            ),
        ];

        let result: Vec<_> = EventQuery::new(events.iter())
            .after(base_epoch + 300.0)
            .collect();

        assert_eq!(result.len(), 0);
    }

    // =========================================================================
    // before Tests
    // =========================================================================

    #[test]
    fn test_EventQuery_before_includes_matching_epoch() {
        let base_epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let events = [create_test_event(
            "Event",
            0,
            EventType::Instantaneous,
            EventAction::Continue,
            0.0,
        )];

        // Filter at exactly the event epoch (should be inclusive)
        let result: Vec<_> = EventQuery::new(events.iter()).before(base_epoch).collect();

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_EventQuery_before_includes_earlier_events() {
        let base_epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let events = [
            create_test_event(
                "Early",
                0,
                EventType::Instantaneous,
                EventAction::Continue,
                0.0,
            ),
            create_test_event(
                "Late",
                1,
                EventType::Instantaneous,
                EventAction::Continue,
                200.0,
            ),
        ];

        let result: Vec<_> = EventQuery::new(events.iter())
            .before(base_epoch + 100.0)
            .collect();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "Early");
    }

    #[test]
    fn test_EventQuery_before_excludes_later_events() {
        let base_epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let events = [
            create_test_event(
                "Early",
                0,
                EventType::Instantaneous,
                EventAction::Continue,
                0.0,
            ),
            create_test_event(
                "Late",
                1,
                EventType::Instantaneous,
                EventAction::Continue,
                200.0,
            ),
        ];

        let result: Vec<_> = EventQuery::new(events.iter())
            .before(base_epoch - 100.0)
            .collect();

        assert_eq!(result.len(), 0);
    }

    // =========================================================================
    // by_event_type Tests
    // =========================================================================

    #[test]
    fn test_EventQuery_by_event_type_instantaneous() {
        let events = [
            create_test_event(
                "Instant",
                0,
                EventType::Instantaneous,
                EventAction::Continue,
                0.0,
            ),
            create_test_event("Window", 1, EventType::Window, EventAction::Continue, 100.0),
        ];

        let result: Vec<_> = EventQuery::new(events.iter())
            .by_event_type(EventType::Instantaneous)
            .collect();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "Instant");
        assert_eq!(result[0].event_type, EventType::Instantaneous);
    }

    #[test]
    fn test_EventQuery_by_event_type_window() {
        let events = [
            create_test_event(
                "Instant",
                0,
                EventType::Instantaneous,
                EventAction::Continue,
                0.0,
            ),
            create_test_event("Window", 1, EventType::Window, EventAction::Continue, 100.0),
        ];

        let result: Vec<_> = EventQuery::new(events.iter())
            .by_event_type(EventType::Window)
            .collect();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "Window");
        assert_eq!(result[0].event_type, EventType::Window);
    }

    #[test]
    fn test_EventQuery_by_event_type_no_match() {
        let events = [create_test_event(
            "Instant",
            0,
            EventType::Instantaneous,
            EventAction::Continue,
            0.0,
        )];

        let result: Vec<_> = EventQuery::new(events.iter())
            .by_event_type(EventType::Window)
            .collect();

        assert_eq!(result.len(), 0);
    }

    // =========================================================================
    // by_action Tests
    // =========================================================================

    #[test]
    fn test_EventQuery_by_action_stop() {
        let events = [
            create_test_event(
                "Continue Event",
                0,
                EventType::Instantaneous,
                EventAction::Continue,
                0.0,
            ),
            create_test_event(
                "Stop Event",
                1,
                EventType::Instantaneous,
                EventAction::Stop,
                100.0,
            ),
        ];

        let result: Vec<_> = EventQuery::new(events.iter())
            .by_action(EventAction::Stop)
            .collect();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "Stop Event");
        assert_eq!(result[0].action, EventAction::Stop);
    }

    #[test]
    fn test_EventQuery_by_action_continue() {
        let events = [
            create_test_event(
                "Continue Event",
                0,
                EventType::Instantaneous,
                EventAction::Continue,
                0.0,
            ),
            create_test_event(
                "Stop Event",
                1,
                EventType::Instantaneous,
                EventAction::Stop,
                100.0,
            ),
        ];

        let result: Vec<_> = EventQuery::new(events.iter())
            .by_action(EventAction::Continue)
            .collect();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "Continue Event");
        assert_eq!(result[0].action, EventAction::Continue);
    }

    #[test]
    fn test_EventQuery_by_action_no_match() {
        let events = [create_test_event(
            "Continue Event",
            0,
            EventType::Instantaneous,
            EventAction::Continue,
            0.0,
        )];

        let result: Vec<_> = EventQuery::new(events.iter())
            .by_action(EventAction::Stop)
            .collect();

        assert_eq!(result.len(), 0);
    }

    // =========================================================================
    // Combined Filter Tests
    // =========================================================================

    #[test]
    fn test_EventQuery_combined_filters() {
        let base_epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let events = [
            create_test_event(
                "Target Event",
                0,
                EventType::Window,
                EventAction::Stop,
                100.0,
            ),
            create_test_event(
                "Target Event",
                1,
                EventType::Instantaneous,
                EventAction::Continue,
                200.0,
            ),
            create_test_event(
                "Other Event",
                2,
                EventType::Window,
                EventAction::Stop,
                150.0,
            ),
            create_test_event(
                "Target Event",
                3,
                EventType::Window,
                EventAction::Continue,
                50.0,
            ),
        ];

        // Chain multiple filters: name + event_type + action + time
        let result: Vec<_> = EventQuery::new(events.iter())
            .by_name_exact("Target Event")
            .by_event_type(EventType::Window)
            .by_action(EventAction::Stop)
            .after(base_epoch + 75.0)
            .collect();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "Target Event");
        assert_eq!(result[0].detector_index, 0);
        assert_eq!(result[0].event_type, EventType::Window);
        assert_eq!(result[0].action, EventAction::Stop);
    }
}
