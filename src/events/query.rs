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
