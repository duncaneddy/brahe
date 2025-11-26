/*!
 * Event detection algorithms
 *
 * Implements bisection search for finding precise event times during
 * numerical propagation.
 */

use super::traits::{
    DDetectedEvent, DEventDetector, EventDirection, SDetectedEvent, SEventDetector,
};
use crate::time::Epoch;
use nalgebra::{DVector, SVector};

/// Direction for bisection search
///
/// Used internally by the bisection algorithm to step forward or backward in time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepDirection {
    /// Step forward in time
    Forward,
    /// Step backward in time
    Backward,
}

/// Find event time using bisection search (static-sized)
///
/// Uses recursive bisection to refine the event time to within the specified
/// tolerance. The algorithm steps in the given direction until the event
/// function changes sign, then recursively bisects the interval.
///
/// # Arguments
/// * `detector` - Event detector to evaluate
/// * `state_fn` - Function to get state at a given time (from integrator)
/// * `time` - Starting time
/// * `direction` - Step direction (forward/backward)
/// * `step` - Step size (seconds)
/// * `current_value` - Event function value at starting time
/// * `params` - Optional parameter vector
/// * `search_start_time` - Start of search window (events before this are not detected)
/// * `search_end_time` - End of search window (events after this are not detected)
///
/// # Returns
/// Event time and state, or None if no event found within search window
#[allow(clippy::too_many_arguments)]
pub(crate) fn bisection_search<const S: usize, const P: usize, F>(
    detector: &dyn SEventDetector<S, P>,
    state_fn: &F,
    time: Epoch,
    direction: StepDirection,
    step: f64,
    mut current_crossing: f64,
    params: Option<&SVector<f64, P>>,
    search_start_time: Epoch,
    search_end_time: Epoch,
) -> Option<(Epoch, SVector<f64, S>)>
where
    F: Fn(Epoch) -> SVector<f64, S>,
{
    let mut current_time = time;
    let mut steps_taken = 0;
    const MAX_STEPS: usize = 10000; // Safety limit

    let time_tol = detector.time_tolerance();
    let value_tol = detector.value_tolerance();
    let dir = detector.direction();
    let threshold = detector.threshold();

    // Check if we're already at the event
    if current_crossing.abs() < value_tol {
        let state = state_fn(time);
        return Some((time, state));
    }

    loop {
        // Take a step
        let next_time = match direction {
            StepDirection::Forward => current_time + step,
            StepDirection::Backward => current_time - step,
        };

        // Check search window bounds
        if next_time < search_start_time || next_time > search_end_time || steps_taken >= MAX_STEPS
        {
            return None;
        }

        current_time = next_time;
        steps_taken += 1;

        // Evaluate event function at new time
        let state = state_fn(current_time);
        let next_value = detector.evaluate(current_time, &state, params);
        let next_crossing = next_value - threshold;

        // Check for zero crossing based on direction
        let crossed = match dir {
            EventDirection::Increasing => current_crossing < 0.0 && next_crossing >= 0.0,
            EventDirection::Decreasing => current_crossing > 0.0 && next_crossing <= 0.0,
            EventDirection::Any => {
                (current_crossing < 0.0 && next_crossing >= 0.0)
                    || (current_crossing > 0.0 && next_crossing <= 0.0)
            }
        };

        if crossed || next_crossing.abs() < value_tol {
            // Found event or close enough to zero
            if step < time_tol {
                return Some((current_time, state));
            } else {
                // Recurse with smaller step
                let new_direction = match direction {
                    StepDirection::Forward => StepDirection::Backward,
                    StepDirection::Backward => StepDirection::Forward,
                };
                return bisection_search(
                    detector,
                    state_fn,
                    current_time,
                    new_direction,
                    step / 2.0,
                    next_crossing,
                    params,
                    search_start_time,
                    search_end_time,
                );
            }
        }

        // Update current_crossing for next iteration
        current_crossing = next_crossing;
    }
}

/// Find event time using bisection search (dynamic-sized)
#[allow(clippy::too_many_arguments)]
pub(crate) fn bisection_search_d<F>(
    detector: &dyn DEventDetector,
    state_fn: &F,
    time: Epoch,
    direction: StepDirection,
    step: f64,
    mut current_crossing: f64,
    params: Option<&DVector<f64>>,
    search_start_time: Epoch,
    search_end_time: Epoch,
) -> Option<(Epoch, DVector<f64>)>
where
    F: Fn(Epoch) -> DVector<f64>,
{
    let mut current_time = time;
    let mut steps_taken = 0;
    const MAX_STEPS: usize = 10000;

    let time_tol = detector.time_tolerance();
    let value_tol = detector.value_tolerance();
    let dir = detector.direction();
    let threshold = detector.threshold();

    // Check if we're already at the event
    if current_crossing.abs() < value_tol {
        let state = state_fn(time);
        return Some((time, state));
    }

    loop {
        let next_time = match direction {
            StepDirection::Forward => current_time + step,
            StepDirection::Backward => current_time - step,
        };

        if next_time < search_start_time || next_time > search_end_time || steps_taken >= MAX_STEPS
        {
            return None;
        }

        current_time = next_time;
        steps_taken += 1;

        let state = state_fn(current_time);
        let next_value = detector.evaluate(current_time, &state, params);
        let next_crossing = next_value - threshold;

        let crossed = match dir {
            EventDirection::Increasing => current_crossing < 0.0 && next_crossing >= 0.0,
            EventDirection::Decreasing => current_crossing > 0.0 && next_crossing <= 0.0,
            EventDirection::Any => {
                (current_crossing < 0.0 && next_crossing >= 0.0)
                    || (current_crossing > 0.0 && next_crossing <= 0.0)
            }
        };

        if crossed || next_crossing.abs() < value_tol {
            if step < time_tol {
                return Some((current_time, state));
            } else {
                let new_direction = match direction {
                    StepDirection::Forward => StepDirection::Backward,
                    StepDirection::Backward => StepDirection::Forward,
                };
                return bisection_search_d(
                    detector,
                    state_fn,
                    current_time,
                    new_direction,
                    step / 2.0,
                    next_crossing,
                    params,
                    search_start_time,
                    search_end_time,
                );
            }
        }

        // Update current_crossing for next iteration
        current_crossing = next_crossing;
    }
}

/// Scan for events in a time interval (static-sized)
///
/// Monitors event function during propagation and triggers bisection search
/// when a potential event is detected.
///
/// # Arguments
/// * `detector` - Event detector
/// * `state_fn` - Function to get state at time
/// * `prev_time` - Previous time
/// * `current_time` - Current time
/// * `prev_state` - Previous state
/// * `current_state` - Current state
/// * `params` - Optional parameters
///
/// # Returns
/// Detected event if found
#[allow(clippy::too_many_arguments)]
pub fn sscan_for_event<const S: usize, const P: usize, F>(
    detector: &dyn SEventDetector<S, P>,
    detector_index: usize,
    state_fn: &F,
    prev_time: Epoch,
    current_time: Epoch,
    prev_state: &SVector<f64, S>,
    current_state: &SVector<f64, S>,
    params: Option<&SVector<f64, P>>,
) -> Option<SDetectedEvent<S>>
where
    F: Fn(Epoch) -> SVector<f64, S>,
{
    // Evaluate monitored values at both times
    let prev_value = detector.evaluate(prev_time, prev_state, params);
    let current_value = detector.evaluate(current_time, current_state, params);
    let threshold = detector.threshold();

    // Compute zero-crossings
    let prev_crossing = prev_value - threshold;
    let current_crossing = current_value - threshold;

    // Check for zero crossing
    let dir = detector.direction();
    let crossed = match dir {
        EventDirection::Increasing => prev_crossing < 0.0 && current_crossing >= 0.0,
        EventDirection::Decreasing => prev_crossing > 0.0 && current_crossing <= 0.0,
        EventDirection::Any => {
            (prev_crossing < 0.0 && current_crossing >= 0.0)
                || (prev_crossing > 0.0 && current_crossing <= 0.0)
        }
    };

    if !crossed && current_crossing.abs() > detector.value_tolerance() {
        return None;
    }

    // Found potential event - refine using bisection
    let dt = (current_time - prev_time).abs();
    let initial_step = dt / 4.0; // Start with quarter of interval

    let result = bisection_search(
        detector,
        state_fn,
        prev_time,
        StepDirection::Forward,
        initial_step,
        prev_crossing,
        params,
        prev_time,
        current_time,
    );

    result.map(|(event_time, event_state)| {
        let event_value = detector.evaluate(event_time, &event_state, params);
        let action = detector.action();
        let event_type = detector.event_type();

        SDetectedEvent {
            window_open: event_time,
            window_close: event_time, // Same for instantaneous events
            entry_state: event_state,
            exit_state: event_state, // Same for instantaneous events
            value: event_value,
            name: detector.name().to_string(),
            action,
            event_type,
            detector_index,
        }
    })
}

/// Scan for events (dynamic-sized)
#[allow(clippy::too_many_arguments)]
pub fn dscan_for_event<F>(
    detector: &dyn DEventDetector,
    detector_index: usize,
    state_fn: &F,
    prev_time: Epoch,
    current_time: Epoch,
    prev_state: &DVector<f64>,
    current_state: &DVector<f64>,
    params: Option<&DVector<f64>>,
) -> Option<DDetectedEvent>
where
    F: Fn(Epoch) -> DVector<f64>,
{
    let prev_value = detector.evaluate(prev_time, prev_state, params);
    let current_value = detector.evaluate(current_time, current_state, params);
    let threshold = detector.threshold();

    let prev_crossing = prev_value - threshold;
    let current_crossing = current_value - threshold;

    let dir = detector.direction();
    let crossed = match dir {
        EventDirection::Increasing => prev_crossing < 0.0 && current_crossing >= 0.0,
        EventDirection::Decreasing => prev_crossing > 0.0 && current_crossing <= 0.0,
        EventDirection::Any => {
            (prev_crossing < 0.0 && current_crossing >= 0.0)
                || (prev_crossing > 0.0 && current_crossing <= 0.0)
        }
    };

    if !crossed && current_crossing.abs() > detector.value_tolerance() {
        return None;
    }

    let dt = (current_time - prev_time).abs();
    let initial_step = dt / 4.0;

    let result = bisection_search_d(
        detector,
        state_fn,
        prev_time,
        StepDirection::Forward,
        initial_step,
        prev_crossing,
        params,
        prev_time,
        current_time,
    );

    result.map(|(event_time, event_state)| {
        let event_value = detector.evaluate(event_time, &event_state, params);
        let action = detector.action();
        let event_type = detector.event_type();

        DDetectedEvent {
            window_open: event_time,
            window_close: event_time,
            entry_state: event_state.clone(),
            exit_state: event_state,
            value: event_value,
            name: detector.name().to_string(),
            action,
            event_type,
            detector_index,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::TimeSystem;
    use nalgebra::Vector6;

    struct SimpleTimeEvent {
        target_time: Epoch,
        name: String,
    }

    impl SEventDetector<6, 0> for SimpleTimeEvent {
        fn evaluate(
            &self,
            t: Epoch,
            _state: &SVector<f64, 6>,
            _params: Option<&SVector<f64, 0>>,
        ) -> f64 {
            t - self.target_time // Returns signed time difference in seconds
        }

        fn threshold(&self) -> f64 {
            0.0 // Event occurs when time reaches target
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    #[test]
    fn test_bisection_search_time_event() {
        let start_epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let target_epoch = start_epoch + 100.0; // 100 seconds later

        let detector = SimpleTimeEvent {
            target_time: target_epoch,
            name: "Test Time Event".to_string(),
        };

        // Simple state function that returns constant state
        let state_fn = |_t: Epoch| Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);

        let initial_value = detector.evaluate(start_epoch, &state_fn(start_epoch), None);
        let initial_crossing = initial_value - detector.threshold();

        let result = bisection_search(
            &detector,
            &state_fn,
            start_epoch,
            StepDirection::Forward,
            10.0, // 10 second initial step
            initial_crossing,
            None,
            start_epoch,
            start_epoch + 200.0,
        );

        assert!(result.is_some());
        let (event_time, _) = result.unwrap();

        // Should find event within tolerance
        let time_error = (event_time - target_epoch).abs();
        assert!(time_error < detector.time_tolerance());
    }

    #[test]
    fn test_scan_for_event() {
        let start_epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let target_epoch = start_epoch + 50.0;

        let detector = SimpleTimeEvent {
            target_time: target_epoch,
            name: "Scan Test".to_string(),
        };

        let state_fn = |_t: Epoch| Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);

        let prev_time = start_epoch;
        let current_time = start_epoch + 100.0; // Step over the event
        let prev_state = state_fn(prev_time);
        let current_state = state_fn(current_time);

        let result = sscan_for_event(
            &detector,
            0,
            &state_fn,
            prev_time,
            current_time,
            &prev_state,
            &current_state,
            None,
        );

        assert!(result.is_some());
        let event = result.unwrap();

        let time_error = (event.window_open - target_epoch).abs();
        assert!(time_error < detector.time_tolerance());
        assert_eq!(event.name, "Scan Test");
        assert_eq!(event.detector_index, 0);
    }
}
