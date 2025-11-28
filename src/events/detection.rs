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
/// Uses bracketing bisection to refine the event time to within the specified
/// tolerance. The algorithm maintains explicit bracket bounds around the event
/// and narrows them on each crossing detection until the bracket width is less
/// than or equal to the time tolerance.
///
/// # Arguments
/// * `detector` - Event detector to evaluate
/// * `state_fn` - Function to get state at a given time (from integrator)
/// * `start_time` - Starting time for this search iteration
/// * `direction` - Step direction (forward/backward)
/// * `step` - Step size (seconds)
/// * `start_crossing` - Event function value at starting time (relative to value)
/// * `params` - Optional parameter vector
/// * `bracket_low` - Lower bound of search bracket (earlier time)
/// * `bracket_high` - Upper bound of search bracket (later time)
///
/// # Returns
/// Event time and state, or None if no event found within search window
#[allow(clippy::too_many_arguments)]
pub(crate) fn bisection_search<const S: usize, const P: usize, F>(
    detector: &dyn SEventDetector<S, P>,
    state_fn: &F,
    start_time: Epoch,
    direction: StepDirection,
    step: f64,
    start_crossing: f64,
    params: Option<&SVector<f64, P>>,
    bracket_low: Epoch,
    bracket_high: Epoch,
) -> Option<(Epoch, SVector<f64, S>)>
where
    F: Fn(Epoch) -> SVector<f64, S>,
{
    let time_tol = detector.time_tolerance();
    let value_tol = detector.value_tolerance();
    let step_factor = detector.step_reduction_factor();
    let target_value = detector.target_value();

    // TERMINATION: Bracket is tight enough
    let bracket_width = (bracket_high - bracket_low).abs();
    if bracket_width <= time_tol {
        // Return the midpoint of the bracket
        let mid_time = bracket_low + bracket_width / 2.0;
        let mid_state = state_fn(mid_time);
        let mid_crossing = detector.evaluate(mid_time, &mid_state, params) - target_value;

        // Validate: prefer midpoint if within tolerance
        if mid_crossing.abs() <= value_tol {
            return Some((mid_time, mid_state));
        }

        // Try endpoints
        let low_state = state_fn(bracket_low);
        let low_crossing = detector.evaluate(bracket_low, &low_state, params) - target_value;
        if low_crossing.abs() <= value_tol {
            return Some((bracket_low, low_state));
        }

        let high_state = state_fn(bracket_high);
        let high_crossing = detector.evaluate(bracket_high, &high_state, params) - target_value;
        if high_crossing.abs() <= value_tol {
            return Some((bracket_high, high_state));
        }

        // Fallback: return midpoint (guaranteed crossing exists between bounds)
        return Some((mid_time, mid_state));
    }

    // Already at event?
    if start_crossing.abs() < value_tol {
        return Some((start_time, state_fn(start_time)));
    }

    let mut current_time = start_time;
    let mut current_crossing = start_crossing;
    let mut steps_taken = 0;
    const MAX_STEPS: usize = 10000;

    loop {
        // Calculate next time based on step direction
        let mut next_time = match direction {
            StepDirection::Forward => current_time + step,
            StepDirection::Backward => current_time - step,
        };

        // Clamp to bracket bounds
        if next_time < bracket_low {
            next_time = bracket_low;
        } else if next_time > bracket_high {
            next_time = bracket_high;
        }

        // No progress possible - validate and return
        if next_time == current_time {
            let state = state_fn(current_time);
            let crossing = detector.evaluate(current_time, &state, params) - target_value;
            if crossing.abs() <= value_tol {
                return Some((current_time, state));
            }
            return None;
        }

        steps_taken += 1;
        if steps_taken >= MAX_STEPS {
            return None;
        }

        // Evaluate at next time
        let next_state = state_fn(next_time);
        let next_crossing = detector.evaluate(next_time, &next_state, params) - target_value;

        // During bisection refinement, we're looking for ANY sign change
        // because we know there's a crossing in the bracket. The original
        // detection direction was already verified by sscan_for_event.
        let crossed = (current_crossing < 0.0 && next_crossing >= 0.0)
            || (current_crossing > 0.0 && next_crossing <= 0.0);

        if crossed || next_crossing.abs() < value_tol {
            // Found crossing! Update bracket based on step direction
            // The crossing is between current_time and next_time
            let (new_low, new_high) = match direction {
                StepDirection::Forward => {
                    // Stepped forward: event between current_time and next_time
                    (current_time, next_time)
                }
                StepDirection::Backward => {
                    // Stepped backward: event between next_time and current_time
                    (next_time, current_time)
                }
            };

            // Calculate new step size as fraction of new bracket width
            let new_bracket_width = (new_high - new_low).abs();
            let new_step = step_factor * new_bracket_width;

            // Reverse direction and continue search from the crossing point
            let new_direction = match direction {
                StepDirection::Forward => StepDirection::Backward,
                StepDirection::Backward => StepDirection::Forward,
            };

            // Recurse with narrowed bracket
            return bisection_search(
                detector,
                state_fn,
                next_time,
                new_direction,
                new_step,
                next_crossing,
                params,
                new_low,
                new_high,
            );
        }

        // No crossing yet, continue stepping
        current_time = next_time;
        current_crossing = next_crossing;
    }
}

/// Find event time using bisection search (dynamic-sized)
///
/// Uses bracketing bisection to refine the event time to within the specified
/// tolerance. See `bisection_search` for algorithm details.
#[allow(clippy::too_many_arguments)]
pub(crate) fn bisection_search_d<F>(
    detector: &dyn DEventDetector,
    state_fn: &F,
    start_time: Epoch,
    direction: StepDirection,
    step: f64,
    start_crossing: f64,
    params: Option<&DVector<f64>>,
    bracket_low: Epoch,
    bracket_high: Epoch,
) -> Option<(Epoch, DVector<f64>)>
where
    F: Fn(Epoch) -> DVector<f64>,
{
    let time_tol = detector.time_tolerance();
    let value_tol = detector.value_tolerance();
    let step_factor = detector.step_reduction_factor();
    let target_value = detector.target_value();

    // TERMINATION: Bracket is tight enough
    let bracket_width = (bracket_high - bracket_low).abs();
    if bracket_width <= time_tol {
        // Return the midpoint of the bracket
        let mid_time = bracket_low + bracket_width / 2.0;
        let mid_state = state_fn(mid_time);
        let mid_crossing = detector.evaluate(mid_time, &mid_state, params) - target_value;

        // Validate: prefer midpoint if within tolerance
        if mid_crossing.abs() <= value_tol {
            return Some((mid_time, mid_state));
        }

        // Try endpoints
        let low_state = state_fn(bracket_low);
        let low_crossing = detector.evaluate(bracket_low, &low_state, params) - target_value;
        if low_crossing.abs() <= value_tol {
            return Some((bracket_low, low_state));
        }

        let high_state = state_fn(bracket_high);
        let high_crossing = detector.evaluate(bracket_high, &high_state, params) - target_value;
        if high_crossing.abs() <= value_tol {
            return Some((bracket_high, high_state));
        }

        // Fallback: return midpoint (guaranteed crossing exists between bounds)
        return Some((mid_time, mid_state));
    }

    // Already at event?
    if start_crossing.abs() < value_tol {
        return Some((start_time, state_fn(start_time)));
    }

    let mut current_time = start_time;
    let mut current_crossing = start_crossing;
    let mut steps_taken = 0;
    const MAX_STEPS: usize = 10000;

    loop {
        // Calculate next time based on step direction
        let mut next_time = match direction {
            StepDirection::Forward => current_time + step,
            StepDirection::Backward => current_time - step,
        };

        // Clamp to bracket bounds
        if next_time < bracket_low {
            next_time = bracket_low;
        } else if next_time > bracket_high {
            next_time = bracket_high;
        }

        // No progress possible - validate and return
        if next_time == current_time {
            let state = state_fn(current_time);
            let crossing = detector.evaluate(current_time, &state, params) - target_value;
            if crossing.abs() <= value_tol {
                return Some((current_time, state));
            }
            return None;
        }

        steps_taken += 1;
        if steps_taken >= MAX_STEPS {
            return None;
        }

        // Evaluate at next time
        let next_state = state_fn(next_time);
        let next_crossing = detector.evaluate(next_time, &next_state, params) - target_value;

        // During bisection refinement, we're looking for ANY sign change
        // because we know there's a crossing in the bracket. The original
        // detection direction was already verified by dscan_for_event.
        let crossed = (current_crossing < 0.0 && next_crossing >= 0.0)
            || (current_crossing > 0.0 && next_crossing <= 0.0);

        if crossed || next_crossing.abs() < value_tol {
            // Found crossing! Update bracket based on step direction
            // The crossing is between current_time and next_time
            let (new_low, new_high) = match direction {
                StepDirection::Forward => {
                    // Stepped forward: event between current_time and next_time
                    (current_time, next_time)
                }
                StepDirection::Backward => {
                    // Stepped backward: event between next_time and current_time
                    (next_time, current_time)
                }
            };

            // Calculate new step size as fraction of new bracket width
            let new_bracket_width = (new_high - new_low).abs();
            let new_step = step_factor * new_bracket_width;

            // Reverse direction and continue search from the crossing point
            let new_direction = match direction {
                StepDirection::Forward => StepDirection::Backward,
                StepDirection::Backward => StepDirection::Forward,
            };

            // Recurse with narrowed bracket
            return bisection_search_d(
                detector,
                state_fn,
                next_time,
                new_direction,
                new_step,
                next_crossing,
                params,
                new_low,
                new_high,
            );
        }

        // No crossing yet, continue stepping
        current_time = next_time;
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
    let target_value = detector.target_value();

    // Compute zero-crossings
    let prev_crossing = prev_value - target_value;
    let current_crossing = current_value - target_value;

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
    let target_value = detector.target_value();

    let prev_crossing = prev_value - target_value;
    let current_crossing = current_value - target_value;

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

        fn target_value(&self) -> f64 {
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
        let initial_crossing = initial_value - detector.target_value();

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

    /// Configurable time event with custom step reduction factor
    struct ConfigurableTimeEvent {
        target_time: Epoch,
        name: String,
        step_factor: f64,
        time_tol: f64,
    }

    impl SEventDetector<6, 0> for ConfigurableTimeEvent {
        fn evaluate(
            &self,
            t: Epoch,
            _state: &SVector<f64, 6>,
            _params: Option<&SVector<f64, 0>>,
        ) -> f64 {
            t - self.target_time
        }

        fn target_value(&self) -> f64 {
            0.0
        }

        fn name(&self) -> &str {
            &self.name
        }

        fn step_reduction_factor(&self) -> f64 {
            self.step_factor
        }

        fn time_tolerance(&self) -> f64 {
            self.time_tol
        }
    }

    #[test]
    fn test_step_reduction_factor_is_used() {
        // Test that a custom step reduction factor is used in the bisection
        let start_epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let target_epoch = start_epoch + 100.0;

        // Very aggressive step factor (0.5) should still converge
        let detector = ConfigurableTimeEvent {
            target_time: target_epoch,
            name: "Aggressive Step".to_string(),
            step_factor: 0.5,
            time_tol: 1e-3,
        };

        let state_fn = |_t: Epoch| Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);

        let initial_value = detector.evaluate(start_epoch, &state_fn(start_epoch), None);
        let initial_crossing = initial_value - detector.target_value();

        let result = bisection_search(
            &detector,
            &state_fn,
            start_epoch,
            StepDirection::Forward,
            10.0,
            initial_crossing,
            None,
            start_epoch,
            start_epoch + 200.0,
        );

        assert!(result.is_some());
        let (event_time, _) = result.unwrap();
        let time_error = (event_time - target_epoch).abs();
        assert!(time_error < detector.time_tolerance());
    }

    #[test]
    fn test_bracket_termination() {
        // Test that search terminates when bracket is tight enough
        let start_epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let target_epoch = start_epoch + 50.0;

        let detector = ConfigurableTimeEvent {
            target_time: target_epoch,
            name: "Tight Bracket".to_string(),
            step_factor: 0.2,
            time_tol: 1e-6, // Very tight tolerance
        };

        let state_fn = |_t: Epoch| Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);

        let initial_value = detector.evaluate(start_epoch, &state_fn(start_epoch), None);
        let initial_crossing = initial_value - detector.target_value();

        let result = bisection_search(
            &detector,
            &state_fn,
            start_epoch,
            StepDirection::Forward,
            5.0,
            initial_crossing,
            None,
            start_epoch,
            start_epoch + 100.0,
        );

        assert!(result.is_some());
        let (event_time, _) = result.unwrap();
        let time_error = (event_time - target_epoch).abs();

        // Should be within the tight tolerance
        assert!(time_error < detector.time_tolerance());
    }

    #[test]
    fn test_event_near_bracket_boundary() {
        // Test finding an event very close to the end of the search window
        let start_epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let target_epoch = start_epoch + 99.5; // Near the end of the 100s window

        let detector = SimpleTimeEvent {
            target_time: target_epoch,
            name: "Near Boundary".to_string(),
        };

        let state_fn = |_t: Epoch| Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);

        let initial_value = detector.evaluate(start_epoch, &state_fn(start_epoch), None);
        let initial_crossing = initial_value - detector.target_value();

        let result = bisection_search(
            &detector,
            &state_fn,
            start_epoch,
            StepDirection::Forward,
            10.0,
            initial_crossing,
            None,
            start_epoch,
            start_epoch + 100.0,
        );

        assert!(result.is_some());
        let (event_time, _) = result.unwrap();
        let time_error = (event_time - target_epoch).abs();
        assert!(time_error < detector.time_tolerance());
    }

    // =========================================================================
    // Dynamic-sized value event tests
    // =========================================================================

    use crate::events::DValueEvent;

    #[test]
    fn test_dscan_value_event_position_crossing() {
        // Simulate SHO: position crosses from positive to negative
        // x_prev = [1.0, 0.0] (position=1, velocity=0)
        // x_new = [-0.4, -0.9] (position=-0.4, velocity=-0.9)
        let start_epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let end_epoch = start_epoch + 2.0; // 2 seconds

        let x_prev = DVector::from_vec(vec![1.0, 0.0]);
        let x_new = DVector::from_vec(vec![-0.4, -0.9]);

        // Clone for closure
        let x_prev_clone = x_prev.clone();
        let x_new_clone = x_new.clone();

        // Linear interpolation for state function
        let state_fn = move |t: Epoch| -> DVector<f64> {
            let alpha = (t - start_epoch) / (end_epoch - start_epoch);
            &x_prev_clone + (&x_new_clone - &x_prev_clone) * alpha
        };

        // Value event: detect when position (state[0]) crosses 0
        let value_fn =
            |_epoch: Epoch, state: &DVector<f64>, _params: Option<&DVector<f64>>| state[0];
        let detector = DValueEvent::new(
            "PositionCrossing",
            value_fn,
            0.0,                        // target value
            EventDirection::Decreasing, // detect positive -> negative crossing
        );

        let result = dscan_for_event(
            &detector,
            0,
            &state_fn,
            start_epoch,
            end_epoch,
            &x_prev,
            &x_new,
            None,
        );

        assert!(
            result.is_some(),
            "Should detect position crossing from 1.0 to -0.4"
        );

        let event = result.unwrap();
        // Position should be close to 0 at event time
        assert!(
            event.value.abs() < 0.01,
            "Event value should be close to target value 0, got {}",
            event.value
        );
    }

    #[test]
    fn test_dscan_value_event_no_crossing() {
        // Position stays positive - no crossing should be detected
        let start_epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let end_epoch = start_epoch + 2.0;

        let x_prev = DVector::from_vec(vec![1.0, 0.0]);
        let x_new = DVector::from_vec(vec![0.5, -0.5]); // Still positive

        let state_fn = |t: Epoch| -> DVector<f64> {
            let alpha = (t - start_epoch) / (end_epoch - start_epoch);
            &x_prev + (&x_new - &x_prev) * alpha
        };

        let value_fn =
            |_epoch: Epoch, state: &DVector<f64>, _params: Option<&DVector<f64>>| state[0];
        let detector = DValueEvent::new(
            "PositionCrossing",
            value_fn,
            0.0,
            EventDirection::Decreasing,
        );

        let result = dscan_for_event(
            &detector,
            0,
            &state_fn,
            start_epoch,
            end_epoch,
            &x_prev,
            &x_new,
            None,
        );

        assert!(
            result.is_none(),
            "Should not detect crossing when position stays positive"
        );
    }

    #[test]
    fn test_dscan_value_event_increasing_direction() {
        // Position crosses from negative to positive (increasing)
        let start_epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let end_epoch = start_epoch + 2.0;

        let x_prev = DVector::from_vec(vec![-0.4, 0.9]);
        let x_new = DVector::from_vec(vec![1.0, 0.0]); // Position goes positive

        let state_fn = |t: Epoch| -> DVector<f64> {
            let alpha = (t - start_epoch) / (end_epoch - start_epoch);
            &x_prev + (&x_new - &x_prev) * alpha
        };

        let value_fn =
            |_epoch: Epoch, state: &DVector<f64>, _params: Option<&DVector<f64>>| state[0];
        let detector = DValueEvent::new(
            "PositionCrossing",
            value_fn,
            0.0,
            EventDirection::Increasing, // detect negative -> positive
        );

        let result = dscan_for_event(
            &detector,
            0,
            &state_fn,
            start_epoch,
            end_epoch,
            &x_prev,
            &x_new,
            None,
        );

        assert!(
            result.is_some(),
            "Should detect increasing crossing from -0.4 to 1.0"
        );
    }
}
