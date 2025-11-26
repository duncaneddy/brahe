/*!
 * Event detection for numerical propagation
 *
 * This module provides infrastructure for detecting and handling events during
 * numerical integration, including:
 *
 * - Zero-crossing detection with bisection refinement
 * - Pre-defined event detectors (time, altitude)
 * - Threshold event detectors (continuous value comparisons)
 * - Binary event detectors (boolean condition transitions)
 * - Event callbacks with side effects
 * - Terminal events that stop propagation
 *
 * # Event Detection Process
 *
 * 1. **Monitoring**: During propagation, event detectors are evaluated at each step
 * 2. **Detection**: When a zero-crossing is detected, bisection search refines the event time
 * 3. **Callback**: Optional callbacks can modify state, parameters, or trigger actions
 * 4. **Continuation**: Propagation continues unless event is terminal
 *
 * # Examples
 *
 * ## Time-based Event
 * ```
 * use brahe::events::STimeEvent;
 * use brahe::time::{Epoch, TimeSystem};
 *
 * let target = Epoch::from_jd(2451545.5, TimeSystem::UTC);
 * let event = STimeEvent::<6, 0>::new(target, "Maneuver")
 *     .with_time_tolerance(1e-6);  // Optional: customize tolerance (default: 1e-6 s)
 * ```
 *
 * ## Altitude Event
 * ```
 * use brahe::events::{SAltitudeEvent, EventDirection};
 *
 * // Detect when geodetic altitude drops below 500 km (6D state)
 * // Note: Requires EOP initialization for ECI->ECEF transformation
 * let event = SAltitudeEvent::<6, 0>::new(
 *     500e3,
 *     "Low Altitude Warning",
 *     EventDirection::Decreasing
 * );
 * ```
 *
 * ## Threshold Event
 * ```
 * use brahe::events::{SThresholdEvent, EventDirection};
 * use nalgebra::SVector;
 *
 * // Detect when mass (state[6]) drops below 100 kg
 * let event = SThresholdEvent::<7, 4>::new(
 *     "Low Fuel",
 *     |_t, state: &SVector<f64, 7>, _params| state[6],
 *     100.0,
 *     EventDirection::Decreasing
 * );
 * ```
 *
 * ## Binary Event
 * ```
 * use brahe::events::{SBinaryEvent, EdgeType};
 * use nalgebra::SVector;
 *
 * # fn is_sunlit(_pos: nalgebra::Vector3<f64>) -> bool { true }
 * // Detect entering eclipse (sunlit → shadow)
 * let event = SBinaryEvent::<7, 4>::new(
 *     "Enter Eclipse",
 *     |_t, state: &SVector<f64, 7>, _params| {
 *         // Returns true if sunlit, false if in shadow
 *         is_sunlit(state.fixed_rows::<3>(0).into())
 *     },
 *     EdgeType::FallingEdge
 * );
 * ```
 */

pub mod common;
pub mod detection;
pub mod premade;
pub mod query;
pub mod traits;

// Re-export main types
pub use common::{
    DBinaryEvent, DThresholdEvent, DTimeEvent, SBinaryEvent, SThresholdEvent, STimeEvent,
};
pub use detection::{dscan_for_event, sscan_for_event};
pub use premade::{DAltitudeEvent, SAltitudeEvent};
pub use query::EventQuery;
pub use traits::{
    DDetectedEvent, DEventCallback, DEventDetector, EdgeType, EventAction, EventDirection,
    EventType, SDetectedEvent, SEventCallback, SEventDetector,
};
