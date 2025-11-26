/*!
 * Pre-made event detectors
 *
 * Domain-specific event detectors for altitude and apsis detection.
 */

use super::common::{DThresholdEvent, SThresholdEvent};
use super::traits::{
    DEventCallback, DEventDetector, EventAction, EventDirection, SEventCallback, SEventDetector,
};
use crate::constants::AngleFormat;
use crate::coordinates::position_ecef_to_geodetic;
use crate::frames::position_eci_to_ecef;
use crate::time::Epoch;
use nalgebra::{DVector, SVector};

/// Macro to implement SEventDetector trait delegation for wrapper types (static-sized)
macro_rules! impl_sevent_detector_delegate {
    ($struct:ty, $inner:ident, $s:ident, $p:ident) => {
        impl<const $s: usize, const $p: usize> SEventDetector<$s, $p> for $struct {
            fn evaluate(
                &self,
                t: Epoch,
                state: &SVector<f64, $s>,
                params: Option<&SVector<f64, $p>>,
            ) -> f64 {
                self.$inner.evaluate(t, state, params)
            }

            fn threshold(&self) -> f64 {
                self.$inner.threshold()
            }

            fn name(&self) -> &str {
                self.$inner.name()
            }

            fn direction(&self) -> EventDirection {
                self.$inner.direction()
            }

            fn time_tolerance(&self) -> f64 {
                self.$inner.time_tolerance()
            }

            fn value_tolerance(&self) -> f64 {
                self.$inner.value_tolerance()
            }

            fn step_reduction_factor(&self) -> f64 {
                self.$inner.step_reduction_factor()
            }

            fn callback(&self) -> Option<&SEventCallback<$s, $p>> {
                self.$inner.callback()
            }

            fn action(&self) -> EventAction {
                self.$inner.action()
            }
        }
    };
}

/// Macro to implement DEventDetector trait delegation for wrapper types (dynamic-sized)
macro_rules! impl_devent_detector_delegate {
    ($struct:ty, $inner:ident) => {
        impl DEventDetector for $struct {
            fn evaluate(
                &self,
                t: Epoch,
                state: &DVector<f64>,
                params: Option<&DVector<f64>>,
            ) -> f64 {
                self.$inner.evaluate(t, state, params)
            }

            fn threshold(&self) -> f64 {
                self.$inner.threshold()
            }

            fn name(&self) -> &str {
                self.$inner.name()
            }

            fn direction(&self) -> EventDirection {
                self.$inner.direction()
            }

            fn time_tolerance(&self) -> f64 {
                self.$inner.time_tolerance()
            }

            fn value_tolerance(&self) -> f64 {
                self.$inner.value_tolerance()
            }

            fn step_reduction_factor(&self) -> f64 {
                self.$inner.step_reduction_factor()
            }

            fn callback(&self) -> Option<&DEventCallback> {
                self.$inner.callback()
            }

            fn action(&self) -> EventAction {
                self.$inner.action()
            }
        }
    };
}

/// Altitude-based event detector (for orbital states, static-sized)
///
/// Convenience wrapper around [`SThresholdEvent`] for detecting geodetic altitude
/// crossings. Assumes the first 3 elements of the state vector are ECI position
/// in meters. Transforms to geodetic coordinates to compute altitude above the
/// WGS84 ellipsoid.
///
/// # Examples
/// ```
/// use brahe::events::SAltitudeEvent;
/// use brahe::events::EventDirection;
///
/// // Detect when altitude drops below 500 km (6D state)
/// let event = SAltitudeEvent::<6, 0>::new(
///     500e3,
///     "Low Altitude Warning",
///     EventDirection::Decreasing
/// );
///
/// // Works with extended states (e.g., 7D with mass)
/// let event = SAltitudeEvent::<7, 4>::new(
///     200e3,
///     "Reentry",
///     EventDirection::Decreasing
/// );
/// ```
pub struct SAltitudeEvent<const S: usize, const P: usize> {
    inner: SThresholdEvent<S, P>,
}

impl<const S: usize, const P: usize> SAltitudeEvent<S, P> {
    /// Create new altitude event
    ///
    /// # Arguments
    /// * `threshold_altitude` - Geodetic altitude threshold in meters above WGS84 ellipsoid
    /// * `name` - Event name
    /// * `direction` - Detection direction
    pub fn new(
        threshold_altitude: f64,
        name: impl Into<String>,
        direction: EventDirection,
    ) -> Self {
        let altitude_fn = |t: Epoch, state: &SVector<f64, S>, _params: Option<&SVector<f64, P>>| {
            // Extract ECI position (first 3 elements)
            let r_eci = state.fixed_rows::<3>(0).into_owned();

            // Transform ECI -> ECEF -> Geodetic
            let r_ecef = position_eci_to_ecef(t, r_eci);
            let geodetic = position_ecef_to_geodetic(r_ecef, AngleFormat::Radians);

            // Return geodetic altitude (index 2: [lon, lat, alt])
            geodetic[2]
        };

        Self {
            inner: SThresholdEvent::new(name, altitude_fn, threshold_altitude, direction),
        }
    }

    /// Set instance number for display name
    pub fn with_instance(mut self, instance: usize) -> Self {
        self.inner = self.inner.with_instance(instance);
        self
    }

    /// Set custom tolerances for event detection
    pub fn with_tolerances(mut self, time_tol: f64, value_tol: f64) -> Self {
        self.inner = self.inner.with_tolerances(time_tol, value_tol);
        self
    }

    /// Set custom step reduction factor for bisection search
    ///
    /// When a zero-crossing is detected, the new step size is set to this
    /// factor times the bracket width. Default: 0.2
    pub fn with_step_reduction_factor(mut self, factor: f64) -> Self {
        self.inner = self.inner.with_step_reduction_factor(factor);
        self
    }

    /// Set event callback
    pub fn with_callback(mut self, callback: SEventCallback<S, P>) -> Self {
        self.inner = self.inner.with_callback(callback);
        self
    }

    /// Mark as terminal event (stops propagation)
    pub fn is_terminal(mut self) -> Self {
        self.inner = self.inner.is_terminal();
        self
    }
}

// Delegate EventDetector trait implementation to inner SThresholdEvent
impl_sevent_detector_delegate!(SAltitudeEvent<S, P>, inner, S, P);

/// Dynamic-sized altitude event detector
///
/// See [`SAltitudeEvent`] for details. This version works with dynamic-sized
/// state vectors. Assumes first 3 elements are ECI position in meters.
pub struct DAltitudeEvent {
    inner: DThresholdEvent,
}

impl DAltitudeEvent {
    /// Create new altitude event
    ///
    /// # Arguments
    /// * `threshold_altitude` - Geodetic altitude threshold in meters above WGS84 ellipsoid
    /// * `name` - Event name
    /// * `direction` - Detection direction
    pub fn new(
        threshold_altitude: f64,
        name: impl Into<String>,
        direction: EventDirection,
    ) -> Self {
        let altitude_fn = |t: Epoch, state: &DVector<f64>, _params: Option<&DVector<f64>>| {
            // Extract ECI position (first 3 elements) and convert to Vector3
            use nalgebra::Vector3;
            let r_eci = Vector3::new(state[0], state[1], state[2]);

            // Transform ECI -> ECEF -> Geodetic
            let r_ecef = position_eci_to_ecef(t, r_eci);
            let geodetic = position_ecef_to_geodetic(r_ecef, AngleFormat::Radians);

            // Return geodetic altitude (index 2: [lon, lat, alt])
            geodetic[2]
        };

        Self {
            inner: DThresholdEvent::new(name, altitude_fn, threshold_altitude, direction),
        }
    }

    /// Set instance number for display name
    pub fn with_instance(mut self, instance: usize) -> Self {
        self.inner = self.inner.with_instance(instance);
        self
    }

    /// Set custom tolerances for event detection
    pub fn with_tolerances(mut self, time_tol: f64, value_tol: f64) -> Self {
        self.inner = self.inner.with_tolerances(time_tol, value_tol);
        self
    }

    /// Set custom step reduction factor for bisection search
    ///
    /// When a zero-crossing is detected, the new step size is set to this
    /// factor times the bracket width. Default: 0.2
    pub fn with_step_reduction_factor(mut self, factor: f64) -> Self {
        self.inner = self.inner.with_step_reduction_factor(factor);
        self
    }

    /// Set event callback
    pub fn with_callback(mut self, callback: DEventCallback) -> Self {
        self.inner = self.inner.with_callback(callback);
        self
    }

    /// Mark as terminal event
    pub fn is_terminal(mut self) -> Self {
        self.inner = self.inner.is_terminal();
        self
    }
}

// Delegate EventDetectorD trait implementation to inner DThresholdEvent
impl_devent_detector_delegate!(DAltitudeEvent, inner);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::R_EARTH;
    use crate::time::TimeSystem;
    use nalgebra::Vector6;

    #[test]
    fn test_altitude_event() {
        use crate::utils::testing::setup_global_test_eop;

        // Initialize EOP for frame transformations
        setup_global_test_eop();

        let threshold = 500e3; // 500 km geodetic altitude
        let event = SAltitudeEvent::<6, 0>::new(threshold, "Altitude Test", EventDirection::Any);

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);

        // Low altitude state (below threshold) - ECI position
        // Using radius ~6878 km from center = ~500 km altitude
        let state_low = Vector6::new(6000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        let val_low = event.evaluate(epoch, &state_low, None);
        // This should be negative (altitude < threshold)
        assert!(val_low < 0.0);

        // High altitude state (above threshold) - ECI position
        // Using radius ~7378 km from center = ~1000 km altitude
        let state_high = Vector6::new(R_EARTH + 1000e3, 0.0, 0.0, 0.0, 7.0e3, 0.0);
        let val_high = event.evaluate(epoch, &state_high, None);
        // This should be positive (altitude > threshold)
        assert!(val_high > 0.0);
    }
}
