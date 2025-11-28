/*!
 * Pre-made event detectors
 *
 * Domain-specific event detectors for altitude, orbital elements, and eclipse detection.
 */

use super::common::{DBinaryEvent, DValueEvent, SBinaryEvent, SValueEvent};
use super::traits::{
    DEventCallback, DEventDetector, EdgeType, EventAction, EventDirection, SEventCallback,
    SEventDetector,
};
use crate::constants::AngleFormat;
use crate::coordinates::{position_ecef_to_geodetic, state_eci_to_koe};
use crate::frames::position_eci_to_ecef;
use crate::orbit_dynamics::{eclipse_conical, sun_position, sun_position_de};
use crate::orbits::{anomaly_mean_to_eccentric, anomaly_mean_to_true};
use crate::propagators::EphemerisSource;
use crate::time::Epoch;
use nalgebra::{DVector, SVector, Vector3, Vector6};

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

            fn target_value(&self) -> f64 {
                self.$inner.target_value()
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

            fn target_value(&self) -> f64 {
                self.$inner.target_value()
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
/// Convenience wrapper around [`SValueEvent`] for detecting geodetic altitude
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
    inner: SValueEvent<S, P>,
}

impl<const S: usize, const P: usize> SAltitudeEvent<S, P> {
    /// Create new altitude event
    ///
    /// # Arguments
    /// * `value_altitude` - Geodetic altitude value in meters above WGS84 ellipsoid
    /// * `name` - Event name
    /// * `direction` - Detection direction
    pub fn new(value_altitude: f64, name: impl Into<String>, direction: EventDirection) -> Self {
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
            inner: SValueEvent::new(name, altitude_fn, value_altitude, direction),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

// Delegate EventDetector trait implementation to inner SValueEvent
impl_sevent_detector_delegate!(SAltitudeEvent<S, P>, inner, S, P);

/// Dynamic-sized altitude event detector
///
/// See [`SAltitudeEvent`] for details. This version works with dynamic-sized
/// state vectors. Assumes first 3 elements are ECI position in meters.
pub struct DAltitudeEvent {
    inner: DValueEvent,
}

impl DAltitudeEvent {
    /// Create new altitude event
    ///
    /// # Arguments
    /// * `value_altitude` - Geodetic altitude value in meters above WGS84 ellipsoid
    /// * `name` - Event name
    /// * `direction` - Detection direction
    pub fn new(value_altitude: f64, name: impl Into<String>, direction: EventDirection) -> Self {
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
            inner: DValueEvent::new(name, altitude_fn, value_altitude, direction),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

// Delegate EventDetectorD trait implementation to inner DValueEvent
impl_devent_detector_delegate!(DAltitudeEvent, inner);

// =============================================================================
// Orbital Element Events
// =============================================================================

/// Semi-major axis event detector (static-sized)
///
/// Detects when semi-major axis crosses a value value. Computes orbital
/// elements from the Cartesian state (assumes first 6 elements are ECI state).
///
/// # Examples
/// ```
/// use brahe::events::{SSemiMajorAxisEvent, EventDirection};
/// use brahe::constants::R_EARTH;
///
/// // Detect when semi-major axis drops below 6800 km
/// let event = SSemiMajorAxisEvent::<6, 0>::new(
///     R_EARTH + 400e3,
///     "Low SMA",
///     EventDirection::Decreasing
/// );
/// ```
pub struct SSemiMajorAxisEvent<const S: usize, const P: usize> {
    inner: SValueEvent<S, P>,
}

impl<const S: usize, const P: usize> SSemiMajorAxisEvent<S, P> {
    /// Create new semi-major axis event
    ///
    /// # Arguments
    /// * `value` - Semi-major axis value in meters
    /// * `name` - Event name
    /// * `direction` - Detection direction
    pub fn new(value: f64, name: impl Into<String>, direction: EventDirection) -> Self {
        let value_fn = |_t: Epoch, state: &SVector<f64, S>, _params: Option<&SVector<f64, P>>| {
            let state6 = Vector6::new(state[0], state[1], state[2], state[3], state[4], state[5]);
            let koe = state_eci_to_koe(state6, AngleFormat::Radians);
            koe[0] // Semi-major axis
        };

        Self {
            inner: SValueEvent::new(name, value_fn, value, direction),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_sevent_detector_delegate!(SSemiMajorAxisEvent<S, P>, inner, S, P);

/// Dynamic-sized semi-major axis event detector
pub struct DSemiMajorAxisEvent {
    inner: DValueEvent,
}

impl DSemiMajorAxisEvent {
    /// Create new semi-major axis event
    pub fn new(value: f64, name: impl Into<String>, direction: EventDirection) -> Self {
        let value_fn = |_t: Epoch, state: &DVector<f64>, _params: Option<&DVector<f64>>| {
            let state6 = Vector6::new(state[0], state[1], state[2], state[3], state[4], state[5]);
            let koe = state_eci_to_koe(state6, AngleFormat::Radians);
            koe[0]
        };

        Self {
            inner: DValueEvent::new(name, value_fn, value, direction),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_devent_detector_delegate!(DSemiMajorAxisEvent, inner);

/// Eccentricity event detector (static-sized)
///
/// Detects when eccentricity crosses a value value.
///
/// # Examples
/// ```
/// use brahe::events::{SEccentricityEvent, EventDirection};
///
/// // Detect when eccentricity exceeds 0.1
/// let event = SEccentricityEvent::<6, 0>::new(
///     0.1,
///     "High Eccentricity",
///     EventDirection::Increasing
/// );
/// ```
pub struct SEccentricityEvent<const S: usize, const P: usize> {
    inner: SValueEvent<S, P>,
}

impl<const S: usize, const P: usize> SEccentricityEvent<S, P> {
    /// Create new eccentricity event
    pub fn new(value: f64, name: impl Into<String>, direction: EventDirection) -> Self {
        let value_fn = |_t: Epoch, state: &SVector<f64, S>, _params: Option<&SVector<f64, P>>| {
            let state6 = Vector6::new(state[0], state[1], state[2], state[3], state[4], state[5]);
            let koe = state_eci_to_koe(state6, AngleFormat::Radians);
            koe[1] // Eccentricity
        };

        Self {
            inner: SValueEvent::new(name, value_fn, value, direction),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_sevent_detector_delegate!(SEccentricityEvent<S, P>, inner, S, P);

/// Dynamic-sized eccentricity event detector
pub struct DEccentricityEvent {
    inner: DValueEvent,
}

impl DEccentricityEvent {
    /// Create new eccentricity event
    pub fn new(value: f64, name: impl Into<String>, direction: EventDirection) -> Self {
        let value_fn = |_t: Epoch, state: &DVector<f64>, _params: Option<&DVector<f64>>| {
            let state6 = Vector6::new(state[0], state[1], state[2], state[3], state[4], state[5]);
            let koe = state_eci_to_koe(state6, AngleFormat::Radians);
            koe[1]
        };

        Self {
            inner: DValueEvent::new(name, value_fn, value, direction),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_devent_detector_delegate!(DEccentricityEvent, inner);

/// Inclination event detector (static-sized)
///
/// Detects when inclination crosses a value value. Returns inclination in radians.
///
/// # Examples
/// ```
/// use brahe::events::{SInclinationEvent, EventDirection};
/// use brahe::AngleFormat;
///
/// // Detect when inclination exceeds 90 degrees
/// let event = SInclinationEvent::<6, 0>::new(
///     90.0,
///     "Retrograde",
///     EventDirection::Increasing,
///     AngleFormat::Degrees
/// );
/// ```
pub struct SInclinationEvent<const S: usize, const P: usize> {
    inner: SValueEvent<S, P>,
}

impl<const S: usize, const P: usize> SInclinationEvent<S, P> {
    /// Create new inclination event
    ///
    /// # Arguments
    /// * `value` - Inclination value
    /// * `name` - Event name
    /// * `direction` - Detection direction
    /// * `angle_format` - Whether value is in degrees or radians
    pub fn new(
        value: f64,
        name: impl Into<String>,
        direction: EventDirection,
        angle_format: AngleFormat,
    ) -> Self {
        // Convert value to radians if needed
        let value_rad = match angle_format {
            AngleFormat::Degrees => value.to_radians(),
            AngleFormat::Radians => value,
        };

        let value_fn = |_t: Epoch, state: &SVector<f64, S>, _params: Option<&SVector<f64, P>>| {
            let state6 = Vector6::new(state[0], state[1], state[2], state[3], state[4], state[5]);
            let koe = state_eci_to_koe(state6, AngleFormat::Radians);
            koe[2] // Inclination
        };

        Self {
            inner: SValueEvent::new(name, value_fn, value_rad, direction),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_sevent_detector_delegate!(SInclinationEvent<S, P>, inner, S, P);

/// Dynamic-sized inclination event detector
pub struct DInclinationEvent {
    inner: DValueEvent,
}

impl DInclinationEvent {
    /// Create new inclination event
    ///
    /// # Arguments
    /// * `value` - Inclination value
    /// * `name` - Event name
    /// * `direction` - Detection direction
    /// * `angle_format` - Whether value is in degrees or radians
    pub fn new(
        value: f64,
        name: impl Into<String>,
        direction: EventDirection,
        angle_format: AngleFormat,
    ) -> Self {
        // Convert value to radians if needed
        let value_rad = match angle_format {
            AngleFormat::Degrees => value.to_radians(),
            AngleFormat::Radians => value,
        };

        let value_fn = |_t: Epoch, state: &DVector<f64>, _params: Option<&DVector<f64>>| {
            let state6 = Vector6::new(state[0], state[1], state[2], state[3], state[4], state[5]);
            let koe = state_eci_to_koe(state6, AngleFormat::Radians);
            koe[2]
        };

        Self {
            inner: DValueEvent::new(name, value_fn, value_rad, direction),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_devent_detector_delegate!(DInclinationEvent, inner);

/// Argument of perigee event detector (static-sized)
///
/// Detects when argument of perigee crosses a value value. Returns in radians.
pub struct SArgumentOfPerigeeEvent<const S: usize, const P: usize> {
    inner: SValueEvent<S, P>,
}

impl<const S: usize, const P: usize> SArgumentOfPerigeeEvent<S, P> {
    /// Create new argument of perigee event
    ///
    /// # Arguments
    /// * `value` - Argument of perigee value
    /// * `name` - Event name
    /// * `direction` - Detection direction
    /// * `angle_format` - Whether value is in degrees or radians
    pub fn new(
        value: f64,
        name: impl Into<String>,
        direction: EventDirection,
        angle_format: AngleFormat,
    ) -> Self {
        let value_rad = match angle_format {
            AngleFormat::Degrees => value.to_radians(),
            AngleFormat::Radians => value,
        };

        let value_fn = |_t: Epoch, state: &SVector<f64, S>, _params: Option<&SVector<f64, P>>| {
            let state6 = Vector6::new(state[0], state[1], state[2], state[3], state[4], state[5]);
            let koe = state_eci_to_koe(state6, AngleFormat::Radians);
            koe[4] // Argument of perigee
        };

        Self {
            inner: SValueEvent::new(name, value_fn, value_rad, direction),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_sevent_detector_delegate!(SArgumentOfPerigeeEvent<S, P>, inner, S, P);

/// Dynamic-sized argument of perigee event detector
pub struct DArgumentOfPerigeeEvent {
    inner: DValueEvent,
}

impl DArgumentOfPerigeeEvent {
    /// Create new argument of perigee event
    ///
    /// # Arguments
    /// * `value` - Argument of perigee value
    /// * `name` - Event name
    /// * `direction` - Detection direction
    /// * `angle_format` - Whether value is in degrees or radians
    pub fn new(
        value: f64,
        name: impl Into<String>,
        direction: EventDirection,
        angle_format: AngleFormat,
    ) -> Self {
        let value_rad = match angle_format {
            AngleFormat::Degrees => value.to_radians(),
            AngleFormat::Radians => value,
        };

        let value_fn = |_t: Epoch, state: &DVector<f64>, _params: Option<&DVector<f64>>| {
            let state6 = Vector6::new(state[0], state[1], state[2], state[3], state[4], state[5]);
            let koe = state_eci_to_koe(state6, AngleFormat::Radians);
            koe[4]
        };

        Self {
            inner: DValueEvent::new(name, value_fn, value_rad, direction),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_devent_detector_delegate!(DArgumentOfPerigeeEvent, inner);

/// Mean anomaly event detector (static-sized)
///
/// Detects when mean anomaly crosses a value value. Returns in radians.
pub struct SMeanAnomalyEvent<const S: usize, const P: usize> {
    inner: SValueEvent<S, P>,
}

impl<const S: usize, const P: usize> SMeanAnomalyEvent<S, P> {
    /// Create new mean anomaly event
    ///
    /// # Arguments
    /// * `value` - Mean anomaly value
    /// * `name` - Event name
    /// * `direction` - Detection direction
    /// * `angle_format` - Whether value is in degrees or radians
    pub fn new(
        value: f64,
        name: impl Into<String>,
        direction: EventDirection,
        angle_format: AngleFormat,
    ) -> Self {
        let value_rad = match angle_format {
            AngleFormat::Degrees => value.to_radians(),
            AngleFormat::Radians => value,
        };

        let value_fn = |_t: Epoch, state: &SVector<f64, S>, _params: Option<&SVector<f64, P>>| {
            let state6 = Vector6::new(state[0], state[1], state[2], state[3], state[4], state[5]);
            let koe = state_eci_to_koe(state6, AngleFormat::Radians);
            koe[5] // Mean anomaly
        };

        Self {
            inner: SValueEvent::new(name, value_fn, value_rad, direction),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_sevent_detector_delegate!(SMeanAnomalyEvent<S, P>, inner, S, P);

/// Dynamic-sized mean anomaly event detector
pub struct DMeanAnomalyEvent {
    inner: DValueEvent,
}

impl DMeanAnomalyEvent {
    /// Create new mean anomaly event
    ///
    /// # Arguments
    /// * `value` - Mean anomaly value
    /// * `name` - Event name
    /// * `direction` - Detection direction
    /// * `angle_format` - Whether value is in degrees or radians
    pub fn new(
        value: f64,
        name: impl Into<String>,
        direction: EventDirection,
        angle_format: AngleFormat,
    ) -> Self {
        let value_rad = match angle_format {
            AngleFormat::Degrees => value.to_radians(),
            AngleFormat::Radians => value,
        };

        let value_fn = |_t: Epoch, state: &DVector<f64>, _params: Option<&DVector<f64>>| {
            let state6 = Vector6::new(state[0], state[1], state[2], state[3], state[4], state[5]);
            let koe = state_eci_to_koe(state6, AngleFormat::Radians);
            koe[5]
        };

        Self {
            inner: DValueEvent::new(name, value_fn, value_rad, direction),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_devent_detector_delegate!(DMeanAnomalyEvent, inner);

/// Eccentric anomaly event detector (static-sized)
///
/// Detects when eccentric anomaly crosses a value value. Returns in radians.
pub struct SEccentricAnomalyEvent<const S: usize, const P: usize> {
    inner: SValueEvent<S, P>,
}

impl<const S: usize, const P: usize> SEccentricAnomalyEvent<S, P> {
    /// Create new eccentric anomaly event
    ///
    /// # Arguments
    /// * `value` - Eccentric anomaly value
    /// * `name` - Event name
    /// * `direction` - Detection direction
    /// * `angle_format` - Whether value is in degrees or radians
    pub fn new(
        value: f64,
        name: impl Into<String>,
        direction: EventDirection,
        angle_format: AngleFormat,
    ) -> Self {
        let value_rad = match angle_format {
            AngleFormat::Degrees => value.to_radians(),
            AngleFormat::Radians => value,
        };

        let value_fn = |_t: Epoch, state: &SVector<f64, S>, _params: Option<&SVector<f64, P>>| {
            let state6 = Vector6::new(state[0], state[1], state[2], state[3], state[4], state[5]);
            let koe = state_eci_to_koe(state6, AngleFormat::Radians);
            let e = koe[1];
            let m = koe[5];
            // Convert mean anomaly to eccentric anomaly
            anomaly_mean_to_eccentric(m, e, AngleFormat::Radians).unwrap_or(m)
        };

        Self {
            inner: SValueEvent::new(name, value_fn, value_rad, direction),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_sevent_detector_delegate!(SEccentricAnomalyEvent<S, P>, inner, S, P);

/// Dynamic-sized eccentric anomaly event detector
pub struct DEccentricAnomalyEvent {
    inner: DValueEvent,
}

impl DEccentricAnomalyEvent {
    /// Create new eccentric anomaly event
    ///
    /// # Arguments
    /// * `value` - Eccentric anomaly value
    /// * `name` - Event name
    /// * `direction` - Detection direction
    /// * `angle_format` - Whether value is in degrees or radians
    pub fn new(
        value: f64,
        name: impl Into<String>,
        direction: EventDirection,
        angle_format: AngleFormat,
    ) -> Self {
        let value_rad = match angle_format {
            AngleFormat::Degrees => value.to_radians(),
            AngleFormat::Radians => value,
        };

        let value_fn = |_t: Epoch, state: &DVector<f64>, _params: Option<&DVector<f64>>| {
            let state6 = Vector6::new(state[0], state[1], state[2], state[3], state[4], state[5]);
            let koe = state_eci_to_koe(state6, AngleFormat::Radians);
            let e = koe[1];
            let m = koe[5];
            anomaly_mean_to_eccentric(m, e, AngleFormat::Radians).unwrap_or(m)
        };

        Self {
            inner: DValueEvent::new(name, value_fn, value_rad, direction),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_devent_detector_delegate!(DEccentricAnomalyEvent, inner);

/// True anomaly event detector (static-sized)
///
/// Detects when true anomaly crosses a value value. Returns in radians.
pub struct STrueAnomalyEvent<const S: usize, const P: usize> {
    inner: SValueEvent<S, P>,
}

impl<const S: usize, const P: usize> STrueAnomalyEvent<S, P> {
    /// Create new true anomaly event
    ///
    /// # Arguments
    /// * `value` - True anomaly value
    /// * `name` - Event name
    /// * `direction` - Detection direction
    /// * `angle_format` - Whether value is in degrees or radians
    pub fn new(
        value: f64,
        name: impl Into<String>,
        direction: EventDirection,
        angle_format: AngleFormat,
    ) -> Self {
        let value_rad = match angle_format {
            AngleFormat::Degrees => value.to_radians(),
            AngleFormat::Radians => value,
        };

        let value_fn = |_t: Epoch, state: &SVector<f64, S>, _params: Option<&SVector<f64, P>>| {
            let state6 = Vector6::new(state[0], state[1], state[2], state[3], state[4], state[5]);
            let koe = state_eci_to_koe(state6, AngleFormat::Radians);
            let e = koe[1];
            let m = koe[5];
            // Convert mean anomaly to true anomaly
            anomaly_mean_to_true(m, e, AngleFormat::Radians).unwrap_or(m)
        };

        Self {
            inner: SValueEvent::new(name, value_fn, value_rad, direction),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_sevent_detector_delegate!(STrueAnomalyEvent<S, P>, inner, S, P);

/// Dynamic-sized true anomaly event detector
pub struct DTrueAnomalyEvent {
    inner: DValueEvent,
}

impl DTrueAnomalyEvent {
    /// Create new true anomaly event
    ///
    /// # Arguments
    /// * `value` - True anomaly value
    /// * `name` - Event name
    /// * `direction` - Detection direction
    /// * `angle_format` - Whether value is in degrees or radians
    pub fn new(
        value: f64,
        name: impl Into<String>,
        direction: EventDirection,
        angle_format: AngleFormat,
    ) -> Self {
        let value_rad = match angle_format {
            AngleFormat::Degrees => value.to_radians(),
            AngleFormat::Radians => value,
        };

        let value_fn = |_t: Epoch, state: &DVector<f64>, _params: Option<&DVector<f64>>| {
            let state6 = Vector6::new(state[0], state[1], state[2], state[3], state[4], state[5]);
            let koe = state_eci_to_koe(state6, AngleFormat::Radians);
            let e = koe[1];
            let m = koe[5];
            anomaly_mean_to_true(m, e, AngleFormat::Radians).unwrap_or(m)
        };

        Self {
            inner: DValueEvent::new(name, value_fn, value_rad, direction),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_devent_detector_delegate!(DTrueAnomalyEvent, inner);

/// Argument of latitude event detector (static-sized)
///
/// Detects when argument of latitude (omega + true_anomaly) crosses a value.
/// Returns in radians.
pub struct SArgumentOfLatitudeEvent<const S: usize, const P: usize> {
    inner: SValueEvent<S, P>,
}

impl<const S: usize, const P: usize> SArgumentOfLatitudeEvent<S, P> {
    /// Create new argument of latitude event
    ///
    /// # Arguments
    /// * `value` - Argument of latitude value
    /// * `name` - Event name
    /// * `direction` - Detection direction
    /// * `angle_format` - Whether value is in degrees or radians
    pub fn new(
        value: f64,
        name: impl Into<String>,
        direction: EventDirection,
        angle_format: AngleFormat,
    ) -> Self {
        let value_rad = match angle_format {
            AngleFormat::Degrees => value.to_radians(),
            AngleFormat::Radians => value,
        };

        let value_fn = |_t: Epoch, state: &SVector<f64, S>, _params: Option<&SVector<f64, P>>| {
            let state6 = Vector6::new(state[0], state[1], state[2], state[3], state[4], state[5]);
            let koe = state_eci_to_koe(state6, AngleFormat::Radians);
            let e = koe[1];
            let omega = koe[4]; // Argument of perigee
            let m = koe[5];
            let nu = anomaly_mean_to_true(m, e, AngleFormat::Radians).unwrap_or(m);
            // Argument of latitude = omega + true anomaly
            let u = omega + nu;
            // Normalize to [0, 2*PI)
            u.rem_euclid(2.0 * std::f64::consts::PI)
        };

        Self {
            inner: SValueEvent::new(name, value_fn, value_rad, direction),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_sevent_detector_delegate!(SArgumentOfLatitudeEvent<S, P>, inner, S, P);

/// Dynamic-sized argument of latitude event detector
pub struct DArgumentOfLatitudeEvent {
    inner: DValueEvent,
}

impl DArgumentOfLatitudeEvent {
    /// Create new argument of latitude event
    ///
    /// # Arguments
    /// * `value` - Argument of latitude value
    /// * `name` - Event name
    /// * `direction` - Detection direction
    /// * `angle_format` - Whether value is in degrees or radians
    pub fn new(
        value: f64,
        name: impl Into<String>,
        direction: EventDirection,
        angle_format: AngleFormat,
    ) -> Self {
        let value_rad = match angle_format {
            AngleFormat::Degrees => value.to_radians(),
            AngleFormat::Radians => value,
        };

        let value_fn = |_t: Epoch, state: &DVector<f64>, _params: Option<&DVector<f64>>| {
            let state6 = Vector6::new(state[0], state[1], state[2], state[3], state[4], state[5]);
            let koe = state_eci_to_koe(state6, AngleFormat::Radians);
            let e = koe[1];
            let omega = koe[4];
            let m = koe[5];
            let nu = anomaly_mean_to_true(m, e, AngleFormat::Radians).unwrap_or(m);
            let u = omega + nu;
            u.rem_euclid(2.0 * std::f64::consts::PI)
        };

        Self {
            inner: DValueEvent::new(name, value_fn, value_rad, direction),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_devent_detector_delegate!(DArgumentOfLatitudeEvent, inner);

/// Ascending node event detector (static-sized)
///
/// Detects when spacecraft crosses the ascending node (equator from south to north).
/// The ascending node occurs when the true argument of latitude crosses 0 with increasing values.
///
/// # Examples
/// ```
/// use brahe::events::SAscendingNodeEvent;
///
/// // Detect ascending node crossing
/// let event = SAscendingNodeEvent::<6, 0>::new("Ascending Node");
/// ```
pub struct SAscendingNodeEvent<const S: usize, const P: usize> {
    inner: SValueEvent<S, P>,
}

impl<const S: usize, const P: usize> SAscendingNodeEvent<S, P> {
    /// Create new ascending node event
    ///
    /// # Arguments
    /// * `name` - Event name
    pub fn new(name: impl Into<String>) -> Self {
        let value_fn = |_t: Epoch, state: &SVector<f64, S>, _params: Option<&SVector<f64, P>>| {
            let state6 = Vector6::new(state[0], state[1], state[2], state[3], state[4], state[5]);
            let koe = state_eci_to_koe(state6, AngleFormat::Radians);
            let e = koe[1];
            let omega = koe[4]; // Argument of perigee
            let m = koe[5]; // Mean anomaly

            // Convert mean anomaly to true anomaly
            let nu = anomaly_mean_to_true(m, e, AngleFormat::Radians).unwrap_or(m);

            // True argument of latitude = omega + nu
            // Normalize to [-pi, pi) for proper zero crossing detection
            // Use rem_euclid to handle negative values correctly
            let u = omega + nu;
            (u + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI
        };

        Self {
            inner: SValueEvent::new(name, value_fn, 0.0, EventDirection::Increasing),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_sevent_detector_delegate!(SAscendingNodeEvent<S, P>, inner, S, P);

/// Dynamic-sized ascending node event detector
pub struct DAscendingNodeEvent {
    inner: DValueEvent,
}

impl DAscendingNodeEvent {
    /// Create new ascending node event
    pub fn new(name: impl Into<String>) -> Self {
        let value_fn = |_t: Epoch, state: &DVector<f64>, _params: Option<&DVector<f64>>| {
            let state6 = Vector6::new(state[0], state[1], state[2], state[3], state[4], state[5]);
            let koe = state_eci_to_koe(state6, AngleFormat::Radians);
            let e = koe[1];
            let omega = koe[4];
            let m = koe[5];

            let nu = anomaly_mean_to_true(m, e, AngleFormat::Radians).unwrap_or(m);
            let u = omega + nu;
            (u + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI
        };

        Self {
            inner: DValueEvent::new(name, value_fn, 0.0, EventDirection::Increasing),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_devent_detector_delegate!(DAscendingNodeEvent, inner);

/// Descending node event detector (static-sized)
///
/// Detects when spacecraft crosses the descending node (equator from north to south).
/// The descending node occurs when the true argument of latitude crosses π with increasing values.
///
/// # Examples
/// ```
/// use brahe::events::SDescendingNodeEvent;
///
/// // Detect descending node crossing
/// let event = SDescendingNodeEvent::<6, 0>::new("Descending Node");
/// ```
pub struct SDescendingNodeEvent<const S: usize, const P: usize> {
    inner: SValueEvent<S, P>,
}

impl<const S: usize, const P: usize> SDescendingNodeEvent<S, P> {
    /// Create new descending node event
    ///
    /// # Arguments
    /// * `name` - Event name
    pub fn new(name: impl Into<String>) -> Self {
        let value_fn = |_t: Epoch, state: &SVector<f64, S>, _params: Option<&SVector<f64, P>>| {
            let state6 = Vector6::new(state[0], state[1], state[2], state[3], state[4], state[5]);
            let koe = state_eci_to_koe(state6, AngleFormat::Radians);
            let e = koe[1];
            let omega = koe[4]; // Argument of perigee
            let m = koe[5]; // Mean anomaly

            // Convert mean anomaly to true anomaly
            let nu = anomaly_mean_to_true(m, e, AngleFormat::Radians).unwrap_or(m);

            // True argument of latitude = omega + nu
            let u = omega + nu;
            // Shift so that π becomes our zero crossing point
            // u - π gives us zero at descending node
            let u_shifted = u - std::f64::consts::PI;
            // Normalize to [-pi, pi) using rem_euclid for correct handling of negative values
            (u_shifted + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI)
                - std::f64::consts::PI
        };

        Self {
            inner: SValueEvent::new(name, value_fn, 0.0, EventDirection::Increasing),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_sevent_detector_delegate!(SDescendingNodeEvent<S, P>, inner, S, P);

/// Dynamic-sized descending node event detector
pub struct DDescendingNodeEvent {
    inner: DValueEvent,
}

impl DDescendingNodeEvent {
    /// Create new descending node event
    pub fn new(name: impl Into<String>) -> Self {
        let value_fn = |_t: Epoch, state: &DVector<f64>, _params: Option<&DVector<f64>>| {
            let state6 = Vector6::new(state[0], state[1], state[2], state[3], state[4], state[5]);
            let koe = state_eci_to_koe(state6, AngleFormat::Radians);
            let e = koe[1];
            let omega = koe[4];
            let m = koe[5];

            let nu = anomaly_mean_to_true(m, e, AngleFormat::Radians).unwrap_or(m);
            let u = omega + nu;
            let u_shifted = u - std::f64::consts::PI;
            (u_shifted + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI)
                - std::f64::consts::PI
        };

        Self {
            inner: DValueEvent::new(name, value_fn, 0.0, EventDirection::Increasing),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_devent_detector_delegate!(DDescendingNodeEvent, inner);

// =============================================================================
// State-Derived Events
// =============================================================================

/// Speed event detector (static-sized)
///
/// Detects when velocity magnitude crosses a value value.
///
/// # Examples
/// ```
/// use brahe::events::{SSpeedEvent, EventDirection};
///
/// // Detect when speed drops below 7 km/s
/// let event = SSpeedEvent::<6, 0>::new(
///     7000.0,
///     "Low Speed",
///     EventDirection::Decreasing
/// );
/// ```
pub struct SSpeedEvent<const S: usize, const P: usize> {
    inner: SValueEvent<S, P>,
}

impl<const S: usize, const P: usize> SSpeedEvent<S, P> {
    /// Create new speed event
    ///
    /// # Arguments
    /// * `value` - Speed value in m/s
    /// * `name` - Event name
    /// * `direction` - Detection direction
    pub fn new(value: f64, name: impl Into<String>, direction: EventDirection) -> Self {
        let value_fn = |_t: Epoch, state: &SVector<f64, S>, _params: Option<&SVector<f64, P>>| {
            // Compute velocity magnitude from state[3:6]
            let v = Vector3::new(state[3], state[4], state[5]);
            v.norm()
        };

        Self {
            inner: SValueEvent::new(name, value_fn, value, direction),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_sevent_detector_delegate!(SSpeedEvent<S, P>, inner, S, P);

/// Dynamic-sized speed event detector
pub struct DSpeedEvent {
    inner: DValueEvent,
}

impl DSpeedEvent {
    /// Create new speed event (value in m/s)
    pub fn new(value: f64, name: impl Into<String>, direction: EventDirection) -> Self {
        let value_fn = |_t: Epoch, state: &DVector<f64>, _params: Option<&DVector<f64>>| {
            let v = Vector3::new(state[3], state[4], state[5]);
            v.norm()
        };

        Self {
            inner: DValueEvent::new(name, value_fn, value, direction),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_devent_detector_delegate!(DSpeedEvent, inner);

/// Longitude event detector (static-sized)
///
/// Detects when geodetic longitude crosses a value value.
/// Requires EOP initialization for ECI->ECEF transformation.
///
/// # Examples
/// ```
/// use brahe::events::{SLongitudeEvent, EventDirection};
/// use brahe::AngleFormat;
///
/// // Detect when crossing prime meridian (0 degrees longitude)
/// let event = SLongitudeEvent::<6, 0>::new(
///     0.0,
///     "Prime Meridian",
///     EventDirection::Any,
///     AngleFormat::Degrees
/// );
/// ```
pub struct SLongitudeEvent<const S: usize, const P: usize> {
    inner: SValueEvent<S, P>,
}

impl<const S: usize, const P: usize> SLongitudeEvent<S, P> {
    /// Create new longitude event
    ///
    /// # Arguments
    /// * `value` - Longitude value
    /// * `name` - Event name
    /// * `direction` - Detection direction
    /// * `angle_format` - Whether value is in degrees or radians
    pub fn new(
        value: f64,
        name: impl Into<String>,
        direction: EventDirection,
        angle_format: AngleFormat,
    ) -> Self {
        let value_rad = match angle_format {
            AngleFormat::Degrees => value.to_radians(),
            AngleFormat::Radians => value,
        };

        let value_fn = |t: Epoch, state: &SVector<f64, S>, _params: Option<&SVector<f64, P>>| {
            let r_eci = state.fixed_rows::<3>(0).into_owned();
            let r_ecef = position_eci_to_ecef(t, r_eci);
            let geodetic = position_ecef_to_geodetic(r_ecef, AngleFormat::Radians);
            geodetic[0] // Longitude
        };

        Self {
            inner: SValueEvent::new(name, value_fn, value_rad, direction),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_sevent_detector_delegate!(SLongitudeEvent<S, P>, inner, S, P);

/// Dynamic-sized longitude event detector
pub struct DLongitudeEvent {
    inner: DValueEvent,
}

impl DLongitudeEvent {
    /// Create new longitude event
    ///
    /// # Arguments
    /// * `value` - Longitude value
    /// * `name` - Event name
    /// * `direction` - Detection direction
    /// * `angle_format` - Whether value is in degrees or radians
    pub fn new(
        value: f64,
        name: impl Into<String>,
        direction: EventDirection,
        angle_format: AngleFormat,
    ) -> Self {
        let value_rad = match angle_format {
            AngleFormat::Degrees => value.to_radians(),
            AngleFormat::Radians => value,
        };

        let value_fn = |t: Epoch, state: &DVector<f64>, _params: Option<&DVector<f64>>| {
            let r_eci = Vector3::new(state[0], state[1], state[2]);
            let r_ecef = position_eci_to_ecef(t, r_eci);
            let geodetic = position_ecef_to_geodetic(r_ecef, AngleFormat::Radians);
            geodetic[0]
        };

        Self {
            inner: DValueEvent::new(name, value_fn, value_rad, direction),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_devent_detector_delegate!(DLongitudeEvent, inner);

/// Latitude event detector (static-sized)
///
/// Detects when geodetic latitude crosses a value value.
/// Requires EOP initialization for ECI->ECEF transformation.
///
/// # Examples
/// ```
/// use brahe::events::{SLatitudeEvent, EventDirection};
/// use brahe::AngleFormat;
///
/// // Detect when crossing equator (0 degrees latitude)
/// let event = SLatitudeEvent::<6, 0>::new(
///     0.0,
///     "Equator Crossing",
///     EventDirection::Any,
///     AngleFormat::Degrees
/// );
/// ```
pub struct SLatitudeEvent<const S: usize, const P: usize> {
    inner: SValueEvent<S, P>,
}

impl<const S: usize, const P: usize> SLatitudeEvent<S, P> {
    /// Create new latitude event
    ///
    /// # Arguments
    /// * `value` - Latitude value
    /// * `name` - Event name
    /// * `direction` - Detection direction
    /// * `angle_format` - Whether value is in degrees or radians
    pub fn new(
        value: f64,
        name: impl Into<String>,
        direction: EventDirection,
        angle_format: AngleFormat,
    ) -> Self {
        let value_rad = match angle_format {
            AngleFormat::Degrees => value.to_radians(),
            AngleFormat::Radians => value,
        };

        let value_fn = |t: Epoch, state: &SVector<f64, S>, _params: Option<&SVector<f64, P>>| {
            let r_eci = state.fixed_rows::<3>(0).into_owned();
            let r_ecef = position_eci_to_ecef(t, r_eci);
            let geodetic = position_ecef_to_geodetic(r_ecef, AngleFormat::Radians);
            geodetic[1] // Latitude
        };

        Self {
            inner: SValueEvent::new(name, value_fn, value_rad, direction),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_sevent_detector_delegate!(SLatitudeEvent<S, P>, inner, S, P);

/// Dynamic-sized latitude event detector
pub struct DLatitudeEvent {
    inner: DValueEvent,
}

impl DLatitudeEvent {
    /// Create new latitude event
    ///
    /// # Arguments
    /// * `value` - Latitude value
    /// * `name` - Event name
    /// * `direction` - Detection direction
    /// * `angle_format` - Whether value is in degrees or radians
    pub fn new(
        value: f64,
        name: impl Into<String>,
        direction: EventDirection,
        angle_format: AngleFormat,
    ) -> Self {
        let value_rad = match angle_format {
            AngleFormat::Degrees => value.to_radians(),
            AngleFormat::Radians => value,
        };

        let value_fn = |t: Epoch, state: &DVector<f64>, _params: Option<&DVector<f64>>| {
            let r_eci = Vector3::new(state[0], state[1], state[2]);
            let r_ecef = position_eci_to_ecef(t, r_eci);
            let geodetic = position_ecef_to_geodetic(r_ecef, AngleFormat::Radians);
            geodetic[1]
        };

        Self {
            inner: DValueEvent::new(name, value_fn, value_rad, direction),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_devent_detector_delegate!(DLatitudeEvent, inner);

// =============================================================================
// Eclipse/Shadow Events
// =============================================================================

/// Helper function to get sun position based on ephemeris source
fn get_sun_position(t: Epoch, source: Option<EphemerisSource>) -> Vector3<f64> {
    match source {
        None | Some(EphemerisSource::LowPrecision) => sun_position(t),
        Some(source) => sun_position_de(t, source).unwrap_or_else(|_| sun_position(t)),
    }
}

/// Umbra event detector (static-sized)
///
/// Detects when spacecraft enters/exits Earth's umbra (full shadow).
/// Uses the conical shadow model.
///
/// # Examples
/// ```
/// use brahe::events::{SUmbraEvent, EdgeType};
/// use brahe::propagators::EphemerisSource;
///
/// // Detect umbra entry with low-precision ephemeris (default)
/// let event = SUmbraEvent::<6, 0>::new("Umbra Entry", EdgeType::RisingEdge, None);
///
/// // Detect umbra exit with high-precision DE440s ephemeris
/// let event = SUmbraEvent::<6, 0>::new("Umbra Exit", EdgeType::FallingEdge, Some(EphemerisSource::DE440s));
/// ```
pub struct SUmbraEvent<const S: usize, const P: usize> {
    inner: SBinaryEvent<S, P>,
}

impl<const S: usize, const P: usize> SUmbraEvent<S, P> {
    /// Create new umbra event
    ///
    /// # Arguments
    /// * `name` - Event name
    /// * `edge` - Edge type (RisingEdge = entering umbra, FallingEdge = exiting umbra)
    /// * `ephemeris_source` - Source for sun position (None for low-precision analytical)
    pub fn new(
        name: impl Into<String>,
        edge: EdgeType,
        ephemeris_source: Option<EphemerisSource>,
    ) -> Self {
        let condition_fn =
            move |t: Epoch, state: &SVector<f64, S>, _params: Option<&SVector<f64, P>>| {
                let r_eci = Vector3::new(state[0], state[1], state[2]);
                let r_sun = get_sun_position(t, ephemeris_source);
                let illumination = eclipse_conical(r_eci, r_sun);
                // In umbra when illumination is exactly 0.0
                illumination == 0.0
            };

        Self {
            inner: SBinaryEvent::new(name, condition_fn, edge),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_sevent_detector_delegate!(SUmbraEvent<S, P>, inner, S, P);

/// Dynamic-sized umbra event detector
pub struct DUmbraEvent {
    inner: DBinaryEvent,
}

impl DUmbraEvent {
    /// Create new umbra event
    ///
    /// # Arguments
    /// * `name` - Event name
    /// * `edge` - Edge type (RisingEdge = entering umbra, FallingEdge = exiting umbra)
    /// * `ephemeris_source` - Source for sun position (None for low-precision analytical)
    pub fn new(
        name: impl Into<String>,
        edge: EdgeType,
        ephemeris_source: Option<EphemerisSource>,
    ) -> Self {
        let condition_fn = move |t: Epoch, state: &DVector<f64>, _params: Option<&DVector<f64>>| {
            let r_eci = Vector3::new(state[0], state[1], state[2]);
            let r_sun = get_sun_position(t, ephemeris_source);
            let illumination = eclipse_conical(r_eci, r_sun);
            illumination == 0.0
        };

        Self {
            inner: DBinaryEvent::new(name, condition_fn, edge),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_devent_detector_delegate!(DUmbraEvent, inner);

/// Penumbra event detector (static-sized)
///
/// Detects when spacecraft enters/exits Earth's penumbra (partial shadow).
/// Uses the conical shadow model.
///
/// # Examples
/// ```
/// use brahe::events::{SPenumbraEvent, EdgeType};
/// use brahe::propagators::EphemerisSource;
///
/// // Detect penumbra entry with low-precision ephemeris
/// let event = SPenumbraEvent::<6, 0>::new("Penumbra Entry", EdgeType::RisingEdge, None);
///
/// // Detect penumbra with high-precision DE440s ephemeris
/// let event = SPenumbraEvent::<6, 0>::new("Penumbra Entry", EdgeType::RisingEdge, Some(EphemerisSource::DE440s));
/// ```
pub struct SPenumbraEvent<const S: usize, const P: usize> {
    inner: SBinaryEvent<S, P>,
}

impl<const S: usize, const P: usize> SPenumbraEvent<S, P> {
    /// Create new penumbra event
    ///
    /// # Arguments
    /// * `name` - Event name
    /// * `edge` - Edge type (RisingEdge = entering penumbra, FallingEdge = exiting penumbra)
    /// * `ephemeris_source` - Source for sun position (None for low-precision analytical)
    pub fn new(
        name: impl Into<String>,
        edge: EdgeType,
        ephemeris_source: Option<EphemerisSource>,
    ) -> Self {
        let condition_fn =
            move |t: Epoch, state: &SVector<f64, S>, _params: Option<&SVector<f64, P>>| {
                let r_eci = Vector3::new(state[0], state[1], state[2]);
                let r_sun = get_sun_position(t, ephemeris_source);
                let illumination = eclipse_conical(r_eci, r_sun);
                // In penumbra when 0 < illumination < 1
                illumination > 0.0 && illumination < 1.0
            };

        Self {
            inner: SBinaryEvent::new(name, condition_fn, edge),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_sevent_detector_delegate!(SPenumbraEvent<S, P>, inner, S, P);

/// Dynamic-sized penumbra event detector
pub struct DPenumbraEvent {
    inner: DBinaryEvent,
}

impl DPenumbraEvent {
    /// Create new penumbra event
    ///
    /// # Arguments
    /// * `name` - Event name
    /// * `edge` - Edge type (RisingEdge = entering penumbra, FallingEdge = exiting penumbra)
    /// * `ephemeris_source` - Source for sun position (None for low-precision analytical)
    pub fn new(
        name: impl Into<String>,
        edge: EdgeType,
        ephemeris_source: Option<EphemerisSource>,
    ) -> Self {
        let condition_fn = move |t: Epoch, state: &DVector<f64>, _params: Option<&DVector<f64>>| {
            let r_eci = Vector3::new(state[0], state[1], state[2]);
            let r_sun = get_sun_position(t, ephemeris_source);
            let illumination = eclipse_conical(r_eci, r_sun);
            illumination > 0.0 && illumination < 1.0
        };

        Self {
            inner: DBinaryEvent::new(name, condition_fn, edge),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_devent_detector_delegate!(DPenumbraEvent, inner);

/// Eclipse event detector (static-sized)
///
/// Detects when spacecraft enters/exits eclipse (either umbra or penumbra).
/// Uses the conical shadow model.
///
/// # Examples
/// ```
/// use brahe::events::{SEclipseEvent, EdgeType};
/// use brahe::propagators::EphemerisSource;
///
/// // Detect any eclipse entry with low-precision ephemeris
/// let event = SEclipseEvent::<6, 0>::new("Eclipse Entry", EdgeType::RisingEdge, None);
///
/// // Detect eclipse exit with high-precision DE440s ephemeris
/// let event = SEclipseEvent::<6, 0>::new("Eclipse Exit", EdgeType::FallingEdge, Some(EphemerisSource::DE440s));
/// ```
pub struct SEclipseEvent<const S: usize, const P: usize> {
    inner: SBinaryEvent<S, P>,
}

impl<const S: usize, const P: usize> SEclipseEvent<S, P> {
    /// Create new eclipse event
    ///
    /// # Arguments
    /// * `name` - Event name
    /// * `edge` - Edge type (RisingEdge = entering eclipse, FallingEdge = exiting eclipse)
    /// * `ephemeris_source` - Source for sun position (None for low-precision analytical)
    pub fn new(
        name: impl Into<String>,
        edge: EdgeType,
        ephemeris_source: Option<EphemerisSource>,
    ) -> Self {
        let condition_fn =
            move |t: Epoch, state: &SVector<f64, S>, _params: Option<&SVector<f64, P>>| {
                let r_eci = Vector3::new(state[0], state[1], state[2]);
                let r_sun = get_sun_position(t, ephemeris_source);
                let illumination = eclipse_conical(r_eci, r_sun);
                // In eclipse when illumination < 1.0 (either umbra or penumbra)
                illumination < 1.0
            };

        Self {
            inner: SBinaryEvent::new(name, condition_fn, edge),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_sevent_detector_delegate!(SEclipseEvent<S, P>, inner, S, P);

/// Dynamic-sized eclipse event detector
pub struct DEclipseEvent {
    inner: DBinaryEvent,
}

impl DEclipseEvent {
    /// Create new eclipse event
    ///
    /// # Arguments
    /// * `name` - Event name
    /// * `edge` - Edge type (RisingEdge = entering eclipse, FallingEdge = exiting eclipse)
    /// * `ephemeris_source` - Source for sun position (None for low-precision analytical)
    pub fn new(
        name: impl Into<String>,
        edge: EdgeType,
        ephemeris_source: Option<EphemerisSource>,
    ) -> Self {
        let condition_fn = move |t: Epoch, state: &DVector<f64>, _params: Option<&DVector<f64>>| {
            let r_eci = Vector3::new(state[0], state[1], state[2]);
            let r_sun = get_sun_position(t, ephemeris_source);
            let illumination = eclipse_conical(r_eci, r_sun);
            illumination < 1.0
        };

        Self {
            inner: DBinaryEvent::new(name, condition_fn, edge),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_devent_detector_delegate!(DEclipseEvent, inner);

/// Sunlit event detector (static-sized)
///
/// Detects when spacecraft enters/exits sunlight (fully illuminated).
/// Uses the conical shadow model.
///
/// # Examples
/// ```
/// use brahe::events::{SSunlitEvent, EdgeType};
/// use brahe::propagators::EphemerisSource;
///
/// // Detect entering sunlight with low-precision ephemeris
/// let event = SSunlitEvent::<6, 0>::new("Enter Sunlight", EdgeType::RisingEdge, None);
///
/// // Detect leaving sunlight with high-precision DE440s ephemeris
/// let event = SSunlitEvent::<6, 0>::new("Leave Sunlight", EdgeType::FallingEdge, Some(EphemerisSource::DE440s));
/// ```
pub struct SSunlitEvent<const S: usize, const P: usize> {
    inner: SBinaryEvent<S, P>,
}

impl<const S: usize, const P: usize> SSunlitEvent<S, P> {
    /// Create new sunlit event
    ///
    /// # Arguments
    /// * `name` - Event name
    /// * `edge` - Edge type (RisingEdge = entering sunlight, FallingEdge = leaving sunlight)
    /// * `ephemeris_source` - Source for sun position (None for low-precision analytical)
    pub fn new(
        name: impl Into<String>,
        edge: EdgeType,
        ephemeris_source: Option<EphemerisSource>,
    ) -> Self {
        let condition_fn =
            move |t: Epoch, state: &SVector<f64, S>, _params: Option<&SVector<f64, P>>| {
                let r_eci = Vector3::new(state[0], state[1], state[2]);
                let r_sun = get_sun_position(t, ephemeris_source);
                let illumination = eclipse_conical(r_eci, r_sun);
                // Sunlit when illumination == 1.0
                illumination == 1.0
            };

        Self {
            inner: SBinaryEvent::new(name, condition_fn, edge),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_sevent_detector_delegate!(SSunlitEvent<S, P>, inner, S, P);

/// Dynamic-sized sunlit event detector
pub struct DSunlitEvent {
    inner: DBinaryEvent,
}

impl DSunlitEvent {
    /// Create new sunlit event
    ///
    /// # Arguments
    /// * `name` - Event name
    /// * `edge` - Edge type (RisingEdge = entering sunlight, FallingEdge = leaving sunlight)
    /// * `ephemeris_source` - Source for sun position (None for low-precision analytical)
    pub fn new(
        name: impl Into<String>,
        edge: EdgeType,
        ephemeris_source: Option<EphemerisSource>,
    ) -> Self {
        let condition_fn = move |t: Epoch, state: &DVector<f64>, _params: Option<&DVector<f64>>| {
            let r_eci = Vector3::new(state[0], state[1], state[2]);
            let r_sun = get_sun_position(t, ephemeris_source);
            let illumination = eclipse_conical(r_eci, r_sun);
            illumination == 1.0
        };

        Self {
            inner: DBinaryEvent::new(name, condition_fn, edge),
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
    pub fn set_terminal(mut self) -> Self {
        self.inner = self.inner.set_terminal();
        self
    }
}

impl_devent_detector_delegate!(DSunlitEvent, inner);

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use crate::constants::R_EARTH;
    use crate::time::TimeSystem;
    use crate::utils::testing::setup_global_test_eop;
    use nalgebra::{DVector, Vector6};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    // =========================================================================
    // SAltitudeEvent Tests
    // =========================================================================

    #[test]
    fn test_SAltitudeEvent_new() {
        setup_global_test_eop();

        let event = SAltitudeEvent::<6, 0>::new(500e3, "Low Altitude", EventDirection::Decreasing);

        assert_eq!(event.name(), "Low Altitude");
        assert_eq!(event.target_value(), 500e3);
        assert_eq!(event.direction(), EventDirection::Decreasing);
        assert_eq!(event.action(), EventAction::Continue);
    }

    #[test]
    fn test_SAltitudeEvent_evaluate() {
        setup_global_test_eop();

        let value = 500e3;
        let event = SAltitudeEvent::<6, 0>::new(value, "Altitude Test", EventDirection::Any);

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);

        // Low altitude state (below value) - ECI position along x-axis
        let state_low = Vector6::new(6000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        let val_low = event.evaluate(epoch, &state_low, None);
        // Should be negative (altitude < value)
        assert!(val_low < 0.0);

        // High altitude state (above value) - ECI position along x-axis
        let state_high = Vector6::new(R_EARTH + 1000e3, 0.0, 0.0, 0.0, 7.0e3, 0.0);
        let val_high = event.evaluate(epoch, &state_high, None);
        // Should be positive (altitude > value)
        assert!(val_high > 0.0);
    }

    #[test]
    fn test_SAltitudeEvent_target_value() {
        setup_global_test_eop();

        let event = SAltitudeEvent::<6, 0>::new(350e3, "Reentry", EventDirection::Decreasing);
        assert_eq!(event.target_value(), 350e3);
    }

    #[test]
    fn test_SAltitudeEvent_name() {
        setup_global_test_eop();

        let event = SAltitudeEvent::<6, 0>::new(500e3, "Test Name", EventDirection::Any);
        assert_eq!(event.name(), "Test Name");
    }

    #[test]
    fn test_SAltitudeEvent_with_instance() {
        setup_global_test_eop();

        let event = SAltitudeEvent::<6, 0>::new(500e3, "Altitude Check", EventDirection::Any)
            .with_instance(3);

        assert_eq!(event.name(), "Altitude Check 3");
    }

    #[test]
    fn test_SAltitudeEvent_with_tolerances() {
        setup_global_test_eop();

        // Default tolerances from SValueEvent
        let event = SAltitudeEvent::<6, 0>::new(500e3, "Test", EventDirection::Any);
        assert_eq!(event.time_tolerance(), 1e-6);
        assert_eq!(event.value_tolerance(), 1e-9);

        // Custom tolerances
        let event = SAltitudeEvent::<6, 0>::new(500e3, "Test", EventDirection::Any)
            .with_tolerances(1e-4, 1e-6);

        assert_eq!(event.time_tolerance(), 1e-4);
        assert_eq!(event.value_tolerance(), 1e-6);
    }

    #[test]
    fn test_SAltitudeEvent_time_tolerance() {
        setup_global_test_eop();

        let event = SAltitudeEvent::<6, 0>::new(500e3, "Test", EventDirection::Any)
            .with_tolerances(5e-5, 1e-9);

        assert_eq!(event.time_tolerance(), 5e-5);
    }

    #[test]
    fn test_SAltitudeEvent_value_tolerance() {
        setup_global_test_eop();

        let event = SAltitudeEvent::<6, 0>::new(500e3, "Test", EventDirection::Any)
            .with_tolerances(1e-6, 5e-8);

        assert_eq!(event.value_tolerance(), 5e-8);
    }

    #[test]
    fn test_SAltitudeEvent_with_step_reduction_factor() {
        setup_global_test_eop();

        // Default
        let event = SAltitudeEvent::<6, 0>::new(500e3, "Test", EventDirection::Any);
        assert_eq!(event.step_reduction_factor(), 0.2);

        // Custom
        let event = SAltitudeEvent::<6, 0>::new(500e3, "Test", EventDirection::Any)
            .with_step_reduction_factor(0.15);
        assert_eq!(event.step_reduction_factor(), 0.15);
    }

    #[test]
    fn test_SAltitudeEvent_step_reduction_factor() {
        setup_global_test_eop();

        let event = SAltitudeEvent::<6, 0>::new(500e3, "Test", EventDirection::Any)
            .with_step_reduction_factor(0.1);

        assert_eq!(event.step_reduction_factor(), 0.1);
    }

    #[test]
    fn test_SAltitudeEvent_with_callback() {
        setup_global_test_eop();

        let called = Arc::new(AtomicBool::new(false));
        let called_clone = called.clone();

        let callback: SEventCallback<6, 0> = Box::new(move |_t, _state, _params| {
            called_clone.store(true, Ordering::SeqCst);
            (None, None, EventAction::Continue)
        });

        let event =
            SAltitudeEvent::<6, 0>::new(500e3, "Test", EventDirection::Any).with_callback(callback);

        // Callback should exist
        assert!(event.callback().is_some());

        // Execute callback
        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state = Vector6::zeros();
        if let Some(cb) = event.callback() {
            cb(epoch, &state, None);
        }
        assert!(called.load(Ordering::SeqCst));
    }

    #[test]
    fn test_SAltitudeEvent_callback_none() {
        setup_global_test_eop();

        let event = SAltitudeEvent::<6, 0>::new(500e3, "Test", EventDirection::Any);
        assert!(event.callback().is_none());
    }

    #[test]
    fn test_SAltitudeEvent_set_terminal() {
        setup_global_test_eop();

        let event = SAltitudeEvent::<6, 0>::new(500e3, "Test", EventDirection::Any);
        assert_eq!(event.action(), EventAction::Continue);

        let event = SAltitudeEvent::<6, 0>::new(500e3, "Test", EventDirection::Any).set_terminal();
        assert_eq!(event.action(), EventAction::Stop);
    }

    #[test]
    fn test_SAltitudeEvent_action_continue() {
        setup_global_test_eop();

        let event = SAltitudeEvent::<6, 0>::new(500e3, "Test", EventDirection::Any);
        assert_eq!(event.action(), EventAction::Continue);
    }

    #[test]
    fn test_SAltitudeEvent_action_stop() {
        setup_global_test_eop();

        let event = SAltitudeEvent::<6, 0>::new(500e3, "Test", EventDirection::Any).set_terminal();
        assert_eq!(event.action(), EventAction::Stop);
    }

    #[test]
    fn test_SAltitudeEvent_direction_increasing() {
        setup_global_test_eop();

        let event = SAltitudeEvent::<6, 0>::new(500e3, "Ascending", EventDirection::Increasing);
        assert_eq!(event.direction(), EventDirection::Increasing);
    }

    #[test]
    fn test_SAltitudeEvent_direction_decreasing() {
        setup_global_test_eop();

        let event = SAltitudeEvent::<6, 0>::new(500e3, "Descending", EventDirection::Decreasing);
        assert_eq!(event.direction(), EventDirection::Decreasing);
    }

    #[test]
    fn test_SAltitudeEvent_direction_any() {
        setup_global_test_eop();

        let event = SAltitudeEvent::<6, 0>::new(500e3, "Any", EventDirection::Any);
        assert_eq!(event.direction(), EventDirection::Any);
    }

    #[test]
    fn test_SAltitudeEvent_7d_state() {
        setup_global_test_eop();

        // Test with 7D state (e.g., with mass)
        let event = SAltitudeEvent::<7, 4>::new(500e3, "7D State", EventDirection::Decreasing);

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state: SVector<f64, 7> =
            SVector::from([R_EARTH + 600e3, 0.0, 0.0, 0.0, 7.5e3, 0.0, 1000.0]);

        let altitude = event.evaluate(epoch, &state, None);
        // Altitude should be approximately 600 km
        assert!(altitude > 500e3);
    }

    #[test]
    fn test_SAltitudeEvent_builder_chaining() {
        setup_global_test_eop();

        let callback: SEventCallback<6, 0> =
            Box::new(|_t, _state, _params| (None, None, EventAction::Stop));

        let event = SAltitudeEvent::<6, 0>::new(200e3, "Reentry", EventDirection::Decreasing)
            .with_instance(1)
            .with_tolerances(1e-5, 1e-8)
            .with_step_reduction_factor(0.1)
            .with_callback(callback)
            .set_terminal();

        assert_eq!(event.name(), "Reentry 1");
        assert_eq!(event.time_tolerance(), 1e-5);
        assert_eq!(event.value_tolerance(), 1e-8);
        assert_eq!(event.step_reduction_factor(), 0.1);
        assert!(event.callback().is_some());
        assert_eq!(event.action(), EventAction::Stop);
    }

    // =========================================================================
    // DAltitudeEvent Tests
    // =========================================================================

    #[test]
    fn test_DAltitudeEvent_new() {
        setup_global_test_eop();

        let event = DAltitudeEvent::new(500e3, "Low Altitude", EventDirection::Decreasing);

        assert_eq!(event.name(), "Low Altitude");
        assert_eq!(event.target_value(), 500e3);
        assert_eq!(event.direction(), EventDirection::Decreasing);
        assert_eq!(event.action(), EventAction::Continue);
    }

    #[test]
    fn test_DAltitudeEvent_evaluate() {
        setup_global_test_eop();

        let value = 500e3;
        let event = DAltitudeEvent::new(value, "Altitude Test", EventDirection::Any);

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);

        // Low altitude state (below value)
        let state_low = DVector::from_vec(vec![6000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let val_low = event.evaluate(epoch, &state_low, None);
        assert!(val_low < 0.0);

        // High altitude state (above value)
        let state_high = DVector::from_vec(vec![R_EARTH + 1000e3, 0.0, 0.0, 0.0, 7.0e3, 0.0]);
        let val_high = event.evaluate(epoch, &state_high, None);
        assert!(val_high > 0.0);
    }

    #[test]
    fn test_DAltitudeEvent_target_value() {
        setup_global_test_eop();

        let event = DAltitudeEvent::new(350e3, "Reentry", EventDirection::Decreasing);
        assert_eq!(event.target_value(), 350e3);
    }

    #[test]
    fn test_DAltitudeEvent_name() {
        setup_global_test_eop();

        let event = DAltitudeEvent::new(500e3, "Test Name", EventDirection::Any);
        assert_eq!(event.name(), "Test Name");
    }

    #[test]
    fn test_DAltitudeEvent_with_instance() {
        setup_global_test_eop();

        let event =
            DAltitudeEvent::new(500e3, "Altitude Check", EventDirection::Any).with_instance(2);

        assert_eq!(event.name(), "Altitude Check 2");
    }

    #[test]
    fn test_DAltitudeEvent_with_tolerances() {
        setup_global_test_eop();

        let event =
            DAltitudeEvent::new(500e3, "Test", EventDirection::Any).with_tolerances(1e-4, 1e-7);

        assert_eq!(event.time_tolerance(), 1e-4);
        assert_eq!(event.value_tolerance(), 1e-7);
    }

    #[test]
    fn test_DAltitudeEvent_time_tolerance() {
        setup_global_test_eop();

        let event =
            DAltitudeEvent::new(500e3, "Test", EventDirection::Any).with_tolerances(2e-5, 1e-9);

        assert_eq!(event.time_tolerance(), 2e-5);
    }

    #[test]
    fn test_DAltitudeEvent_value_tolerance() {
        setup_global_test_eop();

        let event =
            DAltitudeEvent::new(500e3, "Test", EventDirection::Any).with_tolerances(1e-6, 3e-8);

        assert_eq!(event.value_tolerance(), 3e-8);
    }

    #[test]
    fn test_DAltitudeEvent_with_step_reduction_factor() {
        setup_global_test_eop();

        // Default
        let event = DAltitudeEvent::new(500e3, "Test", EventDirection::Any);
        assert_eq!(event.step_reduction_factor(), 0.2);

        // Custom
        let event = DAltitudeEvent::new(500e3, "Test", EventDirection::Any)
            .with_step_reduction_factor(0.25);
        assert_eq!(event.step_reduction_factor(), 0.25);
    }

    #[test]
    fn test_DAltitudeEvent_step_reduction_factor() {
        setup_global_test_eop();

        let event = DAltitudeEvent::new(500e3, "Test", EventDirection::Any)
            .with_step_reduction_factor(0.12);

        assert_eq!(event.step_reduction_factor(), 0.12);
    }

    #[test]
    fn test_DAltitudeEvent_with_callback() {
        setup_global_test_eop();

        let called = Arc::new(AtomicBool::new(false));
        let called_clone = called.clone();

        let callback: DEventCallback = Box::new(move |_t, _state, _params| {
            called_clone.store(true, Ordering::SeqCst);
            (None, None, EventAction::Continue)
        });

        let event = DAltitudeEvent::new(500e3, "Test", EventDirection::Any).with_callback(callback);

        assert!(event.callback().is_some());
    }

    #[test]
    fn test_DAltitudeEvent_callback_none() {
        setup_global_test_eop();

        let event = DAltitudeEvent::new(500e3, "Test", EventDirection::Any);
        assert!(event.callback().is_none());
    }

    #[test]
    fn test_DAltitudeEvent_set_terminal() {
        setup_global_test_eop();

        let event = DAltitudeEvent::new(500e3, "Test", EventDirection::Any);
        assert_eq!(event.action(), EventAction::Continue);

        let event = DAltitudeEvent::new(500e3, "Test", EventDirection::Any).set_terminal();
        assert_eq!(event.action(), EventAction::Stop);
    }

    #[test]
    fn test_DAltitudeEvent_action_continue() {
        setup_global_test_eop();

        let event = DAltitudeEvent::new(500e3, "Test", EventDirection::Any);
        assert_eq!(event.action(), EventAction::Continue);
    }

    #[test]
    fn test_DAltitudeEvent_action_stop() {
        setup_global_test_eop();

        let event = DAltitudeEvent::new(500e3, "Test", EventDirection::Any).set_terminal();
        assert_eq!(event.action(), EventAction::Stop);
    }

    #[test]
    fn test_DAltitudeEvent_direction_increasing() {
        setup_global_test_eop();

        let event = DAltitudeEvent::new(500e3, "Ascending", EventDirection::Increasing);
        assert_eq!(event.direction(), EventDirection::Increasing);
    }

    #[test]
    fn test_DAltitudeEvent_direction_decreasing() {
        setup_global_test_eop();

        let event = DAltitudeEvent::new(500e3, "Descending", EventDirection::Decreasing);
        assert_eq!(event.direction(), EventDirection::Decreasing);
    }

    #[test]
    fn test_DAltitudeEvent_direction_any() {
        setup_global_test_eop();

        let event = DAltitudeEvent::new(500e3, "Any", EventDirection::Any);
        assert_eq!(event.direction(), EventDirection::Any);
    }

    #[test]
    fn test_DAltitudeEvent_builder_chaining() {
        setup_global_test_eop();

        let callback: DEventCallback =
            Box::new(|_t, _state, _params| (None, None, EventAction::Stop));

        let event = DAltitudeEvent::new(200e3, "Reentry", EventDirection::Decreasing)
            .with_instance(1)
            .with_tolerances(1e-5, 1e-8)
            .with_step_reduction_factor(0.1)
            .with_callback(callback)
            .set_terminal();

        assert_eq!(event.name(), "Reentry 1");
        assert_eq!(event.time_tolerance(), 1e-5);
        assert_eq!(event.value_tolerance(), 1e-8);
        assert_eq!(event.step_reduction_factor(), 0.1);
        assert!(event.callback().is_some());
        assert_eq!(event.action(), EventAction::Stop);
    }

    // =========================================================================
    // Orbital Element Event Tests
    // =========================================================================

    #[test]
    fn test_SSemiMajorAxisEvent_new() {
        let event =
            SSemiMajorAxisEvent::<6, 0>::new(7000e3, "SMA Check", EventDirection::Increasing);
        assert_eq!(event.name(), "SMA Check");
        assert_eq!(event.target_value(), 7000e3);
        assert_eq!(event.direction(), EventDirection::Increasing);
    }

    #[test]
    fn test_SSemiMajorAxisEvent_evaluate() {
        let event = SSemiMajorAxisEvent::<6, 0>::new(7000e3, "SMA Check", EventDirection::Any);
        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);

        // Circular orbit: v = sqrt(GM/r) for r = 7000 km
        // GM_EARTH = 3.986004418e14 m^3/s^2
        // v = sqrt(3.986e14 / 7e6) = 7546.05 m/s
        let state = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7546.05, 0.0);
        let val = event.evaluate(epoch, &state, None);
        // evaluate() returns raw SMA value
        // For a circular orbit at r=7000km, SMA should be very close to 7000 km
        assert!((val - 7000e3).abs() < 1000.0); // Within 1 km of expected SMA
    }

    #[test]
    fn test_DSemiMajorAxisEvent_new() {
        let event = DSemiMajorAxisEvent::new(7000e3, "SMA Check", EventDirection::Decreasing);
        assert_eq!(event.name(), "SMA Check");
        assert_eq!(event.target_value(), 7000e3);
        assert_eq!(event.direction(), EventDirection::Decreasing);
    }

    #[test]
    fn test_SEccentricityEvent_new() {
        let event = SEccentricityEvent::<6, 0>::new(0.1, "Ecc value", EventDirection::Increasing);
        assert_eq!(event.name(), "Ecc value");
        assert_eq!(event.target_value(), 0.1);
        assert_eq!(event.direction(), EventDirection::Increasing);
    }

    #[test]
    fn test_SEccentricityEvent_evaluate() {
        let event = SEccentricityEvent::<6, 0>::new(0.1, "Ecc Test", EventDirection::Any);
        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);

        // Circular orbit: v = sqrt(GM/r) at r = 7000 km gives e ≈ 0
        let state = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7546.05, 0.0);
        let val = event.evaluate(epoch, &state, None);
        // evaluate() returns raw eccentricity value
        // For a circular orbit, eccentricity should be very small (close to 0)
        assert!(val >= 0.0); // Eccentricity is always non-negative
        assert!(val < 0.01); // Should be nearly circular
    }

    #[test]
    fn test_DEccentricityEvent_new() {
        let event = DEccentricityEvent::new(0.5, "Ecc Check", EventDirection::Any);
        assert_eq!(event.name(), "Ecc Check");
        assert_eq!(event.target_value(), 0.5);
    }

    #[test]
    fn test_SInclinationEvent_new() {
        let inc_rad = std::f64::consts::PI / 4.0; // 45 degrees
        let event = SInclinationEvent::<6, 0>::new(
            inc_rad,
            "Inc value",
            EventDirection::Increasing,
            AngleFormat::Radians,
        );
        assert_eq!(event.name(), "Inc value");
        assert_eq!(event.target_value(), inc_rad);
    }

    #[test]
    fn test_DInclinationEvent_new() {
        let event =
            DInclinationEvent::new(45.0, "Inc Check", EventDirection::Any, AngleFormat::Degrees);
        assert_eq!(event.name(), "Inc Check");
    }

    #[test]
    fn test_SArgumentOfPerigeeEvent_new() {
        let event = SArgumentOfPerigeeEvent::<6, 0>::new(
            std::f64::consts::PI / 2.0,
            "AoP",
            EventDirection::Increasing,
            AngleFormat::Radians,
        );
        assert_eq!(event.name(), "AoP");
    }

    #[test]
    fn test_DArgumentOfPerigeeEvent_new() {
        let event =
            DArgumentOfPerigeeEvent::new(90.0, "AoP", EventDirection::Any, AngleFormat::Degrees);
        assert_eq!(event.name(), "AoP");
    }

    #[test]
    fn test_SMeanAnomalyEvent_new() {
        let event = SMeanAnomalyEvent::<6, 0>::new(
            0.0,
            "Periapsis",
            EventDirection::Increasing,
            AngleFormat::Radians,
        );
        assert_eq!(event.name(), "Periapsis");
        assert_eq!(event.target_value(), 0.0);
    }

    #[test]
    fn test_DMeanAnomalyEvent_new() {
        let event =
            DMeanAnomalyEvent::new(180.0, "Apoapsis", EventDirection::Any, AngleFormat::Degrees);
        assert_eq!(event.name(), "Apoapsis");
    }

    #[test]
    fn test_SEccentricAnomalyEvent_new() {
        let event = SEccentricAnomalyEvent::<6, 0>::new(
            0.0,
            "Periapsis EA",
            EventDirection::Increasing,
            AngleFormat::Radians,
        );
        assert_eq!(event.name(), "Periapsis EA");
    }

    #[test]
    fn test_DEccentricAnomalyEvent_new() {
        let event =
            DEccentricAnomalyEvent::new(0.0, "EA Check", EventDirection::Any, AngleFormat::Degrees);
        assert_eq!(event.name(), "EA Check");
    }

    #[test]
    fn test_STrueAnomalyEvent_new() {
        let event = STrueAnomalyEvent::<6, 0>::new(
            0.0,
            "Periapsis TA",
            EventDirection::Increasing,
            AngleFormat::Radians,
        );
        assert_eq!(event.name(), "Periapsis TA");
    }

    #[test]
    fn test_DTrueAnomalyEvent_new() {
        let event = DTrueAnomalyEvent::new(
            180.0,
            "Apoapsis TA",
            EventDirection::Any,
            AngleFormat::Degrees,
        );
        assert_eq!(event.name(), "Apoapsis TA");
    }

    #[test]
    fn test_SArgumentOfLatitudeEvent_new() {
        let event = SArgumentOfLatitudeEvent::<6, 0>::new(
            0.5,
            "AoL Check",
            EventDirection::Increasing,
            AngleFormat::Radians,
        );
        assert_eq!(event.name(), "AoL Check");
        assert_eq!(event.target_value(), 0.5);
    }

    #[test]
    fn test_DArgumentOfLatitudeEvent_new() {
        let event =
            DArgumentOfLatitudeEvent::new(30.0, "AoL", EventDirection::Any, AngleFormat::Degrees);
        assert_eq!(event.name(), "AoL");
    }

    // =========================================================================
    // Node Crossing Event Tests
    // =========================================================================

    #[test]
    fn test_SAscendingNodeEvent_new() {
        let event = SAscendingNodeEvent::<6, 0>::new("Ascending Node");
        assert_eq!(event.name(), "Ascending Node");
        assert_eq!(event.target_value(), 0.0); // Zero crossing
        assert_eq!(event.direction(), EventDirection::Increasing);
    }

    #[test]
    fn test_SAscendingNodeEvent_builder_chaining() {
        let event = SAscendingNodeEvent::<6, 0>::new("Asc Node")
            .with_instance(2)
            .with_tolerances(1e-5, 1e-8)
            .with_step_reduction_factor(0.15);
        assert_eq!(event.name(), "Asc Node 2");
        assert_eq!(event.time_tolerance(), 1e-5);
        assert_eq!(event.step_reduction_factor(), 0.15);
    }

    #[test]
    fn test_DAscendingNodeEvent_new() {
        let event = DAscendingNodeEvent::new("Ascending");
        assert_eq!(event.name(), "Ascending");
        assert_eq!(event.target_value(), 0.0);
    }

    #[test]
    fn test_SDescendingNodeEvent_new() {
        let event = SDescendingNodeEvent::<6, 0>::new("Descending Node");
        assert_eq!(event.name(), "Descending Node");
        assert_eq!(event.target_value(), 0.0); // Zero crossing (shifted)
        assert_eq!(event.direction(), EventDirection::Increasing);
    }

    #[test]
    fn test_DDescendingNodeEvent_new() {
        let event = DDescendingNodeEvent::new("Descending");
        assert_eq!(event.name(), "Descending");
    }

    // =========================================================================
    // State-Derived Event Tests
    // =========================================================================

    #[test]
    fn test_SSpeedEvent_new() {
        let event = SSpeedEvent::<6, 0>::new(7500.0, "Speed value", EventDirection::Increasing);
        assert_eq!(event.name(), "Speed value");
        assert_eq!(event.target_value(), 7500.0);
    }

    #[test]
    fn test_SSpeedEvent_evaluate() {
        let event = SSpeedEvent::<6, 0>::new(7000.0, "Speed", EventDirection::Any);
        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);

        // State with known velocity: [3000, 4000, 0] => |v| = 5000 m/s
        let state = Vector6::new(7000e3, 0.0, 0.0, 3000.0, 4000.0, 0.0);
        let val = event.evaluate(epoch, &state, None);
        // evaluate() returns raw speed value (5000), not relative to value
        assert!((val - 5000.0).abs() < 1e-6);
    }

    #[test]
    fn test_DSpeedEvent_new() {
        let event = DSpeedEvent::new(8000.0, "Speed", EventDirection::Decreasing);
        assert_eq!(event.name(), "Speed");
        assert_eq!(event.target_value(), 8000.0);
    }

    #[test]
    fn test_SLongitudeEvent_new() {
        setup_global_test_eop();
        let event = SLongitudeEvent::<6, 0>::new(
            0.0,
            "Prime Meridian",
            EventDirection::Any,
            AngleFormat::Degrees,
        );
        assert_eq!(event.name(), "Prime Meridian");
        assert_eq!(event.target_value(), 0.0);
    }

    #[test]
    fn test_DLongitudeEvent_new() {
        setup_global_test_eop();
        let event = DLongitudeEvent::new(
            45.0,
            "Lon Check",
            EventDirection::Increasing,
            AngleFormat::Degrees,
        );
        assert_eq!(event.name(), "Lon Check");
    }

    #[test]
    fn test_SLatitudeEvent_new() {
        setup_global_test_eop();
        let event = SLatitudeEvent::<6, 0>::new(
            30.0,
            "Latitude Check",
            EventDirection::Any,
            AngleFormat::Degrees,
        );
        assert_eq!(event.name(), "Latitude Check");
        assert_eq!(event.target_value(), 30.0_f64.to_radians());
    }

    #[test]
    fn test_DLatitudeEvent_new() {
        setup_global_test_eop();
        let event = DLatitudeEvent::new(
            0.0,
            "Equator",
            EventDirection::Increasing,
            AngleFormat::Degrees,
        );
        assert_eq!(event.name(), "Equator");
    }

    // =========================================================================
    // Eclipse/Shadow Event Tests
    // =========================================================================

    #[test]
    fn test_SUmbraEvent_new_without_source() {
        let event = SUmbraEvent::<6, 0>::new("Enter Umbra", EdgeType::RisingEdge, None);
        assert_eq!(event.name(), "Enter Umbra");
    }

    #[test]
    fn test_SUmbraEvent_new_with_source() {
        let event = SUmbraEvent::<6, 0>::new(
            "Umbra DE440s",
            EdgeType::FallingEdge,
            Some(crate::propagators::EphemerisSource::DE440s),
        );
        assert_eq!(event.name(), "Umbra DE440s");
    }

    #[test]
    fn test_DUmbraEvent_new() {
        let event = DUmbraEvent::new("Umbra", EdgeType::AnyEdge, None);
        assert_eq!(event.name(), "Umbra");
    }

    #[test]
    fn test_SPenumbraEvent_new() {
        let event = SPenumbraEvent::<6, 0>::new("Penumbra", EdgeType::RisingEdge, None);
        assert_eq!(event.name(), "Penumbra");
    }

    #[test]
    fn test_DPenumbraEvent_new() {
        let event = DPenumbraEvent::new("Penumbra", EdgeType::FallingEdge, None);
        assert_eq!(event.name(), "Penumbra");
    }

    #[test]
    fn test_SEclipseEvent_new() {
        let event = SEclipseEvent::<6, 0>::new("Eclipse", EdgeType::AnyEdge, None);
        assert_eq!(event.name(), "Eclipse");
    }

    #[test]
    fn test_SEclipseEvent_with_ephemeris_sources() {
        use crate::propagators::EphemerisSource;

        // Test all ephemeris source variants
        let event1 = SEclipseEvent::<6, 0>::new(
            "Eclipse LP",
            EdgeType::RisingEdge,
            Some(EphemerisSource::LowPrecision),
        );
        assert_eq!(event1.name(), "Eclipse LP");

        let event2 = SEclipseEvent::<6, 0>::new(
            "Eclipse DE440s",
            EdgeType::RisingEdge,
            Some(EphemerisSource::DE440s),
        );
        assert_eq!(event2.name(), "Eclipse DE440s");

        let event3 = SEclipseEvent::<6, 0>::new(
            "Eclipse DE440",
            EdgeType::RisingEdge,
            Some(EphemerisSource::DE440),
        );
        assert_eq!(event3.name(), "Eclipse DE440");
    }

    #[test]
    fn test_DEclipseEvent_new() {
        let event = DEclipseEvent::new("Eclipse", EdgeType::FallingEdge, None);
        assert_eq!(event.name(), "Eclipse");
    }

    #[test]
    fn test_SSunlitEvent_new() {
        let event = SSunlitEvent::<6, 0>::new("Sunlit", EdgeType::RisingEdge, None);
        assert_eq!(event.name(), "Sunlit");
    }

    #[test]
    fn test_DSunlitEvent_new() {
        let event = DSunlitEvent::new("Sunlit", EdgeType::AnyEdge, None);
        assert_eq!(event.name(), "Sunlit");
    }

    #[test]
    fn test_SUmbraEvent_builder_chaining() {
        let event = SUmbraEvent::<6, 0>::new("Umbra", EdgeType::RisingEdge, None)
            .with_instance(1)
            .with_tolerances(1e-5, 1e-8)
            .set_terminal();
        assert_eq!(event.name(), "Umbra 1");
        assert_eq!(event.time_tolerance(), 1e-5);
        assert_eq!(event.action(), EventAction::Stop);
    }
}
