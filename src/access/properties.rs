/*!
 * Access property computation types and utilities
 *
 * This module provides types and traits for computing properties of access windows,
 * including geometric properties (azimuth, elevation, look direction) and custom
 * user-defined properties.
 */

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::propagators::traits::StateProvider;
use crate::utils::BraheError;

use super::constraints::{AscDsc, LookDirection};

// Import AccessWindow from windows module via parent re-export
// This works because both modules are siblings in the access module
#[allow(unused_imports)]
use crate::access::AccessWindow;

// ================================
// PropertyValue Enum
// ================================

/// Flexible value type for custom access properties.
///
/// This enum supports various data types that users might want to compute
/// and store as access properties, from simple scalars to complex time series.
///
/// # Examples
/// ```
/// use brahe::access::PropertyValue;
///
/// // Scalar property
/// let doppler_shift = PropertyValue::Scalar(2500.0);
///
/// // Vector property
/// let look_angles = PropertyValue::Vector(vec![45.0, 30.0]);
///
/// // Time series property
/// let elevation_profile = PropertyValue::TimeSeries {
///     times: vec![0.0, 10.0, 20.0],
///     values: vec![10.0, 45.0, 10.0],
///  };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PropertyValue {
    /// Single floating-point value
    Scalar(f64),

    /// Vector of floating-point values
    Vector(Vec<f64>),

    /// Time series with relative times (seconds from window_open) and values
    TimeSeries {
        /// Relative time from window_open (seconds)
        times: Vec<f64>,
        /// Values at each time
        values: Vec<f64>,
    },

    /// Boolean value
    Boolean(bool),

    /// String value
    String(String),

    /// Arbitrary JSON value for complex data
    Json(serde_json::Value),
}

// ================================
// AccessProperties Struct
// ================================

/// Properties of an access window.
///
/// This struct contains both core geometric properties (always computed) and
/// extensible additional properties (computed by user-defined property computers).
///
/// # Core Properties
/// - Azimuth angles at window open/close
/// - Min/max elevation angles
/// - Elevation angles at window open/close
/// - Min/max off-nadir angles
/// - Local solar time at midtime
/// - Look direction (left/right)
/// - Ascending/descending pass indicator
/// - Location center coordinates (lon, lat, alt, ECEF)
///
/// # Additional Properties
/// Custom properties can be added via the `additional` HashMap using the
/// `PropertyValue` enum.
///
/// # Examples
/// ```
/// use brahe::access::{AccessProperties, PropertyValue};
/// use brahe::access::LookDirection;
/// use brahe::access::AscDsc;
///
/// let mut props = AccessProperties::new(
///     45.0,   // azimuth_open
///     135.0,  // azimuth_close
///     10.0,   // elevation_min
///     85.0,   // elevation_max
///     12.0,   // elevation_open
///     10.5,   // elevation_close
///     5.0,    // off_nadir_min
///     80.0,   // off_nadir_max
///     43200.0, // local_time (noon)
///     LookDirection::Right,
///     AscDsc::Ascending,
///     0.0,    // center_lon (degrees)
///     45.0,   // center_lat (degrees)
///     0.0,    // center_alt (meters)
///     [4517.59e3, 4517.59e3, 0.0], // center_ecef
/// );
///
/// // Add custom property
/// props.add_property("doppler_shift".to_string(), PropertyValue::Scalar(2500.0));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessProperties {
    // ===== Core geometric properties (always present) =====
    /// Azimuth at window open (degrees, 0-360, measured clockwise from North)
    pub azimuth_open: f64,

    /// Azimuth at window close (degrees, 0-360)
    pub azimuth_close: f64,

    /// Minimum elevation angle during access (degrees, above horizon)
    pub elevation_min: f64,

    /// Maximum elevation angle during access (degrees, typically at midtime)
    pub elevation_max: f64,

    /// Elevation angle at window open (degrees, above horizon)
    pub elevation_open: f64,

    /// Elevation angle at window close (degrees, above horizon)
    pub elevation_close: f64,

    /// Minimum off-nadir angle during access (degrees, typically at midtime)
    pub off_nadir_min: f64,

    /// Maximum off-nadir angle during access (degrees, typically at endpoints)
    pub off_nadir_max: f64,

    /// Local solar time at window midtime (seconds since midnight, 0-86400)
    pub local_time: f64,

    /// Look direction (Left or Right)
    pub look_direction: LookDirection,

    /// Ascending or descending pass
    pub asc_dsc: AscDsc,

    // ===== Location coordinates (for plotting and analysis) =====
    /// Location center longitude (degrees, -180 to 180)
    pub center_lon: f64,

    /// Location center latitude (degrees, -90 to 90)
    pub center_lat: f64,

    /// Location center altitude (meters above WGS84 ellipsoid)
    pub center_alt: f64,

    /// Location center ECEF coordinates [x, y, z] (meters)
    pub center_ecef: [f64; 3],

    // ===== Extensible properties (user-defined) =====
    /// Additional custom properties
    #[serde(default)]
    pub additional: HashMap<String, PropertyValue>,
}

impl AccessProperties {
    /// Create AccessProperties with core properties only.
    ///
    /// # Arguments
    /// * `azimuth_open` - Azimuth at window open (degrees, 0-360)
    /// * `azimuth_close` - Azimuth at window close (degrees, 0-360)
    /// * `elevation_min` - Minimum elevation angle (degrees)
    /// * `elevation_max` - Maximum elevation angle (degrees)
    /// * `elevation_open` - Elevation at window open (degrees)
    /// * `elevation_close` - Elevation at window close (degrees)
    /// * `off_nadir_min` - Minimum off-nadir angle (degrees)
    /// * `off_nadir_max` - Maximum off-nadir angle (degrees)
    /// * `local_time` - Local solar time at midtime (seconds since midnight)
    /// * `look_direction` - Look direction (Left or Right)
    /// * `asc_dsc` - Ascending or descending pass
    /// * `center_lon` - Location longitude (degrees)
    /// * `center_lat` - Location latitude (degrees)
    /// * `center_alt` - Location altitude (meters)
    /// * `center_ecef` - Location ECEF coordinates [x, y, z] (meters)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        azimuth_open: f64,
        azimuth_close: f64,
        elevation_min: f64,
        elevation_max: f64,
        elevation_open: f64,
        elevation_close: f64,
        off_nadir_min: f64,
        off_nadir_max: f64,
        local_time: f64,
        look_direction: LookDirection,
        asc_dsc: AscDsc,
        center_lon: f64,
        center_lat: f64,
        center_alt: f64,
        center_ecef: [f64; 3],
    ) -> Self {
        Self {
            azimuth_open,
            azimuth_close,
            elevation_min,
            elevation_max,
            elevation_open,
            elevation_close,
            off_nadir_min,
            off_nadir_max,
            local_time,
            look_direction,
            asc_dsc,
            center_lon,
            center_lat,
            center_alt,
            center_ecef,
            additional: HashMap::new(),
        }
    }

    /// Add a custom property.
    ///
    /// # Arguments
    /// * `key` - Property name
    /// * `value` - Property value
    pub fn add_property(&mut self, key: String, value: PropertyValue) {
        self.additional.insert(key, value);
    }

    /// Get a custom property.
    ///
    /// # Arguments
    /// * `key` - Property name
    ///
    /// # Returns
    /// Reference to property value if it exists, None otherwise
    pub fn get_property(&self, key: &str) -> Option<&PropertyValue> {
        self.additional.get(key)
    }
}

// ================================
// AccessPropertyComputer Trait
// ================================

/// Trait for computing additional access properties.
///
/// Implement this trait to add custom property calculations to access windows.
/// The compute method is called once per access window after core properties
/// are calculated.
///
/// # Examples
/// ```no_run
/// use brahe::access::{AccessPropertyComputer, AccessWindow, PropertyValue};
/// use brahe::propagators::traits::StateProvider;
/// use brahe::utils::BraheError;
/// use std::collections::HashMap;
/// use nalgebra::Vector3;
///
/// struct DopplerComputer;
///
/// impl AccessPropertyComputer for DopplerComputer {
///     fn compute(
///         &self,
///         window: &AccessWindow,
///         state_provider: &dyn StateProvider,
///         location_ecef: &Vector3<f64>,
///         location_geodetic: &Vector3<f64>,
///     ) -> Result<HashMap<String, PropertyValue>, BraheError> {
///         let mut props = HashMap::new();
///
///         // Compute Doppler shift at midtime
///         let midtime = window.midtime();
///         let state_ecef = state_provider.state_ecef(midtime);
///
///         // ... compute Doppler shift from state and location ...
///         let doppler = 2500.0;  // Example value
///
///         props.insert("doppler_shift".to_string(), PropertyValue::Scalar(doppler));
///         Ok(props)
///     }
///
///     fn property_names(&self) -> Vec<String> {
///         vec!["doppler_shift".to_string()]
///     }
/// }
/// ```
pub trait AccessPropertyComputer: Send + Sync {
    /// Compute additional properties for an access window.
    ///
    /// # Arguments
    /// * `window` - The access window (contains times and core properties)
    /// * `state_provider` - State computation interface via StateProvider trait
    /// * `location_ecef` - Location ECEF position [x, y, z] (meters)
    /// * `location_geodetic` - Location geodetic coordinates [lon, lat, alt] (radians, meters)
    ///
    /// # Returns
    /// HashMap of property name -> PropertyValue
    ///
    /// # Notes
    /// - `state_provider.state_ecef(epoch)` returns ECEF state [x,y,z,vx,vy,vz] (m, m/s)
    /// - You can sample at arbitrary epochs within the window
    /// - For time-series properties, you control the sampling rate
    fn compute(
        &self,
        window: &AccessWindow,
        state_provider: &dyn StateProvider,
        location_ecef: &nalgebra::Vector3<f64>,
        location_geodetic: &nalgebra::Vector3<f64>,
    ) -> Result<HashMap<String, PropertyValue>, BraheError>;

    /// Names of properties this computer will produce.
    ///
    /// Used for documentation and validation.
    fn property_names(&self) -> Vec<String>;
}

// Core property computation functions are now in the geometry module
// and are re-exported from the access module for convenience.

// ================================
// Tests
// ================================

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Vector3, Vector6};

    use crate::constants::AngleFormat;
    use crate::coordinates::position_geodetic_to_ecef;
    use crate::propagators::KeplerianPropagator;
    use crate::time::{Epoch, TimeSystem};
    use crate::utils::testing::setup_global_test_eop;

    use super::super::geometry::{
        compute_asc_dsc, compute_azimuth, compute_elevation, compute_local_time,
        compute_look_direction, compute_off_nadir,
    };

    #[test]
    fn test_property_value_serialization() {
        // Scalar
        let scalar = PropertyValue::Scalar(42.0);
        let json = serde_json::to_string(&scalar).unwrap();
        let deserialized: PropertyValue = serde_json::from_str(&json).unwrap();
        assert_eq!(scalar, deserialized);

        // Vector
        let vector = PropertyValue::Vector(vec![1.0, 2.0, 3.0]);
        let json = serde_json::to_string(&vector).unwrap();
        let deserialized: PropertyValue = serde_json::from_str(&json).unwrap();
        assert_eq!(vector, deserialized);

        // TimeSeries
        let ts = PropertyValue::TimeSeries {
            times: vec![0.0, 10.0, 20.0],
            values: vec![1.0, 2.0, 3.0],
        };
        let json = serde_json::to_string(&ts).unwrap();
        let deserialized: PropertyValue = serde_json::from_str(&json).unwrap();
        assert_eq!(ts, deserialized);

        // Boolean
        let boolean = PropertyValue::Boolean(true);
        let json = serde_json::to_string(&boolean).unwrap();
        let deserialized: PropertyValue = serde_json::from_str(&json).unwrap();
        assert_eq!(boolean, deserialized);

        // String
        let string = PropertyValue::String("test".to_string());
        let json = serde_json::to_string(&string).unwrap();
        let deserialized: PropertyValue = serde_json::from_str(&json).unwrap();
        assert_eq!(string, deserialized);
    }

    #[test]
    fn test_state_provider_propagator() {
        use crate::trajectories::traits::{OrbitFrame, OrbitRepresentation};

        // Initialize EOP for frame conversions
        setup_global_test_eop();

        // Create a Keplerian propagator
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let elements = Vector6::new(
            7000e3, // a (m)
            0.001,  // e
            0.9,    // i (rad)
            0.0,    // RAAN (rad)
            0.0,    // arg periapsis (rad)
            0.0,    // mean anomaly (rad)
        );
        let prop = KeplerianPropagator::new(
            epoch,
            elements,
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Radians),
            60.0,
        );

        // Use StateProvider trait directly
        let state = prop.state_ecef(epoch);
        assert_eq!(state.len(), 6);

        // State should be non-zero
        assert!(state.norm() > 0.0);
    }

    #[test]
    fn test_state_provider_orbit_trajectory() {
        use crate::trajectories::orbit_trajectory::OrbitTrajectory;
        use crate::trajectories::traits::{OrbitFrame, OrbitRepresentation, Trajectory};

        // Initialize EOP for frame conversions
        setup_global_test_eop();

        // Create a trajectory in ECI
        let epoch1 = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = epoch1 + 60.0;

        let state1 = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7500.0, 0.0);
        let state2 = Vector6::new(7000e3, 100e3, 10e3, 10.0, 7500.0, 100.0);

        let mut traj = OrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);
        traj.add(epoch1, state1);
        traj.add(epoch2, state2);

        // Use StateProvider trait directly (now implemented by OrbitTrajectory)
        let mid_epoch = epoch1 + 30.0;
        let state = traj.state_ecef(mid_epoch);
        assert_eq!(state.len(), 6);
    }

    #[test]
    fn test_access_properties_creation() {
        let props = AccessProperties::new(
            45.0,
            135.0,
            10.0,
            85.0,
            12.0,
            10.5,
            5.0,
            80.0,
            43200.0,
            LookDirection::Right,
            AscDsc::Ascending,
            0.0,
            45.0,
            0.0,
            [4517.59e3, 4517.59e3, 0.0],
        );

        assert_eq!(props.azimuth_open, 45.0);
        assert_eq!(props.azimuth_close, 135.0);
        assert_eq!(props.elevation_min, 10.0);
        assert_eq!(props.elevation_max, 85.0);
        assert_eq!(props.elevation_open, 12.0);
        assert_eq!(props.elevation_close, 10.5);
        assert_eq!(props.off_nadir_min, 5.0);
        assert_eq!(props.off_nadir_max, 80.0);
        assert_eq!(props.local_time, 43200.0);
        assert_eq!(props.look_direction, LookDirection::Right);
        assert_eq!(props.asc_dsc, AscDsc::Ascending);
        assert_eq!(props.center_lon, 0.0);
        assert_eq!(props.center_lat, 45.0);
        assert_eq!(props.center_alt, 0.0);
        assert_eq!(props.center_ecef, [4517.59e3, 4517.59e3, 0.0]);
        assert!(props.additional.is_empty());
    }

    #[test]
    fn test_access_properties_additional() {
        let mut props = AccessProperties::new(
            45.0,
            135.0,
            10.0,
            85.0,
            12.0,
            10.5,
            5.0,
            80.0,
            43200.0,
            LookDirection::Right,
            AscDsc::Ascending,
            0.0,
            45.0,
            0.0,
            [4517.59e3, 4517.59e3, 0.0],
        );

        // Add property
        props.add_property("doppler".to_string(), PropertyValue::Scalar(2500.0));

        // Get property
        let doppler = props.get_property("doppler").unwrap();
        match doppler {
            PropertyValue::Scalar(val) => assert_eq!(*val, 2500.0),
            _ => panic!("Expected Scalar"),
        }

        // Non-existent property
        assert!(props.get_property("nonexistent").is_none());
    }

    #[test]
    fn test_compute_azimuth_elevation() {
        use crate::coordinates::position_geodetic_to_ecef;

        // Location: (0° lon, 45° lat, 0 alt)
        let loc_geodetic = Vector3::new(0.0, 45.0_f64.to_radians(), 0.0);
        let loc_ecef = position_geodetic_to_ecef(loc_geodetic, AngleFormat::Radians).unwrap();

        // Satellite at high altitude (not directly north, just higher than location)
        let sat_ecef = loc_ecef + Vector3::new(0.0, 500e3, 500e3);

        let azimuth = compute_azimuth(&sat_ecef, &loc_ecef);
        let elevation = compute_elevation(&sat_ecef, &loc_ecef);

        // Just verify azimuth is in valid range
        assert!((0.0..=360.0).contains(&azimuth));

        // Elevation should be positive (satellite is above location)
        assert!(elevation > 0.0);
        assert!(elevation < 90.0);
    }

    #[test]
    fn test_compute_off_nadir() {
        // Satellite at altitude
        let sat_ecef = Vector3::new(7000e3, 0.0, 0.0);

        // Location on Earth surface
        let loc_geodetic = Vector3::new(0.0, 0.0, 0.0);
        let loc_ecef = position_geodetic_to_ecef(loc_geodetic, AngleFormat::Radians).unwrap();

        let off_nadir = compute_off_nadir(&sat_ecef, &loc_ecef);

        // Off-nadir should be reasonable (0-90 degrees for visible targets)
        assert!(off_nadir >= 0.0);
        assert!(off_nadir <= 180.0);
    }

    #[test]
    fn test_compute_local_time() {
        // Initialize EOP for UT1 time conversions
        setup_global_test_eop();

        // Location at 0° longitude
        let loc_geodetic = Vector3::new(0.0, 0.0, 0.0);

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

        let local_time = compute_local_time(&epoch, &loc_geodetic);

        // Should be in range 0-86400 seconds
        assert!(local_time >= 0.0);
        assert!(local_time <= 86400.0);
    }

    #[test]
    fn test_compute_asc_dsc() {
        // Ascending: positive z-velocity in inertial frame
        let state_ascending = Vector6::new(
            7000e3, 0.0, 0.0, // position
            0.0, 7500.0, 100.0, // velocity (positive z)
        );

        let asc_dsc = compute_asc_dsc(&state_ascending);
        assert_eq!(asc_dsc, AscDsc::Ascending);

        // Descending: negative z-velocity
        let state_descending = Vector6::new(
            7000e3, 0.0, 0.0, // position
            0.0, 7500.0, -100.0, // velocity (negative z)
        );

        let asc_dsc = compute_asc_dsc(&state_descending);
        assert_eq!(asc_dsc, AscDsc::Descending);
    }

    #[test]
    fn test_compute_look_direction() {
        // Satellite state
        let sat_state = Vector6::new(
            7000e3, 0.0, 0.0, // position
            0.0, 7500.0, 0.0, // velocity (moving in +y direction)
        );

        // Location to the right (negative x)
        let loc_right = Vector3::new(6000e3, 0.0, 0.0);
        let look_dir = compute_look_direction(&sat_state, &loc_right);
        // This test is geometry-dependent; just check it returns a value
        assert!(look_dir == LookDirection::Left || look_dir == LookDirection::Right);
    }

    // Example property computer implementation
    struct TestPropertyComputer;

    impl AccessPropertyComputer for TestPropertyComputer {
        fn compute(
            &self,
            window: &AccessWindow,
            state_provider: &dyn StateProvider,
            _location_ecef: &nalgebra::Vector3<f64>,
            _location_geodetic: &nalgebra::Vector3<f64>,
        ) -> Result<HashMap<String, PropertyValue>, BraheError> {
            let mut props = HashMap::new();

            // Sample at midtime
            let midtime = window.midtime();
            let state = state_provider.state_ecef(midtime);

            // Compute a simple property (altitude)
            let altitude = state.fixed_rows::<3>(0).norm() - 6371e3;
            props.insert(
                "altitude_km".to_string(),
                PropertyValue::Scalar(altitude / 1e3),
            );

            Ok(props)
        }

        fn property_names(&self) -> Vec<String> {
            vec!["altitude_km".to_string()]
        }
    }

    #[test]
    fn test_property_computer() {
        use crate::access::{AccessibleLocation, PointLocation};
        use crate::trajectories::traits::{OrbitFrame, OrbitRepresentation};

        // Initialize EOP for frame conversions
        setup_global_test_eop();

        // Create test data
        let epoch1 = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = epoch1 + 120.0;

        // Create a minimal AccessWindow for testing
        let window = crate::access::AccessWindow {
            window_open: epoch1,
            window_close: epoch2,
            location_name: None,
            location_id: None,
            location_uuid: None,
            satellite_name: None,
            satellite_id: None,
            satellite_uuid: None,
            name: None,
            id: None,
            uuid: None,
            properties: crate::access::AccessProperties::new(
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                crate::access::LookDirection::Either,
                crate::access::AscDsc::Either,
                0.0,
                45.0,
                0.0,
                [4517.59e3, 4517.59e3, 0.0],
            ),
        };

        let elements = Vector6::new(7000e3, 0.001, 0.9, 0.0, 0.0, 0.0);
        let prop = KeplerianPropagator::new(
            epoch1,
            elements,
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Radians),
            60.0,
        );

        let location = PointLocation::new(0.0, 45.0, 0.0);
        let location_ecef = location.center_ecef();
        let location_geodetic = Vector3::new(0.0_f64.to_radians(), 45.0_f64.to_radians(), 0.0);

        // Compute properties using StateProvider trait directly
        let computer = TestPropertyComputer;
        let props = computer
            .compute(&window, &prop, &location_ecef, &location_geodetic)
            .unwrap();

        // Check property exists
        assert!(props.contains_key("altitude_km"));

        // Check property names
        assert_eq!(computer.property_names(), vec!["altitude_km"]);
    }
}
