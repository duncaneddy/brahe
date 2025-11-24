/*!
 * Access property computation types and utilities
 *
 * This module provides types and traits for computing properties of access windows,
 * including geometric properties (azimuth, elevation, look direction) and custom
 * user-defined properties.
 */

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::time::Epoch;
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
// SamplingConfig Enum
// ================================

/// Configuration for sampling satellite states during an access window.
///
/// This enum defines how property computers should sample satellite states
/// within an access window. Different sampling strategies are useful for
/// different types of properties.
///
/// # Examples
/// ```
/// use brahe::access::SamplingConfig;
///
/// // Single point at window midpoint
/// let midpoint = SamplingConfig::Midpoint;
///
/// // Sample at start, middle, and end
/// let relative = SamplingConfig::RelativePoints(vec![0.0, 0.5, 1.0]);
///
/// // Sample every 10 seconds
/// let interval = SamplingConfig::FixedInterval { interval: 10.0, offset: 0.0 };
///
/// // Sample at 10 evenly-spaced points
/// let count = SamplingConfig::FixedCount(10);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SamplingConfig {
    /// Single sample at window midpoint
    Midpoint,

    /// Sample at specific relative times (0.0 = window_open, 1.0 = window_close)
    ///
    /// # Example
    /// ```
    /// use brahe::access::SamplingConfig;
    ///
    /// // Sample at start, quarter, middle, three-quarters, and end
    /// let config = SamplingConfig::RelativePoints(vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    /// ```
    RelativePoints(Vec<f64>),

    /// Sample at fixed time intervals (seconds)
    ///
    /// # Example
    /// ```
    /// use brahe::access::SamplingConfig;
    ///
    /// // Sample every 0.1 seconds starting at window open
    /// let config = SamplingConfig::FixedInterval { interval: 0.1, offset: 0.0 };
    /// ```
    FixedInterval {
        /// Time between samples (seconds)
        interval: f64,
        /// Time offset from window_open (seconds)
        offset: f64,
    },

    /// Sample at N evenly-spaced points (including endpoints)
    ///
    /// # Example
    /// ```
    /// use brahe::access::SamplingConfig;
    ///
    /// // Sample at 10 evenly-spaced points
    /// let config = SamplingConfig::FixedCount(10);
    /// ```
    FixedCount(usize),
}

impl SamplingConfig {
    /// Generate sample epochs based on the sampling configuration.
    ///
    /// # Arguments
    /// * `window_open` - Window start time
    /// * `window_close` - Window end time
    ///
    /// # Returns
    /// Vector of sample epochs
    ///
    /// # Panics
    /// - `FixedInterval`: Panics if interval ≤ 0, offset < 0, or no samples generated within window
    /// - `FixedCount`: Panics if count = 0
    /// - `RelativePoints`: Panics if empty or any value outside [0.0, 1.0]
    ///
    /// # Examples
    /// ```
    /// use brahe::access::SamplingConfig;
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let window_open = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let window_close = window_open + 3600.0; // 1 hour later
    ///
    /// // Midpoint
    /// let config = SamplingConfig::Midpoint;
    /// let epochs = config.generate_sample_epochs(window_open, window_close);
    /// assert_eq!(epochs.len(), 1);
    /// assert_eq!(epochs[0], window_open + 1800.0);
    ///
    /// // Relative points
    /// let config = SamplingConfig::RelativePoints(vec![0.0, 0.5, 1.0]);
    /// let epochs = config.generate_sample_epochs(window_open, window_close);
    /// assert_eq!(epochs.len(), 3);
    /// assert_eq!(epochs[0], window_open);
    /// assert_eq!(epochs[1], window_open + 1800.0);
    /// assert_eq!(epochs[2], window_close);
    /// ```
    pub fn generate_sample_epochs(&self, window_open: Epoch, window_close: Epoch) -> Vec<Epoch> {
        let duration = window_close - window_open; // seconds

        match self {
            SamplingConfig::Midpoint => {
                vec![window_open + duration * 0.5]
            }

            SamplingConfig::RelativePoints(relative_times) => {
                if relative_times.is_empty() {
                    panic!("SamplingConfig::RelativePoints: relative_times cannot be empty");
                }

                // Validate all points are in [0.0, 1.0]
                for &t in relative_times.iter() {
                    if !(0.0..=1.0).contains(&t) {
                        panic!(
                            "SamplingConfig::RelativePoints: all relative times must be in [0.0, 1.0], got {}",
                            t
                        );
                    }
                }

                relative_times
                    .iter()
                    .map(|&t| window_open + duration * t)
                    .collect()
            }

            SamplingConfig::FixedInterval { interval, offset } => {
                if *interval <= 0.0 {
                    panic!(
                        "SamplingConfig::FixedInterval: interval must be positive, got {}",
                        interval
                    );
                }

                if *offset < 0.0 {
                    panic!(
                        "SamplingConfig::FixedInterval: offset must be non-negative, got {}",
                        offset
                    );
                }

                if *offset > duration {
                    panic!(
                        "SamplingConfig::FixedInterval: offset ({}) exceeds window duration ({})",
                        offset, duration
                    );
                }

                let mut epochs = Vec::new();
                let mut t = *offset; // seconds from window_open

                while t <= duration {
                    epochs.push(window_open + t);
                    t += interval;
                }

                if epochs.is_empty() {
                    panic!(
                        "SamplingConfig::FixedInterval: no samples generated within window (interval too large)"
                    );
                }

                epochs
            }

            SamplingConfig::FixedCount(count) => {
                if *count == 0 {
                    panic!("SamplingConfig::FixedCount: count must be positive, got 0");
                }

                if *count == 1 {
                    return vec![window_open + duration * 0.5];
                }

                // Generate N evenly-spaced points including endpoints
                (0..*count)
                    .map(|i| {
                        let fraction = i as f64 / (*count as f64 - 1.0);
                        window_open + duration * fraction
                    })
                    .collect()
            }
        }
    }
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
/// are calculated. The sampling configuration determines how many satellite
/// states are provided to the compute method.
///
/// # Examples
/// ```no_run
/// use brahe::access::{AccessPropertyComputer, AccessWindow, PropertyValue, SamplingConfig};
/// use brahe::utils::BraheError;
/// use std::collections::HashMap;
/// use nalgebra::Vector3;
///
/// struct DopplerComputer;
///
/// impl AccessPropertyComputer for DopplerComputer {
///     fn sampling_config(&self) -> SamplingConfig {
///         // Sample every 0.1 seconds
///         SamplingConfig::FixedInterval { interval: 0.1 / 86400.0, offset: 0.0 }
///     }
///
///     fn compute(
///         &self,
///         window: &AccessWindow,
///         sample_epochs: &[f64],
///         sample_states_ecef: &[nalgebra::SVector<f64, 6>],
///         location_ecef: &Vector3<f64>,
///         location_geodetic: &Vector3<f64>,
///     ) -> Result<HashMap<String, PropertyValue>, BraheError> {
///         let mut props = HashMap::new();
///
///         // Compute Doppler shift at each sample point
///         let doppler_values: Vec<f64> = sample_states_ecef.iter().map(|state| {
///             // ... compute Doppler shift from state and location ...
///             2500.0  // Example value
///         }).collect();
///
///         // Convert to relative times (seconds from window open)
///         let relative_times: Vec<f64> = sample_epochs.iter()
///             .map(|&epoch| (epoch - window.window_open.mjd()) * 86400.0)
///             .collect();
///
///         props.insert("doppler_shift".to_string(), PropertyValue::TimeSeries {
///             times: relative_times,
///             values: doppler_values,
///         });
///         Ok(props)
///     }
///
///     fn property_names(&self) -> Vec<String> {
///         vec!["doppler_shift".to_string()]
///     }
/// }
/// ```
pub trait AccessPropertyComputer: Send + Sync {
    /// Return the sampling configuration for this property computer.
    ///
    /// The sampling configuration determines how satellite states are sampled
    /// during the access window. The sampled states are then provided to the
    /// compute method.
    ///
    /// # Returns
    /// The sampling configuration to use for this property computer
    fn sampling_config(&self) -> SamplingConfig;

    /// Compute additional properties for an access window.
    ///
    /// This method receives pre-sampled satellite states based on the sampling
    /// configuration returned by `sampling_config()`. The number of samples
    /// corresponds to the sampling strategy used.
    ///
    /// # Arguments
    /// * `window` - The access window (contains times and core properties)
    /// * `sample_epochs` - Sample epochs in MJD (Modified Julian Date)
    /// * `sample_states_ecef` - ECEF states [x,y,z,vx,vy,vz] at each sample epoch (m, m/s)
    /// * `location_ecef` - Location ECEF position [x, y, z] (meters)
    /// * `location_geodetic` - Location geodetic coordinates [lon, lat, alt] (radians, meters)
    ///
    /// # Returns
    /// HashMap of property name -> PropertyValue
    ///
    /// # Notes
    /// - For single-point sampling (e.g., Midpoint), arrays will have length 1
    /// - For time-series sampling, use PropertyValue::TimeSeries with relative times
    /// - Relative times should be in seconds from window_open: (epoch - window_open.mjd()) * 86400.0
    /// - The return type is automatically detected: single sample -> Scalar/Vector, multiple -> TimeSeries
    fn compute(
        &self,
        window: &AccessWindow,
        sample_epochs: &[f64],
        sample_states_ecef: &[nalgebra::SVector<f64, 6>],
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
// Built-in Property Computers
// ================================

/// Computes Doppler shift for uplink and/or downlink communications.
///
/// This property computer calculates Doppler frequency shifts caused by relative
/// motion between the satellite and ground station. It supports separate uplink
/// and downlink frequency configurations.
///
/// # Physics
///
/// - **Uplink Doppler**: Δf = f₀ × v_los / (c - v_los)
///   - Ground station pre-compensates transmit frequency so satellite receives design frequency
/// - **Downlink Doppler**: Δf = -f₀ × v_los / c
///   - Ground station adjusts receive frequency to match Doppler-shifted spacecraft transmission
///
/// where v_los is the line-of-sight velocity (positive = receding, negative = approaching)
///
/// # Examples
///
/// ```
/// use brahe::access::{DopplerComputer, SamplingConfig};
///
/// // S-band uplink (2.2 GHz) and X-band downlink (8.4 GHz)
/// let computer = DopplerComputer::new(
///     Some(2.2e9),  // uplink frequency (Hz)
///     Some(8.4e9),  // downlink frequency (Hz)
///     SamplingConfig::FixedInterval { interval: 0.1 / 86400.0, offset: 0.0 }
/// );
///
/// // Downlink only
/// let downlink_only = DopplerComputer::new(
///     None,
///     Some(8.4e9),
///     SamplingConfig::Midpoint
/// );
/// ```
#[derive(Clone)]
pub struct DopplerComputer {
    /// Uplink frequency in Hz (optional)
    pub uplink_frequency: Option<f64>,
    /// Downlink frequency in Hz (optional)
    pub downlink_frequency: Option<f64>,
    /// Sampling configuration for time-series computation
    pub sampling_config: SamplingConfig,
}

impl DopplerComputer {
    /// Create a new Doppler computer.
    ///
    /// # Arguments
    ///
    /// * `uplink_frequency` - Optional uplink frequency in Hz
    /// * `downlink_frequency` - Optional downlink frequency in Hz
    /// * `sampling_config` - Sampling configuration for the access window
    ///
    /// # Returns
    ///
    /// New DopplerComputer instance
    ///
    /// # Notes
    ///
    /// At least one frequency (uplink or downlink) must be specified.
    pub fn new(
        uplink_frequency: Option<f64>,
        downlink_frequency: Option<f64>,
        sampling_config: SamplingConfig,
    ) -> Self {
        Self {
            uplink_frequency,
            downlink_frequency,
            sampling_config,
        }
    }
}

impl AccessPropertyComputer for DopplerComputer {
    fn sampling_config(&self) -> SamplingConfig {
        self.sampling_config.clone()
    }

    fn compute(
        &self,
        window: &AccessWindow,
        sample_epochs: &[f64],
        sample_states_ecef: &[nalgebra::SVector<f64, 6>],
        location_ecef: &nalgebra::Vector3<f64>,
        _location_geodetic: &nalgebra::Vector3<f64>,
    ) -> Result<HashMap<String, PropertyValue>, BraheError> {
        let mut props = HashMap::new();

        // Compute line-of-sight velocities at each sample
        let v_los_values: Vec<f64> = sample_states_ecef
            .iter()
            .map(|state| {
                let sat_pos = state.fixed_rows::<3>(0);
                let sat_vel = state.fixed_rows::<3>(3);

                // Line-of-sight vector (from ground station to satellite)
                let los_vec = sat_pos - location_ecef;
                let los_unit = los_vec.normalize();

                // Line-of-sight velocity (positive = receding, negative = approaching)
                sat_vel.dot(&los_unit)
            })
            .collect();

        // Convert to relative times (seconds from window open)
        let relative_times: Vec<f64> = sample_epochs
            .iter()
            .map(|&epoch| (epoch - window.window_open.mjd()) * 86400.0)
            .collect();

        // Compute uplink Doppler if frequency specified
        if let Some(f_uplink) = self.uplink_frequency {
            let doppler_uplink: Vec<f64> = v_los_values
                .iter()
                .map(|&v_los| f_uplink * v_los / (crate::constants::C_LIGHT - v_los))
                .collect();

            let value = if doppler_uplink.len() == 1 {
                PropertyValue::Scalar(doppler_uplink[0])
            } else {
                PropertyValue::TimeSeries {
                    times: relative_times.clone(),
                    values: doppler_uplink,
                }
            };

            props.insert("doppler_uplink".to_string(), value);
        }

        // Compute downlink Doppler if frequency specified
        if let Some(f_downlink) = self.downlink_frequency {
            let doppler_downlink: Vec<f64> = v_los_values
                .iter()
                .map(|&v_los| -f_downlink * v_los / crate::constants::C_LIGHT)
                .collect();

            let value = if doppler_downlink.len() == 1 {
                PropertyValue::Scalar(doppler_downlink[0])
            } else {
                PropertyValue::TimeSeries {
                    times: relative_times,
                    values: doppler_downlink,
                }
            };

            props.insert("doppler_downlink".to_string(), value);
        }

        Ok(props)
    }

    fn property_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        if self.uplink_frequency.is_some() {
            names.push("doppler_uplink".to_string());
        }
        if self.downlink_frequency.is_some() {
            names.push("doppler_downlink".to_string());
        }
        names
    }
}

/// Computes range (distance) between satellite and ground station.
///
/// This property computer calculates the instantaneous slant range from the
/// ground station to the satellite at each sample point.
///
/// # Examples
///
/// ```
/// use brahe::access::{RangeComputer, SamplingConfig};
///
/// // Compute range every second
/// let computer = RangeComputer::new(
///     SamplingConfig::FixedInterval { interval: 1.0 / 86400.0, offset: 0.0 }
/// );
/// ```
#[derive(Clone)]
pub struct RangeComputer {
    /// Sampling configuration for time-series computation
    pub sampling_config: SamplingConfig,
}

impl RangeComputer {
    /// Create a new range computer.
    ///
    /// # Arguments
    ///
    /// * `sampling_config` - Sampling configuration for the access window
    ///
    /// # Returns
    ///
    /// New RangeComputer instance
    pub fn new(sampling_config: SamplingConfig) -> Self {
        Self { sampling_config }
    }
}

impl AccessPropertyComputer for RangeComputer {
    fn sampling_config(&self) -> SamplingConfig {
        self.sampling_config.clone()
    }

    fn compute(
        &self,
        window: &AccessWindow,
        sample_epochs: &[f64],
        sample_states_ecef: &[nalgebra::SVector<f64, 6>],
        location_ecef: &nalgebra::Vector3<f64>,
        _location_geodetic: &nalgebra::Vector3<f64>,
    ) -> Result<HashMap<String, PropertyValue>, BraheError> {
        let mut props = HashMap::new();

        // Compute range at each sample
        let range_values: Vec<f64> = sample_states_ecef
            .iter()
            .map(|state| {
                let sat_pos = state.fixed_rows::<3>(0);
                (sat_pos - location_ecef).norm()
            })
            .collect();

        // Convert to relative times (seconds from window open)
        let relative_times: Vec<f64> = sample_epochs
            .iter()
            .map(|&epoch| (epoch - window.window_open.mjd()) * 86400.0)
            .collect();

        let value = if range_values.len() == 1 {
            PropertyValue::Scalar(range_values[0])
        } else {
            PropertyValue::TimeSeries {
                times: relative_times,
                values: range_values,
            }
        };

        props.insert("range".to_string(), value);

        Ok(props)
    }

    fn property_names(&self) -> Vec<String> {
        vec!["range".to_string()]
    }
}

/// Computes range rate (line-of-sight velocity) between satellite and ground station.
///
/// This property computer calculates the instantaneous rate of change of range
/// (also known as radial velocity or line-of-sight velocity) at each sample point.
///
/// # Sign Convention
///
/// - Positive range rate: satellite moving away from ground station (receding)
/// - Negative range rate: satellite moving toward ground station (approaching)
///
/// # Examples
///
/// ```
/// use brahe::access::{RangeRateComputer, SamplingConfig};
///
/// // Compute range rate at midpoint
/// let computer = RangeRateComputer::new(SamplingConfig::Midpoint);
///
/// // Compute range rate time series
/// let computer = RangeRateComputer::new(
///     SamplingConfig::FixedCount(50)  // 50 evenly-spaced points
/// );
/// ```
#[derive(Clone)]
pub struct RangeRateComputer {
    /// Sampling configuration for time-series computation
    pub sampling_config: SamplingConfig,
}

impl RangeRateComputer {
    /// Create a new range rate computer.
    ///
    /// # Arguments
    ///
    /// * `sampling_config` - Sampling configuration for the access window
    ///
    /// # Returns
    ///
    /// New RangeRateComputer instance
    pub fn new(sampling_config: SamplingConfig) -> Self {
        Self { sampling_config }
    }
}

impl AccessPropertyComputer for RangeRateComputer {
    fn sampling_config(&self) -> SamplingConfig {
        self.sampling_config.clone()
    }

    fn compute(
        &self,
        window: &AccessWindow,
        sample_epochs: &[f64],
        sample_states_ecef: &[nalgebra::SVector<f64, 6>],
        location_ecef: &nalgebra::Vector3<f64>,
        _location_geodetic: &nalgebra::Vector3<f64>,
    ) -> Result<HashMap<String, PropertyValue>, BraheError> {
        let mut props = HashMap::new();

        // Compute range rate at each sample
        let range_rate_values: Vec<f64> = sample_states_ecef
            .iter()
            .map(|state| {
                let sat_pos = state.fixed_rows::<3>(0);
                let sat_vel = state.fixed_rows::<3>(3);

                // Line-of-sight vector (from ground station to satellite)
                let los_vec = sat_pos - location_ecef;
                let los_unit = los_vec.normalize();

                // Range rate (line-of-sight velocity)
                sat_vel.dot(&los_unit)
            })
            .collect();

        // Convert to relative times (seconds from window open)
        let relative_times: Vec<f64> = sample_epochs
            .iter()
            .map(|&epoch| (epoch - window.window_open.mjd()) * 86400.0)
            .collect();

        let value = if range_rate_values.len() == 1 {
            PropertyValue::Scalar(range_rate_values[0])
        } else {
            PropertyValue::TimeSeries {
                times: relative_times,
                values: range_rate_values,
            }
        };

        props.insert("range_rate".to_string(), value);

        Ok(props)
    }

    fn property_names(&self) -> Vec<String> {
        vec!["range_rate".to_string()]
    }
}

// ================================
// Tests
// ================================

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use nalgebra::{Vector3, Vector6};

    use crate::constants::AngleFormat;
    use crate::coordinates::position_geodetic_to_ecef;
    use crate::propagators::KeplerianPropagator;
    use crate::propagators::traits::SOrbitStateProvider;
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
        fn sampling_config(&self) -> SamplingConfig {
            SamplingConfig::Midpoint
        }

        fn compute(
            &self,
            _window: &AccessWindow,
            _sample_epochs: &[f64],
            sample_states_ecef: &[nalgebra::SVector<f64, 6>],
            _location_ecef: &nalgebra::Vector3<f64>,
            _location_geodetic: &nalgebra::Vector3<f64>,
        ) -> Result<HashMap<String, PropertyValue>, BraheError> {
            let mut props = HashMap::new();

            // Use first (and only) sample for midpoint configuration
            let state = &sample_states_ecef[0];

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

        // Get sampling configuration and generate sample epochs
        let computer = TestPropertyComputer;
        let sampling_config = computer.sampling_config();
        let sample_epochs =
            sampling_config.generate_sample_epochs(window.window_open, window.window_close);

        // Get states at sample epochs using StateProvider trait
        let sample_states: Vec<nalgebra::SVector<f64, 6>> = sample_epochs
            .iter()
            .map(|&epoch| prop.state_ecef(epoch))
            .collect();

        // Convert epochs to MJD for property computer interface
        let sample_epochs_mjd: Vec<f64> = sample_epochs.iter().map(|e| e.mjd()).collect();

        // Compute properties with sampled states
        let props = computer
            .compute(
                &window,
                &sample_epochs_mjd,
                &sample_states,
                &location_ecef,
                &location_geodetic,
            )
            .unwrap();

        // Check property exists
        assert!(props.contains_key("altitude_km"));

        // Check property names
        assert_eq!(computer.property_names(), vec!["altitude_km"]);
    }

    // ================================
    // SamplingConfig Tests
    // ================================

    #[test]
    fn test_sampling_config_midpoint() {
        let config = SamplingConfig::Midpoint;
        let window_open =
            crate::time::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let window_close = window_open + 3600.0; // 1 hour later

        let epochs = config.generate_sample_epochs(window_open, window_close);

        assert_eq!(epochs.len(), 1);
        assert_eq!(epochs[0], window_open + 1800.0); // Midpoint
    }

    #[test]
    fn test_sampling_config_relative_points() {
        let config = SamplingConfig::RelativePoints(vec![0.0, 0.5, 1.0]);
        let window_open =
            crate::time::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let window_close = window_open + 3600.0; // 1 hour later

        let epochs = config.generate_sample_epochs(window_open, window_close);

        assert_eq!(epochs.len(), 3);
        assert_eq!(epochs[0], window_open); // Start
        assert_eq!(epochs[1], window_open + 1800.0); // Middle
        assert_eq!(epochs[2], window_close); // End
    }

    #[test]
    #[should_panic(expected = "all relative times must be in [0.0, 1.0]")]
    fn test_sampling_config_relative_points_out_of_bounds_negative() {
        // Values outside [0, 1] should panic
        let config = SamplingConfig::RelativePoints(vec![-0.5, 0.0, 0.5]);
        let window_open =
            crate::time::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let window_close = window_open + 3600.0;

        config.generate_sample_epochs(window_open, window_close);
    }

    #[test]
    #[should_panic(expected = "all relative times must be in [0.0, 1.0]")]
    fn test_sampling_config_relative_points_out_of_bounds_positive() {
        // Values outside [0, 1] should panic
        let config = SamplingConfig::RelativePoints(vec![0.0, 0.5, 1.5]);
        let window_open =
            crate::time::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let window_close = window_open + 3600.0;

        config.generate_sample_epochs(window_open, window_close);
    }

    #[test]
    #[should_panic(expected = "relative_times cannot be empty")]
    fn test_sampling_config_relative_points_empty() {
        // Empty vector should panic
        let config = SamplingConfig::RelativePoints(vec![]);
        let window_open =
            crate::time::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let window_close = window_open + 3600.0;

        config.generate_sample_epochs(window_open, window_close);
    }

    #[test]
    fn test_sampling_config_fixed_interval() {
        let config = SamplingConfig::FixedInterval {
            interval: 600.0, // 10 minutes in seconds
            offset: 0.0,
        };
        let window_open =
            crate::time::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let window_close = window_open + 3000.0; // 50 minutes later

        let epochs = config.generate_sample_epochs(window_open, window_close);

        // Should have samples at 0, 600, 1200, 1800, 2400, 3000 seconds from open
        assert_eq!(epochs.len(), 6);
        assert_eq!(epochs[0], window_open);
        assert_eq!(epochs[1], window_open + 600.0);
        assert_eq!(epochs[2], window_open + 1200.0);
        assert_eq!(epochs[3], window_open + 1800.0);
        assert_eq!(epochs[4], window_open + 2400.0);
        assert_eq!(epochs[5], window_open + 3000.0);
    }

    #[test]
    fn test_sampling_config_fixed_interval_with_offset() {
        let config = SamplingConfig::FixedInterval {
            interval: 1200.0, // 20 minutes in seconds
            offset: 600.0,    // Start at 10 minutes
        };
        let window_open =
            crate::time::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let window_close = window_open + 3000.0; // 50 minutes later

        let epochs = config.generate_sample_epochs(window_open, window_close);

        // Should have samples at 600, 1800, 3000 seconds from open
        assert_eq!(epochs.len(), 3);
        assert_eq!(epochs[0], window_open + 600.0);
        assert_eq!(epochs[1], window_open + 1800.0);
        assert_eq!(epochs[2], window_open + 3000.0);
    }

    #[test]
    #[should_panic(expected = "interval must be positive")]
    fn test_sampling_config_fixed_interval_zero() {
        // Zero interval should panic
        let config = SamplingConfig::FixedInterval {
            interval: 0.0,
            offset: 0.0,
        };
        let window_open =
            crate::time::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let window_close = window_open + 3600.0;

        config.generate_sample_epochs(window_open, window_close);
    }

    #[test]
    #[should_panic(expected = "interval must be positive")]
    fn test_sampling_config_fixed_interval_negative() {
        // Negative interval should panic
        let config = SamplingConfig::FixedInterval {
            interval: -0.1,
            offset: 0.0,
        };
        let window_open =
            crate::time::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let window_close = window_open + 3600.0;

        config.generate_sample_epochs(window_open, window_close);
    }

    #[test]
    #[should_panic(expected = "offset must be non-negative")]
    fn test_sampling_config_fixed_interval_negative_offset() {
        // Negative offset should panic
        let config = SamplingConfig::FixedInterval {
            interval: 600.0,
            offset: -100.0,
        };
        let window_open =
            crate::time::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let window_close = window_open + 3600.0;

        config.generate_sample_epochs(window_open, window_close);
    }

    #[test]
    #[should_panic(expected = "offset")]
    fn test_sampling_config_fixed_interval_offset_beyond_window() {
        // If offset is beyond window duration, should panic
        let config = SamplingConfig::FixedInterval {
            interval: 600.0,
            offset: 4000.0, // Offset beyond window duration
        };
        let window_open =
            crate::time::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let window_close = window_open + 3600.0; // 1 hour = 3600 seconds

        config.generate_sample_epochs(window_open, window_close);
    }

    #[test]
    fn test_sampling_config_fixed_count() {
        let config = SamplingConfig::FixedCount(5);
        let window_open =
            crate::time::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let window_close = window_open + 2400.0; // 40 minutes

        let epochs = config.generate_sample_epochs(window_open, window_close);

        // 5 evenly-spaced points: 0, 600, 1200, 1800, 2400 seconds
        assert_eq!(epochs.len(), 5);
        assert_eq!(epochs[0], window_open);
        assert_eq!(epochs[1], window_open + 600.0);
        assert_eq!(epochs[2], window_open + 1200.0);
        assert_eq!(epochs[3], window_open + 1800.0);
        assert_eq!(epochs[4], window_open + 2400.0);
    }

    #[test]
    fn test_sampling_config_fixed_count_single() {
        let config = SamplingConfig::FixedCount(1);
        let window_open =
            crate::time::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let window_close = window_open + 3600.0;

        let epochs = config.generate_sample_epochs(window_open, window_close);

        // Single point should be at midpoint
        assert_eq!(epochs.len(), 1);
        assert_eq!(epochs[0], window_open + 1800.0);
    }

    #[test]
    fn test_sampling_config_fixed_count_two() {
        let config = SamplingConfig::FixedCount(2);
        let window_open =
            crate::time::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let window_close = window_open + 3600.0;

        let epochs = config.generate_sample_epochs(window_open, window_close);

        // Two points: start and end
        assert_eq!(epochs.len(), 2);
        assert_eq!(epochs[0], window_open);
        assert_eq!(epochs[1], window_close);
    }

    #[test]
    #[should_panic(expected = "count must be positive")]
    fn test_sampling_config_fixed_count_zero() {
        // Zero count should panic
        let config = SamplingConfig::FixedCount(0);
        let window_open =
            crate::time::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let window_close = window_open + 3600.0;

        config.generate_sample_epochs(window_open, window_close);
    }

    #[test]
    fn test_sampling_config_serialization() {
        // Midpoint
        let config = SamplingConfig::Midpoint;
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: SamplingConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, deserialized);

        // RelativePoints
        let config = SamplingConfig::RelativePoints(vec![0.0, 0.5, 1.0]);
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: SamplingConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, deserialized);

        // FixedInterval
        let config = SamplingConfig::FixedInterval {
            interval: 0.1,
            offset: 0.05,
        };
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: SamplingConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, deserialized);

        // FixedCount
        let config = SamplingConfig::FixedCount(10);
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: SamplingConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, deserialized);
    }

    #[test]
    fn test_doppler_computer_downlink() {
        setup_global_test_eop();

        let computer = DopplerComputer::new(
            None,
            Some(2.2e9), // S-band downlink
            SamplingConfig::Midpoint,
        );

        // Create a simple access window
        let window_open = crate::time::Epoch::from_datetime(
            2024,
            1,
            1,
            0,
            0,
            0.0,
            0.0,
            crate::time::TimeSystem::UTC,
        );
        let window_close = window_open + 60.0; // 1 minute window

        // Satellite approaching station (negative range rate = approaching)
        let sample_epochs = vec![window_open.mjd() + 30.0 / 86400.0]; // midpoint

        let location_ecef = nalgebra::Vector3::new(4000000.0, 1000000.0, 4500000.0);
        let sat_pos = nalgebra::Vector3::new(1000000.0, 2000000.0, 3000000.0);

        // Velocity toward station (approaching)
        let to_station = (location_ecef - sat_pos).normalize();
        let approaching_velocity = to_station * 1000.0; // 1000 m/s toward station

        let sample_states = vec![nalgebra::SVector::<f64, 6>::new(
            sat_pos[0],
            sat_pos[1],
            sat_pos[2],
            approaching_velocity[0],
            approaching_velocity[1],
            approaching_velocity[2],
        )];
        let location_geodetic = nalgebra::Vector3::new(0.0, 0.0, 0.0); // not used

        let temp_window = crate::access::AccessWindow {
            window_open,
            window_close,
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
                0.0,
                0.0,
                [0.0, 0.0, 0.0],
            ),
        };

        let result = computer
            .compute(
                &temp_window,
                &sample_epochs,
                &sample_states,
                &location_ecef,
                &location_geodetic,
            )
            .unwrap();

        // Check that doppler_downlink property exists
        assert!(result.contains_key("doppler_downlink"));

        // Verify it's a scalar value (single sample)
        if let PropertyValue::Scalar(doppler) = result.get("doppler_downlink").unwrap() {
            // For approaching satellite, doppler should be positive (frequency increase)
            // Basic sanity check: doppler should be reasonable for typical satellite velocities
            assert!(
                *doppler > 0.0,
                "Doppler should be positive for approaching satellite"
            );
            assert!(
                *doppler < 100000.0,
                "Doppler should be reasonable (<100 kHz)"
            );
        } else {
            panic!("Expected scalar value");
        }
    }

    #[test]
    fn test_doppler_computer_uplink() {
        setup_global_test_eop();

        let computer = DopplerComputer::new(
            Some(2.0e9), // S-band uplink
            None,
            SamplingConfig::Midpoint,
        );

        let window_open = crate::time::Epoch::from_datetime(
            2024,
            1,
            1,
            0,
            0,
            0.0,
            0.0,
            crate::time::TimeSystem::UTC,
        );
        let window_close = window_open + 60.0;

        // Satellite receding from station (positive range rate = receding)
        let sample_epochs = vec![window_open.mjd() + 30.0 / 86400.0];

        let location_ecef = nalgebra::Vector3::new(4000000.0, 1000000.0, 4500000.0);
        let sat_pos = nalgebra::Vector3::new(1000000.0, 2000000.0, 3000000.0);

        // Velocity away from station (receding)
        let from_station = (sat_pos - location_ecef).normalize();
        let receding_velocity = from_station * 1000.0; // 1000 m/s away from station

        let sample_states = vec![nalgebra::SVector::<f64, 6>::new(
            sat_pos[0],
            sat_pos[1],
            sat_pos[2],
            receding_velocity[0],
            receding_velocity[1],
            receding_velocity[2],
        )];
        let location_geodetic = nalgebra::Vector3::new(0.0, 0.0, 0.0);

        let temp_window = crate::access::AccessWindow {
            window_open,
            window_close,
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
                0.0,
                0.0,
                [0.0, 0.0, 0.0],
            ),
        };

        let result = computer
            .compute(
                &temp_window,
                &sample_epochs,
                &sample_states,
                &location_ecef,
                &location_geodetic,
            )
            .unwrap();

        assert!(result.contains_key("doppler_uplink"));

        if let PropertyValue::Scalar(doppler) = result.get("doppler_uplink").unwrap() {
            // For receding satellite, uplink pre-compensation should be positive
            assert!(
                *doppler > 0.0,
                "Uplink doppler should be positive for receding satellite"
            );
            assert!(*doppler < 100000.0, "Doppler should be reasonable");
        } else {
            panic!("Expected scalar value");
        }
    }

    #[test]
    fn test_doppler_computer_both_frequencies() {
        setup_global_test_eop();

        let computer = DopplerComputer::new(
            Some(2.0e9), // uplink
            Some(2.2e9), // downlink
            SamplingConfig::FixedCount(3),
        );

        let window_open = crate::time::Epoch::from_datetime(
            2024,
            1,
            1,
            0,
            0,
            0.0,
            0.0,
            crate::time::TimeSystem::UTC,
        );
        let window_close: crate::time::Epoch = window_open + 120.0; // 2 minutes

        let config = SamplingConfig::FixedCount(3);
        let sample_epochs = config.generate_sample_epochs(window_open, window_close);

        let sample_states = vec![
            nalgebra::SVector::<f64, 6>::new(
                1000000.0, 2000000.0, 3000000.0, -1000.0, -500.0, -200.0,
            ),
            nalgebra::SVector::<f64, 6>::new(
                1010000.0, 2005000.0, 3002000.0, -800.0, -400.0, -150.0,
            ),
            nalgebra::SVector::<f64, 6>::new(
                1020000.0, 2010000.0, 3004000.0, -600.0, -300.0, -100.0,
            ),
        ];

        let location_ecef = nalgebra::Vector3::new(4000000.0, 1000000.0, 4500000.0);
        let location_geodetic = nalgebra::Vector3::new(0.0, 0.0, 0.0);

        // Convert epochs to MJD for property computer interface
        let sample_epochs_mjd: Vec<f64> = sample_epochs.iter().map(|e| e.mjd()).collect();

        let temp_window = crate::access::AccessWindow {
            window_open,
            window_close,
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
                0.0,
                0.0,
                [0.0, 0.0, 0.0],
            ),
        };

        let result = computer
            .compute(
                &temp_window,
                &sample_epochs_mjd,
                &sample_states,
                &location_ecef,
                &location_geodetic,
            )
            .unwrap();

        // Should have both uplink and downlink
        assert!(result.contains_key("doppler_uplink"));
        assert!(result.contains_key("doppler_downlink"));

        // Both should be time series (3 samples)
        match result.get("doppler_uplink").unwrap() {
            PropertyValue::TimeSeries { times, values } => {
                assert_eq!(times.len(), 3);
                assert_eq!(values.len(), 3);
            }
            _ => panic!("Expected time series value"),
        }

        match result.get("doppler_downlink").unwrap() {
            PropertyValue::TimeSeries { times, values } => {
                assert_eq!(times.len(), 3);
                assert_eq!(values.len(), 3);
            }
            _ => panic!("Expected time series value"),
        }
    }

    #[test]
    fn test_range_computer() {
        setup_global_test_eop();

        let computer = RangeComputer::new(SamplingConfig::FixedCount(2));

        let window_open = crate::time::Epoch::from_datetime(
            2024,
            1,
            1,
            0,
            0,
            0.0,
            0.0,
            crate::time::TimeSystem::UTC,
        );
        let window_close: crate::time::Epoch = window_open + 60.0;

        let config = SamplingConfig::FixedCount(2);
        let sample_epochs = config.generate_sample_epochs(window_open, window_close);

        let location_ecef = nalgebra::Vector3::new(4000000.0, 1000000.0, 4500000.0);

        // Two satellite positions at different distances
        let sample_states = vec![
            nalgebra::SVector::<f64, 6>::new(
                location_ecef[0] + 1000000.0,
                location_ecef[1],
                location_ecef[2],
                0.0,
                0.0,
                0.0,
            ),
            nalgebra::SVector::<f64, 6>::new(
                location_ecef[0] + 2000000.0,
                location_ecef[1],
                location_ecef[2],
                0.0,
                0.0,
                0.0,
            ),
        ];

        let location_geodetic = nalgebra::Vector3::new(0.0, 0.0, 0.0);

        // Convert epochs to MJD for property computer interface
        let sample_epochs_mjd: Vec<f64> = sample_epochs.iter().map(|e| e.mjd()).collect();

        let temp_window = crate::access::AccessWindow {
            window_open,
            window_close,
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
                0.0,
                0.0,
                [0.0, 0.0, 0.0],
            ),
        };

        let result = computer
            .compute(
                &temp_window,
                &sample_epochs_mjd,
                &sample_states,
                &location_ecef,
                &location_geodetic,
            )
            .unwrap();

        assert!(result.contains_key("range"));

        match result.get("range").unwrap() {
            PropertyValue::TimeSeries { times, values } => {
                assert_eq!(times.len(), 2);
                assert_eq!(values.len(), 2);

                // First range should be ~1000 km
                assert!(
                    (values[0] - 1000000.0).abs() < 1.0,
                    "First range should be ~1000 km"
                );

                // Second range should be ~2000 km
                assert!(
                    (values[1] - 2000000.0).abs() < 1.0,
                    "Second range should be ~2000 km"
                );
            }
            _ => panic!("Expected time series value"),
        }
    }

    #[test]
    fn test_range_rate_computer() {
        setup_global_test_eop();

        let computer = RangeRateComputer::new(SamplingConfig::Midpoint);

        let window_open = crate::time::Epoch::from_datetime(
            2024,
            1,
            1,
            0,
            0,
            0.0,
            0.0,
            crate::time::TimeSystem::UTC,
        );
        let window_close = window_open + 60.0;

        let sample_epochs = vec![window_open.mjd() + 30.0 / 86400.0];

        let location_ecef = nalgebra::Vector3::new(4000000.0, 1000000.0, 4500000.0);

        // Satellite moving directly away from station
        let sat_to_station =
            location_ecef - nalgebra::Vector3::new(1000000.0, 2000000.0, 3000000.0);
        let los_direction = sat_to_station.normalize();

        // Velocity of 1000 m/s in line-of-sight direction (receding)
        let velocity = -los_direction * 1000.0;

        let sample_states = vec![nalgebra::SVector::<f64, 6>::new(
            1000000.0,
            2000000.0,
            3000000.0,
            velocity[0],
            velocity[1],
            velocity[2],
        )];

        let location_geodetic = nalgebra::Vector3::new(0.0, 0.0, 0.0);

        let temp_window = crate::access::AccessWindow {
            window_open,
            window_close,
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
                0.0,
                0.0,
                [0.0, 0.0, 0.0],
            ),
        };

        let result = computer
            .compute(
                &temp_window,
                &sample_epochs,
                &sample_states,
                &location_ecef,
                &location_geodetic,
            )
            .unwrap();

        assert!(result.contains_key("range_rate"));

        if let PropertyValue::Scalar(range_rate) = result.get("range_rate").unwrap() {
            // Should be ~1000 m/s (receding is positive)
            assert!(
                (*range_rate - 1000.0).abs() < 1.0,
                "Range rate should be ~1000 m/s, got {}",
                range_rate
            );
        } else {
            panic!("Expected scalar value");
        }
    }
}
