/*!
 * Access constraints for satellite visibility and access computation
 *
 * This module provides a flexible constraint system for defining when and how
 * satellites can access ground locations. Constraints can be:
 * - Built-in (elevation, off-nadir, local time, look direction, orbit type)
 * - Composed (AND/OR/NOT logic)
 * - Custom (user-defined in Rust or Python)
 */

use crate::coordinates::position_ecef_to_geodetic;
use crate::time::{Epoch, TimeSystem};
use crate::utils::BraheError;
use nalgebra::{Vector3, Vector6};

use super::geometry::{
    compute_asc_dsc, compute_azimuth, compute_elevation, compute_look_direction, compute_off_nadir,
};

/// Core trait for defining access constraints
///
/// Constraints evaluate whether a satellite state satisfies access criteria
/// at a given epoch. All built-in constraints implement this trait.
pub trait AccessConstraint: Send + Sync + std::any::Any {
    /// Evaluate whether constraints are satisfied
    ///
    /// # Arguments
    /// - `epoch`: Time of evaluation
    /// - `sat_state_ecef`: Satellite state in ECEF frame [x, y, z, vx, vy, vz] (meters, m/s)
    /// - `location_ecef`: Ground location in ECEF frame [x, y, z] (meters)
    ///
    /// # Returns
    /// `true` if constraints are satisfied, `false` otherwise
    fn evaluate(
        &self,
        epoch: &Epoch,
        sat_state_ecef: &Vector6<f64>,
        location_ecef: &Vector3<f64>,
    ) -> bool;

    /// Human-readable name for this constraint
    fn name(&self) -> &str;

    /// Format as string (for Display support)
    fn format_string(&self) -> String {
        self.name().to_string()
    }
}

/// Trait for computing custom access constraints
///
/// Implement this trait to create user-defined constraints in Rust or Python.
/// The evaluate method is called at each time step during access computation
/// to determine if the constraint is satisfied.
///
/// This trait is similar to `AccessConstraint` but allows for more complex
/// constraint logic that may need to maintain internal state or compute
/// values that aren't part of the standard geometric constraints.
///
/// # Examples
/// ```no_run
/// use brahe::access::AccessConstraintComputer;
/// use brahe::time::Epoch;
/// use nalgebra::{Vector3, Vector6};
///
/// struct CustomConstraint;
///
/// impl AccessConstraintComputer for CustomConstraint {
///     fn evaluate(
///         &self,
///         epoch: &Epoch,
///         sat_state_ecef: &Vector6<f64>,
///         location_ecef: &Vector3<f64>,
///     ) -> bool {
///         // Custom constraint logic here
///         // For example, check if satellite is in northern hemisphere
///         sat_state_ecef[2] >= 0.0
///     }
///
///     fn name(&self) -> &str {
///         "CustomConstraint"
///     }
/// }
/// ```
pub trait AccessConstraintComputer: Send + Sync {
    /// Evaluate whether the custom constraint is satisfied
    ///
    /// # Arguments
    /// - `epoch`: Time of evaluation
    /// - `sat_state_ecef`: Satellite state in ECEF frame [x, y, z, vx, vy, vz] (meters, m/s)
    /// - `location_ecef`: Ground location in ECEF frame [x, y, z] (meters)
    ///
    /// # Returns
    /// `true` if constraint is satisfied, `false` otherwise
    fn evaluate(
        &self,
        epoch: &Epoch,
        sat_state_ecef: &Vector6<f64>,
        location_ecef: &Vector3<f64>,
    ) -> bool;

    /// Human-readable name for this constraint computer
    fn name(&self) -> &str;
}

/// Wrapper that adapts an `AccessConstraintComputer` to implement `AccessConstraint`
///
/// This allows constraint computers (which can be user-defined in Python or Rust)
/// to be used anywhere an `AccessConstraint` is expected.
pub struct AccessConstraintComputerWrapper<T: AccessConstraintComputer> {
    computer: T,
}

impl<T: AccessConstraintComputer> AccessConstraintComputerWrapper<T> {
    /// Create a new wrapper around an `AccessConstraintComputer`
    pub fn new(computer: T) -> Self {
        Self { computer }
    }
}

impl<T: AccessConstraintComputer + 'static> AccessConstraint
    for AccessConstraintComputerWrapper<T>
{
    fn evaluate(
        &self,
        epoch: &Epoch,
        sat_state_ecef: &Vector6<f64>,
        location_ecef: &Vector3<f64>,
    ) -> bool {
        self.computer.evaluate(epoch, sat_state_ecef, location_ecef)
    }

    fn name(&self) -> &str {
        self.computer.name()
    }
}

/// Elevation angle constraint
///
/// Constrains access based on the elevation angle of the satellite above
/// the local horizon at the ground location.
///
/// Either or both bounds can be omitted by using `None`.
#[derive(Debug, Clone)]
pub struct ElevationConstraint {
    /// Minimum elevation angle (degrees), if specified
    pub min_elevation_deg: Option<f64>,

    /// Maximum elevation angle (degrees), if specified
    pub max_elevation_deg: Option<f64>,

    name: String,
}

/// Elevation mask constraint
///
/// Constrains access based on azimuth-dependent elevation masks.
/// Useful for ground stations with terrain obstructions or antenna limitations.
///
/// The mask is defined as a list of (azimuth, elevation) pairs in degrees.
/// Linear interpolation is used between points, and the mask wraps at 0°/360°.
#[derive(Debug, Clone)]
pub struct ElevationMaskConstraint {
    /// Elevation mask as (azimuth_deg, min_elevation_deg) pairs
    /// Must be sorted by azimuth in ascending order [0, 360)
    pub mask: Vec<(f64, f64)>,

    name: String,
}

impl ElevationConstraint {
    /// Create a new elevation constraint
    ///
    /// # Arguments
    /// - `min_elevation_deg`: Minimum elevation angle (degrees), or None for no minimum
    /// - `max_elevation_deg`: Maximum elevation angle (degrees), or None for no maximum
    ///
    /// # Returns
    /// `Ok(ElevationConstraint)` if at least one bound is specified
    ///
    /// # Errors
    /// Returns error if both bounds are None (unbounded constraint is meaningless)
    ///
    /// # Example
    /// ```
    /// use brahe::access::ElevationConstraint;
    ///
    /// // Typical ground station constraint: 5° minimum, no maximum
    /// let constraint = ElevationConstraint::new(Some(5.0), None).unwrap();
    ///
    /// // Both bounds specified
    /// let constraint = ElevationConstraint::new(Some(5.0), Some(85.0)).unwrap();
    ///
    /// // Only maximum (e.g., avoid zenith)
    /// let constraint = ElevationConstraint::new(None, Some(85.0)).unwrap();
    ///
    /// // Both None returns error
    /// assert!(ElevationConstraint::new(None, None).is_err());
    /// ```
    pub fn new(
        min_elevation_deg: Option<f64>,
        max_elevation_deg: Option<f64>,
    ) -> Result<Self, BraheError> {
        if min_elevation_deg.is_none() && max_elevation_deg.is_none() {
            return Err(BraheError::Error(
                "At least one bound (min or max) must be specified for ElevationConstraint"
                    .to_string(),
            ));
        }

        let name = match (min_elevation_deg, max_elevation_deg) {
            (Some(min), Some(max)) => format!("ElevationConstraint({:.2}° - {:.2}°)", min, max),
            (Some(min), None) => format!("ElevationConstraint(>= {:.2}°)", min),
            (None, Some(max)) => format!("ElevationConstraint(<= {:.2}°)", max),
            (None, None) => unreachable!(),
        };

        Ok(Self {
            min_elevation_deg,
            max_elevation_deg,
            name,
        })
    }
}

impl AccessConstraint for ElevationConstraint {
    fn evaluate(
        &self,
        _epoch: &Epoch,
        sat_state_ecef: &Vector6<f64>,
        location_ecef: &Vector3<f64>,
    ) -> bool {
        let sat_pos = sat_state_ecef.fixed_rows::<3>(0).into_owned();
        let elevation = compute_elevation(&sat_pos, location_ecef);

        let min_satisfied = self.min_elevation_deg.is_none_or(|min| elevation >= min);
        let max_satisfied = self.max_elevation_deg.is_none_or(|max| elevation <= max);

        min_satisfied && max_satisfied
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl std::fmt::Display for ElevationConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl ElevationMaskConstraint {
    /// Create a new elevation mask constraint
    ///
    /// # Arguments
    /// - `mask`: List of (azimuth_deg, min_elevation_deg) pairs, sorted by azimuth
    ///
    /// # Example
    /// ```
    /// use brahe::access::ElevationMaskConstraint;
    ///
    /// // Ground station with terrain obstruction to the north
    /// let mask = vec![
    ///     (0.0, 15.0),     // North: 15° minimum
    ///     (90.0, 5.0),     // East: 5° minimum
    ///     (180.0, 5.0),    // South: 5° minimum
    ///     (270.0, 5.0),    // West: 5° minimum
    /// ];
    /// let constraint = ElevationMaskConstraint::new(mask);
    /// ```
    pub fn new(mask: Vec<(f64, f64)>) -> Self {
        // Sort mask by azimuth
        let mut mask = mask;
        mask.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Find min and max (az, el) for name
        let (min_az, min_el) = mask
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .cloned()
            .unwrap_or((0.0, 0.0));

        let (max_az, max_el) = mask
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .cloned()
            .unwrap_or((0.0, 0.0));

        Self {
            mask,
            name: format!(
                "ElevationMaskConstraint(Min: {:.2}° at {:.2}°, Max: {:.2}° at {:.2}°)",
                min_el, min_az, max_el, max_az
            ),
        }
    }

    /// Interpolate minimum elevation for a given azimuth
    fn interpolate_min_elevation(&self, azimuth_deg: f64) -> f64 {
        if self.mask.is_empty() {
            return 0.0;
        }

        if self.mask.len() == 1 {
            return self.mask[0].1;
        }

        // Wrap azimuth to [0, 360)
        let az = ((azimuth_deg % 360.0) + 360.0) % 360.0;

        // Find bracketing points
        let mut lower_idx = 0;
        let mut upper_idx = 0;

        for (i, &(mask_az, _)) in self.mask.iter().enumerate() {
            if mask_az <= az {
                lower_idx = i;
            }
            if mask_az >= az && upper_idx == 0 {
                upper_idx = i;
                break;
            }
        }

        let az1 = self.mask[lower_idx].0;
        let el1 = self.mask[lower_idx].1;
        let az2: f64;
        let el2: f64;

        // Handle wraparound
        if upper_idx == 0 {
            // Azimuth is beyond last point, wrap to first
            az2 = self.mask[upper_idx].0 + 360.0; // Wrap
            el2 = self.mask[upper_idx].1;
        } else {
            az2 = self.mask[upper_idx].0;
            el2 = self.mask[upper_idx].1;
        }

        if lower_idx == upper_idx {
            return self.mask[lower_idx].1;
        }

        // Linear interpolation
        let t = (az - az1) / (az2 - az1);
        el1 + t * (el2 - el1)
    }
}

impl AccessConstraint for ElevationMaskConstraint {
    fn evaluate(
        &self,
        _epoch: &Epoch,
        sat_state_ecef: &Vector6<f64>,
        location_ecef: &Vector3<f64>,
    ) -> bool {
        let sat_pos = sat_state_ecef.fixed_rows::<3>(0).into_owned();
        let azimuth = compute_azimuth(&sat_pos, location_ecef);
        let elevation = compute_elevation(&sat_pos, location_ecef);

        let min_elevation = self.interpolate_min_elevation(azimuth);

        elevation >= min_elevation
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl std::fmt::Display for ElevationMaskConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Off-nadir angle constraint
///
/// Constrains access based on the off-nadir angle (angle from satellite nadir
/// to the ground location). Useful for imaging satellites with limited gimbal range.
///
/// Either or both bounds can be omitted by using `None`.
/// Note: Off-nadir angles are always non-negative by definition.
#[derive(Debug, Clone)]
pub struct OffNadirConstraint {
    /// Minimum off-nadir angle (degrees, >= 0)
    pub min_off_nadir_deg: Option<f64>,

    /// Maximum off-nadir angle (degrees, typically < 90)
    pub max_off_nadir_deg: Option<f64>,

    name: String,
}

impl OffNadirConstraint {
    /// Create a new off-nadir constraint
    ///
    /// # Arguments
    /// - `min_off_nadir_deg`: Minimum off-nadir angle (degrees, >= 0), or None for no minimum
    /// - `max_off_nadir_deg`: Maximum off-nadir angle (degrees), or None for no maximum
    ///
    /// # Returns
    /// `Ok(OffNadirConstraint)` if at least one bound is specified and all values are non-negative
    ///
    /// # Errors
    /// - Returns error if both bounds are None (unbounded constraint is meaningless)
    /// - Returns error if any angle is negative (off-nadir is always non-negative by definition)
    ///
    /// # Example
    /// ```
    /// use brahe::access::OffNadirConstraint;
    ///
    /// // Imaging satellite with 45° maximum slew angle
    /// let constraint = OffNadirConstraint::new(None, Some(45.0)).unwrap();
    ///
    /// // Minimum 10° to avoid nadir (e.g., for oblique imaging)
    /// let constraint = OffNadirConstraint::new(Some(10.0), Some(45.0)).unwrap();
    ///
    /// // Both None returns error
    /// assert!(OffNadirConstraint::new(None, None).is_err());
    ///
    /// // Negative angle returns error
    /// assert!(OffNadirConstraint::new(Some(-5.0), Some(45.0)).is_err());
    /// ```
    pub fn new(
        min_off_nadir_deg: Option<f64>,
        max_off_nadir_deg: Option<f64>,
    ) -> Result<Self, BraheError> {
        if min_off_nadir_deg.is_none() && max_off_nadir_deg.is_none() {
            return Err(BraheError::Error(
                "At least one bound (min or max) must be specified for OffNadirConstraint"
                    .to_string(),
            ));
        }

        // Validate that angles are non-negative
        if let Some(min) = min_off_nadir_deg.filter(|&m| m < 0.0) {
            return Err(BraheError::Error(format!(
                "Minimum off-nadir angle must be non-negative, got: {}",
                min
            )));
        }
        if let Some(max) = max_off_nadir_deg.filter(|&m| m < 0.0) {
            return Err(BraheError::Error(format!(
                "Maximum off-nadir angle must be non-negative, got: {}",
                max
            )));
        }

        let name = match (min_off_nadir_deg, max_off_nadir_deg) {
            (Some(min), Some(max)) => format!("OffNadirConstraint({:.1}° - {:.1}°)", min, max),
            (Some(min), None) => format!("OffNadirConstraint(>= {:.1}°)", min),
            (None, Some(max)) => format!("OffNadirConstraint(<= {:.1}°)", max),
            (None, None) => unreachable!(),
        };

        Ok(Self {
            min_off_nadir_deg,
            max_off_nadir_deg,
            name,
        })
    }
}

impl AccessConstraint for OffNadirConstraint {
    fn evaluate(
        &self,
        _epoch: &Epoch,
        sat_state_ecef: &Vector6<f64>,
        location_ecef: &Vector3<f64>,
    ) -> bool {
        let sat_pos = sat_state_ecef.fixed_rows::<3>(0).into_owned();
        let off_nadir = compute_off_nadir(&sat_pos, location_ecef);

        let min_satisfied = self.min_off_nadir_deg.is_none_or(|min| off_nadir >= min);
        let max_satisfied = self.max_off_nadir_deg.is_none_or(|max| off_nadir <= max);

        min_satisfied && max_satisfied
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl std::fmt::Display for OffNadirConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Local time constraint
///
/// Constrains access based on the local solar time at the ground location.
/// Useful for sun-synchronous orbits or daytime-only imaging.
///
/// Time windows are always stored internally as hours [0, 24), but can be
/// initialized using either hours or angles.
#[derive(Debug, Clone)]
pub struct LocalTimeConstraint {
    /// List of allowed time windows in hours since local midnight
    /// Each tuple is (start_hour, end_hour) in range [0.0, 24.0)
    pub time_windows: Vec<(f64, f64)>,

    name: String,
}

impl LocalTimeConstraint {
    /// Create a new local time constraint from military time format
    ///
    /// Default constructor using military time in HHMM format (0-2400).
    /// Examples: 600 = 6:00 AM, 1830 = 6:30 PM, 2200 = 10:00 PM
    ///
    /// # Arguments
    /// - `time_windows_military`: List of (start_military, end_military) tuples in range [0, 2400]
    ///
    /// # Returns
    /// `Ok(LocalTimeConstraint)` if all military times are valid
    ///
    /// # Errors
    /// Returns error if any military time value is outside [0, 2400] or has invalid minutes (>= 60)
    ///
    /// # Example
    /// ```
    /// use brahe::access::LocalTimeConstraint;
    ///
    /// // Only daytime (6 AM to 6 PM local time)
    /// let constraint = LocalTimeConstraint::new(vec![(600, 1800)]).unwrap();
    ///
    /// // Two windows: morning (6-9 AM) and evening (4-7 PM)
    /// let constraint = LocalTimeConstraint::new(vec![(600, 900), (1600, 1900)]).unwrap();
    ///
    /// // Overnight window (10 PM to 2 AM) - handles wrap-around
    /// let constraint = LocalTimeConstraint::new(vec![(2200, 200)]).unwrap();
    /// ```
    pub fn new(time_windows_military: Vec<(u16, u16)>) -> Result<Self, BraheError> {
        // Convert military time to hours
        let mut time_windows: Vec<(f64, f64)> = Vec::new();

        for (start_mil, end_mil) in &time_windows_military {
            // Validate military time format
            if *start_mil > 2400 {
                return Err(BraheError::Error(format!(
                    "Start military time must be in [0, 2400], got: {}",
                    start_mil
                )));
            }
            if *end_mil > 2400 {
                return Err(BraheError::Error(format!(
                    "End military time must be in [0, 2400], got: {}",
                    end_mil
                )));
            }

            // Extract hours and minutes
            let start_h = start_mil / 100;
            let start_m = start_mil % 100;
            let end_h = end_mil / 100;
            let end_m = end_mil % 100;

            // Validate minutes
            if start_m >= 60 {
                return Err(BraheError::Error(format!(
                    "Start minutes must be < 60, got: {} (from military time {})",
                    start_m, start_mil
                )));
            }
            if end_m >= 60 {
                return Err(BraheError::Error(format!(
                    "End minutes must be < 60, got: {} (from military time {})",
                    end_m, end_mil
                )));
            }

            // Convert to decimal hours
            let start_hour = f64::from(start_h) + f64::from(start_m) / 60.0;
            let end_hour = f64::from(end_h) + f64::from(end_m) / 60.0;

            time_windows.push((start_hour, end_hour));
        }

        // Build formatted name with wrap-around handling
        let mut window_parts = Vec::new();

        for (start_hour, end_hour) in &time_windows {
            if start_hour <= end_hour {
                // Normal window (no wrap-around)
                let s_h = start_hour.floor() as u32;
                let s_m = ((start_hour - start_hour.floor()) * 60.0).round() as u32;
                let e_h = end_hour.floor() as u32;
                let e_m = ((end_hour - end_hour.floor()) * 60.0).round() as u32;
                window_parts.push(format!("{:02}:{:02}-{:02}:{:02}", s_h, s_m, e_h, e_m));
            } else {
                // Wrap-around window (e.g., 22:00 to 2:00)
                // Split into two parts: start to midnight, midnight to end
                let s_h = start_hour.floor() as u32;
                let s_m = ((start_hour - start_hour.floor()) * 60.0).round() as u32;
                let e_h = end_hour.floor() as u32;
                let e_m = ((end_hour - end_hour.floor()) * 60.0).round() as u32;

                window_parts.push(format!("{:02}:{:02}-24:00", s_h, s_m));
                window_parts.push(format!("00:00-{:02}:{:02}", e_h, e_m));
            }
        }

        let window_str = window_parts.join(", ");

        Ok(Self {
            time_windows,
            name: format!("LocalTimeConstraint({})", window_str),
        })
    }

    /// Create a new local time constraint from decimal hour windows
    ///
    /// # Arguments
    /// - `time_windows`: List of (start_hour, end_hour) tuples in range [0, 24)
    ///
    /// # Returns
    /// `Ok(LocalTimeConstraint)` if all hours are valid
    ///
    /// # Errors
    /// Returns error if any hour value is outside [0, 24)
    ///
    /// # Example
    /// ```
    /// use brahe::access::LocalTimeConstraint;
    ///
    /// // Only daytime (6 AM to 6 PM local time)
    /// let constraint = LocalTimeConstraint::from_hours(vec![(6.0, 18.0)]).unwrap();
    ///
    /// // Two windows: morning (6-9 AM) and evening (4-7 PM)
    /// let constraint = LocalTimeConstraint::from_hours(vec![(6.0, 9.0), (16.0, 19.0)]).unwrap();
    ///
    /// // Overnight window (10 PM to 2 AM)
    /// let constraint = LocalTimeConstraint::from_hours(vec![(22.0, 2.0)]).unwrap();
    /// ```
    pub fn from_hours(time_windows: Vec<(f64, f64)>) -> Result<Self, BraheError> {
        // Validate windows
        for (start, end) in &time_windows {
            if *start < 0.0 || *start >= 24.0 {
                return Err(BraheError::Error(format!(
                    "Start hour must be in [0, 24), got: {}",
                    start
                )));
            }
            if *end < 0.0 || *end >= 24.0 {
                return Err(BraheError::Error(format!(
                    "End hour must be in [0, 24), got: {}",
                    end
                )));
            }
        }

        // Build formatted name with wrap-around handling
        let mut window_parts = Vec::new();

        for (start_hour, end_hour) in &time_windows {
            if start_hour <= end_hour {
                // Normal window (no wrap-around)
                let s_h = start_hour.floor() as u32;
                let s_m = ((start_hour - start_hour.floor()) * 60.0).round() as u32;
                let e_h = end_hour.floor() as u32;
                let e_m = ((end_hour - end_hour.floor()) * 60.0).round() as u32;
                window_parts.push(format!("{:02}:{:02}-{:02}:{:02}", s_h, s_m, e_h, e_m));
            } else {
                // Wrap-around window (e.g., 22:00 to 2:00)
                // Split into two parts: start to midnight, midnight to end
                let s_h = start_hour.floor() as u32;
                let s_m = ((start_hour - start_hour.floor()) * 60.0).round() as u32;
                let e_h = end_hour.floor() as u32;
                let e_m = ((end_hour - end_hour.floor()) * 60.0).round() as u32;

                window_parts.push(format!("{:02}:{:02}-24:00", s_h, s_m));
                window_parts.push(format!("00:00-{:02}:{:02}", e_h, e_m));
            }
        }

        let window_str = window_parts.join(", ");

        Ok(Self {
            time_windows,
            name: format!("LocalTimeConstraint({})", window_str),
        })
    }

    /// Create a new local time constraint from seconds since midnight
    ///
    /// Provides high precision for sub-minute time windows.
    ///
    /// # Arguments
    /// - `time_windows_seconds`: List of (start_seconds, end_seconds) tuples in range [0, 86400)
    ///
    /// # Returns
    /// `Ok(LocalTimeConstraint)` if all seconds are valid
    ///
    /// # Errors
    /// Returns error if any second value is outside [0, 86400)
    ///
    /// # Example
    /// ```
    /// use brahe::access::LocalTimeConstraint;
    ///
    /// // Only daytime (6 AM to 6 PM local time)
    /// let constraint = LocalTimeConstraint::from_seconds(vec![(21600.0, 64800.0)]).unwrap();
    ///
    /// // High precision: 11:00:00 to 13:00:00
    /// let constraint = LocalTimeConstraint::from_seconds(vec![(39600.0, 46800.0)]).unwrap();
    /// ```
    pub fn from_seconds(time_windows_seconds: Vec<(f64, f64)>) -> Result<Self, BraheError> {
        // Validate and convert seconds to hours
        let mut time_windows: Vec<(f64, f64)> = Vec::new();

        for (start_sec, end_sec) in &time_windows_seconds {
            if *start_sec < 0.0 || *start_sec >= 86400.0 {
                return Err(BraheError::Error(format!(
                    "Start seconds must be in [0, 86400), got: {}",
                    start_sec
                )));
            }
            if *end_sec < 0.0 || *end_sec >= 86400.0 {
                return Err(BraheError::Error(format!(
                    "End seconds must be in [0, 86400), got: {}",
                    end_sec
                )));
            }

            // Convert seconds to hours
            let start_hour = start_sec / 3600.0;
            let end_hour = end_sec / 3600.0;

            time_windows.push((start_hour, end_hour));
        }

        // Build formatted name with wrap-around handling
        let mut window_parts = Vec::new();

        for (start_hour, end_hour) in &time_windows {
            if start_hour <= end_hour {
                // Normal window (no wrap-around)
                let s_h = start_hour.floor() as u32;
                let s_m = ((start_hour - start_hour.floor()) * 60.0).round() as u32;
                let e_h = end_hour.floor() as u32;
                let e_m = ((end_hour - end_hour.floor()) * 60.0).round() as u32;
                window_parts.push(format!("{:02}:{:02}-{:02}:{:02}", s_h, s_m, e_h, e_m));
            } else {
                // Wrap-around window (e.g., 22:00 to 2:00)
                // Split into two parts: start to midnight, midnight to end
                let s_h = start_hour.floor() as u32;
                let s_m = ((start_hour - start_hour.floor()) * 60.0).round() as u32;
                let e_h = end_hour.floor() as u32;
                let e_m = ((end_hour - end_hour.floor()) * 60.0).round() as u32;

                window_parts.push(format!("{:02}:{:02}-24:00", s_h, s_m));
                window_parts.push(format!("00:00-{:02}:{:02}", e_h, e_m));
            }
        }

        let window_str = window_parts.join(", ");

        Ok(Self {
            time_windows,
            name: format!("LocalTimeConstraint({})", window_str),
        })
    }

    /// Compute local solar time in hours since midnight
    fn compute_local_time(&self, epoch: &Epoch, location_ecef: &Vector3<f64>) -> f64 {
        use crate::constants::AngleFormat;

        // Convert location to geodetic
        let geodetic = position_ecef_to_geodetic(*location_ecef, AngleFormat::Radians);

        let longitude_rad = geodetic.x;

        // UTC time in hours since midnight
        let (_year, _month, _day, hour, minute, second, nanosecond) =
            epoch.to_datetime_as_time_system(TimeSystem::UTC);

        let utc_hour =
            f64::from(hour) + f64::from(minute) / 60.0 + second / 3600.0 + nanosecond / 3.6e12;

        // Local solar time = UTC + longitude offset
        let local_time = utc_hour + longitude_rad.to_degrees() / 15.0;

        // Wrap to [0, 24)
        (local_time % 24.0 + 24.0) % 24.0
    }
}

impl AccessConstraint for LocalTimeConstraint {
    fn evaluate(
        &self,
        epoch: &Epoch,
        _sat_state_ecef: &Vector6<f64>,
        location_ecef: &Vector3<f64>,
    ) -> bool {
        let local_time = self.compute_local_time(epoch, location_ecef);

        // Check if local time falls within any allowed window
        self.time_windows.iter().any(|(start, end)| {
            if start <= end {
                local_time >= *start && local_time <= *end
            } else {
                // Handle wrap-around (e.g., 22:00 to 2:00)
                local_time >= *start || local_time <= *end
            }
        })
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl std::fmt::Display for LocalTimeConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Look direction for satellite imaging
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LookDirection {
    /// Left-looking (counterclockwise from velocity vector)
    Left,

    /// Right-looking (clockwise from velocity vector)
    Right,

    /// Either left or right
    Either,
}

impl std::fmt::Display for LookDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LookDirection::Left => write!(f, "Left"),
            LookDirection::Right => write!(f, "Right"),
            LookDirection::Either => write!(f, "Either"),
        }
    }
}

/// Look direction constraint
///
/// Constrains access based on whether the satellite must look left or right
/// relative to its velocity vector to view the ground location.
#[derive(Debug, Clone)]
pub struct LookDirectionConstraint {
    /// Required look direction
    pub allowed: LookDirection,

    name: String,
}

impl LookDirectionConstraint {
    /// Create a new look direction constraint
    ///
    /// # Example
    /// ```
    /// use brahe::access::{LookDirectionConstraint, LookDirection};
    ///
    /// // Satellite can only look right
    /// let constraint = LookDirectionConstraint::new(LookDirection::Right);
    /// ```
    pub fn new(allowed: LookDirection) -> Self {
        let direction_str = match allowed {
            LookDirection::Left => "Left",
            LookDirection::Right => "Right",
            LookDirection::Either => "Either",
        };

        Self {
            allowed,
            name: format!("LookDirectionConstraint({})", direction_str),
        }
    }
}

impl AccessConstraint for LookDirectionConstraint {
    fn evaluate(
        &self,
        _epoch: &Epoch,
        sat_state_ecef: &Vector6<f64>,
        location_ecef: &Vector3<f64>,
    ) -> bool {
        let look_dir = compute_look_direction(sat_state_ecef, location_ecef);

        match self.allowed {
            LookDirection::Either => true,
            allowed => look_dir == allowed,
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl std::fmt::Display for LookDirectionConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Ascending/descending pass type
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AscDsc {
    /// Ascending (moving from south to north)
    Ascending,

    /// Descending (moving from north to south)
    Descending,

    /// Either ascending or descending
    Either,
}

impl std::fmt::Display for AscDsc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AscDsc::Ascending => write!(f, "Ascending"),
            AscDsc::Descending => write!(f, "Descending"),
            AscDsc::Either => write!(f, "Either"),
        }
    }
}

/// Ascending/descending constraint
///
/// Constrains access based on whether the satellite is on an ascending or
/// descending pass (moving north or south).
#[derive(Debug, Clone)]
pub struct AscDscConstraint {
    /// Required pass type
    pub allowed: AscDsc,

    name: String,
}

impl AscDscConstraint {
    /// Create a new ascending/descending constraint
    ///
    /// # Example
    /// ```
    /// use brahe::access::{AscDscConstraint, AscDsc};
    ///
    /// // Only ascending passes
    /// let constraint = AscDscConstraint::new(AscDsc::Ascending);
    /// ```
    pub fn new(allowed: AscDsc) -> Self {
        let type_str = match allowed {
            AscDsc::Ascending => "Ascending",
            AscDsc::Descending => "Descending",
            AscDsc::Either => "Either",
        };

        Self {
            allowed,
            name: format!("AscDscConstraint({})", type_str),
        }
    }
}

impl AccessConstraint for AscDscConstraint {
    fn evaluate(
        &self,
        _epoch: &Epoch,
        sat_state_ecef: &Vector6<f64>,
        _location_ecef: &Vector3<f64>,
    ) -> bool {
        let pass_type = compute_asc_dsc(sat_state_ecef);

        match self.allowed {
            AscDsc::Either => true,
            allowed => pass_type == allowed,
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl std::fmt::Display for AscDscConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Constraint composition for logical combinations
///
/// Supports arbitrary nesting for complex constraint logic.
///
/// # Example
/// ```
/// use brahe::access::*;
///
/// // (Elevation >= 5°) AND (OffNadir <= 45°)
/// let c1 = Box::new(ElevationConstraint::new(Some(5.0), None).unwrap());
/// let c2 = Box::new(OffNadirConstraint::new(None, Some(45.0)).unwrap());
/// let composite = ConstraintComposite::All(vec![c1, c2]);
///
/// // NOT(LookDirection == Right)
/// let c3 = Box::new(LookDirectionConstraint::new(LookDirection::Right));
/// let not_right = ConstraintComposite::Not(c3);
///
/// // Nested: (Elevation >= 5°) AND (NOT(LookDirection == Right) OR (OffNadir <= 30°))
/// let c4 = Box::new(ElevationConstraint::new(Some(5.0), None).unwrap());
/// let c5 = Box::new(LookDirectionConstraint::new(LookDirection::Right));
/// let c6 = Box::new(OffNadirConstraint::new(None, Some(30.0)).unwrap());
/// let inner = ConstraintComposite::Any(vec![
///     Box::new(ConstraintComposite::Not(c5)),
///     c6,
/// ]);
/// let outer = ConstraintComposite::All(vec![c4, Box::new(inner)]);
/// ```
pub enum ConstraintComposite {
    /// All constraints must be satisfied (AND)
    All(Vec<Box<dyn AccessConstraint>>),

    /// At least one constraint must be satisfied (OR)
    Any(Vec<Box<dyn AccessConstraint>>),

    /// Constraint must NOT be satisfied (NOT)
    Not(Box<dyn AccessConstraint>),
}

impl AccessConstraint for ConstraintComposite {
    fn evaluate(
        &self,
        epoch: &Epoch,
        sat_state_ecef: &Vector6<f64>,
        location_ecef: &Vector3<f64>,
    ) -> bool {
        match self {
            ConstraintComposite::All(constraints) => constraints
                .iter()
                .all(|c| c.evaluate(epoch, sat_state_ecef, location_ecef)),

            ConstraintComposite::Any(constraints) => constraints
                .iter()
                .any(|c| c.evaluate(epoch, sat_state_ecef, location_ecef)),

            ConstraintComposite::Not(constraint) => {
                !constraint.evaluate(epoch, sat_state_ecef, location_ecef)
            }
        }
    }

    fn name(&self) -> &str {
        match self {
            ConstraintComposite::All(_) => "All",
            ConstraintComposite::Any(_) => "Any",
            ConstraintComposite::Not(_) => "Not",
        }
    }

    fn format_string(&self) -> String {
        format!("{}", self)
    }
}

impl std::fmt::Display for ConstraintComposite {
    /// Format constraint as nested logical expression
    ///
    /// Examples:
    /// - `ElevationConstraint(>= 5.00°) && OffNadirConstraint(<= 45.0°)`
    /// - `!LookDirectionConstraint(Right)`
    /// - `C1 && (!C2 || C3)`
    /// - `(C1 || C2) && C3` - brackets only when needed for precedence
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstraintComposite::All(constraints) => {
                if constraints.len() == 1 {
                    // Single constraint - no parens, no brackets
                    write!(f, "{}", format_constraint(&*constraints[0]))
                } else {
                    // Multiple constraints - no outer parens unless nested
                    for (i, constraint) in constraints.iter().enumerate() {
                        if i > 0 {
                            write!(f, " && ")?;
                        }
                        write_constraint_with_precedence(f, constraint.as_ref(), Precedence::And)?;
                    }
                    Ok(())
                }
            }
            ConstraintComposite::Any(constraints) => {
                if constraints.len() == 1 {
                    // Single constraint - no parens, no brackets
                    write!(f, "{}", format_constraint(&*constraints[0]))
                } else {
                    // Multiple constraints - no outer parens unless nested
                    for (i, constraint) in constraints.iter().enumerate() {
                        if i > 0 {
                            write!(f, " || ")?;
                        }
                        write_constraint_with_precedence(f, constraint.as_ref(), Precedence::Or)?;
                    }
                    Ok(())
                }
            }
            ConstraintComposite::Not(constraint) => {
                write!(f, "!")?;
                write_constraint_with_precedence(f, constraint.as_ref(), Precedence::Not)
            }
        }
    }
}

impl std::fmt::Debug for ConstraintComposite {
    /// Format constraint showing nested structure
    ///
    /// Examples:
    /// - `All([ElevationConstraint(...), OffNadirConstraint(...)])`
    /// - `Not(LookDirectionConstraint(...))`
    /// - `Any([All([...]), ...])`
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstraintComposite::All(constraints) => f
                .debug_tuple("All")
                .field(&constraints.iter().map(|c| c.name()).collect::<Vec<_>>())
                .finish(),
            ConstraintComposite::Any(constraints) => f
                .debug_tuple("Any")
                .field(&constraints.iter().map(|c| c.name()).collect::<Vec<_>>())
                .finish(),
            ConstraintComposite::Not(constraint) => {
                f.debug_tuple("Not").field(&constraint.name()).finish()
            }
        }
    }
}

/// Operator precedence for proper parenthesization
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Precedence {
    Or = 1,  // Lowest precedence
    And = 2, // Higher than OR
    Not = 3, // Highest precedence
}

/// Write a constraint with proper precedence handling
fn write_constraint_with_precedence(
    f: &mut std::fmt::Formatter<'_>,
    constraint: &dyn AccessConstraint,
    parent_precedence: Precedence,
) -> std::fmt::Result {
    let is_composite =
        constraint.name() == "All" || constraint.name() == "Any" || constraint.name() == "Not";

    if !is_composite {
        // Leaf constraint - just write the name, no brackets
        write!(f, "{}", constraint.name())
    } else {
        // Composite constraint - determine if we need parens based on precedence
        let constraint_precedence = match constraint.name() {
            "All" => Precedence::And,
            "Any" => Precedence::Or,
            "Not" => Precedence::Not,
            _ => Precedence::Not,
        };

        let needs_parens = constraint_precedence < parent_precedence;

        if needs_parens {
            write!(f, "(")?;
        }
        // Use format_string which calls Display for composites
        write!(f, "{}", constraint.format_string())?;
        if needs_parens {
            write!(f, ")")?;
        }
        Ok(())
    }
}

/// Helper to format any constraint for Display
///
/// Uses the format_string method which allows composites to provide their Display output
fn format_constraint(constraint: &dyn AccessConstraint) -> String {
    constraint.format_string()
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::constants::{AngleFormat, R_EARTH};
    use crate::coordinates::coordinate_types::EllipsoidalConversionType;
    use crate::coordinates::{position_geodetic_to_ecef, state_osculating_to_cartesian};
    use crate::frames::state_eci_to_ecef;
    use crate::time::{Epoch, TimeSystem};
    use crate::utils::testing::setup_global_test_eop;

    // -----------------
    // Helper functions
    // -----------------

    /// Test epoch for all geometry tests
    fn test_epoch() -> Epoch {
        Epoch::from_datetime(2025, 10, 16, 0, 0, 0.0, 0.0, TimeSystem::UTC)
    }

    /// Helper function for creating ECEF states from orbital elements
    fn test_sat_ecef_from_oe(oe: Vector6<f64>) -> Vector6<f64> {
        let state_eci = state_osculating_to_cartesian(oe, AngleFormat::Degrees);
        state_eci_to_ecef(test_epoch(), state_eci)
    }

    fn test_location_from_oe(oe: Vector6<f64>) -> Vector3<f64> {
        let loc_ecef_state = test_sat_ecef_from_oe(oe);

        // Convert to geodetic to get ground projection
        let loc_geod = position_ecef_to_geodetic(
            loc_ecef_state.fixed_rows::<3>(0).into_owned(),
            AngleFormat::Radians,
        );

        // Remove altitude to get ground location
        let sub_loc_geod = Vector3::new(loc_geod.x, loc_geod.y, 0.0); // Set altitude to 0
        position_geodetic_to_ecef(sub_loc_geod, AngleFormat::Radians).unwrap()
    }

    /// Test geometry 1: Ascending pass with location slightly west of ground track
    fn test_geometry_west_asc() -> (Epoch, Vector6<f64>, Vector3<f64>) {
        setup_global_test_eop();

        // Satellite orbital elements: [a, e, i, raan, argp, M]
        let sat_oe = Vector6::new(
            R_EARTH + 500e3, // Semi-major axis (500 km altitude)
            0.0,             // Eccentricity
            90.0,            // Inclination (degrees)
            0.0,             // RAAN
            0.0,             // Argument of perigee
            0.0,             // Mean anomaly
        );
        let sat_ecef = test_sat_ecef_from_oe(sat_oe);

        // Ground location a little bit to the west
        let loc_oe = Vector6::new(
            R_EARTH + 500e3, // Semi-major axis
            0.0,             // Eccentricity
            90.0,            // Inclination
            358.0,           // RAAN (degrees)
            0.0,             // Argument of perigee
            0.0,             // Mean anomaly
        );
        let loc_ecef = test_location_from_oe(loc_oe);

        (test_epoch(), sat_ecef, loc_ecef)
    }

    /// Test geometry 2: Descending pass with location will be slightly west of ground track
    fn test_geometry_west_dsc() -> (Epoch, Vector6<f64>, Vector3<f64>) {
        setup_global_test_eop();

        // Satellite orbital elements: [a, e, i, raan, argp, M]
        // Different RAAN and mean anomaly for descending pass
        let sat_oe = Vector6::new(
            R_EARTH + 500e3, // Semi-major axis (500 km altitude)
            0.0,             // Eccentricity
            90.0,            // Inclination (degrees)
            180.0,           // RAAN (different from geometry 1)
            0.0,             // Argument of perigee
            180.0,           // Mean anomaly (opposite side of orbit)
        );
        let sat_ecef = test_sat_ecef_from_oe(sat_oe);

        // Ground location (same as geometry 1)
        let loc_oe = Vector6::new(
            R_EARTH + 500e3, // Semi-major axis
            0.0,             // Eccentricity
            90.0,            // Inclination
            358.0,           // RAAN (degrees)
            0.0,             // Argument of perigee
            0.0,             // Mean anomaly
        );
        let loc_ecef = test_location_from_oe(loc_oe);

        (test_epoch(), sat_ecef, loc_ecef)
    }

    /// Helper to compute satellite ECEF position from ground location and look angles
    ///
    /// Given a ground location (lat/lon), azimuth, elevation, and range,
    /// computes the satellite position in ECEF coordinates.
    ///
    /// # Arguments
    /// - `lat_deg`: Ground station latitude (degrees)
    /// - `lon_deg`: Ground station longitude (degrees)
    /// - `alt_m`: Ground station altitude above ellipsoid (meters)
    /// - `az_deg`: Azimuth angle (degrees, clockwise from North)
    /// - `el_deg`: Elevation angle (degrees above horizon)
    /// - `range_m`: Slant range from ground station to satellite (meters)
    ///
    /// # Returns
    /// Satellite position in ECEF coordinates (meters)
    fn compute_sat_position_from_azel(
        lat_deg: f64,
        lon_deg: f64,
        alt_m: f64,
        az_deg: f64,
        el_deg: f64,
        range_m: f64,
    ) -> Vector3<f64> {
        use crate::coordinates::relative_position_enz_to_ecef;
        use std::f64::consts::PI;

        // Convert ground station to ECEF
        let location_geod = Vector3::new(lat_deg, lon_deg, alt_m);
        let location_ecef = position_geodetic_to_ecef(location_geod, AngleFormat::Degrees).unwrap();

        // Convert azimuth and elevation to radians
        let az_rad = az_deg * PI / 180.0;
        let el_rad = el_deg * PI / 180.0;

        // Convert AzEl to ENZ (inverse of position_enz_to_azel)
        // From the forward function:
        //   az = atan2(e, n)
        //   el = atan2(z, sqrt(e^2 + n^2))
        //   range = norm([e, n, z])
        // Inverse:
        let e = range_m * el_rad.cos() * az_rad.sin();
        let n = range_m * el_rad.cos() * az_rad.cos();
        let z = range_m * el_rad.sin();
        let relative_enz = Vector3::new(e, n, z);

        // Convert relative ENZ to ECEF using the library function
        relative_position_enz_to_ecef(
            location_ecef,
            relative_enz,
            EllipsoidalConversionType::Geodetic,
        )
    }

    // -------------
    // Actual tests
    // -------------

    #[test]
    fn test_constraint_name() {
        let elev = ElevationConstraint::new(Some(5.0), Some(90.0)).unwrap();
        assert_eq!(elev.name(), "ElevationConstraint(5.00° - 90.00°)");

        let elev_min_only = ElevationConstraint::new(Some(5.0), None).unwrap();
        assert_eq!(elev_min_only.name(), "ElevationConstraint(>= 5.00°)");

        let elev_max_only = ElevationConstraint::new(None, Some(90.0)).unwrap();
        assert_eq!(elev_max_only.name(), "ElevationConstraint(<= 90.00°)");

        let off_nadir = OffNadirConstraint::new(Some(10.0), Some(45.0)).unwrap();
        assert_eq!(off_nadir.name(), "OffNadirConstraint(10.0° - 45.0°)");

        let off_nadir_max_only = OffNadirConstraint::new(None, Some(45.0)).unwrap();
        assert_eq!(off_nadir_max_only.name(), "OffNadirConstraint(<= 45.0°)");

        let off_nadir_min_only = OffNadirConstraint::new(Some(10.0), None).unwrap();
        assert_eq!(off_nadir_min_only.name(), "OffNadirConstraint(>= 10.0°)");

        let local_time = LocalTimeConstraint::new(vec![(600, 1800)]).unwrap();
        assert_eq!(local_time.name(), "LocalTimeConstraint(06:00-18:00)");

        let local_time_multi = LocalTimeConstraint::new(vec![(200, 600), (1800, 2200)]).unwrap();
        assert_eq!(
            local_time_multi.name(),
            "LocalTimeConstraint(02:00-06:00, 18:00-22:00)"
        );

        let local_time_wrap = LocalTimeConstraint::new(vec![(2200, 200)]).unwrap();
        assert_eq!(
            local_time_wrap.name(),
            "LocalTimeConstraint(22:00-24:00, 00:00-02:00)"
        );

        let local_time_sec = LocalTimeConstraint::from_seconds(vec![(39600.0, 46800.0)]).unwrap();
        assert_eq!(local_time_sec.name(), "LocalTimeConstraint(11:00-13:00)");

        let local_time_hour = LocalTimeConstraint::from_hours(vec![(11.0, 13.0)]).unwrap();
        assert_eq!(local_time_hour.name(), "LocalTimeConstraint(11:00-13:00)");

        let look_dir = LookDirectionConstraint::new(LookDirection::Right);
        assert_eq!(look_dir.name(), "LookDirectionConstraint(Right)");

        let look_dir = LookDirectionConstraint::new(LookDirection::Left);
        assert_eq!(look_dir.name(), "LookDirectionConstraint(Left)");

        let look_dir = LookDirectionConstraint::new(LookDirection::Either);
        assert_eq!(look_dir.name(), "LookDirectionConstraint(Either)");

        let asc_dsc = AscDscConstraint::new(AscDsc::Ascending);
        assert_eq!(asc_dsc.name(), "AscDscConstraint(Ascending)");

        let asc_dsc = AscDscConstraint::new(AscDsc::Either);
        assert_eq!(asc_dsc.name(), "AscDscConstraint(Either)");

        let asc_dsc = AscDscConstraint::new(AscDsc::Descending);
        assert_eq!(asc_dsc.name(), "AscDscConstraint(Descending)");

        let elev_mask = ElevationMaskConstraint::new(vec![(0.0, 10.0), (180.0, 5.0)]);
        assert_eq!(
            elev_mask.name(),
            "ElevationMaskConstraint(Min: 5.00° at 180.00°, Max: 10.00° at 0.00°)"
        );
    }

    #[test]
    fn test_elevation_constraint_satisfied() {
        setup_global_test_eop();

        let constraint = ElevationConstraint::new(Some(10.0), None).unwrap();

        // Ground station at equator (0°N, 0°E, 0m altitude)
        let location = Vector3::new(0.0, 0.0, 0.0);
        let location_ecef = position_geodetic_to_ecef(location, AngleFormat::Degrees).unwrap();

        // Compute satellite position from ground station looking at:
        // - Azimuth: 90° (due East)
        // - Elevation: 45° (halfway to zenith)
        // - Range: 1000 km slant range
        let sat_pos_ecef = compute_sat_position_from_azel(
            0.0,    // Latitude: 0° (equator)
            0.0,    // Longitude: 0°
            0.0,    // Altitude: 0m (sea level)
            90.0,   // Azimuth: 90° (due East)
            45.0,   // Elevation: 45° (clearly > 10° minimum)
            1000e3, // Range: 1000 km
        );

        let sat_state = Vector6::new(
            sat_pos_ecef.x,
            sat_pos_ecef.y,
            sat_pos_ecef.z,
            0.0, // Velocity doesn't matter for elevation constraint
            0.0,
            0.0,
        );

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        // Should be satisfied (45° elevation > 10° minimum)
        assert!(constraint.evaluate(&epoch, &sat_state, &location_ecef));
    }

    #[test]
    fn test_elevation_constraint_at_limit() {
        setup_global_test_eop();

        // Minimum elevation constraint at 10°
        let constraint = ElevationConstraint::new(Some(10.0), None).unwrap();
        // Ground station at equator (0°N, 0°E, 0m altitude)
        let location = Vector3::new(0.0, 0.0, 0.0);
        let location_ecef = position_geodetic_to_ecef(location, AngleFormat::Degrees).unwrap();

        // Compute satellite position from ground station looking at:
        // - Azimuth: 90° (due East)
        // - Elevation: 10° (at the limit)
        // - Range: 1000 km slant range
        let sat_pos_ecef = compute_sat_position_from_azel(
            0.0,    // Latitude: 0° (equator)
            0.0,    // Longitude: 0°
            0.0,    // Altitude: 0m (sea level)
            90.0,   // Azimuth: 90° (due East)
            10.0,   // Elevation: 10° (at limit)
            1000e3, // Range: 1000 km
        );

        let sat_state = Vector6::new(
            sat_pos_ecef.x,
            sat_pos_ecef.y,
            sat_pos_ecef.z,
            0.0, // Velocity doesn't matter for elevation constraint
            0.0,
            0.0,
        );

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        // Should be satisfied (elevation exactly at 10° minimum)
        assert!(constraint.evaluate(&epoch, &sat_state, &location_ecef));
    }

    #[test]
    fn test_elevation_constraint_violated() {
        // Very high minimum elevation constraint (70-90°) should be violated
        // because realistic satellite-ground geometry has elevation around 66°
        let constraint = ElevationConstraint::new(Some(70.0), Some(90.0)).unwrap();

        // Ground station at equator (0°N, 0°E, 0m altitude)
        let location = Vector3::new(0.0, 0.0, 0.0);
        let location_ecef = position_geodetic_to_ecef(location, AngleFormat::Degrees).unwrap();

        // Compute satellite position from ground station looking at:
        // - Azimuth: 90° (due East)
        // - Elevation: 45° (halfway to zenith)
        // - Range: 1000 km slant range
        let sat_pos_ecef = compute_sat_position_from_azel(
            0.0,    // Latitude: 0° (equator)
            0.0,    // Longitude: 0°
            0.0,    // Altitude: 0m (sea level)
            90.0,   // Azimuth: 90° (due East)
            45.0,   // Elevation: 45° (clearly > 10° minimum)
            1000e3, // Range: 1000 km
        );

        let sat_state = Vector6::new(
            sat_pos_ecef.x,
            sat_pos_ecef.y,
            sat_pos_ecef.z,
            0.0, // Velocity doesn't matter for elevation constraint
            0.0,
            0.0,
        );

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        // Should be violated (elevation below 70°)
        assert!(!constraint.evaluate(&epoch, &sat_state, &location_ecef));
    }

    #[test]
    fn test_elevation_constraint_both_none_error() {
        // Both None should return error
        let result = ElevationConstraint::new(None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_elevation_mask_interpolation() {
        let mask = vec![(0.0, 10.0), (90.0, 10.0), (180.0, 20.0), (270.0, 20.0)];
        let constraint = ElevationMaskConstraint::new(mask);

        // At 45°, should interpolate between 10°
        let min_el = constraint.interpolate_min_elevation(90.0);
        assert_eq!(min_el, 10.0);

        // At 135°, should be 15°
        let min_el = constraint.interpolate_min_elevation(135.0);
        assert_eq!(min_el, 15.0);

        // At 225°, should be 20°
        let min_el = constraint.interpolate_min_elevation(225.0);
        assert_eq!(min_el, 20.0);

        // At 315°, should be 15°
        let min_el = constraint.interpolate_min_elevation(315.0);
        assert_eq!(min_el, 15.0);
    }

    #[test]
    fn test_elevation_mask_constraint() {
        setup_global_test_eop();

        let epoch: Epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        // Mask with higher elevation required to the north
        let mask = vec![
            (0.0, 10.0),   // North: 10° minimum
            (90.0, 10.0),  // East: 10° minimum
            (180.0, 20.0), // South: 20° minimum
            (270.0, 20.0), // West: 20° minimum
        ];
        let constraint = ElevationMaskConstraint::new(mask);

        let location = Vector3::new(0.0, 0.0, 0.0);
        let location_ecef = position_geodetic_to_ecef(location, AngleFormat::Degrees).unwrap();

        // Satellite position to North-East (45° azimuth) at 1000 km range and 30° elevation
        let sat_pos_ecef = compute_sat_position_from_azel(
            0.0,    // Latitude: 0° (equator)
            0.0,    // Longitude: 0°
            0.0,    // Altitude: 0m (sea level)
            45.0,   // Azimuth: 45° (NE)
            30.0,   // Elevation: 30°
            1000e3, // Range: 1000 km
        );
        let sat_state_north = Vector6::new(
            sat_pos_ecef.x,
            sat_pos_ecef.y,
            sat_pos_ecef.z,
            0.0, // Velocity doesn't matter for elevation constraint
            0.0,
            0.0,
        );

        assert!(constraint.evaluate(&epoch, &sat_state_north, &location_ecef));

        // Test South-East (135° azimuth) at 15° elevation - should pass
        let sat_pos_ecef = compute_sat_position_from_azel(
            0.0,    // Latitude: 0° (equator)
            0.0,    // Longitude: 0°
            0.0,    // Altitude: 0m (sea level)
            135.0,  // Azimuth: 135° (SE)
            15.0,   // Elevation: 15°
            1000e3, // Range: 1000 km
        );
        let sat_state_southeast = Vector6::new(
            sat_pos_ecef.x,
            sat_pos_ecef.y,
            sat_pos_ecef.z,
            0.0, // Velocity doesn't matter for elevation constraint
            0.0,
            0.0,
        );

        assert!(constraint.evaluate(&epoch, &sat_state_southeast, &location_ecef));

        // Test South-West (225° azimuth) at 15° elevation - should fail (needs 20°)
        let sat_pos_ecef = compute_sat_position_from_azel(
            0.0,    // Latitude: 0° (equator)
            0.0,    // Longitude: 0°
            0.0,    // Altitude: 0m (sea level)
            225.0,  // Azimuth: 225° (SW)
            15.0,   // Elevation: 15°
            1000e3, // Range: 1000 km
        );
        let sat_state_southwest = Vector6::new(
            sat_pos_ecef.x,
            sat_pos_ecef.y,
            sat_pos_ecef.z,
            0.0, // Velocity doesn't matter for elevation constraint
            0.0,
            0.0,
        );
        assert!(!constraint.evaluate(&epoch, &sat_state_southwest, &location_ecef));
    }

    #[test]
    fn test_off_nadir_constraint() {
        setup_global_test_eop();

        let (epoch, sat_state, location) = test_geometry_west_asc();

        // Test Large off-nadir angle (e.g., 60°) should be satisfied
        let constraint = OffNadirConstraint::new(None, Some(60.0)).unwrap();
        assert!(constraint.evaluate(&epoch, &sat_state, &location));

        // Test Small off-nadir angle (e.g., 5°) should be violated
        let constraint = OffNadirConstraint::new(None, Some(5.0)).unwrap();
        assert!(!constraint.evaluate(&epoch, &sat_state, &location));
    }

    #[test]
    fn test_off_nadir_constraint_both_none_error() {
        // Both None should return error
        let result = OffNadirConstraint::new(None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_off_nadir_constraint_negative_error() {
        // Negative angles should return error
        let result = OffNadirConstraint::new(Some(-5.0), Some(45.0));
        assert!(result.is_err());

        let result = OffNadirConstraint::new(Some(10.0), Some(-5.0));
        assert!(result.is_err());
    }

    #[test]
    fn test_local_time_constraint() {
        setup_global_test_eop();

        // Daytime only: 6 AM to 6 PM
        let constraint = LocalTimeConstraint::new(vec![(1100, 1300)]).unwrap();

        let location = Vector3::new(0.0, 0.0, 0.0);
        let location_ecef = position_geodetic_to_ecef(location, AngleFormat::Degrees).unwrap();

        let sat_state = Vector6::zeros();

        // Noon UTC at 0° longitude = noon local time
        let epoch_noon = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        assert!(constraint.evaluate(&epoch_noon, &sat_state, &location_ecef));

        // Midnight UTC at 0° longitude = midnight local time
        let epoch_midnight = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        assert!(!constraint.evaluate(&epoch_midnight, &sat_state, &location_ecef));
    }

    #[test]
    fn test_local_time_hour_validation() {
        // Out of range hours should return error
        let result = LocalTimeConstraint::from_hours(vec![(25.0, 26.0)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_local_time_military_validation() {
        // Out of range military time should return error
        let result = LocalTimeConstraint::new(vec![(2500, 2600)]);
        assert!(result.is_err());

        // Invalid minutes (>= 60) should return error
        let result = LocalTimeConstraint::new(vec![(1065, 1200)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_local_time_seconds_validation() {
        // Out of range seconds should return error
        let result = LocalTimeConstraint::from_seconds(vec![(90000.0, 95000.0)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_look_direction_constraint_asc() {
        setup_global_test_eop();

        let (epoch, sat_state, location) = test_geometry_west_asc();

        // Look Direction: Left should be satisfied
        let constraint = LookDirectionConstraint::new(LookDirection::Left);
        assert!(constraint.evaluate(&epoch, &sat_state, &location));

        // Look Direction: Right should be violated
        let constraint = LookDirectionConstraint::new(LookDirection::Right);
        assert!(!constraint.evaluate(&epoch, &sat_state, &location));

        // Look Direction: Either should be satisfied
        let constraint = LookDirectionConstraint::new(LookDirection::Either);
        assert!(constraint.evaluate(&epoch, &sat_state, &location));
    }

    #[test]
    fn test_look_direction_constraint_dsc() {
        setup_global_test_eop();
        let (epoch, sat_state, location) = test_geometry_west_dsc();

        // Look Direction: Right should be satisfied
        let constraint = LookDirectionConstraint::new(LookDirection::Right);
        assert!(constraint.evaluate(&epoch, &sat_state, &location));

        // Look Direction: Left should be violated
        let constraint = LookDirectionConstraint::new(LookDirection::Left);
        assert!(!constraint.evaluate(&epoch, &sat_state, &location));

        // Look Direction: Either should be satisfied
        let constraint = LookDirectionConstraint::new(LookDirection::Either);
        assert!(constraint.evaluate(&epoch, &sat_state, &location));
    }

    #[test]
    fn test_asc_dsc_constraint_ascending() {
        let (epoch, sat_state, location) = test_geometry_west_asc();
        let constraint = AscDscConstraint::new(AscDsc::Ascending);
        assert!(constraint.evaluate(&epoch, &sat_state, &location));
    }

    #[test]
    fn test_asc_dsc_constraint_descending() {
        let (epoch, sat_state, location) = test_geometry_west_dsc();
        let constraint = AscDscConstraint::new(AscDsc::Descending);
        assert!(constraint.evaluate(&epoch, &sat_state, &location));
    }

    #[test]
    fn test_asc_dsc_constraint_either() {
        let constraint = AscDscConstraint::new(AscDsc::Either);

        // Ascending case
        let (epoch_asc, sat_state_asc, location_asc) = test_geometry_west_asc();
        assert!(constraint.evaluate(&epoch_asc, &sat_state_asc, &location_asc));

        // Descending case
        let (epoch_dsc, sat_state_dsc, location_dsc) = test_geometry_west_dsc();
        assert!(constraint.evaluate(&epoch_dsc, &sat_state_dsc, &location_dsc));
    }

    #[test]
    fn test_constraint_composite_chaining() {
        // Test complex nested composition
        // Create: (Elevation >= 5°) AND (NOT(LookDirection == Right) OR (OffNadir <= 30°))

        let elev = Box::new(ElevationConstraint::new(Some(5.0), None).unwrap());
        let look = Box::new(LookDirectionConstraint::new(LookDirection::Right));
        let off_nadir = Box::new(OffNadirConstraint::new(None, Some(30.0)).unwrap());

        // NOT(LookDirection == Right)
        let not_right = Box::new(ConstraintComposite::Not(look));

        // (NOT(LookDirection == Right) OR (OffNadir <= 30°))
        let inner_or = Box::new(ConstraintComposite::Any(vec![not_right, off_nadir]));

        // (Elevation >= 5°) AND (inner_or)
        let outer_and = ConstraintComposite::All(vec![elev, inner_or]);

        // Test evaluation with dummy data
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let sat_state = Vector6::new(7000000.0, 0.0, 0.0, 0.0, 7500.0, 5000.0);
        let location = Vector3::new(6378137.0, 0.0, 0.0);

        // Just verify it evaluates without panicking
        let _ = outer_and.evaluate(&epoch, &sat_state, &location);
    }

    #[test]
    fn test_constraint_composite_display() {
        // Test Display implementation for pretty printing
        let c1 = Box::new(ElevationConstraint::new(Some(5.0), None).unwrap());
        let c2 = Box::new(OffNadirConstraint::new(None, Some(45.0)).unwrap());

        let and_composite = ConstraintComposite::All(vec![c1, c2]);
        let display = format!("{}", and_composite);

        // Should contain && and both constraint names, no brackets
        assert!(display.contains("&&"));
        assert!(display.contains("ElevationConstraint"));
        assert!(display.contains("OffNadirConstraint"));
        assert!(!display.contains("[")); // No brackets for leaf constraints

        // Test NOT - single constraint should have no brackets
        let c3 = Box::new(LookDirectionConstraint::new(LookDirection::Right));
        let not_composite = ConstraintComposite::Not(c3);
        let display_not = format!("{}", not_composite);

        assert!(display_not.contains("!"));
        assert!(display_not.contains("LookDirectionConstraint"));
        assert!(!display_not.contains("[")); // No brackets

        // Test OR
        let c4 = Box::new(AscDscConstraint::new(AscDsc::Ascending));
        let c5 = Box::new(AscDscConstraint::new(AscDsc::Descending));
        let or_composite = ConstraintComposite::Any(vec![c4, c5]);
        let display_or = format!("{}", or_composite);

        assert!(display_or.contains("||"));
        assert!(!display_or.contains("[")); // No brackets

        // Test single constraint in composite - should just show the constraint
        let c6 = Box::new(ElevationConstraint::new(Some(10.0), None).unwrap());
        let single_all = ConstraintComposite::All(vec![c6]);
        let display_single = format!("{}", single_all);

        // Debug to see actual output
        eprintln!("Single constraint display: '{}'", display_single);

        // Single constraint should be unwrapped completely
        assert!(display_single.contains("ElevationConstraint"));
        // Should not have extra notation since it's just one constraint
    }

    #[test]
    fn test_constraint_composite_nested_display() {
        // Test nested composite display with precedence
        // C1 && (C2 || C3) - OR has lower precedence, needs parens
        let c1 = Box::new(ElevationConstraint::new(Some(5.0), None).unwrap());
        let c2 = Box::new(OffNadirConstraint::new(None, Some(30.0)).unwrap());
        let c3 = Box::new(LookDirectionConstraint::new(LookDirection::Left));

        let inner_or = Box::new(ConstraintComposite::Any(vec![c2, c3]));
        let outer_and = ConstraintComposite::All(vec![c1, inner_or]);

        let display = format!("{}", outer_and);

        // Should have proper nesting with precedence-based parentheses
        assert!(display.contains("&&"));
        assert!(display.contains("||")); // Inner OR is now properly displayed
        assert!(display.contains("(")); // Parens around OR when nested in AND
        assert!(display.contains(")"));
        assert!(!display.contains("[")); // No brackets

        // Test !C1 && C2 - no parens needed for NOT
        let c4 = Box::new(LookDirectionConstraint::new(LookDirection::Right));
        let c5 = Box::new(OffNadirConstraint::new(None, Some(45.0)).unwrap());
        let not_c4 = Box::new(ConstraintComposite::Not(c4));
        let and_with_not = ConstraintComposite::All(vec![not_c4, c5]);
        let display_not = format!("{}", and_with_not);

        assert!(display_not.contains("!"));
        assert!(display_not.contains("&&"));
        // NOT has highest precedence, shouldn't need extra parens beyond the operator itself

        // Test (C1 && C2) || C3 - AND has higher precedence, needs parens when in OR
        let c6 = Box::new(ElevationConstraint::new(Some(10.0), None).unwrap());
        let c7 = Box::new(OffNadirConstraint::new(None, Some(20.0)).unwrap());
        let c8 = Box::new(AscDscConstraint::new(AscDsc::Ascending));
        let inner_and = Box::new(ConstraintComposite::All(vec![c6, c7]));
        let outer_or = ConstraintComposite::Any(vec![inner_and, c8]);
        let display_or = format!("{}", outer_or);

        assert!(display_or.contains("||"));
        assert!(display_or.contains("&&"));
        assert!(display_or.contains("(")); // Parens around AND when nested in OR
    }

    #[test]
    fn test_elevation_constraint_display() {
        // Min-only constraint
        let constraint_min = ElevationConstraint::new(Some(10.0), None).unwrap();
        let display_min = format!("{}", constraint_min);
        assert_eq!(display_min, "ElevationConstraint(>= 10.00°)");

        // Max-only constraint
        let constraint_max = ElevationConstraint::new(None, Some(80.0)).unwrap();
        let display_max = format!("{}", constraint_max);
        assert_eq!(display_max, "ElevationConstraint(<= 80.00°)");

        // Both min and max
        let constraint_both = ElevationConstraint::new(Some(5.0), Some(75.5)).unwrap();
        let display_both = format!("{}", constraint_both);
        assert_eq!(display_both, "ElevationConstraint(5.00° - 75.50°)");
    }

    #[test]
    fn test_elevation_mask_constraint_display() {
        let mask = vec![(0.0, 10.0), (90.0, 5.0), (180.0, 15.0), (270.0, 8.0)];
        let constraint = ElevationMaskConstraint::new(mask);
        let display = format!("{}", constraint);

        // Should show the range from min to max elevation in the mask
        assert!(display.starts_with("ElevationMaskConstraint"));
        assert!(display.contains("5.00°")); // Min elevation in mask
        assert!(display.contains("15.00°")); // Max elevation in mask
    }

    #[test]
    fn test_off_nadir_constraint_display() {
        // Min-only constraint
        let constraint_min = OffNadirConstraint::new(Some(10.0), None).unwrap();
        let display_min = format!("{}", constraint_min);
        assert_eq!(display_min, "OffNadirConstraint(>= 10.0°)");

        // Max-only constraint
        let constraint_max = OffNadirConstraint::new(None, Some(30.0)).unwrap();
        let display_max = format!("{}", constraint_max);
        assert_eq!(display_max, "OffNadirConstraint(<= 30.0°)");

        // Both min and max
        let constraint_both = OffNadirConstraint::new(Some(5.0), Some(45.0)).unwrap();
        let display_both = format!("{}", constraint_both);
        assert_eq!(display_both, "OffNadirConstraint(5.0° - 45.0°)");
    }

    #[test]
    fn test_local_time_constraint_display() {
        // Single window
        let constraint_single = LocalTimeConstraint::new(vec![(800, 1800)]).unwrap();
        let display_single = format!("{}", constraint_single);
        assert_eq!(display_single, "LocalTimeConstraint(08:00-18:00)");

        // Multiple windows
        let constraint_multi = LocalTimeConstraint::new(vec![(600, 900), (1700, 2000)]).unwrap();
        let display_multi = format!("{}", constraint_multi);
        assert_eq!(
            display_multi,
            "LocalTimeConstraint(06:00-09:00, 17:00-20:00)"
        );

        // Wrap-around window (splits into two windows: 22:00-24:00 and 00:00-02:00)
        let constraint_wrap = LocalTimeConstraint::new(vec![(2200, 200)]).unwrap();
        let display_wrap = format!("{}", constraint_wrap);
        assert_eq!(
            display_wrap,
            "LocalTimeConstraint(22:00-24:00, 00:00-02:00)"
        );
    }

    #[test]
    fn test_look_direction_constraint_display() {
        let constraint_left = LookDirectionConstraint::new(LookDirection::Left);
        assert_eq!(
            format!("{}", constraint_left),
            "LookDirectionConstraint(Left)"
        );

        let constraint_right = LookDirectionConstraint::new(LookDirection::Right);
        assert_eq!(
            format!("{}", constraint_right),
            "LookDirectionConstraint(Right)"
        );

        let constraint_either = LookDirectionConstraint::new(LookDirection::Either);
        assert_eq!(
            format!("{}", constraint_either),
            "LookDirectionConstraint(Either)"
        );
    }

    #[test]
    fn test_asc_dsc_constraint_display() {
        let constraint_asc = AscDscConstraint::new(AscDsc::Ascending);
        assert_eq!(format!("{}", constraint_asc), "AscDscConstraint(Ascending)");

        let constraint_dsc = AscDscConstraint::new(AscDsc::Descending);
        assert_eq!(
            format!("{}", constraint_dsc),
            "AscDscConstraint(Descending)"
        );

        let constraint_either = AscDscConstraint::new(AscDsc::Either);
        assert_eq!(format!("{}", constraint_either), "AscDscConstraint(Either)");
    }

    #[test]
    fn test_look_direction_enum_display() {
        assert_eq!(format!("{}", LookDirection::Left), "Left");
        assert_eq!(format!("{}", LookDirection::Right), "Right");
        assert_eq!(format!("{}", LookDirection::Either), "Either");
    }

    #[test]
    fn test_asc_dsc_enum_display() {
        assert_eq!(format!("{}", AscDsc::Ascending), "Ascending");
        assert_eq!(format!("{}", AscDsc::Descending), "Descending");
        assert_eq!(format!("{}", AscDsc::Either), "Either");
    }

    #[test]
    fn test_local_time_constraint_wrap_around_evaluation() {
        setup_global_test_eop();

        // Wrap-around window: 22:00 to 02:00 (night time)
        let constraint = LocalTimeConstraint::new(vec![(2200, 200)]).unwrap();

        let location = Vector3::new(0.0, 0.0, 0.0); // 0° longitude
        let location_ecef = position_geodetic_to_ecef(location, AngleFormat::Degrees).unwrap();
        let sat_state = Vector6::zeros();

        // 23:00 UTC = 23:00 local (inside window)
        let epoch_2300 = Epoch::from_datetime(2024, 1, 1, 23, 0, 0.0, 0.0, TimeSystem::UTC);
        assert!(
            constraint.evaluate(&epoch_2300, &sat_state, &location_ecef),
            "23:00 should be inside 22:00-02:00 window"
        );

        // 01:00 UTC = 01:00 local (inside window)
        let epoch_0100 = Epoch::from_datetime(2024, 1, 1, 1, 0, 0.0, 0.0, TimeSystem::UTC);
        assert!(
            constraint.evaluate(&epoch_0100, &sat_state, &location_ecef),
            "01:00 should be inside 22:00-02:00 window"
        );

        // 12:00 UTC = 12:00 local (outside window)
        let epoch_1200 = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        assert!(
            !constraint.evaluate(&epoch_1200, &sat_state, &location_ecef),
            "12:00 should be outside 22:00-02:00 window"
        );

        // 03:00 UTC = 03:00 local (outside window - just after wrap)
        let epoch_0300 = Epoch::from_datetime(2024, 1, 1, 3, 0, 0.0, 0.0, TimeSystem::UTC);
        assert!(
            !constraint.evaluate(&epoch_0300, &sat_state, &location_ecef),
            "03:00 should be outside 22:00-02:00 window"
        );
    }

    #[test]
    fn test_local_time_constraint_multiple_windows_evaluation() {
        setup_global_test_eop();

        // Multiple windows: 06:00-09:00 and 17:00-20:00
        let constraint = LocalTimeConstraint::new(vec![(600, 900), (1700, 2000)]).unwrap();

        let location = Vector3::new(0.0, 0.0, 0.0);
        let location_ecef = position_geodetic_to_ecef(location, AngleFormat::Degrees).unwrap();
        let sat_state = Vector6::zeros();

        // 07:00 UTC = 07:00 local (inside first window)
        let epoch_0700 = Epoch::from_datetime(2024, 1, 1, 7, 0, 0.0, 0.0, TimeSystem::UTC);
        assert!(constraint.evaluate(&epoch_0700, &sat_state, &location_ecef));

        // 18:00 UTC = 18:00 local (inside second window)
        let epoch_1800 = Epoch::from_datetime(2024, 1, 1, 18, 0, 0.0, 0.0, TimeSystem::UTC);
        assert!(constraint.evaluate(&epoch_1800, &sat_state, &location_ecef));

        // 12:00 UTC = 12:00 local (between windows - outside)
        let epoch_1200 = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        assert!(!constraint.evaluate(&epoch_1200, &sat_state, &location_ecef));

        // 21:00 UTC = 21:00 local (after both windows - outside)
        let epoch_2100 = Epoch::from_datetime(2024, 1, 1, 21, 0, 0.0, 0.0, TimeSystem::UTC);
        assert!(!constraint.evaluate(&epoch_2100, &sat_state, &location_ecef));
    }

    #[test]
    fn test_local_time_constraint_boundary_cases() {
        setup_global_test_eop();

        // Test at exact boundaries
        let constraint = LocalTimeConstraint::new(vec![(600, 1800)]).unwrap();

        let location = Vector3::new(0.0, 0.0, 0.0);
        let location_ecef = position_geodetic_to_ecef(location, AngleFormat::Degrees).unwrap();
        let sat_state = Vector6::zeros();

        // 06:00 UTC = 06:00 local (start boundary - should be inside)
        let epoch_0600 = Epoch::from_datetime(2024, 1, 1, 6, 0, 0.0, 0.0, TimeSystem::UTC);
        assert!(constraint.evaluate(&epoch_0600, &sat_state, &location_ecef));

        // 18:00 UTC = 18:00 local (end boundary - should be inside)
        let epoch_1800 = Epoch::from_datetime(2024, 1, 1, 18, 0, 0.0, 0.0, TimeSystem::UTC);
        assert!(constraint.evaluate(&epoch_1800, &sat_state, &location_ecef));

        // 05:59 UTC = just before start (outside)
        let epoch_0559 = Epoch::from_datetime(2024, 1, 1, 5, 59, 0.0, 0.0, TimeSystem::UTC);
        assert!(!constraint.evaluate(&epoch_0559, &sat_state, &location_ecef));
    }

    #[test]
    fn test_off_nadir_constraint_evaluate_min_only() {
        setup_global_test_eop();

        // Min-only constraint: off-nadir >= 20°
        let constraint = OffNadirConstraint::new(Some(20.0), None).unwrap();

        // Create test geometry
        let location = Vector3::new(0.0, 0.0, 0.0); // lat, lon, alt in degrees/meters
        let location_ecef = position_geodetic_to_ecef(location, AngleFormat::Degrees).unwrap();

        // Test various satellite positions
        // Satellite at 500 km, equatorial orbit
        let sat_oe = Vector6::new(
            R_EARTH + 500e3, // a
            0.0,             // e
            0.0,             // i
            0.0,             // RAAN
            0.0,             // argp
            0.0,             // M
        );
        let sat_state = test_sat_ecef_from_oe(sat_oe);

        // Evaluate constraint - should not panic
        // (exact result depends on geometry, but evaluation should work)
        let _result = constraint.evaluate(&test_epoch(), &sat_state, &location_ecef);

        // Test with different geometry
        let sat_oe_angled = Vector6::new(
            R_EARTH + 500e3, // a
            0.0,             // e
            45.0,            // i (inclined orbit)
            45.0,            // RAAN
            0.0,             // argp
            0.0,             // M
        );
        let sat_state_angled = test_sat_ecef_from_oe(sat_oe_angled);
        let _result_angled = constraint.evaluate(&test_epoch(), &sat_state_angled, &location_ecef);

        // Both evaluations should complete without panicking
    }

    #[test]
    fn test_off_nadir_constraint_evaluate_both_bounds() {
        setup_global_test_eop();

        // Both bounds: 10° <= off-nadir <= 30°
        let constraint = OffNadirConstraint::new(Some(10.0), Some(30.0)).unwrap();

        let location = Vector3::new(0.0, 0.0, 0.0);
        let location_ecef = position_geodetic_to_ecef(location, AngleFormat::Degrees).unwrap();

        // Test with various satellite positions
        let sat_oe = Vector6::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0);
        let sat_state = test_sat_ecef_from_oe(sat_oe);

        // Evaluate constraint - should not panic
        let _result = constraint.evaluate(&test_epoch(), &sat_state, &location_ecef);

        // Test with another position
        let sat_oe2 = Vector6::new(R_EARTH + 500e3, 0.0, 45.0, 90.0, 0.0, 45.0);
        let sat_state2 = test_sat_ecef_from_oe(sat_oe2);
        let _result2 = constraint.evaluate(&test_epoch(), &sat_state2, &location_ecef);
    }

    #[test]
    fn test_elevation_mask_constraint_evaluate_interpolation() {
        setup_global_test_eop();

        // Create mask with varying elevations by azimuth
        // North (0°): 15°, East (90°): 5°, South (180°): 10°, West (270°): 5°
        let mask = vec![(0.0, 15.0), (90.0, 5.0), (180.0, 10.0), (270.0, 5.0)];
        let constraint = ElevationMaskConstraint::new(mask);

        let location = Vector3::new(0.0, 45.0, 0.0); // lon, lat, alt
        let location_ecef = position_geodetic_to_ecef(location, AngleFormat::Degrees).unwrap();

        // Create satellite at different look angles to test interpolation
        // This is a simplified test - in practice, we'd need precise geometry

        // Test with a satellite that should satisfy the constraint
        let sat_oe = Vector6::new(R_EARTH + 500e3, 0.0, 45.0, 0.0, 0.0, 0.0);
        let sat_state = test_sat_ecef_from_oe(sat_oe);

        // Evaluate constraint (exact result depends on geometry, but should not panic)
        let _result = constraint.evaluate(&test_epoch(), &sat_state, &location_ecef);
        // Can't assert specific result without precise az/el calculation
    }

    #[test]
    fn test_elevation_mask_constraint_evaluate_azimuth_wrap() {
        setup_global_test_eop();

        // Create mask that wraps around 0°/360°
        // Test interpolation near the wrap point
        let mask = vec![
            (0.0, 10.0),
            (90.0, 5.0),
            (180.0, 5.0),
            (270.0, 5.0),
            (350.0, 12.0), // Close to 360° wrap
        ];
        let constraint = ElevationMaskConstraint::new(mask);

        let location = Vector3::new(0.0, 0.0, 0.0);
        let location_ecef = position_geodetic_to_ecef(location, AngleFormat::Degrees).unwrap();

        // Test with satellite geometry
        let sat_oe = Vector6::new(R_EARTH + 500e3, 0.0, 45.0, 0.0, 0.0, 0.0);
        let sat_state = test_sat_ecef_from_oe(sat_oe);

        // Evaluate - should handle wrap-around correctly
        let _result = constraint.evaluate(&test_epoch(), &sat_state, &location_ecef);
        // Should not panic due to wrap-around
    }

    #[test]
    fn test_access_constraint_computer_wrapper_new_and_evaluate() {
        setup_global_test_eop();

        // Create a simple test constraint computer
        struct TestConstraintComputer {
            threshold: f64,
        }

        impl AccessConstraintComputer for TestConstraintComputer {
            fn evaluate(
                &self,
                _epoch: &Epoch,
                sat_state_ecef: &Vector6<f64>,
                _location_ecef: &Vector3<f64>,
            ) -> bool {
                // Simple test: check if satellite altitude is above threshold
                let sat_pos = sat_state_ecef.fixed_rows::<3>(0).into_owned();
                let altitude = sat_pos.norm() - R_EARTH;
                altitude > self.threshold
            }

            fn name(&self) -> &str {
                "TestConstraintComputer"
            }
        }

        // Create wrapper using new()
        let computer = TestConstraintComputer { threshold: 400e3 }; // 400 km threshold
        let wrapper = AccessConstraintComputerWrapper::new(computer);

        // Test evaluate method
        let location = Vector3::new(0.0, 0.0, 0.0);
        let location_ecef = position_geodetic_to_ecef(location, AngleFormat::Degrees).unwrap();

        // Satellite at 500 km (should satisfy: 500 km > 400 km)
        let sat_oe_high = Vector6::new(R_EARTH + 500e3, 0.0, 45.0, 0.0, 0.0, 0.0);
        let sat_state_high = test_sat_ecef_from_oe(sat_oe_high);
        assert!(
            wrapper.evaluate(&test_epoch(), &sat_state_high, &location_ecef),
            "Satellite at 500 km should satisfy altitude > 400 km constraint"
        );

        // Satellite at 300 km (should not satisfy: 300 km < 400 km)
        let sat_oe_low = Vector6::new(R_EARTH + 300e3, 0.0, 45.0, 0.0, 0.0, 0.0);
        let sat_state_low = test_sat_ecef_from_oe(sat_oe_low);
        assert!(
            !wrapper.evaluate(&test_epoch(), &sat_state_low, &location_ecef),
            "Satellite at 300 km should not satisfy altitude > 400 km constraint"
        );

        // Test name() method
        assert_eq!(wrapper.name(), "TestConstraintComputer");
    }
}
