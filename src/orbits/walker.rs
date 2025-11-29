//! Walker constellation generator.
//!
//! This module provides the [`WalkerConstellationGenerator`] for generating
//! Walker constellation patterns using T:P:F notation.
//!
//! # Walker Constellation Patterns
//!
//! The Walker pattern distributes satellites across multiple orbital planes:
//! - **T** (Total): Total number of satellites
//! - **P** (Planes): Number of orbital planes
//! - **F** (Phasing): Phasing factor (0 to P-1)
//!
//! Two pattern types are supported:
//! - **Walker Delta**: 360° RAAN spread (global coverage, e.g., GPS, Galileo)
//! - **Walker Star**: 180° RAAN spread (polar coverage, e.g., Iridium)
//!
//! Geometry:
//! - Each plane contains T/P satellites (must divide evenly)
//! - Planes are evenly spaced in RAAN: spread/P apart (360°/P for Delta, 180°/P for Star)
//! - Satellites within each plane are evenly spaced in mean anomaly: 360°/(T/P) apart
//! - Phase offset between adjacent planes: F × 360°/T
//!
//! # Example
//!
//! ```rust
//! use brahe::orbits::{WalkerConstellationGenerator, WalkerPattern};
//! use brahe::time::{Epoch, TimeSystem};
//! use brahe::constants::{AngleFormat, R_EARTH};
//!
//! // Create a Walker Delta 24:3:1 constellation (e.g., GPS-like)
//! let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
//! let generator = WalkerConstellationGenerator::new(
//!     24,                    // 24 total satellites
//!     3,                     // 3 orbital planes
//!     1,                     // phasing factor 1
//!     R_EARTH + 20200e3,     // semi-major axis (GPS altitude)
//!     0.0,                   // eccentricity (circular)
//!     55.0,                  // inclination (degrees)
//!     0.0,                   // argument of perigee
//!     0.0,                   // reference RAAN
//!     0.0,                   // reference mean anomaly
//!     epoch,
//!     AngleFormat::Degrees,
//!     WalkerPattern::Delta,  // Walker Delta pattern (360° RAAN spread)
//! );
//!
//! // Generate propagators
//! let propagators = generator.as_keplerian_propagators(60.0);
//! assert_eq!(propagators.len(), 24);
//! ```

use std::f64::consts::PI;

use nalgebra::{DVector, Vector6};

use crate::constants::{AngleFormat, DEG2RAD, GM_EARTH, RAD2DEG};
use crate::coordinates::state_koe_to_eci;
use crate::orbits::create_tle_lines;
use crate::propagators::{
    DNumericalOrbitPropagator, ForceModelConfig, KeplerianPropagator, NumericalPropagationConfig,
    SGPPropagator,
};
use crate::time::Epoch;
use crate::utils::{BraheError, Identifiable};

/// Walker constellation pattern type.
///
/// Defines whether the constellation uses a Delta (360°) or Star (180°) RAAN spread.
///
/// # Pattern Types
///
/// - **Delta**: Orbital planes are distributed over 360° of RAAN, providing global coverage.
///   Used by GPS, Galileo, and similar navigation constellations.
///
/// - **Star**: Orbital planes are distributed over 180° of RAAN, providing concentrated
///   polar coverage. Used by Iridium and similar communications constellations.
///
/// # Example
///
/// ```rust
/// use brahe::orbits::WalkerPattern;
///
/// let delta = WalkerPattern::Delta;  // 360° RAAN spread
/// let star = WalkerPattern::Star;    // 180° RAAN spread
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WalkerPattern {
    /// Walker Delta pattern with 360° RAAN spread (global coverage)
    #[default]
    Delta,
    /// Walker Star pattern with 180° RAAN spread (polar coverage)
    Star,
}

/// Generator for Walker constellation patterns using T:P:F notation.
///
/// Walker constellations place satellites in circular or near-circular orbits with:
/// - T total satellites distributed across P planes
/// - Each plane inclined at the same angle
/// - Planes evenly spaced in RAAN (spread depends on pattern type)
/// - Satellites within each plane evenly spaced in mean anomaly
/// - Phase offset F controls inter-plane phasing
///
/// Two pattern types are supported via [`WalkerPattern`]:
/// - **Delta**: 360° RAAN spread (global coverage)
/// - **Star**: 180° RAAN spread (polar coverage)
///
/// All satellites in the constellation share the same semi-major axis, eccentricity,
/// inclination, and argument of perigee. They differ only in RAAN and mean anomaly.
#[derive(Debug, Clone)]
pub struct WalkerConstellationGenerator {
    /// Total number of satellites (T)
    pub total_satellites: usize,
    /// Number of orbital planes (P)
    pub num_planes: usize,
    /// Phasing factor (F), range 0 to P-1
    pub phasing: usize,
    /// Semi-major axis in meters
    pub semi_major_axis: f64,
    /// Eccentricity (dimensionless)
    pub eccentricity: f64,
    /// Inclination in radians (internal storage)
    pub inclination: f64,
    /// Argument of perigee in radians (internal storage)
    pub argument_of_perigee: f64,
    /// Reference RAAN for plane 0 in radians (internal storage)
    pub reference_raan: f64,
    /// Reference mean anomaly for first satellite in radians (internal storage)
    pub reference_mean_anomaly: f64,
    /// Reference epoch for ephemeris generation
    pub epoch: Epoch,
    /// Walker pattern type (Delta or Star)
    pub pattern: WalkerPattern,
    /// Optional base name for satellite naming
    base_name: Option<String>,
}

impl WalkerConstellationGenerator {
    /// Create a new Walker constellation generator.
    ///
    /// # Arguments
    ///
    /// * `t` - Total number of satellites (must be divisible by `p`)
    /// * `p` - Number of orbital planes
    /// * `f` - Phasing factor (0 to p-1)
    /// * `semi_major_axis` - Semi-major axis. Units: (m)
    /// * `eccentricity` - Eccentricity (dimensionless, typically near 0 for Walker)
    /// * `inclination` - Inclination. Units: (rad) or (deg) based on `angle_format`
    /// * `argument_of_perigee` - Argument of perigee. Units: (rad) or (deg)
    /// * `reference_raan` - RAAN for plane 0. Units: (rad) or (deg)
    /// * `reference_mean_anomaly` - Mean anomaly for first satellite. Units: (rad) or (deg)
    /// * `epoch` - Reference epoch for ephemeris generation
    /// * `angle_format` - Format for angular inputs (Degrees or Radians)
    /// * `pattern` - Walker pattern type (Delta for 360° RAAN, Star for 180° RAAN)
    ///
    /// # Returns
    ///
    /// New `WalkerConstellationGenerator` instance
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `p` is zero
    /// - `t` is not divisible by `p`
    /// - `f` >= `p`
    ///
    /// # Example
    ///
    /// ```rust
    /// use brahe::orbits::{WalkerConstellationGenerator, WalkerPattern};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::constants::{AngleFormat, R_EARTH};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// // Walker Delta constellation
    /// let walker = WalkerConstellationGenerator::new(
    ///     12, 3, 1,                    // T:P:F = 12:3:1
    ///     R_EARTH + 780e3,             // ~780 km altitude
    ///     0.001,                       // near-circular
    ///     98.0, 0.0, 0.0, 0.0,         // inclination, argp, raan, M
    ///     epoch,
    ///     AngleFormat::Degrees,
    ///     WalkerPattern::Delta,        // 360° RAAN spread
    /// );
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        t: usize,
        p: usize,
        f: usize,
        semi_major_axis: f64,
        eccentricity: f64,
        inclination: f64,
        argument_of_perigee: f64,
        reference_raan: f64,
        reference_mean_anomaly: f64,
        epoch: Epoch,
        angle_format: AngleFormat,
        pattern: WalkerPattern,
    ) -> Self {
        // Validation
        if p == 0 {
            panic!("Number of planes (P) must be greater than zero");
        }
        if !t.is_multiple_of(p) {
            panic!(
                "Total satellites ({}) must be divisible by number of planes ({})",
                t, p
            );
        }
        if f >= p {
            panic!(
                "Phasing factor ({}) must be less than number of planes ({})",
                f, p
            );
        }

        // Convert angles to radians if needed
        let (inc, argp, raan, ma) = match angle_format {
            AngleFormat::Degrees => (
                inclination * DEG2RAD,
                argument_of_perigee * DEG2RAD,
                reference_raan * DEG2RAD,
                reference_mean_anomaly * DEG2RAD,
            ),
            AngleFormat::Radians => (
                inclination,
                argument_of_perigee,
                reference_raan,
                reference_mean_anomaly,
            ),
        };

        Self {
            total_satellites: t,
            num_planes: p,
            phasing: f,
            semi_major_axis,
            eccentricity,
            inclination: inc,
            argument_of_perigee: argp,
            reference_raan: raan,
            reference_mean_anomaly: ma,
            epoch,
            pattern,
            base_name: None,
        }
    }

    /// Set a base name for satellite naming.
    ///
    /// When set, satellites will be named as "{base_name}-P{plane}-S{sat}"
    /// (e.g., "GPS-P0-S0", "GPS-P0-S1", "GPS-P1-S0", etc.)
    ///
    /// # Arguments
    ///
    /// * `name` - Base name for the constellation satellites
    ///
    /// # Returns
    ///
    /// Self with the base name set
    ///
    /// # Example
    ///
    /// ```rust
    /// use brahe::orbits::{WalkerConstellationGenerator, WalkerPattern};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::constants::{AngleFormat, R_EARTH};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let walker = WalkerConstellationGenerator::new(
    ///     12, 3, 1, R_EARTH + 780e3, 0.001, 98.0, 0.0, 0.0, 0.0,
    ///     epoch, AngleFormat::Degrees, WalkerPattern::Delta,
    /// ).with_base_name("Constellation");
    /// ```
    pub fn with_base_name(mut self, name: &str) -> Self {
        self.base_name = Some(name.to_string());
        self
    }

    /// Get the number of satellites per plane (T/P).
    ///
    /// # Returns
    ///
    /// Number of satellites in each orbital plane
    pub fn satellites_per_plane(&self) -> usize {
        self.total_satellites / self.num_planes
    }

    /// Get the satellite name for a given plane and satellite index.
    ///
    /// Returns `Some(name)` if `with_base_name()` was called, `None` otherwise.
    fn satellite_name(&self, plane_index: usize, sat_index: usize) -> Option<String> {
        self.base_name
            .as_ref()
            .map(|name| format!("{}-P{}-S{}", name, plane_index, sat_index))
    }

    /// Get the global satellite ID for a given plane and satellite index.
    ///
    /// IDs are assigned sequentially: plane 0 satellites get IDs 0..S-1,
    /// plane 1 gets S..2S-1, etc.
    fn satellite_id(&self, plane_index: usize, sat_index: usize) -> u64 {
        (plane_index * self.satellites_per_plane() + sat_index) as u64
    }

    /// Get Keplerian elements for a specific satellite.
    ///
    /// Elements are returned as `[a, e, i, raan, argp, M]` in SI units
    /// (meters for semi-major axis, radians for angles).
    ///
    /// # Arguments
    ///
    /// * `plane_index` - Plane index (0 to P-1)
    /// * `sat_index` - Satellite index within plane (0 to T/P - 1)
    ///
    /// # Returns
    ///
    /// Keplerian elements vector `[a, e, i, raan, argp, M]`
    ///
    /// # Panics
    ///
    /// Panics if plane_index or sat_index are out of bounds
    pub fn satellite_elements(&self, plane_index: usize, sat_index: usize) -> Vector6<f64> {
        let sats_per_plane = self.satellites_per_plane();

        if plane_index >= self.num_planes {
            panic!(
                "Plane index ({}) must be less than number of planes ({})",
                plane_index, self.num_planes
            );
        }
        if sat_index >= sats_per_plane {
            panic!(
                "Satellite index ({}) must be less than satellites per plane ({})",
                sat_index, sats_per_plane
            );
        }

        // RAAN spread depends on pattern: Delta = 360°, Star = 180°
        let raan_spread = match self.pattern {
            WalkerPattern::Delta => 2.0 * PI,
            WalkerPattern::Star => PI,
        };
        let raan_spacing = raan_spread / self.num_planes as f64;
        let raan = self.reference_raan + (plane_index as f64) * raan_spacing;

        // Mean anomaly spacing within plane
        let ma_spacing = 2.0 * PI / sats_per_plane as f64;
        let base_ma = (sat_index as f64) * ma_spacing;

        // Phase offset for this plane: plane_index * phasing * (2*PI / total_satellites)
        let phase_per_satellite = 2.0 * PI / self.total_satellites as f64;
        let phase_offset = (plane_index as f64) * (self.phasing as f64) * phase_per_satellite;

        // Final mean anomaly (modulo 2*PI to keep in [0, 2*PI))
        let mean_anomaly = (self.reference_mean_anomaly + base_ma + phase_offset) % (2.0 * PI);

        // Keep RAAN in [0, 2*PI)
        let raan_normalized = raan % (2.0 * PI);

        Vector6::new(
            self.semi_major_axis,
            self.eccentricity,
            self.inclination,
            raan_normalized,
            self.argument_of_perigee,
            mean_anomaly,
        )
    }

    /// Get Keplerian elements for all satellites.
    ///
    /// Elements are returned in order: all satellites in plane 0, then plane 1, etc.
    /// Each element vector is `[a, e, i, raan, argp, M]` in meters/radians.
    ///
    /// # Returns
    ///
    /// Vector of Keplerian element vectors, one per satellite
    pub fn all_elements(&self) -> Vec<Vector6<f64>> {
        let sats_per_plane = self.satellites_per_plane();
        let mut elements = Vec::with_capacity(self.total_satellites);

        for plane in 0..self.num_planes {
            for sat in 0..sats_per_plane {
                elements.push(self.satellite_elements(plane, sat));
            }
        }

        elements
    }

    /// Generate Keplerian propagators for all satellites in the constellation.
    ///
    /// Each propagator is initialized with the satellite's orbital elements at
    /// the generator's epoch. Propagators are returned in order: all satellites
    /// in plane 0, then plane 1, etc.
    ///
    /// If `with_base_name()` was called, each propagator will be named
    /// "{base_name}-P{plane}-S{sat}" and assigned a unique ID.
    ///
    /// # Arguments
    ///
    /// * `step_size` - Step size in seconds for propagation
    ///
    /// # Returns
    ///
    /// Vector of `KeplerianPropagator` instances, one per satellite
    ///
    /// # Example
    ///
    /// ```rust
    /// use brahe::orbits::{WalkerConstellationGenerator, WalkerPattern};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::constants::{AngleFormat, R_EARTH};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let walker = WalkerConstellationGenerator::new(
    ///     12, 3, 1, R_EARTH + 780e3, 0.001, 98.0, 0.0, 0.0, 0.0,
    ///     epoch, AngleFormat::Degrees, WalkerPattern::Delta,
    /// ).with_base_name("Sat");
    ///
    /// let propagators = walker.as_keplerian_propagators(60.0);
    /// assert_eq!(propagators.len(), 12);
    /// ```
    pub fn as_keplerian_propagators(&self, step_size: f64) -> Vec<KeplerianPropagator> {
        let sats_per_plane = self.satellites_per_plane();
        let mut propagators = Vec::with_capacity(self.total_satellites);

        for plane in 0..self.num_planes {
            for sat in 0..sats_per_plane {
                let elements = self.satellite_elements(plane, sat);

                let mut prop = KeplerianPropagator::from_keplerian(
                    self.epoch,
                    elements,
                    AngleFormat::Radians,
                    step_size,
                );

                // Apply naming and ID
                let id = self.satellite_id(plane, sat);
                prop = prop.with_id(id);

                if let Some(name) = self.satellite_name(plane, sat) {
                    prop = prop.with_name(&name);
                }

                propagators.push(prop);
            }
        }

        propagators
    }

    /// Generate SGP propagators for all satellites in the constellation.
    ///
    /// This method creates TLE data for each satellite using the provided
    /// drag and mean motion derivative parameters. Synthetic NORAD IDs are
    /// generated starting at 99000.
    ///
    /// Note: SGP4 uses mean elements internally. The Keplerian elements from
    /// this generator are treated as osculating elements. For highest fidelity,
    /// consider using `as_keplerian_propagators()` or `as_numerical_propagators()`.
    ///
    /// # Arguments
    ///
    /// * `step_size` - Step size in seconds for propagation
    /// * `bstar` - B* drag term (Earth radii^-1)
    /// * `ndt2` - First derivative of mean motion divided by 2 (rev/day²)
    /// * `nddt6` - Second derivative of mean motion divided by 6 (rev/day³)
    ///
    /// # Returns
    ///
    /// Result containing vector of `SGPPropagator` instances, or error if TLE
    /// generation fails for any satellite.
    ///
    /// # Example
    ///
    /// ```rust
    /// use brahe::orbits::{WalkerConstellationGenerator, WalkerPattern};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::constants::{AngleFormat, R_EARTH};
    /// use brahe::eop::{StaticEOPProvider, set_global_eop_provider};
    ///
    /// // Initialize EOP (required for SGP propagator)
    /// let eop = StaticEOPProvider::from_zero();
    /// set_global_eop_provider(eop);
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let walker = WalkerConstellationGenerator::new(
    ///     6, 3, 1, R_EARTH + 780e3, 0.001, 98.0, 0.0, 0.0, 0.0,
    ///     epoch, AngleFormat::Degrees, WalkerPattern::Delta,
    /// );
    ///
    /// let propagators = walker.as_sgp_propagators(60.0, 0.0, 0.0, 0.0).unwrap();
    /// assert_eq!(propagators.len(), 6);
    /// ```
    pub fn as_sgp_propagators(
        &self,
        step_size: f64,
        bstar: f64,
        ndt2: f64,
        nddt6: f64,
    ) -> Result<Vec<SGPPropagator>, BraheError> {
        let sats_per_plane = self.satellites_per_plane();
        let mut propagators = Vec::with_capacity(self.total_satellites);

        for plane in 0..self.num_planes {
            for sat in 0..sats_per_plane {
                let elements = self.satellite_elements(plane, sat);
                let global_id = self.satellite_id(plane, sat);

                // Generate synthetic NORAD ID (99000+)
                let norad_id = format!("{:05}", 99000 + global_id);

                // Convert semi-major axis to mean motion (rev/day)
                let mean_motion_rad_per_sec =
                    (GM_EARTH / (elements[0] * elements[0] * elements[0])).sqrt();
                let mean_motion_revs_per_day = mean_motion_rad_per_sec * 86400.0 / (2.0 * PI);

                // Convert angles to degrees for TLE
                let inclination_deg = elements[2] * RAD2DEG;
                let raan_deg = elements[3] * RAD2DEG;
                let argp_deg = elements[4] * RAD2DEG;
                let ma_deg = elements[5] * RAD2DEG;

                // Create TLE lines
                let (line1, line2) = create_tle_lines(
                    &self.epoch,
                    &norad_id,
                    'U', // Unclassified
                    "",  // No international designator
                    mean_motion_revs_per_day,
                    elements[1], // eccentricity
                    inclination_deg,
                    raan_deg,
                    argp_deg,
                    ma_deg,
                    ndt2,  // first_derivative (ndot/2)
                    nddt6, // second_derivative (nddot/6)
                    bstar, // B* drag term
                    0,     // ephemeris_type
                    0,     // element_set_number
                    0,     // revolution_number
                )?;

                // Create SGP propagator from TLE
                let mut prop = SGPPropagator::from_tle(&line1, &line2, step_size)?;

                // Apply naming and ID
                prop = prop.with_id(global_id);

                if let Some(name) = self.satellite_name(plane, sat) {
                    prop = prop.with_name(&name);
                }

                propagators.push(prop);
            }
        }

        Ok(propagators)
    }

    /// Generate numerical orbit propagators for all satellites in the constellation.
    ///
    /// This method creates high-fidelity numerical propagators using the specified
    /// force model and propagation configuration. Each satellite is initialized
    /// with its Cartesian ECI state derived from the Keplerian elements.
    ///
    /// # Arguments
    ///
    /// * `propagation_config` - Numerical propagation configuration (integrator settings)
    /// * `force_config` - Force model configuration (gravity, drag, SRP, third-body, etc.)
    /// * `params` - Optional parameter vector `[mass, drag_area, Cd, srp_area, Cr, ...]`
    ///
    /// # Returns
    ///
    /// Result containing vector of `DNumericalOrbitPropagator` instances, or error
    /// if propagator creation fails.
    ///
    /// # Example
    ///
    /// ```rust
    /// use brahe::orbits::{WalkerConstellationGenerator, WalkerPattern};
    /// use brahe::propagators::{NumericalPropagationConfig, ForceModelConfig};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::constants::{AngleFormat, R_EARTH};
    /// use brahe::eop::{StaticEOPProvider, set_global_eop_provider};
    ///
    /// // Initialize EOP
    /// let eop = StaticEOPProvider::from_zero();
    /// set_global_eop_provider(eop);
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let walker = WalkerConstellationGenerator::new(
    ///     6, 3, 1, R_EARTH + 780e3, 0.001, 98.0, 0.0, 0.0, 0.0,
    ///     epoch, AngleFormat::Degrees, WalkerPattern::Delta,
    /// );
    ///
    /// let propagators = walker.as_numerical_propagators(
    ///     NumericalPropagationConfig::default(),
    ///     ForceModelConfig::earth_gravity(),
    ///     None,
    /// ).unwrap();
    /// assert_eq!(propagators.len(), 6);
    /// ```
    pub fn as_numerical_propagators(
        &self,
        propagation_config: NumericalPropagationConfig,
        force_config: ForceModelConfig,
        params: Option<DVector<f64>>,
    ) -> Result<Vec<DNumericalOrbitPropagator>, BraheError> {
        let sats_per_plane = self.satellites_per_plane();
        let mut propagators = Vec::with_capacity(self.total_satellites);

        for plane in 0..self.num_planes {
            for sat in 0..sats_per_plane {
                let elements = self.satellite_elements(plane, sat);

                // Convert Keplerian to Cartesian ECI
                let state_vec6 = state_koe_to_eci(elements, AngleFormat::Radians);
                let state = DVector::from_iterator(6, state_vec6.iter().copied());

                // Create numerical propagator
                let mut prop = DNumericalOrbitPropagator::new(
                    self.epoch,
                    state,
                    propagation_config.clone(),
                    force_config.clone(),
                    params.clone(),
                    None, // additional_dynamics
                    None, // control_input
                    None, // initial_covariance
                )?;

                // Apply naming and ID
                let global_id = self.satellite_id(plane, sat);
                prop.id = Some(global_id);

                if let Some(name) = self.satellite_name(plane, sat) {
                    prop.name = Some(name);
                }

                propagators.push(prop);
            }
        }

        Ok(propagators)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn test_epoch() -> Epoch {
        Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, crate::time::TimeSystem::UTC)
    }

    #[test]
    fn test_walker_new_basic() {
        let walker = WalkerConstellationGenerator::new(
            12,
            3,
            1,
            7000e3,
            0.001,
            98.0,
            0.0,
            0.0,
            0.0,
            test_epoch(),
            AngleFormat::Degrees,
            WalkerPattern::Delta,
        );

        assert_eq!(walker.total_satellites, 12);
        assert_eq!(walker.num_planes, 3);
        assert_eq!(walker.phasing, 1);
        assert_eq!(walker.satellites_per_plane(), 4);
        assert_abs_diff_eq!(walker.semi_major_axis, 7000e3);
        assert_abs_diff_eq!(walker.inclination, 98.0 * DEG2RAD, epsilon = 1e-10);
        assert_eq!(walker.pattern, WalkerPattern::Delta);
    }

    #[test]
    #[should_panic(expected = "Number of planes (P) must be greater than zero")]
    fn test_walker_zero_planes() {
        WalkerConstellationGenerator::new(
            12,
            0,
            0,
            7000e3,
            0.001,
            98.0,
            0.0,
            0.0,
            0.0,
            test_epoch(),
            AngleFormat::Degrees,
            WalkerPattern::Delta,
        );
    }

    #[test]
    #[should_panic(expected = "must be divisible by")]
    fn test_walker_not_divisible() {
        WalkerConstellationGenerator::new(
            10,
            3,
            1,
            7000e3,
            0.001,
            98.0,
            0.0,
            0.0,
            0.0,
            test_epoch(),
            AngleFormat::Degrees,
            WalkerPattern::Delta,
        );
    }

    #[test]
    #[should_panic(expected = "Phasing factor")]
    fn test_walker_phasing_too_large() {
        WalkerConstellationGenerator::new(
            12,
            3,
            3,
            7000e3,
            0.001,
            98.0,
            0.0,
            0.0,
            0.0,
            test_epoch(),
            AngleFormat::Degrees,
            WalkerPattern::Delta,
        );
    }

    #[test]
    fn test_walker_raan_spacing_delta() {
        // Walker Delta: 6 satellites in 3 planes -> planes at RAAN 0, 120, 240 degrees
        let walker = WalkerConstellationGenerator::new(
            6,
            3,
            0,
            7000e3,
            0.0,
            45.0,
            0.0,
            0.0,
            0.0,
            test_epoch(),
            AngleFormat::Degrees,
            WalkerPattern::Delta,
        );

        let elem0 = walker.satellite_elements(0, 0);
        let elem1 = walker.satellite_elements(1, 0);
        let elem2 = walker.satellite_elements(2, 0);

        // RAAN spacing should be 120 degrees for Delta (360/3)
        assert_abs_diff_eq!(elem0[3], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(elem1[3], 120.0 * DEG2RAD, epsilon = 1e-10);
        assert_abs_diff_eq!(elem2[3], 240.0 * DEG2RAD, epsilon = 1e-10);
    }

    #[test]
    fn test_walker_raan_spacing_star() {
        // Walker Star: 6 satellites in 3 planes -> planes at RAAN 0, 60, 120 degrees
        let walker = WalkerConstellationGenerator::new(
            6,
            3,
            0,
            7000e3,
            0.0,
            86.4, // High inclination typical for polar constellations
            0.0,
            0.0,
            0.0,
            test_epoch(),
            AngleFormat::Degrees,
            WalkerPattern::Star,
        );

        let elem0 = walker.satellite_elements(0, 0);
        let elem1 = walker.satellite_elements(1, 0);
        let elem2 = walker.satellite_elements(2, 0);

        // RAAN spacing should be 60 degrees for Star (180/3)
        assert_abs_diff_eq!(elem0[3], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(elem1[3], 60.0 * DEG2RAD, epsilon = 1e-10);
        assert_abs_diff_eq!(elem2[3], 120.0 * DEG2RAD, epsilon = 1e-10);
    }

    #[test]
    fn test_walker_ma_spacing_within_plane() {
        // 6 satellites in 2 planes -> 3 per plane, MA spacing = 120 degrees
        let walker = WalkerConstellationGenerator::new(
            6,
            2,
            0,
            7000e3,
            0.0,
            45.0,
            0.0,
            0.0,
            0.0,
            test_epoch(),
            AngleFormat::Degrees,
            WalkerPattern::Delta,
        );

        let elem0 = walker.satellite_elements(0, 0);
        let elem1 = walker.satellite_elements(0, 1);
        let elem2 = walker.satellite_elements(0, 2);

        // MA spacing should be 120 degrees within plane
        assert_abs_diff_eq!(elem0[5], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(elem1[5], 120.0 * DEG2RAD, epsilon = 1e-10);
        assert_abs_diff_eq!(elem2[5], 240.0 * DEG2RAD, epsilon = 1e-10);
    }

    #[test]
    fn test_walker_phasing() {
        // 12:3:1 constellation
        // Phase offset per plane = 1 * 360/12 = 30 degrees
        let walker = WalkerConstellationGenerator::new(
            12,
            3,
            1,
            7000e3,
            0.0,
            45.0,
            0.0,
            0.0,
            0.0,
            test_epoch(),
            AngleFormat::Degrees,
            WalkerPattern::Delta,
        );

        // First satellite in each plane
        let elem_p0_s0 = walker.satellite_elements(0, 0);
        let elem_p1_s0 = walker.satellite_elements(1, 0);
        let elem_p2_s0 = walker.satellite_elements(2, 0);

        // Plane 0: MA = 0
        // Plane 1: MA = 0 + 1*1*(360/12) = 30 degrees
        // Plane 2: MA = 0 + 2*1*(360/12) = 60 degrees
        assert_abs_diff_eq!(elem_p0_s0[5], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(elem_p1_s0[5], 30.0 * DEG2RAD, epsilon = 1e-10);
        assert_abs_diff_eq!(elem_p2_s0[5], 60.0 * DEG2RAD, epsilon = 1e-10);
    }

    #[test]
    fn test_walker_all_elements() {
        let walker = WalkerConstellationGenerator::new(
            6,
            2,
            0,
            7000e3,
            0.0,
            45.0,
            0.0,
            0.0,
            0.0,
            test_epoch(),
            AngleFormat::Degrees,
            WalkerPattern::Delta,
        );

        let all = walker.all_elements();
        assert_eq!(all.len(), 6);

        // Verify ordering: plane 0 satellites first
        assert_abs_diff_eq!(all[0][3], 0.0, epsilon = 1e-10); // plane 0, RAAN = 0
        assert_abs_diff_eq!(all[1][3], 0.0, epsilon = 1e-10); // plane 0, RAAN = 0
        assert_abs_diff_eq!(all[2][3], 0.0, epsilon = 1e-10); // plane 0, RAAN = 0
        assert_abs_diff_eq!(all[3][3], 180.0 * DEG2RAD, epsilon = 1e-10); // plane 1, RAAN = 180
        assert_abs_diff_eq!(all[4][3], 180.0 * DEG2RAD, epsilon = 1e-10); // plane 1
        assert_abs_diff_eq!(all[5][3], 180.0 * DEG2RAD, epsilon = 1e-10); // plane 1
    }

    #[test]
    fn test_walker_with_base_name() {
        let walker = WalkerConstellationGenerator::new(
            6,
            2,
            0,
            7000e3,
            0.0,
            45.0,
            0.0,
            0.0,
            0.0,
            test_epoch(),
            AngleFormat::Degrees,
            WalkerPattern::Delta,
        )
        .with_base_name("TestSat");

        assert_eq!(
            walker.satellite_name(0, 0),
            Some("TestSat-P0-S0".to_string())
        );
        assert_eq!(
            walker.satellite_name(0, 1),
            Some("TestSat-P0-S1".to_string())
        );
        assert_eq!(
            walker.satellite_name(1, 0),
            Some("TestSat-P1-S0".to_string())
        );
    }

    #[test]
    fn test_walker_satellite_id() {
        let walker = WalkerConstellationGenerator::new(
            6,
            2,
            0,
            7000e3,
            0.0,
            45.0,
            0.0,
            0.0,
            0.0,
            test_epoch(),
            AngleFormat::Degrees,
            WalkerPattern::Delta,
        );

        // 3 satellites per plane
        assert_eq!(walker.satellite_id(0, 0), 0);
        assert_eq!(walker.satellite_id(0, 1), 1);
        assert_eq!(walker.satellite_id(0, 2), 2);
        assert_eq!(walker.satellite_id(1, 0), 3);
        assert_eq!(walker.satellite_id(1, 1), 4);
        assert_eq!(walker.satellite_id(1, 2), 5);
    }

    #[test]
    fn test_walker_as_keplerian_propagators() {
        let walker = WalkerConstellationGenerator::new(
            6,
            2,
            1,
            7000e3,
            0.001,
            45.0,
            0.0,
            0.0,
            0.0,
            test_epoch(),
            AngleFormat::Degrees,
            WalkerPattern::Delta,
        )
        .with_base_name("Sat");

        let props = walker.as_keplerian_propagators(60.0);

        assert_eq!(props.len(), 6);

        // Check that propagators have correct IDs and names
        assert_eq!(props[0].get_id(), Some(0));
        assert_eq!(props[0].get_name(), Some("Sat-P0-S0"));
        assert_eq!(props[3].get_id(), Some(3));
        assert_eq!(props[3].get_name(), Some("Sat-P1-S0"));
    }

    #[test]
    fn test_walker_reference_offsets() {
        // Test with non-zero reference RAAN and MA
        let walker = WalkerConstellationGenerator::new(
            4,
            2,
            0,
            7000e3,
            0.0,
            45.0,
            0.0,
            30.0, // reference RAAN = 30 degrees
            15.0, // reference MA = 15 degrees
            test_epoch(),
            AngleFormat::Degrees,
            WalkerPattern::Delta,
        );

        let elem0 = walker.satellite_elements(0, 0);

        // First satellite should have the reference values
        assert_abs_diff_eq!(elem0[3], 30.0 * DEG2RAD, epsilon = 1e-10); // RAAN
        assert_abs_diff_eq!(elem0[5], 15.0 * DEG2RAD, epsilon = 1e-10); // MA
    }

    #[test]
    fn test_walker_radians_input() {
        let walker = WalkerConstellationGenerator::new(
            4,
            2,
            0,
            7000e3,
            0.0,
            std::f64::consts::FRAC_PI_4, // 45 degrees in radians
            0.0,
            0.0,
            0.0,
            test_epoch(),
            AngleFormat::Radians,
            WalkerPattern::Delta,
        );

        assert_abs_diff_eq!(
            walker.inclination,
            std::f64::consts::FRAC_PI_4,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_walker_star_pattern() {
        // Test Walker Star pattern (Iridium-like: 66:6:2)
        let walker = WalkerConstellationGenerator::new(
            66,
            6,
            2,
            7000e3,
            0.001,
            86.4, // High inclination typical for Iridium
            0.0,
            0.0,
            0.0,
            test_epoch(),
            AngleFormat::Degrees,
            WalkerPattern::Star,
        );

        assert_eq!(walker.pattern, WalkerPattern::Star);
        assert_eq!(walker.total_satellites, 66);
        assert_eq!(walker.num_planes, 6);
        assert_eq!(walker.satellites_per_plane(), 11);

        // For Star pattern, RAAN spacing is 180°/6 = 30°
        let elem0 = walker.satellite_elements(0, 0);
        let elem1 = walker.satellite_elements(1, 0);

        assert_abs_diff_eq!(elem0[3], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(elem1[3], 30.0 * DEG2RAD, epsilon = 1e-10);
    }
}
