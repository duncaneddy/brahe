/*!
 * Force model configuration for numerical orbit propagation
 *
 * This module provides configuration structures for defining the force models
 * used in high-fidelity numerical orbit propagation. Force models include:
 * - Gravity (point mass or spherical harmonic)
 * - Atmospheric drag (with multiple atmospheric models)
 * - Solar radiation pressure (with eclipse modeling)
 * - Third-body perturbations (Sun, Moon, planets)
 * - Relativistic corrections
 */

use serde::{Deserialize, Serialize};

// =============================================================================
// Parameter Source Configuration
// =============================================================================

/// Source for a parameter value
///
/// Allows specifying whether a parameter comes from a fixed value or from
/// an index in the parameter vector. This provides maximum flexibility:
/// - Use `Value(x)` for fixed parameters that don't change
/// - Use `ParameterIndex(i)` for parameters that may vary or be estimated
///
/// # Examples
///
/// ```rust
/// use brahe::propagators::ParameterSource;
///
/// // Fixed drag coefficient
/// let cd = ParameterSource::Value(2.2);
///
/// // Variable mass from parameter vector
/// let mass = ParameterSource::ParameterIndex(0);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ParameterSource {
    /// Use a fixed constant value
    Value(f64),
    /// Use value from parameter vector at given index
    ParameterIndex(usize),
}

impl ParameterSource {
    /// Get the value, either from the fixed value or parameter vector
    ///
    /// # Arguments
    /// * `params` - Optional parameter vector (required if using ParameterIndex)
    ///
    /// # Returns
    /// The parameter value, or default if parameter vector is missing
    pub fn get_value(&self, params: Option<&nalgebra::DVector<f64>>, default: f64) -> f64 {
        match self {
            ParameterSource::Value(v) => *v,
            ParameterSource::ParameterIndex(idx) => params.map(|p| p[*idx]).unwrap_or(default),
        }
    }

    /// Check if this parameter source requires a parameter vector
    pub fn requires_params(&self) -> bool {
        matches!(self, ParameterSource::ParameterIndex(_))
    }
}

// =============================================================================
// Main Force Model Configuration
// =============================================================================

/// Complete force model configuration for numerical orbit propagation
///
/// Defines all perturbation forces to be included in orbital dynamics.
/// Each force can be independently enabled/disabled and configured.
///
/// # Default Configuration
///
/// The default configuration provides a reasonable balance between accuracy
/// and computational efficiency:
/// - 20×20 EGM2008 gravity
/// - Harris-Priester atmospheric drag
/// - Solar radiation pressure with conical eclipse model
/// - Sun and Moon third-body perturbations (low-precision)
/// - Relativity disabled
///
/// # Example
///
/// ```rust
/// use brahe::propagators::ForceModelConfiguration;
/// use brahe::GravityConfiguration;
///
/// // Use default configuration
/// let config = ForceModelConfiguration::default();
///
/// // Or customize
/// let config = ForceModelConfiguration {
///     gravity: GravityConfiguration::PointMass,
///     drag: None,  // Disable drag
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceModelConfiguration {
    /// Gravity model configuration
    pub gravity: GravityConfiguration,

    /// Atmospheric drag configuration (None = disabled)
    pub drag: Option<DragConfiguration>,

    /// Solar radiation pressure configuration (None = disabled)
    pub srp: Option<SolarRadiationPressureConfiguration>,

    /// Third-body perturbations configuration (None = disabled)
    pub third_body: Option<ThirdBodyConfiguration>,

    /// Enable general relativistic corrections
    pub relativity: bool,
}

impl Default for ForceModelConfiguration {
    fn default() -> Self {
        Self {
            gravity: GravityConfiguration::SphericalHarmonic {
                model: GravityModelType::EGM2008_360,
                degree: 20,
                order: 20,
            },
            drag: Some(DragConfiguration {
                model: AtmosphericModel::HarrisPriester,
                area: ParameterSource::ParameterIndex(1),
                cd: ParameterSource::ParameterIndex(2),
            }),
            srp: Some(SolarRadiationPressureConfiguration {
                area: ParameterSource::ParameterIndex(3),
                cr: ParameterSource::ParameterIndex(4),
                eclipse_model: EclipseModel::Conical,
            }),
            third_body: Some(ThirdBodyConfiguration {
                ephemeris_source: EphemerisSource::LowPrecision,
                bodies: vec![ThirdBody::Sun, ThirdBody::Moon],
            }),
            relativity: false,
        }
    }
}

impl ForceModelConfiguration {
    /// Create a high-fidelity force model configuration
    ///
    /// Uses:
    /// - 70×70 EGM2008 gravity
    /// - NRLMSISE-00 atmospheric model
    /// - SRP with conical eclipse
    /// - Sun, Moon, and all planets (DE440s ephemerides)
    /// - Relativistic corrections enabled
    pub fn high_fidelity() -> Self {
        Self {
            gravity: GravityConfiguration::SphericalHarmonic {
                model: GravityModelType::EGM2008_360,
                degree: 70,
                order: 70,
            },
            drag: Some(DragConfiguration {
                model: AtmosphericModel::NRLMSISE00,
                area: ParameterSource::ParameterIndex(1),
                cd: ParameterSource::ParameterIndex(2),
            }),
            srp: Some(SolarRadiationPressureConfiguration {
                area: ParameterSource::ParameterIndex(3),
                cr: ParameterSource::ParameterIndex(4),
                eclipse_model: EclipseModel::Conical,
            }),
            third_body: Some(ThirdBodyConfiguration {
                ephemeris_source: EphemerisSource::DE440s,
                bodies: vec![
                    ThirdBody::Sun,
                    ThirdBody::Moon,
                    ThirdBody::Venus,
                    ThirdBody::Mars,
                    ThirdBody::Jupiter,
                ],
            }),
            relativity: true,
        }
    }

    /// Create a gravity-only configuration (for comparison with Keplerian propagation)
    pub fn gravity_only() -> Self {
        Self {
            gravity: GravityConfiguration::PointMass,
            drag: None,
            srp: None,
            third_body: None,
            relativity: false,
        }
    }

    /// Create a configuration suitable for LEO satellites
    ///
    /// Includes drag and higher-order gravity, but omits SRP and third-body
    /// perturbations which are less significant in LEO.
    pub fn leo_default() -> Self {
        Self {
            gravity: GravityConfiguration::SphericalHarmonic {
                model: GravityModelType::EGM2008_360,
                degree: 30,
                order: 30,
            },
            drag: Some(DragConfiguration {
                model: AtmosphericModel::HarrisPriester,
                area: ParameterSource::ParameterIndex(1),
                cd: ParameterSource::ParameterIndex(2),
            }),
            srp: None,
            third_body: None,
            relativity: false,
        }
    }

    /// Create a configuration suitable for GEO satellites
    ///
    /// Includes SRP and third-body perturbations, which are dominant in GEO.
    /// Omits drag which is negligible at GEO altitudes.
    pub fn geo_default() -> Self {
        Self {
            gravity: GravityConfiguration::SphericalHarmonic {
                model: GravityModelType::EGM2008_360,
                degree: 8,
                order: 8,
            },
            drag: None,
            srp: Some(SolarRadiationPressureConfiguration {
                area: ParameterSource::ParameterIndex(3),
                cr: ParameterSource::ParameterIndex(4),
                eclipse_model: EclipseModel::Conical,
            }),
            third_body: Some(ThirdBodyConfiguration {
                ephemeris_source: EphemerisSource::LowPrecision,
                bodies: vec![ThirdBody::Sun, ThirdBody::Moon],
            }),
            relativity: false,
        }
    }
}

// =============================================================================
// Gravity Configuration
// =============================================================================

/// Gravity model configuration
///
/// Specifies the gravity model to use for computing gravitational acceleration.
/// Can be either simple two-body point mass or high-fidelity spherical harmonic expansion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GravityConfiguration {
    /// Simple two-body point mass gravity
    ///
    /// Uses only the central term (J0). Fast but inaccurate for orbit propagation.
    /// Suitable only for preliminary analysis or comparison with Keplerian propagation.
    PointMass,

    /// Spherical harmonic gravity model
    ///
    /// Includes higher-order terms in the gravity field expansion.
    /// More accurate but computationally expensive.
    SphericalHarmonic {
        /// Gravity model to use
        model: GravityModelType,
        /// Maximum degree (n) of expansion
        degree: usize,
        /// Maximum order (m) of expansion
        order: usize,
    },
}

/// Type of spherical harmonic gravity model
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GravityModelType {
    /// EGM2008 model (360×360)
    ///
    /// Earth Gravitational Model 2008 - state-of-the-art model with 360×360 coefficients.
    /// Most accurate for Earth orbit propagation.
    EGM2008_360,

    /// GGM05S model
    ///
    /// GRACE Gravity Model 05S - alternative high-accuracy model
    GGM05S,

    /// JGM3 model
    ///
    /// Joint Gravity Model 3 - older but still widely used
    JGM3,

    /// Load gravity model from file
    ///
    /// Allows using custom gravity models. File must be in standard GFC format.
    FromFile(String),
}

// =============================================================================
// Drag Configuration
// =============================================================================

/// Atmospheric drag configuration
///
/// Defines the atmospheric density model and spacecraft drag parameters.
///
/// # Parameter Specification
///
/// Parameters can be specified as either:
/// - Fixed values: `ParameterSource::Value(x)`
/// - Parameter vector indices: `ParameterSource::ParameterIndex(i)`
///
/// This allows mixing fixed and variable parameters as needed.
///
/// # Examples
///
/// ```rust
/// use brahe::propagators::{DragConfiguration, AtmosphericModel, ParameterSource};
///
/// // Fixed Cd, variable area
/// let drag = DragConfiguration {
///     model: AtmosphericModel::HarrisPriester,
///     area: ParameterSource::ParameterIndex(1),  // From params[1]
///     cd: ParameterSource::Value(2.2),            // Fixed
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DragConfiguration {
    /// Atmospheric density model to use
    pub model: AtmosphericModel,

    /// Drag cross-sectional area [m²]
    pub area: ParameterSource,

    /// Drag coefficient (dimensionless, typically 2.0-2.5)
    pub cd: ParameterSource,
}

/// Atmospheric density model
///
/// Specifies which model to use for computing atmospheric density.
/// Different models trade off accuracy vs computational cost.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AtmosphericModel {
    /// Harris-Priester atmospheric model
    ///
    /// Modified Harris-Priester model accounting for diurnal density variations.
    /// Valid for altitudes 100-1000 km. Fast and reasonably accurate.
    /// Does not require space weather data.
    HarrisPriester,

    /// NRLMSISE-00 empirical atmospheric model
    ///
    /// Naval Research Laboratory Mass Spectrometer and Incoherent Scatter Radar
    /// Exosphere model. High-fidelity model requiring space weather data (F10.7, Ap).
    /// Valid from ground to thermospheric heights.
    NRLMSISE00,

    /// Simple exponential atmosphere model
    ///
    /// Density varies exponentially with altitude: ρ(h) = ρ₀ exp(-h/H)
    /// Very fast but inaccurate. Suitable only for rough estimates.
    Exponential {
        /// Scale height [m]
        scale_height: f64,
        /// Reference density at reference altitude [kg/m³]
        rho0: f64,
        /// Reference altitude [m]
        h0: f64,
    },
}

// =============================================================================
// Solar Radiation Pressure Configuration
// =============================================================================

/// Solar radiation pressure configuration
///
/// Defines eclipse modeling and spacecraft SRP parameters.
///
/// # Parameter Specification
///
/// Parameters can be specified as either:
/// - Fixed values: `ParameterSource::Value(x)`
/// - Parameter vector indices: `ParameterSource::ParameterIndex(i)`
///
/// # Examples
///
/// ```rust
/// use brahe::propagators::{SolarRadiationPressureConfiguration, EclipseModel, ParameterSource};
///
/// // Variable area and Cr
/// let srp = SolarRadiationPressureConfiguration {
///     area: ParameterSource::ParameterIndex(3),
///     cr: ParameterSource::ParameterIndex(4),
///     eclipse_model: EclipseModel::Conical,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SolarRadiationPressureConfiguration {
    /// SRP cross-sectional area [m²]
    pub area: ParameterSource,

    /// Coefficient of reflectivity (dimensionless, typically 1.0-2.0)
    pub cr: ParameterSource,

    /// Eclipse model for shadow effects
    pub eclipse_model: EclipseModel,
}

/// Eclipse modeling for solar radiation pressure
///
/// Defines how to model Earth's shadow on the spacecraft.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EclipseModel {
    /// No eclipse modeling
    ///
    /// SRP is always fully applied. Fast but inaccurate during eclipse periods.
    None,

    /// Cylindrical shadow model
    ///
    /// Models Earth's shadow as a cylinder. Simple and fast, but ignores penumbra.
    /// Spacecraft is either fully illuminated or fully shadowed.
    Cylindrical,

    /// Conical shadow model
    ///
    /// Models Earth's shadow as a cone, accounting for partial shadowing (penumbra).
    /// More accurate than cylindrical, slightly slower.
    Conical,
}

// =============================================================================
// Third Body Configuration
// =============================================================================

/// Third-body perturbations configuration
///
/// Defines which celestial bodies to include as third-body perturbers
/// and which ephemeris source to use for their positions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ThirdBodyConfiguration {
    /// Source for celestial body ephemerides
    pub ephemeris_source: EphemerisSource,

    /// List of bodies to include as perturbers
    pub bodies: Vec<ThirdBody>,
}

/// Source for celestial body ephemerides
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EphemerisSource {
    /// Low-precision analytical ephemerides
    ///
    /// Uses simplified analytical models for Sun and Moon positions.
    /// Fast but less accurate (~km level errors).
    /// Only Sun and Moon are available.
    LowPrecision,

    /// High-precision JPL DE440s ephemerides
    ///
    /// Uses JPL Development Ephemeris 440 (small bodies version).
    /// High accuracy (~m level) but requires ephemeris file and slower evaluation.
    /// All planets available.
    DE440s,
}

/// Third-body perturber
///
/// Celestial bodies that can act as gravitational perturbers on the spacecraft.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ThirdBody {
    /// Sun
    Sun,
    /// Moon
    Moon,
    /// Mercury
    Mercury,
    /// Venus
    Venus,
    /// Mars
    Mars,
    /// Jupiter
    Jupiter,
    /// Saturn
    Saturn,
    /// Uranus
    Uranus,
    /// Neptune
    Neptune,
}

// =============================================================================
// Default Parameter Vector Layout
// =============================================================================

/// Default parameter vector layout for force model configuration
///
/// Defines standard indices for spacecraft parameters used in force models.
/// Users can customize these indices in the configuration structs.
///
/// # Standard Layout
///
/// Index | Parameter | Units | Typical Value
/// ------|-----------|-------|---------------
/// 0     | mass      | kg    | 1000.0
/// 1     | drag_area | m²    | 10.0
/// 2     | Cd        | -     | 2.2
/// 3     | srp_area  | m²    | 10.0
/// 4     | Cr        | -     | 1.3
///
/// Additional parameters can be appended for custom dynamics.
pub struct DefaultParameterLayout;

impl DefaultParameterLayout {
    /// Mass parameter index
    pub const MASS: usize = 0;
    /// Drag cross-sectional area parameter index
    pub const DRAG_AREA: usize = 1;
    /// Drag coefficient parameter index
    pub const CD: usize = 2;
    /// SRP cross-sectional area parameter index
    pub const SRP_AREA: usize = 3;
    /// Coefficient of reflectivity parameter index
    pub const CR: usize = 4;

    /// Number of standard parameters
    pub const NUM_STANDARD_PARAMS: usize = 5;

    /// Create a default parameter vector with typical values
    ///
    /// # Arguments
    /// * `mass` - Spacecraft mass [kg]
    /// * `drag_area` - Drag cross-sectional area [m²]
    /// * `cd` - Drag coefficient (dimensionless)
    /// * `srp_area` - SRP cross-sectional area [m²]
    /// * `cr` - Coefficient of reflectivity (dimensionless)
    ///
    /// # Returns
    /// Parameter vector: [mass, drag_area, Cd, srp_area, Cr]
    pub fn create(mass: f64, drag_area: f64, cd: f64, srp_area: f64, cr: f64) -> Vec<f64> {
        vec![mass, drag_area, cd, srp_area, cr]
    }

    /// Create a default parameter vector with standard values
    ///
    /// Uses:
    /// - mass = 1000.0 kg
    /// - drag_area = 10.0 m²
    /// - Cd = 2.2
    /// - srp_area = 10.0 m²
    /// - Cr = 1.3
    pub fn default_values() -> Vec<f64> {
        vec![1000.0, 10.0, 2.2, 10.0, 1.3]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_configuration() {
        let config = ForceModelConfiguration::default();

        // Check gravity
        match config.gravity {
            GravityConfiguration::SphericalHarmonic { degree, order, .. } => {
                assert_eq!(degree, 20);
                assert_eq!(order, 20);
            }
            _ => panic!("Expected spherical harmonic gravity"),
        }

        // Check drag is enabled
        assert!(config.drag.is_some());

        // Check SRP is enabled
        assert!(config.srp.is_some());

        // Check third body is enabled
        assert!(config.third_body.is_some());

        // Check relativity is disabled
        assert!(!config.relativity);
    }

    #[test]
    fn test_high_fidelity_configuration() {
        let config = ForceModelConfiguration::high_fidelity();

        // Check high-degree gravity
        match config.gravity {
            GravityConfiguration::SphericalHarmonic { degree, order, .. } => {
                assert_eq!(degree, 70);
                assert_eq!(order, 70);
            }
            _ => panic!("Expected spherical harmonic gravity"),
        }

        // Check NRLMSISE-00
        let drag = config.drag.unwrap();
        assert!(matches!(drag.model, AtmosphericModel::NRLMSISE00));

        // Check DE440s
        let tb = config.third_body.unwrap();
        assert!(matches!(tb.ephemeris_source, EphemerisSource::DE440s));

        // Check relativity enabled
        assert!(config.relativity);
    }

    #[test]
    fn test_gravity_only_configuration() {
        let config = ForceModelConfiguration::gravity_only();

        assert!(matches!(config.gravity, GravityConfiguration::PointMass));
        assert!(config.drag.is_none());
        assert!(config.srp.is_none());
        assert!(config.third_body.is_none());
        assert!(!config.relativity);
    }

    #[test]
    fn test_leo_configuration() {
        let config = ForceModelConfiguration::leo_default();

        // LEO should have drag
        assert!(config.drag.is_some());

        // LEO should not have SRP (less significant)
        assert!(config.srp.is_none());

        // LEO should not have third-body (less significant)
        assert!(config.third_body.is_none());
    }

    #[test]
    fn test_geo_configuration() {
        let config = ForceModelConfiguration::geo_default();

        // GEO should not have drag (negligible at GEO altitude)
        assert!(config.drag.is_none());

        // GEO should have SRP (dominant perturbation)
        assert!(config.srp.is_some());

        // GEO should have third-body (dominant perturbation)
        assert!(config.third_body.is_some());
    }

    #[test]
    fn test_default_parameter_layout() {
        let params = DefaultParameterLayout::default_values();

        assert_eq!(params.len(), DefaultParameterLayout::NUM_STANDARD_PARAMS);
        assert_eq!(params[DefaultParameterLayout::MASS], 1000.0);
        assert_eq!(params[DefaultParameterLayout::DRAG_AREA], 10.0);
        assert_eq!(params[DefaultParameterLayout::CD], 2.2);
        assert_eq!(params[DefaultParameterLayout::SRP_AREA], 10.0);
        assert_eq!(params[DefaultParameterLayout::CR], 1.3);
    }

    #[test]
    fn test_custom_parameter_layout() {
        let params = DefaultParameterLayout::create(500.0, 5.0, 2.0, 8.0, 1.5);

        assert_eq!(params[DefaultParameterLayout::MASS], 500.0);
        assert_eq!(params[DefaultParameterLayout::DRAG_AREA], 5.0);
        assert_eq!(params[DefaultParameterLayout::CD], 2.0);
        assert_eq!(params[DefaultParameterLayout::SRP_AREA], 8.0);
        assert_eq!(params[DefaultParameterLayout::CR], 1.5);
    }

    #[test]
    fn test_serialization() {
        let config = ForceModelConfiguration::default();

        // Serialize to JSON
        let json = serde_json::to_string(&config).unwrap();

        // Deserialize back
        let deserialized: ForceModelConfiguration = serde_json::from_str(&json).unwrap();

        // Check equality (at least for some fields)
        assert!(matches!(
            deserialized.gravity,
            GravityConfiguration::SphericalHarmonic { .. }
        ));
        assert!(deserialized.drag.is_some());
    }
}
