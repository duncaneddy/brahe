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

use crate::orbit_dynamics::gravity::GravityModelType;

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

    /// Spacecraft mass [kg] - used by drag and SRP calculations
    ///
    /// Mass resolution priority:
    /// 1. If params vector exists → Use params[0]
    /// 2. Else if mass is Some(Value(v)) → Use v as fallback
    /// 3. Else if mass is Some(ParameterIndex) or None → Error (when drag/SRP enabled)
    ///
    /// This allows runtime mass updates via parameter vector while providing
    /// a convenient fallback for fixed-mass configurations.
    pub mass: Option<ParameterSource>,
}

impl Default for ForceModelConfiguration {
    fn default() -> Self {
        Self {
            gravity: GravityConfiguration::SphericalHarmonic {
                source: GravityModelSource::default(),
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
            mass: None, // Uses params[0] when available
        }
    }
}

impl ForceModelConfiguration {
    /// Check if this configuration requires a parameter vector
    ///
    /// Returns true if any force model component (drag, SRP) references
    /// a parameter index instead of using a fixed value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use brahe::propagators::ForceModelConfiguration;
    ///
    /// let config = ForceModelConfiguration::default();
    /// assert!(config.requires_params()); // Default uses parameter indices
    ///
    /// let gravity_only = ForceModelConfiguration::gravity_only();
    /// assert!(!gravity_only.requires_params()); // No drag/SRP
    /// ```
    pub fn requires_params(&self) -> bool {
        // Check mass configuration
        if let Some(ref mass) = self.mass
            && mass.requires_params()
        {
            return true;
        }

        // Check drag configuration
        if let Some(ref drag) = self.drag
            && (drag.area.requires_params() || drag.cd.requires_params())
        {
            return true;
        }

        // Check SRP configuration
        if let Some(ref srp) = self.srp
            && (srp.area.requires_params() || srp.cr.requires_params())
        {
            return true;
        }

        false
    }

    /// Get the maximum parameter index required by this configuration
    ///
    /// Returns the highest parameter index referenced, or None if no
    /// parameter indices are used.
    fn max_required_param_index(&self) -> Option<usize> {
        let mut max_idx: Option<usize> = None;

        // Check mass configuration
        if let Some(ParameterSource::ParameterIndex(idx)) = self.mass {
            max_idx = Some(max_idx.map_or(idx, |m| m.max(idx)));
        }

        // Check drag configuration
        if let Some(ref drag) = self.drag {
            if let ParameterSource::ParameterIndex(idx) = drag.area {
                max_idx = Some(max_idx.map_or(idx, |m| m.max(idx)));
            }
            if let ParameterSource::ParameterIndex(idx) = drag.cd {
                max_idx = Some(max_idx.map_or(idx, |m| m.max(idx)));
            }
        }

        // Check SRP configuration
        if let Some(ref srp) = self.srp {
            if let ParameterSource::ParameterIndex(idx) = srp.area {
                max_idx = Some(max_idx.map_or(idx, |m| m.max(idx)));
            }
            if let ParameterSource::ParameterIndex(idx) = srp.cr {
                max_idx = Some(max_idx.map_or(idx, |m| m.max(idx)));
            }
        }

        max_idx
    }

    /// Validate that the provided parameter vector satisfies this configuration
    ///
    /// Returns an error if the configuration references parameter indices but:
    /// - No parameter vector is provided
    /// - The parameter vector is too short
    ///
    /// # Arguments
    /// * `params` - Optional parameter vector to validate
    ///
    /// # Returns
    /// `Ok(())` if valid, `Err(BraheError)` with descriptive message if invalid
    ///
    /// # Example
    ///
    /// ```rust
    /// use brahe::propagators::ForceModelConfiguration;
    /// use nalgebra::DVector;
    ///
    /// let config = ForceModelConfiguration::default();
    ///
    /// // This will fail - default config needs params but none provided
    /// let result = config.validate_params(None);
    /// assert!(result.is_err());
    ///
    /// // This will succeed - params vector is long enough
    /// let params = DVector::from_vec(vec![1000.0, 10.0, 2.2, 10.0, 1.3]);
    /// let result = config.validate_params(Some(&params));
    /// assert!(result.is_ok());
    /// ```
    pub fn validate_params(
        &self,
        params: Option<&nalgebra::DVector<f64>>,
    ) -> Result<(), crate::utils::errors::BraheError> {
        if let Some(max_idx) = self.max_required_param_index() {
            match params {
                None => {
                    return Err(crate::utils::errors::BraheError::Error(format!(
                        "Force model configuration references parameter index {} but no parameter \
                         vector was provided. Use ForceModelConfiguration::gravity_only() for \
                         propagation without parameters, or provide a parameter vector with at \
                         least {} elements.",
                        max_idx,
                        max_idx + 1
                    )));
                }
                Some(p) if p.len() <= max_idx => {
                    return Err(crate::utils::errors::BraheError::Error(format!(
                        "Parameter vector length {} is insufficient; force model configuration \
                         requires at least {} elements (max index: {})",
                        p.len(),
                        max_idx + 1,
                        max_idx
                    )));
                }
                _ => {} // Valid
            }
        }
        Ok(())
    }

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
                source: GravityModelSource::ModelType(GravityModelType::EGM2008_360),
                degree: 120,
                order: 120,
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
                    ThirdBody::Saturn,
                    ThirdBody::Uranus,
                    ThirdBody::Neptune,
                    ThirdBody::Mercury,
                ],
            }),
            relativity: true,
            mass: None,
        }
    }

    /// Create a gravity-only configuration (for comparison with Keplerian propagation)
    pub fn gravity_only() -> Self {
        Self {
            gravity: GravityConfiguration::PointMass,
            drag: None,
            srp: None,
            third_body: Some(ThirdBodyConfiguration {
                ephemeris_source: EphemerisSource::DE440s,
                bodies: vec![ThirdBody::Sun, ThirdBody::Moon],
            }),
            relativity: true,
            mass: None,
        }
    }

    /// Create a configuration suitable for LEO satellites
    ///
    /// Includes drag and higher-order gravity, but omits SRP and third-body
    /// perturbations which are less significant in LEO.
    pub fn leo_default() -> Self {
        Self {
            gravity: GravityConfiguration::SphericalHarmonic {
                source: GravityModelSource::ModelType(GravityModelType::EGM2008_360),
                degree: 30,
                order: 30,
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
                bodies: vec![ThirdBody::Sun, ThirdBody::Moon],
            }),
            relativity: false,
            mass: None,
        }
    }

    /// Create a configuration suitable for GEO satellites
    ///
    /// Includes SRP and third-body perturbations, which are dominant in GEO.
    /// Omits drag which is negligible at GEO altitudes.
    pub fn geo_default() -> Self {
        Self {
            gravity: GravityConfiguration::SphericalHarmonic {
                source: GravityModelSource::ModelType(GravityModelType::EGM2008_360),
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
                ephemeris_source: EphemerisSource::DE440s,
                bodies: vec![ThirdBody::Sun, ThirdBody::Moon],
            }),
            relativity: false,
            mass: None,
        }
    }

    /// Get the mass value from configuration and/or parameters
    ///
    /// Resolution priority:
    /// 1. If params vector exists and mass is ParameterIndex → Use params[index]
    /// 2. If params vector exists and mass is None → Use params[0] (backwards compatible)
    /// 3. If params vector exists and mass is Value → Use params[0] (params takes priority)
    /// 4. If no params and mass is Value → Use the value
    /// 5. If no params and mass is ParameterIndex or None → Error
    ///
    /// # Arguments
    /// * `params` - Optional parameter vector
    ///
    /// # Returns
    /// The mass value in kg, or an error if mass cannot be determined
    pub fn get_mass(
        &self,
        params: Option<&nalgebra::DVector<f64>>,
    ) -> Result<f64, crate::utils::errors::BraheError> {
        // Helper to check if params are actually usable (Some and non-empty)
        let usable_params = params.filter(|p| !p.is_empty());

        match (usable_params, &self.mass) {
            // Params exist and are usable: use param vector
            (Some(p), Some(ParameterSource::ParameterIndex(idx))) => {
                if *idx >= p.len() {
                    return Err(crate::utils::errors::BraheError::Error(format!(
                        "Mass parameter index {} exceeds parameter vector length {}",
                        idx,
                        p.len()
                    )));
                }
                Ok(p[*idx])
            }
            (Some(p), Some(ParameterSource::Value(_))) => {
                // Params take priority over config value - use params[0]
                Ok(p[0])
            }
            (Some(p), None) => {
                // Backwards compatible: use params[0]
                Ok(p[0])
            }
            // No usable params: use config value if it's a fixed value
            (None, Some(ParameterSource::Value(v))) => Ok(*v),
            // No usable params and mass requires params: error
            (None, Some(ParameterSource::ParameterIndex(idx))) => {
                Err(crate::utils::errors::BraheError::Error(format!(
                    "Mass is configured as ParameterIndex({}) but no parameter vector was provided",
                    idx
                )))
            }
            (None, None) => Err(crate::utils::errors::BraheError::Error(
                "No mass specified: either provide a parameter vector or set \
                 ForceModelConfiguration.mass to ParameterSource::Value(mass_kg)"
                    .to_string(),
            )),
        }
    }
}

// =============================================================================
// Gravity Configuration
// =============================================================================

/// Source for the gravity model coefficients
///
/// Specifies where the gravity model should be loaded from. This allows users to either
/// share a single global gravity model across multiple propagators (memory efficient) or
/// load a specific model for each propagator (explicit and independent).
///
/// # Examples
///
/// ```rust
/// use brahe::propagators::GravityModelSource;
/// use brahe::gravity::GravityModelType;
///
/// // Use the global gravity model (requires set_global_gravity_model() first)
/// let source = GravityModelSource::Global;
///
/// // Load a specific model at propagator construction
/// let source = GravityModelSource::ModelType(GravityModelType::JGM3);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GravityModelSource {
    /// Use the global gravity model
    ///
    /// The gravity model must be set via `set_global_gravity_model()` before
    /// propagation. This is memory efficient when multiple propagators share
    /// the same gravity model.
    Global,

    /// Load a specific gravity model type
    ///
    /// The model is loaded at propagator construction time and stored internally.
    /// Each propagator has its own copy of the model coefficients.
    ModelType(GravityModelType),
}

impl Default for GravityModelSource {
    fn default() -> Self {
        GravityModelSource::ModelType(GravityModelType::EGM2008_360)
    }
}

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
        /// Source of the gravity model coefficients
        source: GravityModelSource,
        /// Maximum degree (n) of expansion
        degree: usize,
        /// Maximum order (m) of expansion
        order: usize,
    },
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
                assert_eq!(degree, 120);
                assert_eq!(order, 120);
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

        // Check mass is None (uses params[0] by default)
        assert!(config.mass.is_none());
    }

    #[test]
    fn test_gravity_only_configuration() {
        let config = ForceModelConfiguration::gravity_only();

        assert!(matches!(config.gravity, GravityConfiguration::PointMass));
        assert!(config.drag.is_none());
        assert!(config.srp.is_none());
        // gravity_only includes Sun/Moon third-body and relativity for comparison purposes
        assert!(config.third_body.is_some());
        assert!(config.relativity);
        assert!(config.mass.is_none());
    }

    #[test]
    fn test_leo_configuration() {
        let config = ForceModelConfiguration::leo_default();

        // LEO should have drag (dominant perturbation)
        assert!(config.drag.is_some());

        // LEO also has SRP for completeness
        assert!(config.srp.is_some());

        // LEO has Sun/Moon third-body
        assert!(config.third_body.is_some());

        // Check mass is None (uses params[0] by default)
        assert!(config.mass.is_none());
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

    // =========================================================================
    // Tests for get_mass()
    // =========================================================================

    #[test]
    fn test_get_mass_with_params_and_none_config() {
        // When mass config is None, params[0] should be used
        let config = ForceModelConfiguration {
            mass: None,
            ..ForceModelConfiguration::gravity_only()
        };
        let params = nalgebra::DVector::from_vec(vec![500.0, 10.0, 2.2, 10.0, 1.3]);

        let mass = config.get_mass(Some(&params)).unwrap();
        assert_eq!(mass, 500.0);
    }

    #[test]
    fn test_get_mass_with_params_and_value_config() {
        // When params exist, params[0] takes priority over config value
        let config = ForceModelConfiguration {
            mass: Some(ParameterSource::Value(999.0)),
            ..ForceModelConfiguration::gravity_only()
        };
        let params = nalgebra::DVector::from_vec(vec![500.0, 10.0, 2.2, 10.0, 1.3]);

        let mass = config.get_mass(Some(&params)).unwrap();
        assert_eq!(mass, 500.0); // params[0] takes priority
    }

    #[test]
    fn test_get_mass_with_params_and_param_index_config() {
        // When mass is ParameterIndex, use that index from params
        let config = ForceModelConfiguration {
            mass: Some(ParameterSource::ParameterIndex(3)),
            ..ForceModelConfiguration::gravity_only()
        };
        let params = nalgebra::DVector::from_vec(vec![500.0, 10.0, 2.2, 750.0, 1.3]);

        let mass = config.get_mass(Some(&params)).unwrap();
        assert_eq!(mass, 750.0); // params[3]
    }

    #[test]
    fn test_get_mass_without_params_and_value_config() {
        // When no params and mass is Value, use the value
        let config = ForceModelConfiguration {
            mass: Some(ParameterSource::Value(1500.0)),
            ..ForceModelConfiguration::gravity_only()
        };

        let mass = config.get_mass(None).unwrap();
        assert_eq!(mass, 1500.0);
    }

    #[test]
    fn test_get_mass_without_params_and_none_config_errors() {
        // When no params and mass is None, should error
        let config = ForceModelConfiguration {
            mass: None,
            ..ForceModelConfiguration::gravity_only()
        };

        let result = config.get_mass(None);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_mass_without_params_and_param_index_config_errors() {
        // When no params and mass is ParameterIndex, should error
        let config = ForceModelConfiguration {
            mass: Some(ParameterSource::ParameterIndex(0)),
            ..ForceModelConfiguration::gravity_only()
        };

        let result = config.get_mass(None);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_mass_param_index_out_of_bounds_errors() {
        // When param index exceeds params length, should error
        let config = ForceModelConfiguration {
            mass: Some(ParameterSource::ParameterIndex(10)),
            ..ForceModelConfiguration::gravity_only()
        };
        let params = nalgebra::DVector::from_vec(vec![500.0, 10.0, 2.2]);

        let result = config.get_mass(Some(&params));
        assert!(result.is_err());
    }

    #[test]
    fn test_requires_params_with_mass_param_index() {
        // When mass uses ParameterIndex, requires_params should return true
        let config = ForceModelConfiguration {
            mass: Some(ParameterSource::ParameterIndex(5)),
            drag: None,
            srp: None,
            ..ForceModelConfiguration::gravity_only()
        };

        assert!(config.requires_params());
    }

    #[test]
    fn test_requires_params_with_mass_value() {
        // When mass uses Value, requires_params should return false (if no other params needed)
        let config = ForceModelConfiguration {
            mass: Some(ParameterSource::Value(1000.0)),
            drag: None,
            srp: None,
            ..ForceModelConfiguration::gravity_only()
        };

        assert!(!config.requires_params());
    }
}
