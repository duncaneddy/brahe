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

use std::fmt::Display;

use serde::{Deserialize, Serialize};

use crate::constants::{
    GM_DEIMOS, GM_EARTH, GM_JUPITER, GM_JUPITER_SYSTEM, GM_MARS, GM_MARS_SYSTEM, GM_MERCURY,
    GM_MOON, GM_NEPTUNE, GM_NEPTUNE_SYSTEM, GM_PHOBOS, GM_SATURN, GM_SATURN_SYSTEM, GM_SUN,
    GM_URANUS, GM_URANUS_SYSTEM, GM_VENUS, R_EARTH, R_MARS, R_MOON,
};
use crate::datasets::icgem::ICGEMBody;
use crate::orbit_dynamics::ParallelMode;
use crate::orbit_dynamics::gravity::{GravityModel, GravityModelTideSystem, GravityModelType};
use crate::orbit_dynamics::tides::SolidTideConfig;
use crate::propagators::central_body::CentralBody;
use crate::spice::SPICEKernel;
use crate::utils::BraheError;

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
    /// The parameter value.
    ///
    /// # Errors
    /// Returns [`BraheError::OutOfBoundsError`] if this source is a
    /// [`ParameterSource::ParameterIndex`] but the parameter vector is missing
    /// or the index is out of bounds.
    pub fn get_value(&self, params: Option<&nalgebra::DVector<f64>>) -> Result<f64, BraheError> {
        match self {
            ParameterSource::Value(v) => Ok(*v),
            ParameterSource::ParameterIndex(idx) => match params {
                Some(p) if *idx < p.len() => Ok(p[*idx]),
                Some(p) => Err(BraheError::OutOfBoundsError(format!(
                    "parameter index {} out of bounds for parameter vector of length {}",
                    idx,
                    p.len()
                ))),
                None => Err(BraheError::OutOfBoundsError(format!(
                    "parameter vector missing but parameter index {} was requested",
                    idx
                ))),
            },
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
/// use brahe::propagators::ForceModelConfig;
/// use brahe::GravityConfiguration;
///
/// // Use default configuration
/// let config = ForceModelConfig::default();
///
/// // Or customize
/// let config = ForceModelConfig {
///     gravity: GravityConfiguration::PointMass,
///     drag: None,  // Disable drag
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceModelConfig {
    /// Central body the orbit is propagated relative to.
    ///
    /// Defaults to `CentralBody::Earth`. Determines which options are valid
    /// elsewhere in this configuration — see [`ForceModelConfig::validate`].
    #[serde(default)]
    pub central_body: CentralBody,

    /// Gravity model configuration
    pub gravity: GravityConfiguration,

    /// Atmospheric drag configuration (None = disabled)
    pub drag: Option<DragConfiguration>,

    /// Solar radiation pressure configuration (None = disabled)
    pub srp: Option<SolarRadiationPressureConfiguration>,

    /// Third-body perturbations, one entry per perturbing body (None = disabled).
    ///
    /// Deserializes from a single entry, a list of entries, or bare
    /// [`ThirdBody`] values (which take DE440s ephemerides and point-mass
    /// gravity). Configurations serialized before the per-body flattening
    /// (a `third_body` object with `ephemeris_source` and `bodies`) are
    /// migrated on load, with the pre-split planet names mapped to the
    /// `*Barycenter` variants they then denoted.
    #[serde(default, deserialize_with = "deserialize_third_bodies")]
    pub third_body: Option<Vec<ThirdBodyConfiguration>>,

    /// Enable general relativistic corrections
    pub relativity: bool,

    /// Spacecraft mass [kg] - required when drag or SRP is enabled
    ///
    /// Use `ParameterSource::Value(mass_kg)` for fixed mass or
    /// `ParameterSource::ParameterIndex(idx)` to read from parameter vector.
    pub mass: Option<ParameterSource>,

    /// Inertial-to-body-fixed frame transformation used for body-fixed force terms
    /// (spherical-harmonic and zonal gravity, NRLMSISE-00 density, drag).
    ///
    /// Defaults to `FrameTransformationModel::FullEarthRotation`, which matches the
    /// IAU 2006/2000A rotation used elsewhere in brahe. Set to `EarthRotationOnly`
    /// to trade ~0.07° pole-tilt accuracy for ~1.5x faster ECI↔ECEF rotations.
    #[serde(default)]
    pub frame_transform: FrameTransformationModel,

    /// Tidal corrections to the gravity field. `None` (default) disables tides.
    #[serde(default)]
    pub tides: Option<TidesConfiguration>,
}

impl Default for ForceModelConfig {
    fn default() -> Self {
        Self {
            central_body: CentralBody::default(),
            gravity: GravityConfiguration::SphericalHarmonic {
                source: GravityModelSource::default(),
                degree: 20,
                order: 20,
                parallel: ParallelMode::Auto,
            },
            drag: Some(DragConfiguration {
                model: AtmosphericModel::NRLMSISE00,
                area: ParameterSource::ParameterIndex(1),
                cd: ParameterSource::ParameterIndex(2),
                body: None,
            }),
            srp: Some(SolarRadiationPressureConfiguration {
                area: ParameterSource::ParameterIndex(3),
                cr: ParameterSource::ParameterIndex(4),
                eclipse_model: EclipseModel::Conical,
                occulting_bodies: vec![OccultingBody::Earth],
            }),
            third_body: Some(vec![ThirdBody::Sun.into(), ThirdBody::Moon.into()]),
            relativity: false,
            mass: Some(ParameterSource::ParameterIndex(0)),
            frame_transform: FrameTransformationModel::default(),
            tides: None,
        }
    }
}

impl ForceModelConfig {
    /// Check if this configuration requires a parameter vector
    ///
    /// Returns true if any force model component (drag, SRP) references
    /// a parameter index instead of using a fixed value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use brahe::propagators::ForceModelConfig;
    ///
    /// let config = ForceModelConfig::default();
    /// assert!(config.requires_params()); // Default uses parameter indices
    ///
    /// let earth_gravity = ForceModelConfig::earth_gravity();
    /// assert!(!earth_gravity.requires_params()); // No drag/SRP
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
    /// use brahe::propagators::{CentralBody, ForceModelConfig};
    /// use brahe::propagators::force_model_config::{
    ///     DragConfiguration, AtmosphericModel, ParameterSource,
    ///     GravityConfiguration,
    /// };
    /// use nalgebra::DVector;
    ///
    /// // Create a config with Harris-Priester drag (no space weather needed)
    /// let config = ForceModelConfig {
    ///     central_body: CentralBody::Earth,
    ///     gravity: GravityConfiguration::PointMass,
    ///     drag: Some(DragConfiguration {
    ///         model: AtmosphericModel::HarrisPriester,
    ///         area: ParameterSource::ParameterIndex(1),
    ///         cd: ParameterSource::ParameterIndex(2),
    ///         body: None,
    ///     }),
    ///     srp: None,
    ///     third_body: None,
    ///     relativity: false,
    ///     mass: Some(ParameterSource::ParameterIndex(0)),
    ///     frame_transform: Default::default(),
    ///     tides: None,
    /// };
    ///
    /// // This will fail - config needs params but none provided
    /// let result = config.validate_params(None);
    /// assert!(result.is_err());
    ///
    /// // This will succeed - params vector is long enough
    /// let params = DVector::from_vec(vec![1000.0, 10.0, 2.2]);
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
                         vector was provided. Use ForceModelConfig::earth_gravity() for \
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

        // Check if NRLMSISE-00 atmospheric model is configured but space weather is not initialized
        if let Some(drag_config) = &self.drag
            && matches!(drag_config.model, AtmosphericModel::NRLMSISE00)
            && !crate::space_weather::get_global_sw_initialization()
        {
            return Err(crate::utils::errors::BraheError::Error(
                "NRLMSISE-00 atmospheric model requires space weather data. \
                    Call initialize_sw() before creating a propagator with NRLMSISE-00, \
                    initialize another space weather data provider, \
                    or use AtmosphericModel::HarrisPriester which does not require \
                    space weather data."
                    .to_string(),
            ));
        }

        Ok(())
    }

    /// Create a high-fidelity force model configuration
    ///
    /// Uses:
    /// - 120x120 EGM2008 gravity
    /// - NRLMSISE-00 atmospheric model
    /// - SRP with conical eclipse
    /// - Sun, Moon, and all planets (DE440s ephemerides)
    /// - Relativistic corrections enabled
    /// - Solid Earth tides with frequency-dependent corrections and the solid
    ///   pole tide
    /// - Ocean tides (FES2004, 30x30) with admittance and the ocean pole tide;
    ///   requires the one-time cached download of the IERS FES2004
    ///   coefficient file (see `brahe::orbit_dynamics::ocean_tides`)
    /// - Tidal Sun/Moon positions sourced from DE440s, matching the third-body
    ///   configuration so the two force terms share the ephemeris evaluation.
    ///   The tidal path resolves the DE440s kernel at propagator construction
    ///   (fail-fast; auto-downloads on a cold kernel cache), whereas the
    ///   third-body path loads it lazily on the first step
    pub fn high_fidelity() -> Self {
        Self {
            central_body: CentralBody::Earth,
            gravity: GravityConfiguration::SphericalHarmonic {
                source: GravityModelSource::ModelType(GravityModelType::EGM2008_120),
                degree: 120,
                order: 120,
                parallel: ParallelMode::Auto,
            },
            drag: Some(DragConfiguration {
                model: AtmosphericModel::NRLMSISE00,
                area: ParameterSource::ParameterIndex(1),
                cd: ParameterSource::ParameterIndex(2),
                body: None,
            }),
            srp: Some(SolarRadiationPressureConfiguration {
                area: ParameterSource::ParameterIndex(3),
                cr: ParameterSource::ParameterIndex(4),
                eclipse_model: EclipseModel::Conical,
                occulting_bodies: vec![OccultingBody::Earth],
            }),
            third_body: Some(vec![
                ThirdBody::Sun.into(),
                ThirdBody::Moon.into(),
                ThirdBody::Venus.into(),
                ThirdBody::MarsBarycenter.into(),
                ThirdBody::JupiterBarycenter.into(),
                ThirdBody::SaturnBarycenter.into(),
                ThirdBody::UranusBarycenter.into(),
                ThirdBody::NeptuneBarycenter.into(),
                ThirdBody::Mercury.into(),
            ]),
            relativity: true,
            mass: Some(ParameterSource::ParameterIndex(0)),
            frame_transform: FrameTransformationModel::default(),
            tides: Some(TidesConfiguration {
                permanent: PermanentTideConfig::Auto,
                solid: Some(SolidTideConfig {
                    frequency_dependent: true,
                    pole_tide: true,
                }),
                ocean: Some(OceanTideConfig {
                    degree: 30,
                    order: 30,
                    include_admittance: true,
                    pole_tide: true,
                }),
                // Match the third-body config's DE440s source so tidal Sun/Moon
                // positions come from the same ephemeris and share the cache.
                ephemeris_source: EphemerisSource::DE440s,
            }),
        }
    }

    /// Create a gravity-only configuration (for comparison with Keplerian propagation)
    pub fn earth_gravity() -> Self {
        Self {
            central_body: CentralBody::Earth,
            gravity: GravityConfiguration::SphericalHarmonic {
                source: GravityModelSource::ModelType(GravityModelType::EGM2008_120),
                degree: 20,
                order: 20,
                parallel: ParallelMode::Auto,
            },
            drag: None,
            srp: None,
            third_body: None,
            relativity: false,
            mass: None,
            frame_transform: FrameTransformationModel::default(),
            tides: None,
        }
    }

    /// Create a two-body (point mass) gravity configuration
    ///
    /// Uses only central body gravity with no perturbations.
    /// Produces results equivalent to Keplerian propagation.
    /// Useful for validation and comparison tests.
    pub fn two_body_gravity() -> Self {
        Self {
            central_body: CentralBody::Earth,
            gravity: GravityConfiguration::PointMass,
            drag: None,
            srp: None,
            third_body: None,
            relativity: false,
            mass: None,
            frame_transform: FrameTransformationModel::default(),
            tides: None,
        }
    }

    /// Create a gravity-only configuration (for comparison with Keplerian propagation)
    pub fn conservative_forces() -> Self {
        Self {
            central_body: CentralBody::Earth,
            gravity: GravityConfiguration::SphericalHarmonic {
                source: GravityModelSource::ModelType(GravityModelType::EGM2008_120),
                degree: 80,
                order: 80,
                parallel: ParallelMode::Auto,
            },
            drag: None,
            srp: None,
            third_body: Some(vec![ThirdBody::Sun.into(), ThirdBody::Moon.into()]),
            relativity: true,
            mass: None,
            frame_transform: FrameTransformationModel::default(),
            tides: None,
        }
    }

    /// Create a configuration suitable for LEO satellites
    ///
    /// Includes drag and higher-order gravity, but omits SRP and third-body
    /// perturbations which are less significant in LEO.
    pub fn leo_default() -> Self {
        Self {
            central_body: CentralBody::Earth,
            gravity: GravityConfiguration::SphericalHarmonic {
                source: GravityModelSource::ModelType(GravityModelType::EGM2008_120),
                degree: 30,
                order: 30,
                parallel: ParallelMode::Auto,
            },
            drag: Some(DragConfiguration {
                model: AtmosphericModel::NRLMSISE00,
                area: ParameterSource::ParameterIndex(1),
                cd: ParameterSource::ParameterIndex(2),
                body: None,
            }),
            srp: Some(SolarRadiationPressureConfiguration {
                area: ParameterSource::ParameterIndex(3),
                cr: ParameterSource::ParameterIndex(4),
                eclipse_model: EclipseModel::Conical,
                occulting_bodies: vec![OccultingBody::Earth],
            }),
            third_body: Some(vec![ThirdBody::Sun.into(), ThirdBody::Moon.into()]),
            relativity: false,
            mass: Some(ParameterSource::ParameterIndex(0)),
            frame_transform: FrameTransformationModel::default(),
            tides: None,
        }
    }

    /// Create a configuration suitable for GEO satellites
    ///
    /// Includes SRP and third-body perturbations, which are dominant in GEO.
    /// Omits drag which is negligible at GEO altitudes.
    pub fn geo_default() -> Self {
        Self {
            central_body: CentralBody::Earth,
            gravity: GravityConfiguration::SphericalHarmonic {
                source: GravityModelSource::ModelType(GravityModelType::EGM2008_120),
                degree: 8,
                order: 8,
                parallel: ParallelMode::Auto,
            },
            drag: None,
            srp: Some(SolarRadiationPressureConfiguration {
                area: ParameterSource::ParameterIndex(3),
                cr: ParameterSource::ParameterIndex(4),
                eclipse_model: EclipseModel::Conical,
                occulting_bodies: vec![OccultingBody::Earth],
            }),
            third_body: Some(vec![ThirdBody::Sun.into(), ThirdBody::Moon.into()]),
            relativity: false,
            mass: Some(ParameterSource::ParameterIndex(0)),
            frame_transform: FrameTransformationModel::default(),
            tides: None,
        }
    }
}

// =============================================================================
// Tides Configuration
// =============================================================================

/// Permanent (zero-frequency) tide handling for the static gravity field.
///
/// Controls how the loaded model's C̄20 is reconciled with the solid-tide
/// model, which (IERS §6.2.1) produces the *total* tide including the
/// permanent part — correct only against a conventional tide-free background.
///
/// **Applies only to propagator-owned models
/// ([`GravityModelSource::ModelType`]).** For [`GravityModelSource::Global`] the
/// shared model is read-only, so this setting has no effect — resolve the global
/// model's tide system once, before install, with
/// [`set_global_gravity_model_to_tide_system`](crate::gravity::set_global_gravity_model_to_tide_system).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum PermanentTideConfig {
    /// Read the model's tide_system flag and convert C̄20 to conventional
    /// tide-free (the convention the solid-tide model expects). Unknown flag
    /// => no-op + warning. Default.
    #[default]
    Auto,
    /// Force the field into the given tide system. Source = the model's flagged
    /// system; errors at construction if the flag is Unknown.
    ConvertTo(GravityModelTideSystem),
    /// Leave C̄20 untouched.
    Off,
}

/// FES2004 ocean tide configuration (IERS TN36 §6.3), plus the ocean pole
/// tide (§6.5). Requires the one-time cached download of the IERS FES2004
/// coefficient file (see `brahe::orbit_dynamics::ocean_tides`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct OceanTideConfig {
    /// Truncation degree of the ocean tide expansion (2..=100). Default 20.
    pub degree: usize,
    /// Truncation order (2..=degree). Default 20.
    pub order: usize,
    /// Complement the 18 FES2004 main waves with the ~63 secondary waves of
    /// TN36 Table 6.7 via linear admittance interpolation (Eq. 6.16).
    /// Default true.
    pub include_admittance: bool,
    /// Apply the ocean pole tide (2,1) main term (TN36 Eq. 6.24). Requires
    /// initialized global EOP data. Default false.
    pub pole_tide: bool,
}

impl Default for OceanTideConfig {
    fn default() -> Self {
        OceanTideConfig {
            degree: 20,
            order: 20,
            include_admittance: true,
            pole_tide: false,
        }
    }
}

/// Tidal correction configuration. `None` on `ForceModelConfig` disables all tides.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TidesConfiguration {
    /// Permanent-tide / tide-system handling for the static field.
    pub permanent: PermanentTideConfig,
    /// Solid Earth tides. `None` disables solid tides (permanent-only is valid).
    pub solid: Option<SolidTideConfig>,
    /// Ocean tides. `None` disables ocean tides.
    pub ocean: Option<OceanTideConfig>,
    /// Ephemeris source for the Sun and Moon positions the tidal corrections
    /// are computed from. [`TidesConfiguration::default`] uses
    /// [`EphemerisSource::LowPrecision`] (the analytic geocentric Sun/Moon
    /// ephemerides), which is accurate enough for the ~1e-7 m/s² tidal
    /// perturbation. Set to a SPICE source
    /// ([`EphemerisSource::DE440s`]/[`EphemerisSource::DE440`]/
    /// [`EphemerisSource::SPK`]) to share positions with a third-body
    /// perturbation configured against the same source. Serialized
    /// configurations must carry this key (no serde default, matching the
    /// struct's other fields).
    pub ephemeris_source: EphemerisSource,
}

impl Default for TidesConfiguration {
    fn default() -> Self {
        Self {
            permanent: PermanentTideConfig::Auto,
            solid: None,
            ocean: None,
            ephemeris_source: EphemerisSource::LowPrecision,
        }
    }
}

impl ForceModelConfig {
    /// Create a force model configuration for a specific central body
    ///
    /// Convenience constructor that fills in `frame_transform` with its default
    /// (`FrameTransformationModel::FullEarthRotation`) so callers only need to
    /// specify the options that vary per central body. Does not validate the
    /// resulting configuration — call [`ForceModelConfig::validate`] to check
    /// that the chosen options are compatible with `central_body`.
    ///
    /// # Arguments
    /// * `central_body` - Body the orbit is propagated relative to
    /// * `gravity` - Gravity model configuration
    /// * `drag` - Atmospheric drag configuration (`None` to disable)
    /// * `srp` - Solar radiation pressure configuration (`None` to disable)
    /// * `third_body` - Third-body perturbation entries, one per body (`None` to disable)
    /// * `relativity` - Enable general relativistic corrections
    /// * `mass` - Spacecraft mass source (`None` if not needed)
    ///
    /// # Returns
    /// A `ForceModelConfig` for `central_body` with `frame_transform` set to
    /// its default value.
    ///
    /// # Examples
    /// ```rust
    /// use brahe::propagators::{CentralBody, ForceModelConfig, GravityConfiguration};
    ///
    /// let config = ForceModelConfig::for_body(
    ///     CentralBody::Mars,
    ///     GravityConfiguration::PointMass,
    ///     None,
    ///     None,
    ///     None,
    ///     false,
    ///     None,
    /// );
    /// assert_eq!(config.central_body, CentralBody::Mars);
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn for_body(
        central_body: CentralBody,
        gravity: GravityConfiguration,
        drag: Option<DragConfiguration>,
        srp: Option<SolarRadiationPressureConfiguration>,
        third_body: Option<Vec<ThirdBodyConfiguration>>,
        relativity: bool,
        mass: Option<ParameterSource>,
    ) -> Self {
        Self {
            central_body,
            gravity,
            drag,
            srp,
            third_body,
            relativity,
            mass,
            frame_transform: FrameTransformationModel::default(),
            tides: None,
        }
    }

    /// Create a configuration suitable for propagation about the Moon
    ///
    /// Uses:
    /// - 50×50 GRGM660PRIM lunar gravity model
    /// - No atmospheric drag (the Moon has none)
    /// - Solar radiation pressure with conical eclipse, occulted by the Moon and Earth
    /// - Earth and Sun third-body perturbations (DE440s ephemerides)
    /// - Relativity disabled
    pub fn lunar_default() -> Self {
        Self::for_body(
            CentralBody::Moon,
            GravityConfiguration::SphericalHarmonic {
                source: GravityModelSource::ModelType(GravityModelType::ICGEMModel {
                    body: ICGEMBody::Moon,
                    name: "GRGM660PRIM".to_string(),
                }),
                degree: 50,
                order: 50,
                parallel: ParallelMode::Auto,
            },
            None,
            Some(SolarRadiationPressureConfiguration {
                area: ParameterSource::ParameterIndex(3),
                cr: ParameterSource::ParameterIndex(4),
                eclipse_model: EclipseModel::Conical,
                occulting_bodies: vec![OccultingBody::Moon, OccultingBody::Earth],
            }),
            Some(vec![ThirdBody::Earth.into(), ThirdBody::Sun.into()]),
            false,
            Some(ParameterSource::ParameterIndex(0)),
        )
    }

    /// Create a configuration suitable for propagation about Mars
    ///
    /// Uses:
    /// - 50×50 GMM-2B Mars gravity model (Goddard Mars Model 2B, from Mars
    ///   Global Surveyor tracking; degree/order 80, truncated to 50×50 here).
    ///   The ICGEM celestial catalog lists this model under its file name
    ///   `ggm2bc80`, which is the identifier the ICGEM downloader resolves.
    /// - Exponential atmospheric drag (no NRLMSISE-00/Harris-Priester equivalent for Mars)
    /// - Solar radiation pressure with conical eclipse, occulted by Mars
    /// - Sun third-body perturbations (DE440s ephemerides)
    /// - Relativity disabled
    pub fn mars_default() -> Self {
        Self::for_body(
            CentralBody::Mars,
            GravityConfiguration::SphericalHarmonic {
                source: GravityModelSource::ModelType(GravityModelType::ICGEMModel {
                    body: ICGEMBody::Mars,
                    name: "ggm2bc80".to_string(),
                }),
                degree: 50,
                order: 50,
                parallel: ParallelMode::Auto,
            },
            Some(DragConfiguration {
                model: AtmosphericModel::Exponential {
                    scale_height: 11.1e3,
                    rho0: 0.020,
                    h0: 0.0,
                },
                area: ParameterSource::ParameterIndex(1),
                cd: ParameterSource::ParameterIndex(2),
                body: None,
            }),
            Some(SolarRadiationPressureConfiguration {
                area: ParameterSource::ParameterIndex(3),
                cr: ParameterSource::ParameterIndex(4),
                eclipse_model: EclipseModel::Conical,
                occulting_bodies: vec![OccultingBody::Mars],
            }),
            Some(vec![ThirdBody::Sun.into()]),
            false,
            Some(ParameterSource::ParameterIndex(0)),
        )
    }

    /// Create a configuration suitable for cislunar propagation about the Earth-Moon barycenter
    ///
    /// Uses:
    /// - No central gravity term (`GravityConfiguration::Zero` — the
    ///   Earth-Moon barycenter has no mass of its own; the Earth and Moon
    ///   third bodies below provide the actual gravitational terms)
    /// - No atmospheric drag
    /// - Solar radiation pressure with conical eclipse, occulted by Earth and the Moon
    /// - Earth, Moon, and Sun third-body perturbations (DE440s ephemerides)
    /// - Relativity disabled
    pub fn cislunar_default() -> Self {
        Self::for_body(
            CentralBody::EMB,
            GravityConfiguration::Zero,
            None,
            Some(SolarRadiationPressureConfiguration {
                area: ParameterSource::ParameterIndex(3),
                cr: ParameterSource::ParameterIndex(4),
                eclipse_model: EclipseModel::Conical,
                occulting_bodies: vec![OccultingBody::Earth, OccultingBody::Moon],
            }),
            Some(vec![
                ThirdBody::Earth.into(),
                ThirdBody::Moon.into(),
                ThirdBody::Sun.into(),
            ]),
            false,
            Some(ParameterSource::ParameterIndex(0)),
        )
    }

    /// Validate that this configuration's options are compatible with its central body
    ///
    /// Checks seven classes of central-body-dependent constraints:
    /// 1. Earth-specific central-body options (`GravityConfiguration::EarthZonal` as the
    ///    central gravity field, `FrameTransformationModel::EarthRotationOnly`,
    ///    `EphemerisSource::LowPrecision`, `TidesConfiguration`) are rejected for any
    ///    non-Earth central body.
    /// 2. Per third-body entry: `EphemerisSource::LowPrecision` only models the Sun and
    ///    Moon (and only about Earth); an entry whose NAIF ID matches the central body's
    ///    is rejected (a body cannot perturb an orbit centered on itself);
    ///    `GravityConfiguration::EarthZonal` requires `ThirdBody::Earth`; and
    ///    `GravityConfiguration::SphericalHarmonic` requires the body to have a known
    ///    body-fixed frame ([`ThirdBody::body_fixed_frame`]), which excludes the
    ///    `*Barycenter` variants and `Custom` bodies.
    /// 3. Barycenter central bodies (`CentralBody::is_barycenter`) have no mass or
    ///    rotation of their own, so `GravityConfiguration::SphericalHarmonic` as the
    ///    central gravity field is rejected.
    /// 4. Drag rules key on the *attributed* body — `DragConfiguration::body`, or the
    ///    central body when `None`: `AtmosphericModel::HarrisPriester`/`NRLMSISE00`
    ///    model Earth's atmosphere and require the attributed body to be Earth; the
    ///    attributed body must have a known radius and spin rate (needed for altitude
    ///    and atmospheric co-rotation). Drag about a barycenter central body is
    ///    therefore valid only with an explicit physical `DragConfiguration::body`.
    /// 5. `GravityConfiguration::SphericalHarmonic` on a `CentralBody::Custom` body
    ///    requires `central_body.fixed_frame()` to be set (needed to rotate into the
    ///    body-fixed frame the harmonics are expressed in).
    /// 6. The spherical-harmonic gravity field model must be a gravity field for the
    ///    declared `central_body`; the model's source body must match — see
    ///    [`ForceModelConfig::validate_gravity_model_body`] for how the model's body
    ///    is determined and compared.
    /// 7. `OceanTideConfig::degree` must be in `2..=100` (the FES2004 file's truncation
    ///    limit) and `OceanTideConfig::order` must be in `2..=degree`.
    ///
    /// This method is called automatically at propagator construction (e.g.
    /// `DNumericalOrbitPropagator::new`); it may also be called explicitly ahead
    /// of time on a standalone configuration for early feedback.
    ///
    /// # Returns
    /// `Ok(())` if the configuration is internally consistent, `Err(BraheError)`
    /// naming both the offending option and the central body otherwise.
    ///
    /// # Examples
    /// ```rust
    /// use brahe::propagators::ForceModelConfig;
    ///
    /// let config = ForceModelConfig::lunar_default();
    /// assert!(config.validate().is_ok());
    /// ```
    pub fn validate(&self) -> Result<(), BraheError> {
        let is_earth = matches!(self.central_body, CentralBody::Earth);

        if !is_earth {
            if matches!(self.gravity, GravityConfiguration::EarthZonal { .. }) {
                return Err(BraheError::Error(format!(
                    "GravityConfiguration::EarthZonal requires an Earth central body, but \
                     central_body is {}",
                    self.central_body
                )));
            }

            if matches!(
                self.frame_transform,
                FrameTransformationModel::EarthRotationOnly
            ) {
                return Err(BraheError::Error(format!(
                    "FrameTransformationModel::EarthRotationOnly requires an Earth central \
                     body, but central_body is {}",
                    self.central_body
                )));
            }

            if self.tides.is_some() {
                return Err(BraheError::Error(format!(
                    "TidesConfiguration models solid Earth tides (IERS §6.2) and requires an \
                     Earth central body, but central_body is {}",
                    self.central_body
                )));
            }
        }

        if let Some(ref tides_cfg) = self.tides
            && let Some(ref ocean) = tides_cfg.ocean
        {
            if ocean.degree < 2 || ocean.degree > 100 {
                return Err(BraheError::Error(format!(
                    "OceanTideConfig degree must be in 2..=100 (FES2004 file limit), got {}",
                    ocean.degree
                )));
            }
            if ocean.order > ocean.degree {
                return Err(BraheError::Error(format!(
                    "OceanTideConfig order ({}) must not exceed degree ({})",
                    ocean.order, ocean.degree
                )));
            }
            if ocean.order < 2 {
                return Err(BraheError::Error(format!(
                    "OceanTideConfig order must be at least 2 (FES2004 file limit), got {}",
                    ocean.order
                )));
            }
        }

        if let Some(ref third_body) = self.third_body {
            for tb in third_body {
                if matches!(tb.ephemeris_source, EphemerisSource::LowPrecision) {
                    if !is_earth {
                        return Err(BraheError::Error(format!(
                            "EphemerisSource::LowPrecision requires an Earth central body, but \
                             central_body is {}",
                            self.central_body
                        )));
                    }
                    if !matches!(tb.body, ThirdBody::Sun | ThirdBody::Moon) {
                        return Err(BraheError::Error(format!(
                            "EphemerisSource::LowPrecision only supports Sun and Moon third \
                             bodies, but {:?} was requested",
                            tb.body
                        )));
                    }
                }
                if tb.body.naif_id() == self.central_body.naif_id() {
                    return Err(BraheError::Error(format!(
                        "third body {:?} has the same NAIF ID ({}) as central body {}",
                        tb.body,
                        tb.body.naif_id(),
                        self.central_body
                    )));
                }
                // ThirdBody::MarsBarycenter (NAIF 4) sits inside the planet
                // (~0.1-0.2 m from its center): pairing it with a
                // Mars-centered propagation would divide by a near-zero
                // perturber distance. The planet itself (NAIF 499 ==
                // CentralBody::Mars.naif_id()) is caught by the NAIF-ID
                // equality check above. No other *Barycenter variant needs
                // this guard: the outer-planet system barycenters sit well
                // outside their planets' centers (e.g. the Jupiter system
                // barycenter is ~1e2 km from Jupiter's center), and brahe
                // has no built-in central body for those planets anyway.
                if matches!(tb.body, ThirdBody::MarsBarycenter)
                    && matches!(self.central_body, CentralBody::Mars)
                {
                    return Err(BraheError::Error(
                        "ThirdBody::MarsBarycenter (the Mars system barycenter) cannot perturb \
                         a Mars-centered propagation — Mars's gravity is the central-body force"
                            .to_string(),
                    ));
                }
                match &tb.gravity {
                    GravityConfiguration::PointMass => {}
                    GravityConfiguration::Zero => {
                        return Err(BraheError::Error(format!(
                            "GravityConfiguration::Zero on third body {:?} disables the entry                              entirely; remove the entry instead",
                            tb.body
                        )));
                    }
                    GravityConfiguration::EarthZonal { .. } => {
                        if !matches!(tb.body, ThirdBody::Earth) {
                            return Err(BraheError::Error(format!(
                                "GravityConfiguration::EarthZonal on a third body requires \
                                 ThirdBody::Earth (the J coefficients and reference radius are \
                                 Earth's), but the body is {:?}",
                                tb.body
                            )));
                        }
                    }
                    GravityConfiguration::SphericalHarmonic { .. } => {
                        if tb.body.body_fixed_frame().is_none() {
                            return Err(BraheError::Error(format!(
                                "GravityConfiguration::SphericalHarmonic on third body {:?} \
                                 requires a body-fixed frame to orient the field, but none is \
                                 known (barycenters and Custom bodies are not supported)",
                                tb.body
                            )));
                        }
                    }
                }
            }
        }

        if self.central_body.is_barycenter()
            && matches!(self.gravity, GravityConfiguration::SphericalHarmonic { .. })
        {
            return Err(BraheError::Error(format!(
                "GravityConfiguration::SphericalHarmonic requires a physical central body, \
                 but central_body is a barycenter ({})",
                self.central_body
            )));
        }

        if let Some(ref drag) = self.drag {
            // Rules key on the *attributed* body: the body whose atmosphere
            // produces the drag (the central body unless overridden).
            let attributed = drag.body.as_ref().unwrap_or(&self.central_body);
            match drag.model {
                AtmosphericModel::HarrisPriester => {
                    if !matches!(attributed, CentralBody::Earth) {
                        return Err(BraheError::Error(format!(
                            "AtmosphericModel::HarrisPriester models Earth's atmosphere and \
                             requires the drag body to be Earth, but the drag is attributed \
                             to {}",
                            attributed
                        )));
                    }
                }
                AtmosphericModel::NRLMSISE00 => {
                    if !matches!(attributed, CentralBody::Earth) {
                        return Err(BraheError::Error(format!(
                            "AtmosphericModel::NRLMSISE00 models Earth's atmosphere and \
                             requires the drag body to be Earth, but the drag is attributed \
                             to {}",
                            attributed
                        )));
                    }
                }
                AtmosphericModel::Exponential { .. } => {}
            }
            if attributed.radius().is_none() {
                return Err(BraheError::Error(format!(
                    "drag requires the attributed body's radius to be known, but {} has no \
                     known radius",
                    attributed
                )));
            }
            if attributed.omega_vector().is_none() {
                return Err(BraheError::Error(format!(
                    "drag requires the attributed body's spin rate to be known, but {} has no \
                     known spin rate",
                    attributed
                )));
            }
        }

        if matches!(self.central_body, CentralBody::Custom(_))
            && matches!(self.gravity, GravityConfiguration::SphericalHarmonic { .. })
            && self.central_body.fixed_frame().is_none()
        {
            return Err(BraheError::Error(format!(
                "GravityConfiguration::SphericalHarmonic on a Custom central body requires \
                 fixed_frame to be set, but central_body {} has no fixed frame",
                self.central_body
            )));
        }

        self.validate_gravity_model_body()?;

        Ok(())
    }

    /// Checks that a spherical-harmonic central gravity field's source body
    /// matches `central_body`, so a config is rejected rather than silently
    /// evaluating the wrong body's field (e.g. a bare
    /// `GravityConfiguration::SphericalHarmonic`, which defaults to Earth's
    /// EGM2008_120, attached to a Mars `central_body`).
    ///
    /// No-op for [`GravityConfiguration::Zero`]/`PointMass`/`EarthZonal`, and
    /// for [`GravityModelSource::Global`] (identity can't be checked before
    /// `set_global_gravity_model` populates it; the caller owns that
    /// pairing).
    ///
    /// For [`GravityModelSource::ModelType`]:
    /// - The packaged Earth-only models ([`GravityModelType::EGM2008_120`],
    ///   [`GravityModelType::GGM05S`], [`GravityModelType::JGM3`]) and
    ///   [`GravityModelType::ICGEMModel`] with `ICGEMBody::Earth`/`Moon`/`Mars`
    ///   are checked by NAIF ID against `central_body`, no load required.
    /// - `ICGEMModel` with `ICGEMBody::Venus`/`Ceres`/`Other`, and
    ///   `GravityModelType::FromFile`, have no statically known body: the
    ///   model is loaded (a network fetch on a cold `ICGEMModel` cache; a
    ///   local parse for `FromFile`) and its header GM and reference radius
    ///   are compared against `central_body.gm()`/`radius()` within a 1%
    ///   relative tolerance — this is a body-*identity* check only, not a
    ///   model-*quality* check.
    /// - `central_body.gm() <= 0.0` always errors here, since no real
    ///   gravity model has a non-positive GM.
    ///
    /// # Returns
    /// `Ok(())` if the gravity model's body matches `central_body` (or the check
    /// does not apply), `Err(BraheError)` naming both bodies otherwise.
    fn validate_gravity_model_body(&self) -> Result<(), BraheError> {
        let source = match &self.gravity {
            GravityConfiguration::SphericalHarmonic { source, .. } => source,
            _ => return Ok(()),
        };

        let model_type = match source {
            GravityModelSource::Global => return Ok(()),
            GravityModelSource::ModelType(model_type) => model_type,
        };

        // Bodies with a statically known identity: compare NAIF IDs, no load.
        let known_identity: Option<(i32, &str)> = match model_type {
            GravityModelType::EGM2008_120 | GravityModelType::GGM05S | GravityModelType::JGM3 => {
                Some((399, "Earth"))
            }
            GravityModelType::ICGEMModel { body, .. } => match body {
                ICGEMBody::Earth => Some((399, body.as_name())),
                ICGEMBody::Moon => Some((301, body.as_name())),
                ICGEMBody::Mars => Some((499, body.as_name())),
                ICGEMBody::Venus | ICGEMBody::Ceres | ICGEMBody::Other(_) => None,
            },
            GravityModelType::FromFile(_) => None,
        };

        if let Some((expected_naif_id, model_body_name)) = known_identity {
            if self.central_body.naif_id() != expected_naif_id {
                return Err(BraheError::Error(format!(
                    "gravity model {:?} is a {} model, but central_body is {} (NAIF {}); a \
                     spherical-harmonic central gravity field must model the same body the \
                     orbit is propagated relative to",
                    model_type,
                    model_body_name,
                    self.central_body,
                    self.central_body.naif_id()
                )));
            }
            return Ok(());
        }

        // No statically known body: load the model (via the process-wide cache
        // — an `Arc`, not a deep clone) and compare its header GM and reference
        // radius against the central body.
        let model = GravityModel::shared(model_type)?;
        let central_gm = self.central_body.gm();
        if central_gm <= 0.0 {
            // Every real spherical-harmonic model has a positive GM, so a
            // central body with non-positive GM (e.g. a `Custom` body with an
            // unset/zero `gm`) can never match one. Built-in barycenters
            // (`EMB`/`SSB`, `gm() == 0.0`) are already rejected earlier in
            // `validate()`; this also catches the equivalent `Custom` case,
            // which has no dedicated barycenter check of its own.
            return Err(BraheError::Error(format!(
                "central_body {} has non-positive GM ({:.6e} m^3/s^2), but a spherical-harmonic \
                 gravity model (which always has a positive GM) is attached; spherical-harmonic \
                 gravity requires a physical central body with a known positive GM",
                self.central_body, central_gm
            )));
        }
        Self::check_body_constant_within_tolerance(
            model_type,
            &self.central_body,
            "GM",
            "m^3/s^2",
            model.gm,
            central_gm,
        )?;
        if let Some(central_radius) = self.central_body.radius() {
            Self::check_body_constant_within_tolerance(
                model_type,
                &self.central_body,
                "reference radius",
                "m",
                model.radius,
                central_radius,
            )?;
        }

        Ok(())
    }

    /// Compare a loaded gravity model's header constant against the expected
    /// value for `central_body`, within a 1% relative tolerance.
    ///
    /// # Arguments
    /// - `model_type`: Gravity model being checked (named in the error).
    /// - `central_body`: Central body being checked against (named in the error).
    /// - `label`: Human-readable name of the constant (e.g. `"GM"`), used in the
    ///   error message.
    /// - `unit`: Unit string for the constant (e.g. `"m^3/s^2"`), used in the
    ///   error message.
    /// - `model_value`: The gravity model's header value for this constant.
    /// - `central_value`: `central_body`'s value for this constant.
    ///
    /// # Returns
    /// `Ok(())` if the values agree within 1% relative tolerance,
    /// `Err(BraheError)` naming both otherwise.
    fn check_body_constant_within_tolerance(
        model_type: &GravityModelType,
        central_body: &CentralBody,
        label: &str,
        unit: &str,
        model_value: f64,
        central_value: f64,
    ) -> Result<(), BraheError> {
        let rel_diff = (model_value - central_value).abs() / central_value;
        if rel_diff > 1e-2 {
            return Err(BraheError::Error(format!(
                "gravity model {:?} header {} ({:.6e} {}) does not match central_body {}'s {} \
                 ({:.6e} {}) within 1% relative tolerance; the spherical-harmonic model does not \
                 appear to belong to the propagation's central body",
                model_type, label, model_value, unit, central_body, label, central_value, unit
            )));
        }
        Ok(())
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
    ///
    /// The shared model is read-only: the propagator never mutates it, so any
    /// [`PermanentTideConfig`] in the force model has **no effect** for a global
    /// source. Resolve the model's tide system once, before install, with
    /// [`set_global_gravity_model_to_tide_system`](crate::gravity::set_global_gravity_model_to_tide_system)
    /// (or a manual `convert_tide_system` before `set_global_gravity_model`).
    Global,

    /// Load a specific gravity model type
    ///
    /// The model is loaded at propagator construction time and stored internally.
    /// Each propagator has its own copy of the model coefficients.
    ModelType(GravityModelType),
}

impl Default for GravityModelSource {
    fn default() -> Self {
        GravityModelSource::ModelType(GravityModelType::EGM2008_120)
    }
}

/// Supported max degrees for zonal harmonics. J_N will use J_2..J_N
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub enum ZonalHarmonicsDegree {
    J2,
    J3,
    J4,
    J5,
    J6,
}

impl Display for ZonalHarmonicsDegree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", usize::from(self))
    }
}

impl From<&ZonalHarmonicsDegree> for usize {
    fn from(value: &ZonalHarmonicsDegree) -> Self {
        match value {
            ZonalHarmonicsDegree::J2 => 2,
            ZonalHarmonicsDegree::J3 => 3,
            ZonalHarmonicsDegree::J4 => 4,
            ZonalHarmonicsDegree::J5 => 5,
            ZonalHarmonicsDegree::J6 => 6,
        }
    }
}

impl From<ZonalHarmonicsDegree> for usize {
    fn from(value: ZonalHarmonicsDegree) -> Self {
        (&value).into()
    }
}

/// Controls how the inertial-to-body-fixed rotation is computed for force-model
/// evaluation in the numerical propagator.
///
/// The numerical propagator integrates orbital dynamics in an Earth-centered inertial
/// (ECI) frame, but several perturbations are most naturally evaluated in an Earth-fixed
/// (ECEF) frame: spherical-harmonic and zonal gravity, NRLMSISE-00 atmospheric density,
/// and atmospheric drag (which depends on the relative velocity of the rotating
/// atmosphere). This setting selects the precision/speed trade-off used for that ECI↔ECEF
/// rotation across the entire force model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum FrameTransformationModel {
    /// Apply only the Earth Rotation Angle (ERA) rotation about Earth's z-axis.
    ///
    /// Accounts for sidereal rotation but ignores precession, nutation, and polar motion.
    /// ~1.5x faster than `FullEarthRotation` but introduces small errors (~0.07°) from the
    /// uncorrected pole tilt.
    EarthRotationOnly,

    /// Apply the full IAU 2006/2000A rotation: bias-precession-nutation + ERA + polar motion.
    ///
    /// This is the default. Matches the rotation that brahe uses elsewhere for ECI↔ECEF
    /// conversion, giving results consistent with the rest of the library.
    #[default]
    FullEarthRotation,
}

/// Gravity model configuration
///
/// Specifies the gravity model to use for computing gravitational acceleration.
/// Can be either simple two-body point mass or high-fidelity spherical harmonic expansion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GravityConfiguration {
    /// No gravity term from the propagation center.
    ///
    /// For barycentric propagation centers (`CentralBody::EMB`,
    /// `CentralBody::SSB`), which have no mass of their own: every
    /// gravitational force enters through the third-body entries.
    /// Numerically equivalent to `PointMass` about a barycenter (whose GM is
    /// zero), but explicit that no central attracting body exists.
    Zero,

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
        /// Parallelization policy for the acceleration computation.
        #[serde(default)]
        parallel: ParallelMode,
    },

    /// Earth zonal harmonics (J_2..J_n) only.
    ///
    /// Has the same effect as setting m = 0 in SphericalHarmonic with the
    /// packaged Earth gravity model, but is evaluated via a closed-form
    /// expansion the compiler can vectorise (~50% speedup). Earth-specific
    /// because the J_n coefficients and reference radius are baked in.
    EarthZonal {
        /// Maximum degree (n) of expansion, order (m) is 0
        degree: ZonalHarmonicsDegree,
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
/// // Fixed Cd, variable area, drag about the central body
/// let drag = DragConfiguration {
///     model: AtmosphericModel::HarrisPriester,
///     area: ParameterSource::ParameterIndex(1),  // From params[1]
///     cd: ParameterSource::Value(2.2),            // Fixed
///     body: None,
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

    /// Body whose atmosphere produces the drag. `None` (default) means the
    /// propagation's central body. Set to a different body to evaluate drag
    /// at the object's state relative to that body — e.g. Earth drag on a
    /// cislunar trajectory propagated about `CentralBody::EMB`.
    #[serde(default)]
    pub body: Option<CentralBody>,
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
/// use brahe::propagators::{SolarRadiationPressureConfiguration, EclipseModel, OccultingBody, ParameterSource};
///
/// // Variable area and Cr
/// let srp = SolarRadiationPressureConfiguration {
///     area: ParameterSource::ParameterIndex(3),
///     cr: ParameterSource::ParameterIndex(4),
///     eclipse_model: EclipseModel::Conical,
///     occulting_bodies: vec![OccultingBody::Earth],
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

    /// Occulting bodies to consider for shadow/eclipse determination
    ///
    /// Each body's position (resolved via its `naif_position_id`) and physical
    /// radius are used to compute an illumination fraction; when multiple bodies
    /// are listed, the minimum illumination fraction across all bodies is used.
    /// Defaults to `[OccultingBody::Earth]` for backward compatibility.
    #[serde(default = "default_occulting_bodies")]
    pub occulting_bodies: Vec<OccultingBody>,
}

fn default_occulting_bodies() -> Vec<OccultingBody> {
    vec![OccultingBody::Earth]
}

/// Occulting body for eclipse/shadow modeling in solar radiation pressure calculations
///
/// Identifies a celestial body whose shadow may occult the sun as seen from the
/// spacecraft. Each variant provides the body's mean physical radius (via
/// [`OccultingBody::radius`]) and NAIF ID (via [`OccultingBody::naif_id`]) used in
/// eclipse geometry calculations.
///
/// # Examples
///
/// ```rust
/// use brahe::propagators::OccultingBody;
///
/// let earth = OccultingBody::Earth;
/// assert_eq!(earth.naif_id(), 399);
///
/// let custom = OccultingBody::Custom { name: "Europa".to_string(), naif_id: 502, radius: 1560.8e3 };
/// assert_eq!(custom.radius(), 1560.8e3);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OccultingBody {
    /// Earth
    Earth,

    /// Moon
    Moon,

    /// Mars
    Mars,

    /// User-defined occulting body
    Custom {
        /// Descriptive name of the body
        name: String,
        /// NAIF ID of the physical body (used for both shadow geometry and, unless
        /// overridden via a dedicated position ID, ephemeris position lookups)
        naif_id: i32,
        /// Mean physical radius of the body [m]
        radius: f64,
    },
}

impl OccultingBody {
    /// Mean physical radius of the occulting body [m]
    ///
    /// # Returns
    ///
    /// - Physical radius of the body, in meters.
    pub fn radius(&self) -> f64 {
        match self {
            OccultingBody::Earth => R_EARTH,
            OccultingBody::Moon => R_MOON,
            OccultingBody::Mars => R_MARS,
            OccultingBody::Custom { radius, .. } => *radius,
        }
    }

    /// NAIF ID of the physical occulting body
    ///
    /// This identifies the body whose physical radius defines the shadow geometry.
    /// For Mars this is 499 (the planet itself), which differs from the ID used to
    /// query its position via SPK — see [`OccultingBody::naif_position_id`].
    ///
    /// # Returns
    ///
    /// - NAIF integer ID of the physical body.
    pub fn naif_id(&self) -> i32 {
        match self {
            OccultingBody::Earth => 399,
            OccultingBody::Moon => 301,
            OccultingBody::Mars => 499,
            OccultingBody::Custom { naif_id, .. } => *naif_id,
        }
    }

    /// NAIF ID to use when resolving the occulting body's position via SPK ephemerides
    ///
    /// Identical to [`OccultingBody::naif_id`] for every variant: the occulter's
    /// position is queried for the physical body center. For Mars (NAIF 499) this
    /// requires the `mar099s` satellite ephemeris kernel when the occulter is not
    /// co-located with the propagation's central body (a Mars-centered propagation
    /// short-circuits the query entirely).
    ///
    /// # Returns
    ///
    /// - NAIF integer ID to use for SPK position queries.
    pub fn naif_position_id(&self) -> i32 {
        self.naif_id()
    }
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

/// A single third-body perturber and how its force is modeled.
///
/// Each perturber carries its own ephemeris source and gravity model, so a
/// force configuration can, for example, model the Moon as a point mass from
/// DE440s while modeling Earth with a spherical-harmonic field (see
/// [`GravityConfiguration`]). The default gravity model is
/// [`GravityConfiguration::PointMass`], the classical third-body formulation.
///
/// Deserializes from either a bare [`ThirdBody`] (which takes the DE440s
/// default ephemeris source and point-mass gravity) or the full structure
/// with `ephemeris_source` and `gravity` individually optional.
///
/// # Examples
/// ```rust
/// use brahe::propagators::force_model_config::{
///     GravityConfiguration, ThirdBody, ThirdBodyConfiguration, ZonalHarmonicsDegree,
/// };
///
/// // Point-mass Sun with default (DE440s) ephemerides
/// let sun: ThirdBodyConfiguration = ThirdBody::Sun.into();
///
/// // Earth as a J2 zonal perturber (for a non-Earth-centered propagation)
/// let earth = ThirdBodyConfiguration {
///     gravity: GravityConfiguration::EarthZonal {
///         degree: ZonalHarmonicsDegree::J2,
///     },
///     ..ThirdBody::Earth.into()
/// };
/// ```
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ThirdBodyConfiguration {
    /// The perturbing body.
    pub body: ThirdBody,

    /// Source for the body's ephemeris. Default: [`EphemerisSource::DE440s`].
    pub ephemeris_source: EphemerisSource,

    /// Gravity model for this perturber. Default:
    /// [`GravityConfiguration::PointMass`]. `SphericalHarmonic` and
    /// `EarthZonal` evaluate the field at the object's position relative to
    /// the body — see [`ForceModelConfig::validate`] for the applicable
    /// rules.
    pub gravity: GravityConfiguration,
}

fn default_ephemeris_source() -> EphemerisSource {
    EphemerisSource::DE440s
}

fn default_third_body_gravity() -> GravityConfiguration {
    GravityConfiguration::PointMass
}

impl From<ThirdBody> for ThirdBodyConfiguration {
    fn from(body: ThirdBody) -> Self {
        Self {
            body,
            ephemeris_source: default_ephemeris_source(),
            gravity: default_third_body_gravity(),
        }
    }
}

impl ThirdBodyConfiguration {
    /// Create a point-mass, DE440s-sourced configuration for `body`.
    ///
    /// # Arguments
    /// * `body` - The perturbing body
    ///
    /// # Returns
    /// A `ThirdBodyConfiguration` with [`EphemerisSource::DE440s`] and
    /// [`GravityConfiguration::PointMass`].
    ///
    /// # Examples
    /// ```rust
    /// use brahe::propagators::force_model_config::{ThirdBody, ThirdBodyConfiguration};
    ///
    /// let sun = ThirdBodyConfiguration::new(ThirdBody::Sun);
    /// assert_eq!(sun.body, ThirdBody::Sun);
    /// ```
    pub fn new(body: ThirdBody) -> Self {
        body.into()
    }
}

impl<'de> serde::Deserialize<'de> for ThirdBodyConfiguration {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Full {
            body: ThirdBody,
            #[serde(default = "default_ephemeris_source")]
            ephemeris_source: EphemerisSource,
            #[serde(default = "default_third_body_gravity")]
            gravity: GravityConfiguration,
        }
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            Bare(ThirdBody),
            Full(Full),
        }
        Ok(match Repr::deserialize(deserializer)? {
            Repr::Bare(body) => body.into(),
            Repr::Full(f) => ThirdBodyConfiguration {
                body: f.body,
                ephemeris_source: f.ephemeris_source,
                gravity: f.gravity,
            },
        })
    }
}

fn deserialize_third_bodies<'de, D>(
    deserializer: D,
) -> Result<Option<Vec<ThirdBodyConfiguration>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    /// Pre-flattening third-body configuration shape (one shared ephemeris
    /// source, a bare list of bodies). Distinguishable from the per-body
    /// shapes: a single entry requires a `body` key and a bare body is a
    /// string or a `Custom` map, neither of which carries `bodies`.
    #[derive(Deserialize)]
    struct Legacy {
        ephemeris_source: EphemerisSource,
        bodies: Vec<ThirdBody>,
    }

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Repr {
        One(ThirdBodyConfiguration),
        Many(Vec<ThirdBodyConfiguration>),
        Legacy(Legacy),
    }

    Ok(match Option::<Repr>::deserialize(deserializer)? {
        None => None,
        Some(Repr::One(c)) => Some(vec![c]),
        Some(Repr::Many(v)) => Some(v),
        Some(Repr::Legacy(legacy)) => Some(
            legacy
                .bodies
                .into_iter()
                .map(|body| {
                    // Pre-split configurations used Mars..Neptune to mean the
                    // planetary-system barycenters; remap so a loaded legacy
                    // configuration keeps its original physics.
                    let body = match body {
                        ThirdBody::Mars => ThirdBody::MarsBarycenter,
                        ThirdBody::Jupiter => ThirdBody::JupiterBarycenter,
                        ThirdBody::Saturn => ThirdBody::SaturnBarycenter,
                        ThirdBody::Uranus => ThirdBody::UranusBarycenter,
                        ThirdBody::Neptune => ThirdBody::NeptuneBarycenter,
                        other => other,
                    };
                    ThirdBodyConfiguration {
                        body,
                        ephemeris_source: legacy.ephemeris_source,
                        gravity: GravityConfiguration::PointMass,
                    }
                })
                .collect(),
        ),
    })
}

/// Source for celestial body ephemerides
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum EphemerisSource {
    /// Low-precision analytical ephemerides
    ///
    /// Uses simplified analytical models for Sun and Moon positions.
    /// Fast but less accurate (~km level errors).
    /// Only Sun and Moon are available.
    LowPrecision,

    /// High-precision JPL DE440s ephemerides (1550-2650 CE)
    ///
    /// Uses JPL Development Ephemeris 440 (small bodies version).
    /// High accuracy (~m level) but requires ephemeris file and slower evaluation.
    /// Valid over time range 1550-2650 CE.
    /// All planets available. File size ~33 MB.
    DE440s,

    /// Full-precision JPL DE440 ephemerides (13200 BCE-17191 CE)
    ///
    /// Uses JPL Development Ephemeris 440 (full version).
    /// Highest accuracy (~mm level) but requires larger ephemeris file and slower evaluation.
    /// Valid over extended time range 13200 BCE-17191 CE.
    /// All planets available. File size ~120 MB.
    DE440,

    /// Custom SPK-backed ephemeris source.
    ///
    /// Uses an explicitly selected SPK kernel while keeping the higher-level
    /// force-model interface centered on `EphemerisSource`.
    SPK(SPICEKernel),
}

impl TryFrom<EphemerisSource> for SPICEKernel {
    type Error = BraheError;

    fn try_from(source: EphemerisSource) -> Result<Self, Self::Error> {
        match source {
            EphemerisSource::DE440s => Ok(SPICEKernel::DE440s),
            EphemerisSource::DE440 => Ok(SPICEKernel::DE440),
            EphemerisSource::SPK(kernel) => Ok(kernel),
            EphemerisSource::LowPrecision => Err(BraheError::Error(
                "LowPrecision is not a valid DE kernel - use DE440s, DE440, or EphemerisSource::SPK(...)"
                    .to_string(),
            )),
        }
    }
}

/// Third-body perturber
///
/// Celestial bodies that can act as gravitational perturbers on the spacecraft.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ThirdBody {
    /// Sun
    Sun,
    /// Moon
    Moon,
    /// Mercury
    Mercury,
    /// Venus
    Venus,
    /// Mars (planet center, NAIF 499) with the planet-only GM.
    ///
    /// For the classical third-body formulation about Earth prefer
    /// [`ThirdBody::MarsBarycenter`], which is resolvable from the DE kernels
    /// alone; the planet-center position additionally requires the Mars
    /// satellite ephemeris kernel.
    Mars,
    /// Mars system barycenter (NAIF 4): Mars, Phobos, and Deimos combined,
    /// with the system GM. Used by the default Earth force models.
    MarsBarycenter,
    /// Jupiter (planet center, NAIF 599) with the planet-only GM. See
    /// [`ThirdBody::Mars`] on the planet-center vs barycenter distinction.
    Jupiter,
    /// Jupiter system barycenter (NAIF 5) with the system GM.
    JupiterBarycenter,
    /// Saturn (planet center, NAIF 699) with the planet-only GM.
    Saturn,
    /// Saturn system barycenter (NAIF 6) with the system GM.
    SaturnBarycenter,
    /// Uranus (planet center, NAIF 799) with the planet-only GM.
    Uranus,
    /// Uranus system barycenter (NAIF 7) with the system GM.
    UranusBarycenter,
    /// Neptune (planet center, NAIF 899) with the planet-only GM.
    Neptune,
    /// Neptune system barycenter (NAIF 8) with the system GM.
    NeptuneBarycenter,
    /// Earth
    ///
    /// Only meaningful as a perturber when the central body is not Earth
    /// itself (e.g. `CentralBody::EMB` or `CentralBody::Mars`).
    Earth,
    /// Phobos, the larger of Mars's two moons.
    Phobos,
    /// Deimos, the smaller of Mars's two moons.
    Deimos,
    /// A user-defined perturbing body, for bodies without a dedicated variant.
    Custom {
        /// Human-readable name (e.g. `"Ceres"`).
        name: String,
        /// NAIF ID of the body.
        naif_id: i32,
        /// Gravitational parameter. Units: (m^3/s^2)
        gm: f64,
    },
}

impl ThirdBody {
    /// NAIF ID of the perturbing body.
    ///
    /// # Returns
    /// - `naif_id`: NAIF ID. Planet variants use body centers (Mars is `499`,
    ///   Jupiter `599`, ...); the `*Barycenter` variants use the
    ///   planetary-system barycenters (Mars system is `4`, Jupiter system
    ///   `5`, ...), matching the targets used by the
    ///   `*_barycenter_position_spice` ephemeris functions.
    ///
    /// # Examples
    /// ```
    /// use brahe::propagators::force_model_config::ThirdBody;
    ///
    /// assert_eq!(ThirdBody::Sun.naif_id(), 10);
    /// assert_eq!(ThirdBody::Mars.naif_id(), 499);
    /// assert_eq!(ThirdBody::MarsBarycenter.naif_id(), 4);
    /// assert_eq!(ThirdBody::Phobos.naif_id(), 401);
    /// ```
    pub fn naif_id(&self) -> i32 {
        match self {
            ThirdBody::Sun => 10,
            ThirdBody::Moon => 301,
            ThirdBody::Mercury => 199,
            ThirdBody::Venus => 299,
            ThirdBody::Mars => 499,
            ThirdBody::MarsBarycenter => 4,
            ThirdBody::Jupiter => 599,
            ThirdBody::JupiterBarycenter => 5,
            ThirdBody::Saturn => 699,
            ThirdBody::SaturnBarycenter => 6,
            ThirdBody::Uranus => 799,
            ThirdBody::UranusBarycenter => 7,
            ThirdBody::Neptune => 899,
            ThirdBody::NeptuneBarycenter => 8,
            ThirdBody::Earth => 399,
            ThirdBody::Phobos => 401,
            ThirdBody::Deimos => 402,
            ThirdBody::Custom { naif_id, .. } => *naif_id,
        }
    }

    /// Gravitational parameter of the perturbing body.
    ///
    /// # Returns
    /// - `gm`: Gravitational parameter. Units: (m^3/s^2)
    ///
    /// # Examples
    /// ```
    /// use brahe::propagators::force_model_config::ThirdBody;
    /// use brahe::constants::GM_SUN;
    ///
    /// assert_eq!(ThirdBody::Sun.gm(), GM_SUN);
    /// ```
    pub fn gm(&self) -> f64 {
        match self {
            ThirdBody::Sun => GM_SUN,
            ThirdBody::Moon => GM_MOON,
            ThirdBody::Mercury => GM_MERCURY,
            ThirdBody::Venus => GM_VENUS,
            ThirdBody::Mars => GM_MARS,
            ThirdBody::MarsBarycenter => GM_MARS_SYSTEM,
            ThirdBody::Jupiter => GM_JUPITER,
            ThirdBody::JupiterBarycenter => GM_JUPITER_SYSTEM,
            ThirdBody::Saturn => GM_SATURN,
            ThirdBody::SaturnBarycenter => GM_SATURN_SYSTEM,
            ThirdBody::Uranus => GM_URANUS,
            ThirdBody::UranusBarycenter => GM_URANUS_SYSTEM,
            ThirdBody::Neptune => GM_NEPTUNE,
            ThirdBody::NeptuneBarycenter => GM_NEPTUNE_SYSTEM,
            ThirdBody::Earth => GM_EARTH,
            ThirdBody::Phobos => GM_PHOBOS,
            ThirdBody::Deimos => GM_DEIMOS,
            ThirdBody::Custom { gm, .. } => *gm,
        }
    }

    /// The [`CentralBody`] equivalent of this perturber, if it is a physical
    /// body brahe knows how to treat as a frame/parameter center.
    ///
    /// Returns `None` for the `*Barycenter` variants (no physical body) and
    /// for [`ThirdBody::Custom`] (no registered frames or parameters).
    ///
    /// # Returns
    /// - `Some(CentralBody)` for Sun, Moon, Earth, the planet-center
    ///   variants, Phobos, and Deimos
    /// - `None` for the `*Barycenter` variants and `Custom`
    ///
    /// # Examples
    /// ```
    /// use brahe::propagators::CentralBody;
    /// use brahe::propagators::force_model_config::ThirdBody;
    ///
    /// assert_eq!(ThirdBody::Earth.as_central_body(), Some(CentralBody::Earth));
    /// assert_eq!(ThirdBody::MarsBarycenter.as_central_body(), None);
    /// ```
    pub fn as_central_body(&self) -> Option<CentralBody> {
        match self {
            ThirdBody::MarsBarycenter
            | ThirdBody::JupiterBarycenter
            | ThirdBody::SaturnBarycenter
            | ThirdBody::UranusBarycenter
            | ThirdBody::NeptuneBarycenter
            | ThirdBody::Custom { .. } => None,
            _ => CentralBody::from_naif_id(self.naif_id()).ok(),
        }
    }

    /// The body-fixed reference frame a gravity field attached to this
    /// perturber is expressed in (e.g. ITRF for Earth, LFPA for the Moon,
    /// MCMF for Mars, IAU frames for the other planet centers).
    ///
    /// # Returns
    /// - `Some(ReferenceFrame)` when the body has a known body-fixed frame
    /// - `None` for the `*Barycenter` variants, `Custom` bodies, and bodies
    ///   without a rotation model
    ///
    /// # Examples
    /// ```
    /// use brahe::frames::ReferenceFrame;
    /// use brahe::propagators::force_model_config::ThirdBody;
    ///
    /// assert_eq!(ThirdBody::Earth.body_fixed_frame(), Some(ReferenceFrame::ITRF));
    /// assert_eq!(ThirdBody::NeptuneBarycenter.body_fixed_frame(), None);
    /// ```
    pub fn body_fixed_frame(&self) -> Option<crate::frames::ReferenceFrame> {
        self.as_central_body().and_then(|cb| cb.fixed_frame())
    }
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
        let config = ForceModelConfig::default();

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
    fn test_parametersource_get_value_valid() {
        // Fixed value resolves without a parameter vector; an in-range index
        // resolves from the provided vector.
        assert_eq!(ParameterSource::Value(2.2).get_value(None).unwrap(), 2.2);

        let params = nalgebra::DVector::from_vec(vec![10.0, 20.0]);
        assert_eq!(
            ParameterSource::ParameterIndex(1)
                .get_value(Some(&params))
                .unwrap(),
            20.0
        );
    }

    #[test]
    fn test_parametersource_get_value_missing_params() {
        // A ParameterIndex source with no parameter vector is out of bounds.
        let result = ParameterSource::ParameterIndex(2).get_value(None);
        assert!(matches!(result, Err(BraheError::OutOfBoundsError(_))));
    }

    #[test]
    fn test_parametersource_get_value_index_out_of_bounds() {
        // A ParameterIndex beyond the vector length is out of bounds.
        let params = nalgebra::DVector::from_vec(vec![1.0, 2.0]);
        let result = ParameterSource::ParameterIndex(4).get_value(Some(&params));
        assert!(matches!(result, Err(BraheError::OutOfBoundsError(_))));
    }

    #[test]
    fn test_high_fidelity_configuration() {
        let config = ForceModelConfig::high_fidelity();

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
        assert!(
            tb.iter()
                .all(|e| matches!(e.ephemeris_source, EphemerisSource::DE440s))
        );

        // Check relativity enabled
        assert!(config.relativity);

        // Check mass is configured to use params[0]
        assert!(matches!(
            config.mass,
            Some(ParameterSource::ParameterIndex(0))
        ));

        // Check solid Earth tides enabled with frequency-dependent corrections
        let tides = config.tides.unwrap();
        assert_eq!(tides.permanent, PermanentTideConfig::Auto);
        assert!(tides.solid.unwrap().frequency_dependent);
    }

    #[test]
    fn test_ephemeris_source_to_spk_kernel() {
        assert_eq!(
            SPICEKernel::try_from(EphemerisSource::DE440s).unwrap(),
            SPICEKernel::DE440s
        );
        assert_eq!(
            SPICEKernel::try_from(EphemerisSource::DE440).unwrap(),
            SPICEKernel::DE440
        );
        assert_eq!(
            SPICEKernel::try_from(EphemerisSource::SPK(SPICEKernel::DE440s)).unwrap(),
            SPICEKernel::DE440s
        );
        assert!(SPICEKernel::try_from(EphemerisSource::LowPrecision).is_err());
    }

    #[test]
    fn test_gravity_only_configuration() {
        let config = ForceModelConfig::earth_gravity();

        assert!(matches!(
            config.gravity,
            GravityConfiguration::SphericalHarmonic { .. }
        ));
        assert!(config.drag.is_none());
        assert!(config.srp.is_none());
        assert!(config.third_body.is_none());
        assert!(!config.relativity);
        assert!(config.mass.is_none());
    }

    #[test]
    fn test_leo_configuration() {
        let config = ForceModelConfig::leo_default();

        // LEO should have drag (dominant perturbation)
        assert!(config.drag.is_some());

        // LEO also has SRP for completeness
        assert!(config.srp.is_some());

        // LEO has Sun/Moon third-body
        assert!(config.third_body.is_some());

        // Check mass is configured to use params[0]
        assert!(matches!(
            config.mass,
            Some(ParameterSource::ParameterIndex(0))
        ));
    }

    #[test]
    fn test_geo_configuration() {
        let config = ForceModelConfig::geo_default();

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
        let config = ForceModelConfig::default();

        // Serialize to JSON
        let json = serde_json::to_string(&config).unwrap();

        // Deserialize back
        let deserialized: ForceModelConfig = serde_json::from_str(&json).unwrap();

        // Check equality (at least for some fields)
        assert!(matches!(
            deserialized.gravity,
            GravityConfiguration::SphericalHarmonic { .. }
        ));
        assert!(deserialized.drag.is_some());
    }

    #[test]
    fn test_requires_params_with_mass_param_index() {
        // When mass uses ParameterIndex, requires_params should return true
        let config = ForceModelConfig {
            mass: Some(ParameterSource::ParameterIndex(5)),
            drag: None,
            srp: None,
            ..ForceModelConfig::earth_gravity()
        };

        assert!(config.requires_params());
    }

    #[test]
    fn test_requires_params_with_mass_value() {
        // When mass uses Value, requires_params should return false (if no other params needed)
        let config = ForceModelConfig {
            mass: Some(ParameterSource::Value(1000.0)),
            drag: None,
            srp: None,
            ..ForceModelConfig::earth_gravity()
        };

        assert!(!config.requires_params());
    }

    #[test]
    fn test_tides_config_default_none() {
        let config = ForceModelConfig::default();
        assert!(config.tides.is_none());
    }

    #[test]
    fn test_permanent_tide_default_is_auto() {
        assert_eq!(PermanentTideConfig::default(), PermanentTideConfig::Auto);
    }

    #[test]
    fn test_tides_config_serde_roundtrip() {
        use crate::orbit_dynamics::gravity::GravityModelTideSystem;
        let cfg = TidesConfiguration {
            ephemeris_source: EphemerisSource::LowPrecision,
            permanent: PermanentTideConfig::ConvertTo(GravityModelTideSystem::ZeroTide),
            solid: Some(crate::orbit_dynamics::tides::SolidTideConfig {
                frequency_dependent: true,
                pole_tide: false,
            }),
            ocean: None,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: TidesConfiguration = serde_json::from_str(&json).unwrap();
        assert_eq!(back, cfg);
    }

    #[test]
    fn test_force_config_missing_tides_field_deserializes() {
        // Back-compat: configs serialized before this field still load.
        let json = r#"{"gravity":"PointMass","drag":null,"srp":null,"third_body":null,
            "relativity":false,"mass":null,"frame_transform":"FullEarthRotation"}"#;
        let cfg: ForceModelConfig = serde_json::from_str(json).unwrap();
        assert!(cfg.tides.is_none());
    }

    #[test]
    #[serial_test::parallel]
    fn test_ocean_tide_config_default() {
        let c = OceanTideConfig::default();
        assert_eq!(c.degree, 20);
        assert_eq!(c.order, 20);
        assert!(c.include_admittance);
        assert!(!c.pole_tide);
    }

    #[test]
    #[serial_test::parallel]
    fn test_ocean_tide_config_validation() {
        let mut config = ForceModelConfig::earth_gravity();
        config.tides = Some(TidesConfiguration {
            ephemeris_source: EphemerisSource::LowPrecision,
            permanent: PermanentTideConfig::Auto,
            solid: None,
            ocean: Some(OceanTideConfig {
                degree: 101,
                order: 20,
                include_admittance: true,
                pole_tide: false,
            }),
        });
        assert!(config.validate().is_err(), "degree > 100 must be rejected");

        config.tides = Some(TidesConfiguration {
            ephemeris_source: EphemerisSource::LowPrecision,
            permanent: PermanentTideConfig::Auto,
            solid: None,
            ocean: Some(OceanTideConfig {
                degree: 20,
                order: 30,
                include_admittance: true,
                pole_tide: false,
            }),
        });
        assert!(
            config.validate().is_err(),
            "order > degree must be rejected"
        );

        config.tides = Some(TidesConfiguration {
            ephemeris_source: EphemerisSource::LowPrecision,
            permanent: PermanentTideConfig::Auto,
            solid: None,
            ocean: Some(OceanTideConfig {
                degree: 1,
                order: 1,
                include_admittance: true,
                pole_tide: false,
            }),
        });
        assert!(config.validate().is_err(), "degree < 2 must be rejected");

        config.tides = Some(TidesConfiguration {
            ephemeris_source: EphemerisSource::LowPrecision,
            permanent: PermanentTideConfig::Auto,
            solid: None,
            ocean: Some(OceanTideConfig {
                degree: 20,
                order: 1,
                include_admittance: true,
                pole_tide: false,
            }),
        });
        assert!(config.validate().is_err(), "order < 2 must be rejected");

        config.tides = Some(TidesConfiguration {
            ephemeris_source: EphemerisSource::LowPrecision,
            permanent: PermanentTideConfig::Auto,
            solid: None,
            ocean: Some(OceanTideConfig::default()),
        });
        assert!(config.validate().is_ok());
    }

    #[test]
    #[serial_test::parallel]
    fn test_tides_configuration_serde_roundtrip() {
        let tides = TidesConfiguration {
            ephemeris_source: EphemerisSource::LowPrecision,
            permanent: PermanentTideConfig::Auto,
            solid: Some(SolidTideConfig {
                frequency_dependent: true,
                pole_tide: true,
            }),
            ocean: Some(OceanTideConfig {
                degree: 30,
                order: 30,
                include_admittance: false,
                pole_tide: true,
            }),
        };
        let json = serde_json::to_string(&tides).unwrap();
        let back: TidesConfiguration = serde_json::from_str(&json).unwrap();
        assert_eq!(tides, back);
        // pole_tide has a serde default (matches frequency_dependent convention).
        let solid: SolidTideConfig = serde_json::from_str("{}").unwrap();
        assert!(!solid.pole_tide);
    }

    #[test]
    #[serial_test::parallel]
    fn test_tides_configuration_missing_ocean_key_deserializes_none() {
        // serde_derive deserializes a missing Option field to None even without
        // #[serde(default)], so configurations serialized before the ocean field
        // existed load with ocean tides disabled rather than erroring.
        let tides: TidesConfiguration = serde_json::from_str(
            r#"{"permanent":"Auto","solid":null,"ephemeris_source":"LowPrecision"}"#,
        )
        .unwrap();
        assert_eq!(tides.ocean, None);
        assert_eq!(tides.solid, None);
    }

    #[test]
    #[serial_test::parallel]
    fn test_tides_configuration_missing_ephemeris_source_errors() {
        // `ephemeris_source` carries no serde default (none of this struct's
        // fields do), so a config serialized before the field existed fails to
        // deserialize — a deliberate breaking change; the missing key must be
        // added explicitly.
        let result: Result<TidesConfiguration, _> =
            serde_json::from_str(r#"{"permanent":"Auto","solid":null,"ocean":null}"#);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("ephemeris_source"));
    }

    #[test]
    #[serial_test::parallel]
    fn test_high_fidelity_enables_all_tides() {
        let config = ForceModelConfig::high_fidelity();
        let tides = config.tides.unwrap();
        let solid = tides.solid.unwrap();
        assert!(solid.frequency_dependent);
        assert!(solid.pole_tide);
        let ocean = tides.ocean.unwrap();
        assert_eq!(ocean.degree, 30);
        assert_eq!(ocean.order, 30);
        assert!(ocean.include_admittance);
        assert!(ocean.pole_tide);
        // high_fidelity shares DE440s with its third-body config so tidal and
        // third-body Sun/Moon positions come from the same ephemeris.
        assert_eq!(tides.ephemeris_source, EphemerisSource::DE440s);
    }

    #[test]
    fn test_spherical_harmonic_parallel_serde_default() {
        // A JSON config missing `parallel` deserializes to Auto.
        let json =
            r#"{"SphericalHarmonic":{"source":{"ModelType":"JGM3"},"degree":20,"order":20}}"#;
        let cfg: GravityConfiguration = serde_json::from_str(json).unwrap();
        match &cfg {
            GravityConfiguration::SphericalHarmonic { parallel, .. } => {
                assert_eq!(*parallel, ParallelMode::Auto);
            }
            _ => panic!("expected SphericalHarmonic"),
        }
        // Field round-trips when present (guards against a ParallelMode rename
        // silently breaking serialization).
        let roundtrip: GravityConfiguration =
            serde_json::from_str(&serde_json::to_string(&cfg).unwrap()).unwrap();
        assert_eq!(roundtrip, cfg);
    }

    #[test]
    fn test_srp_config_serde_default_occulting_bodies() {
        // Old serialized config without the field deserializes to [Earth]
        let json = r#"{"area":{"Value":1.0},"cr":{"Value":1.8},"eclipse_model":"Conical"}"#;
        let cfg: SolarRadiationPressureConfiguration = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.occulting_bodies, vec![OccultingBody::Earth]);
    }

    #[test]
    fn test_occulting_body_radius_and_naif_ids() {
        assert_eq!(OccultingBody::Earth.radius(), R_EARTH);
        assert_eq!(OccultingBody::Earth.naif_id(), 399);
        assert_eq!(OccultingBody::Earth.naif_position_id(), 399);

        assert_eq!(OccultingBody::Moon.radius(), R_MOON);
        assert_eq!(OccultingBody::Moon.naif_id(), 301);
        assert_eq!(OccultingBody::Moon.naif_position_id(), 301);

        assert_eq!(OccultingBody::Mars.radius(), R_MARS);
        assert_eq!(OccultingBody::Mars.naif_id(), 499);
        assert_eq!(OccultingBody::Mars.naif_position_id(), 499);

        let custom = OccultingBody::Custom {
            name: "Europa".to_string(),
            naif_id: 502,
            radius: 1560.8e3,
        };
        assert_eq!(custom.radius(), 1560.8e3);
        assert_eq!(custom.naif_id(), 502);
        assert_eq!(custom.naif_position_id(), 502);
    }

    // =========================================================================
    // central_body / for_body / defaults / validate
    // =========================================================================

    #[test]
    fn test_default_config_central_body_earth_and_valid() {
        let cfg = ForceModelConfig::default();
        assert_eq!(cfg.central_body, CentralBody::Earth);
        cfg.validate().unwrap();
    }

    #[test]
    fn test_serde_backward_compat_missing_central_body() {
        let mut v = serde_json::to_value(ForceModelConfig::default()).unwrap();
        v.as_object_mut().unwrap().remove("central_body");
        let cfg: ForceModelConfig = serde_json::from_value(v).unwrap();
        assert_eq!(cfg.central_body, CentralBody::Earth);
    }

    #[test]
    fn test_lunar_mars_cislunar_defaults_valid() {
        for cfg in [
            ForceModelConfig::lunar_default(),
            ForceModelConfig::mars_default(),
            ForceModelConfig::cislunar_default(),
        ] {
            cfg.validate().unwrap();
        }
        assert_eq!(
            ForceModelConfig::lunar_default().central_body,
            CentralBody::Moon
        );
        assert!(ForceModelConfig::lunar_default().drag.is_none());
    }

    #[test]
    fn test_for_body_constructs_expected_fields() {
        let cfg = ForceModelConfig::for_body(
            CentralBody::Mars,
            GravityConfiguration::PointMass,
            None,
            None,
            None,
            true,
            Some(ParameterSource::Value(500.0)),
        );
        assert_eq!(cfg.central_body, CentralBody::Mars);
        assert_eq!(cfg.gravity, GravityConfiguration::PointMass);
        assert!(cfg.drag.is_none());
        assert!(cfg.srp.is_none());
        assert!(cfg.third_body.is_none());
        assert!(cfg.relativity);
        assert_eq!(cfg.mass, Some(ParameterSource::Value(500.0)));
        assert_eq!(
            cfg.frame_transform,
            FrameTransformationModel::FullEarthRotation
        );
    }

    #[test]
    fn test_validate_rejects_harris_priester_non_earth() {
        let cfg = ForceModelConfig {
            tides: None,
            central_body: CentralBody::Moon,
            drag: Some(DragConfiguration {
                model: AtmosphericModel::HarrisPriester,
                area: ParameterSource::ParameterIndex(1),
                cd: ParameterSource::ParameterIndex(2),
                body: None,
            }),
            gravity: GravityConfiguration::PointMass,
            srp: None,
            third_body: None,
            relativity: false,
            mass: None,
            frame_transform: FrameTransformationModel::default(),
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("HarrisPriester"), "{err}");
        assert!(err.contains("Moon"), "{err}");
    }

    #[test]
    fn test_validate_rejects_nrlmsise00_non_earth() {
        let cfg = ForceModelConfig {
            tides: None,
            central_body: CentralBody::Mars,
            drag: Some(DragConfiguration {
                model: AtmosphericModel::NRLMSISE00,
                area: ParameterSource::ParameterIndex(1),
                cd: ParameterSource::ParameterIndex(2),
                body: None,
            }),
            gravity: GravityConfiguration::PointMass,
            srp: None,
            third_body: None,
            relativity: false,
            mass: None,
            frame_transform: FrameTransformationModel::default(),
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("NRLMSISE00"), "{err}");
        assert!(err.contains("Mars"), "{err}");
    }

    #[test]
    fn test_validate_rejects_earth_zonal_non_earth() {
        let cfg = ForceModelConfig {
            tides: None,
            central_body: CentralBody::Moon,
            gravity: GravityConfiguration::EarthZonal {
                degree: ZonalHarmonicsDegree::J2,
            },
            drag: None,
            srp: None,
            third_body: None,
            relativity: false,
            mass: None,
            frame_transform: FrameTransformationModel::default(),
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("EarthZonal"), "{err}");
        assert!(err.contains("Moon"), "{err}");
    }

    #[test]
    fn test_validate_rejects_earth_rotation_only_non_earth() {
        let cfg = ForceModelConfig {
            tides: None,
            central_body: CentralBody::Mars,
            gravity: GravityConfiguration::PointMass,
            drag: None,
            srp: None,
            third_body: None,
            relativity: false,
            mass: None,
            frame_transform: FrameTransformationModel::EarthRotationOnly,
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("EarthRotationOnly"), "{err}");
        assert!(err.contains("Mars"), "{err}");
    }

    #[test]
    fn test_validate_rejects_tides_non_earth() {
        let cfg = ForceModelConfig {
            tides: Some(TidesConfiguration::default()),
            central_body: CentralBody::Moon,
            gravity: GravityConfiguration::PointMass,
            drag: None,
            srp: None,
            third_body: None,
            relativity: false,
            mass: None,
            frame_transform: FrameTransformationModel::default(),
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("TidesConfiguration"), "{err}");
        assert!(err.contains("Moon"), "{err}");
    }

    #[test]
    fn test_validate_rejects_low_precision_ephemeris_non_earth() {
        let cfg = ForceModelConfig {
            tides: None,
            central_body: CentralBody::Moon,
            gravity: GravityConfiguration::PointMass,
            drag: None,
            srp: None,
            third_body: Some(vec![ThirdBodyConfiguration {
                body: ThirdBody::Sun,
                ephemeris_source: EphemerisSource::LowPrecision,
                gravity: GravityConfiguration::PointMass,
            }]),
            relativity: false,
            mass: None,
            frame_transform: FrameTransformationModel::default(),
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("LowPrecision"), "{err}");
        assert!(err.contains("Moon"), "{err}");
    }

    #[test]
    fn test_validate_allows_low_precision_earth_sun_moon() {
        let cfg = ForceModelConfig {
            tides: None,
            central_body: CentralBody::Earth,
            gravity: GravityConfiguration::PointMass,
            drag: None,
            srp: None,
            third_body: Some(vec![
                ThirdBodyConfiguration {
                    body: ThirdBody::Sun,
                    ephemeris_source: EphemerisSource::LowPrecision,
                    gravity: GravityConfiguration::PointMass,
                },
                ThirdBodyConfiguration {
                    body: ThirdBody::Moon,
                    ephemeris_source: EphemerisSource::LowPrecision,
                    gravity: GravityConfiguration::PointMass,
                },
            ]),
            relativity: false,
            mass: None,
            frame_transform: FrameTransformationModel::default(),
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_low_precision_earth_planet() {
        let cfg = ForceModelConfig {
            tides: None,
            central_body: CentralBody::Earth,
            gravity: GravityConfiguration::PointMass,
            drag: None,
            srp: None,
            third_body: Some(vec![ThirdBodyConfiguration {
                body: ThirdBody::Mars,
                ephemeris_source: EphemerisSource::LowPrecision,
                gravity: GravityConfiguration::PointMass,
            }]),
            relativity: false,
            mass: None,
            frame_transform: FrameTransformationModel::default(),
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("LowPrecision"), "{err}");
        assert!(err.contains("Mars"), "{err}");
    }

    #[test]
    fn test_validate_rejects_third_body_same_naif_id_as_central_body() {
        let cfg = ForceModelConfig {
            tides: None,
            central_body: CentralBody::Earth,
            gravity: GravityConfiguration::PointMass,
            drag: None,
            srp: None,
            third_body: Some(vec![ThirdBody::Earth.into()]),
            relativity: false,
            mass: None,
            frame_transform: FrameTransformationModel::default(),
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("NAIF"), "{err}");
    }

    #[test]
    fn test_validate_rejects_spherical_harmonic_barycenter() {
        let cfg = ForceModelConfig {
            tides: None,
            central_body: CentralBody::EMB,
            gravity: GravityConfiguration::SphericalHarmonic {
                source: GravityModelSource::default(),
                degree: 20,
                order: 20,
                parallel: ParallelMode::Auto,
            },
            drag: None,
            srp: None,
            third_body: None,
            relativity: false,
            mass: None,
            frame_transform: FrameTransformationModel::default(),
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("SphericalHarmonic"), "{err}");
        assert!(err.contains("Earth-Moon Barycenter"), "{err}");
    }

    #[test]
    fn test_validate_rejects_drag_barycenter() {
        let cfg = ForceModelConfig {
            tides: None,
            central_body: CentralBody::SSB,
            gravity: GravityConfiguration::PointMass,
            drag: Some(DragConfiguration {
                model: AtmosphericModel::Exponential {
                    scale_height: 10e3,
                    rho0: 1.0,
                    h0: 0.0,
                },
                area: ParameterSource::ParameterIndex(1),
                cd: ParameterSource::ParameterIndex(2),
                body: None,
            }),
            srp: None,
            third_body: None,
            relativity: false,
            mass: None,
            frame_transform: FrameTransformationModel::default(),
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.to_lowercase().contains("drag"), "{err}");
        assert!(err.contains("Solar System Barycenter"), "{err}");
    }

    #[test]
    fn test_validate_rejects_drag_without_radius_and_omega() {
        let custom = CentralBody::Custom(crate::propagators::central_body::CustomBody {
            name: "TestBody".to_string(),
            naif_id: 12345,
            gm: 1.0e10,
            radius: None,
            omega: None,
            fixed_frame: None,
        });
        let cfg = ForceModelConfig {
            tides: None,
            central_body: custom,
            gravity: GravityConfiguration::PointMass,
            drag: Some(DragConfiguration {
                model: AtmosphericModel::Exponential {
                    scale_height: 10e3,
                    rho0: 1.0,
                    h0: 0.0,
                },
                area: ParameterSource::ParameterIndex(1),
                cd: ParameterSource::ParameterIndex(2),
                body: None,
            }),
            srp: None,
            third_body: None,
            relativity: false,
            mass: None,
            frame_transform: FrameTransformationModel::default(),
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.to_lowercase().contains("drag"), "{err}");
        assert!(err.contains("TestBody"), "{err}");
    }

    #[test]
    fn test_validate_rejects_custom_spherical_harmonic_without_fixed_frame() {
        let custom = CentralBody::Custom(crate::propagators::central_body::CustomBody {
            name: "TestBody".to_string(),
            naif_id: 12345,
            gm: 1.0e10,
            radius: Some(1.0e6),
            omega: None,
            fixed_frame: None,
        });
        let cfg = ForceModelConfig {
            tides: None,
            central_body: custom,
            gravity: GravityConfiguration::SphericalHarmonic {
                source: GravityModelSource::default(),
                degree: 20,
                order: 20,
                parallel: ParallelMode::Auto,
            },
            drag: None,
            srp: None,
            third_body: None,
            relativity: false,
            mass: None,
            frame_transform: FrameTransformationModel::default(),
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("SphericalHarmonic"), "{err}");
        assert!(err.contains("TestBody"), "{err}");
    }

    #[test]
    #[cfg_attr(not(feature = "integration"), ignore)]
    #[serial_test::serial]
    fn test_lunar_mars_gravity_models_downloadable() {
        use crate::orbit_dynamics::gravity::GravityModel;

        for cfg in [
            ForceModelConfig::lunar_default(),
            ForceModelConfig::mars_default(),
        ] {
            if let GravityConfiguration::SphericalHarmonic {
                source: GravityModelSource::ModelType(mt),
                ..
            } = &cfg.gravity
            {
                GravityModel::from_model_type(mt).unwrap();
            } else {
                panic!("expected SH gravity");
            }
        }
    }

    #[test]
    #[serial_test::parallel]
    fn test_third_body_barycenter_planet_split() {
        use crate::constants::{
            GM_JUPITER, GM_JUPITER_SYSTEM, GM_MARS, GM_MARS_SYSTEM, GM_NEPTUNE, GM_NEPTUNE_SYSTEM,
            GM_SATURN, GM_SATURN_SYSTEM, GM_URANUS, GM_URANUS_SYSTEM,
        };
        // Planet-center variants
        assert_eq!(ThirdBody::Mars.naif_id(), 499);
        assert_eq!(ThirdBody::Mars.gm(), GM_MARS);
        assert_eq!(ThirdBody::Jupiter.naif_id(), 599);
        assert_eq!(ThirdBody::Jupiter.gm(), GM_JUPITER);
        assert_eq!(ThirdBody::Saturn.naif_id(), 699);
        assert_eq!(ThirdBody::Saturn.gm(), GM_SATURN);
        assert_eq!(ThirdBody::Uranus.naif_id(), 799);
        assert_eq!(ThirdBody::Uranus.gm(), GM_URANUS);
        assert_eq!(ThirdBody::Neptune.naif_id(), 899);
        assert_eq!(ThirdBody::Neptune.gm(), GM_NEPTUNE);
        // Barycenter variants
        assert_eq!(ThirdBody::MarsBarycenter.naif_id(), 4);
        assert_eq!(ThirdBody::MarsBarycenter.gm(), GM_MARS_SYSTEM);
        assert_eq!(ThirdBody::JupiterBarycenter.naif_id(), 5);
        assert_eq!(ThirdBody::JupiterBarycenter.gm(), GM_JUPITER_SYSTEM);
        assert_eq!(ThirdBody::SaturnBarycenter.naif_id(), 6);
        assert_eq!(ThirdBody::SaturnBarycenter.gm(), GM_SATURN_SYSTEM);
        assert_eq!(ThirdBody::UranusBarycenter.naif_id(), 7);
        assert_eq!(ThirdBody::UranusBarycenter.gm(), GM_URANUS_SYSTEM);
        assert_eq!(ThirdBody::NeptuneBarycenter.naif_id(), 8);
        assert_eq!(ThirdBody::NeptuneBarycenter.gm(), GM_NEPTUNE_SYSTEM);
    }

    #[test]
    #[serial_test::parallel]
    fn test_third_body_as_central_body_and_fixed_frame() {
        use crate::frames::ReferenceFrame;
        assert_eq!(ThirdBody::Earth.as_central_body(), Some(CentralBody::Earth));
        assert_eq!(ThirdBody::Moon.as_central_body(), Some(CentralBody::Moon));
        assert_eq!(ThirdBody::Mars.as_central_body(), Some(CentralBody::Mars));
        assert_eq!(
            ThirdBody::Earth.body_fixed_frame(),
            Some(ReferenceFrame::ITRF)
        );
        assert_eq!(
            ThirdBody::Moon.body_fixed_frame(),
            Some(ReferenceFrame::LFPA)
        );
        assert_eq!(
            ThirdBody::Mars.body_fixed_frame(),
            Some(ReferenceFrame::MCMF)
        );
        // Planet centers resolve through from_naif_id (IAU frames)
        assert!(ThirdBody::Jupiter.body_fixed_frame().is_some());
        // Barycenters and Custom bodies have no body-fixed frame
        assert!(ThirdBody::MarsBarycenter.as_central_body().is_none());
        assert!(ThirdBody::JupiterBarycenter.body_fixed_frame().is_none());
        let custom = ThirdBody::Custom {
            name: "Ceres".to_string(),
            naif_id: 2000001,
            gm: 6.26325e10,
        };
        assert!(custom.as_central_body().is_none());
        assert!(custom.body_fixed_frame().is_none());
    }

    #[test]
    #[serial_test::parallel]
    fn test_validate_third_body_gravity_rules() {
        // EarthZonal on a non-Earth third body is rejected
        let mut config = ForceModelConfig::cislunar_default();
        config.third_body = Some(vec![ThirdBodyConfiguration {
            gravity: GravityConfiguration::EarthZonal {
                degree: ZonalHarmonicsDegree::J2,
            },
            ..ThirdBody::Moon.into()
        }]);
        assert!(config.validate().is_err());

        // EarthZonal on ThirdBody::Earth is accepted
        config.third_body = Some(vec![ThirdBodyConfiguration {
            gravity: GravityConfiguration::EarthZonal {
                degree: ZonalHarmonicsDegree::J2,
            },
            ..ThirdBody::Earth.into()
        }]);
        assert!(config.validate().is_ok());

        // SphericalHarmonic on a barycenter variant is rejected (no fixed frame)
        config.third_body = Some(vec![ThirdBodyConfiguration {
            gravity: GravityConfiguration::SphericalHarmonic {
                source: GravityModelSource::default(),
                degree: 8,
                order: 8,
                parallel: ParallelMode::Never,
            },
            ..ThirdBody::JupiterBarycenter.into()
        }]);
        assert!(config.validate().is_err());

        // SphericalHarmonic on Earth as a third body is accepted
        config.third_body = Some(vec![ThirdBodyConfiguration {
            gravity: GravityConfiguration::SphericalHarmonic {
                source: GravityModelSource::default(),
                degree: 8,
                order: 8,
                parallel: ParallelMode::Never,
            },
            ..ThirdBody::Earth.into()
        }]);
        assert!(config.validate().is_ok());

        // SphericalHarmonic on a Custom third body is rejected
        config.third_body = Some(vec![ThirdBodyConfiguration {
            gravity: GravityConfiguration::SphericalHarmonic {
                source: GravityModelSource::default(),
                degree: 4,
                order: 4,
                parallel: ParallelMode::Never,
            },
            ..ThirdBody::Custom {
                name: "Ceres".to_string(),
                naif_id: 2000001,
                gm: 6.26325e10,
            }
            .into()
        }]);
        assert!(config.validate().is_err());
    }

    #[test]
    #[serial_test::parallel]
    fn test_validate_attributed_drag_body() {
        // EMB central + NRLMSISE-00 drag attributed to Earth: accepted
        let mut config = ForceModelConfig::cislunar_default();
        config.mass = Some(ParameterSource::Value(1000.0));
        config.drag = Some(DragConfiguration {
            model: AtmosphericModel::NRLMSISE00,
            area: ParameterSource::Value(10.0),
            cd: ParameterSource::Value(2.2),
            body: Some(CentralBody::Earth),
        });
        assert!(config.validate().is_ok());

        // EMB central + drag with no attributed body: rejected (barycenter)
        config.drag.as_mut().unwrap().body = None;
        assert!(config.validate().is_err());

        // Harris-Priester attributed to the Moon: rejected (Earth-only model)
        config.drag = Some(DragConfiguration {
            model: AtmosphericModel::HarrisPriester,
            area: ParameterSource::Value(10.0),
            cd: ParameterSource::Value(2.2),
            body: Some(CentralBody::Moon),
        });
        assert!(config.validate().is_err());

        // Drag attributed to a barycenter: rejected (no radius/spin)
        config.drag = Some(DragConfiguration {
            model: AtmosphericModel::Exponential {
                scale_height: 8500.0,
                rho0: 1.225,
                h0: 0.0,
            },
            area: ParameterSource::Value(10.0),
            cd: ParameterSource::Value(2.2),
            body: Some(CentralBody::EMB),
        });
        assert!(config.validate().is_err());

        // Earth central with NRLMSISE-00, no attribution: still accepted
        let config = ForceModelConfig::leo_default();
        assert!(config.validate().is_ok());
    }

    #[test]
    #[serial_test::parallel]
    fn test_drag_configuration_body_serde_default() {
        // Configurations serialized before the field existed load as
        // body: None (drag about the central body).
        let json = r#"{
            "model": "HarrisPriester",
            "area": {"Value": 10.0},
            "cd": {"Value": 2.2}
        }"#;
        let drag: DragConfiguration = serde_json::from_str(json).unwrap();
        assert!(drag.body.is_none());
    }

    #[test]
    #[serial_test::parallel]
    fn test_mars_system_gm_is_not_the_component_sum() {
        use crate::constants::{GM_DEIMOS, GM_MARS, GM_MARS_SYSTEM, GM_PHOBOS};

        // gm_de440.tpc combines two estimation solutions: the DE440
        // planetary-solution barycenter GM (BODY4_GM, via Konopliv et al.
        // 2016) and the older Horizons satellite-solution body GMs
        // (BODY499/401/402). Their sum therefore does NOT reproduce the
        // system value; this guard documents the known ~1.4e6 m^3/s^2
        // offset so it is not "fixed" into an inconsistency with the DE
        // kernels' barycenter dynamics.
        let component_sum = GM_MARS + GM_PHOBOS + GM_DEIMOS;
        let diff = GM_MARS_SYSTEM - component_sum;
        assert!(
            diff.abs() > 1.0e6,
            "expected the known solution-epoch offset, got {diff} m^3/s^2"
        );
        assert!(
            diff.abs() < 2.0e6,
            "system/component mismatch larger than the documented offset: {diff} m^3/s^2"
        );
    }

    #[test]
    #[serial_test::parallel]
    fn test_gravity_configuration_zero() {
        // cislunar_default uses the explicit no-central-gravity variant
        let config = ForceModelConfig::cislunar_default();
        assert_eq!(config.gravity, GravityConfiguration::Zero);
        assert!(config.validate().is_ok());

        // Zero on a third-body entry disables the entry and is rejected
        let mut config = ForceModelConfig::cislunar_default();
        config.third_body = Some(vec![ThirdBodyConfiguration {
            gravity: GravityConfiguration::Zero,
            ..ThirdBody::Sun.into()
        }]);
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("Zero"), "{err}");

        // Serde round trip
        let json = serde_json::to_string(&GravityConfiguration::Zero).unwrap();
        assert_eq!(
            serde_json::from_str::<GravityConfiguration>(&json).unwrap(),
            GravityConfiguration::Zero
        );
    }

    #[test]
    #[serial_test::parallel]
    fn test_third_body_configuration_from_body() {
        let cfg = ThirdBodyConfiguration::from(ThirdBody::Sun);
        assert_eq!(cfg.body, ThirdBody::Sun);
        assert_eq!(cfg.ephemeris_source, EphemerisSource::DE440s);
        assert_eq!(cfg.gravity, GravityConfiguration::PointMass);
        assert_eq!(
            ThirdBodyConfiguration::new(ThirdBody::Moon).body,
            ThirdBody::Moon
        );
    }

    #[test]
    #[serial_test::parallel]
    fn test_third_bodies_serde_round_trip() {
        let config = ForceModelConfig {
            third_body: Some(vec![
                ThirdBody::Sun.into(),
                ThirdBodyConfiguration {
                    body: ThirdBody::Earth,
                    ephemeris_source: EphemerisSource::DE440,
                    gravity: GravityConfiguration::EarthZonal {
                        degree: ZonalHarmonicsDegree::J2,
                    },
                },
            ]),
            ..ForceModelConfig::two_body_gravity()
        };
        let json = serde_json::to_string(&config).unwrap();
        let back: ForceModelConfig = serde_json::from_str(&json).unwrap();
        let tbs = back.third_body.unwrap();
        assert_eq!(tbs.len(), 2);
        assert_eq!(tbs[0].body, ThirdBody::Sun);
        assert_eq!(tbs[0].gravity, GravityConfiguration::PointMass);
        assert_eq!(tbs[1].ephemeris_source, EphemerisSource::DE440);
    }

    #[test]
    #[serial_test::parallel]
    fn test_third_bodies_deserialize_one_or_many_and_bare_bodies() {
        // Single object coerces to a one-element vec
        let json = r#"{
            "gravity": "PointMass", "drag": null, "srp": null,
            "relativity": false, "mass": null,
            "third_body": {"body": "Sun"}
        }"#;
        let config: ForceModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.third_body.unwrap().len(), 1);

        // Bare bodies inside an array coerce to default configurations
        let json = r#"{
            "gravity": "PointMass", "drag": null, "srp": null,
            "relativity": false, "mass": null,
            "third_body": ["Sun", {"body": "Moon"}]
        }"#;
        let config: ForceModelConfig = serde_json::from_str(json).unwrap();
        let tbs = config.third_body.unwrap();
        assert_eq!(tbs[0].body, ThirdBody::Sun);
        assert_eq!(tbs[0].ephemeris_source, EphemerisSource::DE440s);
        assert_eq!(tbs[0].gravity, GravityConfiguration::PointMass);
        assert_eq!(tbs[1].body, ThirdBody::Moon);

        // Omitted field means None
        let json = r#"{
            "gravity": "PointMass", "drag": null, "srp": null,
            "relativity": false, "mass": null
        }"#;
        let config: ForceModelConfig = serde_json::from_str(json).unwrap();
        assert!(config.third_body.is_none());
    }

    #[test]
    #[serial_test::parallel]
    fn test_legacy_third_body_field_migrates_on_load() {
        // Configurations serialized before the per-body flattening carry a
        // `third_body` object; it migrates to `third_body` entries with the
        // legacy shared ephemeris source and point-mass gravity.
        let json = r#"{
            "gravity": "PointMass", "drag": null, "srp": null,
            "relativity": false, "mass": null,
            "third_body": {"ephemeris_source": "DE440", "bodies": ["Sun", "Moon"]}
        }"#;
        let config: ForceModelConfig = serde_json::from_str(json).unwrap();
        let tbs = config.third_body.unwrap();
        assert_eq!(tbs.len(), 2);
        assert_eq!(tbs[0].body, ThirdBody::Sun);
        assert_eq!(tbs[0].ephemeris_source, EphemerisSource::DE440);
        assert_eq!(tbs[0].gravity, GravityConfiguration::PointMass);
        assert_eq!(tbs[1].body, ThirdBody::Moon);

        // Pre-split planet names meant the system barycenters; the migration
        // preserves that physics.
        let json = r#"{
            "gravity": "PointMass", "drag": null, "srp": null,
            "relativity": false, "mass": null,
            "third_body": {"ephemeris_source": "DE440s", "bodies": ["Mars", "Jupiter", "Venus"]}
        }"#;
        let config: ForceModelConfig = serde_json::from_str(json).unwrap();
        let tbs = config.third_body.unwrap();
        assert_eq!(tbs[0].body, ThirdBody::MarsBarycenter);
        assert_eq!(tbs[1].body, ThirdBody::JupiterBarycenter);
        assert_eq!(tbs[2].body, ThirdBody::Venus);

        // A null legacy field is treated as absent.
        let json = r#"{
            "gravity": "PointMass", "drag": null, "srp": null,
            "relativity": false, "mass": null, "third_body": null
        }"#;
        let config: ForceModelConfig = serde_json::from_str(json).unwrap();
        assert!(config.third_body.is_none());
    }

    #[test]
    #[serial_test::parallel]
    fn test_high_fidelity_uses_barycenter_variants() {
        let config = ForceModelConfig::high_fidelity();
        let tb = config.third_body.unwrap();
        let bodies: Vec<ThirdBody> = tb.iter().map(|e| e.body.clone()).collect();
        assert!(bodies.contains(&ThirdBody::MarsBarycenter));
        assert!(bodies.contains(&ThirdBody::JupiterBarycenter));
        assert!(bodies.contains(&ThirdBody::SaturnBarycenter));
        assert!(bodies.contains(&ThirdBody::UranusBarycenter));
        assert!(bodies.contains(&ThirdBody::NeptuneBarycenter));
        assert!(!bodies.contains(&ThirdBody::Mars));
    }

    #[test]
    #[serial_test::parallel]
    fn test_validate_rejects_mars_bodies_for_mars_central() {
        // Both the planet (same NAIF ID as the central body) and the
        // barycenter (inside the planet) must be rejected for a
        // Mars-centered propagation.
        for body in [ThirdBody::Mars, ThirdBody::MarsBarycenter] {
            let config = ForceModelConfig {
                central_body: CentralBody::Mars,
                third_body: Some(vec![body.into()]),
                ..ForceModelConfig::two_body_gravity()
            };
            assert!(config.validate().is_err());
        }
    }

    // =========================================================================
    // validate_gravity_model_body (issue #400)
    // =========================================================================

    #[test]
    #[serial_test::parallel]
    fn test_validate_rejects_egm2008_on_mars_central() {
        // A bare SphericalHarmonic configuration (no explicit model_type)
        // silently defaults to Earth's EGM2008_120; paired with a Mars
        // central body this must now error instead of silently evaluating
        // Earth's field at Mars-orbit radii.
        let config = ForceModelConfig {
            central_body: CentralBody::Mars,
            gravity: GravityConfiguration::SphericalHarmonic {
                source: GravityModelSource::default(),
                degree: 8,
                order: 8,
                parallel: ParallelMode::Auto,
            },
            ..ForceModelConfig::two_body_gravity()
        };
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("Earth"), "{err}");
        assert!(err.contains("Mars"), "{err}");
    }

    #[test]
    #[serial_test::parallel]
    fn test_validate_rejects_ggm05s_and_jgm3_on_non_earth_central() {
        for model_type in [GravityModelType::GGM05S, GravityModelType::JGM3] {
            let config = ForceModelConfig {
                central_body: CentralBody::Moon,
                gravity: GravityConfiguration::SphericalHarmonic {
                    source: GravityModelSource::ModelType(model_type),
                    degree: 8,
                    order: 8,
                    parallel: ParallelMode::Auto,
                },
                ..ForceModelConfig::two_body_gravity()
            };
            let err = config.validate().unwrap_err().to_string();
            assert!(err.contains("Earth"), "{err}");
            assert!(err.contains("Moon"), "{err}");
        }
    }

    #[test]
    #[serial_test::parallel]
    fn test_validate_rejects_icgem_body_mismatch() {
        // ICGEMModel's body is known statically (no load/network needed to
        // detect the mismatch): Mars model paired with a Moon central body.
        let config = ForceModelConfig {
            central_body: CentralBody::Moon,
            gravity: GravityConfiguration::SphericalHarmonic {
                source: GravityModelSource::ModelType(GravityModelType::ICGEMModel {
                    body: ICGEMBody::Mars,
                    name: "ggm2bc80".to_string(),
                }),
                degree: 8,
                order: 8,
                parallel: ParallelMode::Auto,
            },
            ..ForceModelConfig::two_body_gravity()
        };
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("Mars"), "{err}");
        assert!(err.contains("Moon"), "{err}");
    }

    #[test]
    #[serial_test::parallel]
    fn test_validate_allows_matching_icgem_body() {
        // Same body (Moon), different model name than lunar_default() — the
        // check is identity-based, not tied to a specific packaged name.
        let config = ForceModelConfig {
            central_body: CentralBody::Moon,
            gravity: GravityConfiguration::SphericalHarmonic {
                source: GravityModelSource::ModelType(GravityModelType::ICGEMModel {
                    body: ICGEMBody::Moon,
                    name: "GRGM1200B".to_string(),
                }),
                degree: 8,
                order: 8,
                parallel: ParallelMode::Auto,
            },
            ..ForceModelConfig::two_body_gravity()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    #[serial_test::parallel]
    fn test_validate_allows_default_configs_matching_central_body() {
        // Every named constructor pairs its packaged/ICGEM gravity model with
        // the central body it belongs to; none of these should require a
        // network fetch to validate (Earth packaged models, and Moon/Mars
        // ICGEM models, all resolve statically).
        for config in [
            ForceModelConfig::default(),
            ForceModelConfig::high_fidelity(),
            ForceModelConfig::earth_gravity(),
            ForceModelConfig::two_body_gravity(),
            ForceModelConfig::conservative_forces(),
            ForceModelConfig::leo_default(),
            ForceModelConfig::geo_default(),
            ForceModelConfig::lunar_default(),
            ForceModelConfig::mars_default(),
            ForceModelConfig::cislunar_default(),
        ] {
            assert!(config.validate().is_ok());
        }
    }

    #[test]
    #[serial_test::parallel]
    fn test_validate_gravity_model_body_from_file_header_match_and_mismatch() {
        // GravityModelType::FromFile has no statically known body, so the
        // check falls back to loading the model and comparing its header GM
        // and reference radius. Uses a local Moon (GRGM660PRIM) header
        // fixture — no network access required.
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = std::path::Path::new(&manifest_dir)
            .join("test_assets")
            .join("grgm660prim_header_sample.gfc");
        let model_type = GravityModelType::from_file(&path).unwrap();

        // Matching central body (Moon): header GM/radius agree within tolerance.
        let matching = ForceModelConfig {
            central_body: CentralBody::Moon,
            gravity: GravityConfiguration::SphericalHarmonic {
                source: GravityModelSource::ModelType(model_type.clone()),
                degree: 2,
                order: 2,
                parallel: ParallelMode::Auto,
            },
            ..ForceModelConfig::two_body_gravity()
        };
        assert!(matching.validate().is_ok());

        // Mismatched central body (Mars): header GM/radius are Moon-scale,
        // well outside the 1% tolerance against Mars's GM/radius.
        let mismatched = ForceModelConfig {
            central_body: CentralBody::Mars,
            gravity: GravityConfiguration::SphericalHarmonic {
                source: GravityModelSource::ModelType(model_type),
                degree: 2,
                order: 2,
                parallel: ParallelMode::Auto,
            },
            ..ForceModelConfig::two_body_gravity()
        };
        let err = mismatched.validate().unwrap_err().to_string();
        assert!(err.contains("Mars"), "{err}");
    }

    #[test]
    #[serial_test::parallel]
    fn test_validate_rejects_from_file_on_custom_body_with_zero_gm() {
        // A `Custom` central body has no dedicated barycenter check (unlike
        // `EMB`/`SSB`), so a `gm: 0.0` body must still be rejected here rather
        // than silently skipping the GM/radius comparison (which would
        // otherwise let any FromFile/ICGEM Venus/Ceres/Other model "pass" a
        // massless custom body with no check at all).
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = std::path::Path::new(&manifest_dir)
            .join("test_assets")
            .join("grgm660prim_header_sample.gfc");
        let model_type = GravityModelType::from_file(&path).unwrap();

        let custom = CentralBody::Custom(crate::propagators::central_body::CustomBody {
            name: "Massless".to_string(),
            naif_id: -1,
            gm: 0.0,
            radius: None,
            omega: None,
            fixed_frame: Some(crate::frames::ReferenceFrame::BodyFixedIAU(-1)),
        });
        let config = ForceModelConfig {
            central_body: custom,
            gravity: GravityConfiguration::SphericalHarmonic {
                source: GravityModelSource::ModelType(model_type),
                degree: 2,
                order: 2,
                parallel: ParallelMode::Auto,
            },
            ..ForceModelConfig::two_body_gravity()
        };
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("non-positive GM"), "{err}");
    }
}
