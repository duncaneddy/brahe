/*!
 * The `CentralBody` / `CustomBody` abstraction identifies which body an
 * orbit is propagated relative to, and bundles the physical properties
 * (GM, radius, spin rate, body-fixed frame) a force model needs to
 * evaluate gravity, drag, and third-body perturbations for that body.
 *
 * Five bodies are built in — `Earth`, `Moon`, `Mars`, and the two
 * barycenters `EMB` (Earth-Moon) and `SSB` (Solar System) — because they
 * have dedicated named frames elsewhere in [`crate::frames`]
 * (`GCRF`/`ITRF`, `LCI`/`LFPA`, `MCI`/`MCMF`, `EMBI`, `SSBI`). Any other
 * body is represented as [`CentralBody::Custom`], which carries its own
 * [`CustomBody`] record. [`CentralBody::from_naif_id`] constructs a
 * built-in variant for the NAIF IDs above and a pre-populated `Custom`
 * record (drawn from an embedded table) for a fixed set of other
 * commonly used bodies.
 *
 * # NAIF ID / origin table
 *
 * | Body    | NAIF ID | Inertial frame | Fixed frame       |
 * |---------|---------|-----------------|-------------------|
 * | Earth   | 399     | `GCRF`          | `ITRF`            |
 * | Moon    | 301     | `LCI`           | `LFPA`            |
 * | Mars    | 4       | `MCI`           | `MCMF`            |
 * | EMB     | 3       | `EMBI`          | none              |
 * | SSB     | 0       | `SSBI`          | none              |
 *
 * `Mars` uses NAIF ID **499** (the Mars body center), matching the
 * origin of the `MCI` frame. Ephemeris legs between the body center and
 * the Mars system barycenter (NAIF 4) resolve through the `mar099s`
 * satellite ephemeris kernel, which is auto-loaded on first use.
 * [`CentralBody::from_naif_id`] accepts both 4 and 499 and maps either
 * to `CentralBody::Mars` (the system barycenter sits inside the planet,
 * ~0.1-0.2 m from its center).
 */

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

use crate::constants::{
    GM_DEIMOS, GM_EARTH, GM_JUPITER, GM_MARS, GM_MERCURY, GM_MOON, GM_NEPTUNE, GM_PHOBOS,
    GM_SATURN, GM_SUN, GM_URANUS, GM_VENUS, OMEGA_EARTH, OMEGA_MARS, OMEGA_MOON, R_EARTH, R_MARS,
    R_MOON, R_SUN,
};
use crate::frames::{ReferenceFrame, iau_rotation_model_ids};
use crate::utils::BraheError;

/// A user-defined central body, for propagation about a body without a
/// dedicated [`CentralBody`] variant (e.g. a planet, moon, or asteroid
/// beyond `Earth`/`Moon`/`Mars`).
///
/// [`CentralBody::from_naif_id`] returns pre-populated `CustomBody`
/// records for a fixed set of commonly used bodies (see its
/// documentation for the embedded table and sources); construct one
/// directly for any other body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CustomBody {
    /// Human-readable name (e.g. `"Enceladus"`).
    pub name: String,
    /// NAIF ID of the body.
    ///
    /// For a body without a catalogued NAIF ID (e.g. a newly observed
    /// asteroid), self-assign a unique negative ID, mirroring NAIF's own
    /// convention for non-catalogued objects. The ID is used for frame
    /// identity and force-model validation; ephemeris queries against it
    /// will surface an SPK lookup error unless a kernel covering that ID
    /// is loaded, and a body-fixed frame can be supplied without any
    /// ephemeris via [`crate::frames::register_custom_frame`] and
    /// [`crate::frames::ReferenceFrame::BodyFixedCustom`].
    pub naif_id: i32,
    /// Gravitational parameter. Units: (m^3/s^2)
    pub gm: f64,
    /// Mean or equatorial radius, if known. Units: (m)
    pub radius: Option<f64>,
    /// Body-fixed axial spin vector, if known. Units: (rad/s)
    ///
    /// Not populated by [`CentralBody::from_naif_id`] — set this
    /// directly when a force model needs it (e.g. atmospheric
    /// co-rotation for drag).
    pub omega: Option<Vector3<f64>>,
    /// Body-fixed reference frame, required for spherical-harmonic
    /// gravity and body-fixed rotations. `None` if no rotation model is
    /// available.
    pub fixed_frame: Option<ReferenceFrame>,
}

/// The central body an orbit is propagated relative to.
///
/// `Earth`, `Moon`, and `Mars` are built in because they have dedicated
/// named inertial/fixed frame pairs elsewhere in [`crate::frames`]. `EMB`
/// and `SSB` are the Earth-Moon and Solar System barycenters — useful as
/// propagation origins for heliocentric or cislunar trajectories, but
/// they have no physical radius, spin, or fixed frame. Any other body is
/// represented as `Custom`.
///
/// See the module-level documentation for the NAIF ID / frame table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CentralBody {
    /// Earth (NAIF ID 399).
    Earth,
    /// Moon (NAIF ID 301).
    Moon,
    /// Mars system barycenter (NAIF ID 4).
    Mars,
    /// Earth-Moon barycenter (NAIF ID 3).
    EMB,
    /// Solar System barycenter (NAIF ID 0).
    SSB,
    /// A user-defined or table-derived body.
    Custom(CustomBody),
}

impl CentralBody {
    /// Gravitational parameter of the central body.
    ///
    /// # Returns
    /// - `gm`: Gravitational parameter. Units: (m^3/s^2). `0.0` for the
    ///   `EMB` and `SSB` barycenters, which have no mass of their own.
    ///
    /// # Examples
    /// ```
    /// use brahe::propagators::CentralBody;
    /// use brahe::constants::GM_EARTH;
    ///
    /// assert_eq!(CentralBody::Earth.gm(), GM_EARTH);
    /// assert_eq!(CentralBody::EMB.gm(), 0.0);
    /// ```
    pub fn gm(&self) -> f64 {
        match self {
            CentralBody::Earth => GM_EARTH,
            CentralBody::Moon => GM_MOON,
            CentralBody::Mars => GM_MARS,
            CentralBody::EMB => 0.0,
            CentralBody::SSB => 0.0,
            CentralBody::Custom(c) => c.gm,
        }
    }

    /// Mean or equatorial radius of the central body.
    ///
    /// # Returns
    /// - `radius`: Radius, if known. Units: (m). `None` for the `EMB`
    ///   and `SSB` barycenters.
    ///
    /// # Examples
    /// ```
    /// use brahe::propagators::CentralBody;
    /// use brahe::constants::R_EARTH;
    ///
    /// assert_eq!(CentralBody::Earth.radius(), Some(R_EARTH));
    /// assert_eq!(CentralBody::SSB.radius(), None);
    /// ```
    pub fn radius(&self) -> Option<f64> {
        match self {
            CentralBody::Earth => Some(R_EARTH),
            CentralBody::Moon => Some(R_MOON),
            CentralBody::Mars => Some(R_MARS),
            CentralBody::EMB => None,
            CentralBody::SSB => None,
            CentralBody::Custom(c) => c.radius,
        }
    }

    /// NAIF ID of the central body.
    ///
    /// # Returns
    /// - `naif_id`: NAIF ID. `399` for `Earth`, `301` for `Moon`, `499`
    ///   for `Mars` (body center — see module-level docs), `3` for
    ///   `EMB`, `0` for `SSB`.
    ///
    /// # Examples
    /// ```
    /// use brahe::propagators::CentralBody;
    ///
    /// assert_eq!(CentralBody::Earth.naif_id(), 399);
    /// assert_eq!(CentralBody::Mars.naif_id(), 499);
    /// ```
    pub fn naif_id(&self) -> i32 {
        match self {
            CentralBody::Earth => 399,
            CentralBody::Moon => 301,
            CentralBody::Mars => 499,
            CentralBody::EMB => 3,
            CentralBody::SSB => 0,
            CentralBody::Custom(c) => c.naif_id,
        }
    }

    /// Body-fixed axial spin vector of the central body.
    ///
    /// # Returns
    /// - `omega`: Spin vector expressed in the body's inertial frame,
    ///   if known. Units: (rad/s). `None` for the `EMB`/`SSB`
    ///   barycenters and for `Custom` bodies unless `omega` was set
    ///   explicitly.
    ///
    /// # Examples
    /// ```
    /// use brahe::propagators::CentralBody;
    /// use brahe::constants::OMEGA_EARTH;
    ///
    /// assert_eq!(
    ///     CentralBody::Earth.omega_vector(),
    ///     Some(nalgebra::Vector3::new(0.0, 0.0, OMEGA_EARTH))
    /// );
    /// assert_eq!(CentralBody::EMB.omega_vector(), None);
    /// ```
    pub fn omega_vector(&self) -> Option<Vector3<f64>> {
        match self {
            CentralBody::Earth => Some(Vector3::new(0.0, 0.0, OMEGA_EARTH)),
            CentralBody::Moon => Some(Vector3::new(0.0, 0.0, OMEGA_MOON)),
            CentralBody::Mars => Some(Vector3::new(0.0, 0.0, OMEGA_MARS)),
            CentralBody::EMB => None,
            CentralBody::SSB => None,
            CentralBody::Custom(c) => c.omega,
        }
    }

    /// ICRF-aligned inertial reference frame centered on this body.
    ///
    /// # Returns
    /// - `frame`: `GCRF` for `Earth`, `LCI` for `Moon`, `MCI` for
    ///   `Mars`, `EMBI` for `EMB`, `SSBI` for `SSB`, and
    ///   `BodyCenteredICRF(naif_id)` for `Custom` bodies.
    ///
    /// # Examples
    /// ```
    /// use brahe::propagators::CentralBody;
    /// use brahe::frames::ReferenceFrame;
    ///
    /// assert_eq!(CentralBody::Moon.inertial_frame(), ReferenceFrame::LCI);
    /// ```
    pub fn inertial_frame(&self) -> ReferenceFrame {
        match self {
            CentralBody::Earth => ReferenceFrame::GCRF,
            CentralBody::Moon => ReferenceFrame::LCI,
            CentralBody::Mars => ReferenceFrame::MCI,
            CentralBody::EMB => ReferenceFrame::EMBI,
            CentralBody::SSB => ReferenceFrame::SSBI,
            CentralBody::Custom(c) => ReferenceFrame::BodyCenteredICRF(c.naif_id),
        }
    }

    /// Body-fixed reference frame of this body, if one is defined.
    ///
    /// # Returns
    /// - `frame`: `ITRF` for `Earth`, `LFPA` for `Moon`, `MCMF` for
    ///   `Mars`, `None` for `EMB`/`SSB` (barycenters have no fixed
    ///   frame), and `custom.fixed_frame` for `Custom` bodies.
    ///
    /// # Examples
    /// ```
    /// use brahe::propagators::CentralBody;
    /// use brahe::frames::ReferenceFrame;
    ///
    /// assert_eq!(CentralBody::Moon.fixed_frame(), Some(ReferenceFrame::LFPA));
    /// assert_eq!(CentralBody::EMB.fixed_frame(), None);
    /// ```
    pub fn fixed_frame(&self) -> Option<ReferenceFrame> {
        match self {
            CentralBody::Earth => Some(ReferenceFrame::ITRF),
            CentralBody::Moon => Some(ReferenceFrame::LFPA),
            CentralBody::Mars => Some(ReferenceFrame::MCMF),
            CentralBody::EMB => None,
            CentralBody::SSB => None,
            CentralBody::Custom(c) => c.fixed_frame,
        }
    }

    /// Whether this central body is a barycenter (`EMB` or `SSB`).
    ///
    /// Barycenters have no mass, radius, spin, or fixed frame of their
    /// own — they are only useful as inertial propagation origins.
    ///
    /// # Returns
    /// - `is_barycenter`: `true` for `EMB`/`SSB`, `false` otherwise.
    ///
    /// # Examples
    /// ```
    /// use brahe::propagators::CentralBody;
    ///
    /// assert!(CentralBody::SSB.is_barycenter());
    /// assert!(!CentralBody::Earth.is_barycenter());
    /// ```
    pub fn is_barycenter(&self) -> bool {
        matches!(self, CentralBody::EMB | CentralBody::SSB)
    }

    /// Constructs a `CentralBody` from a NAIF ID.
    ///
    /// `399`, `301`, `4`/`499`, `3`, and `0` map to the built-in `Earth`,
    /// `Moon`, `Mars`, `EMB`, and `SSB` variants, respectively (see the
    /// module-level documentation for the `Mars` = 4 vs. 499 caveat).
    /// A fixed table of other commonly used bodies maps to a
    /// pre-populated `Custom` variant. Radii below were transcribed from
    /// [`pck00011.tpc`](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc)
    /// `BODYxxx_RADII`, fetched from the NAIF generic kernel archive.
    /// Planetary radii use the equatorial (`a`) semi-axis, since
    /// oblateness is significant for the giant planets; the small,
    /// markedly non-spherical moons (Phobos, Deimos, Enceladus, Titan)
    /// and Mercury (nearly spherical, but the reference triaxial figure
    /// is not) use the mean of the three `pck00011.tpc` semi-axes,
    /// `(a + b + c) / 3`, matching how these bodies are conventionally
    /// tabulated. GMs use the crate's existing named constants where one
    /// exists (see [`crate::constants::physical`] for each constant's
    /// own source); Enceladus and Titan have no such constant, so their
    /// GM is transcribed directly from
    /// [`gm_de440.tpc`](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc)
    /// `BODYxxx_GM` (also fetched from the NAIF generic kernel archive).
    /// Note that some named GM constants (sourced from Montenbruck &
    /// Gill, 2012) differ from `gm_de440.tpc`'s current DE440-based
    /// values at the 4th-5th significant digit; this is a pre-existing,
    /// out-of-scope discrepancy in those constants, not in this table.
    ///
    /// | NAIF ID | Body      | GM                                  | Radius source (`BODYxxx_RADII`, km)   | Radius (m) |
    /// |---------|-----------|--------------------------------------|----------------------------------------|------------|
    /// | 199     | Mercury   | `GM_MERCURY`                          | (2440.53, 2440.53, 2438.26), mean       | 2,439,773.33 |
    /// | 299     | Venus     | `GM_VENUS`                            | (6051.8, 6051.8, 6051.8), mean          | 6,051,800.0 |
    /// | 599     | Jupiter   | `GM_JUPITER`                          | (71492, 71492, 66854), equatorial       | 71,492,000.0 |
    /// | 699     | Saturn    | `GM_SATURN`                           | (60268, 60268, 54364), equatorial       | 60,268,000.0 |
    /// | 799     | Uranus    | `GM_URANUS`                           | (25559, 25559, 24973), equatorial       | 25,559,000.0 |
    /// | 899     | Neptune   | `GM_NEPTUNE`                          | (24764, 24764, 24341), equatorial       | 24,764,000.0 |
    /// | 10      | Sun       | `GM_SUN`                              | `R_SUN`                                 | 695,700,000.0 |
    /// | 401     | Phobos    | `GM_PHOBOS`                           | (13.0, 11.4, 9.1), mean                 | 11,166.67 |
    /// | 402     | Deimos    | `GM_DEIMOS`                           | (7.8, 6.0, 5.1), mean                   | 6,300.0 |
    /// | 602     | Enceladus | 7.210366688598896e9 (`BODY602_GM`)    | (256.6, 251.4, 248.3), mean             | 252,100.0 |
    /// | 606     | Titan     | 8.978137095521046e12 (`BODY606_GM`)   | (2575.15, 2574.78, 2574.47), mean       | 2,574,800.0 |
    ///
    /// Every `Custom` entry's `fixed_frame` is
    /// `Some(ReferenceFrame::BodyFixedIAU(naif_id))` when `naif_id` is
    /// in [`iau_rotation_model_ids`] (true for every body in the table
    /// above), and `None` otherwise. `omega` is always `None` — set it
    /// directly on the returned `CustomBody` if a force model needs it.
    ///
    /// # Arguments
    /// - `naif_id`: NAIF ID of the body.
    ///
    /// # Returns
    /// - `body`: The corresponding `CentralBody`, or `Err(BraheError)`
    ///   if `naif_id` is not a built-in body or in the embedded table.
    ///
    /// # Examples
    /// ```
    /// use brahe::propagators::CentralBody;
    ///
    /// assert_eq!(CentralBody::from_naif_id(301).unwrap(), CentralBody::Moon);
    /// assert!(CentralBody::from_naif_id(602).is_ok()); // Enceladus
    /// assert!(CentralBody::from_naif_id(-42).is_err());
    /// ```
    pub fn from_naif_id(naif_id: i32) -> Result<CentralBody, BraheError> {
        let custom = |name: &str, naif_id: i32, gm: f64, radius: f64| -> CentralBody {
            let fixed_frame = if iau_rotation_model_ids().contains(&naif_id) {
                Some(ReferenceFrame::BodyFixedIAU(naif_id))
            } else {
                None
            };
            CentralBody::Custom(CustomBody {
                name: name.to_string(),
                naif_id,
                gm,
                radius: Some(radius),
                omega: None,
                fixed_frame,
            })
        };

        match naif_id {
            399 => Ok(CentralBody::Earth),
            301 => Ok(CentralBody::Moon),
            4 | 499 => Ok(CentralBody::Mars),
            3 => Ok(CentralBody::EMB),
            0 => Ok(CentralBody::SSB),
            199 => Ok(custom("Mercury", 199, GM_MERCURY, 2_439_773.333_333_333_3)),
            299 => Ok(custom("Venus", 299, GM_VENUS, 6_051_800.0)),
            599 => Ok(custom("Jupiter", 599, GM_JUPITER, 71_492_000.0)),
            699 => Ok(custom("Saturn", 699, GM_SATURN, 60_268_000.0)),
            799 => Ok(custom("Uranus", 799, GM_URANUS, 25_559_000.0)),
            899 => Ok(custom("Neptune", 899, GM_NEPTUNE, 24_764_000.0)),
            10 => Ok(custom("Sun", 10, GM_SUN, R_SUN)),
            401 => Ok(custom("Phobos", 401, GM_PHOBOS, 11_166.666_666_666_666)),
            402 => Ok(custom("Deimos", 402, GM_DEIMOS, 6_300.0)),
            602 => Ok(custom("Enceladus", 602, 7.210366688598896e9, 252_100.0)),
            606 => Ok(custom("Titan", 606, 8.978137095521046e12, 2_574_800.0)),
            _ => Err(BraheError::Error(format!(
                "No CentralBody for NAIF ID {}. Use CentralBody::Custom directly for bodies \
                 outside the built-in from_naif_id table.",
                naif_id
            ))),
        }
    }
}

impl Default for CentralBody {
    /// Defaults to `Earth`, matching the default central body used
    /// throughout brahe's force models and propagators.
    ///
    /// # Examples
    /// ```
    /// use brahe::propagators::CentralBody;
    ///
    /// assert_eq!(CentralBody::default(), CentralBody::Earth);
    /// ```
    fn default() -> Self {
        CentralBody::Earth
    }
}

impl std::fmt::Display for CentralBody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CentralBody::Earth => write!(f, "Earth"),
            CentralBody::Moon => write!(f, "Moon"),
            CentralBody::Mars => write!(f, "Mars"),
            CentralBody::EMB => write!(f, "Earth-Moon Barycenter"),
            CentralBody::SSB => write!(f, "Solar System Barycenter"),
            CentralBody::Custom(c) => write!(f, "{}", c.name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_central_body_properties() {
        assert_eq!(CentralBody::Earth.naif_id(), 399);
        assert_eq!(CentralBody::Moon.gm(), GM_MOON);
        assert_eq!(CentralBody::Mars.naif_id(), 499);
        assert_eq!(CentralBody::EMB.gm(), 0.0);
        assert!(CentralBody::SSB.is_barycenter());
        assert_eq!(CentralBody::Moon.fixed_frame(), Some(ReferenceFrame::LFPA));
        assert_eq!(CentralBody::EMB.fixed_frame(), None);
        assert_eq!(CentralBody::Moon.inertial_frame(), ReferenceFrame::LCI);
    }

    #[test]
    fn test_from_naif_id() {
        assert_eq!(CentralBody::from_naif_id(301).unwrap(), CentralBody::Moon);
        let enceladus = CentralBody::from_naif_id(602).unwrap();
        match &enceladus {
            CentralBody::Custom(c) => {
                assert_eq!(c.naif_id, 602);
                assert!(c.gm > 7.0e9 && c.gm < 7.4e9);
                assert_eq!(c.fixed_frame, Some(ReferenceFrame::BodyFixedIAU(602)));
            }
            _ => panic!("expected Custom"),
        }
        assert!(CentralBody::from_naif_id(-42).is_err());
    }

    #[test]
    fn test_central_body_serde_roundtrip() {
        let cb = CentralBody::from_naif_id(602).unwrap();
        let s = serde_json::to_string(&cb).unwrap();
        assert_eq!(serde_json::from_str::<CentralBody>(&s).unwrap(), cb);
    }

    #[test]
    fn test_from_naif_id_mars_accepts_both_ids() {
        assert_eq!(CentralBody::from_naif_id(4).unwrap(), CentralBody::Mars);
        assert_eq!(CentralBody::from_naif_id(499).unwrap(), CentralBody::Mars);
    }

    #[test]
    fn test_custom_body_display() {
        let enceladus = CentralBody::from_naif_id(602).unwrap();
        assert_eq!(enceladus.to_string(), "Enceladus");
        assert_eq!(CentralBody::Earth.to_string(), "Earth");
    }

    #[test]
    fn test_earth_moon_mars_omega_and_frames() {
        assert_eq!(
            CentralBody::Earth.omega_vector(),
            Some(Vector3::new(0.0, 0.0, OMEGA_EARTH))
        );
        assert_eq!(CentralBody::Mars.fixed_frame(), Some(ReferenceFrame::MCMF));
        assert_eq!(CentralBody::Mars.inertial_frame(), ReferenceFrame::MCI);
        assert_eq!(CentralBody::Earth.radius(), Some(R_EARTH));
        assert_eq!(CentralBody::SSB.radius(), None);
        assert_eq!(CentralBody::SSB.omega_vector(), None);
        assert_eq!(CentralBody::EMB.inertial_frame(), ReferenceFrame::EMBI);
        assert_eq!(CentralBody::SSB.inertial_frame(), ReferenceFrame::SSBI);
    }

    #[test]
    fn test_central_body_default_is_earth() {
        assert_eq!(CentralBody::default(), CentralBody::Earth);
    }

    #[test]
    fn test_central_body_all_builtin_accessors() {
        // Exercise every accessor arm for each built-in variant so no match
        // arm is left uncovered.
        // gm
        assert_eq!(CentralBody::Earth.gm(), GM_EARTH);
        assert_eq!(CentralBody::Moon.gm(), GM_MOON);
        assert_eq!(CentralBody::Mars.gm(), GM_MARS);
        assert_eq!(CentralBody::EMB.gm(), 0.0);
        assert_eq!(CentralBody::SSB.gm(), 0.0);
        // radius
        assert_eq!(CentralBody::Earth.radius(), Some(R_EARTH));
        assert_eq!(CentralBody::Moon.radius(), Some(R_MOON));
        assert_eq!(CentralBody::Mars.radius(), Some(R_MARS));
        assert_eq!(CentralBody::EMB.radius(), None);
        assert_eq!(CentralBody::SSB.radius(), None);
        // naif_id
        assert_eq!(CentralBody::Earth.naif_id(), 399);
        assert_eq!(CentralBody::Moon.naif_id(), 301);
        assert_eq!(CentralBody::Mars.naif_id(), 499);
        assert_eq!(CentralBody::EMB.naif_id(), 3);
        assert_eq!(CentralBody::SSB.naif_id(), 0);
        // omega_vector
        assert_eq!(
            CentralBody::Earth.omega_vector(),
            Some(Vector3::new(0.0, 0.0, OMEGA_EARTH))
        );
        assert_eq!(
            CentralBody::Moon.omega_vector(),
            Some(Vector3::new(0.0, 0.0, OMEGA_MOON))
        );
        assert_eq!(
            CentralBody::Mars.omega_vector(),
            Some(Vector3::new(0.0, 0.0, OMEGA_MARS))
        );
        assert_eq!(CentralBody::EMB.omega_vector(), None);
        assert_eq!(CentralBody::SSB.omega_vector(), None);
        // inertial_frame
        assert_eq!(CentralBody::Earth.inertial_frame(), ReferenceFrame::GCRF);
        assert_eq!(CentralBody::Moon.inertial_frame(), ReferenceFrame::LCI);
        assert_eq!(CentralBody::Mars.inertial_frame(), ReferenceFrame::MCI);
        assert_eq!(CentralBody::EMB.inertial_frame(), ReferenceFrame::EMBI);
        assert_eq!(CentralBody::SSB.inertial_frame(), ReferenceFrame::SSBI);
        // fixed_frame
        assert_eq!(CentralBody::Earth.fixed_frame(), Some(ReferenceFrame::ITRF));
        assert_eq!(CentralBody::Moon.fixed_frame(), Some(ReferenceFrame::LFPA));
        assert_eq!(CentralBody::Mars.fixed_frame(), Some(ReferenceFrame::MCMF));
        assert_eq!(CentralBody::EMB.fixed_frame(), None);
        assert_eq!(CentralBody::SSB.fixed_frame(), None);
        // is_barycenter
        assert!(!CentralBody::Earth.is_barycenter());
        assert!(!CentralBody::Moon.is_barycenter());
        assert!(!CentralBody::Mars.is_barycenter());
        assert!(CentralBody::EMB.is_barycenter());
        assert!(CentralBody::SSB.is_barycenter());
        // Display
        assert_eq!(CentralBody::Earth.to_string(), "Earth");
        assert_eq!(CentralBody::Moon.to_string(), "Moon");
        assert_eq!(CentralBody::Mars.to_string(), "Mars");
        assert_eq!(CentralBody::EMB.to_string(), "Earth-Moon Barycenter");
        assert_eq!(CentralBody::SSB.to_string(), "Solar System Barycenter");
    }

    #[test]
    fn test_from_naif_id_builtin_barycenters() {
        assert_eq!(CentralBody::from_naif_id(399).unwrap(), CentralBody::Earth);
        assert_eq!(CentralBody::from_naif_id(3).unwrap(), CentralBody::EMB);
        assert_eq!(CentralBody::from_naif_id(0).unwrap(), CentralBody::SSB);
    }

    #[test]
    fn test_custom_body_omega_and_fixed_frame_accessors() {
        // A Custom body with an explicit spin and fixed frame exercises the
        // `Custom(c)` arms of omega_vector/fixed_frame/inertial_frame.
        let body = CentralBody::Custom(CustomBody {
            name: "Widget".to_string(),
            naif_id: -42,
            gm: 1.0,
            radius: Some(2.0),
            omega: Some(Vector3::new(0.0, 0.0, 3.0)),
            fixed_frame: Some(ReferenceFrame::BodyFixedIAU(-42)),
        });
        assert_eq!(body.gm(), 1.0);
        assert_eq!(body.radius(), Some(2.0));
        assert_eq!(body.naif_id(), -42);
        assert_eq!(body.omega_vector(), Some(Vector3::new(0.0, 0.0, 3.0)));
        assert_eq!(body.inertial_frame(), ReferenceFrame::BodyCenteredICRF(-42));
        assert_eq!(body.fixed_frame(), Some(ReferenceFrame::BodyFixedIAU(-42)));
        assert!(!body.is_barycenter());
    }

    #[test]
    fn test_from_naif_id_all_table_entries_have_bodyfixediau() {
        for id in [199, 299, 599, 699, 799, 899, 10, 401, 402, 602, 606] {
            let body = CentralBody::from_naif_id(id).unwrap();
            assert_eq!(body.naif_id(), id);
            assert_eq!(
                body.fixed_frame(),
                Some(ReferenceFrame::BodyFixedIAU(id)),
                "naif_id {id} should have a BodyFixedIAU frame"
            );
            assert!(body.gm() > 0.0);
            assert!(body.radius().is_some());
            assert!(!body.is_barycenter());
        }
    }
}
