/*!
 * Orientation half of a celestial reference frame.
 *
 * [`FrameAxes`] names every orientation the celestial frame router can
 * evaluate, independent of the body a frame is centered on. A
 * [`CelestialFrame`](super::CelestialFrame) pairs one `FrameAxes` value
 * with one NAIF center; see
 * [`CelestialFrame::centered`](super::CelestialFrame::centered).
 */

use std::fmt;
use std::str::FromStr;

use crate::utils::BraheError;

/// Orientation of a celestial frame, independent of its origin.
///
/// The variants correspond one-to-one with the orientations the celestial
/// frame router evaluates. Rotating axes (`ITRF`, the body-fixed families,
/// and the synodic families) carry transport-velocity terms that depend
/// only on the axes' angular velocity, so they apply unchanged to a frame
/// centered on a body other than the orientation's usual center.
///
/// [`Display`](fmt::Display) prints the variant name, with the payload in
/// parentheses for the parameterized variants. [`FromStr`] parses the
/// eleven argument-free names case-insensitively; the parameterized
/// variants are constructed directly.
///
/// # Examples
/// ```
/// use brahe::frames::{CelestialFrame, FrameAxes};
///
/// assert_eq!(CelestialFrame::MCMF.axes(), FrameAxes::MarsFixed);
/// assert_eq!("tod".parse::<FrameAxes>().unwrap(), FrameAxes::TOD);
/// assert_eq!(FrameAxes::BodyFixedIAU(499).to_string(), "BodyFixedIAU(499)");
/// ```
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum FrameAxes {
    /// ICRF axes (identity).
    ICRF,
    /// Earth mean equator and equinox of J2000.0 (frame bias from ICRF).
    EME2000,
    /// Earth mean equator and equinox of date (bias-precession).
    MOD,
    /// Earth true equator and equinox of date (bias-precession-nutation
    /// with Earth orientation parameter corrections).
    TOD,
    /// Earth true equator and mean equinox of date, anchored to GMST 1982
    /// (the SGP4 output frame).
    TEME,
    /// Earth-fixed (ITRF): bias-precession-nutation, Earth rotation, and
    /// polar motion.
    ITRF,
    /// Lunar principal-axis frame from the loaded binary PCK
    /// (`MOON_PA_DE440`).
    LunarPA,
    /// Lunar mean-Earth/polar-axis frame.
    LunarME,
    /// Mars body-fixed frame (IAU/WGCCRE Mars rotation model).
    MarsFixed,
    /// Earth-Moon rotating axes (x̂ from Earth to Moon).
    EMR,
    /// Sun-Earth rotating axes (x̂ from Sun to Earth).
    SER,
    /// Geocentric solar ecliptic axes (x̂ from Earth to Sun).
    GSE,
    /// IAU/WGCCRE body-fixed axes of the given NAIF ID.
    BodyFixedIAU(i32),
    /// Body-fixed axes evaluated from a loaded binary PCK frame class ID.
    BodyFixedPCK(i32),
    /// Body-fixed axes from a user-registered rotation callback, keyed by
    /// its registry key.
    BodyFixedCustom(u32),
    /// Generic two-body synodic axes (x̂ from `primary` toward
    /// `secondary`).
    Synodic {
        /// NAIF ID of the primary body (x̂ base).
        primary: i32,
        /// NAIF ID of the secondary body (x̂ target).
        secondary: i32,
    },
}

impl fmt::Display for FrameAxes {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FrameAxes::ICRF => write!(f, "ICRF"),
            FrameAxes::EME2000 => write!(f, "EME2000"),
            FrameAxes::MOD => write!(f, "MOD"),
            FrameAxes::TOD => write!(f, "TOD"),
            FrameAxes::TEME => write!(f, "TEME"),
            FrameAxes::ITRF => write!(f, "ITRF"),
            FrameAxes::LunarPA => write!(f, "LunarPA"),
            FrameAxes::LunarME => write!(f, "LunarME"),
            FrameAxes::MarsFixed => write!(f, "MarsFixed"),
            FrameAxes::EMR => write!(f, "EMR"),
            FrameAxes::SER => write!(f, "SER"),
            FrameAxes::GSE => write!(f, "GSE"),
            FrameAxes::BodyFixedIAU(id) => write!(f, "BodyFixedIAU({})", id),
            FrameAxes::BodyFixedPCK(frame_id) => write!(f, "BodyFixedPCK({})", frame_id),
            FrameAxes::BodyFixedCustom(key) => write!(f, "BodyFixedCustom({})", key),
            FrameAxes::Synodic { primary, secondary } => {
                write!(f, "Synodic({},{})", primary, secondary)
            }
        }
    }
}

impl FromStr for FrameAxes {
    type Err = BraheError;

    /// Parses the eleven argument-free axes names case-insensitively.
    ///
    /// The parameterized variants (`BodyFixedIAU`, `BodyFixedPCK`,
    /// `BodyFixedCustom`, `Synodic`) are not parseable from a string;
    /// construct them directly.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_uppercase().as_str() {
            "ICRF" => Ok(FrameAxes::ICRF),
            "EME2000" => Ok(FrameAxes::EME2000),
            "MOD" => Ok(FrameAxes::MOD),
            "TOD" => Ok(FrameAxes::TOD),
            "TEME" => Ok(FrameAxes::TEME),
            "ITRF" => Ok(FrameAxes::ITRF),
            "LUNARPA" => Ok(FrameAxes::LunarPA),
            "LUNARME" => Ok(FrameAxes::LunarME),
            "MARSFIXED" => Ok(FrameAxes::MarsFixed),
            "EMR" => Ok(FrameAxes::EMR),
            "SER" => Ok(FrameAxes::SER),
            "GSE" => Ok(FrameAxes::GSE),
            _ => Err(BraheError::ParseError(format!(
                "Unknown frame axes '{}'. Supported: ICRF, EME2000, MOD, TOD, TEME, ITRF, \
                 LunarPA, LunarME, MarsFixed, EMR, SER, GSE",
                s
            ))),
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use serial_test::parallel;

    use super::*;

    #[test]
    #[parallel]
    fn test_frameaxes_display() {
        assert_eq!(FrameAxes::ICRF.to_string(), "ICRF");
        assert_eq!(FrameAxes::TEME.to_string(), "TEME");
        assert_eq!(FrameAxes::LunarPA.to_string(), "LunarPA");
        assert_eq!(FrameAxes::LunarME.to_string(), "LunarME");
        assert_eq!(FrameAxes::MarsFixed.to_string(), "MarsFixed");
        assert_eq!(
            FrameAxes::BodyFixedPCK(31008).to_string(),
            "BodyFixedPCK(31008)"
        );
        assert_eq!(
            FrameAxes::BodyFixedCustom(42).to_string(),
            "BodyFixedCustom(42)"
        );
        assert_eq!(
            FrameAxes::Synodic {
                primary: 399,
                secondary: 301
            }
            .to_string(),
            "Synodic(399,301)"
        );
    }

    #[test]
    #[parallel]
    fn test_frameaxes_from_str_case_insensitive() {
        assert_eq!("icrf".parse::<FrameAxes>().unwrap(), FrameAxes::ICRF);
        assert_eq!("  MoD ".parse::<FrameAxes>().unwrap(), FrameAxes::MOD);
        assert_eq!("teme".parse::<FrameAxes>().unwrap(), FrameAxes::TEME);
        assert_eq!("lunarpa".parse::<FrameAxes>().unwrap(), FrameAxes::LunarPA);
        assert_eq!(
            "MARSFIXED".parse::<FrameAxes>().unwrap(),
            FrameAxes::MarsFixed
        );
        assert!("Synodic(399,301)".parse::<FrameAxes>().is_err());
        assert!("".parse::<FrameAxes>().is_err());
    }

    #[test]
    #[parallel]
    fn test_frameaxes_serde_round_trip() {
        for a in [
            FrameAxes::ICRF,
            FrameAxes::BodyFixedIAU(499),
            FrameAxes::BodyFixedPCK(31008),
            FrameAxes::BodyFixedCustom(7),
            FrameAxes::Synodic {
                primary: 10,
                secondary: 399,
            },
        ] {
            let s = serde_json::to_string(&a).unwrap();
            assert_eq!(serde_json::from_str::<FrameAxes>(&s).unwrap(), a);
        }
    }
}
