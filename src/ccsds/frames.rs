/*!
Typed reference-frame vocabularies for CCSDS Attitude Data Messages (ADM).

CCSDS 504.0-B-2 Annex B3 requires the values of `REF_FRAME_A`, `REF_FRAME_B`,
`ANGVEL_FRAME`, `INERTIA_REF_FRAME`, and `MAN_REF_FRAME` to be drawn from three
SANA registries:

- <https://sanaregistry.org/r/celestial_body_reference_frames>
- <https://sanaregistry.org/r/orbit_relative_reference_frames>
- <https://sanaregistry.org/r/spacecraft_body_reference_frames>

Each registry is mirrored by one enum here; [`ADMReferenceFrame`] is the sum
type used by ADM frame keywords. Non-registry tokens parse into `Other(String)`
variants so that files round-trip verbatim; conversion to native brahe types
(via `crate::ccsds::interop`) fails for such frames with a descriptive error.
*/

use std::fmt;

use serde::{Deserialize, Serialize};

/// Celestial-body reference frame, per the SANA registry
/// <https://sanaregistry.org/r/celestial_body_reference_frames> (CCSDS
/// 504.0-B-2 annex B3).
///
/// Parametrized realizations carry their realization designator:
/// `ITRF(Some(2014))` is the token `ITRF2014`, `ITRF(None)` is bare `ITRF`;
/// `GCRF(Some(2))` is `GCRF2`; `MoonPA(Some(440))` is `MOON_PA440`.
/// Tokens not in the registry parse into [`CCSDSCelestialBodyFrame::Other`]
/// and round-trip verbatim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CCSDSCelestialBodyFrame {
    /// Inertial evaluation of a celestial-body fixed frame at a reference epoch.
    AlignCB,
    /// Inertial evaluation of the Earth-fixed frame at a reference epoch.
    AlignEarth,
    /// FK4 mean equator and equinox of Besselian epoch 1950.
    B1950,
    /// Celestial Intermediate Reference System (IAU 2000A).
    CIRS,
    /// DGFI-TUM realization of the ITRS, with realization year.
    DTRF(Option<u16>),
    /// Earth-Fixed Greenwich rotating frame (no polar motion).
    EFG,
    /// Earth Mean Equator and Equinox of J2000.0.
    EME2000,
    /// Rotating celestial-body fixed frame.
    FixedCB,
    /// Rotating Earth-fixed frame.
    FixedEarth,
    /// Geocentric Celestial Reference Frame, with optional realization number.
    GCRF(Option<u8>),
    /// Greenwich True-Of-Date rotating frame.
    GTOD,
    /// International Celestial Reference Frame, with optional realization number.
    ICRF(Option<u8>),
    /// Celestial-body centered frame with constant rotation offset from ICRF.
    InertialCB,
    /// International Terrestrial Reference Frame, with optional realization year.
    ITRF(Option<u16>),
    /// Mean equator and equinox of J2000 epoch (quasi-inertial).
    J2000,
    /// J2000 variant per the SANA registry.
    J2000A,
    /// Ecliptic frame at J2000 epoch.
    J2000Ecliptic,
    /// Mean-of-date frame for a celestial body.
    MODCB,
    /// Earth mean-of-date frame.
    MODEarth,
    /// Moon mean-of-date frame.
    MODMoon,
    /// Mean-of-epoch frame for a celestial body.
    MOECB,
    /// Earth mean-of-epoch frame.
    MOEEarth,
    /// Moon Mean-Earth/rotation-axis body-fixed frame.
    MoonME,
    /// Moon mean equator and IAU node of epoch frame.
    MoonMEIAUE,
    /// Lunar principal-axes frame, with optional DE-ephemeris designator.
    MoonPA(Option<u16>),
    /// True Equator Mean Equinox of date (TLE frame).
    TEMEOfDate,
    /// True Equator Mean Equinox of epoch.
    TEMEOfEpoch,
    /// Terrestrial Intermediate Reference System.
    TIRS,
    /// True-of-date frame for a celestial body.
    TODCB,
    /// Earth true-of-date frame.
    TODEarth,
    /// Moon true-of-date frame.
    TODMoon,
    /// True-of-epoch frame for a celestial body.
    TOECB,
    /// Earth true-of-epoch frame.
    TOEEarth,
    /// Moon true-of-epoch frame.
    TOEMoon,
    /// True ecliptic frame of date.
    TrueEcliptic,
    /// Launch go-inertial frame.
    UVWGOInertial,
    /// World Geodetic System 1984 Earth-fixed frame.
    WGS84,
    /// Any token not present in the SANA registry; round-trips verbatim.
    Other(String),
}

impl fmt::Display for CCSDSCelestialBodyFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlignCB => write!(f, "ALIGN_CB"),
            Self::AlignEarth => write!(f, "ALIGN_EARTH"),
            Self::B1950 => write!(f, "B1950"),
            Self::CIRS => write!(f, "CIRS"),
            Self::DTRF(None) => write!(f, "DTRF"),
            Self::DTRF(Some(y)) => write!(f, "DTRF{}", y),
            Self::EFG => write!(f, "EFG"),
            Self::EME2000 => write!(f, "EME2000"),
            Self::FixedCB => write!(f, "FIXED_CB"),
            Self::FixedEarth => write!(f, "FIXED_EARTH"),
            Self::GCRF(None) => write!(f, "GCRF"),
            Self::GCRF(Some(n)) => write!(f, "GCRF{}", n),
            Self::GTOD => write!(f, "GTOD"),
            Self::ICRF(None) => write!(f, "ICRF"),
            Self::ICRF(Some(n)) => write!(f, "ICRF{}", n),
            Self::InertialCB => write!(f, "INERTIAL_CB"),
            Self::ITRF(None) => write!(f, "ITRF"),
            Self::ITRF(Some(y)) => write!(f, "ITRF{}", y),
            Self::J2000 => write!(f, "J2000"),
            Self::J2000A => write!(f, "J2000A"),
            Self::J2000Ecliptic => write!(f, "J2000_ECLIPTIC"),
            Self::MODCB => write!(f, "MOD_CB"),
            Self::MODEarth => write!(f, "MOD_EARTH"),
            Self::MODMoon => write!(f, "MOD_MOON"),
            Self::MOECB => write!(f, "MOE_CB"),
            Self::MOEEarth => write!(f, "MOE_EARTH"),
            Self::MoonME => write!(f, "MOON_ME"),
            Self::MoonMEIAUE => write!(f, "MOON_MEIAUE"),
            Self::MoonPA(None) => write!(f, "MOON_PA"),
            Self::MoonPA(Some(n)) => write!(f, "MOON_PA{}", n),
            Self::TEMEOfDate => write!(f, "TEMEOFDATE"),
            Self::TEMEOfEpoch => write!(f, "TEMEOFEPOCH"),
            Self::TIRS => write!(f, "TIRS"),
            Self::TODCB => write!(f, "TOD_CB"),
            Self::TODEarth => write!(f, "TOD_EARTH"),
            Self::TODMoon => write!(f, "TOD_MOON"),
            Self::TOECB => write!(f, "TOE_CB"),
            Self::TOEEarth => write!(f, "TOE_EARTH"),
            Self::TOEMoon => write!(f, "TOE_MOON"),
            Self::TrueEcliptic => write!(f, "TRUE_ECLIPTIC"),
            Self::UVWGOInertial => write!(f, "UVW_GO_INERTIAL"),
            Self::WGS84 => write!(f, "WGS84"),
            Self::Other(s) => write!(f, "{}", s),
        }
    }
}

impl CCSDSCelestialBodyFrame {
    /// Parses a SANA celestial-body frame token. Infallible: unknown tokens
    /// return [`CCSDSCelestialBodyFrame::Other`] preserving the input verbatim.
    /// Matching is case-insensitive; hyphenated realization designators
    /// (`ITRF-2014`) are accepted.
    ///
    /// # Arguments
    ///
    /// * `s` - The token string to parse (whitespace is trimmed).
    ///
    /// # Returns
    ///
    /// A [`CCSDSCelestialBodyFrame`] variant representing the parsed token.
    /// Unknown tokens return [`CCSDSCelestialBodyFrame::Other`] with the
    /// original string preserved.
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::ccsds::CCSDSCelestialBodyFrame;
    ///
    /// assert_eq!(
    ///     CCSDSCelestialBodyFrame::parse("ITRF2014"),
    ///     CCSDSCelestialBodyFrame::ITRF(Some(2014))
    /// );
    /// assert_eq!(
    ///     CCSDSCelestialBodyFrame::parse("EME2000"),
    ///     CCSDSCelestialBodyFrame::EME2000
    /// );
    /// assert_eq!(
    ///     CCSDSCelestialBodyFrame::parse("CUSTOM_FRAME").to_string(),
    ///     "CUSTOM_FRAME"
    /// );
    /// ```
    pub fn parse(s: &str) -> Self {
        let token = s.trim();
        let upper = token.to_uppercase();
        // Fixed tokens first
        match upper.as_str() {
            "ALIGN_CB" => return Self::AlignCB,
            "ALIGN_EARTH" => return Self::AlignEarth,
            "B1950" => return Self::B1950,
            "CIRS" => return Self::CIRS,
            "EFG" => return Self::EFG,
            "EME2000" => return Self::EME2000,
            "FIXED_CB" => return Self::FixedCB,
            "FIXED_EARTH" => return Self::FixedEarth,
            "GTOD" => return Self::GTOD,
            "INERTIAL_CB" => return Self::InertialCB,
            "J2000" => return Self::J2000,
            "J2000A" => return Self::J2000A,
            "J2000_ECLIPTIC" => return Self::J2000Ecliptic,
            "MOD_CB" => return Self::MODCB,
            "MOD_EARTH" => return Self::MODEarth,
            "MOD_MOON" => return Self::MODMoon,
            "MOE_CB" => return Self::MOECB,
            "MOE_EARTH" => return Self::MOEEarth,
            "MOON_ME" => return Self::MoonME,
            "MOON_MEIAUE" => return Self::MoonMEIAUE,
            "TEMEOFDATE" => return Self::TEMEOfDate,
            "TEMEOFEPOCH" => return Self::TEMEOfEpoch,
            "TIRS" => return Self::TIRS,
            "TOD_CB" => return Self::TODCB,
            "TOD_EARTH" => return Self::TODEarth,
            "TOD_MOON" => return Self::TODMoon,
            "TOE_CB" => return Self::TOECB,
            "TOE_EARTH" => return Self::TOEEarth,
            "TOE_MOON" => return Self::TOEMoon,
            "TRUE_ECLIPTIC" => return Self::TrueEcliptic,
            "UVW_GO_INERTIAL" => return Self::UVWGOInertial,
            "WGS84" => return Self::WGS84,
            _ => {}
        }
        // Parametrized families: FAMILY[ -]?<digits>
        if let Some(v) = parse_numeric_suffix(&upper, "ITRF") {
            return Self::ITRF(v);
        }
        if let Some(v) = parse_numeric_suffix(&upper, "DTRF") {
            return Self::DTRF(v);
        }
        if let Some(v) = parse_numeric_suffix(&upper, "GCRF") {
            if let Some(n) = v {
                if let Ok(n_u8) = u8::try_from(n) {
                    return Self::GCRF(Some(n_u8));
                }
            } else {
                return Self::GCRF(None);
            }
        }
        if let Some(v) = parse_numeric_suffix(&upper, "ICRF") {
            if let Some(n) = v {
                if let Ok(n_u8) = u8::try_from(n) {
                    return Self::ICRF(Some(n_u8));
                }
            } else {
                return Self::ICRF(None);
            }
        }
        if let Some(v) = parse_numeric_suffix(&upper, "MOON_PA") {
            return Self::MoonPA(v);
        }
        Self::Other(token.to_string())
    }
}

/// Matches `PREFIX`, `PREFIX<digits>`, or `PREFIX-<digits>` (case already
/// upper). Returns `None` when the token does not belong to the family,
/// `Some(None)` for the bare prefix, `Some(Some(n))` for a numeric suffix.
fn parse_numeric_suffix(upper: &str, prefix: &str) -> Option<Option<u16>> {
    let rest = upper.strip_prefix(prefix)?;
    if rest.is_empty() {
        return Some(None);
    }
    let digits = rest.strip_prefix('-').unwrap_or(rest);
    digits.parse::<u16>().ok().map(Some)
}

/// Orbit-relative reference frame, per the SANA registry
/// <https://sanaregistry.org/r/orbit_relative_reference_frames>.
///
/// `*Rotating` variants are true local orbital frames; `*Inertial` variants
/// are quasi-inertial snapshots evaluated at each time of interest. `EQW` and
/// `PQW` exist only in inertial form.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CCSDSOrbitRelativeFrame {
    /// Equinoctial frame (inertial snapshot).
    EQWInertial,
    /// Local-Vertical Local-Horizontal, inertial snapshot.
    LVLHInertial,
    /// Local-Vertical Local-Horizontal, rotating.
    LVLHRotating,
    /// Nadir/Sun/normal frame, inertial snapshot.
    NSWInertial,
    /// Nadir/Sun/normal frame, rotating.
    NSWRotating,
    /// Normal/tangential/cross-track frame, inertial snapshot.
    NTWInertial,
    /// Normal/tangential/cross-track frame, rotating.
    NTWRotating,
    /// Perifocal frame (inertial snapshot).
    PQWInertial,
    /// Radial/along-track/cross-track frame (brahe: RTN), inertial snapshot.
    RSWInertial,
    /// Radial/along-track/cross-track frame (brahe: RTN), rotating.
    RSWRotating,
    /// Topocentric south/east/zenith frame, inertial snapshot.
    SEZInertial,
    /// Topocentric south/east/zenith frame, rotating.
    SEZRotating,
    /// Tangential/normal/cross-track frame, inertial snapshot.
    TNWInertial,
    /// Tangential/normal/cross-track frame, rotating.
    TNWRotating,
    /// Velocity/normal/co-normal frame, inertial snapshot.
    VNCInertial,
    /// Velocity/normal/co-normal frame, rotating.
    VNCRotating,
    /// Any token not present in the SANA registry; round-trips verbatim.
    Other(String),
}

impl fmt::Display for CCSDSOrbitRelativeFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let token = match self {
            Self::EQWInertial => "EQW_INERTIAL",
            Self::LVLHInertial => "LVLH_INERTIAL",
            Self::LVLHRotating => "LVLH_ROTATING",
            Self::NSWInertial => "NSW_INERTIAL",
            Self::NSWRotating => "NSW_ROTATING",
            Self::NTWInertial => "NTW_INERTIAL",
            Self::NTWRotating => "NTW_ROTATING",
            Self::PQWInertial => "PQW_INERTIAL",
            Self::RSWInertial => "RSW_INERTIAL",
            Self::RSWRotating => "RSW_ROTATING",
            Self::SEZInertial => "SEZ_INERTIAL",
            Self::SEZRotating => "SEZ_ROTATING",
            Self::TNWInertial => "TNW_INERTIAL",
            Self::TNWRotating => "TNW_ROTATING",
            Self::VNCInertial => "VNC_INERTIAL",
            Self::VNCRotating => "VNC_ROTATING",
            Self::Other(s) => return write!(f, "{}", s),
        };
        write!(f, "{}", token)
    }
}

impl CCSDSOrbitRelativeFrame {
    /// Parses a SANA orbit-relative frame token. Infallible: unknown tokens
    /// return [`CCSDSOrbitRelativeFrame::Other`] preserving the input
    /// verbatim. Matching is case-insensitive.
    ///
    /// # Arguments
    ///
    /// * `s` - The token string to parse (whitespace is trimmed).
    ///
    /// # Returns
    ///
    /// A [`CCSDSOrbitRelativeFrame`] variant representing the parsed token.
    /// Unknown tokens return [`CCSDSOrbitRelativeFrame::Other`] with the
    /// original string preserved.
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::ccsds::CCSDSOrbitRelativeFrame;
    ///
    /// assert_eq!(
    ///     CCSDSOrbitRelativeFrame::parse("RSW_ROTATING"),
    ///     CCSDSOrbitRelativeFrame::RSWRotating
    /// );
    /// assert_eq!(
    ///     CCSDSOrbitRelativeFrame::parse("lvlh_inertial"),
    ///     CCSDSOrbitRelativeFrame::LVLHInertial
    /// );
    /// ```
    pub fn parse(s: &str) -> Self {
        let token = s.trim();
        match token.to_uppercase().as_str() {
            "EQW_INERTIAL" => Self::EQWInertial,
            "LVLH_INERTIAL" => Self::LVLHInertial,
            "LVLH_ROTATING" => Self::LVLHRotating,
            "NSW_INERTIAL" => Self::NSWInertial,
            "NSW_ROTATING" => Self::NSWRotating,
            "NTW_INERTIAL" => Self::NTWInertial,
            "NTW_ROTATING" => Self::NTWRotating,
            "PQW_INERTIAL" => Self::PQWInertial,
            "RSW_INERTIAL" => Self::RSWInertial,
            "RSW_ROTATING" => Self::RSWRotating,
            "SEZ_INERTIAL" => Self::SEZInertial,
            "SEZ_ROTATING" => Self::SEZRotating,
            "TNW_INERTIAL" => Self::TNWInertial,
            "TNW_ROTATING" => Self::TNWRotating,
            "VNC_INERTIAL" => Self::VNCInertial,
            "VNC_ROTATING" => Self::VNCRotating,
            _ => Self::Other(token.to_string()),
        }
    }
}

/// Spacecraft-body reference frame, per the SANA registry
/// <https://sanaregistry.org/r/spacecraft_body_reference_frames>.
///
/// Each family carries an optional instance designator: `SCBody(Some("1"))`
/// is the token `SC_BODY_1`, `SCBody(None)` is `SC_BODY`. Designators are
/// strings because the standard's own examples use letters (`INSTRUMENT_A`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CCSDSSpacecraftBodyFrame {
    /// Accelerometer frame.
    ACC(Option<String>),
    /// Actuator frame.
    Actuator(Option<String>),
    /// Autonomous star tracker frame.
    AST(Option<String>),
    /// Coarse sun sensor frame.
    CSS(Option<String>),
    /// Digital sun sensor frame.
    DSS(Option<String>),
    /// Earth sensor assembly frame.
    ESA(Option<String>),
    /// Gyroscope frame.
    GyroFrame(Option<String>),
    /// Inertial measurement unit frame.
    IMUFrame(Option<String>),
    /// Instrument frame.
    Instrument(Option<String>),
    /// Magnetic torque assembly frame.
    MTA(Option<String>),
    /// Reaction wheel frame.
    RW(Option<String>),
    /// Solar array frame.
    SA(Option<String>),
    /// Spacecraft body frame.
    SCBody(Option<String>),
    /// Generic sensor frame.
    Sensor(Option<String>),
    /// Star tracker frame.
    StarTracker(Option<String>),
    /// Three-axis magnetometer frame.
    TAM(Option<String>),
    /// Any token not matching a registry family; round-trips verbatim.
    Other(String),
}

impl fmt::Display for CCSDSSpacecraftBodyFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (family, suffix) = match self {
            Self::ACC(s) => ("ACC", s),
            Self::Actuator(s) => ("ACTUATOR", s),
            Self::AST(s) => ("AST", s),
            Self::CSS(s) => ("CSS", s),
            Self::DSS(s) => ("DSS", s),
            Self::ESA(s) => ("ESA", s),
            Self::GyroFrame(s) => ("GYRO_FRAME", s),
            Self::IMUFrame(s) => ("IMU_FRAME", s),
            Self::Instrument(s) => ("INSTRUMENT", s),
            Self::MTA(s) => ("MTA", s),
            Self::RW(s) => ("RW", s),
            Self::SA(s) => ("SA", s),
            Self::SCBody(s) => ("SC_BODY", s),
            Self::Sensor(s) => ("SENSOR", s),
            Self::StarTracker(s) => ("STARTRACKER", s),
            Self::TAM(s) => ("TAM", s),
            Self::Other(t) => return write!(f, "{}", t),
        };
        match suffix {
            Some(i) => write!(f, "{}_{}", family, i),
            None => write!(f, "{}", family),
        }
    }
}

impl CCSDSSpacecraftBodyFrame {
    /// Parses a SANA spacecraft-body frame token. Infallible: tokens not
    /// matching any registry family return
    /// [`CCSDSSpacecraftBodyFrame::Other`] preserving the input verbatim.
    ///
    /// # Arguments
    ///
    /// * `s` - The token string to parse (whitespace is trimmed).
    ///
    /// # Returns
    ///
    /// A [`CCSDSSpacecraftBodyFrame`] variant representing the parsed token.
    /// Unknown tokens return [`CCSDSSpacecraftBodyFrame::Other`] with the
    /// original string preserved.
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::ccsds::CCSDSSpacecraftBodyFrame;
    ///
    /// assert_eq!(
    ///     CCSDSSpacecraftBodyFrame::parse("SC_BODY_1"),
    ///     CCSDSSpacecraftBodyFrame::SCBody(Some("1".to_string()))
    /// );
    /// assert_eq!(
    ///     CCSDSSpacecraftBodyFrame::parse("INSTRUMENT_A"),
    ///     CCSDSSpacecraftBodyFrame::Instrument(Some("A".to_string()))
    /// );
    /// ```
    pub fn parse(s: &str) -> Self {
        let token = s.trim();
        let upper = token.to_uppercase();
        const FAMILIES: &[&str] = &[
            "STARTRACKER",
            "GYRO_FRAME",
            "IMU_FRAME",
            "INSTRUMENT",
            "ACTUATOR",
            "SC_BODY",
            "SENSOR",
            "ACC",
            "AST",
            "CSS",
            "DSS",
            "ESA",
            "MTA",
            "RW",
            "SA",
            "TAM",
        ];
        for family in FAMILIES {
            if let Some(rest) = upper.strip_prefix(family) {
                let suffix = match rest {
                    "" => None,
                    _ => match rest.strip_prefix('_') {
                        Some(i) if !i.is_empty() => Some(i.to_string()),
                        _ => continue,
                    },
                };
                return Self::from_family(family, suffix);
            }
        }
        Self::Other(token.to_string())
    }

    fn from_family(family: &str, suffix: Option<String>) -> Self {
        match family {
            "ACC" => Self::ACC(suffix),
            "ACTUATOR" => Self::Actuator(suffix),
            "AST" => Self::AST(suffix),
            "CSS" => Self::CSS(suffix),
            "DSS" => Self::DSS(suffix),
            "ESA" => Self::ESA(suffix),
            "GYRO_FRAME" => Self::GyroFrame(suffix),
            "IMU_FRAME" => Self::IMUFrame(suffix),
            "INSTRUMENT" => Self::Instrument(suffix),
            "MTA" => Self::MTA(suffix),
            "RW" => Self::RW(suffix),
            "SA" => Self::SA(suffix),
            "SC_BODY" => Self::SCBody(suffix),
            "SENSOR" => Self::Sensor(suffix),
            "STARTRACKER" => Self::StarTracker(suffix),
            "TAM" => Self::TAM(suffix),
            _ => unreachable!("from_family called with unknown family"),
        }
    }
}

/// Reference frame value for the ADM keywords `REF_FRAME_A`, `REF_FRAME_B`,
/// `ANGVEL_FRAME`, `INERTIA_REF_FRAME`, and `MAN_REF_FRAME` (CCSDS 504.0-B-2
/// annex B3): one of the three SANA registry vocabularies, or a non-registry
/// token preserved verbatim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ADMReferenceFrame {
    /// A celestial-body reference frame.
    Celestial(CCSDSCelestialBodyFrame),
    /// An orbit-relative reference frame.
    OrbitRelative(CCSDSOrbitRelativeFrame),
    /// A spacecraft-body reference frame.
    Spacecraft(CCSDSSpacecraftBodyFrame),
    /// A token present in none of the three registries; round-trips verbatim.
    Other(String),
}

impl fmt::Display for ADMReferenceFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Celestial(frame) => write!(f, "{}", frame),
            Self::OrbitRelative(frame) => write!(f, "{}", frame),
            Self::Spacecraft(frame) => write!(f, "{}", frame),
            Self::Other(s) => write!(f, "{}", s),
        }
    }
}

impl ADMReferenceFrame {
    /// Parses an ADM frame token by trying the three SANA vocabularies in
    /// order (celestial, orbit-relative, spacecraft; the token sets are
    /// disjoint, so order does not affect the result). Tokens matching no
    /// registry return [`ADMReferenceFrame::Other`].
    ///
    /// # Arguments
    ///
    /// * `s` - The token string to parse (whitespace is trimmed).
    ///
    /// # Returns
    ///
    /// An [`ADMReferenceFrame`] variant representing the parsed token.
    /// Tokens present in any of the three SANA registries return the
    /// appropriate nested variant; unknown tokens return
    /// [`ADMReferenceFrame::Other`] with the original string preserved.
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::ccsds::ADMReferenceFrame;
    ///
    /// assert_eq!(
    ///     ADMReferenceFrame::parse("ITRF2014").to_string(),
    ///     "ITRF2014"
    /// );
    /// assert_eq!(
    ///     ADMReferenceFrame::parse("RSW_ROTATING").to_string(),
    ///     "RSW_ROTATING"
    /// );
    /// ```
    pub fn parse(s: &str) -> Self {
        match CCSDSCelestialBodyFrame::parse(s) {
            CCSDSCelestialBodyFrame::Other(_) => {}
            frame => return Self::Celestial(frame),
        }
        match CCSDSOrbitRelativeFrame::parse(s) {
            CCSDSOrbitRelativeFrame::Other(_) => {}
            frame => return Self::OrbitRelative(frame),
        }
        match CCSDSSpacecraftBodyFrame::parse(s) {
            CCSDSSpacecraftBodyFrame::Other(_) => {}
            frame => return Self::Spacecraft(frame),
        }
        Self::Other(s.trim().to_string())
    }
}

impl From<CCSDSCelestialBodyFrame> for ADMReferenceFrame {
    fn from(frame: CCSDSCelestialBodyFrame) -> Self {
        Self::Celestial(frame)
    }
}

impl From<CCSDSOrbitRelativeFrame> for ADMReferenceFrame {
    fn from(frame: CCSDSOrbitRelativeFrame) -> Self {
        Self::OrbitRelative(frame)
    }
}

impl From<CCSDSSpacecraftBodyFrame> for ADMReferenceFrame {
    fn from(frame: CCSDSSpacecraftBodyFrame) -> Self {
        Self::Spacecraft(frame)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use serial_test::parallel;

    use super::*;

    #[test]
    #[parallel]
    fn test_celestial_body_frame_display() {
        assert_eq!(CCSDSCelestialBodyFrame::ICRF(None).to_string(), "ICRF");
        assert_eq!(CCSDSCelestialBodyFrame::ICRF(Some(3)).to_string(), "ICRF3");
        assert_eq!(CCSDSCelestialBodyFrame::GCRF(None).to_string(), "GCRF");
        assert_eq!(CCSDSCelestialBodyFrame::GCRF(Some(2)).to_string(), "GCRF2");
        assert_eq!(CCSDSCelestialBodyFrame::EME2000.to_string(), "EME2000");
        assert_eq!(CCSDSCelestialBodyFrame::J2000.to_string(), "J2000");
        assert_eq!(CCSDSCelestialBodyFrame::ITRF(None).to_string(), "ITRF");
        assert_eq!(
            CCSDSCelestialBodyFrame::ITRF(Some(2014)).to_string(),
            "ITRF2014"
        );
        assert_eq!(
            CCSDSCelestialBodyFrame::DTRF(Some(2014)).to_string(),
            "DTRF2014"
        );
        assert_eq!(CCSDSCelestialBodyFrame::MoonPA(None).to_string(), "MOON_PA");
        assert_eq!(
            CCSDSCelestialBodyFrame::MoonPA(Some(440)).to_string(),
            "MOON_PA440"
        );
        assert_eq!(CCSDSCelestialBodyFrame::MoonME.to_string(), "MOON_ME");
        assert_eq!(
            CCSDSCelestialBodyFrame::MoonMEIAUE.to_string(),
            "MOON_MEIAUE"
        );
        assert_eq!(CCSDSCelestialBodyFrame::AlignCB.to_string(), "ALIGN_CB");
        assert_eq!(
            CCSDSCelestialBodyFrame::AlignEarth.to_string(),
            "ALIGN_EARTH"
        );
        assert_eq!(CCSDSCelestialBodyFrame::B1950.to_string(), "B1950");
        assert_eq!(CCSDSCelestialBodyFrame::CIRS.to_string(), "CIRS");
        assert_eq!(CCSDSCelestialBodyFrame::TIRS.to_string(), "TIRS");
        assert_eq!(CCSDSCelestialBodyFrame::EFG.to_string(), "EFG");
        assert_eq!(CCSDSCelestialBodyFrame::GTOD.to_string(), "GTOD");
        assert_eq!(CCSDSCelestialBodyFrame::FixedCB.to_string(), "FIXED_CB");
        assert_eq!(
            CCSDSCelestialBodyFrame::FixedEarth.to_string(),
            "FIXED_EARTH"
        );
        assert_eq!(
            CCSDSCelestialBodyFrame::InertialCB.to_string(),
            "INERTIAL_CB"
        );
        assert_eq!(CCSDSCelestialBodyFrame::J2000A.to_string(), "J2000A");
        assert_eq!(
            CCSDSCelestialBodyFrame::J2000Ecliptic.to_string(),
            "J2000_ECLIPTIC"
        );
        assert_eq!(CCSDSCelestialBodyFrame::MODCB.to_string(), "MOD_CB");
        assert_eq!(CCSDSCelestialBodyFrame::MODEarth.to_string(), "MOD_EARTH");
        assert_eq!(CCSDSCelestialBodyFrame::MODMoon.to_string(), "MOD_MOON");
        assert_eq!(CCSDSCelestialBodyFrame::MOECB.to_string(), "MOE_CB");
        assert_eq!(CCSDSCelestialBodyFrame::MOEEarth.to_string(), "MOE_EARTH");
        assert_eq!(
            CCSDSCelestialBodyFrame::TEMEOfDate.to_string(),
            "TEMEOFDATE"
        );
        assert_eq!(
            CCSDSCelestialBodyFrame::TEMEOfEpoch.to_string(),
            "TEMEOFEPOCH"
        );
        assert_eq!(CCSDSCelestialBodyFrame::TODCB.to_string(), "TOD_CB");
        assert_eq!(CCSDSCelestialBodyFrame::TODEarth.to_string(), "TOD_EARTH");
        assert_eq!(CCSDSCelestialBodyFrame::TODMoon.to_string(), "TOD_MOON");
        assert_eq!(CCSDSCelestialBodyFrame::TOECB.to_string(), "TOE_CB");
        assert_eq!(CCSDSCelestialBodyFrame::TOEEarth.to_string(), "TOE_EARTH");
        assert_eq!(CCSDSCelestialBodyFrame::TOEMoon.to_string(), "TOE_MOON");
        assert_eq!(
            CCSDSCelestialBodyFrame::TrueEcliptic.to_string(),
            "TRUE_ECLIPTIC"
        );
        assert_eq!(
            CCSDSCelestialBodyFrame::UVWGOInertial.to_string(),
            "UVW_GO_INERTIAL"
        );
        assert_eq!(CCSDSCelestialBodyFrame::WGS84.to_string(), "WGS84");
        assert_eq!(
            CCSDSCelestialBodyFrame::Other("CUSTOM_FRAME".to_string()).to_string(),
            "CUSTOM_FRAME"
        );
    }

    #[test]
    #[parallel]
    fn test_celestial_body_frame_parse() {
        assert_eq!(
            CCSDSCelestialBodyFrame::parse("ICRF"),
            CCSDSCelestialBodyFrame::ICRF(None)
        );
        assert_eq!(
            CCSDSCelestialBodyFrame::parse("ICRF3"),
            CCSDSCelestialBodyFrame::ICRF(Some(3))
        );
        assert_eq!(
            CCSDSCelestialBodyFrame::parse("GCRF"),
            CCSDSCelestialBodyFrame::GCRF(None)
        );
        assert_eq!(
            CCSDSCelestialBodyFrame::parse("GCRF2"),
            CCSDSCelestialBodyFrame::GCRF(Some(2))
        );
        assert_eq!(
            CCSDSCelestialBodyFrame::parse("ITRF2014"),
            CCSDSCelestialBodyFrame::ITRF(Some(2014))
        );
        assert_eq!(
            CCSDSCelestialBodyFrame::parse("ITRF-2014"),
            CCSDSCelestialBodyFrame::ITRF(Some(2014))
        );
        assert_eq!(
            CCSDSCelestialBodyFrame::parse("ITRF"),
            CCSDSCelestialBodyFrame::ITRF(None)
        );
        assert_eq!(
            CCSDSCelestialBodyFrame::parse("DTRF2020"),
            CCSDSCelestialBodyFrame::DTRF(Some(2020))
        );
        assert_eq!(
            CCSDSCelestialBodyFrame::parse("MOON_PA"),
            CCSDSCelestialBodyFrame::MoonPA(None)
        );
        assert_eq!(
            CCSDSCelestialBodyFrame::parse("MOON_PA440"),
            CCSDSCelestialBodyFrame::MoonPA(Some(440))
        );
        assert_eq!(
            CCSDSCelestialBodyFrame::parse("MOON_ME"),
            CCSDSCelestialBodyFrame::MoonME
        );
        assert_eq!(
            CCSDSCelestialBodyFrame::parse("EME2000"),
            CCSDSCelestialBodyFrame::EME2000
        );
        assert_eq!(
            CCSDSCelestialBodyFrame::parse("eme2000"),
            CCSDSCelestialBodyFrame::EME2000
        );
        assert_eq!(
            CCSDSCelestialBodyFrame::parse("SOME_FRAME"),
            CCSDSCelestialBodyFrame::Other("SOME_FRAME".to_string())
        );
    }

    #[test]
    #[parallel]
    fn test_celestial_body_frame_roundtrip() {
        for token in [
            "ALIGN_CB",
            "ALIGN_EARTH",
            "B1950",
            "CIRS",
            "DTRF2014",
            "EFG",
            "EME2000",
            "FIXED_CB",
            "FIXED_EARTH",
            "GCRF",
            "GCRF2",
            "GTOD",
            "ICRF",
            "ICRF3",
            "INERTIAL_CB",
            "ITRF",
            "ITRF2014",
            "J2000",
            "J2000A",
            "J2000_ECLIPTIC",
            "MOD_CB",
            "MOD_EARTH",
            "MOD_MOON",
            "MOE_CB",
            "MOE_EARTH",
            "MOON_ME",
            "MOON_MEIAUE",
            "MOON_PA",
            "MOON_PA440",
            "TEMEOFDATE",
            "TEMEOFEPOCH",
            "TIRS",
            "TOD_CB",
            "TOD_EARTH",
            "TOD_MOON",
            "TOE_CB",
            "TOE_EARTH",
            "TOE_MOON",
            "TRUE_ECLIPTIC",
            "UVW_GO_INERTIAL",
            "WGS84",
            "NONREGISTRY",
        ] {
            assert_eq!(CCSDSCelestialBodyFrame::parse(token).to_string(), token);
        }
    }

    #[test]
    #[parallel]
    fn test_orbit_relative_frame_roundtrip() {
        for token in [
            "EQW_INERTIAL",
            "LVLH_INERTIAL",
            "LVLH_ROTATING",
            "NSW_INERTIAL",
            "NSW_ROTATING",
            "NTW_INERTIAL",
            "NTW_ROTATING",
            "PQW_INERTIAL",
            "RSW_INERTIAL",
            "RSW_ROTATING",
            "SEZ_INERTIAL",
            "SEZ_ROTATING",
            "TNW_INERTIAL",
            "TNW_ROTATING",
            "VNC_INERTIAL",
            "VNC_ROTATING",
            "LVLH_CUSTOM",
        ] {
            assert_eq!(CCSDSOrbitRelativeFrame::parse(token).to_string(), token);
        }
        assert_eq!(
            CCSDSOrbitRelativeFrame::parse("RSW_ROTATING"),
            CCSDSOrbitRelativeFrame::RSWRotating
        );
        assert_eq!(
            CCSDSOrbitRelativeFrame::parse("LVLH_CUSTOM"),
            CCSDSOrbitRelativeFrame::Other("LVLH_CUSTOM".to_string())
        );
    }

    #[test]
    #[parallel]
    fn test_spacecraft_body_frame_parse() {
        assert_eq!(
            CCSDSSpacecraftBodyFrame::parse("SC_BODY_1"),
            CCSDSSpacecraftBodyFrame::SCBody(Some("1".to_string()))
        );
        assert_eq!(
            CCSDSSpacecraftBodyFrame::parse("SC_BODY"),
            CCSDSSpacecraftBodyFrame::SCBody(None)
        );
        assert_eq!(
            CCSDSSpacecraftBodyFrame::parse("INSTRUMENT_A"),
            CCSDSSpacecraftBodyFrame::Instrument(Some("A".to_string()))
        );
        assert_eq!(
            CCSDSSpacecraftBodyFrame::parse("GYRO_FRAME_2"),
            CCSDSSpacecraftBodyFrame::GyroFrame(Some("2".to_string()))
        );
        assert_eq!(
            CCSDSSpacecraftBodyFrame::parse("STARTRACKER_3"),
            CCSDSSpacecraftBodyFrame::StarTracker(Some("3".to_string()))
        );
        assert_eq!(
            CCSDSSpacecraftBodyFrame::parse("IMU_FRAME_1"),
            CCSDSSpacecraftBodyFrame::IMUFrame(Some("1".to_string()))
        );
        assert_eq!(
            CCSDSSpacecraftBodyFrame::parse("WHEEL_9"),
            CCSDSSpacecraftBodyFrame::Other("WHEEL_9".to_string())
        );
    }

    #[test]
    #[parallel]
    fn test_spacecraft_body_frame_roundtrip() {
        for token in [
            "ACC_1",
            "ACTUATOR_1",
            "AST_1",
            "CSS_2",
            "DSS_1",
            "ESA_1",
            "GYRO_FRAME_1",
            "IMU_FRAME_2",
            "INSTRUMENT_A",
            "MTA_1",
            "RW_4",
            "SA_1",
            "SC_BODY_1",
            "SENSOR_10",
            "STARTRACKER_2",
            "TAM_1",
            "SC_BODY",
            "WHEEL_9",
        ] {
            assert_eq!(CCSDSSpacecraftBodyFrame::parse(token).to_string(), token);
        }
    }

    #[test]
    #[parallel]
    fn test_adm_reference_frame_parse_dispatch() {
        assert_eq!(
            ADMReferenceFrame::parse("EME2000"),
            ADMReferenceFrame::Celestial(CCSDSCelestialBodyFrame::EME2000)
        );
        assert_eq!(
            ADMReferenceFrame::parse("RSW_ROTATING"),
            ADMReferenceFrame::OrbitRelative(CCSDSOrbitRelativeFrame::RSWRotating)
        );
        assert_eq!(
            ADMReferenceFrame::parse("SC_BODY_1"),
            ADMReferenceFrame::Spacecraft(CCSDSSpacecraftBodyFrame::SCBody(Some("1".to_string())))
        );
        assert_eq!(
            ADMReferenceFrame::parse("BODY_FRAME_A"),
            ADMReferenceFrame::Other("BODY_FRAME_A".to_string())
        );
    }

    #[test]
    #[parallel]
    fn test_adm_reference_frame_roundtrip() {
        for token in [
            "ICRF",
            "ITRF2014",
            "LVLH_ROTATING",
            "INSTRUMENT_A",
            "SC_BODY",
            "BODY_FRAME_A",
        ] {
            assert_eq!(ADMReferenceFrame::parse(token).to_string(), token);
        }
    }

    #[test]
    #[parallel]
    fn test_celestial_body_frame_dtrf_bare() {
        assert_eq!(
            CCSDSCelestialBodyFrame::parse("DTRF"),
            CCSDSCelestialBodyFrame::DTRF(None)
        );
        assert_eq!(CCSDSCelestialBodyFrame::parse("DTRF").to_string(), "DTRF");
    }

    #[test]
    #[parallel]
    fn test_spacecraft_body_frame_parse_non_underscore_suffix() {
        // "AST12" matches the "AST" family prefix but the remainder is not
        // underscore-delimited, so it does not form a valid instance
        // designator and the token falls through to `Other`.
        assert_eq!(
            CCSDSSpacecraftBodyFrame::parse("AST12"),
            CCSDSSpacecraftBodyFrame::Other("AST12".to_string())
        );
    }

    #[test]
    #[parallel]
    fn test_celestial_body_frame_from_impl() {
        let frame: ADMReferenceFrame = CCSDSCelestialBodyFrame::EME2000.into();
        assert_eq!(
            frame,
            ADMReferenceFrame::Celestial(CCSDSCelestialBodyFrame::EME2000)
        );
    }

    #[test]
    #[parallel]
    fn test_orbit_relative_frame_from_impl() {
        let frame: ADMReferenceFrame = CCSDSOrbitRelativeFrame::RSWRotating.into();
        assert_eq!(
            frame,
            ADMReferenceFrame::OrbitRelative(CCSDSOrbitRelativeFrame::RSWRotating)
        );
    }

    #[test]
    #[parallel]
    fn test_spacecraft_body_frame_from_impl() {
        let frame: ADMReferenceFrame = CCSDSSpacecraftBodyFrame::SCBody(None).into();
        assert_eq!(
            frame,
            ADMReferenceFrame::Spacecraft(CCSDSSpacecraftBodyFrame::SCBody(None))
        );
    }

    #[test]
    #[parallel]
    fn test_celestial_body_frame_numeric_suffix_overflow() {
        assert_eq!(
            CCSDSCelestialBodyFrame::parse("GCRF999"),
            CCSDSCelestialBodyFrame::Other("GCRF999".to_string())
        );
        assert_eq!(
            CCSDSCelestialBodyFrame::parse("ICRF300"),
            CCSDSCelestialBodyFrame::Other("ICRF300".to_string())
        );
        assert_eq!(
            CCSDSCelestialBodyFrame::parse("GCRF999").to_string(),
            "GCRF999"
        );
    }
}
