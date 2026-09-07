/*!
 * Typed NAIF body and frame identifiers.
 */

use crate::utils::errors::BraheError;
use std::fmt;
use std::str::FromStr;

/// NAIF integer ID codes for solar-system bodies.
///
/// Named variants cover the planets, planetary-system barycenters, and
/// major natural satellites; [`NAIFId::Id`] carries any other raw NAIF ID
/// (see the NAIF integer ID codes reference:
/// <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html>).
///
/// Equality and hashing compare the underlying integer ID, so
/// `NAIFId::Sun == NAIFId::Id(10)`.
///
/// # Examples
/// ```
/// use brahe::spice::NAIFId;
///
/// assert_eq!(NAIFId::Earth.id(), 399);
/// assert_eq!(NAIFId::Sun, NAIFId::Id(10));
/// ```
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum NAIFId {
    /// Solar System Barycenter.
    SolarSystemBarycenter,
    /// Mercury planetary-system barycenter.
    MercuryBarycenter,
    /// Venus planetary-system barycenter.
    VenusBarycenter,
    /// Earth-Moon barycenter.
    EarthMoonBarycenter,
    /// Mars planetary-system barycenter.
    MarsBarycenter,
    /// Jupiter planetary-system barycenter.
    JupiterBarycenter,
    /// Saturn planetary-system barycenter.
    SaturnBarycenter,
    /// Uranus planetary-system barycenter.
    UranusBarycenter,
    /// Neptune planetary-system barycenter.
    NeptuneBarycenter,
    /// Pluto planetary-system barycenter.
    PlutoBarycenter,
    /// Sun.
    Sun,
    /// Mercury body center.
    Mercury,
    /// Venus body center.
    Venus,
    /// Earth body center.
    Earth,
    /// Moon body center.
    Moon,
    /// Mars body center.
    Mars,
    /// Jupiter body center.
    Jupiter,
    /// Saturn body center.
    Saturn,
    /// Uranus body center.
    Uranus,
    /// Neptune body center.
    Neptune,
    /// Pluto body center.
    Pluto,
    /// Phobos, moon of Mars.
    Phobos,
    /// Deimos, moon of Mars.
    Deimos,
    /// Io, moon of Jupiter.
    Io,
    /// Europa, moon of Jupiter.
    Europa,
    /// Ganymede, moon of Jupiter.
    Ganymede,
    /// Callisto, moon of Jupiter.
    Callisto,
    /// Titan, moon of Saturn.
    Titan,
    /// Ariel, moon of Uranus.
    Ariel,
    /// Umbriel, moon of Uranus.
    Umbriel,
    /// Titania, moon of Uranus.
    Titania,
    /// Oberon, moon of Uranus.
    Oberon,
    /// Miranda, moon of Uranus.
    Miranda,
    /// Triton, moon of Neptune.
    Triton,
    /// Charon, moon of Pluto.
    Charon,
    /// Any other raw NAIF ID (e.g. spacecraft or minor bodies).
    Id(i32),
}

/// Named `NAIFId` variants paired with their raw NAIF integer ID and their
/// canonical NAIF body name, used to keep `From<i32>`, [`NAIFId::name`], and
/// [`NAIFId::from_name`] from drifting apart.
const NAMED: &[(NAIFId, i32, &str)] = &[
    (NAIFId::SolarSystemBarycenter, 0, "SOLAR SYSTEM BARYCENTER"),
    (NAIFId::MercuryBarycenter, 1, "MERCURY BARYCENTER"),
    (NAIFId::VenusBarycenter, 2, "VENUS BARYCENTER"),
    (NAIFId::EarthMoonBarycenter, 3, "EARTH MOON BARYCENTER"),
    (NAIFId::MarsBarycenter, 4, "MARS BARYCENTER"),
    (NAIFId::JupiterBarycenter, 5, "JUPITER BARYCENTER"),
    (NAIFId::SaturnBarycenter, 6, "SATURN BARYCENTER"),
    (NAIFId::UranusBarycenter, 7, "URANUS BARYCENTER"),
    (NAIFId::NeptuneBarycenter, 8, "NEPTUNE BARYCENTER"),
    (NAIFId::PlutoBarycenter, 9, "PLUTO BARYCENTER"),
    (NAIFId::Sun, 10, "SUN"),
    (NAIFId::Mercury, 199, "MERCURY"),
    (NAIFId::Venus, 299, "VENUS"),
    (NAIFId::Earth, 399, "EARTH"),
    (NAIFId::Moon, 301, "MOON"),
    (NAIFId::Mars, 499, "MARS"),
    (NAIFId::Jupiter, 599, "JUPITER"),
    (NAIFId::Saturn, 699, "SATURN"),
    (NAIFId::Uranus, 799, "URANUS"),
    (NAIFId::Neptune, 899, "NEPTUNE"),
    (NAIFId::Pluto, 999, "PLUTO"),
    (NAIFId::Phobos, 401, "PHOBOS"),
    (NAIFId::Deimos, 402, "DEIMOS"),
    (NAIFId::Io, 501, "IO"),
    (NAIFId::Europa, 502, "EUROPA"),
    (NAIFId::Ganymede, 503, "GANYMEDE"),
    (NAIFId::Callisto, 504, "CALLISTO"),
    (NAIFId::Titan, 606, "TITAN"),
    (NAIFId::Ariel, 701, "ARIEL"),
    (NAIFId::Umbriel, 702, "UMBRIEL"),
    (NAIFId::Titania, 703, "TITANIA"),
    (NAIFId::Oberon, 704, "OBERON"),
    (NAIFId::Miranda, 705, "MIRANDA"),
    (NAIFId::Triton, 801, "TRITON"),
    (NAIFId::Charon, 901, "CHARON"),
];

/// Alternate spellings accepted by [`NAIFId::from_name`], paired with the
/// variant they resolve to. These are alias inputs only; [`NAIFId::name`]
/// always returns the canonical spelling from `NAMED`.
const ALIASES: &[(&str, NAIFId)] = &[
    ("SSB", NAIFId::SolarSystemBarycenter),
    ("EMB", NAIFId::EarthMoonBarycenter),
    ("EARTH BARYCENTER", NAIFId::EarthMoonBarycenter),
];

impl NAIFId {
    /// The raw NAIF integer ID code.
    ///
    /// # Returns
    /// - The NAIF integer ID for this body
    ///
    /// # Examples
    /// ```
    /// use brahe::spice::NAIFId;
    ///
    /// assert_eq!(NAIFId::Moon.id(), 301);
    /// assert_eq!(NAIFId::Id(-42).id(), -42);
    /// ```
    pub const fn id(self) -> i32 {
        match self {
            NAIFId::SolarSystemBarycenter => 0,
            NAIFId::MercuryBarycenter => 1,
            NAIFId::VenusBarycenter => 2,
            NAIFId::EarthMoonBarycenter => 3,
            NAIFId::MarsBarycenter => 4,
            NAIFId::JupiterBarycenter => 5,
            NAIFId::SaturnBarycenter => 6,
            NAIFId::UranusBarycenter => 7,
            NAIFId::NeptuneBarycenter => 8,
            NAIFId::PlutoBarycenter => 9,
            NAIFId::Sun => 10,
            NAIFId::Mercury => 199,
            NAIFId::Venus => 299,
            NAIFId::Earth => 399,
            NAIFId::Moon => 301,
            NAIFId::Mars => 499,
            NAIFId::Jupiter => 599,
            NAIFId::Saturn => 699,
            NAIFId::Uranus => 799,
            NAIFId::Neptune => 899,
            NAIFId::Pluto => 999,
            NAIFId::Phobos => 401,
            NAIFId::Deimos => 402,
            NAIFId::Io => 501,
            NAIFId::Europa => 502,
            NAIFId::Ganymede => 503,
            NAIFId::Callisto => 504,
            NAIFId::Titan => 606,
            NAIFId::Ariel => 701,
            NAIFId::Umbriel => 702,
            NAIFId::Titania => 703,
            NAIFId::Oberon => 704,
            NAIFId::Miranda => 705,
            NAIFId::Triton => 801,
            NAIFId::Charon => 901,
            NAIFId::Id(raw) => raw,
        }
    }

    /// The canonical NAIF body name.
    ///
    /// Named variants return their NAIF body name (e.g. `"EARTH MOON
    /// BARYCENTER"`); [`NAIFId::Id`] returns its raw integer ID formatted
    /// as a string.
    ///
    /// # Returns
    /// - The NAIF body name, or the raw integer ID as a string for
    ///   [`NAIFId::Id`]
    ///
    /// # Examples
    /// ```
    /// use brahe::spice::NAIFId;
    ///
    /// assert_eq!(NAIFId::Earth.name(), "EARTH");
    /// assert_eq!(NAIFId::EarthMoonBarycenter.name(), "EARTH MOON BARYCENTER");
    /// assert_eq!(NAIFId::Id(2000001).name(), "2000001");
    /// ```
    pub fn name(&self) -> String {
        match self {
            NAIFId::Id(raw) => raw.to_string(),
            _ => NAMED
                .iter()
                .find(|(_, id, _)| *id == self.id())
                .map(|(_, _, name)| name.to_string())
                .unwrap_or_else(|| self.id().to_string()),
        }
    }

    /// Resolves a NAIF body name or integer ID string to a [`NAIFId`].
    ///
    /// Matching is case-insensitive and trims surrounding whitespace;
    /// underscores are treated as spaces and runs of whitespace collapse to a
    /// single space (so `"EARTH_MOON_BARYCENTER"` and
    /// `"EARTH   MOON   BARYCENTER"` both match `"EARTH MOON BARYCENTER"`).
    /// The abbreviations `"SSB"`, `"EMB"`, and `"EARTH BARYCENTER"` are
    /// accepted as aliases for the solar system and Earth-Moon barycenters. A
    /// string that does not match a known name or alias is parsed as an
    /// integer NAIF ID; if that also fails, an error is returned.
    ///
    /// # Arguments
    /// - `name`: A NAIF body name (e.g. `"MARS BARYCENTER"`), an accepted
    ///   alias (e.g. `"SSB"`), or an integer NAIF ID string (e.g. `"2000001"`)
    ///
    /// # Returns
    /// - The matching [`NAIFId`], or a [`BraheError::Error`] if `name` is
    ///   neither a known NAIF body name nor a valid integer ID
    ///
    /// # Examples
    /// ```
    /// use brahe::spice::NAIFId;
    ///
    /// assert_eq!(
    ///     NAIFId::from_name("mars barycenter").unwrap(),
    ///     NAIFId::MarsBarycenter
    /// );
    /// assert_eq!(NAIFId::from_name("SSB").unwrap(), NAIFId::SolarSystemBarycenter);
    /// assert_eq!(NAIFId::from_name("emb").unwrap(), NAIFId::EarthMoonBarycenter);
    /// assert_eq!(NAIFId::from_name("2000001").unwrap(), NAIFId::Id(2000001));
    /// assert!(NAIFId::from_name("not a body").is_err());
    /// ```
    pub fn from_name(name: &str) -> Result<NAIFId, BraheError> {
        let trimmed = name.trim();
        let normalized = trimmed
            .to_uppercase()
            .replace('_', " ")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        if let Some((variant, _, _)) = NAMED.iter().find(|(_, _, n)| *n == normalized) {
            return Ok(*variant);
        }
        if let Some((_, variant)) = ALIASES.iter().find(|(alias, _)| *alias == normalized) {
            return Ok(*variant);
        }
        if let Ok(raw) = normalized.parse::<i32>() {
            return Ok(NAIFId::Id(raw));
        }
        Err(BraheError::Error(format!(
            "'{}' is not a known NAIF body name or ID",
            trimmed
        )))
    }
}

impl fmt::Display for NAIFId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl From<i32> for NAIFId {
    /// Canonicalizes a raw NAIF integer ID to its named variant when one
    /// exists, otherwise wraps it in [`NAIFId::Id`].
    fn from(raw: i32) -> Self {
        NAMED
            .iter()
            .find(|(_, id, _)| *id == raw)
            .map(|(variant, _, _)| *variant)
            .unwrap_or(NAIFId::Id(raw))
    }
}

impl From<NAIFId> for i32 {
    fn from(id: NAIFId) -> Self {
        id.id()
    }
}

impl FromStr for NAIFId {
    type Err = BraheError;

    /// Parses a NAIF body name, accepted alias, or integer ID string.
    ///
    /// Delegates to [`NAIFId::from_name`], so matching is case-insensitive,
    /// underscores are treated as spaces, and a string that names no known
    /// body is parsed as an integer NAIF ID.
    ///
    /// # Arguments
    /// - `s`: A NAIF body name (e.g. `"MARS BARYCENTER"`), an accepted alias
    ///   (e.g. `"SSB"`), or an integer NAIF ID string (e.g. `"2000001"`)
    ///
    /// # Returns
    /// - The matching [`NAIFId`], or a [`BraheError::Error`] if `s` is
    ///   neither a known NAIF body name nor a valid integer ID
    ///
    /// # Examples
    /// ```
    /// use brahe::spice::NAIFId;
    ///
    /// assert_eq!(
    ///     "MARS BARYCENTER".parse::<NAIFId>().unwrap(),
    ///     NAIFId::MarsBarycenter
    /// );
    /// assert_eq!("2000001".parse::<NAIFId>().unwrap(), NAIFId::Id(2000001));
    /// assert!("not a body".parse::<NAIFId>().is_err());
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        NAIFId::from_name(s)
    }
}

impl PartialEq for NAIFId {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}
impl Eq for NAIFId {}
impl std::hash::Hash for NAIFId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id().hash(state);
    }
}

/// NAIF frame class ID codes for PCK body-fixed reference frames.
///
/// [`FrameId::Id`] carries any raw frame class ID not otherwise named (see
/// the NAIF Frames Required Reading:
/// <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/frames.html>).
///
/// Equality and hashing compare the underlying integer ID, so
/// `FrameId::MoonPaDe440 == FrameId::Id(31008)`.
///
/// # Examples
/// ```
/// use brahe::spice::FrameId;
///
/// assert_eq!(FrameId::MoonPaDe440.id(), 31008);
/// assert_eq!(FrameId::MoonPaDe440, FrameId::Id(31008));
/// ```
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum FrameId {
    /// MOON_PA_DE440 principal-axis lunar body-fixed frame.
    MoonPaDe440,
    /// Any other raw NAIF frame class ID.
    Id(i32),
}

impl FrameId {
    /// The raw NAIF frame class ID code.
    ///
    /// # Returns
    /// - The NAIF integer frame class ID
    ///
    /// # Examples
    /// ```
    /// use brahe::spice::FrameId;
    ///
    /// assert_eq!(FrameId::MoonPaDe440.id(), 31008);
    /// assert_eq!(FrameId::Id(31006).id(), 31006);
    /// ```
    pub const fn id(self) -> i32 {
        match self {
            FrameId::MoonPaDe440 => 31008,
            FrameId::Id(raw) => raw,
        }
    }
}

impl From<i32> for FrameId {
    fn from(raw: i32) -> Self {
        FrameId::Id(raw)
    }
}

impl From<FrameId> for i32 {
    fn from(id: FrameId) -> Self {
        id.id()
    }
}

impl PartialEq for FrameId {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}
impl Eq for FrameId {}
impl std::hash::Hash for FrameId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id().hash(state);
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use serial_test::parallel;

    #[test]
    #[parallel]
    fn test_naif_id_values() {
        assert_eq!(NAIFId::SolarSystemBarycenter.id(), 0);
        assert_eq!(NAIFId::EarthMoonBarycenter.id(), 3);
        assert_eq!(NAIFId::Sun.id(), 10);
        assert_eq!(NAIFId::Earth.id(), 399);
        assert_eq!(NAIFId::Moon.id(), 301);
        assert_eq!(NAIFId::Mars.id(), 499);
        assert_eq!(NAIFId::Pluto.id(), 999);
        assert_eq!(NAIFId::Titan.id(), 606);
        assert_eq!(NAIFId::Id(-42).id(), -42);
    }

    #[test]
    #[parallel]
    fn test_naif_id_equality_across_forms() {
        assert_eq!(NAIFId::Sun, NAIFId::Id(10));
        assert_eq!(NAIFId::from(399), NAIFId::Earth);
        let x: i32 = NAIFId::JupiterBarycenter.into();
        assert_eq!(x, 5);
    }

    #[test]
    #[parallel]
    fn test_frame_id_values() {
        assert_eq!(FrameId::MoonPaDe440.id(), 31008);
        assert_eq!(FrameId::Id(31006).id(), 31006);
        assert_eq!(FrameId::MoonPaDe440, FrameId::Id(31008));
    }

    #[test]
    #[parallel]
    fn test_naif_id_all_variants_exhaustive() {
        // Every named variant maps to its documented NAIF integer ID,
        // exercising all arms of `NAIFId::id`.
        let cases = [
            (NAIFId::SolarSystemBarycenter, 0),
            (NAIFId::MercuryBarycenter, 1),
            (NAIFId::VenusBarycenter, 2),
            (NAIFId::EarthMoonBarycenter, 3),
            (NAIFId::MarsBarycenter, 4),
            (NAIFId::JupiterBarycenter, 5),
            (NAIFId::SaturnBarycenter, 6),
            (NAIFId::UranusBarycenter, 7),
            (NAIFId::NeptuneBarycenter, 8),
            (NAIFId::PlutoBarycenter, 9),
            (NAIFId::Sun, 10),
            (NAIFId::Mercury, 199),
            (NAIFId::Venus, 299),
            (NAIFId::Earth, 399),
            (NAIFId::Moon, 301),
            (NAIFId::Mars, 499),
            (NAIFId::Jupiter, 599),
            (NAIFId::Saturn, 699),
            (NAIFId::Uranus, 799),
            (NAIFId::Neptune, 899),
            (NAIFId::Pluto, 999),
            (NAIFId::Phobos, 401),
            (NAIFId::Deimos, 402),
            (NAIFId::Io, 501),
            (NAIFId::Europa, 502),
            (NAIFId::Ganymede, 503),
            (NAIFId::Callisto, 504),
            (NAIFId::Titan, 606),
            (NAIFId::Ariel, 701),
            (NAIFId::Umbriel, 702),
            (NAIFId::Titania, 703),
            (NAIFId::Oberon, 704),
            (NAIFId::Miranda, 705),
            (NAIFId::Triton, 801),
            (NAIFId::Charon, 901),
            (NAIFId::Id(-42), -42),
        ];
        for (variant, expected) in cases {
            assert_eq!(variant.id(), expected);
        }
    }

    #[test]
    #[parallel]
    fn test_naif_id_hash_matches_by_id() {
        use std::collections::HashSet;
        // Hash follows the integer ID, so a named variant and its raw-ID
        // form collapse to a single set entry.
        let mut set = HashSet::new();
        set.insert(NAIFId::Sun);
        assert!(!set.insert(NAIFId::Id(10)));
        assert_eq!(set.len(), 1);
        // A distinct ID is a separate entry.
        assert!(set.insert(NAIFId::Earth));
        assert_eq!(set.len(), 2);
    }

    #[test]
    #[parallel]
    fn test_naifid_from_i32_canonicalizes() {
        assert!(matches!(NAIFId::from(399), NAIFId::Earth));
        assert!(matches!(NAIFId::from(4), NAIFId::MarsBarycenter));
        assert!(matches!(NAIFId::from(-42), NAIFId::Id(-42)));
    }

    #[test]
    #[parallel]
    fn test_naifid_name_and_from_name_round_trip() {
        for id in [
            NAIFId::SolarSystemBarycenter,
            NAIFId::EarthMoonBarycenter,
            NAIFId::Earth,
            NAIFId::Moon,
            NAIFId::Mars,
            NAIFId::MarsBarycenter,
            NAIFId::Titan,
        ] {
            assert_eq!(NAIFId::from_name(&id.name()).unwrap(), id);
        }
        assert_eq!(NAIFId::Earth.name(), "EARTH");
        assert_eq!(NAIFId::EarthMoonBarycenter.name(), "EARTH MOON BARYCENTER");
        assert_eq!(
            NAIFId::from_name(" mars barycenter ").unwrap(),
            NAIFId::MarsBarycenter
        );
        assert_eq!(
            NAIFId::from_name("EARTH_MOON_BARYCENTER").unwrap(),
            NAIFId::EarthMoonBarycenter
        );
        assert_eq!(NAIFId::from_name("2000001").unwrap(), NAIFId::Id(2000001));
        assert_eq!(NAIFId::Id(2000001).name(), "2000001");
        let err = NAIFId::from_name("PLANET X").unwrap_err();
        assert!(err.to_string().contains("PLANET X"));
        assert_eq!(format!("{}", NAIFId::Moon), "MOON");
    }

    #[test]
    #[parallel]
    fn test_naifid_from_name_aliases() {
        assert_eq!(
            NAIFId::from_name("SSB").unwrap(),
            NAIFId::SolarSystemBarycenter
        );
        assert_eq!(
            NAIFId::from_name(" ssb ").unwrap(),
            NAIFId::SolarSystemBarycenter
        );
        assert_eq!(
            NAIFId::from_name("emb").unwrap(),
            NAIFId::EarthMoonBarycenter
        );
        assert_eq!(
            NAIFId::from_name("EARTH         BARYCENTER").unwrap(),
            NAIFId::EarthMoonBarycenter
        );
        assert_eq!(
            NAIFId::from_name("Mars_Barycenter").unwrap(),
            NAIFId::MarsBarycenter
        );
        assert_eq!(
            NAIFId::from_name("EARTH\tMOON\nBARYCENTER").unwrap(),
            NAIFId::EarthMoonBarycenter
        );
        // Aliases are input spellings only; `name` stays canonical.
        assert_eq!(
            NAIFId::SolarSystemBarycenter.name(),
            "SOLAR SYSTEM BARYCENTER"
        );
        assert_eq!(NAIFId::EarthMoonBarycenter.name(), "EARTH MOON BARYCENTER");
        assert_eq!(NAIFId::MarsBarycenter.name(), "MARS BARYCENTER");
    }

    #[test]
    #[parallel]
    fn test_naifid_from_str() {
        assert_eq!(
            "MARS BARYCENTER".parse::<NAIFId>().unwrap(),
            NAIFId::MarsBarycenter
        );
        assert_eq!("mars".parse::<NAIFId>().unwrap(), NAIFId::Mars);
        assert_eq!(
            "ssb".parse::<NAIFId>().unwrap(),
            NAIFId::SolarSystemBarycenter
        );
        assert_eq!("2000001".parse::<NAIFId>().unwrap(), NAIFId::Id(2000001));
        assert_eq!(
            NAIFId::from_str("EARTH_MOON_BARYCENTER").unwrap(),
            NAIFId::EarthMoonBarycenter
        );

        let err = "PLANET X".parse::<NAIFId>().unwrap_err();
        assert!(
            err.to_string().contains("not a known NAIF body name or ID"),
            "unexpected error message: {}",
            err
        );
    }

    #[test]
    #[parallel]
    fn test_frame_id_from_impls_and_hash() {
        use std::collections::HashSet;
        // From<i32> yields the raw-ID form; the named variant compares equal.
        assert_eq!(FrameId::from(31008), FrameId::MoonPaDe440);
        // From<FrameId> for i32 returns the underlying class ID.
        let raw: i32 = FrameId::MoonPaDe440.into();
        assert_eq!(raw, 31008);
        let raw_other: i32 = FrameId::Id(31006).into();
        assert_eq!(raw_other, 31006);
        // Hash follows the class ID: named and raw forms collapse to one entry.
        let mut set = HashSet::new();
        set.insert(FrameId::MoonPaDe440);
        assert!(!set.insert(FrameId::Id(31008)));
        assert_eq!(set.len(), 1);
        assert!(set.insert(FrameId::Id(31006)));
        assert_eq!(set.len(), 2);
    }
}
