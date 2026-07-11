/*!
 * Typed NAIF body and frame identifiers.
 */

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
}

impl From<i32> for NAIFId {
    fn from(raw: i32) -> Self {
        NAIFId::Id(raw)
    }
}

impl From<NAIFId> for i32 {
    fn from(id: NAIFId) -> Self {
        id.id()
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

    #[test]
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
    fn test_naif_id_equality_across_forms() {
        assert_eq!(NAIFId::Sun, NAIFId::Id(10));
        assert_eq!(NAIFId::from(399), NAIFId::Earth);
        let x: i32 = NAIFId::JupiterBarycenter.into();
        assert_eq!(x, 5);
    }

    #[test]
    fn test_frame_id_values() {
        assert_eq!(FrameId::MoonPaDe440.id(), 31008);
        assert_eq!(FrameId::Id(31006).id(), 31006);
        assert_eq!(FrameId::MoonPaDe440, FrameId::Id(31008));
    }
}
