/*!
 * Origin half of a celestial reference frame.
 *
 * [`FrameCenter`] names the body or barycenter a frame is centered on,
 * independent of the frame's orientation. A
 * [`CelestialFrame`](super::CelestialFrame) pairs one `FrameCenter` value
 * with one [`FrameAxes`](super::FrameAxes) value; see
 * [`CelestialFrame::centered`](super::CelestialFrame::centered).
 */

use std::fmt;
use std::str::FromStr;

use crate::spice::NAIFId;
use crate::utils::BraheError;

use super::transform::synodic_barycenter_id;

/// Origin of a celestial frame, independent of its axes.
///
/// [`FrameCenter::Body`] names a NAIF-catalogued body or system
/// barycenter, including the ones that have no name in the catalogue
/// ([`NAIFId::Id`] carries any raw ID, such as a self-assigned negative
/// center). [`FrameCenter::Barycenter`] names the GM-weighted barycenter
/// of a two-body pair, the origin the synodic frames compute analytically
/// rather than reading from an ephemeris; it is a different point from a
/// catalogued system barycenter such as [`NAIFId::EarthMoonBarycenter`],
/// which is a `Body`.
///
/// [`FrameCenter::naif_id`] is the identity the frame router uses: two
/// centers with the same `naif_id` translate identically, and
/// [`CelestialFrame::centered`](super::CelestialFrame::centered)
/// canonicalizes on it. Equality and hashing are structural instead, so a
/// `Barycenter` value and the `Body` built from the same synthetic ID are
/// distinct values that share one `naif_id`.
///
/// [`Display`](fmt::Display) prints [`FrameCenter::name`]. [`FromStr`]
/// parses NAIF body names and integer IDs into `Body`; the
/// `BARYCENTER(..)` form `name` prints for a `Barycenter` is not parsed
/// back.
///
/// Equality and hashing compare [`FrameCenter::naif_id`], the identity the
/// frame router translates by, so a `Body` built from a synthetic synodic
/// barycenter ID equals the `Barycenter` that encodes to that ID, exactly as
/// [`NAIFId`] compares by its integer code.
///
/// # Examples
/// ```
/// use brahe::frames::FrameCenter;
/// use brahe::spice::NAIFId;
///
/// assert_eq!(FrameCenter::from(NAIFId::Mars).naif_id(), 499);
/// assert_eq!(FrameCenter::from(499).name(), "MARS");
/// assert_eq!(
///     FrameCenter::Barycenter {
///         primary: NAIFId::Sun,
///         secondary: NAIFId::Earth,
///     }
///     .name(),
///     "BARYCENTER(SUN, EARTH)"
/// );
/// ```
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum FrameCenter {
    /// A NAIF-catalogued body or system barycenter.
    Body(NAIFId),
    /// GM-weighted barycenter of a two-body pair, as used by the synodic
    /// frames.
    Barycenter {
        /// NAIF ID of the primary body.
        primary: NAIFId,
        /// NAIF ID of the secondary body.
        secondary: NAIFId,
    },
}

impl FrameCenter {
    /// NAIF ID of this origin.
    ///
    /// This is the identity the frame router translates on: a
    /// [`FrameCenter::Body`] returns its own ID, and a
    /// [`FrameCenter::Barycenter`] returns the synthetic ID
    /// [`synodic_barycenter_id`] encodes for its pair.
    ///
    /// # Returns
    /// - `naif_id`: NAIF ID of the origin
    ///
    /// # Examples
    /// ```
    /// use brahe::frames::{synodic_barycenter_id, FrameCenter};
    /// use brahe::spice::NAIFId;
    ///
    /// assert_eq!(FrameCenter::Body(NAIFId::Earth).naif_id(), 399);
    /// assert_eq!(
    ///     FrameCenter::Barycenter {
    ///         primary: NAIFId::Earth,
    ///         secondary: NAIFId::Moon,
    ///     }
    ///     .naif_id(),
    ///     synodic_barycenter_id(399, 301)
    /// );
    /// ```
    pub fn naif_id(&self) -> i32 {
        match self {
            FrameCenter::Body(id) => id.id(),
            FrameCenter::Barycenter { primary, secondary } => {
                synodic_barycenter_id(primary.id(), secondary.id())
            }
        }
    }

    /// Name of this origin.
    ///
    /// A [`FrameCenter::Body`] returns the NAIF body name (or the raw
    /// integer ID for an uncatalogued one; see [`NAIFId::name`]), and a
    /// [`FrameCenter::Barycenter`] returns `BARYCENTER(primary,
    /// secondary)` with both bodies named the same way.
    ///
    /// # Returns
    /// - `name`: Name of the origin
    ///
    /// # Examples
    /// ```
    /// use brahe::frames::FrameCenter;
    /// use brahe::spice::NAIFId;
    ///
    /// assert_eq!(FrameCenter::Body(NAIFId::Earth).name(), "EARTH");
    /// assert_eq!(FrameCenter::Body(NAIFId::Id(-42)).name(), "-42");
    /// assert_eq!(
    ///     FrameCenter::Barycenter {
    ///         primary: NAIFId::Earth,
    ///         secondary: NAIFId::Moon,
    ///     }
    ///     .name(),
    ///     "BARYCENTER(EARTH, MOON)"
    /// );
    /// ```
    pub fn name(&self) -> String {
        match self {
            FrameCenter::Body(id) => id.name(),
            FrameCenter::Barycenter { primary, secondary } => {
                format!("BARYCENTER({}, {})", primary.name(), secondary.name())
            }
        }
    }

    /// The single body at this origin, when there is one.
    ///
    /// # Returns
    /// - `body`: The [`NAIFId`] of a [`FrameCenter::Body`], or `None` for
    ///   a [`FrameCenter::Barycenter`]
    ///
    /// # Examples
    /// ```
    /// use brahe::frames::FrameCenter;
    /// use brahe::spice::NAIFId;
    ///
    /// assert_eq!(FrameCenter::Body(NAIFId::Mars).body(), Some(NAIFId::Mars));
    /// assert_eq!(
    ///     FrameCenter::Barycenter {
    ///         primary: NAIFId::Earth,
    ///         secondary: NAIFId::Moon,
    ///     }
    ///     .body(),
    ///     None
    /// );
    /// ```
    pub fn body(&self) -> Option<NAIFId> {
        match self {
            FrameCenter::Body(id) => Some(*id),
            FrameCenter::Barycenter { .. } => None,
        }
    }
}

impl PartialEq for FrameCenter {
    fn eq(&self, other: &Self) -> bool {
        self.naif_id() == other.naif_id()
    }
}

impl Eq for FrameCenter {}

impl std::hash::Hash for FrameCenter {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.naif_id().hash(state);
    }
}

impl From<NAIFId> for FrameCenter {
    /// Wraps a [`NAIFId`] as a [`FrameCenter::Body`].
    ///
    /// # Arguments
    /// - `id`: NAIF ID of the body or system barycenter
    ///
    /// # Returns
    /// - `center`: The [`FrameCenter::Body`] holding `id`
    fn from(id: NAIFId) -> Self {
        FrameCenter::Body(id)
    }
}

impl From<i32> for FrameCenter {
    /// Wraps a raw NAIF integer ID as a [`FrameCenter::Body`], through the
    /// canonicalizing [`From<i32>`](NAIFId) conversion of [`NAIFId`].
    ///
    /// Any ID is accepted, including a self-assigned negative center and a
    /// synthetic synodic-barycenter ID (see
    /// [`synodic_barycenter_id`]); the latter stays a `Body` rather than
    /// being decoded back into a [`FrameCenter::Barycenter`].
    ///
    /// # Arguments
    /// - `id`: Raw NAIF integer ID
    ///
    /// # Returns
    /// - `center`: The [`FrameCenter::Body`] holding `id`
    fn from(id: i32) -> Self {
        FrameCenter::Body(NAIFId::from(id))
    }
}

impl fmt::Display for FrameCenter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl FromStr for FrameCenter {
    type Err = BraheError;

    /// Parses a NAIF body name, accepted alias, or integer ID string into a
    /// [`FrameCenter::Body`], delegating to [`NAIFId::from_name`].
    ///
    /// The `BARYCENTER(primary, secondary)` form [`FrameCenter::name`]
    /// prints for a [`FrameCenter::Barycenter`] is not parsed; construct
    /// that variant directly.
    ///
    /// # Arguments
    /// - `s`: A NAIF body name (e.g. `"MARS BARYCENTER"`), an accepted
    ///   alias (e.g. `"SSB"`), or an integer NAIF ID string
    ///
    /// # Returns
    /// - The matching [`FrameCenter::Body`], or a [`BraheError::Error`] if
    ///   `s` names no known body and is not a valid integer ID
    ///
    /// # Examples
    /// ```
    /// use brahe::frames::FrameCenter;
    /// use brahe::spice::NAIFId;
    ///
    /// assert_eq!(
    ///     "mars".parse::<FrameCenter>().unwrap(),
    ///     FrameCenter::Body(NAIFId::Mars)
    /// );
    /// assert!("BARYCENTER(EARTH, MOON)".parse::<FrameCenter>().is_err());
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        NAIFId::from_name(s).map(FrameCenter::Body)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use serial_test::parallel;

    use super::*;
    use crate::frames::SUN_EARTH_BARYCENTER_ID;

    #[test]
    #[parallel]
    fn test_framecenter_from_conversions() {
        assert_eq!(
            FrameCenter::from(NAIFId::Mars),
            FrameCenter::Body(NAIFId::Mars)
        );
        assert_eq!(FrameCenter::from(499), FrameCenter::Body(NAIFId::Mars));
        assert_eq!(FrameCenter::from(-42), FrameCenter::Body(NAIFId::Id(-42)));

        // A synthetic synodic-barycenter ID stays a `Body`, and compares equal
        // to the `Barycenter` that encodes to the same NAIF ID.
        let seb = FrameCenter::from(SUN_EARTH_BARYCENTER_ID);
        assert!(matches!(seb, FrameCenter::Body(NAIFId::Id(id)) if id == SUN_EARTH_BARYCENTER_ID));
        let barycenter = FrameCenter::Barycenter {
            primary: NAIFId::Sun,
            secondary: NAIFId::Earth,
        };
        assert_eq!(seb, barycenter);
        assert_eq!(seb.body(), Some(NAIFId::Id(SUN_EARTH_BARYCENTER_ID)));
        assert_eq!(barycenter.body(), None);
        let mut set = std::collections::HashSet::new();
        set.insert(seb);
        assert!(set.contains(&barycenter));
    }

    #[test]
    #[parallel]
    fn test_framecenter_naif_id() {
        assert_eq!(FrameCenter::Body(NAIFId::Earth).naif_id(), 399);
        assert_eq!(FrameCenter::Body(NAIFId::Id(-20001)).naif_id(), -20001);
        assert_eq!(
            FrameCenter::Barycenter {
                primary: NAIFId::Earth,
                secondary: NAIFId::Moon
            }
            .naif_id(),
            synodic_barycenter_id(399, 301)
        );
        assert_eq!(
            FrameCenter::Barycenter {
                primary: NAIFId::Sun,
                secondary: NAIFId::Earth
            }
            .naif_id(),
            SUN_EARTH_BARYCENTER_ID
        );
    }

    #[test]
    #[parallel]
    fn test_framecenter_name_and_body() {
        assert_eq!(FrameCenter::Body(NAIFId::Earth).name(), "EARTH");
        assert_eq!(
            FrameCenter::Body(NAIFId::EarthMoonBarycenter).name(),
            "EARTH MOON BARYCENTER"
        );
        assert_eq!(FrameCenter::Body(NAIFId::Id(-42)).name(), "-42");
        assert_eq!(
            FrameCenter::Barycenter {
                primary: NAIFId::Earth,
                secondary: NAIFId::Moon
            }
            .name(),
            "BARYCENTER(EARTH, MOON)"
        );
        assert_eq!(
            FrameCenter::Barycenter {
                primary: NAIFId::Earth,
                secondary: NAIFId::Moon
            }
            .to_string(),
            "BARYCENTER(EARTH, MOON)"
        );

        assert_eq!(FrameCenter::Body(NAIFId::Mars).body(), Some(NAIFId::Mars));
        assert_eq!(
            FrameCenter::Barycenter {
                primary: NAIFId::Earth,
                secondary: NAIFId::Moon
            }
            .body(),
            None
        );
    }

    #[test]
    #[parallel]
    fn test_framecenter_from_str() {
        for center in [
            FrameCenter::Body(NAIFId::Earth),
            FrameCenter::Body(NAIFId::MarsBarycenter),
            FrameCenter::Body(NAIFId::Id(2000001)),
        ] {
            assert_eq!(center.name().parse::<FrameCenter>().unwrap(), center);
        }
        assert_eq!(
            "mars".parse::<FrameCenter>().unwrap(),
            FrameCenter::Body(NAIFId::Mars)
        );
        assert_eq!(
            "-42".parse::<FrameCenter>().unwrap(),
            FrameCenter::Body(NAIFId::Id(-42))
        );
        assert!("BARYCENTER(EARTH, MOON)".parse::<FrameCenter>().is_err());
        assert!("not a body".parse::<FrameCenter>().is_err());
    }

    #[test]
    #[parallel]
    fn test_framecenter_serde_round_trip() {
        for center in [
            FrameCenter::Body(NAIFId::Earth),
            FrameCenter::Body(NAIFId::Id(-20001)),
            FrameCenter::Barycenter {
                primary: NAIFId::Earth,
                secondary: NAIFId::Moon,
            },
        ] {
            let s = serde_json::to_string(&center).unwrap();
            assert_eq!(serde_json::from_str::<FrameCenter>(&s).unwrap(), center);
        }
    }
}
