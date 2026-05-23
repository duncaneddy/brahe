//! ICGEM body enum and string-form conversion.

use serde::{Deserialize, Serialize};

/// A celestial body cataloged on ICGEM.
///
/// Known bodies map to specific enum variants; arbitrary bodies fall back to
/// `Other(name)` so the catalog stays useful if ICGEM adds new bodies before
/// Brahe is updated. Names stored in `Other` are normalized to lowercase to
/// give case-insensitive matching against the ICGEM "Object" column.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ICGEMBody {
    /// Earth — uses the ICGEM `tom_longtime` listing page.
    Earth,
    /// Earth's Moon.
    Moon,
    /// Mars.
    Mars,
    /// Venus.
    Venus,
    /// Dwarf planet Ceres.
    Ceres,
    /// Any other body present in the ICGEM celestial catalog, stored as a
    /// lowercase name (e.g. `Other("pluto")`).
    Other(String),
}

impl ICGEMBody {
    /// Parse a body name (case-insensitive). Known bodies map to their
    /// dedicated variants; anything else becomes `Other(lowercased)`.
    ///
    /// The ICGEM celestial catalog labels the Moon as "Moon (of the Earth)";
    /// any name that is exactly "moon" or starts with "moon " (with a trailing
    /// space) is mapped to `ICGEMBody::Moon`. The trailing-space guard prevents
    /// a hypothetical body like "moonlet-x" from being misclassified.
    pub fn from_name(s: &str) -> Self {
        let lower = s.trim().to_lowercase();
        match lower.as_str() {
            "earth" => ICGEMBody::Earth,
            "mars" => ICGEMBody::Mars,
            "venus" => ICGEMBody::Venus,
            "ceres" => ICGEMBody::Ceres,
            other if other == "moon" || other.starts_with("moon ") => ICGEMBody::Moon,
            other => ICGEMBody::Other(other.to_string()),
        }
    }

    /// Canonical display name (capitalized for known bodies, lowercase for `Other`).
    pub fn as_name(&self) -> &str {
        match self {
            ICGEMBody::Earth => "Earth",
            ICGEMBody::Moon => "Moon",
            ICGEMBody::Mars => "Mars",
            ICGEMBody::Venus => "Venus",
            ICGEMBody::Ceres => "Ceres",
            ICGEMBody::Other(name) => name.as_str(),
        }
    }

    /// True for Earth, which uses the `tom_longtime` listing page. Everything
    /// else uses `tom_celestial`.
    pub fn is_earth(&self) -> bool {
        matches!(self, ICGEMBody::Earth)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_from_name_known_bodies_case_insensitive() {
        assert_eq!(ICGEMBody::from_name("Earth"), ICGEMBody::Earth);
        assert_eq!(ICGEMBody::from_name("earth"), ICGEMBody::Earth);
        assert_eq!(ICGEMBody::from_name("  MOON "), ICGEMBody::Moon);
        assert_eq!(ICGEMBody::from_name("Mars"), ICGEMBody::Mars);
        assert_eq!(ICGEMBody::from_name("Venus"), ICGEMBody::Venus);
        assert_eq!(ICGEMBody::from_name("Ceres"), ICGEMBody::Ceres);
    }

    #[test]
    fn test_from_name_unknown_falls_through_to_other_lowercased() {
        assert_eq!(
            ICGEMBody::from_name("Pluto"),
            ICGEMBody::Other("pluto".to_string())
        );
        assert_eq!(
            ICGEMBody::from_name("Bennu"),
            ICGEMBody::Other("bennu".to_string())
        );
    }

    #[test]
    fn test_as_name_round_trip() {
        for body in [
            ICGEMBody::Earth,
            ICGEMBody::Moon,
            ICGEMBody::Mars,
            ICGEMBody::Venus,
            ICGEMBody::Ceres,
            ICGEMBody::Other("pluto".to_string()),
        ] {
            assert_eq!(ICGEMBody::from_name(body.as_name()), body);
        }
    }

    #[test]
    fn test_is_earth() {
        assert!(ICGEMBody::Earth.is_earth());
        assert!(!ICGEMBody::Moon.is_earth());
        assert!(!ICGEMBody::Other("earth-like".to_string()).is_earth());
    }
}
