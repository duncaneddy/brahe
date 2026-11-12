/*!
 * IGS analysis centers.
 *
 * The variants are the three-letter center codes that appear in IGS product
 * file names, with the institution names from the IGS Analysis Center
 * Coordinator page, <https://igs.org/acc/#ac-list>. That page abbreviates a
 * few centers differently from their file-name codes (CODE is `COD`, USNO is
 * `USN`, GOP-RIGTC is `GOP`, ESA/ESOC is `ESA`); the file-name code is used
 * here. `IGS` is the code of the combined IGS products.
 */

/// Analysis center that produced an IGS product.
///
/// Codes that are not listed, such as the short-name combination codes `IGL`
/// and `IGV`, are kept, uppercased, in [`IGSAnalysisCenter::Other`].
///
/// # Examples
///
/// ```
/// use brahe::igs::IGSAnalysisCenter;
///
/// assert_eq!(IGSAnalysisCenter::parse("cod"), IGSAnalysisCenter::COD);
/// assert_eq!(IGSAnalysisCenter::JPL.code(), "JPL");
/// assert_eq!(IGSAnalysisCenter::JPL.name(), Some("Jet Propulsion Laboratory"));
/// assert_eq!(IGSAnalysisCenter::parse("igl"), IGSAnalysisCenter::Other("IGL".to_string()));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[allow(clippy::upper_case_acronyms)]
pub enum IGSAnalysisCenter {
    /// Center for Orbit Determination in Europe (`COD`).
    COD,
    /// Natural Resources Canada (`EMR`).
    EMR,
    /// European Space Agency/ESOC (`ESA`).
    ESA,
    /// GFZ Helmholtz Centre for Geosciences (`GFZ`).
    GFZ,
    /// Geodetic Observatory Pecny (`GOP`).
    GOP,
    /// Space geodesy team of the CNES (`GRG`).
    GRG,
    /// Geospatial Information Authority of Japan and Japan Aerospace Exploration Agency (`JGX`).
    JGX,
    /// Jet Propulsion Laboratory (`JPL`).
    JPL,
    /// Massachusetts Institute of Technology (`MIT`).
    MIT,
    /// NOAA/National Geodetic Survey (`NGS`).
    NGS,
    /// Scripps Institution of Oceanography (`SIO`).
    SIO,
    /// U.S. Naval Observatory (`USN`).
    USN,
    /// Wuhan University (`WHU`).
    WHU,
    /// International GNSS Service combined products (`IGS`).
    IGS,
    /// Any other center code, stored uppercased.
    Other(String),
}

impl IGSAnalysisCenter {
    /// Parses a center code.
    ///
    /// Matching is case-insensitive. Unlisted codes are kept, uppercased, in
    /// [`IGSAnalysisCenter::Other`].
    ///
    /// # Arguments
    /// * `code` - Three-letter center code, for example `"COD"` or `"cod"`
    ///
    /// # Returns
    /// * `IGSAnalysisCenter`: The parsed center
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::IGSAnalysisCenter;
    ///
    /// assert_eq!(IGSAnalysisCenter::parse("gfz"), IGSAnalysisCenter::GFZ);
    /// assert_eq!(IGSAnalysisCenter::parse("xyz"), IGSAnalysisCenter::Other("XYZ".to_string()));
    /// ```
    pub fn parse(code: &str) -> Self {
        let upper = code.to_ascii_uppercase();
        match upper.as_str() {
            "COD" => Self::COD,
            "EMR" => Self::EMR,
            "ESA" => Self::ESA,
            "GFZ" => Self::GFZ,
            "GOP" => Self::GOP,
            "GRG" => Self::GRG,
            "JGX" => Self::JGX,
            "JPL" => Self::JPL,
            "MIT" => Self::MIT,
            "NGS" => Self::NGS,
            "SIO" => Self::SIO,
            "USN" => Self::USN,
            "WHU" => Self::WHU,
            "IGS" => Self::IGS,
            _ => Self::Other(upper),
        }
    }

    /// Returns the three-letter code used in product file names.
    ///
    /// # Returns
    /// * `&str`: The uppercase center code, for example `"COD"`
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::IGSAnalysisCenter;
    ///
    /// assert_eq!(IGSAnalysisCenter::COD.code(), "COD");
    /// assert_eq!(IGSAnalysisCenter::Other("IGL".to_string()).code(), "IGL");
    /// ```
    pub fn code(&self) -> &str {
        match self {
            Self::COD => "COD",
            Self::EMR => "EMR",
            Self::ESA => "ESA",
            Self::GFZ => "GFZ",
            Self::GOP => "GOP",
            Self::GRG => "GRG",
            Self::JGX => "JGX",
            Self::JPL => "JPL",
            Self::MIT => "MIT",
            Self::NGS => "NGS",
            Self::SIO => "SIO",
            Self::USN => "USN",
            Self::WHU => "WHU",
            Self::IGS => "IGS",
            Self::Other(code) => code,
        }
    }

    /// Returns the name of the institution.
    ///
    /// # Returns
    /// * `Option<&'static str>`: The institution name, or `None` for [`IGSAnalysisCenter::Other`]
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::IGSAnalysisCenter;
    ///
    /// assert_eq!(IGSAnalysisCenter::WHU.name(), Some("Wuhan University"));
    /// assert_eq!(IGSAnalysisCenter::parse("xyz").name(), None);
    /// ```
    pub fn name(&self) -> Option<&'static str> {
        match self {
            Self::COD => Some("Center for Orbit Determination in Europe"),
            Self::EMR => Some("Natural Resources Canada"),
            Self::ESA => Some("European Space Agency/ESOC"),
            Self::GFZ => Some("GFZ Helmholtz Centre for Geosciences"),
            Self::GOP => Some("Geodetic Observatory Pecny"),
            Self::GRG => Some("Space geodesy team of the CNES"),
            Self::JGX => Some(
                "Geospatial Information Authority of Japan and Japan Aerospace Exploration Agency",
            ),
            Self::JPL => Some("Jet Propulsion Laboratory"),
            Self::MIT => Some("Massachusetts Institute of Technology"),
            Self::NGS => Some("NOAA/National Geodetic Survey"),
            Self::SIO => Some("Scripps Institution of Oceanography"),
            Self::USN => Some("U.S. Naval Observatory"),
            Self::WHU => Some("Wuhan University"),
            Self::IGS => Some("International GNSS Service (combined products)"),
            Self::Other(_) => None,
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use rstest::rstest;
    use serial_test::parallel;

    #[rstest]
    #[case("COD", IGSAnalysisCenter::COD)]
    #[case("EMR", IGSAnalysisCenter::EMR)]
    #[case("ESA", IGSAnalysisCenter::ESA)]
    #[case("GFZ", IGSAnalysisCenter::GFZ)]
    #[case("GOP", IGSAnalysisCenter::GOP)]
    #[case("GRG", IGSAnalysisCenter::GRG)]
    #[case("JGX", IGSAnalysisCenter::JGX)]
    #[case("JPL", IGSAnalysisCenter::JPL)]
    #[case("MIT", IGSAnalysisCenter::MIT)]
    #[case("NGS", IGSAnalysisCenter::NGS)]
    #[case("SIO", IGSAnalysisCenter::SIO)]
    #[case("USN", IGSAnalysisCenter::USN)]
    #[case("WHU", IGSAnalysisCenter::WHU)]
    #[case("IGS", IGSAnalysisCenter::IGS)]
    #[parallel]
    fn test_igs_analysis_center_code_round_trip(
        #[case] code: &str,
        #[case] center: IGSAnalysisCenter,
    ) {
        assert_eq!(IGSAnalysisCenter::parse(code), center);
        assert_eq!(IGSAnalysisCenter::parse(&code.to_ascii_lowercase()), center);
        assert_eq!(center.code(), code);
        assert!(center.name().is_some_and(|name| !name.is_empty()));
    }

    #[test]
    #[parallel]
    fn test_igs_analysis_center_parse_case_insensitive() {
        assert_eq!(IGSAnalysisCenter::parse("Cod"), IGSAnalysisCenter::COD);
        assert_eq!(
            IGSAnalysisCenter::COD.name(),
            Some("Center for Orbit Determination in Europe")
        );
    }

    #[test]
    #[parallel]
    fn test_igs_analysis_center_parse_unknown() {
        let center = IGSAnalysisCenter::parse("xyz");
        assert_eq!(center, IGSAnalysisCenter::Other("XYZ".to_string()));
        assert_eq!(center.code(), "XYZ");
        assert_eq!(center.name(), None);
        assert_eq!(
            IGSAnalysisCenter::parse("CODE"),
            IGSAnalysisCenter::Other("CODE".to_string())
        );
    }
}
