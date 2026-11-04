/*!
 * IGS analysis centers.
 *
 * The table lists the three-letter center codes that appear in IGS product
 * file names, with the institution names from the IGS Analysis Center
 * Coordinator page, <https://igs.org/acc/#ac-list>. That page abbreviates a
 * few centers differently from their file-name codes (CODE is `COD`, USNO is
 * `USN`, GOP-RIGTC is `GOP`, ESA/ESOC is `ESA`); the file-name code is used
 * here. `IGS` is the code of the combined IGS products.
 */

/// An IGS analysis center.
///
/// # Examples
///
/// ```
/// use brahe::igs::IGSAnalysisCenter;
///
/// let center = IGSAnalysisCenter::from_code("cod").unwrap();
/// assert_eq!(center.code, "COD");
/// assert_eq!(center.name, "Center for Orbit Determination in Europe");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IGSAnalysisCenter {
    /// Three-letter code used in product file names, uppercase.
    pub code: &'static str,
    /// Full name of the institution.
    pub name: &'static str,
}

static ANALYSIS_CENTERS: [IGSAnalysisCenter; 14] = [
    IGSAnalysisCenter {
        code: "COD",
        name: "Center for Orbit Determination in Europe",
    },
    IGSAnalysisCenter {
        code: "EMR",
        name: "Natural Resources Canada",
    },
    IGSAnalysisCenter {
        code: "ESA",
        name: "European Space Agency/ESOC",
    },
    IGSAnalysisCenter {
        code: "GFZ",
        name: "GFZ Helmholtz Centre for Geosciences",
    },
    IGSAnalysisCenter {
        code: "GOP",
        name: "Geodetic Observatory Pecny",
    },
    IGSAnalysisCenter {
        code: "GRG",
        name: "Space geodesy team of the CNES",
    },
    IGSAnalysisCenter {
        code: "JGX",
        name: "Geospatial Information Authority of Japan and Japan Aerospace Exploration Agency",
    },
    IGSAnalysisCenter {
        code: "JPL",
        name: "Jet Propulsion Laboratory",
    },
    IGSAnalysisCenter {
        code: "MIT",
        name: "Massachusetts Institute of Technology",
    },
    IGSAnalysisCenter {
        code: "NGS",
        name: "NOAA/National Geodetic Survey",
    },
    IGSAnalysisCenter {
        code: "SIO",
        name: "Scripps Institution of Oceanography",
    },
    IGSAnalysisCenter {
        code: "USN",
        name: "U.S. Naval Observatory",
    },
    IGSAnalysisCenter {
        code: "WHU",
        name: "Wuhan University",
    },
    IGSAnalysisCenter {
        code: "IGS",
        name: "International GNSS Service (combined products)",
    },
];

impl IGSAnalysisCenter {
    /// Looks up an analysis center by its product file-name code.
    ///
    /// Matching is case-insensitive. Short-name combination and variant codes
    /// such as `IGL`, `IGV` and `COF` are not analysis centers in the table
    /// and return `None`.
    ///
    /// # Arguments
    /// * `code` - Three-letter center code, for example `"COD"` or `"cod"`
    ///
    /// # Returns
    /// * `Some(&IGSAnalysisCenter)`: The matching center
    /// * `None`: If the code is not in the table
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::IGSAnalysisCenter;
    ///
    /// assert_eq!(IGSAnalysisCenter::from_code("gfz").unwrap().code, "GFZ");
    /// assert!(IGSAnalysisCenter::from_code("XYZ").is_none());
    /// ```
    pub fn from_code(code: &str) -> Option<&'static IGSAnalysisCenter> {
        ANALYSIS_CENTERS
            .iter()
            .find(|center| center.code.eq_ignore_ascii_case(code))
    }
}

/// Returns every analysis center in the table.
///
/// # Returns
/// * `&'static [IGSAnalysisCenter]`: The centers, ordered by code with `IGS` last
///
/// # Examples
///
/// ```
/// use brahe::igs::list_analysis_centers;
///
/// let codes: Vec<&str> = list_analysis_centers().iter().map(|c| c.code).collect();
/// assert!(codes.contains(&"JPL"));
/// assert!(codes.contains(&"IGS"));
/// ```
pub fn list_analysis_centers() -> &'static [IGSAnalysisCenter] {
    &ANALYSIS_CENTERS
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use serial_test::parallel;

    #[test]
    #[parallel]
    fn test_igs_analysis_center_from_code_case_insensitive() {
        let center = IGSAnalysisCenter::from_code("cod").unwrap();
        assert_eq!(center.code, "COD");
        assert_eq!(center.name, "Center for Orbit Determination in Europe");
        assert_eq!(IGSAnalysisCenter::from_code("Cod"), Some(center));
    }

    #[test]
    #[parallel]
    fn test_igs_analysis_center_from_code_unknown() {
        assert!(IGSAnalysisCenter::from_code("XYZ").is_none());
        assert!(IGSAnalysisCenter::from_code("").is_none());
        assert!(IGSAnalysisCenter::from_code("CODE").is_none());
    }

    #[test]
    #[parallel]
    fn test_igs_list_analysis_centers_contents() {
        let codes: Vec<&str> = list_analysis_centers().iter().map(|c| c.code).collect();
        for expected in [
            "COD", "EMR", "ESA", "GFZ", "GRG", "JPL", "MIT", "NGS", "SIO", "USN", "WHU", "IGS",
        ] {
            assert!(codes.contains(&expected), "missing {}", expected);
        }
    }

    #[test]
    #[parallel]
    fn test_igs_list_analysis_centers_well_formed() {
        let centers = list_analysis_centers();
        for (i, center) in centers.iter().enumerate() {
            assert_eq!(center.code.len(), 3);
            assert!(center.code.chars().all(|c| c.is_ascii_uppercase()));
            assert!(!center.name.is_empty());
            assert!(
                centers[i + 1..]
                    .iter()
                    .all(|other| other.code != center.code),
                "duplicate code {}",
                center.code
            );
        }
    }
}
