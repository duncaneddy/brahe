/*!
 * Shared KVN tokenizing and writing helpers.
 *
 * KVN lines are either key=value pairs, comments, data lines of
 * space-separated numbers, or section markers such as `META_START`. Every
 * message parser consumes the same token stream, and the block writers
 * shared between messages live here alongside it.
 */

use crate::ccsds::common::{covariance_to_lower_triangular, round_ccsds_value};

/// A parsed KVN line token.
#[derive(Debug)]
pub(super) enum KVNToken {
    /// A key=value pair
    KeyValue { key: String, value: String },
    /// A comment line
    Comment(String),
    /// A data line (space-separated values)
    DataLine(Vec<String>),
    /// An empty or blank line
    Empty,
}

/// Section markers that appear as standalone keywords (no `=`).
const SECTION_MARKERS: &[&str] = &[
    "META_START",
    "META_STOP",
    "COVARIANCE_START",
    "COVARIANCE_STOP",
    "QUAT_START",
    "QUAT_STOP",
    "EULER_START",
    "EULER_STOP",
    "ANGVEL_START",
    "ANGVEL_STOP",
    "SPIN_START",
    "SPIN_STOP",
    "INERTIA_START",
    "INERTIA_STOP",
    "MAN_START",
    "MAN_STOP",
    "DATA_START",
    "DATA_STOP",
];

/// Tokenize a single KVN line.
pub(super) fn tokenize_line(line: &str) -> KVNToken {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return KVNToken::Empty;
    }

    // Check for COMMENT
    if let Some(rest) = trimmed.strip_prefix("COMMENT") {
        let text = rest.trim().to_string();
        return KVNToken::Comment(text);
    }

    // Check for standalone section markers (no '=' sign)
    if SECTION_MARKERS.contains(&trimmed) {
        return KVNToken::KeyValue {
            key: trimmed.to_string(),
            value: String::new(),
        };
    }

    // Check for key=value
    if let Some(eq_pos) = trimmed.find('=') {
        let key = trimmed[..eq_pos].trim().to_string();
        let value = trimmed[eq_pos + 1..].trim().to_string();
        return KVNToken::KeyValue { key, value };
    }

    // Otherwise it's a data line (space-separated tokens)
    let parts: Vec<String> = trimmed.split_whitespace().map(|s| s.to_string()).collect();
    if parts.is_empty() {
        KVNToken::Empty
    } else {
        KVNToken::DataLine(parts)
    }
}

/// Shared helper: write spacecraft parameters to KVN output.
pub(super) fn write_kvn_spacecraft_params(
    out: &mut String,
    sp: &Option<crate::ccsds::common::CCSDSSpacecraftParameters>,
) {
    if let Some(sp) = sp {
        for comment in &sp.comments {
            out.push_str(&format!("COMMENT {}\n", comment));
        }
        if let Some(mass) = sp.mass {
            out.push_str(&format!("MASS = {:.6}\n", mass));
        }
        if let Some(sra) = sp.solar_rad_area {
            out.push_str(&format!("SOLAR_RAD_AREA = {:.6}\n", sra));
        }
        if let Some(src) = sp.solar_rad_coeff {
            out.push_str(&format!("SOLAR_RAD_COEFF = {:.6}\n", src));
        }
        if let Some(da) = sp.drag_area {
            out.push_str(&format!("DRAG_AREA = {:.6}\n", da));
        }
        if let Some(dc) = sp.drag_coeff {
            out.push_str(&format!("DRAG_COEFF = {:.6}\n", dc));
        }
    }
}

/// Shared helper: write covariance as named key=value elements (for OMM KVN).
pub(super) fn write_kvn_covariance_elements(
    out: &mut String,
    matrix: &nalgebra::SMatrix<f64, 6, 6>,
) {
    // Convert m² → km² (factor 1e-6)
    let values = covariance_to_lower_triangular(matrix, 1e-6).map(round_ccsds_value);
    let names = [
        "CX_X",
        "CY_X",
        "CY_Y",
        "CZ_X",
        "CZ_Y",
        "CZ_Z",
        "CX_DOT_X",
        "CX_DOT_Y",
        "CX_DOT_Z",
        "CX_DOT_X_DOT",
        "CY_DOT_X",
        "CY_DOT_Y",
        "CY_DOT_Z",
        "CY_DOT_X_DOT",
        "CY_DOT_Y_DOT",
        "CZ_DOT_X",
        "CZ_DOT_Y",
        "CZ_DOT_Z",
        "CZ_DOT_X_DOT",
        "CZ_DOT_Y_DOT",
        "CZ_DOT_Z_DOT",
    ];
    for (i, name) in names.iter().enumerate() {
        out.push_str(&format!("{} = {:.15e}\n", name, values[i]));
    }
}

/// Shared helper: write user-defined parameters to KVN output.
pub(super) fn write_kvn_user_defined(
    out: &mut String,
    ud: &Option<crate::ccsds::common::CCSDSUserDefined>,
) {
    if let Some(ud) = ud {
        out.push('\n');
        for (k, v) in &ud.parameters {
            out.push_str(&format!("USER_DEFINED_{} = {}\n", k, v));
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    use crate::ccsds::kvn::parse_oem;

    use crate::ccsds::kvn::{write_cdm, write_oem, write_omm, write_opm};

    use serial_test::parallel;
    #[test]
    #[parallel]
    fn test_tokenize_empty() {
        match tokenize_line("") {
            KVNToken::Empty => {}
            _ => panic!("Expected Empty"),
        }
    }

    #[test]
    #[parallel]
    fn test_tokenize_comment() {
        match tokenize_line("COMMENT This is a test") {
            KVNToken::Comment(text) => assert_eq!(text, "This is a test"),
            _ => panic!("Expected Comment"),
        }
    }

    #[test]
    #[parallel]
    fn test_tokenize_key_value() {
        match tokenize_line("OBJECT_NAME = ISS") {
            KVNToken::KeyValue { key, value } => {
                assert_eq!(key, "OBJECT_NAME");
                assert_eq!(value, "ISS");
            }
            _ => panic!("Expected KeyValue"),
        }
    }

    #[test]
    #[parallel]
    fn test_tokenize_data_line() {
        match tokenize_line(
            "2017-04-11T22:31:43.121856 2906.275 4076.358 4561.364 -6.879 1.450 3.081",
        ) {
            KVNToken::DataLine(parts) => {
                assert_eq!(parts.len(), 7);
                assert_eq!(parts[0], "2017-04-11T22:31:43.121856");
            }
            _ => panic!("Expected DataLine"),
        }
    }

    #[test]
    #[parallel]
    fn test_parse_ccsds_datetime_doy_utc() {
        // Test DOY format with a supported time system (UTC)
        use crate::ccsds::common::{CCSDSTimeSystem, parse_ccsds_datetime};

        let epoch = parse_ccsds_datetime("1996-200T16:00:00", &CCSDSTimeSystem::UTC).unwrap();
        // 1996 is a leap year, DOY 200 = July 18
        let (y, m, d, h, min, s, _ns) = epoch.to_datetime();
        assert_eq!(y, 1996);
        assert_eq!(m, 7);
        assert_eq!(d, 18);
        assert_eq!(h, 16);
        assert_eq!(min, 0);
        assert!((s - 0.0).abs() < 1e-6);
    }

    // ------------------------------------------------------------------
    // Header keyword order
    //
    // CCSDS 502.0-B-3 section 7.4.8 (and 508.0-P-1.1 section 6.3.1.9) fix the
    // order of KVN assignments to that of the header tables, which place
    // COMMENT immediately after the version line and CLASSIFICATION after it.
    // ------------------------------------------------------------------

    /// Index of the first line whose trimmed form starts with `keyword`.
    ///
    /// # Arguments
    ///
    /// - `written`: Serialized KVN message to search
    /// - `keyword`: KVN keyword to locate, matched against the start of each
    ///   line after leading whitespace is trimmed
    ///
    /// # Returns
    ///
    /// - `usize`: Zero-based index of the first matching line
    ///
    /// # Panics
    ///
    /// Panics if no line starts with `keyword`.
    fn line_index_of(written: &str, keyword: &str) -> usize {
        written
            .lines()
            .position(|line| line.trim_start().starts_with(keyword))
            .unwrap_or_else(|| panic!("'{}' missing from written message", keyword))
    }

    /// Assert the version line comes first, then COMMENT, then CLASSIFICATION.
    ///
    /// # Arguments
    ///
    /// - `written`: Serialized KVN message whose header order is checked
    /// - `vers_keyword`: Message-specific version keyword, such as
    ///   `CCSDS_OEM_VERS`
    ///
    /// # Panics
    ///
    /// Panics if any of the three keywords is absent, or if they do not appear
    /// in the order fixed by the header tables.
    fn assert_header_order(written: &str, vers_keyword: &str) {
        let vers = line_index_of(written, vers_keyword);
        let comment = line_index_of(written, "COMMENT");
        let classification = line_index_of(written, "CLASSIFICATION");

        assert!(
            vers < comment && comment < classification,
            "expected {} < COMMENT < CLASSIFICATION, got {} < {} < {} in:\n{}",
            vers_keyword,
            vers,
            comment,
            classification,
            written
        );
    }

    #[test]
    #[serial_test::parallel]
    fn test_oem_write_header_comment_before_classification() {
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        let mut oem = parse_oem(&content).unwrap();
        oem.header.classification = Some("public, test-data".to_string());
        oem.header.comments = vec!["first header comment".to_string(), "second".to_string()];

        let written = write_oem(&oem).unwrap();
        assert_header_order(&written, "CCSDS_OEM_VERS");

        // Header comments must round-trip as header comments, not be absorbed
        // into the first segment's metadata comments.
        let reparsed = parse_oem(&written).unwrap();
        assert_eq!(reparsed.header.comments, oem.header.comments);
        assert_eq!(reparsed.header.classification, oem.header.classification);
        assert_eq!(
            reparsed.segments[0].metadata.comments,
            oem.segments[0].metadata.comments
        );
    }

    #[test]
    #[serial_test::parallel]
    fn test_omm_write_header_comment_before_classification() {
        let content = std::fs::read_to_string("test_assets/ccsds/omm/OMMExample2.txt").unwrap();
        let mut omm = crate::ccsds::kvn::parse_omm(&content).unwrap();
        omm.header.classification = Some("public, test-data".to_string());
        omm.header.comments = vec!["first header comment".to_string(), "second".to_string()];

        let written = write_omm(&omm).unwrap();
        assert_header_order(&written, "CCSDS_OMM_VERS");

        let reparsed = crate::ccsds::kvn::parse_omm(&written).unwrap();
        assert_eq!(reparsed.header.comments, omm.header.comments);
        assert_eq!(reparsed.header.classification, omm.header.classification);
        assert_eq!(reparsed.metadata.comments, omm.metadata.comments);
    }

    #[test]
    #[serial_test::parallel]
    fn test_opm_write_header_comment_before_classification() {
        let content = std::fs::read_to_string("test_assets/ccsds/opm/OPMExample1.txt").unwrap();
        let mut opm = crate::ccsds::kvn::parse_opm(&content).unwrap();
        opm.header.classification = Some("public, test-data".to_string());
        opm.header.comments = vec!["first header comment".to_string(), "second".to_string()];

        let written = write_opm(&opm).unwrap();
        assert_header_order(&written, "CCSDS_OPM_VERS");

        let reparsed = crate::ccsds::kvn::parse_opm(&written).unwrap();
        assert_eq!(reparsed.header.comments, opm.header.comments);
        assert_eq!(reparsed.header.classification, opm.header.classification);
        assert_eq!(reparsed.metadata.comments, opm.metadata.comments);
    }

    #[test]
    #[serial_test::parallel]
    fn test_cdm_write_header_comment_before_classification() {
        let content = std::fs::read_to_string("test_assets/ccsds/cdm/CDMExample1.txt").unwrap();
        let mut cdm = crate::ccsds::kvn::parse_cdm(&content).unwrap();
        cdm.header.classification = Some("public, test-data".to_string());
        cdm.header.comments = vec!["first header comment".to_string(), "second".to_string()];

        let written = write_cdm(&cdm).unwrap();
        assert_header_order(&written, "CCSDS_CDM_VERS");

        let reparsed = crate::ccsds::kvn::parse_cdm(&written).unwrap();
        assert_eq!(reparsed.header.comments, cdm.header.comments);
        assert_eq!(reparsed.header.classification, cdm.header.classification);
        assert_eq!(
            reparsed.relative_metadata.comments,
            cdm.relative_metadata.comments
        );
    }
}
