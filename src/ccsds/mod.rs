/*!
 * CCSDS Orbit Data Message (ODM) support.
 *
 * This module provides parsing and writing of CCSDS standard orbit data
 * messages in three encoding formats: KVN (text), XML, and JSON.
 *
 * Supported message types:
 * - **OEM** (Orbit Ephemeris Message): Time-series of state vectors
 * - **OMM** (Orbit Mean-elements Message): Mean elements for SGP4 propagation
 * - **OPM** (Orbit Parameter Message): Single state vector with optional extras
 *
 * Reference: CCSDS 502.0-B-3 (Orbit Data Messages), April 2023
 *
 * # Usage
 *
 * ```no_run
 * use brahe::ccsds::oem::OEM;
 *
 * let oem = OEM::from_file("ephemeris.oem").unwrap();
 * for segment in &oem.segments {
 *     println!("Object: {}", segment.metadata.object_name);
 *     println!("States: {}", segment.states.len());
 * }
 * ```
 */

pub mod common;
pub mod error;
pub mod interop;
pub mod json;
pub mod kvn;
pub mod oem;
pub mod omm;
pub mod opm;
pub mod xml;

// Re-export commonly used types at the ccsds module level
pub use common::{CCSDSFormat, CCSDSRefFrame, CCSDSTimeSystem};
pub use oem::{OEM, OEMMetadata, OEMSegment, OEMStateVector};
pub use omm::OMM;
pub use opm::OPM;

/// Auto-detect the encoding format of a CCSDS message string.
///
/// Detection logic:
/// - Starts with `<?xml` or `<`: XML
/// - Starts with `{` or `[`: JSON
/// - Otherwise: KVN (default)
///
/// # Arguments
///
/// * `content` - String content of the CCSDS message
///
/// # Returns
///
/// * `CCSDSFormat` - Detected format
pub fn detect_format(content: &str) -> CCSDSFormat {
    let trimmed = content.trim_start();
    if trimmed.starts_with("<?xml") || trimmed.starts_with('<') {
        CCSDSFormat::XML
    } else if trimmed.starts_with('{') || trimmed.starts_with('[') {
        CCSDSFormat::JSON
    } else {
        CCSDSFormat::KVN
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_detect_format_kvn() {
        assert_eq!(detect_format("CCSDS_OEM_VERS = 3.0\n"), CCSDSFormat::KVN);
    }

    #[test]
    fn test_detect_format_xml() {
        assert_eq!(
            detect_format("<?xml version=\"1.0\"?>\n<oem>"),
            CCSDSFormat::XML
        );
        assert_eq!(detect_format("<oem>"), CCSDSFormat::XML);
    }

    #[test]
    fn test_detect_format_json() {
        assert_eq!(detect_format("{\"header\": {}}"), CCSDSFormat::JSON);
        assert_eq!(detect_format("[{\"header\": {}}]"), CCSDSFormat::JSON);
    }

    #[test]
    fn test_detect_format_whitespace() {
        assert_eq!(
            detect_format("  \n  CCSDS_OEM_VERS = 3.0"),
            CCSDSFormat::KVN
        );
        assert_eq!(detect_format("  \n  <?xml"), CCSDSFormat::XML);
    }
}
