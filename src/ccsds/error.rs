/*!
 * CCSDS-specific error helpers.
 *
 * Provides convenience functions for creating [`BraheError`] variants with
 * consistent formatting for CCSDS message parsing and writing operations.
 */

use crate::utils::errors::BraheError;

/// Create a parse error for a CCSDS message type.
///
/// # Arguments
///
/// * `msg_type` - The CCSDS message type (e.g., "OEM", "OMM", "OPM")
/// * `detail` - Specific detail about what went wrong
///
/// # Returns
///
/// * `BraheError` - A `ParseError` variant with formatted message
pub(crate) fn ccsds_parse_error(msg_type: &str, detail: &str) -> BraheError {
    BraheError::ParseError(format!("CCSDS {}: {}", msg_type, detail))
}

/// Create an error for a missing required field in a CCSDS message.
///
/// # Arguments
///
/// * `msg_type` - The CCSDS message type (e.g., "OEM", "OMM", "OPM")
/// * `field` - The name of the missing required field
///
/// # Returns
///
/// * `BraheError` - A `ParseError` variant with formatted message
pub(crate) fn ccsds_missing_field(msg_type: &str, field: &str) -> BraheError {
    BraheError::ParseError(format!(
        "CCSDS {}: missing required field '{}'",
        msg_type, field
    ))
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_ccsds_parse_error() {
        let err = ccsds_parse_error("OEM", "invalid format version");
        match err {
            BraheError::ParseError(msg) => {
                assert_eq!(msg, "CCSDS OEM: invalid format version");
            }
            _ => panic!("Expected ParseError"),
        }
    }

    #[test]
    fn test_ccsds_missing_field() {
        let err = ccsds_missing_field("OPM", "OBJECT_NAME");
        match err {
            BraheError::ParseError(msg) => {
                assert_eq!(msg, "CCSDS OPM: missing required field 'OBJECT_NAME'");
            }
            _ => panic!("Expected ParseError"),
        }
    }
}
