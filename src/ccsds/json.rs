/*!
 * CCSDS JSON format support.
 *
 * Stub — implemented in Stage 2 (OEM), Stage 5 (OMM), Stage 6 (OPM).
 */

use crate::ccsds::error::ccsds_parse_error;
use crate::utils::errors::BraheError;

/// Parse an OEM message from JSON format.
pub fn parse_oem_json(_content: &str) -> Result<crate::ccsds::oem::OEM, BraheError> {
    Err(ccsds_parse_error("OEM", "JSON parsing not yet implemented"))
}

/// Parse an OMM message from JSON format.
pub fn parse_omm_json(_content: &str) -> Result<crate::ccsds::omm::OMM, BraheError> {
    Err(ccsds_parse_error("OMM", "JSON parsing not yet implemented"))
}

/// Parse an OPM message from JSON format.
pub fn parse_opm_json(_content: &str) -> Result<crate::ccsds::opm::OPM, BraheError> {
    Err(ccsds_parse_error("OPM", "JSON parsing not yet implemented"))
}

/// Write an OEM message to JSON format.
pub fn write_oem_json(_oem: &crate::ccsds::oem::OEM) -> Result<String, BraheError> {
    Err(BraheError::Error(
        "OEM JSON writer not yet implemented".to_string(),
    ))
}

/// Write an OMM message to JSON format.
pub fn write_omm_json(_omm: &crate::ccsds::omm::OMM) -> Result<String, BraheError> {
    Err(BraheError::Error(
        "OMM JSON writer not yet implemented".to_string(),
    ))
}

/// Write an OPM message to JSON format.
pub fn write_opm_json(_opm: &crate::ccsds::opm::OPM) -> Result<String, BraheError> {
    Err(BraheError::Error(
        "OPM JSON writer not yet implemented".to_string(),
    ))
}
