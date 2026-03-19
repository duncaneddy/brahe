/*!
 * XML writer for CCSDS OEM, OMM, and OPM messages.
 *
 * Stub — implemented in Stage 3 (OEM), Stage 5 (OMM), Stage 6 (OPM).
 */

use crate::utils::errors::BraheError;

/// Write an OEM message to XML format.
pub fn write_oem_xml(_oem: &crate::ccsds::oem::OEM) -> Result<String, BraheError> {
    Err(BraheError::Error(
        "OEM XML writer not yet implemented".to_string(),
    ))
}

/// Write an OMM message to XML format.
pub fn write_omm_xml(_omm: &crate::ccsds::omm::OMM) -> Result<String, BraheError> {
    Err(BraheError::Error(
        "OMM XML writer not yet implemented".to_string(),
    ))
}

/// Write an OPM message to XML format.
pub fn write_opm_xml(_opm: &crate::ccsds::opm::OPM) -> Result<String, BraheError> {
    Err(BraheError::Error(
        "OPM XML writer not yet implemented".to_string(),
    ))
}
