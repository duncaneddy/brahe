/*!
 * CCSDS XML format support.
 *
 * Stub — implemented in Stage 2 (OEM), Stage 5 (OMM), Stage 6 (OPM).
 */

mod parser;
mod writer;

pub use parser::parse_oem_xml;
pub use parser::parse_omm_xml;
pub use parser::parse_opm_xml;
pub use writer::write_oem_xml;
pub use writer::write_omm_xml;
pub use writer::write_opm_xml;
