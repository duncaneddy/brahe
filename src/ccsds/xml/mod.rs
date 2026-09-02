/*!
 * CCSDS XML format support.
 *
 * Each message type has its own reader and writer module; the intermediate
 * serde structs and block writers they share live in `common`.
 */

mod aem;
mod apm;
mod cdm;
mod common;
mod oem;
mod omm;
mod opm;

pub use aem::{parse_aem_xml, write_aem_xml};
pub use apm::{parse_apm_xml, write_apm_xml};
pub use cdm::{parse_cdm_xml, write_cdm_xml};
pub use oem::{parse_oem_xml, write_oem_xml};
pub use omm::{parse_omm_xml, write_omm_xml};
pub use opm::{parse_opm_xml, write_opm_xml};
