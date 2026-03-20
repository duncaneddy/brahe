/*!
 * CCSDS KVN (Keyword=Value Notation) format support.
 *
 * KVN is the original text-based CCSDS format. Lines are either:
 * - Key=Value pairs (e.g., `OBJECT_NAME = ISS`)
 * - Comments (e.g., `COMMENT This is a comment`)
 * - Data lines (space-separated numeric values, e.g., ephemeris entries)
 * - Section markers (`META_START`, `META_STOP`, `COVARIANCE_START`, `COVARIANCE_STOP`)
 */

mod parser;
mod writer;

pub use parser::parse_cdm;
pub use parser::parse_oem;
pub use parser::parse_omm;
pub use parser::parse_opm;
pub use writer::write_cdm;
pub use writer::write_oem;
pub use writer::write_omm;
pub use writer::write_opm;
