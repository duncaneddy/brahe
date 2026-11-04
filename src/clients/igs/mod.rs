/*!
 * Client support for International GNSS Service (IGS) precise products.
 *
 * IGS analysis centers publish precise orbit, clock, and Earth-rotation
 * products on public archives such as NASA CDDIS. [`IGSProductFileName`]
 * parses the archive's two file-name conventions, the long
 * `CCCVPPPTTT_YYYYDDDHHMM_LEN_SMP_CNT.FMT` names used from GPS week 2238 and
 * the short `cccwwwwd.ext` names used before it, into typed fields.
 */

pub mod file_name;

pub use file_name::{
    IGSCompression, IGSContentType, IGSFileFormat, IGSNameStyle, IGSProductFileName,
    IGSSolutionType,
};
