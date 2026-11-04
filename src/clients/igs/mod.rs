/*!
 * Client support for International GNSS Service (IGS) precise products.
 *
 * IGS analysis centers publish precise orbit, clock, and Earth-rotation
 * products on public archives such as NASA CDDIS. [`IGSProductFileName`]
 * parses the archive's two file-name conventions, the long
 * `CCCVPPPTTT_YYYYDDDHHMM_LEN_SMP_CNT.FMT` names used from GPS week 2238 and
 * the short `cccwwwwd.ext` names used before it, into typed fields.
 * [`IGSAnalysisCenter`] lists the analysis centers that publish them.
 * [`IGSProductsClient`] lists an archive's GPS week directories as
 * [`IGSProduct`] values, selected with an [`IGSProductFilter`].
 */

pub mod centers;
pub mod client;
pub mod file_name;
pub mod product;
pub mod source;

pub use centers::{IGSAnalysisCenter, list_analysis_centers};
pub use client::{DEFAULT_IGS_LISTING_MAX_AGE, IGSProductsClient};
pub use file_name::{
    IGSCompression, IGSContentType, IGSFileFormat, IGSNameStyle, IGSProductFileName,
    IGSSolutionType,
};
pub use product::{IGSProduct, IGSProductFilter};
pub use source::IGSProductSource;
