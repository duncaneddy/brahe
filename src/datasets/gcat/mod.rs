/*!
 * GCAT (General Catalog of Artificial Space Objects) module.
 *
 * Provides access to Jonathan McDowell's GCAT catalogs:
 * - **SATCAT** (`satcat.tsv`) - Object catalog with physical/orbital properties
 * - **PSATCAT** (`psatcat.tsv`) - Payload-specific metadata
 *
 * Data is downloaded from `https://planet4589.org/space/gcat/tsv/cat/`
 * with 24-hour file-based caching by default.
 */

pub mod catalog;
pub mod fetch;
pub mod parser;
pub mod records;

pub use catalog::{GCATPsatcat, GCATSatcat};
pub use records::{GCATPsatcatRecord, GCATSatcatRecord};

use crate::utils::BraheError;

/// Default base URL for GCAT TSV files.
const DEFAULT_BASE_URL: &str = "https://planet4589.org/space/gcat/tsv/cat";

/// Default cache max age in seconds (24 hours).
const DEFAULT_CACHE_MAX_AGE: f64 = 86400.0;

/// Download and parse the GCAT SATCAT catalog.
///
/// Fetches the SATCAT TSV file from GCAT with file-based caching.
/// Returns a `GCATSatcat` container with search and filter methods.
///
/// # Arguments
///
/// * `cache_max_age` - Maximum cache age in seconds. Defaults to 86400 (24 hours).
///   Pass `Some(0.0)` to force a fresh download.
///
/// # Returns
///
/// * `Result<GCATSatcat, BraheError>` - Parsed SATCAT catalog container
///
/// # Examples
/// ```no_run
/// use brahe::datasets::gcat::get_satcat;
/// let satcat = get_satcat(None).unwrap();  // default 24h cache
/// println!("Loaded {} records", satcat.len());
/// ```
pub fn get_satcat(cache_max_age: Option<f64>) -> Result<GCATSatcat, BraheError> {
    let max_age = cache_max_age.unwrap_or(DEFAULT_CACHE_MAX_AGE);
    let url = format!("{}/satcat.tsv", DEFAULT_BASE_URL);
    let data = fetch::fetch_with_cache(&url, "satcat.tsv", max_age)?;
    let records = parser::parse_satcat_tsv(&data)?;
    Ok(GCATSatcat::new(records))
}

/// Download and parse the GCAT PSATCAT catalog.
///
/// Fetches the PSATCAT TSV file from GCAT with file-based caching.
/// Returns a `GCATPsatcat` container with search and filter methods.
///
/// # Arguments
///
/// * `cache_max_age` - Maximum cache age in seconds. Defaults to 86400 (24 hours).
///   Pass `Some(0.0)` to force a fresh download.
///
/// # Returns
///
/// * `Result<GCATPsatcat, BraheError>` - Parsed PSATCAT catalog container
///
/// # Examples
/// ```no_run
/// use brahe::datasets::gcat::get_psatcat;
/// let psatcat = get_psatcat(None).unwrap();
/// println!("Loaded {} records", psatcat.len());
/// ```
pub fn get_psatcat(cache_max_age: Option<f64>) -> Result<GCATPsatcat, BraheError> {
    let max_age = cache_max_age.unwrap_or(DEFAULT_CACHE_MAX_AGE);
    let url = format!("{}/psatcat.tsv", DEFAULT_BASE_URL);
    let data = fetch::fetch_with_cache(&url, "psatcat.tsv", max_age)?;
    let records = parser::parse_psatcat_tsv(&data)?;
    Ok(GCATPsatcat::new(records))
}
