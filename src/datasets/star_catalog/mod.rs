/*!
 * Star catalog module.
 *
 * Provides access to fixed-epoch star catalogs (FK5, Hipparcos, Tycho-2)
 * used for reference-frame realization and star-based attitude
 * determination. Unlike other datasets, star catalogs are static: once
 * published they do not change, so cached copies never go stale.
 *
 * Data is downloaded from `https://www.simplespacedata.org/star_catalog/cds`.
 */

pub(crate) mod fetch;
pub mod fk5;
pub mod hipparcos;
pub mod traits;
pub mod tycho2;

pub use fk5::{FK5Catalog, FK5Record};
pub use hipparcos::{HipparcosCatalog, HipparcosRecord};
pub use traits::StarRecord;
pub use tycho2::{Tycho2Catalog, Tycho2Record};

use crate::utils::BraheError;

/// Default base URL for star catalog data files.
pub const DEFAULT_BASE_URL: &str = "https://www.simplespacedata.org/star_catalog/cds";

/// Parse an optional `f64` value from a fixed-width text field.
///
/// Returns `None` for empty or whitespace-only fields, or if the trimmed
/// text cannot be parsed as a floating-point number.
pub(crate) fn opt_f64(value: &str) -> Option<f64> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        trimmed.parse::<f64>().ok()
    }
}

/// Parse an optional `String` value from a fixed-width text field.
///
/// Returns `None` for empty or whitespace-only fields.
pub(crate) fn opt_string(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Download and parse the FK5 star catalog.
///
/// Fetches the fixed-width FK5 catalog text file with file-based caching.
/// FK5 is a fixed, published catalog, so the default cache never expires.
///
/// # Arguments
///
/// * `cache_max_age` - Maximum cache age in seconds. `None` means the
///   cached copy never goes stale (the default; appropriate since FK5 does
///   not change once published). Pass `Some(0.0)` to force a fresh download.
///
/// # Returns
///
/// * `Result<FK5Catalog, BraheError>` - Parsed FK5 catalog container
///
/// # Examples
/// ```no_run
/// use brahe::datasets::star_catalog::get_fk5_catalog;
/// let catalog = get_fk5_catalog(None).unwrap();
/// println!("Loaded {} records", catalog.len());
/// ```
pub fn get_fk5_catalog(cache_max_age: Option<f64>) -> Result<FK5Catalog, BraheError> {
    let url = format!("{}/fk5/latest/FK5_Catalog.txt", DEFAULT_BASE_URL);
    let data = fetch::fetch_with_cache(&url, "FK5_Catalog.txt", cache_max_age)?;
    let records = fk5::parse_fk5_text(&data)?;
    Ok(FK5Catalog::new(records))
}

/// Download and parse the Hipparcos star catalog.
///
/// Fetches the pipe-delimited Hipparcos catalog text file with file-based
/// caching. Hipparcos is a fixed, published catalog, so the default cache
/// never expires.
///
/// # Arguments
///
/// * `cache_max_age` - Maximum cache age in seconds. `None` means the
///   cached copy never goes stale (the default; appropriate since Hipparcos
///   does not change once published). Pass `Some(0.0)` to force a fresh download.
///
/// # Returns
///
/// * `Result<HipparcosCatalog, BraheError>` - Parsed Hipparcos catalog container
///
/// # Examples
/// ```no_run
/// use brahe::datasets::star_catalog::get_hipparcos_catalog;
/// let catalog = get_hipparcos_catalog(None).unwrap();
/// println!("Loaded {} records", catalog.len());
/// ```
pub fn get_hipparcos_catalog(cache_max_age: Option<f64>) -> Result<HipparcosCatalog, BraheError> {
    let url = format!(
        "{}/hipparcos/latest/Hipparcos_Catalog.txt",
        DEFAULT_BASE_URL
    );
    let data = fetch::fetch_with_cache(&url, "Hipparcos_Catalog.txt", cache_max_age)?;
    let records = hipparcos::parse_hipparcos_text(&data)?;
    Ok(HipparcosCatalog::new(records))
}

/// Download and parse the Tycho-2 star catalog.
///
/// Fetches the pipe-delimited Tycho-2 catalog text file with file-based
/// caching. Tycho-2 is a fixed, published catalog, so the default cache
/// never expires. The source file is large (~526 MB, ~2.54 million
/// records), so the first call may take some time.
///
/// # Arguments
///
/// * `cache_max_age` - Maximum cache age in seconds. `None` means the
///   cached copy never goes stale (the default; appropriate since Tycho-2
///   does not change once published). Pass `Some(0.0)` to force a fresh download.
///
/// # Returns
///
/// * `Result<Tycho2Catalog, BraheError>` - Parsed Tycho-2 catalog container
///
/// # Examples
/// ```no_run
/// use brahe::datasets::star_catalog::get_tycho2_catalog;
/// let catalog = get_tycho2_catalog(None).unwrap();
/// println!("Loaded {} records", catalog.len());
/// ```
pub fn get_tycho2_catalog(cache_max_age: Option<f64>) -> Result<Tycho2Catalog, BraheError> {
    let url = format!("{}/tycho2/latest/Tycho2_Catalog.txt", DEFAULT_BASE_URL);
    let data = fetch::fetch_with_cache(&url, "Tycho2_Catalog.txt", cache_max_age)?;
    let records = tycho2::parse_tycho2_text(&data)?;
    Ok(Tycho2Catalog::new(records))
}
