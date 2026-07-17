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

/// Default base URL for star catalog data files.
pub const DEFAULT_BASE_URL: &str = "https://www.simplespacedata.org/star_catalog/cds";
