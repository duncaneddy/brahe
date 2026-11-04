/*!
 * Blocking client for IGS precise-product archives.
 */

use std::collections::HashSet;
use std::env;
use std::fmt;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::sleep;

use crate::clients::igs::file_name::IGSProductFileName;
use crate::clients::igs::product::{IGSProduct, IGSProductFilter};
use crate::clients::igs::source::IGSProductSource;
use crate::constants::SECONDS_PER_DAY;
use crate::time::Epoch;
use crate::utils::cache::is_older_than;
use crate::utils::download::{backoff_delay, is_retryable_error};
use crate::utils::fs::validate_path_component;
use crate::utils::network::{CacheDecision, cache_policy, ensure_online};
use crate::utils::{BraheError, atomic_write, get_igs_cache_dir};

/// Default seconds a cached directory listing is served before it is refreshed.
pub const DEFAULT_IGS_LISTING_MAX_AGE: f64 = 3600.0;

const DEFAULT_MAX_RETRIES: u32 = 3;

/// How long before a range start a product can begin and still cover the
/// range from an earlier archive week: the longest span of the products
/// issued more often than weekly (three days, for SIO ultra-rapid ERP files).
const RANGE_LOOKBACK: f64 = 3.0 * SECONDS_PER_DAY;

/// How long after a range end a product can be filed and still cover the
/// range: short ultra-rapid coverage starts one day before the named time
/// that sets the archive week.
const RANGE_LOOKAHEAD: f64 = SECONDS_PER_DAY;

/// Seconds in a GPS week.
const SECONDS_PER_WEEK: f64 = 7.0 * SECONDS_PER_DAY;

/// Environment variable holding the NASA Earthdata username used for CDDIS.
const CDDIS_USERNAME_VAR: &str = "CDDIS_EARTHDATA_USERNAME";

/// Environment variable holding the NASA Earthdata password used for CDDIS.
const CDDIS_PASSWORD_VAR: &str = "CDDIS_EARTHDATA_PASSWORD";

const GARNER_WARNING: &str = "Warning: CDDIS_EARTHDATA_USERNAME/CDDIS_EARTHDATA_PASSWORD not set; using the Garner (SIO) mirror over unencrypted HTTP. Set the variables or choose a source explicitly to silence this warning.";

static GARNER_WARNED: AtomicBool = AtomicBool::new(false);

/// Allows the next default selection that falls back to Garner to warn again.
#[cfg(test)]
pub(crate) fn reset_garner_warning() {
    GARNER_WARNED.store(false, Ordering::SeqCst);
}

/// Reads the CDDIS credentials from the environment when both are non-empty.
fn credentials_from_env() -> Option<(String, String)> {
    let username = env::var(CDDIS_USERNAME_VAR)
        .ok()
        .filter(|v| !v.is_empty())?;
    let password = env::var(CDDIS_PASSWORD_VAR)
        .ok()
        .filter(|v| !v.is_empty())?;
    Some((username, password))
}

/// Chooses the source for a client built without an explicit one.
///
/// An explicit source is returned as is. Otherwise CDDIS is chosen when
/// credentials are available and Garner when they are not, printing the
/// Garner warning the first time that happens in the process. The boolean is
/// `true` when this call printed the warning.
fn select_source(
    explicit: Option<IGSProductSource>,
    credentials: Option<&(String, String)>,
) -> (IGSProductSource, bool) {
    if let Some(source) = explicit {
        return (source, false);
    }
    if credentials.is_some() {
        return (IGSProductSource::CDDIS, false);
    }
    let warned = !GARNER_WARNED.swap(true, Ordering::SeqCst);
    if warned {
        eprintln!("{GARNER_WARNING}");
    }
    (IGSProductSource::Garner, warned)
}

/// Client for IGS precise-product archives.
///
/// The client reads from one of the sources in [`IGSProductSource`]. A client
/// built with [`IGSProductsClient::new`] uses CDDIS when the
/// `CDDIS_EARTHDATA_USERNAME` and `CDDIS_EARTHDATA_PASSWORD` environment
/// variables are both non-empty. Otherwise it prints a warning, once per
/// process, and uses the Garner mirror, which is served over unencrypted HTTP.
/// A client built with an explicit source or base URL uses exactly that
/// source and never falls back to another.
///
/// Requests are blocking, retried on transient failures, and honour
/// `BRAHE_NETWORK_MODE`. Listings and downloaded files are cached under
/// [`get_igs_cache_dir`](crate::utils::get_igs_cache_dir). The cache
/// directory is not coordinated across processes or across clients sharing
/// it.
///
/// Credentials are held privately and are never printed by `Debug`.
///
/// # Examples
///
/// ```
/// use brahe::igs::{IGSProductSource, IGSProductsClient};
///
/// let client = IGSProductsClient::with_source(IGSProductSource::BKG);
/// assert_eq!(client.source(), IGSProductSource::BKG);
/// assert_eq!(client.base_url(), "https://igs.bkg.bund.de/root_ftp/IGS/products/");
/// ```
#[derive(Clone)]
pub struct IGSProductsClient {
    source: IGSProductSource,
    base_url: String,
    credentials: Option<(String, String)>,
    listing_max_age: f64,
    max_retries: u32,
    agent: ureq::Agent,
}

impl fmt::Debug for IGSProductsClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IGSProductsClient")
            .field("source", &self.source)
            .field("base_url", &self.base_url)
            .field("has_credentials", &self.credentials.is_some())
            .field("listing_max_age", &self.listing_max_age)
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

impl Default for IGSProductsClient {
    fn default() -> Self {
        Self::new()
    }
}

impl IGSProductsClient {
    fn build(
        source: IGSProductSource,
        base_url: &str,
        credentials: Option<(String, String)>,
    ) -> Self {
        let mut base_url = base_url.to_string();
        if !base_url.ends_with('/') {
            base_url.push('/');
        }
        Self {
            source,
            base_url,
            credentials,
            listing_max_age: DEFAULT_IGS_LISTING_MAX_AGE,
            max_retries: DEFAULT_MAX_RETRIES,
            agent: ureq::Agent::new_with_defaults(),
        }
    }

    /// Creates a client with the default source.
    ///
    /// The source is CDDIS when `CDDIS_EARTHDATA_USERNAME` and
    /// `CDDIS_EARTHDATA_PASSWORD` are both set and non-empty, and the Garner
    /// mirror otherwise. The Garner fallback prints a warning once per
    /// process because that mirror is served over unencrypted HTTP.
    ///
    /// # Returns
    /// * `IGSProductsClient`: Client using the default source and base URL
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::IGSProductsClient;
    ///
    /// let client = IGSProductsClient::new();
    /// assert!(client.base_url().ends_with('/'));
    /// ```
    pub fn new() -> Self {
        let credentials = credentials_from_env();
        let (source, _) = select_source(None, credentials.as_ref());
        Self::build(source, source.default_base_url(), credentials)
    }

    /// Creates a client for a specific source.
    ///
    /// No warning is printed and no other source is tried. A CDDIS client
    /// takes its credentials from the environment variables
    /// `CDDIS_EARTHDATA_USERNAME` and `CDDIS_EARTHDATA_PASSWORD`.
    ///
    /// # Arguments
    /// * `source` - Archive to read from
    ///
    /// # Returns
    /// * `IGSProductsClient`: Client using the source's default base URL
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::{IGSProductSource, IGSProductsClient};
    ///
    /// let client = IGSProductsClient::with_source(IGSProductSource::Garner);
    /// assert_eq!(client.source(), IGSProductSource::Garner);
    /// ```
    pub fn with_source(source: IGSProductSource) -> Self {
        Self::with_base_url(source, source.default_base_url())
    }

    /// Creates a CDDIS client with explicit NASA Earthdata credentials.
    ///
    /// # Arguments
    /// * `username` - NASA Earthdata username
    /// * `password` - NASA Earthdata password
    ///
    /// # Returns
    /// * `IGSProductsClient`: CDDIS client using the default CDDIS base URL
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::{IGSProductSource, IGSProductsClient};
    ///
    /// let client = IGSProductsClient::with_cddis_credentials("user", "password");
    /// assert_eq!(client.source(), IGSProductSource::CDDIS);
    /// assert!(!format!("{client:?}").contains("password"));
    /// ```
    pub fn with_cddis_credentials(username: &str, password: &str) -> Self {
        Self::build(
            IGSProductSource::CDDIS,
            IGSProductSource::CDDIS.default_base_url(),
            Some((username.to_string(), password.to_string())),
        )
    }

    /// Creates a client for a source served from a different base URL, for
    /// mirrors and tests.
    ///
    /// A trailing `/` is added to `base_url` when it is missing. A CDDIS
    /// client takes its credentials from the environment variables
    /// `CDDIS_EARTHDATA_USERNAME` and `CDDIS_EARTHDATA_PASSWORD`.
    ///
    /// # Arguments
    /// * `source` - Archive the base URL stands in for
    /// * `base_url` - Directory URL that contains one subdirectory per GPS week
    ///
    /// # Returns
    /// * `IGSProductsClient`: Client using exactly this source and base URL
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::{IGSProductSource, IGSProductsClient};
    ///
    /// let client = IGSProductsClient::with_base_url(IGSProductSource::BKG, "http://127.0.0.1:8080/products");
    /// assert_eq!(client.base_url(), "http://127.0.0.1:8080/products/");
    /// ```
    pub fn with_base_url(source: IGSProductSource, base_url: &str) -> Self {
        let credentials = match source {
            IGSProductSource::CDDIS => credentials_from_env(),
            _ => None,
        };
        Self::build(source, base_url, credentials)
    }

    /// Sets how long a cached directory listing is served before it is refreshed.
    ///
    /// # Arguments
    /// * `listing_max_age` - Maximum listing age in seconds (default 3600)
    ///
    /// # Returns
    /// * `IGSProductsClient`: The client, for chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::{IGSProductSource, IGSProductsClient};
    ///
    /// let client = IGSProductsClient::with_source(IGSProductSource::Garner).with_cache_age(600.0);
    /// ```
    pub fn with_cache_age(mut self, listing_max_age: f64) -> Self {
        self.listing_max_age = listing_max_age;
        self
    }

    /// Sets the number of retries for transient failures.
    ///
    /// # Arguments
    /// * `max_retries` - Extra attempts after the first (default 3)
    ///
    /// # Returns
    /// * `IGSProductsClient`: The client, for chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::{IGSProductSource, IGSProductsClient};
    ///
    /// let client = IGSProductsClient::with_source(IGSProductSource::Garner).max_retries(0);
    /// ```
    pub fn max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Source the client reads from.
    ///
    /// # Returns
    /// * `IGSProductSource`: The archive
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::{IGSProductSource, IGSProductsClient};
    ///
    /// let client = IGSProductsClient::with_source(IGSProductSource::BKG);
    /// assert_eq!(client.source(), IGSProductSource::BKG);
    /// ```
    pub fn source(&self) -> IGSProductSource {
        self.source
    }

    /// Base URL of the product directory, with a trailing slash.
    ///
    /// # Returns
    /// * `&str`: The directory URL
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::{IGSProductSource, IGSProductsClient};
    ///
    /// let client = IGSProductsClient::with_source(IGSProductSource::Garner);
    /// assert_eq!(client.base_url(), "http://garner.ucsd.edu/pub/products/");
    /// ```
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Directory holding cached listings and product files.
    ///
    /// # Returns
    /// * `Ok(PathBuf)`: `$BRAHE_CACHE/igs`, created if missing
    /// * `Err(BraheError)`: If the cache directory cannot be created
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::{IGSProductSource, IGSProductsClient};
    ///
    /// let dir = IGSProductsClient::with_source(IGSProductSource::Garner).cache_dir().unwrap();
    /// assert!(dir.ends_with("igs"));
    /// ```
    pub fn cache_dir(&self) -> Result<PathBuf, BraheError> {
        Ok(PathBuf::from(get_igs_cache_dir()?))
    }

    /// Lists the GPS weeks that have a directory on the source.
    ///
    /// Reads the source's top-level listing and keeps the entries named by a
    /// four-digit week number, such as `0576/` or `2340/`. Other entries are
    /// ignored.
    ///
    /// # Returns
    /// * `Ok(Vec<u32>)`: GPS week numbers in ascending order
    /// * `Err(BraheError)`: If the listing cannot be served from the cache or fetched
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::igs::{IGSProductSource, IGSProductsClient};
    ///
    /// let weeks = IGSProductsClient::with_source(IGSProductSource::Garner).list_weeks().unwrap();
    /// assert_eq!(weeks.first(), Some(&576));
    /// ```
    pub fn list_weeks(&self) -> Result<Vec<u32>, BraheError> {
        Ok(listed_weeks(&self.listing(None)?))
    }

    /// Lists the file names in the GPS week directory containing `epoch`.
    ///
    /// Every file is returned as published, including names that are not IGS
    /// products such as `readme.txt`. Subdirectories, sort links, and the
    /// parent-directory link are left out.
    ///
    /// # Arguments
    /// * `epoch` - Any instant in the GPS week to list
    ///
    /// # Returns
    /// * `Ok(Vec<String>)`: File names in listing order
    /// * `Err(BraheError)`: If the week directory does not exist on the source, or the listing cannot be served from the cache or fetched
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::igs::{IGSProductSource, IGSProductsClient};
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let client = IGSProductsClient::with_source(IGSProductSource::Garner);
    /// let epoch = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// let files = client.list_files(epoch).unwrap();
    /// assert!(files.iter().any(|f| f == "COD0OPSFIN_20243150000_01D_05M_ORB.SP3.gz"));
    /// ```
    pub fn list_files(&self, epoch: Epoch) -> Result<Vec<String>, BraheError> {
        let week = epoch.gps_date().0;
        Ok(self
            .week_files(week)?
            .into_iter()
            .map(|(name, _)| name)
            .collect())
    }

    /// Lists the products in the GPS week directory containing `epoch` that
    /// match `filter`.
    ///
    /// Files whose names parse as IGS product names are returned; other files
    /// are skipped.
    ///
    /// # Arguments
    /// * `epoch` - Any instant in the GPS week to list
    /// * `filter` - Criteria the products must match
    ///
    /// # Returns
    /// * `Ok(Vec<IGSProduct>)`: Matching products in listing order
    /// * `Err(BraheError)`: If the week directory does not exist on the source, or the listing cannot be served from the cache or fetched
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::igs::{IGSContentType, IGSProductFilter, IGSProductSource, IGSProductsClient, IGSSolutionType};
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let client = IGSProductsClient::with_source(IGSProductSource::Garner);
    /// let epoch = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// let filter = IGSProductFilter::new()
    ///     .with_center("COD")
    ///     .with_solution(IGSSolutionType::Final)
    ///     .with_content(IGSContentType::Orbit);
    /// for product in client.list_products(epoch, &filter).unwrap() {
    ///     println!("{} {}", product.name, product.url);
    /// }
    /// ```
    pub fn list_products(
        &self,
        epoch: Epoch,
        filter: &IGSProductFilter,
    ) -> Result<Vec<IGSProduct>, BraheError> {
        let week = epoch.gps_date().0;
        self.week_products(week, filter)
    }

    /// Lists the products matching `filter` whose coverage overlaps
    /// `[start, end]`.
    ///
    /// A product covers `[start_epoch, start_epoch + span)`. A product with
    /// no span, such as a short-name weekly file, covers the GPS week of its
    /// archive directory. The week directories containing `start` through
    /// `end` are listed, plus those holding products that begin up to three
    /// days before `start` or are filed up to one day after `end` (short
    /// ultra-rapid products begin a day before their named time). Such an
    /// extra week that does not exist on the source is skipped.
    ///
    /// # Arguments
    /// * `start` - Start of the time range
    /// * `end` - End of the time range, inclusive
    /// * `filter` - Criteria the products must match
    ///
    /// # Returns
    /// * `Ok(Vec<IGSProduct>)`: Matching products ordered by start epoch, then name
    /// * `Err(BraheError)`: If `end` is before `start`, a week directory containing part of the range does not exist on the source, or a listing cannot be served from the cache or fetched
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::igs::{IGSContentType, IGSProductFilter, IGSProductSource, IGSProductsClient};
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let client = IGSProductsClient::with_source(IGSProductSource::Garner);
    /// let start = Epoch::from_datetime(2024, 11, 13, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// let end = Epoch::from_datetime(2024, 11, 20, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// let filter = IGSProductFilter::new().with_center("IGS").with_content(IGSContentType::Orbit);
    /// let products = client.list_products_between(start, end, &filter).unwrap();
    /// ```
    pub fn list_products_between(
        &self,
        start: Epoch,
        end: Epoch,
        filter: &IGSProductFilter,
    ) -> Result<Vec<IGSProduct>, BraheError> {
        if end < start {
            return Err(BraheError::Error(format!(
                "IGS product range end {end} is before its start {start}"
            )));
        }
        let first_week = start.gps_date().0;
        let last_week = end.gps_date().0;
        let lookback_week = (start - RANGE_LOOKBACK).gps_date().0;
        let lookahead_week = (end + RANGE_LOOKAHEAD).gps_date().0;

        let mut products = Vec::new();
        for week in lookback_week..=lookahead_week {
            let listing = if (first_week..=last_week).contains(&week) {
                self.listing(Some(week))?
            } else {
                match self.fetch_listing(Some(week))? {
                    Some(listing) => listing,
                    None => continue,
                }
            };
            products.extend(
                self.products_from_listing(week, &listing, filter)
                    .into_iter()
                    .filter(|product| covers(product, start, end)),
            );
        }
        products.sort_by(|a, b| {
            a.file_name
                .start_epoch()
                .cmp(&b.file_name.start_epoch())
                .then_with(|| a.name.cmp(&b.name))
        });
        Ok(products)
    }

    /// Lists the files of a week directory with their sizes.
    ///
    /// # Arguments
    /// * `week` - GPS week of the directory
    ///
    /// # Returns
    /// * `Ok(Vec<(String, Option<u64>)>)`: File names and sizes in bytes, in listing order
    /// * `Err(BraheError)`: If the directory does not exist or the listing cannot be obtained
    fn week_files(&self, week: u32) -> Result<Vec<(String, Option<u64>)>, BraheError> {
        Ok(self.parse_files(&self.listing(Some(week))?))
    }

    /// Lists the products of a week directory that match `filter`.
    ///
    /// # Arguments
    /// * `week` - GPS week of the directory
    /// * `filter` - Criteria the products must match
    ///
    /// # Returns
    /// * `Ok(Vec<IGSProduct>)`: Matching products in listing order
    /// * `Err(BraheError)`: If the directory does not exist or the listing cannot be obtained
    fn week_products(
        &self,
        week: u32,
        filter: &IGSProductFilter,
    ) -> Result<Vec<IGSProduct>, BraheError> {
        let listing = self.listing(Some(week))?;
        Ok(self.products_from_listing(week, &listing, filter))
    }

    /// Builds the products of a week listing that match `filter`, skipping
    /// files whose names are not IGS product names.
    ///
    /// # Arguments
    /// * `week` - GPS week of the listed directory
    /// * `listing` - Listing body as served
    /// * `filter` - Criteria the products must match
    ///
    /// # Returns
    /// * `Vec<IGSProduct>`: Matching products in listing order
    fn products_from_listing(
        &self,
        week: u32,
        listing: &str,
        filter: &IGSProductFilter,
    ) -> Vec<IGSProduct> {
        self.parse_files(listing)
            .into_iter()
            .filter_map(|(name, size)| {
                let file_name = IGSProductFileName::parse(&name).ok()?;
                Some(IGSProduct {
                    url: format!("{}{week:04}/{name}", self.base_url),
                    file_name,
                    name,
                    size,
                    gps_week: week,
                })
            })
            .filter(|product| filter.matches(product))
            .collect()
    }

    /// Reads the files of a week listing in the source's listing format.
    ///
    /// # Arguments
    /// * `listing` - Listing body as served
    ///
    /// # Returns
    /// * `Vec<(String, Option<u64>)>`: File names and sizes in bytes, in listing order
    fn parse_files(&self, listing: &str) -> Vec<(String, Option<u64>)> {
        match self.source {
            IGSProductSource::CDDIS => plain_listing_files(listing),
            IGSProductSource::Garner | IGSProductSource::BKG => html_listing_files(listing),
        }
    }

    /// URL of the top-level listing (`week` of `None`) or of a week listing.
    ///
    /// # Arguments
    /// * `week` - GPS week of the directory, or `None` for the top level
    ///
    /// # Returns
    /// * `String`: The listing URL
    fn listing_url(&self, week: Option<u32>) -> String {
        match (week, self.source) {
            (None, _) => self.base_url.clone(),
            (Some(week), IGSProductSource::CDDIS) => format!("{}{week:04}/*?list", self.base_url),
            (Some(week), _) => format!("{}{week:04}/", self.base_url),
        }
    }

    /// Description of a listing used in messages, naming the source and week.
    ///
    /// # Arguments
    /// * `week` - GPS week of the directory, or `None` for the top level
    ///
    /// # Returns
    /// * `String`: The description
    fn listing_resource(&self, week: Option<u32>) -> String {
        match week {
            Some(week) => format!("IGS {} listing for GPS week {week:04}", self.source),
            None => format!("IGS {} top-level listing", self.source),
        }
    }

    /// Cache file of the top-level listing (`week` of `None`) or of a week listing.
    ///
    /// # Arguments
    /// * `week` - GPS week of the directory, or `None` for the top level
    ///
    /// # Returns
    /// * `Ok(PathBuf)`: `listings/<source>/<week>.txt` or `listings/<source>/root.txt` under the cache directory
    /// * `Err(BraheError)`: If the cache directory cannot be created
    fn listing_cache_path(&self, week: Option<u32>) -> Result<PathBuf, BraheError> {
        let file = match week {
            Some(week) => format!("{week:04}.txt"),
            None => "root.txt".to_string(),
        };
        Ok(self
            .cache_dir()?
            .join("listings")
            .join(self.source.to_string().to_lowercase())
            .join(file))
    }

    /// Returns a listing, failing when its directory does not exist on the source.
    ///
    /// # Arguments
    /// * `week` - GPS week of the directory, or `None` for the top level
    ///
    /// # Returns
    /// * `Ok(String)`: The listing body
    /// * `Err(BraheError)`: If the directory does not exist, or the listing cannot be served from the cache or fetched
    fn listing(&self, week: Option<u32>) -> Result<String, BraheError> {
        self.fetch_listing(week)?.ok_or_else(|| {
            BraheError::Error(format!(
                "{} not found: {} answered 404",
                self.listing_resource(week),
                self.listing_url(week)
            ))
        })
    }

    /// Returns a listing from the cache while it is younger than
    /// `listing_max_age`, and from the source otherwise.
    ///
    /// `BRAHE_NETWORK_MODE` applies: `offline` serves a cached listing of any
    /// age, `offline-strict` serves only a fresh one, and both reject a
    /// listing that is not cached. A fetched listing is written to the cache.
    ///
    /// # Arguments
    /// * `week` - GPS week of the directory, or `None` for the top level
    ///
    /// # Returns
    /// * `Ok(Some(String))`: The listing body
    /// * `Ok(None)`: If the source answered 404
    /// * `Err(BraheError)`: If the network mode forbids the request, every attempt fails, or the cache cannot be read or written
    fn fetch_listing(&self, week: Option<u32>) -> Result<Option<String>, BraheError> {
        let resource = self.listing_resource(week);
        let path = self.listing_cache_path(week)?;
        if path.is_file() {
            let stale = is_older_than(&path, self.listing_max_age)?;
            if cache_policy(&resource, stale)? == CacheDecision::Serve {
                return fs::read_to_string(&path).map(Some).map_err(|e| {
                    BraheError::IoError(format!(
                        "Failed to read cached {resource} {}: {e}",
                        path.display()
                    ))
                });
            }
        }
        let Some(body) = self.fetch_text(&self.listing_url(week), &resource)? else {
            return Ok(None);
        };
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                BraheError::IoError(format!(
                    "Failed to create IGS listing cache directory {}: {e}",
                    parent.display()
                ))
            })?;
        }
        atomic_write(&path, body.as_bytes()).map_err(|e| {
            BraheError::IoError(format!(
                "Failed to write cached {resource} {}: {e}",
                path.display()
            ))
        })?;
        Ok(Some(body))
    }

    /// GETs `url` as text, retrying transient failures with backoff up to
    /// `max_retries` times.
    ///
    /// # Arguments
    /// * `url` - The URL to fetch
    /// * `resource` - Description of what is fetched, used in messages
    ///
    /// # Returns
    /// * `Ok(Some(String))`: The response body
    /// * `Ok(None)`: If the server answered 404
    /// * `Err(BraheError)`: If the network mode forbids the request or every attempt fails
    fn fetch_text(&self, url: &str, resource: &str) -> Result<Option<String>, BraheError> {
        let mut attempt = 0;
        loop {
            ensure_online(url, &format!("{resource} ({url})"))?;
            if attempt > 0 {
                sleep(backoff_delay(attempt));
            }
            match self.agent.get(url).call() {
                Ok(mut response) => match response.body_mut().read_to_string() {
                    Ok(body) => return Ok(Some(body)),
                    Err(_) if attempt < self.max_retries => {}
                    Err(e) => {
                        return Err(BraheError::IoError(format!(
                            "Failed to read {resource} from {url}: {e}"
                        )));
                    }
                },
                Err(ureq::Error::StatusCode(404)) => return Ok(None),
                Err(e) if attempt < self.max_retries && is_retryable_error(&e) => {}
                Err(e) => {
                    return Err(BraheError::IoError(format!(
                        "{resource} request {url} failed: {e}"
                    )));
                }
            }
            attempt += 1;
        }
    }
}

/// Whether a product's coverage overlaps `[start, end]`.
///
/// A product covers `[start_epoch, start_epoch + span)`, or its archive GPS
/// week when it has no span.
///
/// # Arguments
/// * `product` - Product to test
/// * `start` - Start of the range
/// * `end` - End of the range, inclusive
///
/// # Returns
/// * `bool`: `true` when the coverage and the range share an instant
fn covers(product: &IGSProduct, start: Epoch, end: Epoch) -> bool {
    let (begin, span) = match product.file_name.span() {
        Some(span) => (product.file_name.start_epoch(), span),
        None => (
            Epoch::from_gps_date(product.gps_week, 0.0),
            SECONDS_PER_WEEK,
        ),
    };
    begin <= end && begin + span > start
}

/// Extracts the `href` attribute values of an HTML page, in document order.
///
/// # Arguments
/// * `html` - Page source
///
/// # Returns
/// * `Vec<&str>`: The double-quoted `href` values
fn html_hrefs(html: &str) -> Vec<&str> {
    let mut hrefs = Vec::new();
    let mut rest = html;
    while let Some(start) = rest.find("href=\"") {
        rest = &rest[start + "href=\"".len()..];
        let Some(end) = rest.find('"') else {
            break;
        };
        hrefs.push(&rest[..end]);
        rest = &rest[end + 1..];
    }
    hrefs
}

/// Whether a listed name is a file inside the listed directory.
///
/// Sort links (`?C=N;O=D`), fragments, and anything containing a path
/// separator, such as the parent-directory link or a subdirectory `name/`,
/// are not.
///
/// # Arguments
/// * `name` - Listed name
///
/// # Returns
/// * `bool`: `true` for a plain file name
fn is_listed_file(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with(['?', '#'])
        && validate_path_component("listed file name", name).is_ok()
}

/// Reads the files of an Apache-style HTML directory listing.
///
/// The listing's human-readable sizes are not exact, so no size is reported.
///
/// # Arguments
/// * `html` - Listing page source
///
/// # Returns
/// * `Vec<(String, Option<u64>)>`: File names, each once, in listing order, with no size
fn html_listing_files(html: &str) -> Vec<(String, Option<u64>)> {
    let mut seen = HashSet::new();
    html_hrefs(html)
        .into_iter()
        .filter(|href| is_listed_file(href) && seen.insert(*href))
        .map(|href| (href.to_string(), None))
        .collect()
}

/// Reads the files of a plain-text listing with one `name size` line per
/// entry.
///
/// Blank lines, lines starting with `#`, and entries that are not plain file
/// names, such as subdirectories, are skipped. A size that is not an integer
/// is reported as unknown.
///
/// # Arguments
/// * `text` - Listing body
///
/// # Returns
/// * `Vec<(String, Option<u64>)>`: File names and sizes in bytes, in listing order
fn plain_listing_files(text: &str) -> Vec<(String, Option<u64>)> {
    text.lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.starts_with('#') {
                return None;
            }
            let mut fields = line.split_whitespace();
            let name = fields.next()?;
            if !is_listed_file(name) {
                return None;
            }
            let size = fields.next().and_then(|size| size.parse().ok());
            Some((name.to_string(), size))
        })
        .collect()
}

/// Reads the GPS weeks of a top-level listing: the entries whose last path
/// segment is a four-digit number, such as `0576/`.
///
/// # Arguments
/// * `html` - Listing page source
///
/// # Returns
/// * `Vec<u32>`: Week numbers in ascending order, without duplicates
fn listed_weeks(html: &str) -> Vec<u32> {
    let mut weeks: Vec<u32> = html_hrefs(html)
        .into_iter()
        .filter(|href| !href.starts_with('?'))
        .filter_map(|href| {
            let segment = href.trim_end_matches('/').rsplit('/').next()?;
            if segment.len() == 4 && segment.bytes().all(|b| b.is_ascii_digit()) {
                segment.parse().ok()
            } else {
                None
            }
        })
        .collect();
    weeks.sort_unstable();
    weeks.dedup();
    weeks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::igs::file_name::{IGSContentType, IGSFileFormat, IGSSolutionType};
    use crate::time::TimeSystem;
    use crate::utils::testing::{CacheRedirect, NetworkModeGuard};
    use httpmock::prelude::*;
    use serial_test::{parallel, serial};
    use std::time::{Duration, SystemTime};

    /// Sets the CDDIS credential variables for a test and restores the prior
    /// values on drop.
    struct EnvGuard {
        saved: [(&'static str, Option<String>); 2],
    }

    impl EnvGuard {
        fn set(username: Option<&str>, password: Option<&str>) -> Self {
            let saved = [
                (CDDIS_USERNAME_VAR, env::var(CDDIS_USERNAME_VAR).ok()),
                (CDDIS_PASSWORD_VAR, env::var(CDDIS_PASSWORD_VAR).ok()),
            ];
            Self::apply(CDDIS_USERNAME_VAR, username);
            Self::apply(CDDIS_PASSWORD_VAR, password);
            Self { saved }
        }

        fn apply(name: &str, value: Option<&str>) {
            // SAFETY: only used from #[serial] tests, so no other thread reads
            // the environment concurrently.
            unsafe {
                match value {
                    Some(v) => env::set_var(name, v),
                    None => env::remove_var(name),
                }
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (name, value) in &self.saved {
                Self::apply(name, value.as_deref());
            }
        }
    }

    #[test]
    #[serial]
    fn test_igs_products_client_select_source_with_credentials_is_cddis() {
        let _env = EnvGuard::set(Some("user"), Some("secret"));
        reset_garner_warning();
        let credentials = credentials_from_env();
        assert!(credentials.is_some());
        let (source, warned) = select_source(None, credentials.as_ref());
        assert_eq!(source, IGSProductSource::CDDIS);
        assert!(!warned);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_select_source_with_empty_variable_is_garner() {
        let _env = EnvGuard::set(Some("user"), Some(""));
        reset_garner_warning();
        let credentials = credentials_from_env();
        assert!(credentials.is_none());
        let (source, warned) = select_source(None, credentials.as_ref());
        assert_eq!(source, IGSProductSource::Garner);
        assert!(warned);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_select_source_with_missing_variable_is_garner() {
        let _env = EnvGuard::set(None, Some("secret"));
        reset_garner_warning();
        let (source, warned) = select_source(None, credentials_from_env().as_ref());
        assert_eq!(source, IGSProductSource::Garner);
        assert!(warned);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_new_warns_once() {
        let _env = EnvGuard::set(None, None);
        reset_garner_warning();
        let (_, first) = select_source(None, credentials_from_env().as_ref());
        let (_, second) = select_source(None, credentials_from_env().as_ref());
        assert!(first);
        assert!(!second);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_explicit_source_never_warns() {
        reset_garner_warning();
        let (source, warned) = select_source(Some(IGSProductSource::Garner), None);
        assert_eq!(source, IGSProductSource::Garner);
        assert!(!warned);
        // The explicit selection did not consume the one-time warning.
        let (_, warned) = select_source(None, None);
        assert!(warned);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_new_uses_environment() {
        reset_garner_warning();
        let _env = EnvGuard::set(Some("user"), Some("secret"));
        let client = IGSProductsClient::new();
        assert_eq!(client.source(), IGSProductSource::CDDIS);
        assert_eq!(
            client.base_url(),
            IGSProductSource::CDDIS.default_base_url()
        );
        drop(_env);

        let _env = EnvGuard::set(None, None);
        let client = IGSProductsClient::default();
        assert_eq!(client.source(), IGSProductSource::Garner);
        assert_eq!(
            client.base_url(),
            IGSProductSource::Garner.default_base_url()
        );
    }

    #[test]
    #[serial]
    fn test_igs_products_client_with_source_uses_default_base_url() {
        for source in [
            IGSProductSource::CDDIS,
            IGSProductSource::Garner,
            IGSProductSource::BKG,
        ] {
            let client = IGSProductsClient::with_source(source);
            assert_eq!(client.source(), source);
            assert_eq!(client.base_url(), source.default_base_url());
        }
    }

    #[test]
    #[serial]
    fn test_igs_products_client_with_cddis_credentials() {
        let _env = EnvGuard::set(None, None);
        let client = IGSProductsClient::with_cddis_credentials("user", "secret");
        assert_eq!(client.source(), IGSProductSource::CDDIS);
        assert_eq!(
            client.base_url(),
            IGSProductSource::CDDIS.default_base_url()
        );
        assert_eq!(
            client.credentials,
            Some(("user".to_string(), "secret".to_string()))
        );
    }

    #[test]
    #[serial]
    fn test_igs_products_client_with_cddis_source_reads_environment_credentials() {
        let _env = EnvGuard::set(Some("user"), Some("secret"));
        let client = IGSProductsClient::with_source(IGSProductSource::CDDIS);
        assert_eq!(
            client.credentials,
            Some(("user".to_string(), "secret".to_string()))
        );
        let client = IGSProductsClient::with_source(IGSProductSource::BKG);
        assert_eq!(client.credentials, None);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_with_base_url_normalises_trailing_slash() {
        let client =
            IGSProductsClient::with_base_url(IGSProductSource::BKG, "http://localhost:1/x/");
        assert_eq!(client.source(), IGSProductSource::BKG);
        assert_eq!(client.base_url(), "http://localhost:1/x/");

        let client =
            IGSProductsClient::with_base_url(IGSProductSource::BKG, "http://localhost:1/x");
        assert_eq!(client.base_url(), "http://localhost:1/x/");
    }

    #[test]
    #[serial]
    fn test_igs_products_client_builders() {
        let client = IGSProductsClient::with_source(IGSProductSource::Garner);
        assert_eq!(client.listing_max_age, DEFAULT_IGS_LISTING_MAX_AGE);
        assert_eq!(client.max_retries, 3);

        let client = client.with_cache_age(600.0).max_retries(0);
        assert_eq!(client.listing_max_age, 600.0);
        assert_eq!(client.max_retries, 0);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_debug_omits_credentials() {
        let client = IGSProductsClient::with_cddis_credentials("u", "secret");
        let debug = format!("{client:?}");
        assert!(!debug.contains("secret"));
        assert!(debug.contains("CDDIS"));
    }

    #[test]
    #[serial]
    fn test_igs_products_client_cache_dir() {
        let redirect = CacheRedirect::new();
        let dir = IGSProductsClient::with_source(IGSProductSource::Garner)
            .cache_dir()
            .unwrap();
        assert!(dir.ends_with("igs"));
        assert!(dir.is_dir());
        assert!(dir.starts_with(redirect.cache_path()));
    }

    const GARNER_PATH: &str = "/pub/products/";

    fn asset(name: &str) -> String {
        fs::read_to_string(format!("test_assets/igs/{name}")).unwrap()
    }

    fn gps(year: u32, month: u8, day: u8, hour: u8) -> Epoch {
        Epoch::from_datetime(year, month, day, hour, 0, 0.0, 0.0, TimeSystem::GPS)
    }

    /// An instant in GPS week 2340 (2024-11-10 through 2024-11-16).
    fn week_2340() -> Epoch {
        gps(2024, 11, 12, 12)
    }

    /// A minimal Apache-style listing of `names`, with a sort link and a
    /// parent link.
    fn apache_listing(names: &[&str]) -> String {
        let rows: String = names
            .iter()
            .map(|name| {
                format!(
                    "<tr><td><a href=\"{name}\">{name}</a></td><td align=\"right\">1.0M</td></tr>\n"
                )
            })
            .collect();
        format!(
            "<html><body><table>\n<tr><th><a href=\"?C=N;O=D\">Name</a></th></tr>\n\
             <tr><td><a href=\"/pub/products/\">Parent Directory</a></td></tr>\n{rows}</table></body></html>\n"
        )
    }

    fn mock_listing<'a>(server: &'a MockServer, path: &str, body: String) -> httpmock::Mock<'a> {
        server.mock(|when, then| {
            when.method(GET).path(path.to_string());
            then.status(200).body(body);
        })
    }

    fn garner_client(server: &MockServer) -> IGSProductsClient {
        IGSProductsClient::with_base_url(IGSProductSource::Garner, &server.url(GARNER_PATH))
    }

    fn names(products: &[IGSProduct]) -> Vec<&str> {
        products.iter().map(|p| p.name.as_str()).collect()
    }

    fn age_file(path: &std::path::Path, seconds: u64) {
        let file = fs::OpenOptions::new().write(true).open(path).unwrap();
        file.set_modified(SystemTime::now() - Duration::from_secs(seconds))
            .unwrap();
    }

    #[test]
    #[serial]
    fn test_igs_products_client_list_weeks() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let root = mock_listing(&server, GARNER_PATH, asset("garner_root.html"));
        let client = garner_client(&server);
        assert_eq!(
            client.list_weeks().unwrap(),
            vec![576, 577, 578, 579, 580, 2339, 2340, 2341, 2442, 2443]
        );
        root.assert_calls(1);
        let cached = client.cache_dir().unwrap().join("listings/garner/root.txt");
        assert_eq!(
            fs::read_to_string(cached).unwrap(),
            asset("garner_root.html")
        );
    }

    #[test]
    #[serial]
    fn test_igs_products_client_list_files_skips_listing_noise() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        for (source, fixture) in [
            (IGSProductSource::Garner, "garner_2340.html"),
            (IGSProductSource::BKG, "bkg_2340.html"),
        ] {
            let path = format!("/{source}/2340/");
            let week = mock_listing(&server, &path, asset(fixture));
            let client =
                IGSProductsClient::with_base_url(source, &server.url(format!("/{source}/")));
            let files = client.list_files(week_2340()).unwrap();
            week.assert_calls(1);

            assert!(files.contains(&"readme.txt".to_string()), "{source}");
            assert!(
                files.contains(&"IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.gz".to_string()),
                "{source}"
            );
            for file in &files {
                assert!(!file.starts_with('?'), "{source}: {file}");
                assert!(!file.contains('/'), "{source}: {file}");
                assert!(!file.starts_with("repro3"), "{source}: {file}");
            }
            let expected = html_hrefs(&asset(fixture))
                .into_iter()
                .filter(|href| !href.starts_with('?') && !href.contains('/'))
                .count();
            assert_eq!(files.len(), expected, "{source}");
        }
    }

    #[test]
    #[serial]
    fn test_igs_products_client_list_products_center_is_case_insensitive() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let week = mock_listing(&server, "/pub/products/2340/", asset("garner_2340.html"));
        let client = garner_client(&server);

        let lower = client
            .list_products(week_2340(), &IGSProductFilter::new().with_center("cod"))
            .unwrap();
        let upper = client
            .list_products(week_2340(), &IGSProductFilter::new().with_center("COD"))
            .unwrap();
        assert_eq!(lower, upper);
        assert!(lower.len() >= 20, "{}", lower.len());
        assert!(lower.iter().all(|p| p.file_name.center() == "COD"));
        assert!(names(&lower).contains(&"COD0OPSFIN_20243150000_01D_05M_ORB.SP3.gz"));
        assert!(!names(&lower).iter().any(|name| name.starts_with("ESA")));
        week.assert_calls(1);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_list_products_filters_solution_and_content() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        mock_listing(&server, "/pub/products/2340/", asset("garner_2340.html"));
        let client = garner_client(&server);

        let filter = IGSProductFilter::new()
            .with_solution(IGSSolutionType::Final)
            .with_content(IGSContentType::Orbit);
        let products = client.list_products(week_2340(), &filter).unwrap();
        assert!(!products.is_empty());
        for product in &products {
            assert_eq!(product.file_name.solution(), &IGSSolutionType::Final);
            assert_eq!(product.file_name.content(), &IGSContentType::Orbit);
        }
        let listed = names(&products);
        assert!(listed.contains(&"COD0OPSFIN_20243150000_01D_05M_ORB.SP3.gz"));
        assert!(listed.contains(&"IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.gz"));
        assert!(listed.contains(&"JPL0OPSFIN_20243150000_01D_05M_ORB.SP3.gz"));
        assert!(!listed.contains(&"COD0OPSRAP_20243150000_01D_05M_ORB.SP3.gz"));
        assert!(!listed.contains(&"COD0OPSFIN_20243150000_01D_30S_CLK.CLK.gz"));

        let clocks = client
            .list_products(
                week_2340(),
                &IGSProductFilter::new()
                    .with_center("ESA")
                    .with_format(IGSFileFormat::CLK),
            )
            .unwrap();
        assert_eq!(
            names(&clocks),
            vec![
                "ESA0OPSFIN_20243150000_01D_30S_CLK.CLK.gz",
                "ESA0OPSRAP_20243150000_01D_05M_CLK.CLK.gz",
            ]
        );
    }

    #[test]
    #[serial]
    fn test_igs_products_client_list_products_skips_unparseable_names() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        mock_listing(&server, "/pub/products/2340/", asset("garner_2340.html"));
        let client = garner_client(&server);

        let files = client.list_files(week_2340()).unwrap();
        let products = client
            .list_products(week_2340(), &IGSProductFilter::new())
            .unwrap();
        let listed = names(&products);
        for raw in ["readme.txt", "index315.html", "gpgga4.315.rap.Z"] {
            assert!(files.contains(&raw.to_string()), "{raw}");
            assert!(!listed.contains(&raw), "{raw}");
        }
        assert!(listed.contains(&"igr23400.sp3.Z"));
        assert!(products.len() < files.len());
        for product in &products {
            assert!(files.contains(&product.name));
            assert_eq!(product.file_name.to_string(), product.name);
            assert_eq!(
                product.url,
                format!("{}2340/{}", client.base_url(), product.name)
            );
            assert_eq!(product.size, None);
            assert_eq!(product.gps_week, 2340);
        }
    }

    #[test]
    #[serial]
    fn test_igs_products_client_cddis_listing_reports_sizes() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let week = server.mock(|when, then| {
            when.method(GET)
                .path("/archive/gnss/products/2340/*")
                .query_param_exists("list");
            then.status(200).body(asset("cddis_2340.txt"));
        });
        let client = IGSProductsClient::with_base_url(
            IGSProductSource::CDDIS,
            &server.url("/archive/gnss/products/"),
        );

        let files = client.list_files(week_2340()).unwrap();
        assert_eq!(files.len(), asset("cddis_2340.txt").lines().count());
        assert!(files.contains(&"readme.txt".to_string()));

        let products = client
            .list_products(
                week_2340(),
                &IGSProductFilter::new()
                    .with_center("cod")
                    .with_solution(IGSSolutionType::Final)
                    .with_content(IGSContentType::Orbit),
            )
            .unwrap();
        let orbit = products
            .iter()
            .find(|p| p.name == "COD0OPSFIN_20243150000_01D_05M_ORB.SP3.gz")
            .unwrap();
        assert_eq!(orbit.size, Some(617472));
        assert_eq!(
            orbit.url,
            server.url("/archive/gnss/products/2340/COD0OPSFIN_20243150000_01D_05M_ORB.SP3.gz")
        );
        assert!(products.iter().all(|p| p.size.is_some()));
        week.assert_calls(1);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_list_products_between_spans_weeks() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let before = mock_listing(&server, "/pub/products/2339/", apache_listing(&[]));
        let week_a = mock_listing(&server, "/pub/products/2340/", asset("garner_2340.html"));
        let week_b = mock_listing(
            &server,
            "/pub/products/2341/",
            apache_listing(&[
                "COD0OPSFIN_20243220000_01D_05M_ORB.SP3.gz",
                "COD0OPSFIN_20243240000_01D_05M_ORB.SP3.gz",
                "COD0OPSFIN_20243250000_01D_05M_ORB.SP3.gz",
                "COD0OPSFIN_20243260000_01D_05M_ORB.SP3.gz",
                "igs23417.erp.Z",
                "igu23413_00.sp3.Z",
                "igu23414_00.sp3.Z",
                "igu23414_18.sp3.Z",
                "readme.txt",
            ]),
        );
        let after = mock_listing(&server, "/pub/products/2342/", apache_listing(&[]));
        let client = garner_client(&server);
        let start = gps(2024, 11, 13, 0);
        let end = gps(2024, 11, 20, 12);

        let cod = client
            .list_products_between(start, end, &IGSProductFilter::new().with_center("cod"))
            .unwrap();
        assert_eq!(
            names(&cod),
            vec![
                "COD0OPSFIN_20243150000_07D_01D_ERP.ERP.gz",
                "COD0OPSFIN_20243150000_07D_01D_SUM.SUM.gz",
                "COD0OPSFIN_20243210000_01D_05M_ORB.SP3.gz",
                "COD0OPSFIN_20243220000_01D_05M_ORB.SP3.gz",
                "COD0OPSFIN_20243240000_01D_05M_ORB.SP3.gz",
                "COD0OPSFIN_20243250000_01D_05M_ORB.SP3.gz",
            ]
        );
        assert_eq!(cod[3].gps_week, 2341);
        assert!(
            cod[3]
                .url
                .ends_with("/2341/COD0OPSFIN_20243220000_01D_05M_ORB.SP3.gz")
        );

        let igs = client
            .list_products_between(start, end, &IGSProductFilter::new().with_center("IGS"))
            .unwrap();
        let listed = names(&igs);
        for kept in [
            "igs23417.erp.Z",
            "igu23406_18.sp3.Z",
            "igu23413_00.sp3.Z",
            "igu23414_00.sp3.Z",
            "IGS0OPSFIN_20243150000_07D_01D_ERP.ERP.gz",
            "IGS0OPSULT_20243211800_02D_15M_ORB.SP3.gz",
        ] {
            assert!(listed.contains(&kept), "{kept}");
        }
        for dropped in [
            "igu23414_18.sp3.Z",
            "igu23400_00.sp3.Z",
            "igr23400.sp3.Z",
            "IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.gz",
            "IGS0OPSULT_20243150000_02D_15M_ORB.SP3.gz",
        ] {
            assert!(!listed.contains(&dropped), "{dropped}");
        }
        assert!(
            igs.windows(2)
                .all(|w| w[0].file_name.start_epoch() <= w[1].file_name.start_epoch())
        );

        week_a.assert_calls(1);
        week_b.assert_calls(1);
        before.assert_calls(0);
        after.assert_calls(0);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_list_products_between_pads_and_skips_missing_padding_weeks() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let before = server.mock(|when, then| {
            when.method(GET).path("/pub/products/2339/");
            then.status(404);
        });
        let week = mock_listing(&server, "/pub/products/2340/", asset("garner_2340.html"));
        let after = mock_listing(
            &server,
            "/pub/products/2341/",
            apache_listing(&["igu23410_00.sp3.Z", "igu23410_18.sp3.Z"]),
        );
        let client = garner_client(&server);

        // Saturday of week 2340: the ultra-rapid file named Sunday 00:00 of
        // week 2341 starts at Saturday 00:00 and covers the range.
        let products = client
            .list_products_between(
                gps(2024, 11, 16, 6),
                gps(2024, 11, 16, 12),
                &IGSProductFilter::new().with_center("igs"),
            )
            .unwrap();
        let listed = names(&products);
        assert!(listed.contains(&"igu23410_00.sp3.Z"));
        assert!(!listed.contains(&"igu23410_18.sp3.Z"));
        after.assert_calls(1);
        before.assert_calls(0);

        // Sunday of week 2340: the earlier padding week is requested, and its
        // 404 is skipped.
        let products = client
            .list_products_between(
                gps(2024, 11, 10, 0),
                gps(2024, 11, 10, 6),
                &IGSProductFilter::new().with_center("COD"),
            )
            .unwrap();
        assert!(names(&products).contains(&"COD0OPSFIN_20243150000_01D_05M_ORB.SP3.gz"));
        before.assert_calls(1);
        week.assert_calls(1);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_list_products_between_rejects_reversed_range() {
        let client =
            IGSProductsClient::with_base_url(IGSProductSource::Garner, "http://127.0.0.1:1/");
        let err = client
            .list_products_between(
                gps(2024, 11, 13, 0),
                gps(2024, 11, 12, 0),
                &IGSProductFilter::new(),
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains("before its start"), "{err}");
    }

    #[test]
    #[serial]
    fn test_igs_products_client_missing_week_names_week_and_source() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let missing = server.mock(|when, then| {
            when.method(GET).path("/pub/products/2340/");
            then.status(404);
        });
        let client = garner_client(&server);
        let err = client
            .list_products(week_2340(), &IGSProductFilter::new())
            .unwrap_err()
            .to_string();
        assert!(err.contains("Garner"), "{err}");
        assert!(err.contains("2340"), "{err}");
        assert!(err.contains("404"), "{err}");

        let err = client
            .list_products_between(week_2340(), week_2340(), &IGSProductFilter::new())
            .unwrap_err()
            .to_string();
        assert!(err.contains("GPS week 2340"), "{err}");
        missing.assert_calls(2);
        assert!(
            !client
                .cache_dir()
                .unwrap()
                .join("listings/garner/2340.txt")
                .exists()
        );
    }

    #[test]
    #[serial]
    fn test_igs_products_client_listing_is_cached() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let week = mock_listing(&server, "/pub/products/2340/", asset("garner_2340.html"));
        let client = garner_client(&server);

        let first = client.list_files(week_2340()).unwrap();
        let second = client.list_files(week_2340()).unwrap();
        let products = client
            .list_products(week_2340(), &IGSProductFilter::new())
            .unwrap();
        assert_eq!(first, second);
        assert!(!products.is_empty());
        week.assert_calls(1);

        let path = client.cache_dir().unwrap().join("listings/garner/2340.txt");
        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            asset("garner_2340.html")
        );

        age_file(&path, 7200);
        client.list_files(week_2340()).unwrap();
        week.assert_calls(2);

        let client = garner_client(&server).with_cache_age(10_000.0);
        age_file(&path, 7200);
        client.list_files(week_2340()).unwrap();
        week.assert_calls(2);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_listing_offline_serves_cache() {
        let _cache = CacheRedirect::new();
        let online = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        mock_listing(&server, "/pub/products/2340/", asset("garner_2340.html"));
        let expected = garner_client(&server).list_files(week_2340()).unwrap();
        drop(online);

        let _mode = NetworkModeGuard::set(Some("offline"));
        let client = IGSProductsClient::with_base_url(
            IGSProductSource::Garner,
            "https://brahe-network-mode-test.invalid/pub/products/",
        )
        .max_retries(0);
        let path = client.cache_dir().unwrap().join("listings/garner/2340.txt");
        age_file(&path, 7200);
        assert_eq!(client.list_files(week_2340()).unwrap(), expected);

        let err = client
            .list_files(gps(2024, 11, 18, 0))
            .unwrap_err()
            .to_string();
        assert!(err.contains("BRAHE_NETWORK_MODE is offline"), "{err}");
        assert!(err.contains("GPS week 2341"), "{err}");
    }

    #[test]
    #[serial]
    fn test_igs_products_client_listing_offline_strict_rejects_stale_cache() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("offline-strict"));
        let client = IGSProductsClient::with_base_url(
            IGSProductSource::BKG,
            "https://brahe-network-mode-test.invalid/products/",
        );
        let path = client.cache_dir().unwrap().join("listings/bkg/2340.txt");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, asset("bkg_2340.html")).unwrap();
        assert!(
            client
                .list_files(week_2340())
                .unwrap()
                .contains(&"readme.txt".to_string())
        );

        age_file(&path, 7200);
        let err = client.list_files(week_2340()).unwrap_err().to_string();
        assert!(err.contains("offline-strict"), "{err}");
    }

    #[test]
    #[serial]
    fn test_igs_products_client_listing_retries_server_errors() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let failing = server.mock(|when, then| {
            when.method(GET).path("/pub/products/2340/");
            then.status(503);
        });
        let client = garner_client(&server).max_retries(1);
        let err = client.list_files(week_2340()).unwrap_err().to_string();
        assert!(err.contains("503"), "{err}");
        assert!(err.contains("Garner listing for GPS week 2340"), "{err}");
        failing.assert_calls(2);
    }

    #[test]
    #[parallel]
    fn test_igs_html_listing_files() {
        let html = concat!(
            "<a href=\"?C=N;O=D\">Name</a><a href=\"/pub/\">Parent</a>",
            "<a href=\"sub/\">sub/</a><a href=\"#top\">top</a><a href=\"..\">up</a>",
            "<a href=\"igr23400.sp3.Z\">igr23400.sp3.Z</a><a href=\"igr23400.sp3.Z\">again</a>",
            "<a href=\"readme.txt\">readme.txt</a><a href=\"unterminated"
        );
        assert_eq!(
            html_listing_files(html),
            vec![
                ("igr23400.sp3.Z".to_string(), None),
                ("readme.txt".to_string(), None),
            ]
        );
    }

    #[test]
    #[parallel]
    fn test_igs_plain_listing_files() {
        let text = "# comment\n\nigr23400.sp3.Z 1024\nsubdir/ 0\nreadme.txt\n  odd.txt  big \n";
        assert_eq!(
            plain_listing_files(text),
            vec![
                ("igr23400.sp3.Z".to_string(), Some(1024)),
                ("readme.txt".to_string(), None),
                ("odd.txt".to_string(), None),
            ]
        );
    }

    #[test]
    #[parallel]
    fn test_igs_listed_weeks() {
        let html = concat!(
            "<a href=\"?C=N;O=D\">Name</a><a href=\"/pub/\">Parent</a>",
            "<a href=\"2340/\">2340/</a><a href=\"0576/\">0576/</a><a href=\"860/\">860/</a>",
            "<a href=\"SAVE.1200/\">SAVE.1200/</a><a href=\"/pub/products/2341/\">2341</a>",
            "<a href=\"2340\">2340</a><a href=\"20p2\">20p2</a>"
        );
        assert_eq!(listed_weeks(html), vec![576, 2340, 2341]);
    }

    #[test]
    #[parallel]
    fn test_igs_covers_uses_archive_week_without_span() {
        let product = |name: &str, week: u32| IGSProduct {
            file_name: IGSProductFileName::parse(name).unwrap(),
            name: name.to_string(),
            url: String::new(),
            size: None,
            gps_week: week,
        };
        let weekly = product("igs23407.erp.Z", 2340);
        assert!(covers(&weekly, gps(2024, 11, 16, 23), gps(2024, 11, 17, 1)));
        assert!(!covers(&weekly, gps(2024, 11, 17, 0), gps(2024, 11, 18, 0)));
        assert!(!covers(&weekly, gps(2024, 11, 1, 0), gps(2024, 11, 9, 23)));

        let daily = product("COD0OPSFIN_20243150000_01D_05M_ORB.SP3.gz", 2340);
        assert!(covers(&daily, gps(2024, 11, 10, 0), gps(2024, 11, 10, 0)));
        assert!(covers(&daily, gps(2024, 11, 9, 0), gps(2024, 11, 10, 0)));
        assert!(!covers(&daily, gps(2024, 11, 11, 0), gps(2024, 11, 12, 0)));
    }
}
