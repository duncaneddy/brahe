/*!
 * Blocking client for IGS precise-product archives.
 */

use std::cmp::Ordering as CmpOrdering;
use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::sleep;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use ureq::http::{Response, Uri};

use crate::clients::igs::file_name::{
    IGS_LONG_NAME_FIRST_WEEK, IGSCompression, IGSContentType, IGSFileFormat, IGSNameStyle,
    IGSProductFileName, IGSSolutionType, short_code,
};
use crate::clients::igs::product::{IGSProduct, IGSProductFilter, IGSRangeOptions};
use crate::clients::igs::source::IGSProductSource;
use crate::constants::SECONDS_PER_DAY;
use crate::datafiles::compression::{
    Compression, UNIX_COMPRESS_UNSUPPORTED, compression_of, decompress,
};
use crate::datafiles::sp3::SP3;
use crate::time::{Epoch, TimeSystem};
use crate::utils::cache::is_older_than;
use crate::utils::download::{backoff_delay, is_retryable_error};
use crate::utils::fs::{
    copy_file, is_within, move_file, resolve_destination, validate_path_component,
};
use crate::utils::network::{CacheDecision, cache_policy, ensure_online, network_mode};
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

/// NASA Earthdata login service that CDDIS redirects to for authentication.
const URS_URL: &str = "https://urs.earthdata.nasa.gov/";

/// Most redirects followed for one request.
const MAX_REDIRECTS: usize = 10;

/// Largest response body read into memory, in bytes (1 GiB).
const MAX_RESPONSE_BYTES: u64 = 1 << 30;

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
    urs_url: String,
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
        // Redirects are followed by the client itself, so that the network
        // mode is checked for every host contacted and CDDIS credentials
        // reach only the Earthdata login service.
        let agent = ureq::Agent::config_builder()
            .max_redirects(0)
            .build()
            .into();
        Self {
            source,
            base_url,
            credentials,
            listing_max_age: DEFAULT_IGS_LISTING_MAX_AGE,
            max_retries: DEFAULT_MAX_RETRIES,
            agent,
            urs_url: URS_URL.to_string(),
        }
    }

    /// Sets the URL prefix treated as the NASA Earthdata login service, the
    /// only place CDDIS credentials are sent.
    ///
    /// # Arguments
    /// * `urs_url` - Scheme, host, optional port, and path prefix of the login service
    ///
    /// # Returns
    /// * `IGSProductsClient`: The client, for chaining
    #[cfg(test)]
    pub(crate) fn with_urs_url(mut self, urs_url: &str) -> Self {
        self.urs_url = urs_url.to_string();
        self
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
    /// Listings are cached per source, so a client with a different base URL
    /// shares the listing cache of the source it stands in for.
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
    /// When `BRAHE_NETWORK_MODE` forbids fetching a listing, the extra weeks
    /// are served only from the cache and skipped when they are not cached,
    /// so offline results are limited to cached listings: a product filed in
    /// an uncached extra week can be absent. A week containing part of the
    /// range must still be cached.
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
                match self.fetch_listing(Some(week), true)? {
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

    /// Downloads a product into the cache and returns the cached path.
    ///
    /// Files are cached under `<cache_dir>/<source>/<week>/<name>`, with the
    /// source in lowercase, the product's four-digit archive week, and the
    /// name as published. Published products do not change, so a cached file
    /// is returned without a request and without an age check. When the file
    /// is not cached, `BRAHE_NETWORK_MODE` must allow the request. A CDDIS
    /// download signs in to NASA Earthdata with the client's credentials,
    /// which are sent only to the Earthdata login service.
    ///
    /// # Arguments
    /// * `product` - Product to download, as returned by a listing
    ///
    /// # Returns
    /// * `Ok(PathBuf)`: Path of the cached file, compressed as published
    /// * `Err(BraheError)`: If the product's name is not a plain IGS product file name, the file does not exist on the source, the network mode forbids the request, the Earthdata login fails, the body does not match the name's compression, or the cache cannot be written
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
    ///     .with_center("IGS")
    ///     .with_solution(IGSSolutionType::Final)
    ///     .with_content(IGSContentType::Orbit);
    /// let product = &client.list_products(epoch, &filter).unwrap()[0];
    /// let path = client.download(product).unwrap();
    /// println!("{}", path.display());
    /// ```
    pub fn download(&self, product: &IGSProduct) -> Result<PathBuf, BraheError> {
        let path = self.product_cache_path(product)?;
        let resource = self.product_resource(product);
        if path.is_file() {
            match cache_policy(&resource, false)? {
                CacheDecision::Serve => return Ok(path),
                CacheDecision::Refresh => {}
            }
        }
        let body = self.fetch_bytes(&product.url, &resource)?.ok_or_else(|| {
            BraheError::Error(format!(
                "{resource} not found: {} answered 404",
                product.url
            ))
        })?;
        self.check_product_body(product, &body)?;
        atomic_write(&path, &body).map_err(|e| {
            BraheError::IoError(format!(
                "Failed to write cached {resource} {}: {e}",
                path.display()
            ))
        })?;
        Ok(path)
    }

    /// Downloads a product, if it is not cached, and writes it to `destination`.
    ///
    /// An existing directory, or a path whose last component has no
    /// extension, is treated as a directory (created if missing). A gzip
    /// product is written decompressed, in a directory under its published
    /// name without `.gz`. Unix-compressed (`.Z`) and uncompressed products
    /// are written unchanged under their published name. A path with an
    /// extension is the file written. With `keep_cached` false the cached
    /// file is removed, so a later [`IGSProductsClient::download`] downloads
    /// it again; with `keep_cached` true it stays.
    ///
    /// # Arguments
    /// * `product` - Product to save, as returned by a listing
    /// * `destination` - Directory or file path, outside the cache directory
    /// * `keep_cached` - Leave the cached file in place
    ///
    /// # Returns
    /// * `Ok(PathBuf)`: Path written
    /// * `Err(BraheError)`: If `destination` resolves inside the cache directory, which is checked before anything is downloaded, the download fails, the gzip data is invalid, or the file cannot be written
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::igs::{IGSContentType, IGSProductFilter, IGSProductSource, IGSProductsClient};
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let client = IGSProductsClient::with_source(IGSProductSource::Garner);
    /// let epoch = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// let filter = IGSProductFilter::new().with_center("COD").with_content(IGSContentType::Orbit);
    /// let product = &client.list_products(epoch, &filter).unwrap()[0];
    /// let path = client.save(product, "./igs", false).unwrap();
    /// println!("{}", path.display());
    /// ```
    pub fn save<P: AsRef<Path>>(
        &self,
        product: &IGSProduct,
        destination: P,
        keep_cached: bool,
    ) -> Result<PathBuf, BraheError> {
        let destination = destination.as_ref();
        if is_within(destination, &self.cache_dir()?) {
            return Err(BraheError::Error(format!(
                "save destination {} is inside the IGS cache directory",
                destination.display()
            )));
        }
        let cached = self.download(product)?;
        let target = resolve_destination(destination, saved_name(product))?;
        write_saved(product, &cached, &target, keep_cached)?;
        Ok(target)
    }

    /// Downloads every product matching `filter` whose coverage overlaps
    /// `[start, end]`.
    ///
    /// The products are those [`IGSProductsClient::list_products_between`]
    /// returns, each downloaded as [`IGSProductsClient::download`] does.
    ///
    /// # Arguments
    /// * `start` - Start of the time range
    /// * `end` - End of the time range, inclusive
    /// * `filter` - Criteria the products must match
    ///
    /// # Returns
    /// * `Ok(Vec<PathBuf>)`: Cached paths, ordered by product start epoch, then name
    /// * `Err(BraheError)`: If the listing fails or any download fails
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::igs::{IGSContentType, IGSProductFilter, IGSProductSource, IGSProductsClient, IGSSolutionType};
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let client = IGSProductsClient::with_source(IGSProductSource::Garner);
    /// let start = Epoch::from_datetime(2024, 11, 13, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// let end = Epoch::from_datetime(2024, 11, 15, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// let filter = IGSProductFilter::new()
    ///     .with_center("IGS")
    ///     .with_solution(IGSSolutionType::Final)
    ///     .with_content(IGSContentType::Orbit);
    /// let paths = client.download_between(start, end, &filter).unwrap();
    /// ```
    pub fn download_between(
        &self,
        start: Epoch,
        end: Epoch,
        filter: &IGSProductFilter,
    ) -> Result<Vec<PathBuf>, BraheError> {
        self.list_products_between(start, end, filter)?
            .iter()
            .map(|product| self.download(product))
            .collect()
    }

    /// Downloads every product [`IGSProductsClient::download_between`] would
    /// and writes each into `directory`.
    ///
    /// Every product is downloaded before any is written. Each is then
    /// written as [`IGSProductsClient::save`] writes into a directory: gzip
    /// products decompressed under their name without `.gz`, other products
    /// unchanged. `directory` is created if missing. With `keep_cached`
    /// false the cached files are removed; with `keep_cached` true they stay.
    ///
    /// # Arguments
    /// * `start` - Start of the time range
    /// * `end` - End of the time range, inclusive
    /// * `filter` - Criteria the products must match
    /// * `directory` - Directory to write into, outside the cache directory
    /// * `keep_cached` - Leave the cached files in place
    ///
    /// # Returns
    /// * `Ok(Vec<PathBuf>)`: Paths written, ordered by product start epoch, then name
    /// * `Err(BraheError)`: If `directory` is an existing file or resolves inside the cache directory, both checked before anything is downloaded, or if the listing, a download, or a write fails
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::igs::{IGSContentType, IGSProductFilter, IGSProductSource, IGSProductsClient, IGSSolutionType};
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let client = IGSProductsClient::with_source(IGSProductSource::Garner);
    /// let start = Epoch::from_datetime(2024, 11, 13, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// let end = Epoch::from_datetime(2024, 11, 15, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// let filter = IGSProductFilter::new()
    ///     .with_center("IGS")
    ///     .with_solution(IGSSolutionType::Final)
    ///     .with_content(IGSContentType::Orbit);
    /// let paths = client.save_between(start, end, &filter, "./igs", false).unwrap();
    /// ```
    pub fn save_between<P: AsRef<Path>>(
        &self,
        start: Epoch,
        end: Epoch,
        filter: &IGSProductFilter,
        directory: P,
        keep_cached: bool,
    ) -> Result<Vec<PathBuf>, BraheError> {
        let directory = directory.as_ref();
        if directory.is_file() {
            return Err(BraheError::Error(format!(
                "save_between directory {} is a file, expected a directory",
                directory.display()
            )));
        }
        if is_within(directory, &self.cache_dir()?) {
            return Err(BraheError::Error(format!(
                "save_between directory {} is inside the IGS cache directory",
                directory.display()
            )));
        }
        let products = self.list_products_between(start, end, filter)?;
        let cached = products
            .iter()
            .map(|product| self.download(product))
            .collect::<Result<Vec<_>, _>>()?;
        fs::create_dir_all(directory).map_err(|e| {
            BraheError::IoError(format!("Failed to create {}: {e}", directory.display()))
        })?;
        products
            .iter()
            .zip(cached)
            .map(|(product, cached)| {
                let target = directory.join(saved_name(product));
                write_saved(product, &cached, &target, keep_cached)?;
                Ok(target)
            })
            .collect()
    }

    /// Downloads an SP3 orbit product, if it is not cached, and parses it.
    ///
    /// The product is downloaded as [`IGSProductsClient::download`] does and
    /// read with [`SP3::from_file`], which decompresses gzip data. Unix
    /// compress (`.Z`) data cannot be decompressed, so a `.Z` product is
    /// rejected before anything is downloaded; download it with
    /// [`IGSProductsClient::save`] and decompress it externally.
    ///
    /// # Arguments
    /// * `product` - SP3 product to download, as returned by a listing
    ///
    /// # Returns
    /// * `Ok(SP3)`: The parsed file
    /// * `Err(BraheError)`: If the product is not an SP3 file or is Unix-compressed, both checked before anything is downloaded, or if the download or parse fails
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
    ///     .with_center("IGS")
    ///     .with_solution(IGSSolutionType::Final)
    ///     .with_content(IGSContentType::Orbit);
    /// let product = &client.list_products(epoch, &filter).unwrap()[0];
    /// let sp3 = client.download_sp3(product).unwrap();
    /// println!("{} satellites", sp3.num_satellites());
    /// ```
    pub fn download_sp3(&self, product: &IGSProduct) -> Result<SP3, BraheError> {
        if !product.file_name.is_sp3() {
            return Err(BraheError::Error(format!(
                "{} is not an SP3 file (format {})",
                self.product_resource(product),
                product.file_name.format().code()
            )));
        }
        self.reject_unix_compress(product)?;
        self.parse_sp3(product, &self.download(product)?)
    }

    /// Downloads and parses the daily SP3 orbit files of one analysis center
    /// and solution that cover `[start, end]`.
    ///
    /// The files are the daily orbit files of every GPS day from the day
    /// containing `start` through the day containing `end`. With
    /// `options.pad` set, the days before and after are added when the source
    /// has their files, so that [`SP3Satellite::from_files`](crate::datafiles::sp3::SP3Satellite::from_files)
    /// uses centered fits at the span edges. Each day's file is found in its
    /// GPS week listing:
    ///
    /// - From GPS week 2238, the long-name one-day (`01D`) `ORB` SP3 file of
    ///   `center` and `solution` starting at 00:00. When several sampling
    ///   intervals are published, the coarsest is used unless
    ///   `options.sampling` is set, in which case only that interval is used.
    ///   Files of the `OPS` campaign are preferred, then the lowest version.
    /// - Before GPS week 2238, the short-name daily SP3 file of `center` and
    ///   `solution`, such as `cod21003.sp3` (rapid and ultra-rapid products
    ///   use the center's own code, such as `igr`).
    ///
    /// In both conventions, uncompressed and gzip files are preferred to
    /// Unix-compressed (`.Z`) ones before sampling is compared.
    ///
    /// Every file is chosen, and every error below is raised, before
    /// anything is downloaded. Ultra-rapid orbits are rejected: their issues
    /// every six hours overlap one another, so they do not chain into one
    /// time series. A day inside the span with no file, or whose file is only
    /// published Unix-compressed, is an error; such a padding day is skipped.
    /// A range crossing week 2238 returns short-name and long-name files
    /// together.
    ///
    /// # Arguments
    /// * `start` - Start of the time range
    /// * `end` - End of the time range, inclusive
    /// * `center` - Analysis center code, in any case, for example `"COD"`
    /// * `solution` - Solution type, for example [`IGSSolutionType::Final`]
    /// * `options` - Padding and sampling selection
    ///
    /// # Returns
    /// * `Ok(Vec<SP3>)`: Parsed daily files in time order
    /// * `Err(BraheError)`: If `solution` is ultra-rapid, `end` is before `start`, a week listing containing part of the span cannot be obtained, a day inside the span has no file on the source (including when its week directory answers 404) or only a Unix-compressed one, or a download or parse fails; a parse error names the product
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::datafiles::sp3::SP3Satellite;
    /// use brahe::igs::{IGSProductSource, IGSProductsClient, IGSRangeOptions, IGSSolutionType};
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let client = IGSProductsClient::with_source(IGSProductSource::Garner);
    /// let start = Epoch::from_datetime(2024, 11, 11, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// let end = Epoch::from_datetime(2024, 11, 12, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// let files = client
    ///     .download_sp3_range(start, end, "COD", IGSSolutionType::Final, &IGSRangeOptions::new())
    ///     .unwrap();
    /// let satellites = SP3Satellite::from_files(&files).unwrap();
    /// ```
    pub fn download_sp3_range(
        &self,
        start: Epoch,
        end: Epoch,
        center: &str,
        solution: IGSSolutionType,
        options: &IGSRangeOptions,
    ) -> Result<Vec<SP3>, BraheError> {
        let products = self.sp3_range_products(start, end, center, &solution, options)?;
        let paths = products
            .iter()
            .map(|product| self.download(product))
            .collect::<Result<Vec<_>, _>>()?;
        products
            .iter()
            .zip(&paths)
            .map(|(product, path)| self.parse_sp3(product, path))
            .collect()
    }

    /// Parses a downloaded SP3 product, naming the product in any error.
    ///
    /// # Arguments
    /// * `product` - Product the file was downloaded for
    /// * `path` - Cached file of the product
    ///
    /// # Returns
    /// * `Ok(SP3)`: The parsed file
    /// * `Err(BraheError)`: The parse or read error, of the same kind, with its message prefixed by the product's source and name
    fn parse_sp3(&self, product: &IGSProduct, path: &Path) -> Result<SP3, BraheError> {
        SP3::from_file(path).map_err(|e| {
            let resource = self.product_resource(product);
            match e {
                BraheError::ParseError(message) => {
                    BraheError::ParseError(format!("{resource}: {message}"))
                }
                BraheError::IoError(message) => {
                    BraheError::IoError(format!("{resource}: {message}"))
                }
                other => BraheError::Error(format!("{resource}: {other}")),
            }
        })
    }

    /// Chooses the daily SP3 orbit products [`IGSProductsClient::download_sp3_range`]
    /// downloads, without downloading them.
    ///
    /// # Arguments
    /// * `start` - Start of the time range
    /// * `end` - End of the time range, inclusive
    /// * `center` - Analysis center code, in any case
    /// * `solution` - Solution type
    /// * `options` - Padding and sampling selection
    ///
    /// # Returns
    /// * `Ok(Vec<IGSProduct>)`: One product per covered day, in time order
    /// * `Err(BraheError)`: As for [`IGSProductsClient::download_sp3_range`], other than download and parse failures
    fn sp3_range_products(
        &self,
        start: Epoch,
        end: Epoch,
        center: &str,
        solution: &IGSSolutionType,
        options: &IGSRangeOptions,
    ) -> Result<Vec<IGSProduct>, BraheError> {
        if *solution == IGSSolutionType::UltraRapid {
            return Err(BraheError::Error(format!(
                "IGS SP3 ranges of ultra-rapid ({}) orbits are not supported: successive issues overlap in time and do not chain; download single issues with download_sp3",
                solution.code()
            )));
        }
        if end < start {
            return Err(BraheError::Error(format!(
                "IGS SP3 range end {end} is before its start {start}"
            )));
        }
        let first_day = gps_day(start);
        let last_day = gps_day(end);
        let span_days = first_day..=last_day;
        let (low_day, high_day) = if options.pad {
            (first_day.saturating_sub(1), last_day + 1)
        } else {
            (first_day, last_day)
        };
        let span_weeks = first_day / 7..=last_day / 7;
        let filter = IGSProductFilter::new()
            .with_center(center)
            .with_solution(solution.clone())
            .with_content(IGSContentType::Orbit)
            .with_format(IGSFileFormat::SP3);

        let mut week_products = BTreeMap::new();
        let mut absent_weeks = HashSet::new();
        for week in low_day / 7..=high_day / 7 {
            let listing = self.fetch_listing(Some(week), !span_weeks.contains(&week))?;
            if listing.is_none() {
                absent_weeks.insert(week);
            }
            let products = listing
                .map(|listing| self.products_from_listing(week, &listing, &filter))
                .unwrap_or_default();
            week_products.insert(week, products);
        }

        let mut selected: Vec<IGSProduct> = Vec::new();
        let mut missing = Vec::new();
        for day in low_day..=high_day {
            let in_span = span_days.contains(&day);
            match daily_orbit(&week_products[&(day / 7)], day, options.sampling) {
                Some(product)
                    if product.file_name.compression() == IGSCompression::UnixCompress =>
                {
                    if in_span {
                        self.reject_unix_compress(product)?;
                    }
                }
                Some(product) => selected.push(product.clone()),
                None if in_span => missing.push(day),
                None => {}
            }
        }
        if let Some(&day) = missing.first() {
            let sampling = options.sampling.or_else(|| {
                selected
                    .iter()
                    .filter_map(|product| product.file_name.sampling())
                    .max_by(f64::total_cmp)
            });
            let week = day / 7;
            let reason = if absent_weeks.contains(&week) {
                format!(
                    "the GPS week {week:04} listing {} answered 404",
                    self.listing_url(Some(week))
                )
            } else {
                format!(
                    "the GPS week {week:04} listing {} has no daily {} {} SP3 orbit file for that day",
                    self.listing_url(Some(week)),
                    center.to_ascii_uppercase(),
                    solution.code()
                )
            };
            return Err(BraheError::Error(format!(
                "IGS {} product {} not found: {reason}",
                self.source,
                expected_daily_orbit_name(center, solution, day, sampling),
            )));
        }
        Ok(selected)
    }

    /// Fails for a Unix-compressed (`.Z`) product, which cannot be parsed.
    ///
    /// # Arguments
    /// * `product` - Product to check
    ///
    /// # Returns
    /// * `Ok(())`: The product is not Unix-compressed
    /// * `Err(BraheError)`: An error naming the product and stating [`UNIX_COMPRESS_UNSUPPORTED`]
    fn reject_unix_compress(&self, product: &IGSProduct) -> Result<(), BraheError> {
        if product.file_name.compression() == IGSCompression::UnixCompress {
            return Err(BraheError::Error(format!(
                "{}: {UNIX_COMPRESS_UNSUPPORTED}",
                self.product_resource(product)
            )));
        }
        Ok(())
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
        self.fetch_listing(week, false)?.ok_or_else(|| {
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
    /// * `optional` - Whether a listing the network mode does not allow to be served or fetched is reported as absent instead of as an error
    ///
    /// # Returns
    /// * `Ok(Some(String))`: The listing body
    /// * `Ok(None)`: If the source answered 404, or `optional` is set and the network mode forbids serving or fetching the listing
    /// * `Err(BraheError)`: If the network mode forbids the request and `optional` is not set, `BRAHE_NETWORK_MODE` holds an unrecognized value, every attempt fails, or the cache cannot be read or written
    fn fetch_listing(
        &self,
        week: Option<u32>,
        optional: bool,
    ) -> Result<Option<String>, BraheError> {
        let resource = self.listing_resource(week);
        let path = self.listing_cache_path(week)?;
        let url = self.listing_url(week);
        // Surface an unrecognized mode before refusals are treated as absence.
        network_mode()?;
        if path.is_file() {
            let stale = is_older_than(&path, self.listing_max_age)?;
            match cache_policy(&resource, stale) {
                Ok(CacheDecision::Serve) => {
                    return fs::read_to_string(&path).map(Some).map_err(|e| {
                        BraheError::IoError(format!(
                            "Failed to read cached {resource} {}: {e}",
                            path.display()
                        ))
                    });
                }
                Ok(CacheDecision::Refresh) => {}
                Err(_) if optional => return Ok(None),
                Err(e) => return Err(e),
            }
        }
        if optional && ensure_online(&url, &format!("{resource} ({url})")).is_err() {
            return Ok(None);
        }
        let Some(body) = self.fetch_text(&url, &resource)? else {
            return Ok(None);
        };
        if self.source == IGSProductSource::CDDIS && week.is_some() && is_html(body.as_bytes()) {
            return Err(BraheError::Error(format!(
                "{resource} from {url} is an HTML page, not a file list{}",
                self.login_hint()
            )));
        }
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

    /// Cache path of a product file.
    ///
    /// # Arguments
    /// * `product` - Product whose cache path is built
    ///
    /// # Returns
    /// * `Ok(PathBuf)`: `<source>/<week>/<name>` under the cache directory
    /// * `Err(BraheError)`: If the product's name is not a plain file name equal to its parsed file name, or the cache directory cannot be created
    fn product_cache_path(&self, product: &IGSProduct) -> Result<PathBuf, BraheError> {
        if !is_listed_file(&product.name) || product.file_name.to_string() != product.name {
            return Err(BraheError::Error(format!(
                "invalid IGS product name '{}': expected a plain file name equal to its parsed file name '{}'",
                product.name, product.file_name
            )));
        }
        Ok(self
            .cache_dir()?
            .join(self.source.to_string().to_lowercase())
            .join(format!("{:04}", product.gps_week))
            .join(&product.name))
    }

    /// Description of a product used in messages, naming the source and file.
    ///
    /// # Arguments
    /// * `product` - Product described
    ///
    /// # Returns
    /// * `String`: The description
    fn product_resource(&self, product: &IGSProduct) -> String {
        format!("IGS {} product {}", self.source, product.name)
    }

    /// Checks that a downloaded body is the file its name describes before
    /// it is cached.
    ///
    /// A `.gz` or `.Z` product must start with that format's magic bytes, and
    /// a `.gz` product must decompress completely within the decompression
    /// size limit. An uncompressed CDDIS product must not be an HTML page.
    ///
    /// # Arguments
    /// * `product` - Product the body was downloaded for
    /// * `body` - Response body
    ///
    /// # Returns
    /// * `Ok(())`: The body matches the product
    /// * `Err(BraheError)`: If the body does not match, which for CDDIS usually means the Earthdata login failed
    fn check_product_body(&self, product: &IGSProduct, body: &[u8]) -> Result<(), BraheError> {
        let expected = match product.file_name.compression() {
            IGSCompression::Gzip => Some((Compression::Gzip, "gzip")),
            IGSCompression::UnixCompress => Some((Compression::UnixCompress, "Unix compress")),
            IGSCompression::None => None,
        };
        let mismatch = match expected {
            Some((compression, format)) if compression_of(body) != compression => {
                Some(format!("is not {format} data"))
            }
            Some((Compression::Gzip, _)) => decompress(body)
                .err()
                .map(|e| format!("is not valid gzip data ({e})")),
            None if self.source == IGSProductSource::CDDIS && is_html(body) => {
                Some("is an HTML page".to_string())
            }
            _ => None,
        };
        match mismatch {
            Some(detail) => Err(BraheError::Error(format!(
                "{} downloaded from {} {detail}{}",
                self.product_resource(product),
                product.url,
                self.login_hint()
            ))),
            None => Ok(()),
        }
    }

    /// Hint appended to errors that a failed Earthdata login can cause.
    ///
    /// # Returns
    /// * `&str`: The hint for CDDIS, empty for other sources
    fn login_hint(&self) -> &'static str {
        match self.source {
            IGSProductSource::CDDIS => {
                "; the NASA Earthdata login likely failed, so check CDDIS_EARTHDATA_USERNAME and CDDIS_EARTHDATA_PASSWORD"
            }
            IGSProductSource::Garner | IGSProductSource::BKG => "",
        }
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
    /// * `Err(BraheError)`: If the network mode forbids the request, every attempt fails, or the body is not UTF-8
    fn fetch_text(&self, url: &str, resource: &str) -> Result<Option<String>, BraheError> {
        self.fetch_bytes(url, resource)?
            .map(|body| {
                String::from_utf8(body).map_err(|e| {
                    BraheError::IoError(format!("{resource} from {url} is not UTF-8 text: {e}"))
                })
            })
            .transpose()
    }

    /// GETs `url`, retrying transient failures with backoff up to
    /// `max_retries` times.
    ///
    /// # Arguments
    /// * `url` - The URL to fetch
    /// * `resource` - Description of what is fetched, used in messages
    ///
    /// # Returns
    /// * `Ok(Some(Vec<u8>))`: The response body
    /// * `Ok(None)`: If the server answered 404
    /// * `Err(BraheError)`: If the network mode forbids a request, the Earthdata login fails, or every attempt fails
    fn fetch_bytes(&self, url: &str, resource: &str) -> Result<Option<Vec<u8>>, BraheError> {
        let mut attempt = 0;
        loop {
            ensure_online(url, &format!("{resource} ({url})"))?;
            if attempt > 0 {
                sleep(backoff_delay(attempt));
            }
            match self.request(url, resource) {
                Ok(mut response) => {
                    match response
                        .body_mut()
                        .with_config()
                        .limit(MAX_RESPONSE_BYTES)
                        .read_to_vec()
                    {
                        Ok(body) => return Ok(Some(body)),
                        Err(_) if attempt < self.max_retries => {}
                        Err(e) => {
                            return Err(BraheError::IoError(format!(
                                "Failed to read {resource} from {url}: {e}"
                            )));
                        }
                    }
                }
                Err(RequestError::NotFound) => return Ok(None),
                Err(RequestError::Failed(e)) => return Err(e),
                Err(RequestError::Http(e))
                    if attempt < self.max_retries && is_retryable_error(&e) => {}
                Err(RequestError::Http(e)) => {
                    return Err(BraheError::IoError(format!(
                        "{resource} request {url} failed: {e}"
                    )));
                }
            }
            attempt += 1;
        }
    }

    /// Sends one GET for `url`.
    ///
    /// Up to [`MAX_REDIRECTS`] redirects are followed one at a time, checking
    /// `BRAHE_NETWORK_MODE` before each request, with the agent's cookie jar
    /// carrying cookies between them. For CDDIS, HTTP Basic credentials are
    /// added only to requests for the Earthdata login service, only a final
    /// `200` from outside the login service is accepted, and a 404 during
    /// the sign-in (from the login service until the redirect back to `url`)
    /// is a sign-in failure rather than a missing file.
    ///
    /// # Arguments
    /// * `url` - The URL to fetch
    /// * `resource` - Description of what is fetched, used in messages
    ///
    /// # Returns
    /// * `Ok(Response)`: The final `200` response, with its body unread
    /// * `Err(RequestError)`: `NotFound` for a 404, `Http` for a transport or status failure that may be retried, `Failed` for a failure that must not be
    fn request(&self, url: &str, resource: &str) -> Result<Response<ureq::Body>, RequestError> {
        let failed = |detail: String| RequestError::Failed(BraheError::Error(detail));
        let mut current = url.to_string();
        let mut signing_in = false;
        for _ in 0..=MAX_REDIRECTS {
            ensure_online(&current, &format!("{resource} ({current})"))
                .map_err(RequestError::Failed)?;
            let uri: Uri = current
                .parse()
                .map_err(|e| failed(format!("{resource}: invalid URL {current}: {e}")))?;
            let login = self.source == IGSProductSource::CDDIS && self.is_login_service(&uri);
            if login {
                signing_in = true;
            } else if current == url {
                signing_in = false;
            }
            let mut request = self.agent.get(&current);
            if login {
                let (username, password) = self.credentials.as_ref().ok_or_else(|| {
                    failed(format!(
                        "{resource} requires NASA Earthdata credentials: set {CDDIS_USERNAME_VAR} and {CDDIS_PASSWORD_VAR} or use IGSProductsClient::with_cddis_credentials"
                    ))
                })?;
                let token = BASE64.encode(format!("{username}:{password}"));
                request = request.header("Authorization", format!("Basic {token}"));
            }
            let response = match request.call() {
                Ok(response) => response,
                Err(ureq::Error::StatusCode(status @ (401 | 403))) if login => {
                    return Err(failed(format!(
                        "{resource}: NASA Earthdata login at {} rejected the credentials (HTTP {status}); check {CDDIS_USERNAME_VAR} and {CDDIS_PASSWORD_VAR}",
                        self.urs_url
                    )));
                }
                Err(ureq::Error::StatusCode(404)) if signing_in => {
                    return Err(failed(format!(
                        "{resource}: NASA Earthdata sign-in step {current} answered 404{}",
                        self.login_hint()
                    )));
                }
                Err(e) => return Err(RequestError::from(e)),
            };
            let status = response.status();
            if status.as_u16() == 200 {
                if login {
                    return Err(failed(format!(
                        "{resource}: NASA Earthdata login at {current} answered with a page instead of redirecting back to CDDIS{}",
                        self.login_hint()
                    )));
                }
                return Ok(response);
            }
            if !status.is_redirection() {
                return Err(failed(format!(
                    "{resource} request {current} answered unexpected HTTP status {status}"
                )));
            }
            let location = response
                .headers()
                .get("location")
                .and_then(|value| value.to_str().ok())
                .ok_or_else(|| {
                    failed(format!(
                        "{resource} request {current} answered {status} without a Location header"
                    ))
                })?;
            current = resolve_location(&uri, location).ok_or_else(|| {
                failed(format!(
                    "{resource} request {current} redirected to an invalid location {location}"
                ))
            })?;
        }
        Err(failed(format!(
            "{resource} request {url} redirected more than {MAX_REDIRECTS} times"
        )))
    }

    /// Whether `uri` addresses the NASA Earthdata login service: the same
    /// scheme, host, and port as the configured login URL, under its path.
    ///
    /// # Arguments
    /// * `uri` - Request target
    ///
    /// # Returns
    /// * `bool`: `true` when credentials may be sent to `uri`
    fn is_login_service(&self, uri: &Uri) -> bool {
        let Ok(login) = self.urs_url.parse::<Uri>() else {
            return false;
        };
        let (Some(host), Some(login_host)) = (uri.host(), login.host()) else {
            return false;
        };
        let (Some(scheme), Some(login_scheme)) = (uri.scheme_str(), login.scheme_str()) else {
            return false;
        };
        scheme.eq_ignore_ascii_case(login_scheme)
            && host.eq_ignore_ascii_case(login_host)
            && effective_port(uri) == effective_port(&login)
            && uri.path().starts_with(login.path())
    }
}

/// Failure of one request attempt.
enum RequestError {
    /// The server answered 404.
    NotFound,
    /// A transport or HTTP status failure, retried when transient.
    Http(ureq::Error),
    /// A failure that retrying cannot fix.
    Failed(BraheError),
}

impl From<ureq::Error> for RequestError {
    fn from(e: ureq::Error) -> Self {
        match e {
            ureq::Error::StatusCode(404) => RequestError::NotFound,
            e => RequestError::Http(e),
        }
    }
}

/// Port a URI connects to: its explicit port, or the scheme's default.
///
/// # Arguments
/// * `uri` - The URI
///
/// # Returns
/// * `Option<u16>`: The port, or `None` for a scheme other than `http` or `https` without an explicit port
fn effective_port(uri: &Uri) -> Option<u16> {
    let scheme = uri.scheme_str()?.to_ascii_lowercase();
    uri.port_u16().or(match scheme.as_str() {
        "https" => Some(443),
        "http" => Some(80),
        _ => None,
    })
}

/// Resolves a redirect `Location` against the URL that answered it.
///
/// # Arguments
/// * `base` - URL of the redirecting request
/// * `location` - `Location` header value: absolute with a scheme in any case, scheme-relative (`//host/path`), absolute-path (`/path`), query-only (`?query`, applied to the base path), or relative to the base directory
///
/// # Returns
/// * `Option<String>`: The absolute URL, or `None` if `base` has no scheme or host, or the location's scheme is not `http` or `https`
fn resolve_location(base: &Uri, location: &str) -> Option<String> {
    if let Some(scheme) = location_scheme(location) {
        return ["http", "https"]
            .iter()
            .any(|supported| scheme.eq_ignore_ascii_case(supported))
            .then(|| location.to_string());
    }
    let scheme = base.scheme_str()?;
    if location.starts_with("//") {
        return Some(format!("{scheme}:{location}"));
    }
    let authority = base.authority()?;
    if location.starts_with('/') {
        return Some(format!("{scheme}://{authority}{location}"));
    }
    let path = base.path();
    if location.starts_with('?') {
        return Some(format!("{scheme}://{authority}{path}{location}"));
    }
    let directory = &path[..path.rfind('/').map_or(0, |i| i + 1)];
    let directory = if directory.is_empty() { "/" } else { directory };
    Some(format!("{scheme}://{authority}{directory}{location}"))
}

/// Scheme of an absolute URL reference: the text before the first `:` when
/// it starts with a letter, holds only letters, digits, `+`, `-`, and `.`,
/// and precedes any `/`, `?`, or `#`.
///
/// # Arguments
/// * `reference` - URL reference
///
/// # Returns
/// * `Option<&str>`: The scheme as written, or `None` for a relative reference
fn location_scheme(reference: &str) -> Option<&str> {
    let end = reference.find([':', '/', '?', '#'])?;
    let scheme = &reference[..end];
    (reference[end..].starts_with(':')
        && scheme.starts_with(|c: char| c.is_ascii_alphabetic())
        && scheme
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.')))
    .then_some(scheme)
}

/// Whether a body is an HTML page: its first non-whitespace byte is `<`.
///
/// # Arguments
/// * `body` - Response body
///
/// # Returns
/// * `bool`: `true` for a body that starts with a markup tag
fn is_html(body: &[u8]) -> bool {
    body.iter()
        .find(|b| !b.is_ascii_whitespace())
        .is_some_and(|&b| b == b'<')
}

/// Name a product is saved under in a directory: the published name, without
/// `.gz` for a gzip product.
///
/// # Arguments
/// * `product` - Product saved
///
/// # Returns
/// * `&str`: The file name
fn saved_name(product: &IGSProduct) -> &str {
    match product.file_name.compression() {
        IGSCompression::Gzip => product.name.strip_suffix(".gz").unwrap_or(&product.name),
        IGSCompression::UnixCompress | IGSCompression::None => &product.name,
    }
}

/// Writes a cached product to `target`, decompressing gzip products.
///
/// # Arguments
/// * `product` - Product written
/// * `cached` - Cached file of the product
/// * `target` - File to write
/// * `keep_cached` - Leave the cached file in place; otherwise it is removed
///
/// # Returns
/// * `Ok(())`: The file was written
/// * `Err(BraheError)`: If the cached file cannot be read or removed, its gzip data is invalid, or `target` cannot be written
fn write_saved(
    product: &IGSProduct,
    cached: &Path,
    target: &Path,
    keep_cached: bool,
) -> Result<(), BraheError> {
    if product.file_name.compression() != IGSCompression::Gzip {
        return if keep_cached {
            copy_file(cached, target)
        } else {
            move_file(cached, target)
        };
    }
    let bytes = fs::read(cached)
        .map_err(|e| BraheError::IoError(format!("Failed to read {}: {e}", cached.display())))?;
    let plain = decompress(&bytes)
        .map_err(|e| BraheError::Error(format!("IGS product {}: {e}", product.name)))?;
    atomic_write(target, plain)
        .map_err(|e| BraheError::IoError(format!("Failed to write {}: {e}", target.display())))?;
    if !keep_cached {
        fs::remove_file(cached).map_err(|e| {
            BraheError::IoError(format!("Failed to remove {}: {e}", cached.display()))
        })?;
    }
    Ok(())
}

/// Day count since the GPS epoch of the GPS day containing `epoch`.
///
/// # Arguments
/// * `epoch` - Instant to locate
///
/// # Returns
/// * `u32`: `7 * week + day of week`
fn gps_day(epoch: Epoch) -> u32 {
    let (week, seconds) = epoch.gps_date();
    week * 7 + ((seconds / SECONDS_PER_DAY).floor() as u32).min(6)
}

/// Start of a GPS day.
///
/// # Arguments
/// * `day` - Day count since the GPS epoch
///
/// # Returns
/// * `Epoch`: 00:00 GPS time of that day
fn gps_day_start(day: u32) -> Epoch {
    Epoch::from_gps_date(day / 7, f64::from(day % 7) * SECONDS_PER_DAY)
}

/// The daily SP3 orbit file for `day` among one week's orbit products.
///
/// Products in the naming convention of the day's week are considered: long
/// names from week 2238, short names before it. A candidate spans one day,
/// starts at the day's 00:00, and, for long names when `sampling` is set,
/// has that sampling interval. The first candidate in [`daily_orbit_preference`]
/// order is returned.
///
/// # Arguments
/// * `products` - SP3 orbit products of the day's week listing
/// * `day` - Day count since the GPS epoch
/// * `sampling` - Required sampling interval of long-name files, in seconds
///
/// # Returns
/// * `Option<&IGSProduct>`: The chosen file, or `None` when there is none
fn daily_orbit(products: &[IGSProduct], day: u32, sampling: Option<f64>) -> Option<&IGSProduct> {
    let style = if day / 7 >= IGS_LONG_NAME_FIRST_WEEK {
        IGSNameStyle::Long
    } else {
        IGSNameStyle::Short
    };
    let day_start = gps_day_start(day);
    products
        .iter()
        .filter(|product| {
            let name = &product.file_name;
            name.name_style() == style
                && name.span() == Some(SECONDS_PER_DAY)
                && (name.start_epoch() - day_start).abs() < 1.0
                && (style == IGSNameStyle::Short
                    || sampling.is_none_or(|sampling| {
                        name.sampling()
                            .is_some_and(|published| (published - sampling).abs() < 1e-6)
                    }))
        })
        .min_by(|a, b| daily_orbit_preference(a, b))
}

/// Preference order of daily orbit files for the same day: files that are
/// not Unix-compressed first, since those cannot be parsed, then coarsest
/// sampling, then the `OPS` campaign, then the lowest version, then `.sp3`
/// short names before other SP3 extensions, then name.
///
/// # Arguments
/// * `a` - First product
/// * `b` - Second product
///
/// # Returns
/// * `Ordering`: `Less` when `a` is preferred
fn daily_orbit_preference(a: &IGSProduct, b: &IGSProduct) -> CmpOrdering {
    let (x, y) = (&a.file_name, &b.file_name);
    let unix = |name: &IGSProductFileName| name.compression() == IGSCompression::UnixCompress;
    let other_campaign = |name: &IGSProductFileName| name.campaign().is_some_and(|c| c != "OPS");
    let other_extension = |product: &IGSProduct| {
        product.file_name.name_style() == IGSNameStyle::Short && !product.name.contains(".sp3")
    };
    unix(x)
        .cmp(&unix(y))
        .then_with(|| {
            y.sampling()
                .unwrap_or(0.0)
                .total_cmp(&x.sampling().unwrap_or(0.0))
        })
        .then_with(|| other_campaign(x).cmp(&other_campaign(y)))
        .then_with(|| x.version().cmp(&y.version()))
        .then_with(|| other_extension(a).cmp(&other_extension(b)))
        .then_with(|| a.name.cmp(&b.name))
}

/// Name of the daily SP3 orbit file expected for `day`, used in errors.
///
/// From week 2238 this is the operational long name,
/// `CCC0OPSTTT_YYYYDDD0000_01D_SMP_ORB.SP3.gz`, with `*` for an unknown
/// sampling. Before it, the short name `cccwwwwd.sp3`, followed by
/// `[.gz|.Z]`.
///
/// # Arguments
/// * `center` - Analysis center code, in any case
/// * `solution` - Solution type
/// * `day` - Day count since the GPS epoch
/// * `sampling` - Sampling interval in seconds, if known
///
/// # Returns
/// * `String`: The expected name
fn expected_daily_orbit_name(
    center: &str,
    solution: &IGSSolutionType,
    day: u32,
    sampling: Option<f64>,
) -> String {
    let week = day / 7;
    if week < IGS_LONG_NAME_FIRST_WEEK {
        return format!(
            "{}{week:04}{}.sp3[.gz|.Z]",
            short_code(center, solution),
            day % 7
        );
    }
    // Read the calendar date at noon so that rounding cannot move it a day.
    let noon = gps_day_start(day) + SECONDS_PER_DAY / 2.0;
    let year = noon.to_datetime_as_time_system(TimeSystem::GPS).0;
    let day_of_year = noon.day_of_year_as_time_system(TimeSystem::GPS).floor() as u32;
    let sampling = sampling.map_or_else(|| "*".to_string(), duration_code);
    format!(
        "{}0OPS{}_{year:04}{day_of_year:03}0000_01D_{sampling}_ORB.SP3.gz",
        center.to_ascii_uppercase(),
        solution.code()
    )
}

/// Long-name duration code of an interval, such as `15M` for 900 seconds.
///
/// # Arguments
/// * `seconds` - Interval in seconds
///
/// # Returns
/// * `String`: Two-digit count and unit (`D`, `H`, `M`, or `S`), using the largest unit that divides the interval
fn duration_code(seconds: f64) -> String {
    let seconds = seconds.round() as u64;
    let (count, unit) = [(86400, 'D'), (3600, 'H'), (60, 'M')]
        .into_iter()
        .find(|(size, _)| seconds > 0 && seconds.is_multiple_of(*size))
        .map_or((seconds, 'S'), |(size, unit)| (seconds / size, unit));
    format!("{count:02}{unit}")
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
/// Sort links (`?C=N;O=D`), fragments, scheme links such as `mailto:`, and
/// anything containing a path separator, such as the parent-directory link or
/// a subdirectory `name/`, are not.
///
/// # Arguments
/// * `name` - Listed name
///
/// # Returns
/// * `bool`: `true` for a plain file name
fn is_listed_file(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with(['?', '#'])
        && !name.contains(':')
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
    use crate::datafiles::compression::gzip;
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

    #[test]
    #[parallel]
    fn test_igs_listed_file_rejects_scheme_links() {
        assert!(!is_listed_file("mailto:igs@example.org"));
        assert!(!is_listed_file("javascript:void(0)"));
        assert!(is_listed_file("igr23400.sp3.Z"));
        let html = "<a href=\"mailto:igs@example.org\">mail</a><a href=\"readme.txt\">readme</a>";
        assert_eq!(
            html_listing_files(html),
            vec![("readme.txt".to_string(), None)]
        );
    }

    fn offline_garner_client() -> IGSProductsClient {
        IGSProductsClient::with_base_url(
            IGSProductSource::Garner,
            "https://brahe-network-mode-test.invalid/pub/products/",
        )
        .max_retries(0)
    }

    #[test]
    #[serial]
    fn test_igs_products_client_list_products_between_offline_repeats_online_query() {
        let _cache = CacheRedirect::new();
        let online = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        mock_listing(&server, "/pub/products/2340/", asset("garner_2340.html"));
        let missing = server.mock(|when, then| {
            when.method(GET).path("/pub/products/2341/");
            then.status(404);
        });
        let filter = IGSProductFilter::new().with_center("COD");
        let (start, end) = (gps(2024, 11, 16, 6), gps(2024, 11, 16, 12));
        let expected = garner_client(&server)
            .list_products_between(start, end, &filter)
            .unwrap();
        assert!(names(&expected).contains(&"COD0OPSFIN_20243210000_01D_05M_ORB.SP3.gz"));
        missing.assert_calls(1);
        drop(online);

        let _mode = NetworkModeGuard::set(Some("offline"));
        let offline = offline_garner_client()
            .list_products_between(start, end, &filter)
            .unwrap();
        assert_eq!(names(&offline), names(&expected));
    }

    #[test]
    #[serial]
    fn test_igs_products_client_list_products_between_offline_skips_uncached_padding_week() {
        let _cache = CacheRedirect::new();
        let online = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        mock_listing(&server, "/pub/products/2340/", asset("garner_2340.html"));
        garner_client(&server)
            .list_products(week_2340(), &IGSProductFilter::new())
            .unwrap();
        drop(online);

        let filter = IGSProductFilter::new().with_center("COD");
        let (start, end) = (gps(2024, 11, 11, 0), gps(2024, 11, 12, 0));
        let client = offline_garner_client();
        let mode = NetworkModeGuard::set(Some("offline"));
        let products = client.list_products_between(start, end, &filter).unwrap();
        assert!(names(&products).contains(&"COD0OPSFIN_20243160000_01D_05M_ORB.SP3.gz"));
        drop(mode);

        // A stale cached padding week is skipped under offline-strict, while
        // the fresh in-range week is served.
        let padding = client.cache_dir().unwrap().join("listings/garner/2339.txt");
        fs::write(
            &padding,
            apache_listing(&["COD0OPSFIN_20243140000_01D_05M_ORB.SP3.gz"]),
        )
        .unwrap();
        age_file(&padding, 7200);
        let _mode = NetworkModeGuard::set(Some("offline-strict"));
        let strict = client.list_products_between(start, end, &filter).unwrap();
        assert_eq!(names(&strict), names(&products));
    }

    #[test]
    #[serial]
    fn test_igs_products_client_list_products_between_offline_uncached_range_week_errors() {
        let _cache = CacheRedirect::new();
        let online = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        mock_listing(&server, "/pub/products/2340/", asset("garner_2340.html"));
        garner_client(&server).list_files(week_2340()).unwrap();
        drop(online);

        let _mode = NetworkModeGuard::set(Some("offline"));
        let err = offline_garner_client()
            .list_products_between(
                gps(2024, 11, 13, 0),
                gps(2024, 11, 20, 0),
                &IGSProductFilter::new(),
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains("BRAHE_NETWORK_MODE is offline"), "{err}");
        assert!(err.contains("GPS week 2341"), "{err}");
    }

    const SP3_NAME: &str = "COD0OPSFIN_20243150000_01D_05M_ORB.SP3.gz";
    const SP3_TEXT: &str = "#dP2024 11 10  0  0  0.00000000     289 ORBIT IGS20 FIT  COD\n";
    const SHORT_Z_NAME: &str = "igr23400.sp3.Z";
    const SHORT_Z_BYTES: [u8; 6] = [0x1f, 0x9d, 0x90, 0x23, 0x64, 0x50];

    /// A product of `client` named `name`, listed in archive week `week`.
    fn product(client: &IGSProductsClient, name: &str, week: u32) -> IGSProduct {
        IGSProduct {
            file_name: IGSProductFileName::parse(name).unwrap(),
            name: name.to_string(),
            url: format!("{}{week:04}/{name}", client.base_url()),
            size: None,
            gps_week: week,
        }
    }

    fn mock_file<'a>(server: &'a MockServer, path: &str, body: Vec<u8>) -> httpmock::Mock<'a> {
        server.mock(|when, then| {
            when.method(GET).path(path.to_string());
            then.status(200).body(body);
        })
    }

    /// A CDDIS client for `server` with credentials `u:p` whose login
    /// service is the server's `/oauth/` path.
    fn cddis_client(server: &MockServer) -> IGSProductsClient {
        IGSProductsClient::build(
            IGSProductSource::CDDIS,
            &server.url("/archive/"),
            Some(("u".to_string(), "p".to_string())),
        )
        .with_urs_url(&server.url("/oauth/"))
    }

    /// Mocks the Earthdata sign-in for `path` on `server`: without the
    /// session cookie `path` redirects to the login service, which requires
    /// Basic `u:p` and redirects to `/proxyauth` setting the cookie, which
    /// redirects back to `path`. With the cookie, `path` answers `body`.
    /// Returns the file, login, and proxyauth mocks plus a mock counting
    /// file requests that carry an `Authorization` header.
    fn mock_cddis_flow<'a>(
        server: &'a MockServer,
        path: &str,
        body: Vec<u8>,
    ) -> [httpmock::Mock<'a>; 4] {
        let leaked = server.mock(|when, then| {
            when.method(GET)
                .path(path.to_string())
                .header_exists("authorization");
            then.status(500);
        });
        let file = server.mock(|when, then| {
            when.method(GET)
                .path(path.to_string())
                .header_missing("authorization")
                .cookie("s", "1");
            then.status(200).body(body);
        });
        server.mock(|when, then| {
            when.method(GET)
                .path(path.to_string())
                .header_missing("authorization")
                .cookie_missing("s");
            then.status(302).header(
                "Location",
                "/oauth/authorize?client_id=c&response_type=code",
            );
        });
        let login = server.mock(|when, then| {
            when.method(GET)
                .path("/oauth/authorize")
                .query_param("client_id", "c")
                .header("authorization", "Basic dTpw");
            then.status(302)
                .header("Location", "/proxyauth?code=abc")
                .header("Set-Cookie", "s=1; Path=/");
        });
        let proxy = server.mock(|when, then| {
            when.method(GET)
                .path("/proxyauth")
                .header_missing("authorization");
            then.status(302).header("Location", server.url(path));
        });
        [file, login, proxy, leaked]
    }

    #[test]
    #[serial]
    fn test_igs_products_client_download_caches_file() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let body = gzip(SP3_TEXT.as_bytes()).unwrap();
        let file = mock_file(
            &server,
            &format!("/pub/products/2340/{SP3_NAME}"),
            body.clone(),
        );
        let client = garner_client(&server);
        let product = product(&client, SP3_NAME, 2340);

        let path = client.download(&product).unwrap();
        assert_eq!(
            path,
            client
                .cache_dir()
                .unwrap()
                .join("garner/2340")
                .join(SP3_NAME)
        );
        assert_eq!(fs::read(&path).unwrap(), body);
        assert_eq!(client.download(&product).unwrap(), path);
        file.assert_calls(1);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_download_offline_serves_cache_and_rejects_uncached() {
        let _cache = CacheRedirect::new();
        let online = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        mock_file(
            &server,
            &format!("/pub/products/2340/{SP3_NAME}"),
            gzip(SP3_TEXT.as_bytes()).unwrap(),
        );
        let cached = product(&garner_client(&server), SP3_NAME, 2340);
        let path = garner_client(&server).download(&cached).unwrap();
        drop(online);

        let _mode = NetworkModeGuard::set(Some("offline-strict"));
        let client = offline_garner_client();
        assert_eq!(
            client.download(&product(&client, SP3_NAME, 2340)).unwrap(),
            path
        );
        let err = client
            .download(&product(
                &client,
                "COD0OPSFIN_20243160000_01D_05M_ORB.SP3.gz",
                2340,
            ))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("BRAHE_NETWORK_MODE is offline-strict"),
            "{err}"
        );
        assert!(
            err.contains("COD0OPSFIN_20243160000_01D_05M_ORB.SP3.gz"),
            "{err}"
        );
    }

    #[test]
    #[serial]
    fn test_igs_products_client_download_missing_product_names_file_and_source() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let missing = server.mock(|when, then| {
            when.method(GET)
                .path(format!("/pub/products/2340/{SP3_NAME}"));
            then.status(404);
        });
        let client = garner_client(&server);
        let err = client
            .download(&product(&client, SP3_NAME, 2340))
            .unwrap_err()
            .to_string();
        assert!(err.contains("Garner"), "{err}");
        assert!(err.contains(SP3_NAME), "{err}");
        assert!(err.contains("404"), "{err}");
        missing.assert_calls(1);
        assert!(
            !client
                .cache_dir()
                .unwrap()
                .join("garner/2340")
                .join(SP3_NAME)
                .exists()
        );
    }

    #[test]
    #[serial]
    fn test_igs_products_client_download_rejects_body_not_matching_compression() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        mock_file(
            &server,
            &format!("/pub/products/2340/{SP3_NAME}"),
            b"<html>error</html>".to_vec(),
        );
        let client = garner_client(&server);
        let err = client
            .download(&product(&client, SP3_NAME, 2340))
            .unwrap_err()
            .to_string();
        assert!(err.contains("is not gzip data"), "{err}");
        assert!(
            !client
                .cache_dir()
                .unwrap()
                .join("garner/2340")
                .join(SP3_NAME)
                .exists()
        );
    }

    #[test]
    #[serial]
    fn test_igs_products_client_download_rejects_unsafe_names() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let any = server.mock(|when, then| {
            when.any_request();
            then.status(200).body(gzip(SP3_TEXT.as_bytes()).unwrap());
        });
        let client = garner_client(&server);
        for name in [
            format!("../{SP3_NAME}"),
            format!("..\\{SP3_NAME}"),
            format!("C:{SP3_NAME}"),
            "..".to_string(),
            String::new(),
        ] {
            let mut unsafe_product = product(&client, SP3_NAME, 2340);
            unsafe_product.name = name.clone();
            let err = client.download(&unsafe_product).unwrap_err().to_string();
            assert!(err.contains("invalid IGS product name"), "{name}: {err}");
        }
        any.assert_calls(0);
    }

    #[test]
    #[parallel]
    fn test_igs_listings_drop_traversal_names() {
        let html = apache_listing(&[
            "../COD0OPSFIN_20243150000_01D_05M_ORB.SP3.gz",
            "..%2F..%2Fetc",
            "C:COD0OPSFIN_20243150000_01D_05M_ORB.SP3.gz",
            SP3_NAME,
        ]);
        assert_eq!(
            html_listing_files(&html),
            vec![
                ("..%2F..%2Fetc".to_string(), None),
                (SP3_NAME.to_string(), None)
            ]
        );
        let text = format!(
            "../../{SP3_NAME} 10\n..\\{SP3_NAME} 10\nC:{SP3_NAME} 10\n.. 1\n{SP3_NAME} 12\n"
        );
        assert_eq!(
            plain_listing_files(&text),
            vec![(SP3_NAME.to_string(), Some(12))]
        );
        let client = IGSProductsClient::with_source(IGSProductSource::Garner);
        let products = client.products_from_listing(2340, &html, &IGSProductFilter::new());
        assert_eq!(names(&products), vec![SP3_NAME]);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_save_gzip_decompresses() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let out = tempfile::tempdir().unwrap();
        let server = MockServer::start();
        let file = mock_file(
            &server,
            &format!("/pub/products/2340/{SP3_NAME}"),
            gzip(SP3_TEXT.as_bytes()).unwrap(),
        );
        let client = garner_client(&server);
        let product = product(&client, SP3_NAME, 2340);
        let cached = client
            .cache_dir()
            .unwrap()
            .join("garner/2340")
            .join(SP3_NAME);

        let saved = client.save(&product, out.path(), true).unwrap();
        assert_eq!(
            saved,
            out.path().join("COD0OPSFIN_20243150000_01D_05M_ORB.SP3")
        );
        assert_eq!(fs::read_to_string(&saved).unwrap(), SP3_TEXT);
        assert!(cached.is_file());

        let custom = out.path().join("nested").join("custom.sp3");
        let saved = client.save(&product, &custom, false).unwrap();
        assert_eq!(saved, custom);
        assert_eq!(fs::read_to_string(&custom).unwrap(), SP3_TEXT);
        assert!(!cached.exists());
        file.assert_calls(1);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_save_unix_compress_unchanged() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let out = tempfile::tempdir().unwrap();
        let server = MockServer::start();
        mock_file(
            &server,
            &format!("/pub/products/2340/{SHORT_Z_NAME}"),
            SHORT_Z_BYTES.to_vec(),
        );
        let client = garner_client(&server);
        let product = product(&client, SHORT_Z_NAME, 2340);

        let saved = client.save(&product, out.path(), false).unwrap();
        assert_eq!(saved, out.path().join(SHORT_Z_NAME));
        assert_eq!(fs::read(&saved).unwrap(), SHORT_Z_BYTES);
        assert!(
            !client
                .cache_dir()
                .unwrap()
                .join("garner/2340")
                .join(SHORT_Z_NAME)
                .exists()
        );
    }

    #[test]
    #[serial]
    fn test_igs_products_client_save_into_cache_errors_before_download() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let any = server.mock(|when, then| {
            when.any_request();
            then.status(200).body(gzip(SP3_TEXT.as_bytes()).unwrap());
        });
        let client = garner_client(&server);
        let product = product(&client, SP3_NAME, 2340);
        let cache = client.cache_dir().unwrap();
        for destination in [cache.clone(), cache.join("saved").join("out.sp3")] {
            let err = client
                .save(&product, &destination, false)
                .unwrap_err()
                .to_string();
            assert!(err.contains("inside the IGS cache directory"), "{err}");
        }
        any.assert_calls(0);
    }

    /// Mocks a Garner week 2340 listing of three COD daily orbits, listed
    /// out of time order, and their gzip files. Returns the names in time
    /// order.
    fn mock_week_of_orbits(server: &MockServer) -> [&'static str; 3] {
        let ordered = [
            "COD0OPSFIN_20243150000_01D_05M_ORB.SP3.gz",
            "COD0OPSFIN_20243160000_01D_05M_ORB.SP3.gz",
            "COD0OPSFIN_20243170000_01D_05M_ORB.SP3.gz",
        ];
        mock_listing(
            server,
            "/pub/products/2340/",
            apache_listing(&[
                ordered[2],
                ordered[0],
                "COD0OPSFIN_20243150000_01D_30S_CLK.CLK.gz",
                ordered[1],
            ]),
        );
        for name in ordered {
            mock_file(
                server,
                &format!("/pub/products/2340/{name}"),
                gzip(format!("{name}\n").as_bytes()).unwrap(),
            );
        }
        ordered
    }

    #[test]
    #[serial]
    fn test_igs_products_client_download_between_orders_by_start() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        mock_listing(&server, "/pub/products/2339/", apache_listing(&[]));
        let ordered = mock_week_of_orbits(&server);
        let client = garner_client(&server);
        let filter = IGSProductFilter::new().with_content(IGSContentType::Orbit);

        let paths = client
            .download_between(gps(2024, 11, 10, 0), gps(2024, 11, 12, 12), &filter)
            .unwrap();
        let week_dir = client.cache_dir().unwrap().join("garner/2340");
        let expected: Vec<PathBuf> = ordered.iter().map(|name| week_dir.join(name)).collect();
        assert_eq!(paths, expected);
        assert!(paths.iter().all(|path| path.is_file()));
    }

    #[test]
    #[serial]
    fn test_igs_products_client_save_between_writes_in_time_order() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let out = tempfile::tempdir().unwrap();
        let server = MockServer::start();
        mock_listing(&server, "/pub/products/2339/", apache_listing(&[]));
        let ordered = mock_week_of_orbits(&server);
        let client = garner_client(&server);
        let filter = IGSProductFilter::new().with_content(IGSContentType::Orbit);
        let directory = out.path().join("orbits");

        let paths = client
            .save_between(
                gps(2024, 11, 10, 0),
                gps(2024, 11, 12, 12),
                &filter,
                &directory,
                false,
            )
            .unwrap();
        let expected: Vec<PathBuf> = ordered
            .iter()
            .map(|name| directory.join(name.strip_suffix(".gz").unwrap()))
            .collect();
        assert_eq!(paths, expected);
        for (path, name) in paths.iter().zip(ordered) {
            assert_eq!(fs::read_to_string(path).unwrap(), format!("{name}\n"));
        }
        let week_dir = client.cache_dir().unwrap().join("garner/2340");
        assert_eq!(fs::read_dir(&week_dir).unwrap().count(), 0);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_save_between_rejects_bad_directory_before_download() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let out = tempfile::tempdir().unwrap();
        let server = MockServer::start();
        let any = server.mock(|when, then| {
            when.any_request();
            then.status(200).body(apache_listing(&[]));
        });
        let client = garner_client(&server);
        let file = out.path().join("file.txt");
        fs::write(&file, "x").unwrap();
        let (start, end) = (gps(2024, 11, 12, 0), gps(2024, 11, 13, 0));
        let filter = IGSProductFilter::new();

        let inside = client.cache_dir().unwrap().join("saved");
        let err = client
            .save_between(start, end, &filter, &inside, false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("inside the IGS cache directory"), "{err}");
        let err = client
            .save_between(start, end, &filter, &file, false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("is a file"), "{err}");
        any.assert_calls(0);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_cddis_download_signs_in_through_earthdata() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let body = gzip(SP3_TEXT.as_bytes()).unwrap();
        let [file, login, proxy, leaked] =
            mock_cddis_flow(&server, &format!("/archive/2340/{SP3_NAME}"), body.clone());
        let client = cddis_client(&server);

        let path = client.download(&product(&client, SP3_NAME, 2340)).unwrap();
        assert_eq!(
            path,
            client
                .cache_dir()
                .unwrap()
                .join("cddis/2340")
                .join(SP3_NAME)
        );
        assert_eq!(fs::read(&path).unwrap(), body);
        file.assert_calls(1);
        login.assert_calls(1);
        proxy.assert_calls(1);
        leaked.assert_calls(0);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_cddis_listing_signs_in_through_earthdata() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let [file, login, _proxy, leaked] = mock_cddis_flow(
            &server,
            "/archive/2340/*",
            asset("cddis_2340.txt").into_bytes(),
        );
        let client = cddis_client(&server);

        let files = client.list_files(week_2340()).unwrap();
        assert!(files.contains(&SP3_NAME.to_string()));
        file.assert_calls(1);
        login.assert_calls(1);
        leaked.assert_calls(0);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_cddis_html_listing_is_rejected() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/archive/2340/*");
            then.status(200)
                .body("\n  <!DOCTYPE html><html><body>Earthdata Login</body></html>");
        });
        let client = cddis_client(&server);

        let err = client.list_files(week_2340()).unwrap_err().to_string();
        assert!(err.contains("HTML page"), "{err}");
        assert!(err.contains("CDDIS_EARTHDATA_USERNAME"), "{err}");
        assert!(
            !client
                .cache_dir()
                .unwrap()
                .join("listings/cddis/2340.txt")
                .exists()
        );
    }

    #[test]
    #[serial]
    fn test_igs_products_client_cddis_login_failures() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let path = format!("/archive/2340/{SP3_NAME}");
        server.mock(|when, then| {
            when.method(GET).path(path.clone());
            then.status(302)
                .header("Location", "/oauth/authorize?client_id=c");
        });
        let rejected = server.mock(|when, then| {
            when.method(GET)
                .path("/oauth/authorize")
                .header_exists("authorization");
            then.status(401);
        });
        let client = cddis_client(&server);
        let err = client
            .download(&product(&client, SP3_NAME, 2340))
            .unwrap_err()
            .to_string();
        assert!(err.contains("rejected the credentials"), "{err}");
        assert!(err.contains(SP3_NAME), "{err}");
        rejected.assert_calls(1);

        let anonymous =
            IGSProductsClient::build(IGSProductSource::CDDIS, &server.url("/archive/"), None)
                .with_urs_url(&server.url("/oauth/"));
        let err = anonymous
            .download(&product(&anonymous, SP3_NAME, 2340))
            .unwrap_err()
            .to_string();
        assert!(err.contains("requires NASA Earthdata credentials"), "{err}");
        rejected.assert_calls(1);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_cddis_login_page_and_redirect_loop_are_errors() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path(format!("/archive/2340/{SP3_NAME}"));
            then.status(302).header("Location", "/oauth/authorize");
        });
        server.mock(|when, then| {
            when.method(GET).path("/oauth/authorize");
            then.status(200).body("<html>Earthdata Login</html>");
        });
        let loop_mock = server.mock(|when, then| {
            when.method(GET).path("/archive/2340/igr23400.sp3.Z");
            then.status(302).header("Location", "igr23400.sp3.Z");
        });
        let client = cddis_client(&server);
        let err = client
            .download(&product(&client, SP3_NAME, 2340))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("instead of redirecting back to CDDIS"),
            "{err}"
        );

        let err = client
            .download(&product(&client, "igr23400.sp3.Z", 2340))
            .unwrap_err()
            .to_string();
        assert!(err.contains("redirected more than 10 times"), "{err}");
        loop_mock.assert_calls(MAX_REDIRECTS + 1);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_cddis_unexpected_responses_are_errors() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/archive/2340/igr23400.sp3.Z");
            then.status(204);
        });
        server.mock(|when, then| {
            when.method(GET).path("/archive/2340/igr23401.sp3.Z");
            then.status(302);
        });
        let client = cddis_client(&server);

        let err = client
            .download(&product(&client, "igr23400.sp3.Z", 2340))
            .unwrap_err()
            .to_string();
        assert!(err.contains("unexpected HTTP status 204"), "{err}");

        let err = client
            .download(&product(&client, "igr23401.sp3.Z", 2340))
            .unwrap_err()
            .to_string();
        assert!(err.contains("without a Location header"), "{err}");
    }

    #[test]
    #[serial]
    fn test_igs_products_client_cddis_html_product_is_rejected() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        mock_file(
            &server,
            "/archive/2340/igr23400.sp3",
            b"<html>Earthdata Login</html>".to_vec(),
        );
        let client = cddis_client(&server);

        let err = client
            .download(&product(&client, "igr23400.sp3", 2340))
            .unwrap_err()
            .to_string();
        assert!(err.contains("is an HTML page"), "{err}");
        assert!(err.contains("CDDIS_EARTHDATA_USERNAME"), "{err}");
        assert!(
            !client
                .cache_dir()
                .unwrap()
                .join("cddis/2340/igr23400.sp3")
                .exists()
        );
    }

    #[test]
    #[serial]
    fn test_igs_products_client_listing_rejects_non_utf8_body() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        mock_file(&server, "/pub/products/2340/", vec![0xff, 0xfe, 0x00]);
        let client = garner_client(&server);

        let err = client.list_files(week_2340()).unwrap_err().to_string();
        assert!(err.contains("is not UTF-8 text"), "{err}");
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
    fn test_igs_products_client_save_unix_compress_keep_cached_copies() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let out = tempfile::tempdir().unwrap();
        let server = MockServer::start();
        mock_file(
            &server,
            &format!("/pub/products/2340/{SHORT_Z_NAME}"),
            SHORT_Z_BYTES.to_vec(),
        );
        let client = garner_client(&server);
        let product = product(&client, SHORT_Z_NAME, 2340);

        let saved = client.save(&product, out.path(), true).unwrap();
        assert_eq!(fs::read(&saved).unwrap(), SHORT_Z_BYTES);
        assert_eq!(
            fs::read(
                client
                    .cache_dir()
                    .unwrap()
                    .join("garner/2340")
                    .join(SHORT_Z_NAME)
            )
            .unwrap(),
            SHORT_Z_BYTES
        );
    }

    #[test]
    #[parallel]
    fn test_igs_resolve_location() {
        let base: Uri = "https://cddis.nasa.gov/archive/gnss/products/2340/file.gz?x=1"
            .parse()
            .unwrap();
        for (location, expected) in [
            (
                "https://urs.earthdata.nasa.gov/oauth/authorize?a=b",
                "https://urs.earthdata.nasa.gov/oauth/authorize?a=b",
            ),
            (
                "//urs.earthdata.nasa.gov/oauth",
                "https://urs.earthdata.nasa.gov/oauth",
            ),
            (
                "/proxyauth?code=1",
                "https://cddis.nasa.gov/proxyauth?code=1",
            ),
            (
                "other.gz",
                "https://cddis.nasa.gov/archive/gnss/products/2340/other.gz",
            ),
        ] {
            assert_eq!(resolve_location(&base, location).as_deref(), Some(expected));
        }
        for (location, expected) in [
            (
                "HTTPS://URS.earthdata.nasa.gov/oauth",
                Some("HTTPS://URS.earthdata.nasa.gov/oauth"),
            ),
            (
                "Http://mirror.example/a.gz",
                Some("Http://mirror.example/a.gz"),
            ),
            (
                "?list",
                Some("https://cddis.nasa.gov/archive/gnss/products/2340/file.gz?list"),
            ),
            ("ftp://mirror.example/a.gz", None),
            ("mailto:user@example.com", None),
            ("dir:name/x.gz", None),
        ] {
            assert_eq!(
                resolve_location(&base, location).as_deref(),
                expected,
                "{location}"
            );
        }
        let relative: Uri = "/no/host".parse().unwrap();
        assert_eq!(resolve_location(&relative, "/x"), None);
    }

    #[test]
    #[parallel]
    fn test_igs_products_client_is_login_service() {
        let client = IGSProductsClient::with_cddis_credentials("u", "p");
        for (url, expected) in [
            ("https://urs.earthdata.nasa.gov/oauth/authorize", true),
            ("https://URS.earthdata.nasa.gov:443/oauth/authorize", true),
            ("HTTPS://urs.earthdata.nasa.gov/oauth/authorize", true),
            ("http://urs.earthdata.nasa.gov/oauth/authorize", false),
            ("https://urs.earthdata.nasa.gov:8443/oauth/authorize", false),
            ("https://urs.earthdata.nasa.gov.example.com/oauth", false),
            ("https://cddis.nasa.gov/archive/gnss/products/", false),
            ("https://cddis.nasa.gov/proxyauth", false),
        ] {
            let uri: Uri = url.parse().unwrap();
            assert_eq!(client.is_login_service(&uri), expected, "{url}");
        }
        let scoped = client.with_urs_url("http://127.0.0.1:8080/oauth/");
        assert!(scoped.is_login_service(&"http://127.0.0.1:8080/oauth/authorize".parse().unwrap()));
        assert!(!scoped.is_login_service(&"http://127.0.0.1:8080/file".parse().unwrap()));
    }

    #[test]
    #[parallel]
    fn test_igs_is_html() {
        assert!(is_html(b"<html>"));
        assert!(is_html(b" \n\t<!DOCTYPE html>"));
        assert!(!is_html(b"#dP2024"));
        assert!(!is_html(b""));
    }

    const SP3_FIXTURE: &str = "test_assets/sp3/IGS0OPSFIN_20243150000_01D_15M_ORB_subset.SP3.gz";

    fn sp3_fixture() -> Vec<u8> {
        fs::read(SP3_FIXTURE).unwrap()
    }

    /// Name of the COD final daily orbit for 2024 day of year `doy` at
    /// sampling code `sampling`.
    fn cod_daily(doy: u32, sampling: &str) -> String {
        format!("COD0OPSFIN_2024{doy:03}0000_01D_{sampling}_ORB.SP3.gz")
    }

    /// Mocks a Garner week 2340 listing holding COD final daily orbits at
    /// 05M and 15M for each 2024 day of year in `days`, plus products of other
    /// solutions, contents, spans, and centers, and serves the SP3 fixture
    /// for each listed file. Returns the file mocks keyed by name.
    fn mock_cod_week_2340<'a>(
        server: &'a MockServer,
        days: &[u32],
    ) -> BTreeMap<String, httpmock::Mock<'a>> {
        let mut listed: Vec<String> = days
            .iter()
            .flat_map(|&doy| [cod_daily(doy, "05M"), cod_daily(doy, "15M")])
            .collect();
        listed.extend(
            [
                "COD0OPSRAP_20243160000_01D_15M_ORB.SP3.gz",
                "COD0OPSULT_20243160000_02D_05M_ORB.SP3.gz",
                "COD0OPSFIN_20243160000_01D_30S_CLK.CLK.gz",
                "COD0OPSFIN_20243150000_07D_01D_ERP.ERP.gz",
                "ESA0OPSFIN_20243160000_01D_15M_ORB.SP3.gz",
            ]
            .map(String::from),
        );
        let refs: Vec<&str> = listed.iter().map(String::as_str).collect();
        mock_listing(server, "/pub/products/2340/", apache_listing(&refs));
        listed
            .iter()
            .map(|name| {
                let mock = mock_file(server, &format!("/pub/products/2340/{name}"), sp3_fixture());
                (name.clone(), mock)
            })
            .collect()
    }

    /// Asserts that each file mock in `files` was requested once when its
    /// name is in `expected` and never otherwise.
    fn assert_downloaded(files: &BTreeMap<String, httpmock::Mock<'_>>, expected: &[String]) {
        for (name, mock) in files {
            let calls = usize::from(expected.contains(name));
            assert_eq!(mock.calls(), calls, "{name}");
        }
    }

    #[test]
    #[serial]
    fn test_igs_products_client_download_sp3_parses_file() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let name = "IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.gz";
        let file = mock_file(
            &server,
            &format!("/pub/products/2340/{name}"),
            sp3_fixture(),
        );
        let client = garner_client(&server);

        let sp3 = client.download_sp3(&product(&client, name, 2340)).unwrap();
        assert_eq!(sp3, SP3::from_file(SP3_FIXTURE).unwrap());
        file.assert_calls(1);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_download_sp3_rejects_before_download() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let any = server.mock(|when, then| {
            when.any_request();
            then.status(200).body(sp3_fixture());
        });
        let client = garner_client(&server);

        let err = client
            .download_sp3(&product(&client, "cod21000.sp3.Z", 2100))
            .unwrap_err()
            .to_string();
        assert!(err.contains(UNIX_COMPRESS_UNSUPPORTED), "{err}");
        assert!(err.contains("cod21000.sp3.Z"), "{err}");

        let err = client
            .download_sp3(&product(
                &client,
                "COD0OPSFIN_20243150000_01D_30S_CLK.CLK.gz",
                2340,
            ))
            .unwrap_err()
            .to_string();
        assert!(err.contains("not an SP3"), "{err}");
        any.assert_calls(0);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_download_sp3_range_pads_with_coarsest_sampling() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let files = mock_cod_week_2340(&server, &[315, 316, 317, 318]);
        let client = garner_client(&server);
        let (start, end) = (gps(2024, 11, 11, 0), gps(2024, 11, 12, 0));
        let options = IGSRangeOptions::new();

        let sp3s = client
            .download_sp3_range(start, end, "cod", IGSSolutionType::Final, &options)
            .unwrap();
        let expected: Vec<String> = (315..=318).map(|doy| cod_daily(doy, "15M")).collect();
        assert_eq!(sp3s.len(), 4);
        assert!(sp3s.iter().all(|sp3| *sp3 == sp3s[0]));
        assert_downloaded(&files, &expected);
        let chosen = client
            .sp3_range_products(start, end, "COD", &IGSSolutionType::Final, &options)
            .unwrap();
        assert_eq!(names(&chosen), expected);

        let fine = options.with_sampling(300.0);
        let sp3s = client
            .download_sp3_range(start, end, "COD", IGSSolutionType::Final, &fine)
            .unwrap();
        assert_eq!(sp3s.len(), 4);
        let chosen = client
            .sp3_range_products(start, end, "COD", &IGSSolutionType::Final, &fine)
            .unwrap();
        let expected_fine: Vec<String> = (315..=318).map(|doy| cod_daily(doy, "05M")).collect();
        assert_eq!(names(&chosen), expected_fine);
        let both: Vec<String> = expected.into_iter().chain(expected_fine).collect();
        assert_downloaded(&files, &both);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_download_sp3_range_without_padding() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let files = mock_cod_week_2340(&server, &[315, 316, 317, 318]);
        let client = garner_client(&server);
        let (start, end) = (gps(2024, 11, 11, 0), gps(2024, 11, 12, 0));
        let options = IGSRangeOptions::new().with_pad(false);

        let sp3s = client
            .download_sp3_range(start, end, "COD", IGSSolutionType::Final, &options)
            .unwrap();
        assert_eq!(sp3s.len(), 2);
        let expected = vec![cod_daily(316, "15M"), cod_daily(317, "15M")];
        assert_downloaded(&files, &expected);
        let chosen = client
            .sp3_range_products(start, end, "COD", &IGSSolutionType::Final, &options)
            .unwrap();
        assert_eq!(names(&chosen), expected);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_download_sp3_range_skips_missing_padding() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let files = mock_cod_week_2340(&server, &[316, 317, 318]);
        let before = server.mock(|when, then| {
            when.method(GET).path("/pub/products/2339/");
            then.status(404);
        });
        let client = garner_client(&server);
        let options = IGSRangeOptions::new();

        // Day 315, the padding day before Monday 2024-11-11, is missing.
        let sp3s = client
            .download_sp3_range(
                gps(2024, 11, 11, 0),
                gps(2024, 11, 12, 0),
                "COD",
                IGSSolutionType::Final,
                &options,
            )
            .unwrap();
        assert_eq!(sp3s.len(), 3);
        let expected: Vec<String> = (316..=318).map(|doy| cod_daily(doy, "15M")).collect();
        assert_downloaded(&files, &expected);
        before.assert_calls(0);

        // Inside the span, the missing day 315 is an error. The padding day
        // before it is in week 2339, whose 404 is skipped.
        let chosen = client
            .sp3_range_products(
                gps(2024, 11, 11, 6),
                gps(2024, 11, 11, 12),
                "COD",
                &IGSSolutionType::Final,
                &options,
            )
            .unwrap();
        assert_eq!(
            names(&chosen),
            [cod_daily(316, "15M"), cod_daily(317, "15M")]
        );
        let err = client
            .sp3_range_products(
                gps(2024, 11, 10, 6),
                gps(2024, 11, 11, 12),
                "COD",
                &IGSSolutionType::Final,
                &options,
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains(&cod_daily(315, "15M")), "{err}");
        before.assert_calls(1);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_download_sp3_range_skips_missing_padding_week() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        mock_cod_week_2340(&server, &[315, 316]);
        let before = server.mock(|when, then| {
            when.method(GET).path("/pub/products/2339/");
            then.status(404);
        });
        let client = garner_client(&server);

        let chosen = client
            .sp3_range_products(
                gps(2024, 11, 10, 6),
                gps(2024, 11, 10, 12),
                "COD",
                &IGSSolutionType::Final,
                &IGSRangeOptions::new(),
            )
            .unwrap();
        assert_eq!(
            names(&chosen),
            [cod_daily(315, "15M"), cod_daily(316, "15M")]
        );
        before.assert_calls(1);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_download_sp3_range_missing_day_names_file_and_source() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let files = mock_cod_week_2340(&server, &[315, 316, 318]);
        let client = garner_client(&server);
        let (start, end) = (gps(2024, 11, 11, 0), gps(2024, 11, 12, 0));

        let err = client
            .download_sp3_range(
                start,
                end,
                "COD",
                IGSSolutionType::Final,
                &IGSRangeOptions::new(),
            )
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("COD0OPSFIN_20243170000_01D_15M_ORB.SP3.gz"),
            "{err}"
        );
        assert!(err.contains("Garner"), "{err}");

        let err = client
            .download_sp3_range(
                start,
                end,
                "COD",
                IGSSolutionType::Final,
                &IGSRangeOptions::new().with_sampling(30.0),
            )
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("COD0OPSFIN_20243160000_01D_30S_ORB.SP3.gz"),
            "{err}"
        );
        assert_downloaded(&files, &[]);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_download_sp3_range_rejects_ultra_rapid_before_request() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let any = server.mock(|when, then| {
            when.any_request();
            then.status(200).body(apache_listing(&[]));
        });
        let client = garner_client(&server);

        let err = client
            .download_sp3_range(
                gps(2024, 11, 11, 0),
                gps(2024, 11, 12, 0),
                "IGS",
                IGSSolutionType::UltraRapid,
                &IGSRangeOptions::new(),
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains("ultra-rapid"), "{err}");

        let err = client
            .download_sp3_range(
                gps(2024, 11, 12, 0),
                gps(2024, 11, 11, 0),
                "IGS",
                IGSSolutionType::Final,
                &IGSRangeOptions::new(),
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains("before its start"), "{err}");
        any.assert_calls(0);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_download_sp3_range_unix_compress() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        mock_listing(
            &server,
            "/pub/products/2100/",
            apache_listing(&[
                "cod21000.sp3.Z",
                "cod21001.sp3.gz",
                "cod21002.sp3.Z",
                "cod21003.sp3.Z",
                "cod21003.sp3.gz",
                "cod21003.clk.Z",
            ]),
        );
        let files = server.mock(|when, then| {
            when.method(GET).path_prefix("/pub/products/2100/cod");
            then.status(200).body(sp3_fixture());
        });
        let client = garner_client(&server);
        let options = IGSRangeOptions::new();

        // Sunday 2020-04-05 is day 0 of week 2100.
        let err = client
            .download_sp3_range(
                gps(2020, 4, 5, 6),
                gps(2020, 4, 6, 6),
                "cod",
                IGSSolutionType::Final,
                &options,
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains(UNIX_COMPRESS_UNSUPPORTED), "{err}");
        assert!(err.contains("cod21000.sp3.Z"), "{err}");
        files.assert_calls(0);

        // The `.Z` padding days are skipped, and gzip is preferred to `.Z`.
        let chosen = client
            .sp3_range_products(
                gps(2020, 4, 6, 6),
                gps(2020, 4, 6, 12),
                "COD",
                &IGSSolutionType::Final,
                &options,
            )
            .unwrap();
        assert_eq!(names(&chosen), ["cod21001.sp3.gz"]);
        let chosen = client
            .sp3_range_products(
                gps(2020, 4, 8, 6),
                gps(2020, 4, 8, 12),
                "COD",
                &IGSSolutionType::Final,
                &options.clone().with_pad(false),
            )
            .unwrap();
        assert_eq!(names(&chosen), ["cod21003.sp3.gz"]);
        files.assert_calls(0);

        // A missing short-name day names the expected short file.
        let err = client
            .sp3_range_products(
                gps(2020, 4, 9, 6),
                gps(2020, 4, 9, 12),
                "COD",
                &IGSSolutionType::Rapid,
                &options,
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains("cor21004.sp3"), "{err}");
    }

    #[test]
    #[serial]
    fn test_igs_products_client_download_sp3_range_crosses_long_name_week() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let plain = decompress(&sp3_fixture()).unwrap().into_owned();
        let short = ["cod22375.sp3", "cod22376.sp3"];
        let long = [
            "COD0OPSFIN_20223310000_01D_05M_ORB.SP3.gz",
            "COD0OPSFIN_20223320000_01D_05M_ORB.SP3.gz",
        ];
        // Each week also lists a file in the other convention, which is not
        // used.
        mock_listing(
            &server,
            "/pub/products/2237/",
            apache_listing(&[
                short[0],
                short[1],
                "COD0OPSFIN_20223300000_01D_05M_ORB.SP3.gz",
            ]),
        );
        mock_listing(
            &server,
            "/pub/products/2238/",
            apache_listing(&[long[0], long[1], "cod22380.sp3.Z"]),
        );
        let short_files: Vec<_> = short
            .iter()
            .map(|name| {
                mock_file(
                    &server,
                    &format!("/pub/products/2237/{name}"),
                    plain.clone(),
                )
            })
            .collect();
        let long_files: Vec<_> = long
            .iter()
            .map(|name| {
                mock_file(
                    &server,
                    &format!("/pub/products/2238/{name}"),
                    sp3_fixture(),
                )
            })
            .collect();
        let client = garner_client(&server);
        // Saturday 2022-11-26 is the last day of week 2237.
        let (start, end) = (gps(2022, 11, 26, 12), gps(2022, 11, 27, 12));
        let options = IGSRangeOptions::new();

        let sp3s = client
            .download_sp3_range(start, end, "COD", IGSSolutionType::Final, &options)
            .unwrap();
        assert_eq!(sp3s.len(), 4);
        let fixture = SP3::from_file(SP3_FIXTURE).unwrap();
        assert!(sp3s.iter().all(|sp3| *sp3 == fixture));
        for mock in short_files.iter().chain(&long_files) {
            mock.assert_calls(1);
        }
        let chosen = client
            .sp3_range_products(start, end, "COD", &IGSSolutionType::Final, &options)
            .unwrap();
        assert_eq!(names(&chosen), [short[0], short[1], long[0], long[1]]);
    }

    #[test]
    #[parallel]
    fn test_igs_gps_day_and_expected_daily_orbit_name() {
        assert_eq!(gps_day(gps(2024, 11, 10, 0)), 2340 * 7);
        assert_eq!(gps_day(gps(2024, 11, 16, 23)), 2340 * 7 + 6);
        assert_eq!(gps_day_start(2340 * 7 + 2), gps(2024, 11, 12, 0));
        assert_eq!(
            expected_daily_orbit_name("cod", &IGSSolutionType::Final, 2340 * 7 + 2, Some(900.0)),
            "COD0OPSFIN_20243170000_01D_15M_ORB.SP3.gz"
        );
        assert_eq!(
            expected_daily_orbit_name("IGS", &IGSSolutionType::Rapid, 2238 * 7, None),
            "IGS0OPSRAP_20223310000_01D_*_ORB.SP3.gz"
        );
        assert_eq!(
            expected_daily_orbit_name("IGS", &IGSSolutionType::Rapid, 2237 * 7 + 6, None),
            "igr22376.sp3[.gz|.Z]"
        );
        assert_eq!(duration_code(900.0), "15M");
        assert_eq!(duration_code(30.0), "30S");
        assert_eq!(duration_code(7200.0), "02H");
        assert_eq!(duration_code(86400.0), "01D");
    }

    #[test]
    #[parallel]
    fn test_igs_daily_orbit_preference() {
        let listed = |name: &str| IGSProduct {
            file_name: IGSProductFileName::parse(name).unwrap(),
            name: name.to_string(),
            url: String::new(),
            size: None,
            gps_week: 2340,
        };
        let products: Vec<IGSProduct> = [
            "GFZ0MGXRAP_20243150000_01D_05M_ORB.SP3.gz",
            "GFZ1OPSRAP_20243150000_01D_05M_ORB.SP3.gz",
            "GFZ0OPSRAP_20243150000_01D_05M_ORB.SP3.gz",
            "GFZ0OPSRAP_20243150600_01D_15M_ORB.SP3.gz",
            "GFZ0MGXRAP_20243160000_01D_15M_ORB.SP3.gz",
        ]
        .into_iter()
        .map(listed)
        .collect();
        let day = 2340 * 7;
        assert_eq!(
            daily_orbit(&products, day, None).unwrap().name,
            "GFZ0OPSRAP_20243150000_01D_05M_ORB.SP3.gz"
        );
        assert!(daily_orbit(&products, day, Some(900.0)).is_none());
        assert_eq!(
            daily_orbit(&products, day + 1, None).unwrap().name,
            "GFZ0MGXRAP_20243160000_01D_15M_ORB.SP3.gz"
        );

        let short: Vec<IGSProduct> = ["cod21000.eph.Z", "cod21000.sp3.Z", "cod21000.eph"]
            .into_iter()
            .map(listed)
            .collect();
        assert_eq!(
            daily_orbit(&short, 2100 * 7, None).unwrap().name,
            "cod21000.eph"
        );
        assert_eq!(
            daily_orbit(&short[..2], 2100 * 7, None).unwrap().name,
            "cod21000.sp3.Z"
        );
    }

    const INTEGRATION_FINAL_ORBIT: &str = "IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.gz";

    fn integration_week_2340() -> Epoch {
        Epoch::from_datetime(2024, 11, 12, 12, 0, 0.0, 0.0, TimeSystem::GPS)
    }

    fn assert_integration_final_orbit(client: &IGSProductsClient) {
        let products = client
            .list_products(
                integration_week_2340(),
                &IGSProductFilter::new()
                    .with_center("IGS")
                    .with_solution(IGSSolutionType::Final)
                    .with_content(IGSContentType::Orbit)
                    .with_format(IGSFileFormat::SP3),
            )
            .unwrap();
        let product = products
            .iter()
            .find(|p| p.name == INTEGRATION_FINAL_ORBIT)
            .unwrap_or_else(|| panic!("{INTEGRATION_FINAL_ORBIT} not listed"));
        let sp3 = client.download_sp3(product).unwrap();
        assert_eq!(sp3.num_epochs(), 96);
        assert!(!sp3.satellite_ids().is_empty());
    }

    #[test]
    #[serial]
    #[cfg_attr(not(feature = "integration"), ignore)]
    fn test_integration_garner_listing_and_download() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(None);
        let client = IGSProductsClient::with_source(IGSProductSource::Garner);
        let files = client.list_files(integration_week_2340()).unwrap();
        assert!(files.iter().any(|f| f == INTEGRATION_FINAL_ORBIT));
        assert_integration_final_orbit(&client);
    }

    #[test]
    #[serial]
    #[cfg_attr(not(feature = "integration"), ignore)]
    fn test_integration_garner_raw_unix_compress_download() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(None);
        let client = IGSProductsClient::with_source(IGSProductSource::Garner);
        let epoch = Epoch::from_datetime(2020, 4, 8, 12, 0, 0.0, 0.0, TimeSystem::GPS);
        let products = client
            .list_products(epoch, &IGSProductFilter::new().with_center("IGS"))
            .unwrap();
        let product = products
            .iter()
            .find(|p| p.name == "igs21000.sp3.Z")
            .unwrap_or_else(|| panic!("igs21000.sp3.Z not listed"));
        let path = client.download(product).unwrap();
        let bytes = fs::read(path).unwrap();
        assert_eq!(&bytes[..2], &[0x1f, 0x9d]);
        assert!(client.download_sp3(product).is_err());
    }

    #[test]
    #[serial]
    #[cfg_attr(not(feature = "integration"), ignore)]
    fn test_integration_bkg_listing_and_download() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(None);
        let client = IGSProductsClient::with_source(IGSProductSource::BKG);
        let files = client.list_files(integration_week_2340()).unwrap();
        assert!(files.iter().any(|f| f == INTEGRATION_FINAL_ORBIT));
        assert_integration_final_orbit(&client);
    }

    #[test]
    #[serial]
    #[cfg_attr(not(feature = "integration"), ignore)]
    fn test_integration_cddis_listing_and_download() {
        let username = env::var(CDDIS_USERNAME_VAR).unwrap_or_default();
        let password = env::var(CDDIS_PASSWORD_VAR).unwrap_or_default();
        if username.is_empty() || password.is_empty() {
            eprintln!("skipping: CDDIS credentials not set");
            return;
        }
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(None);
        let client = IGSProductsClient::with_cddis_credentials(&username, &password);
        let files = client.list_files(integration_week_2340()).unwrap();
        assert!(files.iter().any(|f| f == INTEGRATION_FINAL_ORBIT));
        assert_integration_final_orbit(&client);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_download_rejects_truncated_gzip_without_caching() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let body = gzip(SP3_TEXT.as_bytes()).unwrap();
        let path = format!("/pub/products/2340/{SP3_NAME}");
        let mut truncated = mock_file(&server, &path, body[..body.len() - 8].to_vec());
        let client = garner_client(&server);
        let product = product(&client, SP3_NAME, 2340);
        let cached = client
            .cache_dir()
            .unwrap()
            .join("garner/2340")
            .join(SP3_NAME);

        let err = client.download(&product).unwrap_err().to_string();
        assert!(err.contains("is not valid gzip data"), "{err}");
        assert!(err.contains(SP3_NAME), "{err}");
        assert!(!cached.exists());
        truncated.assert_calls(1);
        truncated.delete();

        let complete = mock_file(&server, &path, body.clone());
        assert_eq!(client.download(&product).unwrap(), cached);
        assert_eq!(fs::read(&cached).unwrap(), body);
        complete.assert_calls(1);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_redirects_are_checked_against_network_mode() {
        let _cache = CacheRedirect::new();
        let online = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let body = gzip(SP3_TEXT.as_bytes()).unwrap();
        let moved = server.mock(|when, then| {
            when.method(GET)
                .path(format!("/pub/products/2340/{SP3_NAME}"));
            then.status(302)
                .header("Location", format!("/mirror/{SP3_NAME}"));
        });
        let mirror = mock_file(&server, &format!("/mirror/{SP3_NAME}"), body.clone());
        let client = garner_client(&server);
        let path = client.download(&product(&client, SP3_NAME, 2340)).unwrap();
        assert_eq!(fs::read(path).unwrap(), body);
        moved.assert_calls(1);
        mirror.assert_calls(1);
        drop(online);

        let name = "COD0OPSFIN_20243160000_01D_05M_ORB.SP3.gz";
        let external = server.mock(|when, then| {
            when.method(GET).path(format!("/pub/products/2340/{name}"));
            then.status(302).header(
                "Location",
                format!("http://brahe-redirect-test.invalid/{name}"),
            );
        });
        for mode in ["offline", "offline-strict"] {
            let _mode = NetworkModeGuard::set(Some(mode));
            let err = client
                .download(&product(&client, name, 2340))
                .unwrap_err()
                .to_string();
            assert!(
                err.contains(&format!("BRAHE_NETWORK_MODE is {mode}")),
                "{err}"
            );
            assert!(err.contains("brahe-redirect-test.invalid"), "{err}");
        }
        external.assert_calls(2);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_cddis_sign_in_404_is_not_missing_product() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        for (missing, location) in [
            ("/oauth/authorize", "/oauth/authorize?client_id=c"),
            ("/proxyauth", "/proxyauth?code=abc"),
        ] {
            let server = MockServer::start();
            server.mock(|when, then| {
                when.method(GET).path(format!("/archive/2340/{SP3_NAME}"));
                then.status(302)
                    .header("Location", "/oauth/authorize?client_id=c");
            });
            if missing == "/proxyauth" {
                server.mock(|when, then| {
                    when.method(GET).path("/oauth/authorize");
                    then.status(302).header("Location", location);
                });
            }
            server.mock(|when, then| {
                when.method(GET).path(missing);
                then.status(404);
            });
            let client = cddis_client(&server);
            let err = client
                .download(&product(&client, SP3_NAME, 2340))
                .unwrap_err()
                .to_string();
            assert!(err.contains("sign-in step"), "{missing}: {err}");
            assert!(err.contains(missing), "{missing}: {err}");
            assert!(!err.contains("not found"), "{missing}: {err}");
        }
    }

    #[test]
    #[serial]
    fn test_igs_products_client_cddis_missing_file_after_sign_in_is_not_found() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let path = format!("/archive/2340/{SP3_NAME}");
        server.mock(|when, then| {
            when.method(GET).path(path.clone()).cookie_missing("s");
            then.status(302)
                .header("Location", "/oauth/authorize?client_id=c");
        });
        server.mock(|when, then| {
            when.method(GET).path("/oauth/authorize");
            then.status(302)
                .header("Location", "/proxyauth?code=abc")
                .header("Set-Cookie", "s=1; Path=/");
        });
        server.mock(|when, then| {
            when.method(GET).path("/proxyauth");
            then.status(302).header("Location", server.url(&path));
        });
        server.mock(|when, then| {
            when.method(GET).path(path.clone()).cookie("s", "1");
            then.status(404);
        });
        let client = cddis_client(&server);
        let err = client
            .download(&product(&client, SP3_NAME, 2340))
            .unwrap_err()
            .to_string();
        assert!(err.contains("not found"), "{err}");
        assert!(err.contains("CDDIS"), "{err}");
        assert!(err.contains(SP3_NAME), "{err}");
    }

    #[test]
    #[serial]
    fn test_igs_products_client_download_sp3_range_prefers_parseable_finer_file() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let coarse_z = "COD0OPSFIN_20243160000_01D_15M_ORB.SP3.Z";
        let fine_gz = cod_daily(316, "05M");
        let only_z = "COD0OPSFIN_20243170000_01D_15M_ORB.SP3.Z";
        mock_listing(
            &server,
            "/pub/products/2340/",
            apache_listing(&[coarse_z, &fine_gz, only_z]),
        );
        let fine = mock_file(
            &server,
            &format!("/pub/products/2340/{fine_gz}"),
            sp3_fixture(),
        );
        let unix = server.mock(|when, then| {
            when.method(GET).path_suffix(".SP3.Z");
            then.status(200).body(SHORT_Z_BYTES);
        });
        let client = garner_client(&server);
        let options = IGSRangeOptions::new().with_pad(false);

        let files = client
            .download_sp3_range(
                gps(2024, 11, 11, 6),
                gps(2024, 11, 11, 12),
                "COD",
                IGSSolutionType::Final,
                &options,
            )
            .unwrap();
        assert_eq!(files, vec![SP3::from_file(SP3_FIXTURE).unwrap()]);
        fine.assert_calls(1);

        let err = client
            .download_sp3_range(
                gps(2024, 11, 12, 6),
                gps(2024, 11, 12, 12),
                "COD",
                IGSSolutionType::Final,
                &options,
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains(UNIX_COMPRESS_UNSUPPORTED), "{err}");
        assert!(err.contains(only_z), "{err}");
        unix.assert_calls(0);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_download_sp3_range_missing_week_names_expected_file() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let week = server.mock(|when, then| {
            when.method(GET).path("/pub/products/2341/");
            then.status(404);
        });
        let client = garner_client(&server);

        // Monday 2024-11-18 is day of year 323, in GPS week 2341.
        let err = client
            .download_sp3_range(
                gps(2024, 11, 18, 6),
                gps(2024, 11, 18, 12),
                "COD",
                IGSSolutionType::Final,
                &IGSRangeOptions::new().with_pad(false),
            )
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("IGS Garner product COD0OPSFIN_20243230000_01D_*_ORB.SP3.gz not found"),
            "{err}"
        );
        assert!(err.contains("2341"), "{err}");
        assert!(err.contains("answered 404"), "{err}");
        week.assert_calls(1);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_download_sp3_parse_errors_name_product() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let name = cod_daily(316, "15M");
        mock_listing(&server, "/pub/products/2340/", apache_listing(&[&name]));
        mock_file(
            &server,
            &format!("/pub/products/2340/{name}"),
            gzip(b"not an SP3 file\n").unwrap(),
        );
        let client = garner_client(&server);

        let err = client
            .download_sp3(&product(&client, &name, 2340))
            .unwrap_err();
        assert!(matches!(err, BraheError::ParseError(_)), "{err}");
        assert!(
            err.to_string()
                .contains(&format!("IGS Garner product {name}")),
            "{err}"
        );

        let err = client
            .download_sp3_range(
                gps(2024, 11, 11, 6),
                gps(2024, 11, 11, 12),
                "COD",
                IGSSolutionType::Final,
                &IGSRangeOptions::new().with_pad(false),
            )
            .unwrap_err();
        assert!(matches!(err, BraheError::ParseError(_)), "{err}");
        assert!(
            err.to_string()
                .contains(&format!("IGS Garner product {name}")),
            "{err}"
        );
    }
}
