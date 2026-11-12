/*!
 * Blocking client for IGS precise-product archives.
 */

use std::cmp::Ordering as CmpOrdering;
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::thread::sleep;

use crate::clients::igs::centers::IGSAnalysisCenter;
use crate::clients::igs::file_name::{
    IGS_LONG_NAME_FIRST_WEEK, IGSCompression, IGSContentType, IGSNameStyle, IGSProductFileName,
    IGSSolutionType,
};
use crate::clients::igs::product::{IGSProduct, IGSProductFilter, IGSRangeOptions};
use crate::clients::igs::source::IGSProductSource;
use crate::constants::SECONDS_PER_DAY;
use crate::datafiles::sp3::SP3;
use crate::time::Epoch;
use crate::utils::cache::is_older_than;
use crate::utils::compression::{
    Compression, UNIX_COMPRESS_UNSUPPORTED, compression_of, decompress,
};
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

/// Largest response body read into memory, in bytes (1 GiB).
const MAX_RESPONSE_BYTES: u64 = 1 << 30;

/// Client for IGS precise-product archives.
///
/// The client reads from one of the sources in [`IGSProductSource`] and never
/// falls back to another.
///
/// Requests are blocking, retried on transient failures, and honour
/// `BRAHE_NETWORK_MODE`. Listings and downloaded files are cached under
/// [`get_igs_cache_dir`](crate::utils::get_igs_cache_dir). The cache
/// directory is not coordinated across processes or across clients sharing
/// it.
///
/// # Examples
///
/// ```
/// use brahe::igs::{IGSProductSource, IGSProductsClient};
///
/// let client = IGSProductsClient::new(IGSProductSource::BKG);
/// assert_eq!(client.source(), IGSProductSource::BKG);
/// assert_eq!(client.base_url(), "https://igs.bkg.bund.de/root_ftp/IGS/products/");
/// ```
#[derive(Debug, Clone)]
pub struct IGSProductsClient {
    source: IGSProductSource,
    base_url: String,
    listing_max_age: f64,
    max_retries: u32,
    agent: ureq::Agent,
}

impl IGSProductsClient {
    /// Creates a client for a source at its default base URL.
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
    /// let client = IGSProductsClient::new(IGSProductSource::Garner);
    /// assert_eq!(client.source(), IGSProductSource::Garner);
    /// assert_eq!(client.base_url(), "http://garner.ucsd.edu/pub/products/");
    /// ```
    pub fn new(source: IGSProductSource) -> Self {
        Self::with_base_url(source, source.default_base_url())
    }

    /// Creates a client for a source served from a different base URL, for
    /// mirrors and tests.
    ///
    /// Listings are cached per source, so a client with a different base URL
    /// shares the listing cache of the source it stands in for.
    ///
    /// A trailing `/` is added to `base_url` when it is missing.
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
        let mut base_url = base_url.to_string();
        if !base_url.ends_with('/') {
            base_url.push('/');
        }
        // A source served over HTTPS is never followed to a plain HTTP redirect.
        let agent = ureq::Agent::config_builder()
            .https_only(base_url.starts_with("https://"))
            .build()
            .into();
        Self {
            source,
            base_url,
            listing_max_age: DEFAULT_IGS_LISTING_MAX_AGE,
            max_retries: DEFAULT_MAX_RETRIES,
            agent,
        }
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
    /// let client = IGSProductsClient::new(IGSProductSource::Garner).with_cache_age(600.0);
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
    /// let client = IGSProductsClient::new(IGSProductSource::Garner).max_retries(0);
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
    /// let client = IGSProductsClient::new(IGSProductSource::BKG);
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
    /// let client = IGSProductsClient::new(IGSProductSource::Garner);
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
    /// let dir = IGSProductsClient::new(IGSProductSource::Garner).cache_dir().unwrap();
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
    /// let weeks = IGSProductsClient::new(IGSProductSource::Garner).list_weeks().unwrap();
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
    /// let client = IGSProductsClient::new(IGSProductSource::Garner);
    /// let epoch = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// let files = client.list_files(epoch).unwrap();
    /// assert!(files.iter().any(|f| f == "COD0OPSFIN_20243150000_01D_05M_ORB.SP3.gz"));
    /// ```
    pub fn list_files(&self, epoch: Epoch) -> Result<Vec<String>, BraheError> {
        let week = epoch.gps_date().0;
        Ok(html_listing_files(&self.listing(Some(week))?))
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
    /// use brahe::igs::{IGSAnalysisCenter, IGSContentType, IGSProductFilter, IGSProductSource, IGSProductsClient, IGSSolutionType};
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let client = IGSProductsClient::new(IGSProductSource::Garner);
    /// let epoch = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// let filter = IGSProductFilter::new()
    ///     .with_center(IGSAnalysisCenter::COD)
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
    /// use brahe::igs::{IGSAnalysisCenter, IGSContentType, IGSProductFilter, IGSProductSource, IGSProductsClient};
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let client = IGSProductsClient::new(IGSProductSource::Garner);
    /// let start = Epoch::from_datetime(2024, 11, 13, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// let end = Epoch::from_datetime(2024, 11, 20, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// let filter = IGSProductFilter::new().with_center(IGSAnalysisCenter::IGS).with_content(IGSContentType::Orbit);
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
    /// is not cached, `BRAHE_NETWORK_MODE` must allow the request.
    ///
    /// # Arguments
    /// * `product` - Product to download, as returned by a listing
    ///
    /// # Returns
    /// * `Ok(PathBuf)`: Path of the cached file, compressed as published
    /// * `Err(BraheError)`: If the product's name is not a plain IGS product file name, the file does not exist on the source, the network mode forbids the request, the body does not match the name's compression, or the cache cannot be written
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::igs::{IGSAnalysisCenter, IGSContentType, IGSProductFilter, IGSProductSource, IGSProductsClient, IGSSolutionType};
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let client = IGSProductsClient::new(IGSProductSource::Garner);
    /// let epoch = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// let filter = IGSProductFilter::new()
    ///     .with_center(IGSAnalysisCenter::IGS)
    ///     .with_solution(IGSSolutionType::Final)
    ///     .with_content(IGSContentType::Orbit);
    /// let product = &client.list_products(epoch, &filter).unwrap()[0];
    /// let path = client.download(product).unwrap();
    /// println!("{}", path.display());
    /// ```
    pub fn download(&self, product: &IGSProduct) -> Result<PathBuf, BraheError> {
        self.fetch_product(product)?
            .ok_or_else(|| self.product_not_found(product))
    }

    /// Downloads a product into the cache, reporting a 404 as absence.
    ///
    /// This is [`IGSProductsClient::download`] with the source's 404 for the
    /// file returned as `Ok(None)` instead of an error.
    ///
    /// # Arguments
    /// * `product` - Product to download
    ///
    /// # Returns
    /// * `Ok(Some(PathBuf))`: Path of the cached file
    /// * `Ok(None)`: If the file is not cached and the source answered 404
    /// * `Err(BraheError)`: As for [`IGSProductsClient::download`], other than a missing file
    fn fetch_product(&self, product: &IGSProduct) -> Result<Option<PathBuf>, BraheError> {
        let path = self.product_cache_path(product)?;
        let resource = self.product_resource(product);
        if path.is_file() {
            match cache_policy(&resource, false)? {
                CacheDecision::Serve => return Ok(Some(path)),
                CacheDecision::Refresh => {}
            }
        }
        let Some(body) = self.fetch_bytes(&product.url, &resource)? else {
            return Ok(None);
        };
        self.check_product_body(product, &body)?;
        atomic_write(&path, &body).map_err(|e| {
            BraheError::IoError(format!(
                "Failed to write cached {resource} {}: {e}",
                path.display()
            ))
        })?;
        Ok(Some(path))
    }

    /// Error for a product file the source answered 404 for.
    ///
    /// # Arguments
    /// * `product` - Product that was not found
    ///
    /// # Returns
    /// * `BraheError`: An error naming the source, the file, and its URL
    fn product_not_found(&self, product: &IGSProduct) -> BraheError {
        BraheError::Error(format!(
            "{} not found: {} answered 404",
            self.product_resource(product),
            product.url
        ))
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
    /// use brahe::igs::{IGSAnalysisCenter, IGSContentType, IGSProductFilter, IGSProductSource, IGSProductsClient};
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let client = IGSProductsClient::new(IGSProductSource::Garner);
    /// let epoch = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// let filter = IGSProductFilter::new().with_center(IGSAnalysisCenter::COD).with_content(IGSContentType::Orbit);
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
    /// use brahe::igs::{IGSAnalysisCenter, IGSContentType, IGSProductFilter, IGSProductSource, IGSProductsClient, IGSSolutionType};
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let client = IGSProductsClient::new(IGSProductSource::Garner);
    /// let epoch = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// let filter = IGSProductFilter::new()
    ///     .with_center(IGSAnalysisCenter::IGS)
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
    /// Every file is chosen, and every listing and selection error below is
    /// raised, before anything is downloaded. Ultra-rapid orbits are
    /// rejected: their issues every six hours overlap one another, so they do
    /// not chain into one time series. A day inside the span with no file, or
    /// whose file is only published Unix-compressed, is an error; such a
    /// padding day is skipped. A padding file that is listed but whose
    /// download answers 404 is also skipped, and its day is left out of the
    /// result; any other failure to download or parse a padding file, and
    /// any failure for a file inside the span, including a 404, is an error.
    /// A range crossing week 2238 returns short-name and long-name files
    /// together.
    ///
    /// # Arguments
    /// * `start` - Start of the time range
    /// * `end` - End of the time range, inclusive
    /// * `center` - Analysis center, for example [`IGSAnalysisCenter::COD`]
    /// * `solution` - Solution type, for example [`IGSSolutionType::Final`]
    /// * `options` - Padding and sampling selection
    ///
    /// # Returns
    /// * `Ok(Vec<SP3>)`: Parsed daily files in time order
    /// * `Err(BraheError)`: If `solution` is ultra-rapid, `end` is before `start`, a week listing containing part of the span cannot be obtained, a day inside the span has no file on the source (including when its week directory answers 404) or only a Unix-compressed one, or a download or parse fails other than a 404 for a padding file; a parse error names the product
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::datafiles::sp3::SP3Satellite;
    /// use brahe::igs::{
    ///     IGSAnalysisCenter, IGSProductSource, IGSProductsClient, IGSRangeOptions, IGSSolutionType,
    /// };
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let client = IGSProductsClient::new(IGSProductSource::Garner);
    /// let start = Epoch::from_datetime(2024, 11, 11, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// let end = Epoch::from_datetime(2024, 11, 12, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// let files = client
    ///     .download_sp3_range(
    ///         start,
    ///         end,
    ///         IGSAnalysisCenter::COD,
    ///         IGSSolutionType::Final,
    ///         &IGSRangeOptions::new(),
    ///     )
    ///     .unwrap();
    /// let satellites = SP3Satellite::from_files(&files).unwrap();
    /// ```
    pub fn download_sp3_range(
        &self,
        start: Epoch,
        end: Epoch,
        center: IGSAnalysisCenter,
        solution: IGSSolutionType,
        options: &IGSRangeOptions,
    ) -> Result<Vec<SP3>, BraheError> {
        let files = self.sp3_range_files(start, end, &center, &solution, options)?;
        let mut downloaded = Vec::with_capacity(files.len());
        for file in &files {
            match self.fetch_product(&file.product)? {
                Some(path) => downloaded.push((&file.product, path)),
                None if file.padding => {}
                None => return Err(self.product_not_found(&file.product)),
            }
        }
        downloaded
            .iter()
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

    /// Chooses the daily SP3 orbit files [`IGSProductsClient::download_sp3_range`]
    /// downloads, without downloading them.
    ///
    /// # Arguments
    /// * `start` - Start of the time range
    /// * `end` - End of the time range, inclusive
    /// * `center` - Analysis center
    /// * `solution` - Solution type
    /// * `options` - Padding and sampling selection
    ///
    /// # Returns
    /// * `Ok(Vec<SP3RangeFile>)`: One file per covered day, in time order, marked as padding or inside the span
    /// * `Err(BraheError)`: As for [`IGSProductsClient::download_sp3_range`], other than download and parse failures
    fn sp3_range_files(
        &self,
        start: Epoch,
        end: Epoch,
        center: &IGSAnalysisCenter,
        solution: &IGSSolutionType,
        options: &IGSRangeOptions,
    ) -> Result<Vec<SP3RangeFile>, BraheError> {
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
            .with_center(center.clone())
            .with_solution(solution.clone())
            .with_content(IGSContentType::Orbit);

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

        let mut selected: Vec<SP3RangeFile> = Vec::new();
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
                Some(product) => selected.push(SP3RangeFile {
                    product: product.clone(),
                    padding: !in_span,
                }),
                None if in_span => missing.push(day),
                None => {}
            }
        }
        if let Some(&day) = missing.first() {
            let week = day / 7;
            let listing = self.listing_url(Some(week));
            let reason = if absent_weeks.contains(&week) {
                format!("{listing} answered 404")
            } else {
                format!("{listing} lists none")
            };
            let sampling = options
                .sampling
                .map(|sampling| format!(" with {sampling} s sampling"))
                .unwrap_or_default();
            return Err(BraheError::Error(format!(
                "IGS {} has no daily {} {} SP3 orbit file{sampling} for {} (GPS week {week:04}, day {}): {reason}",
                self.source,
                center.code(),
                solution.code(),
                gps_day_start(day),
                day % 7
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
        html_listing_files(listing)
            .into_iter()
            .filter_map(|name| {
                let file_name = IGSProductFileName::parse(&name).ok()?;
                Some(IGSProduct {
                    url: format!("{}{week:04}/{name}", self.base_url),
                    file_name,
                    name,
                    gps_week: week,
                })
            })
            .filter(|product| filter.matches(product))
            .collect()
    }

    /// URL of the top-level listing (`week` of `None`) or of a week listing.
    ///
    /// # Arguments
    /// * `week` - GPS week of the directory, or `None` for the top level
    ///
    /// # Returns
    /// * `String`: The listing URL
    fn listing_url(&self, week: Option<u32>) -> String {
        match week {
            None => self.base_url.clone(),
            Some(week) => format!("{}{week:04}/", self.base_url),
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
    /// size limit.
    ///
    /// # Arguments
    /// * `product` - Product the body was downloaded for
    /// * `body` - Response body
    ///
    /// # Returns
    /// * `Ok(())`: The body matches the product
    /// * `Err(BraheError)`: If the body does not match
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
            _ => None,
        };
        match mismatch {
            Some(detail) => Err(BraheError::Error(format!(
                "{} downloaded from {} {detail}",
                self.product_resource(product),
                product.url
            ))),
            None => Ok(()),
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
    /// * `Err(BraheError)`: If the network mode forbids the request or every attempt fails
    fn fetch_bytes(&self, url: &str, resource: &str) -> Result<Option<Vec<u8>>, BraheError> {
        let mut attempt = 0;
        loop {
            ensure_online(url, &format!("{resource} ({url})"))?;
            if attempt > 0 {
                sleep(backoff_delay(attempt));
            }
            match self.agent.get(url).call() {
                Ok(response) if response.status().as_u16() != 200 => {
                    return Err(BraheError::IoError(format!(
                        "{resource} request {url} answered unexpected HTTP status {}",
                        response.status()
                    )));
                }
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

/// Name a product is saved under in a directory: the published name, without
/// `.gz` for a gzip product.
///
/// # Arguments
/// * `product` - Product saved
///
/// # Returns
/// * `&str`: The file name
///
/// # Examples
///
/// ```ignore
/// let name = "IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.gz";
/// let product = IGSProduct {
///     file_name: IGSProductFileName::parse(name).unwrap(),
///     name: name.to_string(),
///     url: format!("https://igs.bkg.bund.de/root_ftp/IGS/products/2340/{name}"),
///     gps_week: 2340,
/// };
/// assert_eq!(saved_name(&product), "IGS0OPSFIN_20243150000_01D_15M_ORB.SP3");
/// ```
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
///
/// # Examples
///
/// ```ignore
/// // `product` is a gzip product whose cached copy is `cached`.
/// write_saved(&product, &cached, Path::new("orbits/IGS0OPSFIN_20243150000_01D_15M_ORB.SP3"), true)?;
/// ```
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

/// A daily SP3 orbit file chosen for a range download.
#[derive(Debug, Clone)]
struct SP3RangeFile {
    /// The chosen product.
    product: IGSProduct,
    /// Whether the file's day is a padding day outside the requested span.
    padding: bool,
}

/// Day count since the GPS epoch of the GPS day containing `epoch`.
///
/// # Arguments
/// * `epoch` - Instant to locate
///
/// # Returns
/// * `u32`: `7 * week + day of week`
///
/// # Examples
///
/// ```ignore
/// let epoch = Epoch::from_datetime(2024, 11, 10, 12, 0, 0.0, 0.0, TimeSystem::GPS);
/// assert_eq!(gps_day(epoch), 2340 * 7);
/// ```
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
///
/// # Examples
///
/// ```ignore
/// let start = gps_day_start(2340 * 7 + 1);
/// assert_eq!(start, Epoch::from_datetime(2024, 11, 11, 0, 0, 0.0, 0.0, TimeSystem::GPS));
/// ```
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
///
/// # Examples
///
/// ```ignore
/// // `products` is the SP3 orbit listing of week 2340.
/// let chosen = daily_orbit(&products, 2340 * 7, Some(900.0)).unwrap();
/// assert_eq!(chosen.name, "IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.gz");
/// ```
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
                && name.is_sp3()
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
///
/// # Examples
///
/// ```ignore
/// // `coarse` has 15M sampling and `fine` 05M sampling, for the same day.
/// assert_eq!(daily_orbit_preference(&coarse, &fine), CmpOrdering::Less);
/// ```
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
///
/// # Examples
///
/// ```ignore
/// // `product` is a one-day file starting 2024-11-10 00:00 GPS.
/// let noon = Epoch::from_datetime(2024, 11, 10, 12, 0, 0.0, 0.0, TimeSystem::GPS);
/// assert!(covers(&product, noon, noon));
/// assert!(!covers(&product, noon + 86400.0, noon + 86400.0));
/// ```
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
///
/// # Examples
///
/// ```ignore
/// assert_eq!(html_hrefs(r#"<a href="2340/">2340/</a> <a href="?C=N;O=D">Name</a>"#), ["2340/", "?C=N;O=D"]);
/// ```
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
///
/// # Examples
///
/// ```ignore
/// assert!(is_listed_file("IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.gz"));
/// assert!(!is_listed_file("?C=N;O=D"));
/// assert!(!is_listed_file("../"));
/// ```
fn is_listed_file(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with(['?', '#'])
        && !name.contains(':')
        && validate_path_component("listed file name", name).is_ok()
}

/// Reads the files of an Apache-style HTML directory listing.
///
/// # Arguments
/// * `html` - Listing page source
///
/// # Returns
/// * `Vec<String>`: File names, each once, in listing order
///
/// # Examples
///
/// ```ignore
/// let html = r#"<a href="../">Parent</a> <a href="igs23400.sp3.Z">igs23400.sp3.Z</a>"#;
/// assert_eq!(html_listing_files(html), ["igs23400.sp3.Z"]);
/// ```
fn html_listing_files(html: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    html_hrefs(html)
        .into_iter()
        .filter(|href| is_listed_file(href) && seen.insert(*href))
        .map(str::to_string)
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
///
/// # Examples
///
/// ```ignore
/// let html = r#"<a href="2341/">2341/</a> <a href="2340/">2340/</a> <a href="mgex/">mgex/</a>"#;
/// assert_eq!(listed_weeks(html), [2340, 2341]);
/// ```
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
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::clients::igs::file_name::{IGSContentType, IGSSolutionType};
    use crate::time::TimeSystem;
    use crate::utils::compression::gzip;
    use crate::utils::testing::{CacheRedirect, NetworkModeGuard};
    use httpmock::prelude::*;
    use serial_test::{parallel, serial};
    use std::time::{Duration, SystemTime};

    #[test]
    #[serial]
    fn test_igs_products_client_new_uses_default_base_url() {
        for source in [IGSProductSource::Garner, IGSProductSource::BKG] {
            let client = IGSProductsClient::new(source);
            assert_eq!(client.source(), source);
            assert_eq!(client.base_url(), source.default_base_url());
        }
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
        let client = IGSProductsClient::new(IGSProductSource::Garner);
        assert_eq!(client.listing_max_age, DEFAULT_IGS_LISTING_MAX_AGE);
        assert_eq!(client.max_retries, 3);

        let client = client.with_cache_age(600.0).max_retries(0);
        assert_eq!(client.listing_max_age, 600.0);
        assert_eq!(client.max_retries, 0);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_https_source_refuses_plain_http() {
        let garner = IGSProductsClient::new(IGSProductSource::Garner);
        assert!(!garner.agent.config().https_only());
        let bkg = IGSProductsClient::new(IGSProductSource::BKG);
        assert!(bkg.agent.config().https_only());
    }

    #[test]
    #[serial]
    fn test_igs_products_client_listing_rejects_non_200_status() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let week = server.mock(|when, then| {
            when.method(GET).path("/pub/products/2340/");
            then.status(204);
        });
        let client = garner_client(&server);

        let err = client.list_files(week_2340()).unwrap_err().to_string();
        assert!(err.contains("unexpected HTTP status 204"), "{err}");
        week.assert_calls(1);
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
    fn test_igs_products_client_cache_dir() {
        let redirect = CacheRedirect::new();
        let dir = IGSProductsClient::new(IGSProductSource::Garner)
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
            .list_products(
                week_2340(),
                &IGSProductFilter::new().with_center(IGSAnalysisCenter::parse("cod")),
            )
            .unwrap();
        let upper = client
            .list_products(
                week_2340(),
                &IGSProductFilter::new().with_center(IGSAnalysisCenter::COD),
            )
            .unwrap();
        assert_eq!(lower, upper);
        assert!(lower.len() >= 20, "{}", lower.len());
        assert!(
            lower
                .iter()
                .all(|p| *p.file_name.center() == IGSAnalysisCenter::COD)
        );
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
                    .with_center(IGSAnalysisCenter::ESA)
                    .with_content(IGSContentType::Clock),
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
            assert_eq!(product.gps_week, 2340);
        }
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
            .list_products_between(
                start,
                end,
                &IGSProductFilter::new().with_center(IGSAnalysisCenter::parse("cod")),
            )
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
            .list_products_between(
                start,
                end,
                &IGSProductFilter::new().with_center(IGSAnalysisCenter::IGS),
            )
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
                &IGSProductFilter::new().with_center(IGSAnalysisCenter::parse("igs")),
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
                &IGSProductFilter::new().with_center(IGSAnalysisCenter::COD),
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
            vec!["igr23400.sp3.Z".to_string(), "readme.txt".to_string()]
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
        assert_eq!(html_listing_files(html), vec!["readme.txt".to_string()]);
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
        let filter = IGSProductFilter::new().with_center(IGSAnalysisCenter::COD);
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

        let filter = IGSProductFilter::new().with_center(IGSAnalysisCenter::COD);
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
            gps_week: week,
        }
    }

    fn mock_file<'a>(server: &'a MockServer, path: &str, body: Vec<u8>) -> httpmock::Mock<'a> {
        server.mock(|when, then| {
            when.method(GET).path(path.to_string());
            then.status(200).body(body);
        })
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
            vec!["..%2F..%2Fetc".to_string(), SP3_NAME.to_string()]
        );
        let client = IGSProductsClient::new(IGSProductSource::Garner);
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

    const SP3_FIXTURE: &str = "test_assets/sp3/IGS0OPSFIN_20243150000_01D_15M_ORB_subset.SP3.gz";

    impl IGSProductsClient {
        /// The products [`IGSProductsClient::sp3_range_files`] chooses, in order.
        fn sp3_range_products(
            &self,
            start: Epoch,
            end: Epoch,
            center: &IGSAnalysisCenter,
            solution: &IGSSolutionType,
            options: &IGSRangeOptions,
        ) -> Result<Vec<IGSProduct>, BraheError> {
            Ok(self
                .sp3_range_files(start, end, center, solution, options)?
                .into_iter()
                .map(|file| file.product)
                .collect())
        }
    }

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
            .download_sp3_range(
                start,
                end,
                IGSAnalysisCenter::parse("cod"),
                IGSSolutionType::Final,
                &options,
            )
            .unwrap();
        let expected: Vec<String> = (315..=318).map(|doy| cod_daily(doy, "15M")).collect();
        assert_eq!(sp3s.len(), 4);
        assert!(sp3s.iter().all(|sp3| *sp3 == sp3s[0]));
        assert_downloaded(&files, &expected);
        let chosen = client
            .sp3_range_products(
                start,
                end,
                &IGSAnalysisCenter::COD,
                &IGSSolutionType::Final,
                &options,
            )
            .unwrap();
        assert_eq!(names(&chosen), expected);

        let fine = options.with_sampling(300.0);
        let sp3s = client
            .download_sp3_range(
                start,
                end,
                IGSAnalysisCenter::COD,
                IGSSolutionType::Final,
                &fine,
            )
            .unwrap();
        assert_eq!(sp3s.len(), 4);
        let chosen = client
            .sp3_range_products(
                start,
                end,
                &IGSAnalysisCenter::COD,
                &IGSSolutionType::Final,
                &fine,
            )
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
            .download_sp3_range(
                start,
                end,
                IGSAnalysisCenter::COD,
                IGSSolutionType::Final,
                &options,
            )
            .unwrap();
        assert_eq!(sp3s.len(), 2);
        let expected = vec![cod_daily(316, "15M"), cod_daily(317, "15M")];
        assert_downloaded(&files, &expected);
        let chosen = client
            .sp3_range_products(
                start,
                end,
                &IGSAnalysisCenter::COD,
                &IGSSolutionType::Final,
                &options,
            )
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
                IGSAnalysisCenter::COD,
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
                &IGSAnalysisCenter::COD,
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
                &IGSAnalysisCenter::COD,
                &IGSSolutionType::Final,
                &options,
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains("for 2024-11-10"), "{err}");
        before.assert_calls(1);
    }

    /// Replaces the file mock for `name` in `files` with one answering `status`.
    fn fail_file<'a>(
        server: &'a MockServer,
        files: &mut BTreeMap<String, httpmock::Mock<'a>>,
        name: &str,
        status: u16,
    ) {
        files.remove(name).unwrap().delete();
        let failing = server.mock(|when, then| {
            when.method(GET).path(format!("/pub/products/2340/{name}"));
            then.status(status);
        });
        files.insert(name.to_string(), failing);
    }

    #[test]
    #[serial]
    fn test_igs_products_client_download_sp3_range_skips_padding_file_not_found() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let mut files = mock_cod_week_2340(&server, &[315, 316, 317, 318, 319]);
        fail_file(&server, &mut files, &cod_daily(318, "15M"), 404);
        let client = garner_client(&server);
        let options = IGSRangeOptions::new();
        let (start, end) = (gps(2024, 11, 11, 0), gps(2024, 11, 12, 0));

        let chosen = client
            .sp3_range_files(
                start,
                end,
                &IGSAnalysisCenter::COD,
                &IGSSolutionType::Final,
                &options,
            )
            .unwrap();
        let padding: Vec<bool> = chosen.iter().map(|file| file.padding).collect();
        assert_eq!(padding, [true, false, false, true]);

        // Day 318, the padding day after the span, is listed but answers 404.
        let sp3s = client
            .download_sp3_range(
                start,
                end,
                IGSAnalysisCenter::COD,
                IGSSolutionType::Final,
                &options,
            )
            .unwrap();
        assert_eq!(sp3s.len(), 3);
        let expected: Vec<String> = (315..=318).map(|doy| cod_daily(doy, "15M")).collect();
        assert_downloaded(&files, &expected);
        let week_dir = client.cache_dir().unwrap().join("garner/2340");
        assert!(!week_dir.join(cod_daily(318, "15M")).exists());

        // Inside the span the same 404 is an error naming the file.
        let err = client
            .download_sp3_range(
                gps(2024, 11, 12, 0),
                gps(2024, 11, 13, 0),
                IGSAnalysisCenter::COD,
                IGSSolutionType::Final,
                &options,
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains("not found"), "{err}");
        assert!(err.contains("Garner"), "{err}");
        assert!(err.contains(&cod_daily(318, "15M")), "{err}");
    }

    #[test]
    #[serial]
    fn test_igs_products_client_download_sp3_range_padding_server_error_fails() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let mut files = mock_cod_week_2340(&server, &[315, 316, 317, 318]);
        fail_file(&server, &mut files, &cod_daily(318, "15M"), 503);
        let client = garner_client(&server).max_retries(1);

        let err = client
            .download_sp3_range(
                gps(2024, 11, 11, 0),
                gps(2024, 11, 12, 0),
                IGSAnalysisCenter::COD,
                IGSSolutionType::Final,
                &IGSRangeOptions::new(),
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains("503"), "{err}");
        assert!(err.contains(&cod_daily(318, "15M")), "{err}");
        files[&cod_daily(318, "15M")].assert_calls(2);
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
                &IGSAnalysisCenter::COD,
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
    fn test_igs_products_client_download_sp3_range_missing_day_names_day_and_source() {
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
                IGSAnalysisCenter::COD,
                IGSSolutionType::Final,
                &IGSRangeOptions::new(),
            )
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("IGS Garner has no daily COD FIN SP3 orbit file for 2024-11-12"),
            "{err}"
        );
        assert!(err.contains("GPS week 2340, day 2"), "{err}");
        assert!(err.contains("/pub/products/2340/ lists none"), "{err}");

        let err = client
            .download_sp3_range(
                start,
                end,
                IGSAnalysisCenter::COD,
                IGSSolutionType::Final,
                &IGSRangeOptions::new().with_sampling(30.0),
            )
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("SP3 orbit file with 30 s sampling for 2024-11-11"),
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
                IGSAnalysisCenter::IGS,
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
                IGSAnalysisCenter::IGS,
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
                IGSAnalysisCenter::parse("cod"),
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
                &IGSAnalysisCenter::COD,
                &IGSSolutionType::Final,
                &options,
            )
            .unwrap();
        assert_eq!(names(&chosen), ["cod21001.sp3.gz"]);
        let chosen = client
            .sp3_range_products(
                gps(2020, 4, 8, 6),
                gps(2020, 4, 8, 12),
                &IGSAnalysisCenter::COD,
                &IGSSolutionType::Final,
                &options.clone().with_pad(false),
            )
            .unwrap();
        assert_eq!(names(&chosen), ["cod21003.sp3.gz"]);
        files.assert_calls(0);

        // A missing short-name day names the day.
        let err = client
            .sp3_range_products(
                gps(2020, 4, 9, 6),
                gps(2020, 4, 9, 12),
                &IGSAnalysisCenter::COD,
                &IGSSolutionType::Rapid,
                &options,
            )
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("no daily COD RAP SP3 orbit file for 2020-04-09"),
            "{err}"
        );
        assert!(err.contains("GPS week 2100, day 4"), "{err}");
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
            .download_sp3_range(
                start,
                end,
                IGSAnalysisCenter::COD,
                IGSSolutionType::Final,
                &options,
            )
            .unwrap();
        assert_eq!(sp3s.len(), 4);
        let fixture = SP3::from_file(SP3_FIXTURE).unwrap();
        assert!(sp3s.iter().all(|sp3| *sp3 == fixture));
        for mock in short_files.iter().chain(&long_files) {
            mock.assert_calls(1);
        }
        let chosen = client
            .sp3_range_products(
                start,
                end,
                &IGSAnalysisCenter::COD,
                &IGSSolutionType::Final,
                &options,
            )
            .unwrap();
        assert_eq!(names(&chosen), [short[0], short[1], long[0], long[1]]);
    }

    #[test]
    #[parallel]
    fn test_igs_gps_day() {
        assert_eq!(gps_day(gps(2024, 11, 10, 0)), 2340 * 7);
        assert_eq!(gps_day(gps(2024, 11, 16, 23)), 2340 * 7 + 6);
        assert_eq!(gps_day_start(2340 * 7 + 2), gps(2024, 11, 12, 0));
    }

    #[test]
    #[parallel]
    fn test_igs_daily_orbit_skips_other_formats() {
        let listed = |name: &str| IGSProduct {
            file_name: IGSProductFileName::parse(name).unwrap(),
            name: name.to_string(),
            url: String::new(),
            gps_week: 2340,
        };
        let products = vec![
            listed("GFZ0OPSRAP_20243150000_01D_05M_ORB.OBX.gz"),
            listed("GFZ0OPSRAP_20243150000_01D_15M_ORB.SP3.gz"),
        ];
        assert_eq!(
            daily_orbit(&products, 2340 * 7, None).unwrap().name,
            "GFZ0OPSRAP_20243150000_01D_15M_ORB.SP3.gz"
        );
        assert!(daily_orbit(&products[..1], 2340 * 7, None).is_none());
    }

    #[test]
    #[parallel]
    fn test_igs_daily_orbit_preference() {
        let listed = |name: &str| IGSProduct {
            file_name: IGSProductFileName::parse(name).unwrap(),
            name: name.to_string(),
            url: String::new(),
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
                    .with_center(IGSAnalysisCenter::IGS)
                    .with_solution(IGSSolutionType::Final)
                    .with_content(IGSContentType::Orbit),
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
        let client = IGSProductsClient::new(IGSProductSource::Garner);
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
        let client = IGSProductsClient::new(IGSProductSource::Garner);
        let epoch = Epoch::from_datetime(2020, 4, 8, 12, 0, 0.0, 0.0, TimeSystem::GPS);
        let products = client
            .list_products(
                epoch,
                &IGSProductFilter::new().with_center(IGSAnalysisCenter::IGS),
            )
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
        let client = IGSProductsClient::new(IGSProductSource::BKG);
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
                IGSAnalysisCenter::COD,
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
                IGSAnalysisCenter::COD,
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
    fn test_igs_products_client_download_sp3_range_missing_week_names_day_and_listing() {
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
                IGSAnalysisCenter::COD,
                IGSSolutionType::Final,
                &IGSRangeOptions::new().with_pad(false),
            )
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("IGS Garner has no daily COD FIN SP3 orbit file for 2024-11-18"),
            "{err}"
        );
        assert!(err.contains("GPS week 2341, day 1"), "{err}");
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
                IGSAnalysisCenter::COD,
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
