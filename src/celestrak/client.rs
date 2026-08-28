/*!
 * HTTP client for the CelestrakClient API.
 *
 * Provides access to CelestrakClient endpoints with file-based caching
 * and typed query execution. No authentication is required.
 */

use std::fs;
use std::path::Path;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant, SystemTime};

use crate::celestrak::filter::{apply_filters, apply_limit, apply_order_by};
use crate::celestrak::query::{CelestrakQuery, LocalSelector};
use crate::celestrak::responses::CelestrakSATCATRecord;
use crate::celestrak::types::{CelestrakOutputFormat, SupGPSource};
use crate::propagators::SGPPropagator;
use crate::types::GPRecord;
use crate::utils::network::{
    CacheDecision, NetworkMode, cache_policy, ensure_online, network_mode,
};
use crate::utils::{BraheError, atomic_write, get_celestrak_cache_dir};

/// Default base URL for the CelestrakClient API.
const DEFAULT_BASE_URL: &str = "https://celestrak.org";

/// Default maximum cache age in seconds (7200.0, 2 hours).
const DEFAULT_MAX_CACHE_AGE: f64 = 7200.0;

/// Default maximum number of retries for transient HTTP errors.
const DEFAULT_MAX_RETRIES: u32 = 3;

/// Minimum interval between HTTP requests to avoid overwhelming the server.
const MIN_REQUEST_INTERVAL: Duration = Duration::from_millis(500);

/// Celestrak group whose cached response answers single-object lookups locally.
const LOCAL_CATALOG_GROUP: &str = "active";

/// Process-global tracker for the last HTTP request time, shared across all
/// `CelestrakClient` instances to enforce rate limiting.
static LAST_REQUEST_TIME: LazyLock<Mutex<Option<Instant>>> = LazyLock::new(|| Mutex::new(None));

/// CelestrakClient API client with caching.
///
/// Provides typed query execution for GP, supplemental GP, and SATCAT
/// data from CelestrakClient. Responses are cached locally to reduce
/// server load and improve performance. The `BRAHE_NETWORK_MODE`
/// environment variable controls whether a stale cached response is
/// refreshed or served; see [`crate::utils::network`].
///
/// Single-object lookups (`get_gp_by_catnr`, `get_gp_by_intdes`,
/// `get_gp_by_name`, and `query_gp` with exactly one of those selectors)
/// are answered from a cached copy of the `active` group when the object
/// is in it, so a series of lookups costs one request; objects not in
/// `active` (debris, inactive objects) are requested individually. A
/// `name` search that matches in `active` returns only active objects.
///
/// # Examples
///
/// ```no_run
/// use brahe::celestrak::*;
///
/// let client = CelestrakClient::new();
///
/// let query = CelestrakQuery::gp()
///     .group("stations")
///     .format(CelestrakOutputFormat::Json);
///
/// let records = client.query_gp(&query).unwrap();
/// println!("Found {} records", records.len());
/// ```
pub struct CelestrakClient {
    base_url: String,
    cache_max_age: f64,
    max_retries: u32,
    agent: ureq::Agent,
}

impl CelestrakClient {
    /// Create a new CelestrakClient client with default settings.
    ///
    /// Uses the default base URL (`https://celestrak.org`) and
    /// a 2-hour cache TTL.
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::celestrak::CelestrakClient;
    ///
    /// let client = CelestrakClient::new();
    /// ```
    pub fn new() -> Self {
        CelestrakClient {
            base_url: DEFAULT_BASE_URL.to_string(),
            cache_max_age: DEFAULT_MAX_CACHE_AGE,
            max_retries: DEFAULT_MAX_RETRIES,
            agent: ureq::Agent::new_with_defaults(),
        }
    }

    /// Create a new CelestrakClient client with a custom cache duration.
    ///
    /// # Arguments
    ///
    /// * `cache_max_age` - Maximum cache age in seconds
    pub fn with_cache_age(cache_max_age: f64) -> Self {
        CelestrakClient {
            base_url: DEFAULT_BASE_URL.to_string(),
            cache_max_age,
            max_retries: DEFAULT_MAX_RETRIES,
            agent: ureq::Agent::new_with_defaults(),
        }
    }

    /// Create a new CelestrakClient client with a custom base URL.
    ///
    /// Useful for testing against a mock server.
    ///
    /// # Arguments
    ///
    /// * `base_url` - Custom base URL
    pub fn with_base_url(base_url: &str) -> Self {
        CelestrakClient {
            base_url: base_url.trim_end_matches('/').to_string(),
            cache_max_age: DEFAULT_MAX_CACHE_AGE,
            max_retries: DEFAULT_MAX_RETRIES,
            agent: ureq::Agent::new_with_defaults(),
        }
    }

    /// Create a new CelestrakClient client with a custom base URL and cache duration.
    ///
    /// # Arguments
    ///
    /// * `base_url` - Custom base URL
    /// * `cache_max_age` - Maximum cache age in seconds
    pub fn with_base_url_and_cache_age(base_url: &str, cache_max_age: f64) -> Self {
        CelestrakClient {
            base_url: base_url.trim_end_matches('/').to_string(),
            cache_max_age,
            max_retries: DEFAULT_MAX_RETRIES,
            agent: ureq::Agent::new_with_defaults(),
        }
    }

    /// Set the maximum number of retries for transient HTTP errors.
    ///
    /// # Arguments
    ///
    /// * `max_retries` - Maximum retry attempts (0 disables retries)
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::celestrak::CelestrakClient;
    ///
    /// let client = CelestrakClient::new().max_retries(5);
    /// ```
    pub fn max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Execute a query and return the raw response body as a string.
    ///
    /// Uses cached data if available and fresh enough. The output format
    /// used is whatever was specified in the query (or the default for the endpoint).
    ///
    /// # Arguments
    ///
    /// * `query` - The query to execute
    ///
    /// # Returns
    ///
    /// * `Ok(String)` - Raw response body
    /// * `Err(BraheError)` - On network or cache errors
    pub fn query_raw(&self, query: &CelestrakQuery) -> Result<String, BraheError> {
        let url = self.build_full_url(query);
        self.fetch_with_cache(&url)
    }

    /// Execute a query and save the response to a file.
    ///
    /// # Arguments
    ///
    /// * `query` - The query to execute
    /// * `filepath` - Path to save the response to
    ///
    /// # Returns
    ///
    /// * `Ok(())` - File saved successfully
    /// * `Err(BraheError)` - On network, cache, or I/O errors
    pub fn download(&self, query: &CelestrakQuery, filepath: &Path) -> Result<(), BraheError> {
        let body = self.query_raw(query)?;

        // Create parent directories if needed
        if let Some(parent) = filepath.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| BraheError::IoError(format!("Failed to create directories: {}", e)))?;
        }

        atomic_write(filepath, body.as_bytes())
            .map_err(|e| BraheError::IoError(format!("Failed to write file: {}", e)))
    }

    /// Execute a GP query and return typed GP records.
    ///
    /// Forces JSON format internally for deserialization. Applies any
    /// client-side filters, ordering, and limit specified in the query.
    ///
    /// Works for both GP and SupGP query types. A GP query whose
    /// server-side parameters are exactly one of `CATNR`, `INTDES`, or
    /// `NAME` is resolved in three steps: the exact per-object cache file,
    /// then the cached `active` group (a match returns immediately and is
    /// not written to the per-object cache key), then a per-object
    /// request. A `name` search that matches in `active` returns only
    /// active objects; a search with no match in `active` still goes to
    /// the server. `catnr` and `intdes` results are unaffected. Any other
    /// query (a group, `special`, `source`, combined selectors, and so
    /// on) is always sent to the server.
    ///
    /// # Arguments
    ///
    /// * `query` - The query to execute
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<GPRecord>)` - Typed GP records (same type as SpaceTrack!)
    /// * `Err(BraheError)` - On network, cache, or parse errors
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::celestrak::*;
    ///
    /// let client = CelestrakClient::new();
    /// let query = CelestrakQuery::gp()
    ///     .group("stations");
    ///
    /// let records = client.query_gp(&query).unwrap();
    /// for record in &records {
    ///     println!("{:?}: {:?}", record.object_name, record.norad_cat_id);
    /// }
    /// ```
    pub fn query_gp(&self, query: &CelestrakQuery) -> Result<Vec<GPRecord>, BraheError> {
        // Force JSON format for deserialization (Celestrak defaults to 3LE, not JSON)
        let json_query = if query.output_format().is_some_and(|f| f.is_json()) {
            query.clone()
        } else {
            query.clone().format(CelestrakOutputFormat::Json)
        };

        let mut records = match query.local_selector() {
            Some(selector) => self.resolve_single_object(&json_query, &selector)?,
            None => Self::parse_gp_records(&self.query_raw(&json_query)?)?,
        };

        // Apply client-side processing
        records = apply_filters(records, query.client_side_filters());
        apply_order_by(&mut records, query.client_side_order_by());
        records = apply_limit(records, query.client_side_limit());

        Ok(records)
    }

    /// Execute a SATCAT query and return typed SATCAT records.
    ///
    /// Forces JSON format internally for deserialization. Applies any
    /// client-side filters, ordering, and limit specified in the query.
    ///
    /// # Arguments
    ///
    /// * `query` - The query to execute
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<CelestrakSATCATRecord>)` - Typed SATCAT records
    /// * `Err(BraheError)` - On network, cache, or parse errors
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::celestrak::*;
    ///
    /// let client = CelestrakClient::new();
    /// let query = CelestrakQuery::satcat()
    ///     .active(true)
    ///     .payloads(true);
    ///
    /// let records = client.query_satcat(&query).unwrap();
    /// println!("Found {} active payloads", records.len());
    /// ```
    pub fn query_satcat(
        &self,
        query: &CelestrakQuery,
    ) -> Result<Vec<CelestrakSATCATRecord>, BraheError> {
        // Force JSON format for deserialization (Celestrak defaults to 3LE, not JSON)
        let json_query = if query.output_format().is_some_and(|f| f.is_json()) {
            query.clone()
        } else {
            query.clone().format(CelestrakOutputFormat::Json)
        };

        let body = self.query_raw(&json_query)?;
        let mut records: Vec<CelestrakSATCATRecord> = serde_json::from_str(&body).map_err(|e| {
            BraheError::ParseError(format!(
                "Failed to parse CelestrakClient SATCAT response: {}",
                e
            ))
        })?;

        // Apply client-side processing
        records = apply_filters(records, query.client_side_filters());
        apply_order_by(&mut records, query.client_side_order_by());
        records = apply_limit(records, query.client_side_limit());

        Ok(records)
    }

    // -- Convenience methods --

    /// Look up GP records by NORAD catalog number.
    ///
    /// # Arguments
    ///
    /// * `catnr` - NORAD catalog number (e.g., 25544 for ISS)
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<GPRecord>)` - Matching GP records, resolved from the cached
    ///   `active` group when the object is in it, otherwise requested
    ///   individually
    /// * `Err(BraheError)` - On network, cache, or parse errors
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::celestrak::CelestrakClient;
    ///
    /// let client = CelestrakClient::new();
    /// let records = client.get_gp_by_catnr(25544).unwrap();
    /// println!("ISS: {:?}", records[0].object_name);
    /// ```
    pub fn get_gp_by_catnr(&self, catnr: u32) -> Result<Vec<GPRecord>, BraheError> {
        let query = CelestrakQuery::gp().catnr(catnr);
        self.query_gp(&query)
    }

    /// Look up GP records by satellite group name.
    ///
    /// # Arguments
    ///
    /// * `group` - Group name (e.g., "stations", "active", "gnss")
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<GPRecord>)` - GP records in the group
    /// * `Err(BraheError)` - On network, cache, or parse errors
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::celestrak::CelestrakClient;
    ///
    /// let client = CelestrakClient::new();
    /// let records = client.get_gp_by_group("stations").unwrap();
    /// println!("Found {} records", records.len());
    /// ```
    pub fn get_gp_by_group(&self, group: &str) -> Result<Vec<GPRecord>, BraheError> {
        let query = CelestrakQuery::gp().group(group);
        self.query_gp(&query)
    }

    /// Look up GP records by satellite name (substring match).
    ///
    /// # Arguments
    ///
    /// * `name` - Satellite name to search for (partial match supported)
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<GPRecord>)` - Matching GP records. When the cached `active`
    ///   group contains a match, only active objects are returned; a search
    ///   with no match in `active` is sent to the server, which may include
    ///   inactive objects
    /// * `Err(BraheError)` - On network, cache, or parse errors
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::celestrak::CelestrakClient;
    ///
    /// let client = CelestrakClient::new();
    /// let records = client.get_gp_by_name("ISS").unwrap();
    /// ```
    pub fn get_gp_by_name(&self, name: &str) -> Result<Vec<GPRecord>, BraheError> {
        let query = CelestrakQuery::gp().name_search(name);
        self.query_gp(&query)
    }

    /// Look up GP records by international designator.
    ///
    /// # Arguments
    ///
    /// * `intdes` - International designator (e.g., "1998-067A")
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<GPRecord>)` - Matching GP records, resolved from the cached
    ///   `active` group when the object is in it, otherwise requested
    ///   individually
    /// * `Err(BraheError)` - On network, cache, or parse errors
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::celestrak::CelestrakClient;
    ///
    /// let client = CelestrakClient::new();
    /// let records = client.get_gp_by_intdes("1998-067A").unwrap();
    /// ```
    pub fn get_gp_by_intdes(&self, intdes: &str) -> Result<Vec<GPRecord>, BraheError> {
        let query = CelestrakQuery::gp().intdes(intdes);
        self.query_gp(&query)
    }

    /// Look up supplemental GP records by source.
    ///
    /// # Arguments
    ///
    /// * `source` - The supplemental data source
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<GPRecord>)` - GP records from the supplemental source
    /// * `Err(BraheError)` - On network, cache, or parse errors
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::celestrak::{CelestrakClient, SupGPSource};
    ///
    /// let client = CelestrakClient::new();
    /// let records = client.get_sup_gp(SupGPSource::Starlink).unwrap();
    /// ```
    pub fn get_sup_gp(&self, source: SupGPSource) -> Result<Vec<GPRecord>, BraheError> {
        let query = CelestrakQuery::sup_gp().source(source);
        self.query_gp(&query)
    }

    /// Look up SATCAT records by NORAD catalog number.
    ///
    /// # Arguments
    ///
    /// * `catnr` - NORAD catalog number
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<CelestrakSATCATRecord>)` - Matching SATCAT records
    /// * `Err(BraheError)` - On network, cache, or parse errors
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::celestrak::CelestrakClient;
    ///
    /// let client = CelestrakClient::new();
    /// let records = client.get_satcat_by_catnr(25544).unwrap();
    /// println!("ISS: {:?}", records[0].object_name);
    /// ```
    pub fn get_satcat_by_catnr(
        &self,
        catnr: u32,
    ) -> Result<Vec<CelestrakSATCATRecord>, BraheError> {
        let query = CelestrakQuery::satcat().catnr(catnr);
        self.query_satcat(&query)
    }

    /// Look up a satellite by NORAD catalog number and return an SGP4 propagator.
    ///
    /// Queries GP data for the given catalog number and creates an
    /// `SGPPropagator` from the first result.
    ///
    /// # Arguments
    ///
    /// * `catnr` - NORAD catalog number
    /// * `step_size` - Propagator step size in seconds
    ///
    /// # Returns
    ///
    /// * `Ok(SGPPropagator)` - Ready-to-use propagator
    /// * `Err(BraheError)` - If no records found or propagator creation fails
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::celestrak::CelestrakClient;
    ///
    /// let client = CelestrakClient::new();
    /// let propagator = client.get_sgp_propagator_by_catnr(25544, 60.0).unwrap();
    /// ```
    pub fn get_sgp_propagator_by_catnr(
        &self,
        catnr: u32,
        step_size: f64,
    ) -> Result<SGPPropagator, BraheError> {
        let records = self.get_gp_by_catnr(catnr)?;
        let record = records.first().ok_or_else(|| {
            BraheError::Error(format!(
                "No GP records found for NORAD catalog number {}",
                catnr
            ))
        })?;
        SGPPropagator::from_gp_record(record, step_size)
    }

    // -- Internal helpers --

    /// Build the full URL for a query.
    fn build_full_url(&self, query: &CelestrakQuery) -> String {
        let endpoint = query.query_type().endpoint_path();
        let params = query.build_url();
        if params.is_empty() {
            format!("{}{}", self.base_url, endpoint)
        } else {
            format!("{}{}?{}", self.base_url, endpoint, params)
        }
    }

    /// Fetch a URL with file-based caching.
    ///
    /// The `BRAHE_NETWORK_MODE` environment variable controls whether a stale
    /// cached response is refreshed or served; see [`crate::utils::network`].
    ///
    /// # Arguments
    ///
    /// * `url` - Full URL to fetch, used both as the cache key source and as
    ///   the request target if the cache is missing or refreshed
    ///
    /// # Returns
    ///
    /// * `Ok(String)` - Response body, from cache or freshly downloaded
    /// * `Err(BraheError)` - On cache I/O errors, or if `BRAHE_NETWORK_MODE`
    ///   forbids the request needed to fill or refresh the cache
    fn fetch_with_cache(&self, url: &str) -> Result<String, BraheError> {
        let cache_key = self.cache_key_for_url(url);

        // Check cache
        if let Some(cached) = self.read_cache(&cache_key)? {
            return Ok(cached);
        }

        // Fetch from network
        let body = self.execute_get(url)?;

        // Write to cache
        self.write_cache(&cache_key, &body)?;

        Ok(body)
    }

    /// Parse a JSON GP response body into records.
    ///
    /// # Arguments
    ///
    /// * `body` - JSON array of GP records as returned by Celestrak
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<GPRecord>)` - Parsed records
    /// * `Err(BraheError)` - If the body is not a JSON array of GP records
    fn parse_gp_records(body: &str) -> Result<Vec<GPRecord>, BraheError> {
        serde_json::from_str(body).map_err(|e| {
            BraheError::ParseError(format!(
                "Failed to parse CelestrakClient GP response: {}",
                e
            ))
        })
    }

    /// Resolve a single-object GP query: exact per-object cache, then the
    /// cached `active` group, then a per-object request.
    ///
    /// # Arguments
    ///
    /// * `json_query` - The query with JSON output format applied
    /// * `selector` - The object the query names
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<GPRecord>)` - Matching records from whichever step answered
    /// * `Err(BraheError)` - On cache I/O or parse errors, or when the
    ///   per-object request is needed and `BRAHE_NETWORK_MODE` forbids it
    fn resolve_single_object(
        &self,
        json_query: &CelestrakQuery,
        selector: &LocalSelector,
    ) -> Result<Vec<GPRecord>, BraheError> {
        let url = self.build_full_url(json_query);
        let cache_key = self.cache_key_for_url(&url);
        if let Some(body) = self.read_cache(&cache_key)? {
            return Self::parse_gp_records(&body);
        }
        if let Some(records) = self.resolve_from_active(selector)? {
            return Ok(records);
        }
        Self::parse_gp_records(&self.query_raw(json_query)?)
    }

    /// Load the `active` group through the ordinary cached fetch path.
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<GPRecord>)` - Every record in the `active` group
    /// * `Err(BraheError)` - If the group is neither servable from cache nor
    ///   fetchable under the current `BRAHE_NETWORK_MODE`
    fn active_catalog(&self) -> Result<Vec<GPRecord>, BraheError> {
        let query = CelestrakQuery::gp()
            .group(LOCAL_CATALOG_GROUP)
            .format(CelestrakOutputFormat::Json);
        let url = self.build_full_url(&query);
        Self::parse_gp_records(&self.fetch_with_cache(&url)?)
    }

    /// Match a selector against the `active` group.
    ///
    /// # Arguments
    ///
    /// * `selector` - The object to look for
    ///
    /// # Returns
    ///
    /// * `Ok(Some(Vec<GPRecord>))` - One or more active records matched
    /// * `Ok(None)` - No active record matched, or the group could not be
    ///   obtained (a warning is printed in `online` mode); the caller falls
    ///   back to a per-object request
    /// * `Err(BraheError)` - If the network mode cannot be read
    fn resolve_from_active(
        &self,
        selector: &LocalSelector,
    ) -> Result<Option<Vec<GPRecord>>, BraheError> {
        let catalog = match self.active_catalog() {
            Ok(catalog) => catalog,
            Err(e) => {
                if network_mode()? == NetworkMode::Online {
                    eprintln!(
                        "Warning: Celestrak active catalog unavailable ({}); requesting the object directly",
                        e
                    );
                }
                return Ok(None);
            }
        };
        let matches: Vec<GPRecord> = catalog
            .into_iter()
            .filter(|record| selector.matches(record))
            .collect();
        Ok((!matches.is_empty()).then_some(matches))
    }

    /// Generate a cache key from a URL.
    fn cache_key_for_url(&self, url: &str) -> String {
        // Use a simple sanitization: replace non-alphanumeric chars with underscores
        url.chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '.' {
                    c
                } else {
                    '_'
                }
            })
            .collect()
    }

    /// Read cached data, applying the `BRAHE_NETWORK_MODE` cache policy.
    ///
    /// # Arguments
    ///
    /// * `cache_key` - Cache file name, as produced by [`Self::cache_key_for_url`]
    ///
    /// # Returns
    ///
    /// * `Ok(Some(String))` - Cached body, either fresh or served stale under
    ///   `offline`
    /// * `Ok(None)` - No cache file exists, or it is stale and `online` mode
    ///   allows a refresh
    /// * `Err(BraheError)` - On cache I/O errors, or if the cache file is
    ///   stale and `offline-strict` forbids serving it
    fn read_cache(&self, cache_key: &str) -> Result<Option<String>, BraheError> {
        let cache_dir = get_celestrak_cache_dir()?;
        let cache_path = Path::new(&cache_dir).join(cache_key);

        if !cache_path.exists() {
            return Ok(None);
        }

        let stale = self.is_cache_stale(&cache_path)?;
        let resource = format!("Celestrak cached response {cache_key}");
        if cache_policy(&resource, stale)? == CacheDecision::Refresh {
            return Ok(None);
        }

        let contents = fs::read_to_string(&cache_path)
            .map_err(|e| BraheError::IoError(format!("Failed to read cache file: {}", e)))?;

        Ok(Some(contents))
    }

    /// Write data to the cache.
    fn write_cache(&self, cache_key: &str, data: &str) -> Result<(), BraheError> {
        let cache_dir = get_celestrak_cache_dir()?;
        let cache_path = Path::new(&cache_dir).join(cache_key);

        atomic_write(&cache_path, data.as_bytes())
            .map_err(|e| BraheError::IoError(format!("Failed to write cache file: {}", e)))
    }

    /// Check if a cache file is older than the maximum cache age.
    fn is_cache_stale(&self, path: &Path) -> Result<bool, BraheError> {
        let metadata = fs::metadata(path)
            .map_err(|e| BraheError::IoError(format!("Failed to read file metadata: {}", e)))?;

        let modified = metadata.modified().map_err(|e| {
            BraheError::IoError(format!("Failed to read file modification time: {}", e))
        })?;

        let age = SystemTime::now()
            .duration_since(modified)
            .unwrap_or_default();

        Ok(age.as_secs_f64() > self.cache_max_age)
    }

    /// Execute an HTTP GET request and return the response body.
    ///
    /// Retries on transient errors (server errors, connection resets, timeouts)
    /// with exponential backoff and jitter. Enforces a minimum interval between
    /// requests via a process-global rate limiter.
    ///
    /// # Arguments
    ///
    /// * `url` - Full URL to request
    ///
    /// # Returns
    ///
    /// * `Ok(String)` - Response body
    /// * `Err(BraheError)` - If `BRAHE_NETWORK_MODE` is `offline` or
    ///   `offline-strict` and `url` is not a loopback address, so no request
    ///   is attempted, or if the request fails after exhausting retries
    fn execute_get(&self, url: &str) -> Result<String, BraheError> {
        ensure_online(url, &format!("Celestrak request {url}"))?;

        let mut last_error = None;

        for attempt in 0..=self.max_retries {
            Self::rate_limit_wait();

            if attempt > 0 {
                let base_delay = Duration::from_secs(1) * 2u32.pow(attempt - 1);
                let nanos = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .subsec_nanos();
                let jitter = Duration::from_millis((nanos % 500) as u64);
                std::thread::sleep(base_delay + jitter);
            }

            match self.agent.get(url).call() {
                Ok(mut response) => {
                    return response.body_mut().read_to_string().map_err(|e| {
                        BraheError::IoError(format!(
                            "Failed to read CelestrakClient response: {}",
                            e
                        ))
                    });
                }
                Err(e) => {
                    if attempt < self.max_retries && Self::is_retryable_error(&e) {
                        last_error = Some(e);
                        continue;
                    }
                    return Err(BraheError::IoError(format!(
                        "CelestrakClient request failed: {}",
                        e
                    )));
                }
            }
        }

        // Only reachable if max_retries > 0 and all attempts were retryable errors
        Err(BraheError::IoError(format!(
            "CelestrakClient request failed: {}",
            last_error.unwrap()
        )))
    }

    /// Check whether an HTTP error is transient and worth retrying.
    fn is_retryable_error(e: &ureq::Error) -> bool {
        matches!(
            e,
            ureq::Error::StatusCode(429 | 500 | 502 | 503 | 504)
                | ureq::Error::Io(_)
                | ureq::Error::Timeout(_)
                | ureq::Error::ConnectionFailed
        )
    }

    /// Enforce a minimum interval between HTTP requests process-wide.
    fn rate_limit_wait() {
        let mut last_time = LAST_REQUEST_TIME.lock().unwrap();
        if let Some(last) = *last_time {
            let elapsed = last.elapsed();
            if elapsed < MIN_REQUEST_INTERVAL {
                std::thread::sleep(MIN_REQUEST_INTERVAL - elapsed);
            }
        }
        *last_time = Some(Instant::now());
    }
}

impl Default for CelestrakClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::utils::testing::{CacheRedirect, NetworkModeGuard, setup_global_test_eop};
    use httpmock::prelude::*;
    use serial_test::serial;
    use std::fs;
    use std::time::{Duration, SystemTime};

    #[test]
    #[serial_test::parallel]
    fn test_client_creation() {
        let client = CelestrakClient::new();
        assert_eq!(client.base_url, DEFAULT_BASE_URL);
        assert_eq!(client.cache_max_age, DEFAULT_MAX_CACHE_AGE);
    }

    #[test]
    #[serial_test::parallel]
    fn test_client_with_cache_age() {
        let client = CelestrakClient::with_cache_age(3600.0);
        assert_eq!(client.base_url, DEFAULT_BASE_URL);
        assert_eq!(client.cache_max_age, 3600.0);
    }

    #[test]
    #[serial_test::parallel]
    fn test_client_with_base_url() {
        let client = CelestrakClient::with_base_url("https://test.celestrak.org/");
        assert_eq!(client.base_url, "https://test.celestrak.org");
    }

    #[test]
    #[serial_test::parallel]
    fn test_client_with_base_url_no_trailing_slash() {
        let client = CelestrakClient::with_base_url("https://test.celestrak.org");
        assert_eq!(client.base_url, "https://test.celestrak.org");
    }

    #[test]
    #[serial_test::parallel]
    fn test_client_with_base_url_and_cache_age() {
        let client =
            CelestrakClient::with_base_url_and_cache_age("https://test.celestrak.org", 1800.0);
        assert_eq!(client.base_url, "https://test.celestrak.org");
        assert_eq!(client.cache_max_age, 1800.0);
    }

    #[test]
    #[serial_test::parallel]
    fn test_client_default() {
        let client = CelestrakClient::default();
        assert_eq!(client.base_url, DEFAULT_BASE_URL);
    }

    #[test]
    #[serial_test::parallel]
    fn test_build_full_url_gp_with_params() {
        let client = CelestrakClient::new();
        let query = CelestrakQuery::gp().group("stations");
        let url = client.build_full_url(&query);
        assert_eq!(
            url,
            "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations"
        );
    }

    #[test]
    #[serial_test::parallel]
    fn test_build_full_url_gp_empty() {
        let client = CelestrakClient::new();
        let query = CelestrakQuery::gp();
        let url = client.build_full_url(&query);
        assert_eq!(url, "https://celestrak.org/NORAD/elements/gp.php");
    }

    #[test]
    #[serial_test::parallel]
    fn test_build_full_url_sup_gp() {
        let client = CelestrakClient::new();
        let query = CelestrakQuery::sup_gp().source(crate::celestrak::SupGPSource::SpaceX);
        let url = client.build_full_url(&query);
        assert_eq!(
            url,
            "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?SOURCE=spacex"
        );
    }

    #[test]
    #[serial_test::parallel]
    fn test_build_full_url_satcat() {
        let client = CelestrakClient::new();
        let query = CelestrakQuery::satcat().active(true);
        let url = client.build_full_url(&query);
        assert_eq!(url, "https://celestrak.org/satcat/records.php?ACTIVE=Y");
    }

    #[test]
    #[serial_test::serial]
    fn test_query_raw_gp() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(GET)
                .path("/NORAD/elements/gp.php")
                .query_param("GROUP", "stations")
                .query_param("FORMAT", "JSON");
            then.status(200)
                .body(r#"[{"OBJECT_NAME":"ISS (ZARYA)","NORAD_CAT_ID":"25544"}]"#);
        });

        let client = CelestrakClient::with_base_url_and_cache_age(&server.base_url(), 0.0);

        let query = CelestrakQuery::gp()
            .group("stations")
            .format(CelestrakOutputFormat::Json);

        let result = client.query_raw(&query);
        assert!(result.is_ok());
        assert!(result.unwrap().contains("ISS"));
    }

    #[test]
    #[serial_test::serial]
    fn test_query_gp_typed() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(GET)
                .path("/NORAD/elements/gp.php")
                .query_param("GROUP", "stations");
            then.status(200).body(
                r#"[{
                    "OBJECT_NAME": "ISS (ZARYA)",
                    "NORAD_CAT_ID": "25544",
                    "INCLINATION": "51.6400",
                    "ECCENTRICITY": "0.00010000"
                }]"#,
            );
        });

        let client = CelestrakClient::with_base_url_and_cache_age(&server.base_url(), 0.0);

        let query = CelestrakQuery::gp().group("stations");
        let result = client.query_gp(&query);
        assert!(result.is_ok());
        let records = result.unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].object_name.as_deref(), Some("ISS (ZARYA)"));
        assert_eq!(records[0].norad_cat_id, Some(25544));
    }

    #[test]
    #[serial_test::serial]
    fn test_query_gp_with_client_side_filter() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(GET).path("/NORAD/elements/gp.php");
            then.status(200).body(
                r#"[
                    {"OBJECT_NAME": "ISS (ZARYA)", "NORAD_CAT_ID": "25544", "INCLINATION": "51.64", "OBJECT_TYPE": "PAYLOAD"},
                    {"OBJECT_NAME": "COSMOS DEB", "NORAD_CAT_ID": "33767", "INCLINATION": "74.03", "OBJECT_TYPE": "DEBRIS"},
                    {"OBJECT_NAME": "NOAA 18", "NORAD_CAT_ID": "28654", "INCLINATION": "98.70", "OBJECT_TYPE": "PAYLOAD"}
                ]"#,
            );
        });

        let client = CelestrakClient::with_base_url_and_cache_age(&server.base_url(), 0.0);

        let query = CelestrakQuery::gp()
            .group("active")
            .filter("OBJECT_TYPE", "<>DEBRIS")
            .filter("INCLINATION", ">60");

        let result = client.query_gp(&query);
        assert!(result.is_ok());
        let records = result.unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].object_name.as_deref(), Some("NOAA 18"));
    }

    #[test]
    #[serial_test::serial]
    fn test_query_gp_with_order_and_limit() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(GET).path("/NORAD/elements/gp.php");
            then.status(200).body(
                r#"[
                    {"OBJECT_NAME": "A", "NORAD_CAT_ID": "100", "INCLINATION": "30"},
                    {"OBJECT_NAME": "B", "NORAD_CAT_ID": "200", "INCLINATION": "60"},
                    {"OBJECT_NAME": "C", "NORAD_CAT_ID": "300", "INCLINATION": "90"}
                ]"#,
            );
        });

        let client = CelestrakClient::with_base_url_and_cache_age(&server.base_url(), 0.0);

        let query = CelestrakQuery::gp()
            .group("active")
            .order_by("INCLINATION", false)
            .limit(2);

        let result = client.query_gp(&query);
        assert!(result.is_ok());
        let records = result.unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].object_name.as_deref(), Some("C"));
        assert_eq!(records[1].object_name.as_deref(), Some("B"));
    }

    #[test]
    #[serial_test::serial]
    fn test_query_satcat_typed() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(GET)
                .path("/satcat/records.php")
                .query_param("ACTIVE", "Y");
            then.status(200).body(
                r#"[{
                    "OBJECT_NAME": "ISS (ZARYA)",
                    "NORAD_CAT_ID": "25544",
                    "OBJECT_TYPE": "PAY",
                    "OWNER": "ISS"
                }]"#,
            );
        });

        let client = CelestrakClient::with_base_url_and_cache_age(&server.base_url(), 0.0);

        let query = CelestrakQuery::satcat().active(true);
        let result = client.query_satcat(&query);
        assert!(result.is_ok());
        let records = result.unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].object_name.as_deref(), Some("ISS (ZARYA)"));
    }

    #[test]
    #[serial_test::serial]
    fn test_query_raw_tle_format() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();

        let tle_data = "ISS (ZARYA)\n1 25544U 98067A   24015.50000000\n2 25544  51.6400";
        server.mock(|when, then| {
            when.method(GET)
                .path("/NORAD/elements/gp.php")
                .query_param("GROUP", "stations")
                .query_param("FORMAT", "3LE");
            then.status(200).body(tle_data);
        });

        let client = CelestrakClient::with_base_url_and_cache_age(&server.base_url(), 0.0);

        let query = CelestrakQuery::gp()
            .group("stations")
            .format(CelestrakOutputFormat::ThreeLe);

        let result = client.query_raw(&query);
        assert!(result.is_ok());
        assert!(result.unwrap().contains("25544"));
    }

    #[test]
    #[serial_test::serial]
    fn test_http_error_404() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(GET).path("/NORAD/elements/gp.php");
            then.status(404).body("Not Found");
        });

        let client = CelestrakClient::with_base_url_and_cache_age(&server.base_url(), 0.0);

        let query = CelestrakQuery::gp().group("nonexistent");
        let result = client.query_raw(&query);
        assert!(result.is_err());
    }

    #[test]
    #[serial_test::serial]
    fn test_invalid_json_response() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(GET).path("/NORAD/elements/gp.php");
            then.status(200).body("this is not json");
        });

        let client = CelestrakClient::with_base_url_and_cache_age(&server.base_url(), 0.0);

        let query = CelestrakQuery::gp().group("stations");
        let result = client.query_gp(&query);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("parse"));
    }

    #[test]
    #[serial_test::serial]
    fn test_empty_json_response() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(GET).path("/NORAD/elements/gp.php");
            then.status(200).body("[]");
        });

        let client = CelestrakClient::with_base_url_and_cache_age(&server.base_url(), 0.0);

        let query = CelestrakQuery::gp().group("stations");
        let result = client.query_gp(&query);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    #[serial_test::serial]
    fn test_download_to_file() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(GET).path("/NORAD/elements/gp.php");
            then.status(200).body("test data content");
        });

        let client = CelestrakClient::with_base_url_and_cache_age(&server.base_url(), 0.0);

        let temp_dir = std::env::temp_dir().join("brahe_test_celestrak_download");
        let _ = fs::remove_dir_all(&temp_dir);
        let filepath = temp_dir.join("test_output.txt");

        let query = CelestrakQuery::gp()
            .group("stations")
            .format(CelestrakOutputFormat::ThreeLe);

        let result = client.download(&query, &filepath);
        assert!(result.is_ok());
        assert!(filepath.exists());

        let contents = fs::read_to_string(&filepath).unwrap();
        assert_eq!(contents, "test data content");

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    #[serial_test::parallel]
    fn test_cache_key_generation() {
        let client = CelestrakClient::new();
        let key = client.cache_key_for_url("https://celestrak.org/gp.php?GROUP=stations");
        assert!(key.contains("celestrak.org"));
        assert!(key.contains("GROUP"));
        assert!(!key.contains("?"));
        assert!(!key.contains("/"));
    }

    const ISS_GP_JSON: &str = r#"[{"OBJECT_NAME":"ISS (ZARYA)","NORAD_CAT_ID":"25544"}]"#;

    /// Write `body` into the redirected Celestrak cache under the key the client
    /// would compute for `url`, and back-date its mtime by `age`.
    fn seed_celestrak_cache(client: &CelestrakClient, url: &str, body: &str, age: Duration) {
        let dir = crate::utils::get_celestrak_cache_dir().unwrap();
        let path = std::path::Path::new(&dir).join(client.cache_key_for_url(url));
        fs::write(&path, body).unwrap();
        let mtime = SystemTime::now() - age;
        let file = fs::File::options().write(true).open(&path).unwrap();
        file.set_modified(mtime).unwrap();
    }

    #[test]
    #[serial]
    fn test_offline_serves_stale_cache_without_request() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("offline"));
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/NORAD/elements/gp.php");
            then.status(200).body(ISS_GP_JSON);
        });
        let client = CelestrakClient::with_base_url(&server.base_url());
        let url = format!(
            "{}/NORAD/elements/gp.php?CATNR=25544&FORMAT=JSON",
            server.base_url()
        );
        seed_celestrak_cache(&client, &url, ISS_GP_JSON, Duration::from_secs(30 * 86400));

        let records = client.get_gp_by_catnr(25544).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].norad_cat_id, Some(25544));
        assert_eq!(mock.calls(), 0);
    }

    #[test]
    #[serial]
    fn test_offline_miss_errors_without_request() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("offline"));
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/NORAD/elements/gp.php");
            then.status(200).body(ISS_GP_JSON);
        });
        let client = CelestrakClient::with_base_url("https://brahe-network-mode-test.invalid");

        let err = client.get_gp_by_catnr(25544).unwrap_err().to_string();
        assert!(
            err.starts_with("BRAHE_NETWORK_MODE is offline; Celestrak request "),
            "{err}"
        );
        assert_eq!(mock.calls(), 0);
    }

    #[test]
    #[serial]
    fn test_offline_strict_stale_cache_errors() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("offline-strict"));
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/NORAD/elements/gp.php");
            then.status(200).body(ISS_GP_JSON);
        });
        let client = CelestrakClient::with_base_url(&server.base_url());
        let url = format!(
            "{}/NORAD/elements/gp.php?CATNR=25544&FORMAT=JSON",
            server.base_url()
        );
        seed_celestrak_cache(&client, &url, ISS_GP_JSON, Duration::from_secs(30 * 86400));

        let err = client.get_gp_by_catnr(25544).unwrap_err().to_string();
        assert!(err.contains("is older than its cache limit"), "{err}");
        assert_eq!(mock.calls(), 0);
    }

    #[test]
    #[serial]
    fn test_offline_strict_fresh_cache_is_served() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("offline-strict"));
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/NORAD/elements/gp.php");
            then.status(200).body(ISS_GP_JSON);
        });
        let client = CelestrakClient::with_base_url(&server.base_url());
        let url = format!(
            "{}/NORAD/elements/gp.php?CATNR=25544&FORMAT=JSON",
            server.base_url()
        );
        seed_celestrak_cache(&client, &url, ISS_GP_JSON, Duration::from_secs(60));

        let records = client.get_gp_by_catnr(25544).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(mock.calls(), 0);
    }

    // -- Convenience method tests --

    #[test]
    #[serial_test::serial]
    fn test_get_gp_by_catnr() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(GET)
                .path("/NORAD/elements/gp.php")
                .query_param("CATNR", "25544");
            then.status(200).body(
                r#"[{
                    "OBJECT_NAME": "ISS (ZARYA)",
                    "NORAD_CAT_ID": "25544",
                    "INCLINATION": "51.6400"
                }]"#,
            );
        });

        let client = CelestrakClient::with_base_url_and_cache_age(&server.base_url(), 0.0);
        let records = client.get_gp_by_catnr(25544).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].norad_cat_id, Some(25544));
    }

    #[test]
    #[serial_test::serial]
    fn test_get_gp_by_group() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(GET)
                .path("/NORAD/elements/gp.php")
                .query_param("GROUP", "stations");
            then.status(200).body(
                r#"[{
                    "OBJECT_NAME": "ISS (ZARYA)",
                    "NORAD_CAT_ID": "25544"
                }]"#,
            );
        });

        let client = CelestrakClient::with_base_url_and_cache_age(&server.base_url(), 0.0);
        let records = client.get_gp_by_group("stations").unwrap();
        assert_eq!(records.len(), 1);
    }

    #[test]
    #[serial_test::serial]
    fn test_get_gp_by_name() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(GET)
                .path("/NORAD/elements/gp.php")
                .query_param("NAME", "ISS");
            then.status(200).body(
                r#"[{
                    "OBJECT_NAME": "ISS (ZARYA)",
                    "NORAD_CAT_ID": "25544"
                }]"#,
            );
        });

        let client = CelestrakClient::with_base_url_and_cache_age(&server.base_url(), 0.0);
        let records = client.get_gp_by_name("ISS").unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].object_name.as_deref(), Some("ISS (ZARYA)"));
    }

    #[test]
    #[serial_test::serial]
    fn test_get_gp_by_intdes() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(GET)
                .path("/NORAD/elements/gp.php")
                .query_param("INTDES", "1998-067A");
            then.status(200).body(
                r#"[{
                    "OBJECT_NAME": "ISS (ZARYA)",
                    "NORAD_CAT_ID": "25544"
                }]"#,
            );
        });

        let client = CelestrakClient::with_base_url_and_cache_age(&server.base_url(), 0.0);
        let records = client.get_gp_by_intdes("1998-067A").unwrap();
        assert_eq!(records.len(), 1);
    }

    #[test]
    #[serial_test::serial]
    fn test_get_sup_gp() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(GET)
                .path("/NORAD/elements/supplemental/sup-gp.php")
                .query_param("SOURCE", "spacex");
            then.status(200).body(
                r#"[{
                    "OBJECT_NAME": "STARLINK-1234",
                    "NORAD_CAT_ID": "44000"
                }]"#,
            );
        });

        let client = CelestrakClient::with_base_url_and_cache_age(&server.base_url(), 0.0);
        let records = client.get_sup_gp(SupGPSource::SpaceX).unwrap();
        assert_eq!(records.len(), 1);
    }

    #[test]
    #[serial_test::serial]
    fn test_get_satcat_by_catnr() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(GET)
                .path("/satcat/records.php")
                .query_param("CATNR", "25544");
            then.status(200).body(
                r#"[{
                    "OBJECT_NAME": "ISS (ZARYA)",
                    "NORAD_CAT_ID": "25544",
                    "OBJECT_TYPE": "PAY"
                }]"#,
            );
        });

        let client = CelestrakClient::with_base_url_and_cache_age(&server.base_url(), 0.0);
        let records = client.get_satcat_by_catnr(25544).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].norad_cat_id, Some(25544));
    }

    #[test]
    #[serial_test::serial]
    fn test_get_sgp_propagator_by_catnr_empty_results() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(GET)
                .path("/NORAD/elements/gp.php")
                .query_param("CATNR", "99999");
            then.status(200).body("[]");
        });

        let client = CelestrakClient::with_base_url_and_cache_age(&server.base_url(), 0.0);
        let result = client.get_sgp_propagator_by_catnr(99999, 60.0);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("No GP records found")
        );
    }

    // -- Retry behavior tests --

    #[test]
    #[serial_test::serial]
    fn test_retry_on_503() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();

        let mock = server.mock(|when, then| {
            when.method(GET).path("/NORAD/elements/gp.php");
            then.status(503);
        });

        let client =
            CelestrakClient::with_base_url_and_cache_age(&server.base_url(), 0.0).max_retries(1);

        let query = CelestrakQuery::gp().group("stations");
        let result = client.query_raw(&query);
        assert!(result.is_err());
        mock.assert_calls(2); // 1 initial + 1 retry
    }

    #[test]
    #[serial_test::serial]
    fn test_no_retry_on_404() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();

        let mock = server.mock(|when, then| {
            when.method(GET).path("/NORAD/elements/gp.php");
            then.status(404);
        });

        let client =
            CelestrakClient::with_base_url_and_cache_age(&server.base_url(), 0.0).max_retries(3);

        let query = CelestrakQuery::gp().group("nonexistent");
        let result = client.query_raw(&query);
        assert!(result.is_err());
        mock.assert_calls(1); // No retry for 404
    }

    #[test]
    #[serial_test::serial]
    fn test_max_retries_zero_no_retry() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();

        let mock = server.mock(|when, then| {
            when.method(GET).path("/NORAD/elements/gp.php");
            then.status(503);
        });

        let client =
            CelestrakClient::with_base_url_and_cache_age(&server.base_url(), 0.0).max_retries(0);

        let query = CelestrakQuery::gp().group("stations");
        let result = client.query_raw(&query);
        assert!(result.is_err());
        mock.assert_calls(1); // No retries with max_retries=0
    }

    #[test]
    #[serial_test::parallel]
    fn test_max_retries_builder() {
        let client = CelestrakClient::new().max_retries(5);
        assert_eq!(client.max_retries, 5);

        let client = CelestrakClient::new().max_retries(0);
        assert_eq!(client.max_retries, 0);
    }

    // -- CI-gated live integration tests --

    #[test]
    #[cfg_attr(not(feature = "integration"), ignore)]
    #[serial_test::serial]
    fn test_integration_gp_by_group() {
        let client = CelestrakClient::with_cache_age(0.0);
        let records = client.get_gp_by_group("stations").expect("GP query failed");
        assert!(!records.is_empty(), "Expected at least one GP record");
    }

    #[test]
    #[cfg_attr(not(feature = "integration"), ignore)]
    #[serial_test::serial]
    fn test_integration_gp_by_catnr() {
        let client = CelestrakClient::with_cache_age(0.0);
        let records = client.get_gp_by_catnr(25544).expect("GP query failed");
        assert!(!records.is_empty(), "Expected ISS GP record");
        assert_eq!(records[0].norad_cat_id, Some(25544));
    }

    #[test]
    #[cfg_attr(not(feature = "integration"), ignore)]
    #[serial_test::serial]
    fn test_integration_gp_by_name() {
        let client = CelestrakClient::with_cache_age(0.0);
        let records = client.get_gp_by_name("ISS").expect("GP query failed");
        assert!(
            !records.is_empty(),
            "Expected at least one record matching ISS"
        );
    }

    #[test]
    #[cfg_attr(not(feature = "integration"), ignore)]
    #[serial_test::serial]
    fn test_integration_satcat() {
        let client = CelestrakClient::with_cache_age(0.0);
        let records = client
            .get_satcat_by_catnr(25544)
            .expect("SATCAT query failed");
        assert!(!records.is_empty(), "Expected ISS SATCAT record");
        assert_eq!(records[0].norad_cat_id, Some(25544));
    }

    #[test]
    #[cfg_attr(not(feature = "integration"), ignore)]
    #[serial_test::serial]
    fn test_integration_get_sgp_propagator_by_catnr() {
        let client = CelestrakClient::with_cache_age(0.0);
        let propagator = client
            .get_sgp_propagator_by_catnr(25544, 60.0)
            .expect("SGP propagator creation failed");
        assert_eq!(propagator.norad_id, 25544);
    }

    // -- Active-group resolution tests --

    const ACTIVE_JSON: &str = r#"[
 {"OBJECT_NAME":"ISS (ZARYA)","OBJECT_ID":"1998-067A","EPOCH":"2026-08-27T12:00:00.000000","MEAN_MOTION":15.49,"ECCENTRICITY":0.0006,"INCLINATION":51.64,"RA_OF_ASC_NODE":120.5,"ARG_OF_PERICENTER":30.2,"MEAN_ANOMALY":329.9,"EPHEMERIS_TYPE":0,"CLASSIFICATION_TYPE":"U","NORAD_CAT_ID":25544,"ELEMENT_SET_NO":999,"REV_AT_EPOCH":54000,"BSTAR":0.0001,"MEAN_MOTION_DOT":0.0001,"MEAN_MOTION_DDOT":0},
 {"OBJECT_NAME":"ISS (NAUKA)","OBJECT_ID":"2021-066A","EPOCH":"2026-08-27T12:00:00.000000","MEAN_MOTION":15.49,"ECCENTRICITY":0.0006,"INCLINATION":51.64,"RA_OF_ASC_NODE":120.5,"ARG_OF_PERICENTER":30.2,"MEAN_ANOMALY":329.9,"EPHEMERIS_TYPE":0,"CLASSIFICATION_TYPE":"U","NORAD_CAT_ID":49044,"ELEMENT_SET_NO":999,"REV_AT_EPOCH":28000,"BSTAR":0.0001,"MEAN_MOTION_DOT":0.0001,"MEAN_MOTION_DDOT":0},
 {"OBJECT_NAME":"NISAR","OBJECT_ID":"2025-158A","EPOCH":"2026-08-27T12:00:00.000000","MEAN_MOTION":14.3,"ECCENTRICITY":0.0002,"INCLINATION":98.4,"RA_OF_ASC_NODE":200.0,"ARG_OF_PERICENTER":90.0,"MEAN_ANOMALY":270.0,"EPHEMERIS_TYPE":0,"CLASSIFICATION_TYPE":"U","NORAD_CAT_ID":65053,"ELEMENT_SET_NO":999,"REV_AT_EPOCH":5000,"BSTAR":0.00005,"MEAN_MOTION_DOT":0.00001,"MEAN_MOTION_DDOT":0}
]"#;

    const SINGLE_JSON: &str = r#"[{"OBJECT_NAME":"COSMOS 2251 DEB","OBJECT_ID":"1993-036AAB","EPOCH":"2026-08-27T12:00:00.000000","MEAN_MOTION":14.1,"ECCENTRICITY":0.01,"INCLINATION":74.0,"RA_OF_ASC_NODE":10.0,"ARG_OF_PERICENTER":20.0,"MEAN_ANOMALY":30.0,"EPHEMERIS_TYPE":0,"CLASSIFICATION_TYPE":"U","NORAD_CAT_ID":34427,"ELEMENT_SET_NO":999,"REV_AT_EPOCH":1,"BSTAR":0.0001,"MEAN_MOTION_DOT":0.0001,"MEAN_MOTION_DDOT":0}]"#;

    /// Mock the `active` group and a per-object endpoint; returns (active, single).
    fn mock_active_and_single<'a>(
        server: &'a MockServer,
        param: &str,
        value: &str,
    ) -> (httpmock::Mock<'a>, httpmock::Mock<'a>) {
        let active = server.mock(|when, then| {
            when.method(GET)
                .path("/NORAD/elements/gp.php")
                .query_param("GROUP", "active")
                .query_param("FORMAT", "JSON");
            then.status(200).body(ACTIVE_JSON);
        });
        let single = server.mock(|when, then| {
            when.method(GET)
                .path("/NORAD/elements/gp.php")
                .query_param(param, value)
                .query_param("FORMAT", "JSON");
            then.status(200).body(SINGLE_JSON);
        });
        (active, single)
    }

    #[test]
    #[serial]
    fn test_catnr_resolves_from_active_without_per_object_request() {
        let _cache = CacheRedirect::new();
        let server = MockServer::start();
        let (active, single) = mock_active_and_single(&server, "CATNR", "25544");
        let client = CelestrakClient::with_base_url(&server.base_url());

        let records = client.get_gp_by_catnr(25544).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].norad_cat_id, Some(25544));
        active.assert_calls(1);
        single.assert_calls(0);

        // A second client shares the on-disk active cache: no request at all.
        let other = CelestrakClient::with_base_url(&server.base_url());
        let records = other.get_gp_by_catnr(65053).unwrap();
        assert_eq!(records[0].object_name.as_deref(), Some("NISAR"));
        active.assert_calls(1);
        single.assert_calls(0);
    }

    #[test]
    #[serial]
    fn test_name_resolves_case_insensitive_substring_from_active() {
        let _cache = CacheRedirect::new();
        let server = MockServer::start();
        let (active, single) = mock_active_and_single(&server, "NAME", "iss");
        let client = CelestrakClient::with_base_url(&server.base_url());

        let records = client.get_gp_by_name("iss").unwrap();
        let names: Vec<_> = records
            .iter()
            .filter_map(|r| r.object_name.as_deref())
            .collect();
        assert_eq!(names, vec!["ISS (ZARYA)", "ISS (NAUKA)"]);
        active.assert_calls(1);
        single.assert_calls(0);
    }

    #[test]
    #[serial]
    fn test_intdes_resolves_from_active() {
        let _cache = CacheRedirect::new();
        let server = MockServer::start();
        let (active, single) = mock_active_and_single(&server, "INTDES", "2025-158a");
        let client = CelestrakClient::with_base_url(&server.base_url());

        let records = client.get_gp_by_intdes("2025-158a").unwrap();
        assert_eq!(records[0].norad_cat_id, Some(65053));
        active.assert_calls(1);
        single.assert_calls(0);
    }

    #[test]
    #[serial]
    fn test_catnr_absent_from_active_falls_through_to_per_object_request() {
        let _cache = CacheRedirect::new();
        let server = MockServer::start();
        let (active, single) = mock_active_and_single(&server, "CATNR", "34427");
        let client = CelestrakClient::with_base_url(&server.base_url());

        let records = client.get_gp_by_catnr(34427).unwrap();
        assert_eq!(records[0].object_name.as_deref(), Some("COSMOS 2251 DEB"));
        active.assert_calls(1);
        single.assert_calls(1);
    }

    #[test]
    #[serial]
    fn test_name_without_active_match_goes_to_server() {
        let _cache = CacheRedirect::new();
        let server = MockServer::start();
        let (active, single) = mock_active_and_single(&server, "NAME", "COSMOS");
        let client = CelestrakClient::with_base_url(&server.base_url());

        let records = client.get_gp_by_name("COSMOS").unwrap();
        assert_eq!(records.len(), 1);
        active.assert_calls(1);
        single.assert_calls(1);
    }

    #[test]
    #[serial]
    fn test_exact_per_object_cache_takes_precedence_over_active() {
        let _cache = CacheRedirect::new();
        let server = MockServer::start();
        let (active, single) = mock_active_and_single(&server, "CATNR", "25544");
        let client = CelestrakClient::with_base_url(&server.base_url());
        let url = format!(
            "{}/NORAD/elements/gp.php?CATNR=25544&FORMAT=JSON",
            server.base_url()
        );
        seed_celestrak_cache(&client, &url, SINGLE_JSON, Duration::from_secs(60));

        let records = client.get_gp_by_catnr(25544).unwrap();
        assert_eq!(records[0].object_name.as_deref(), Some("COSMOS 2251 DEB"));
        active.assert_calls(0);
        single.assert_calls(0);
    }

    #[test]
    #[serial]
    fn test_active_unavailable_online_falls_back_to_per_object_request() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let active = server.mock(|when, then| {
            when.method(GET)
                .path("/NORAD/elements/gp.php")
                .query_param("GROUP", "active");
            then.status(404);
        });
        let single = server.mock(|when, then| {
            when.method(GET)
                .path("/NORAD/elements/gp.php")
                .query_param("CATNR", "34427");
            then.status(200).body(SINGLE_JSON);
        });
        let client = CelestrakClient::with_base_url(&server.base_url());

        let records = client.get_gp_by_catnr(34427).unwrap();
        assert_eq!(records[0].norad_cat_id, Some(34427));
        active.assert_calls(1);
        single.assert_calls(1);
    }

    #[test]
    #[serial]
    fn test_active_resolution_does_not_write_per_object_cache() {
        let _cache = CacheRedirect::new();
        let server = MockServer::start();
        let (_active, _single) = mock_active_and_single(&server, "CATNR", "25544");
        let client = CelestrakClient::with_base_url(&server.base_url());
        client.get_gp_by_catnr(25544).unwrap();

        let url = format!(
            "{}/NORAD/elements/gp.php?CATNR=25544&FORMAT=JSON",
            server.base_url()
        );
        let dir = crate::utils::get_celestrak_cache_dir().unwrap();
        let per_object = std::path::Path::new(&dir).join(client.cache_key_for_url(&url));
        assert!(!per_object.exists());
        let active_url = format!(
            "{}/NORAD/elements/gp.php?GROUP=active&FORMAT=JSON",
            server.base_url()
        );
        let active_file = std::path::Path::new(&dir).join(client.cache_key_for_url(&active_url));
        assert!(active_file.exists());
    }

    #[test]
    #[serial]
    fn test_group_query_never_touches_active() {
        let _cache = CacheRedirect::new();
        let server = MockServer::start();
        let active = server.mock(|when, then| {
            when.method(GET)
                .path("/NORAD/elements/gp.php")
                .query_param("GROUP", "active");
            then.status(200).body(ACTIVE_JSON);
        });
        let stations = server.mock(|when, then| {
            when.method(GET)
                .path("/NORAD/elements/gp.php")
                .query_param("GROUP", "stations");
            then.status(200).body(SINGLE_JSON);
        });
        let client = CelestrakClient::with_base_url(&server.base_url());

        client.get_gp_by_group("stations").unwrap();
        active.assert_calls(0);
        stations.assert_calls(1);
    }

    #[test]
    #[serial]
    fn test_query_raw_with_catnr_never_touches_active() {
        let _cache = CacheRedirect::new();
        let server = MockServer::start();
        let (active, single) = mock_active_and_single(&server, "CATNR", "34427");
        let client = CelestrakClient::with_base_url(&server.base_url());

        let body = client
            .query_raw(
                &CelestrakQuery::gp()
                    .catnr(34427)
                    .format(CelestrakOutputFormat::Json),
            )
            .unwrap();
        assert_eq!(body, SINGLE_JSON);
        active.assert_calls(0);
        single.assert_calls(1);
    }

    #[test]
    #[serial]
    fn test_client_side_filters_apply_after_active_resolution() {
        let _cache = CacheRedirect::new();
        let server = MockServer::start();
        let (active, single) = mock_active_and_single(&server, "NAME", "ISS");
        let client = CelestrakClient::with_base_url(&server.base_url());

        let query = CelestrakQuery::gp().name_search("ISS").limit(1);
        let records = client.query_gp(&query).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].object_name.as_deref(), Some("ISS (ZARYA)"));
        active.assert_calls(1);
        single.assert_calls(0);
    }

    #[test]
    #[serial]
    fn test_offline_serves_stale_active_and_errors_on_full_miss() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("offline"));
        let server = MockServer::start();
        let (active, single) = mock_active_and_single(&server, "CATNR", "25544");
        let client = CelestrakClient::with_base_url(&server.base_url());
        let active_url = format!(
            "{}/NORAD/elements/gp.php?GROUP=active&FORMAT=JSON",
            server.base_url()
        );
        seed_celestrak_cache(
            &client,
            &active_url,
            ACTIVE_JSON,
            Duration::from_secs(30 * 86400),
        );

        let records = client.get_gp_by_catnr(25544).unwrap();
        assert_eq!(records[0].norad_cat_id, Some(25544));
        active.assert_calls(0);
        single.assert_calls(0);

        let err = client.get_gp_by_catnr(34427).unwrap_err().to_string();
        assert!(
            err.starts_with("BRAHE_NETWORK_MODE is offline; Celestrak request "),
            "{err}"
        );
        active.assert_calls(0);
        single.assert_calls(0);
    }

    #[test]
    #[serial]
    fn test_sgp_propagator_by_catnr_resolves_from_active() {
        setup_global_test_eop();
        let _cache = CacheRedirect::new();
        let server = MockServer::start();
        let (active, single) = mock_active_and_single(&server, "CATNR", "25544");
        let client = CelestrakClient::with_base_url(&server.base_url());

        let propagator = client.get_sgp_propagator_by_catnr(25544, 60.0).unwrap();
        assert_eq!(propagator.norad_id, 25544);
        active.assert_calls(1);
        single.assert_calls(0);
    }
}
