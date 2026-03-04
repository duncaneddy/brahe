/*!
 * HTTP client for the CelestrakClient API.
 *
 * Provides access to CelestrakClient endpoints with file-based caching
 * and typed query execution. No authentication is required.
 */

use std::fs;
use std::path::Path;
use std::time::SystemTime;

use crate::celestrak::filter::{apply_filters, apply_limit, apply_order_by};
use crate::celestrak::query::CelestrakQuery;
use crate::celestrak::responses::CelestrakSATCATRecord;
use crate::celestrak::types::{CelestrakOutputFormat, SupGPSource};
use crate::propagators::SGPPropagator;
use crate::types::GPRecord;
use crate::utils::{BraheError, atomic_write, get_celestrak_cache_dir};

/// Default base URL for the CelestrakClient API.
const DEFAULT_BASE_URL: &str = "https://celestrak.org";

/// Default maximum cache age in seconds (6 hours).
const DEFAULT_MAX_CACHE_AGE: f64 = 21600.0;

/// CelestrakClient API client with caching.
///
/// Provides typed query execution for GP, supplemental GP, and SATCAT
/// data from CelestrakClient. Responses are cached locally to reduce
/// server load and improve performance.
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
    agent: ureq::Agent,
}

impl CelestrakClient {
    /// Create a new CelestrakClient client with default settings.
    ///
    /// Uses the default base URL (`https://celestrak.org`) and
    /// a 6-hour cache TTL.
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
            agent: ureq::Agent::new_with_defaults(),
        }
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
    /// Works for both GP and SupGP query types.
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

        let body = self.query_raw(&json_query)?;
        let mut records: Vec<GPRecord> = serde_json::from_str(&body).map_err(|e| {
            BraheError::ParseError(format!(
                "Failed to parse CelestrakClient GP response: {}",
                e
            ))
        })?;

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
    /// * `Ok(Vec<GPRecord>)` - Matching GP records
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
    /// * `Ok(Vec<GPRecord>)` - Matching GP records
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
    /// * `Ok(Vec<GPRecord>)` - Matching GP records
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

    /// Read cached data if it exists and is fresh.
    fn read_cache(&self, cache_key: &str) -> Result<Option<String>, BraheError> {
        let cache_dir = get_celestrak_cache_dir()?;
        let cache_path = Path::new(&cache_dir).join(cache_key);

        if !cache_path.exists() {
            return Ok(None);
        }

        if self.is_cache_stale(&cache_path)? {
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
    fn execute_get(&self, url: &str) -> Result<String, BraheError> {
        let mut response =
            self.agent.get(url).call().map_err(|e| {
                BraheError::IoError(format!("CelestrakClient request failed: {}", e))
            })?;

        response.body_mut().read_to_string().map_err(|e| {
            BraheError::IoError(format!("Failed to read CelestrakClient response: {}", e))
        })
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
    use httpmock::prelude::*;

    #[test]
    fn test_client_creation() {
        let client = CelestrakClient::new();
        assert_eq!(client.base_url, DEFAULT_BASE_URL);
        assert_eq!(client.cache_max_age, DEFAULT_MAX_CACHE_AGE);
    }

    #[test]
    fn test_client_with_cache_age() {
        let client = CelestrakClient::with_cache_age(3600.0);
        assert_eq!(client.base_url, DEFAULT_BASE_URL);
        assert_eq!(client.cache_max_age, 3600.0);
    }

    #[test]
    fn test_client_with_base_url() {
        let client = CelestrakClient::with_base_url("https://test.celestrak.org/");
        assert_eq!(client.base_url, "https://test.celestrak.org");
    }

    #[test]
    fn test_client_with_base_url_no_trailing_slash() {
        let client = CelestrakClient::with_base_url("https://test.celestrak.org");
        assert_eq!(client.base_url, "https://test.celestrak.org");
    }

    #[test]
    fn test_client_with_base_url_and_cache_age() {
        let client =
            CelestrakClient::with_base_url_and_cache_age("https://test.celestrak.org", 1800.0);
        assert_eq!(client.base_url, "https://test.celestrak.org");
        assert_eq!(client.cache_max_age, 1800.0);
    }

    #[test]
    fn test_client_default() {
        let client = CelestrakClient::default();
        assert_eq!(client.base_url, DEFAULT_BASE_URL);
    }

    #[test]
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
    fn test_build_full_url_gp_empty() {
        let client = CelestrakClient::new();
        let query = CelestrakQuery::gp();
        let url = client.build_full_url(&query);
        assert_eq!(url, "https://celestrak.org/NORAD/elements/gp.php");
    }

    #[test]
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
    fn test_build_full_url_satcat() {
        let client = CelestrakClient::new();
        let query = CelestrakQuery::satcat().active(true);
        let url = client.build_full_url(&query);
        assert_eq!(url, "https://celestrak.org/satcat/records.php?ACTIVE=Y");
    }

    #[test]
    fn test_query_raw_gp() {
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
    fn test_query_gp_typed() {
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
    fn test_query_gp_with_client_side_filter() {
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
    fn test_query_gp_with_order_and_limit() {
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
    fn test_query_satcat_typed() {
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
    fn test_query_raw_tle_format() {
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
    fn test_http_error_404() {
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
    fn test_invalid_json_response() {
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
    fn test_empty_json_response() {
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
    fn test_download_to_file() {
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
    fn test_cache_key_generation() {
        let client = CelestrakClient::new();
        let key = client.cache_key_for_url("https://celestrak.org/gp.php?GROUP=stations");
        assert!(key.contains("celestrak.org"));
        assert!(key.contains("GROUP"));
        assert!(!key.contains("?"));
        assert!(!key.contains("/"));
    }

    // -- Convenience method tests --

    #[test]
    fn test_get_gp_by_catnr() {
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
    fn test_get_gp_by_group() {
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
    fn test_get_gp_by_name() {
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
    fn test_get_gp_by_intdes() {
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
    fn test_get_sup_gp() {
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
    fn test_get_satcat_by_catnr() {
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
    fn test_get_sgp_propagator_by_catnr_empty_results() {
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

    // -- CI-gated live integration tests --

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    fn test_integration_gp_by_group() {
        let client = CelestrakClient::with_cache_age(0.0);
        let records = client.get_gp_by_group("stations").expect("GP query failed");
        assert!(!records.is_empty(), "Expected at least one GP record");
    }

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    fn test_integration_gp_by_catnr() {
        let client = CelestrakClient::with_cache_age(0.0);
        let records = client.get_gp_by_catnr(25544).expect("GP query failed");
        assert!(!records.is_empty(), "Expected ISS GP record");
        assert_eq!(records[0].norad_cat_id, Some(25544));
    }

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    fn test_integration_gp_by_name() {
        let client = CelestrakClient::with_cache_age(0.0);
        let records = client.get_gp_by_name("ISS").expect("GP query failed");
        assert!(
            !records.is_empty(),
            "Expected at least one record matching ISS"
        );
    }

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    fn test_integration_satcat() {
        let client = CelestrakClient::with_cache_age(0.0);
        let records = client
            .get_satcat_by_catnr(25544)
            .expect("SATCAT query failed");
        assert!(!records.is_empty(), "Expected ISS SATCAT record");
        assert_eq!(records[0].norad_cat_id, Some(25544));
    }

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    fn test_integration_get_sgp_propagator_by_catnr() {
        let client = CelestrakClient::with_cache_age(0.0);
        let propagator = client
            .get_sgp_propagator_by_catnr(25544, 60.0)
            .expect("SGP propagator creation failed");
        assert_eq!(propagator.norad_id, 25544);
    }
}
