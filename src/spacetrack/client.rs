/*!
 * SpaceTrack HTTP client with authentication and query execution.
 *
 * Handles session management via cookie-based authentication against
 * the Space-Track.org API. Supports automatic authentication on first
 * query and re-authentication on session expiry (401 responses).
 */

use std::sync::Mutex;
use std::time::Duration;

use crate::spacetrack::query::SpaceTrackQuery;
use crate::spacetrack::rate_limiter::{RateLimitConfig, RateLimiter};
use crate::spacetrack::responses::{
    FileShareFileRecord, FolderRecord, SATCATRecord, SPEphemerisFileRecord,
};
use crate::types::GPRecord;
use crate::utils::BraheError;

/// Default base URL for the Space-Track.org API.
const DEFAULT_BASE_URL: &str = "https://www.space-track.org";

/// SpaceTrack API client with session-based authentication.
///
/// The client lazily authenticates on the first query and re-authenticates
/// automatically if the session expires (HTTP 401 response).
///
/// # Examples
///
/// ```no_run
/// use brahe::spacetrack::*;
///
/// let client = SpaceTrackClient::new("user@example.com", "password");
///
/// let query = SpaceTrackQuery::new(RequestClass::GP)
///     .filter("NORAD_CAT_ID", "25544")
///     .order_by("EPOCH", SortOrder::Desc)
///     .limit(1);
///
/// let json = client.query_json(&query).unwrap();
/// println!("Records: {}", json.len());
/// ```
pub struct SpaceTrackClient {
    identity: String,
    password: String,
    base_url: String,
    agent: Mutex<ureq::Agent>,
    authenticated: Mutex<bool>,
    rate_limiter: Mutex<RateLimiter>,
}

impl SpaceTrackClient {
    /// Create a new SpaceTrack client.
    ///
    /// Authentication is deferred until the first query is made.
    ///
    /// # Arguments
    ///
    /// * `identity` - Space-Track.org login email
    /// * `password` - Space-Track.org password
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::spacetrack::SpaceTrackClient;
    ///
    /// let client = SpaceTrackClient::new("user@example.com", "password");
    /// ```
    pub fn new(identity: &str, password: &str) -> Self {
        SpaceTrackClient {
            identity: identity.to_string(),
            password: password.to_string(),
            base_url: DEFAULT_BASE_URL.to_string(),
            agent: Mutex::new(ureq::Agent::new_with_defaults()),
            authenticated: Mutex::new(false),
            rate_limiter: Mutex::new(RateLimiter::new(RateLimitConfig::default())),
        }
    }

    /// Create a new SpaceTrack client with a custom base URL.
    ///
    /// Useful for testing against a mock server or the Space-Track test server.
    ///
    /// # Arguments
    ///
    /// * `identity` - Space-Track.org login email
    /// * `password` - Space-Track.org password
    /// * `base_url` - Custom base URL (e.g., `"https://for-testing-only.space-track.org"`)
    pub fn with_base_url(identity: &str, password: &str, base_url: &str) -> Self {
        SpaceTrackClient {
            identity: identity.to_string(),
            password: password.to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
            agent: Mutex::new(ureq::Agent::new_with_defaults()),
            authenticated: Mutex::new(false),
            rate_limiter: Mutex::new(RateLimiter::new(RateLimitConfig::default())),
        }
    }

    /// Create a new SpaceTrack client with a custom rate limit configuration.
    ///
    /// # Arguments
    ///
    /// * `identity` - Space-Track.org login email
    /// * `password` - Space-Track.org password
    /// * `config` - Rate limit configuration
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::spacetrack::{SpaceTrackClient, RateLimitConfig};
    ///
    /// let config = RateLimitConfig { max_per_minute: 10, max_per_hour: 100 };
    /// let client = SpaceTrackClient::with_rate_limit("user@example.com", "password", config);
    /// ```
    pub fn with_rate_limit(identity: &str, password: &str, config: RateLimitConfig) -> Self {
        SpaceTrackClient {
            identity: identity.to_string(),
            password: password.to_string(),
            base_url: DEFAULT_BASE_URL.to_string(),
            agent: Mutex::new(ureq::Agent::new_with_defaults()),
            authenticated: Mutex::new(false),
            rate_limiter: Mutex::new(RateLimiter::new(config)),
        }
    }

    /// Create a new SpaceTrack client with a custom base URL and rate limit configuration.
    ///
    /// # Arguments
    ///
    /// * `identity` - Space-Track.org login email
    /// * `password` - Space-Track.org password
    /// * `base_url` - Custom base URL
    /// * `config` - Rate limit configuration
    pub fn with_base_url_and_rate_limit(
        identity: &str,
        password: &str,
        base_url: &str,
        config: RateLimitConfig,
    ) -> Self {
        SpaceTrackClient {
            identity: identity.to_string(),
            password: password.to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
            agent: Mutex::new(ureq::Agent::new_with_defaults()),
            authenticated: Mutex::new(false),
            rate_limiter: Mutex::new(RateLimiter::new(config)),
        }
    }

    /// Wait for rate limit clearance before making an HTTP request.
    ///
    /// Acquires the rate limiter lock, computes the required wait duration,
    /// releases the lock, then sleeps for the computed duration.
    fn wait_for_rate_limit(&self) -> Result<(), BraheError> {
        let wait = {
            let mut limiter = self.rate_limiter.lock().map_err(|e| {
                BraheError::Error(format!("Failed to acquire lock on rate limiter: {}", e))
            })?;
            limiter.acquire()
        };

        if wait > Duration::ZERO {
            std::thread::sleep(wait);
        }

        Ok(())
    }

    /// Explicitly authenticate with Space-Track.org.
    ///
    /// This is called automatically on the first query, but can be called
    /// explicitly to verify credentials early.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if authentication succeeds
    /// * `Err(BraheError)` if authentication fails
    pub fn authenticate(&self) -> Result<(), BraheError> {
        self.wait_for_rate_limit()?;

        let url = format!("{}/ajaxauth/login", self.base_url);

        let form_data = format!(
            "identity={}&password={}",
            urlencoded(&self.identity),
            urlencoded(&self.password)
        );

        let agent = self.agent.lock().map_err(|e| {
            BraheError::Error(format!("Failed to acquire lock on HTTP agent: {}", e))
        })?;

        let mut response = agent
            .post(&url)
            .content_type("application/x-www-form-urlencoded")
            .send(form_data.as_str())
            .map_err(|e| {
                BraheError::IoError(format!("SpaceTrack authentication request failed: {}", e))
            })?;

        let body = response.body_mut().read_to_string().map_err(|e| {
            BraheError::IoError(format!(
                "Failed to read SpaceTrack authentication response: {}",
                e
            ))
        })?;

        // Check for login failure - Space-Track returns JSON with Login field
        if body.contains("\"Login\":\"Failed\"") || body.contains("\"Login\": \"Failed\"") {
            return Err(BraheError::IoError(
                "SpaceTrack authentication failed: invalid credentials".to_string(),
            ));
        }

        let mut auth = self.authenticated.lock().map_err(|e| {
            BraheError::Error(format!("Failed to acquire lock on auth state: {}", e))
        })?;
        *auth = true;

        Ok(())
    }

    /// Execute a query and return the raw response body as a string.
    ///
    /// Auto-authenticates on first query and re-authenticates on 401.
    ///
    /// # Arguments
    ///
    /// * `query` - The query to execute
    ///
    /// # Returns
    ///
    /// * `Ok(String)` - Raw response body
    /// * `Err(BraheError)` - On network, auth, or HTTP errors
    pub fn query_raw(&self, query: &SpaceTrackQuery) -> Result<String, BraheError> {
        let url = format!("{}{}", self.base_url, query.build());
        self.authenticated_get_string(&url)
    }

    /// Execute a query and return the response as parsed JSON values.
    ///
    /// The query format must be JSON (the default). Returns an error if
    /// a non-JSON format is specified.
    ///
    /// # Arguments
    ///
    /// * `query` - The query to execute
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<serde_json::Value>)` - Parsed JSON array
    /// * `Err(BraheError)` - On network, auth, parse, or format errors
    pub fn query_json(
        &self,
        query: &SpaceTrackQuery,
    ) -> Result<Vec<serde_json::Value>, BraheError> {
        if !query.output_format().is_json() {
            return Err(BraheError::Error(
                "query_json requires JSON output format".to_string(),
            ));
        }

        let body = self.query_raw(query)?;
        serde_json::from_str(&body).map_err(|e| {
            BraheError::ParseError(format!("Failed to parse SpaceTrack JSON response: {}", e))
        })
    }

    /// Execute a GP query and return typed GP records.
    ///
    /// The query must use JSON format (the default) and should query the
    /// GP or GP_HISTORY request class for meaningful results.
    ///
    /// # Arguments
    ///
    /// * `query` - The query to execute
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<GPRecord>)` - Typed GP records
    /// * `Err(BraheError)` - On network, auth, parse, or format errors
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::spacetrack::*;
    ///
    /// let client = SpaceTrackClient::new("user@example.com", "password");
    /// let query = SpaceTrackQuery::new(RequestClass::GP)
    ///     .filter("NORAD_CAT_ID", "25544")
    ///     .limit(1);
    ///
    /// let records = client.query_gp(&query).unwrap();
    /// println!("Object: {:?}", records[0].object_name);
    /// ```
    pub fn query_gp(&self, query: &SpaceTrackQuery) -> Result<Vec<GPRecord>, BraheError> {
        if !query.output_format().is_json() {
            return Err(BraheError::Error(
                "query_gp requires JSON output format".to_string(),
            ));
        }

        let body = self.query_raw(query)?;
        serde_json::from_str(&body).map_err(|e| {
            BraheError::ParseError(format!("Failed to parse SpaceTrack GP response: {}", e))
        })
    }

    /// Execute a SATCAT query and return typed SATCAT records.
    ///
    /// The query must use JSON format (the default) and should query the
    /// SATCAT request class for meaningful results.
    ///
    /// # Arguments
    ///
    /// * `query` - The query to execute
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<SATCATRecord>)` - Typed SATCAT records
    /// * `Err(BraheError)` - On network, auth, parse, or format errors
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::spacetrack::*;
    ///
    /// let client = SpaceTrackClient::new("user@example.com", "password");
    /// let query = SpaceTrackQuery::new(RequestClass::SATCAT)
    ///     .filter("NORAD_CAT_ID", "25544")
    ///     .limit(1);
    ///
    /// let records = client.query_satcat(&query).unwrap();
    /// println!("Name: {:?}", records[0].satname);
    /// ```
    pub fn query_satcat(&self, query: &SpaceTrackQuery) -> Result<Vec<SATCATRecord>, BraheError> {
        if !query.output_format().is_json() {
            return Err(BraheError::Error(
                "query_satcat requires JSON output format".to_string(),
            ));
        }

        let body = self.query_raw(query)?;
        serde_json::from_str(&body).map_err(|e| {
            BraheError::ParseError(format!("Failed to parse SpaceTrack SATCAT response: {}", e))
        })
    }

    // ========================================
    // FileShare operations
    // ========================================

    /// Upload a file to the Space-Track file share.
    ///
    /// # Arguments
    ///
    /// * `folder_id` - Target folder identifier
    /// * `file_name` - Name for the uploaded file
    /// * `file_data` - File content as bytes
    ///
    /// # Returns
    ///
    /// * `Ok(String)` - Server response (typically JSON confirmation)
    /// * `Err(BraheError)` - On network, auth, or upload errors
    pub fn fileshare_upload(
        &self,
        folder_id: &str,
        file_name: &str,
        file_data: &[u8],
    ) -> Result<String, BraheError> {
        let url = format!("{}/fileshare/upload/folder_id/{}", self.base_url, folder_id);
        self.authenticated_post_multipart(&url, file_name, file_data)
    }

    /// Download a file from the Space-Track file share.
    ///
    /// # Arguments
    ///
    /// * `file_id` - File identifier to download
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<u8>)` - File content as bytes
    /// * `Err(BraheError)` - On network, auth, or download errors
    pub fn fileshare_download(&self, file_id: &str) -> Result<Vec<u8>, BraheError> {
        let url = format!("{}/fileshare/download/file_id/{}", self.base_url, file_id);
        self.authenticated_get_binary(&url)
    }

    /// Download all files in a folder from the Space-Track file share.
    ///
    /// # Arguments
    ///
    /// * `folder_id` - Folder identifier to download
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<u8>)` - Folder content as bytes (typically a zip archive)
    /// * `Err(BraheError)` - On network, auth, or download errors
    pub fn fileshare_download_folder(&self, folder_id: &str) -> Result<Vec<u8>, BraheError> {
        let url = format!(
            "{}/fileshare/download/folder_id/{}",
            self.base_url, folder_id
        );
        self.authenticated_get_binary(&url)
    }

    /// List files in the Space-Track file share.
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<FileShareFileRecord>)` - File metadata records
    /// * `Err(BraheError)` - On network, auth, or parse errors
    pub fn fileshare_list_files(&self) -> Result<Vec<FileShareFileRecord>, BraheError> {
        let url = format!("{}/fileshare/query/class/file", self.base_url);
        let body = self.authenticated_get_string(&url)?;
        serde_json::from_str(&body).map_err(|e| {
            BraheError::ParseError(format!(
                "Failed to parse SpaceTrack fileshare file listing: {}",
                e
            ))
        })
    }

    /// List folders in the Space-Track file share.
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<FolderRecord>)` - Folder metadata records
    /// * `Err(BraheError)` - On network, auth, or parse errors
    pub fn fileshare_list_folders(&self) -> Result<Vec<FolderRecord>, BraheError> {
        let url = format!("{}/fileshare/query/class/folder", self.base_url);
        let body = self.authenticated_get_string(&url)?;
        serde_json::from_str(&body).map_err(|e| {
            BraheError::ParseError(format!(
                "Failed to parse SpaceTrack fileshare folder listing: {}",
                e
            ))
        })
    }

    /// Delete a file from the Space-Track file share.
    ///
    /// # Arguments
    ///
    /// * `file_id` - File identifier to delete
    ///
    /// # Returns
    ///
    /// * `Ok(String)` - Server response
    /// * `Err(BraheError)` - On network, auth, or deletion errors
    pub fn fileshare_delete(&self, file_id: &str) -> Result<String, BraheError> {
        let url = format!(
            "{}/fileshare/query/class/delete/file_id/{}",
            self.base_url, file_id
        );
        self.authenticated_get_string(&url)
    }

    // ========================================
    // SP Ephemeris operations
    // ========================================

    /// Download an SP ephemeris file from Space-Track.
    ///
    /// # Arguments
    ///
    /// * `file_id` - SP ephemeris file identifier
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<u8>)` - Ephemeris file content as bytes
    /// * `Err(BraheError)` - On network, auth, or download errors
    pub fn spephemeris_download(&self, file_id: &str) -> Result<Vec<u8>, BraheError> {
        let url = format!("{}/spephemeris/download/file_id/{}", self.base_url, file_id);
        self.authenticated_get_binary(&url)
    }

    /// List available SP ephemeris files.
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<SPEphemerisFileRecord>)` - Ephemeris file metadata records
    /// * `Err(BraheError)` - On network, auth, or parse errors
    pub fn spephemeris_list_files(&self) -> Result<Vec<SPEphemerisFileRecord>, BraheError> {
        let url = format!("{}/spephemeris/query/class/file", self.base_url);
        let body = self.authenticated_get_string(&url)?;
        serde_json::from_str(&body).map_err(|e| {
            BraheError::ParseError(format!(
                "Failed to parse SpaceTrack SP ephemeris file listing: {}",
                e
            ))
        })
    }

    /// List SP ephemeris file history.
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<serde_json::Value>)` - File history records as generic JSON
    /// * `Err(BraheError)` - On network, auth, or parse errors
    pub fn spephemeris_file_history(&self) -> Result<Vec<serde_json::Value>, BraheError> {
        let url = format!("{}/spephemeris/query/class/file_history", self.base_url);
        let body = self.authenticated_get_string(&url)?;
        serde_json::from_str(&body).map_err(|e| {
            BraheError::ParseError(format!(
                "Failed to parse SpaceTrack SP ephemeris file history: {}",
                e
            ))
        })
    }

    // ========================================
    // Public Files operations (no auth required)
    // ========================================

    /// Download a public file from Space-Track.
    ///
    /// This operation does not require authentication.
    ///
    /// # Arguments
    ///
    /// * `file_name` - Name of the public file to download
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<u8>)` - File content as bytes
    /// * `Err(BraheError)` - On network or download errors
    pub fn publicfiles_download(&self, file_name: &str) -> Result<Vec<u8>, BraheError> {
        let url = format!(
            "{}/publicfiles/query/class/download?name={}",
            self.base_url,
            urlencoded(file_name)
        );
        self.execute_get_binary(&url)
    }

    /// List public file directories on Space-Track.
    ///
    /// This operation does not require authentication.
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<serde_json::Value>)` - Directory listing as generic JSON
    /// * `Err(BraheError)` - On network or parse errors
    pub fn publicfiles_list_dirs(&self) -> Result<Vec<serde_json::Value>, BraheError> {
        let url = format!("{}/publicfiles/query/class/dirs", self.base_url);
        let body = self.execute_get(&url)?;
        serde_json::from_str(&body).map_err(|e| {
            BraheError::ParseError(format!(
                "Failed to parse SpaceTrack public files directory listing: {}",
                e
            ))
        })
    }

    /// Execute an authenticated GET request returning a string response.
    ///
    /// Auto-authenticates on first call and re-authenticates on 401.
    fn authenticated_get_string(&self, url: &str) -> Result<String, BraheError> {
        self.ensure_authenticated()?;

        match self.execute_get(url) {
            Ok(body) => Ok(body),
            Err(e) => {
                let err_str = e.to_string();
                if err_str.contains("401") || err_str.contains("Unauthorized") {
                    self.authenticate()?;
                    self.execute_get(url)
                } else {
                    Err(e)
                }
            }
        }
    }

    /// Execute an authenticated GET request returning a binary response.
    ///
    /// Auto-authenticates on first call and re-authenticates on 401.
    fn authenticated_get_binary(&self, url: &str) -> Result<Vec<u8>, BraheError> {
        self.ensure_authenticated()?;

        match self.execute_get_binary(url) {
            Ok(body) => Ok(body),
            Err(e) => {
                let err_str = e.to_string();
                if err_str.contains("401") || err_str.contains("Unauthorized") {
                    self.authenticate()?;
                    self.execute_get_binary(url)
                } else {
                    Err(e)
                }
            }
        }
    }

    /// Execute an authenticated POST request with multipart form data returning a string response.
    ///
    /// Auto-authenticates on first call and re-authenticates on 401.
    fn authenticated_post_multipart(
        &self,
        url: &str,
        file_name: &str,
        file_data: &[u8],
    ) -> Result<String, BraheError> {
        self.ensure_authenticated()?;

        match self.execute_post_multipart(url, file_name, file_data) {
            Ok(body) => Ok(body),
            Err(e) => {
                let err_str = e.to_string();
                if err_str.contains("401") || err_str.contains("Unauthorized") {
                    self.authenticate()?;
                    self.execute_post_multipart(url, file_name, file_data)
                } else {
                    Err(e)
                }
            }
        }
    }

    /// Ensure the client is authenticated, authenticating if necessary.
    fn ensure_authenticated(&self) -> Result<(), BraheError> {
        let auth = self.authenticated.lock().map_err(|e| {
            BraheError::Error(format!("Failed to acquire lock on auth state: {}", e))
        })?;

        if !*auth {
            drop(auth); // Release lock before authenticate() acquires it
            self.authenticate()?;
        }

        Ok(())
    }

    /// Execute an HTTP GET request and return the response body as a string.
    fn execute_get(&self, url: &str) -> Result<String, BraheError> {
        self.wait_for_rate_limit()?;

        let agent = self.agent.lock().map_err(|e| {
            BraheError::Error(format!("Failed to acquire lock on HTTP agent: {}", e))
        })?;

        let mut response = agent
            .get(url)
            .call()
            .map_err(|e| BraheError::IoError(format!("SpaceTrack query request failed: {}", e)))?;

        response
            .body_mut()
            .read_to_string()
            .map_err(|e| BraheError::IoError(format!("Failed to read SpaceTrack response: {}", e)))
    }

    /// Execute an HTTP GET request and return the response body as bytes.
    fn execute_get_binary(&self, url: &str) -> Result<Vec<u8>, BraheError> {
        self.wait_for_rate_limit()?;

        let agent = self.agent.lock().map_err(|e| {
            BraheError::Error(format!("Failed to acquire lock on HTTP agent: {}", e))
        })?;

        let mut response = agent
            .get(url)
            .call()
            .map_err(|e| BraheError::IoError(format!("SpaceTrack request failed: {}", e)))?;

        response.body_mut().read_to_vec().map_err(|e| {
            BraheError::IoError(format!("Failed to read SpaceTrack binary response: {}", e))
        })
    }

    /// Execute an HTTP POST request with multipart form data.
    fn execute_post_multipart(
        &self,
        url: &str,
        file_name: &str,
        file_data: &[u8],
    ) -> Result<String, BraheError> {
        self.wait_for_rate_limit()?;

        use ureq::unversioned::multipart::{Form, Part};

        let agent = self.agent.lock().map_err(|e| {
            BraheError::Error(format!("Failed to acquire lock on HTTP agent: {}", e))
        })?;

        let form = Form::new().part("file", Part::bytes(file_data).file_name(file_name));

        let mut response = agent
            .post(url)
            .send(form)
            .map_err(|e| BraheError::IoError(format!("SpaceTrack upload request failed: {}", e)))?;

        response.body_mut().read_to_string().map_err(|e| {
            BraheError::IoError(format!("Failed to read SpaceTrack upload response: {}", e))
        })
    }
}

/// Simple URL-encoding for form data fields.
///
/// Encodes special characters that need escaping in
/// application/x-www-form-urlencoded content.
fn urlencoded(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    for c in input.chars() {
        match c {
            ' ' => result.push_str("%20"),
            '!' => result.push_str("%21"),
            '#' => result.push_str("%23"),
            '$' => result.push_str("%24"),
            '%' => result.push_str("%25"),
            '&' => result.push_str("%26"),
            '\'' => result.push_str("%27"),
            '(' => result.push_str("%28"),
            ')' => result.push_str("%29"),
            '*' => result.push_str("%2A"),
            '+' => result.push_str("%2B"),
            ',' => result.push_str("%2C"),
            '/' => result.push_str("%2F"),
            ':' => result.push_str("%3A"),
            ';' => result.push_str("%3B"),
            '=' => result.push_str("%3D"),
            '?' => result.push_str("%3F"),
            '@' => result.push_str("%40"),
            '[' => result.push_str("%5B"),
            ']' => result.push_str("%5D"),
            _ => result.push(c),
        }
    }
    result
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::spacetrack::{OutputFormat, RequestClass, SortOrder};
    use httpmock::prelude::*;

    #[test]
    fn test_client_creation() {
        let client = SpaceTrackClient::new("user@example.com", "password123");
        assert_eq!(client.identity, "user@example.com");
        assert_eq!(client.password, "password123");
        assert_eq!(client.base_url, DEFAULT_BASE_URL);
    }

    #[test]
    fn test_client_with_base_url() {
        let client = SpaceTrackClient::with_base_url(
            "user@example.com",
            "password123",
            "https://test.space-track.org/",
        );
        assert_eq!(client.base_url, "https://test.space-track.org");
    }

    #[test]
    fn test_client_with_base_url_no_trailing_slash() {
        let client = SpaceTrackClient::with_base_url(
            "user@example.com",
            "password123",
            "https://test.space-track.org",
        );
        assert_eq!(client.base_url, "https://test.space-track.org");
    }

    #[test]
    fn test_urlencoded() {
        assert_eq!(urlencoded("hello"), "hello");
        assert_eq!(urlencoded("hello world"), "hello%20world");
        assert_eq!(urlencoded("user@example.com"), "user%40example.com");
        assert_eq!(urlencoded("pass&word"), "pass%26word");
        assert_eq!(urlencoded("a=b"), "a%3Db");
    }

    #[test]
    fn test_successful_authentication() {
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(POST)
                .path("/ajaxauth/login")
                .header("content-type", "application/x-www-form-urlencoded");
            then.status(200).body("");
        });

        let client =
            SpaceTrackClient::with_base_url("user@example.com", "password", &server.base_url());

        let result = client.authenticate();
        assert!(result.is_ok());
    }

    #[test]
    fn test_failed_authentication() {
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(POST).path("/ajaxauth/login");
            then.status(200).body(
                r#"{"Login":"Failed","help":"https://www.space-track.org/documentation#/api"}"#,
            );
        });

        let client =
            SpaceTrackClient::with_base_url("bad@example.com", "wrong", &server.base_url());

        let result = client.authenticate();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("invalid credentials")
        );
    }

    #[test]
    fn test_auto_auth_on_first_query() {
        let server = MockServer::start();

        let auth_mock = server.mock(|when, then| {
            when.method(POST).path("/ajaxauth/login");
            then.status(200).body("");
        });

        let query_mock = server.mock(|when, then| {
            when.method(GET)
                .path_includes("/basicspacedata/query/class/gp");
            then.status(200)
                .body(r#"[{"OBJECT_NAME":"ISS","NORAD_CAT_ID":"25544"}]"#);
        });

        let client =
            SpaceTrackClient::with_base_url("user@example.com", "password", &server.base_url());

        let query = SpaceTrackQuery::new(RequestClass::GP)
            .filter("NORAD_CAT_ID", "25544")
            .limit(1);

        let result = client.query_json(&query);
        assert!(result.is_ok());

        auth_mock.assert();
        query_mock.assert();
    }

    #[test]
    fn test_reauth_on_401() {
        let server = MockServer::start();

        // Auth always succeeds
        server.mock(|when, then| {
            when.method(POST).path("/ajaxauth/login");
            then.status(200).body("");
        });

        // Query always returns 401 - after first 401, client re-auths and retries
        let query_mock = server.mock(|when, then| {
            when.method(GET)
                .path_includes("/basicspacedata/query/class/gp");
            then.status(401).body("Unauthorized");
        });

        let client =
            SpaceTrackClient::with_base_url("user@example.com", "password", &server.base_url());

        let query = SpaceTrackQuery::new(RequestClass::GP)
            .filter("NORAD_CAT_ID", "25544")
            .limit(1);

        // First attempt gets 401, triggers re-auth, second attempt also gets 401
        let result = client.query_raw(&query);
        assert!(result.is_err());

        // The query endpoint should be called twice: initial + retry after reauth
        query_mock.assert_calls(2);
    }

    #[test]
    fn test_query_raw() {
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(POST).path("/ajaxauth/login");
            then.status(200).body("");
        });

        server.mock(|when, then| {
            when.method(GET)
                .path_includes("/basicspacedata/query/class/gp");
            then.status(200)
                .body("1 25544U 98067A   24015.50000000\n2 25544  51.6400");
        });

        let client =
            SpaceTrackClient::with_base_url("user@example.com", "password", &server.base_url());

        let query = SpaceTrackQuery::new(RequestClass::GP)
            .filter("NORAD_CAT_ID", "25544")
            .format(OutputFormat::TLE);

        let result = client.query_raw(&query);
        assert!(result.is_ok());
        assert!(result.unwrap().contains("25544"));
    }

    #[test]
    fn test_query_json() {
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(POST).path("/ajaxauth/login");
            then.status(200).body("");
        });

        server.mock(|when, then| {
            when.method(GET)
                .path_includes("/basicspacedata/query/class/gp");
            then.status(200)
                .body(r#"[{"OBJECT_NAME":"ISS (ZARYA)","NORAD_CAT_ID":"25544"}]"#);
        });

        let client =
            SpaceTrackClient::with_base_url("user@example.com", "password", &server.base_url());

        let query = SpaceTrackQuery::new(RequestClass::GP)
            .filter("NORAD_CAT_ID", "25544")
            .limit(1);

        let result = client.query_json(&query);
        assert!(result.is_ok());
        let records = result.unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0]["OBJECT_NAME"], "ISS (ZARYA)");
    }

    #[test]
    fn test_query_json_rejects_non_json_format() {
        let client = SpaceTrackClient::new("user@example.com", "password");

        let query = SpaceTrackQuery::new(RequestClass::GP)
            .filter("NORAD_CAT_ID", "25544")
            .format(OutputFormat::TLE);

        let result = client.query_json(&query);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("requires JSON output format")
        );
    }

    #[test]
    fn test_query_gp() {
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(POST).path("/ajaxauth/login");
            then.status(200).body("");
        });

        server.mock(|when, then| {
            when.method(GET)
                .path_includes("/basicspacedata/query/class/gp");
            then.status(200).body(
                r#"[{
                    "OBJECT_NAME": "ISS (ZARYA)",
                    "NORAD_CAT_ID": "25544",
                    "EPOCH": "2024-01-15T12:00:00.000",
                    "MEAN_MOTION": "15.50000000",
                    "ECCENTRICITY": "0.00010000",
                    "INCLINATION": "51.6400"
                }]"#,
            );
        });

        let client =
            SpaceTrackClient::with_base_url("user@example.com", "password", &server.base_url());

        let query = SpaceTrackQuery::new(RequestClass::GP)
            .filter("NORAD_CAT_ID", "25544")
            .limit(1);

        let result = client.query_gp(&query);
        assert!(result.is_ok());
        let records = result.unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].object_name.as_deref(), Some("ISS (ZARYA)"));
        assert_eq!(records[0].norad_cat_id, Some(25544));
    }

    #[test]
    fn test_query_gp_rejects_non_json_format() {
        let client = SpaceTrackClient::new("user@example.com", "password");

        let query = SpaceTrackQuery::new(RequestClass::GP).format(OutputFormat::TLE);

        let result = client.query_gp(&query);
        assert!(result.is_err());
    }

    #[test]
    fn test_query_satcat() {
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(POST).path("/ajaxauth/login");
            then.status(200).body("");
        });

        server.mock(|when, then| {
            when.method(GET)
                .path_includes("/basicspacedata/query/class/satcat");
            then.status(200).body(
                r#"[{
                    "SATNAME": "ISS (ZARYA)",
                    "NORAD_CAT_ID": "25544",
                    "OBJECT_TYPE": "PAY",
                    "COUNTRY": "ISS"
                }]"#,
            );
        });

        let client =
            SpaceTrackClient::with_base_url("user@example.com", "password", &server.base_url());

        let query = SpaceTrackQuery::new(RequestClass::SATCAT)
            .filter("NORAD_CAT_ID", "25544")
            .limit(1);

        let result = client.query_satcat(&query);
        assert!(result.is_ok());
        let records = result.unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].satname.as_deref(), Some("ISS (ZARYA)"));
    }

    #[test]
    fn test_query_satcat_rejects_non_json_format() {
        let client = SpaceTrackClient::new("user@example.com", "password");

        let query = SpaceTrackQuery::new(RequestClass::SATCAT).format(OutputFormat::CSV);

        let result = client.query_satcat(&query);
        assert!(result.is_err());
    }

    #[test]
    fn test_http_error_500() {
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(POST).path("/ajaxauth/login");
            then.status(200).body("");
        });

        server.mock(|when, then| {
            when.method(GET)
                .path_includes("/basicspacedata/query/class/gp");
            then.status(500).body("Internal Server Error");
        });

        let client =
            SpaceTrackClient::with_base_url("user@example.com", "password", &server.base_url());

        let query = SpaceTrackQuery::new(RequestClass::GP)
            .filter("NORAD_CAT_ID", "25544")
            .limit(1);

        let result = client.query_raw(&query);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_json_response() {
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(POST).path("/ajaxauth/login");
            then.status(200).body("");
        });

        server.mock(|when, then| {
            when.method(GET)
                .path_includes("/basicspacedata/query/class/gp");
            then.status(200).body("this is not json");
        });

        let client =
            SpaceTrackClient::with_base_url("user@example.com", "password", &server.base_url());

        let query = SpaceTrackQuery::new(RequestClass::GP).limit(1);

        let result = client.query_json(&query);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("parse"));
    }

    #[test]
    fn test_query_json_rejects_all_non_json_formats() {
        let client = SpaceTrackClient::new("user@example.com", "password");
        let non_json_formats = vec![
            OutputFormat::XML,
            OutputFormat::HTML,
            OutputFormat::CSV,
            OutputFormat::TLE,
            OutputFormat::ThreeLe,
            OutputFormat::KVN,
        ];

        for fmt in non_json_formats {
            let query = SpaceTrackQuery::new(RequestClass::GP).format(fmt);
            let result = client.query_json(&query);
            assert!(
                result.is_err(),
                "query_json should reject format {}",
                fmt.as_str()
            );
        }
    }

    #[test]
    fn test_query_gp_rejects_all_non_json_formats() {
        let client = SpaceTrackClient::new("user@example.com", "password");
        let non_json_formats = vec![
            OutputFormat::XML,
            OutputFormat::HTML,
            OutputFormat::CSV,
            OutputFormat::TLE,
            OutputFormat::ThreeLe,
            OutputFormat::KVN,
        ];

        for fmt in non_json_formats {
            let query = SpaceTrackQuery::new(RequestClass::GP).format(fmt);
            let result = client.query_gp(&query);
            assert!(
                result.is_err(),
                "query_gp should reject format {}",
                fmt.as_str()
            );
        }
    }

    #[test]
    fn test_query_satcat_rejects_all_non_json_formats() {
        let client = SpaceTrackClient::new("user@example.com", "password");
        let non_json_formats = vec![
            OutputFormat::XML,
            OutputFormat::HTML,
            OutputFormat::CSV,
            OutputFormat::TLE,
            OutputFormat::ThreeLe,
            OutputFormat::KVN,
        ];

        for fmt in non_json_formats {
            let query = SpaceTrackQuery::new(RequestClass::SATCAT).format(fmt);
            let result = client.query_satcat(&query);
            assert!(
                result.is_err(),
                "query_satcat should reject format {}",
                fmt.as_str()
            );
        }
    }

    // Integration tests against the real SpaceTrack test server
    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    fn test_integration_auth() {
        let user = std::env::var("TEST_SPACETRACK_USER")
            .expect("TEST_SPACETRACK_USER env var must be set");
        let pass = std::env::var("TEST_SPACETRACK_PASS")
            .expect("TEST_SPACETRACK_PASS env var must be set");
        let base_url = std::env::var("TEST_SPACETRACK_BASE_URL")
            .expect("TEST_SPACETRACK_BASE_URL env var must be set");

        let client = SpaceTrackClient::with_base_url(&user, &pass, &base_url);
        let result = client.authenticate();
        assert!(result.is_ok(), "Authentication failed: {:?}", result.err());
    }

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    fn test_integration_gp_query() {
        let user = std::env::var("TEST_SPACETRACK_USER")
            .expect("TEST_SPACETRACK_USER env var must be set");
        let pass = std::env::var("TEST_SPACETRACK_PASS")
            .expect("TEST_SPACETRACK_PASS env var must be set");
        let base_url = std::env::var("TEST_SPACETRACK_BASE_URL")
            .expect("TEST_SPACETRACK_BASE_URL env var must be set");

        let client = SpaceTrackClient::with_base_url(&user, &pass, &base_url);

        let query = SpaceTrackQuery::new(RequestClass::GP)
            .filter("NORAD_CAT_ID", "25544")
            .order_by("EPOCH", SortOrder::Desc)
            .limit(1);

        let records = client.query_gp(&query).expect("GP query failed");
        assert!(!records.is_empty(), "Expected at least one GP record");
        assert_eq!(records[0].norad_cat_id, Some(25544));
    }

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    fn test_integration_satcat_query() {
        let user = std::env::var("TEST_SPACETRACK_USER")
            .expect("TEST_SPACETRACK_USER env var must be set");
        let pass = std::env::var("TEST_SPACETRACK_PASS")
            .expect("TEST_SPACETRACK_PASS env var must be set");
        let base_url = std::env::var("TEST_SPACETRACK_BASE_URL")
            .expect("TEST_SPACETRACK_BASE_URL env var must be set");

        let client = SpaceTrackClient::with_base_url(&user, &pass, &base_url);

        let query = SpaceTrackQuery::new(RequestClass::SATCAT)
            .filter("NORAD_CAT_ID", "25544")
            .limit(1);

        let records = client.query_satcat(&query).expect("SATCAT query failed");
        assert!(!records.is_empty(), "Expected at least one SATCAT record");
        assert_eq!(records[0].norad_cat_id, Some(25544));
    }

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    fn test_integration_tle_format() {
        let user = std::env::var("TEST_SPACETRACK_USER")
            .expect("TEST_SPACETRACK_USER env var must be set");
        let pass = std::env::var("TEST_SPACETRACK_PASS")
            .expect("TEST_SPACETRACK_PASS env var must be set");
        let base_url = std::env::var("TEST_SPACETRACK_BASE_URL")
            .expect("TEST_SPACETRACK_BASE_URL env var must be set");

        let client = SpaceTrackClient::with_base_url(&user, &pass, &base_url);

        let query = SpaceTrackQuery::new(RequestClass::GP)
            .filter("NORAD_CAT_ID", "25544")
            .order_by("EPOCH", SortOrder::Desc)
            .limit(1)
            .format(OutputFormat::TLE);

        let raw = client.query_raw(&query).expect("TLE query failed");
        assert!(!raw.trim().is_empty(), "Expected TLE data");
    }

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    fn test_integration_query_with_operators() {
        use crate::spacetrack::operators;

        let user = std::env::var("TEST_SPACETRACK_USER")
            .expect("TEST_SPACETRACK_USER env var must be set");
        let pass = std::env::var("TEST_SPACETRACK_PASS")
            .expect("TEST_SPACETRACK_PASS env var must be set");
        let base_url = std::env::var("TEST_SPACETRACK_BASE_URL")
            .expect("TEST_SPACETRACK_BASE_URL env var must be set");

        let client = SpaceTrackClient::with_base_url(&user, &pass, &base_url);

        let query = SpaceTrackQuery::new(RequestClass::GP)
            .filter(
                "NORAD_CAT_ID",
                &operators::inclusive_range("25544", "25550"),
            )
            .filter(
                "EPOCH",
                &operators::greater_than(operators::now_offset(-3650)),
            )
            .order_by("NORAD_CAT_ID", SortOrder::Asc)
            .limit(5);

        let records = client.query_gp(&query).expect("Operator query failed");
        assert!(!records.is_empty(), "Expected at least one record");
    }

    // -- FileShare tests --

    #[test]
    fn test_fileshare_upload() {
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(POST).path("/ajaxauth/login");
            then.status(200).body("");
        });

        let upload_mock = server.mock(|when, then| {
            when.method(POST)
                .path_includes("/fileshare/upload/folder_id/100");
            then.status(200).body(r#"{"status":"ok"}"#);
        });

        let client =
            SpaceTrackClient::with_base_url("user@example.com", "password", &server.base_url());

        let result = client.fileshare_upload("100", "test.txt", b"hello world");
        assert!(result.is_ok());
        assert!(result.unwrap().contains("ok"));
        upload_mock.assert();
    }

    #[test]
    fn test_fileshare_download() {
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(POST).path("/ajaxauth/login");
            then.status(200).body("");
        });

        let file_data: Vec<u8> = vec![1, 2, 3, 4, 5];
        server.mock(|when, then| {
            when.method(GET).path("/fileshare/download/file_id/12345");
            then.status(200).body(&file_data);
        });

        let client =
            SpaceTrackClient::with_base_url("user@example.com", "password", &server.base_url());

        let result = client.fileshare_download("12345");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), file_data);
    }

    #[test]
    fn test_fileshare_download_folder() {
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(POST).path("/ajaxauth/login");
            then.status(200).body("");
        });

        let zip_data: Vec<u8> = vec![0x50, 0x4B, 0x03, 0x04]; // ZIP magic bytes
        server.mock(|when, then| {
            when.method(GET).path("/fileshare/download/folder_id/100");
            then.status(200).body(&zip_data);
        });

        let client =
            SpaceTrackClient::with_base_url("user@example.com", "password", &server.base_url());

        let result = client.fileshare_download_folder("100");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), zip_data);
    }

    #[test]
    fn test_fileshare_list_files() {
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(POST).path("/ajaxauth/login");
            then.status(200).body("");
        });

        server.mock(|when, then| {
            when.method(GET).path("/fileshare/query/class/file");
            then.status(200)
                .body(r#"[{"FILE_ID":"12345","FILE_NAME":"data.txt","FOLDER_ID":"100"}]"#);
        });

        let client =
            SpaceTrackClient::with_base_url("user@example.com", "password", &server.base_url());

        let result = client.fileshare_list_files();
        assert!(result.is_ok());
        let files = result.unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].file_id.as_deref(), Some("12345"));
        assert_eq!(files[0].file_name.as_deref(), Some("data.txt"));
    }

    #[test]
    fn test_fileshare_list_folders() {
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(POST).path("/ajaxauth/login");
            then.status(200).body("");
        });

        server.mock(|when, then| {
            when.method(GET).path("/fileshare/query/class/folder");
            then.status(200)
                .body(r#"[{"FOLDER_ID":"100","FOLDER_NAME":"my_data"}]"#);
        });

        let client =
            SpaceTrackClient::with_base_url("user@example.com", "password", &server.base_url());

        let result = client.fileshare_list_folders();
        assert!(result.is_ok());
        let folders = result.unwrap();
        assert_eq!(folders.len(), 1);
        assert_eq!(folders[0].folder_id.as_deref(), Some("100"));
        assert_eq!(folders[0].folder_name.as_deref(), Some("my_data"));
    }

    #[test]
    fn test_fileshare_delete() {
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(POST).path("/ajaxauth/login");
            then.status(200).body("");
        });

        server.mock(|when, then| {
            when.method(GET)
                .path("/fileshare/query/class/delete/file_id/12345");
            then.status(200).body(r#"{"status":"deleted"}"#);
        });

        let client =
            SpaceTrackClient::with_base_url("user@example.com", "password", &server.base_url());

        let result = client.fileshare_delete("12345");
        assert!(result.is_ok());
        assert!(result.unwrap().contains("deleted"));
    }

    // -- SP Ephemeris tests --

    #[test]
    fn test_spephemeris_download() {
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(POST).path("/ajaxauth/login");
            then.status(200).body("");
        });

        let ephem_data: Vec<u8> = vec![0xDE, 0xAD, 0xBE, 0xEF];
        server.mock(|when, then| {
            when.method(GET).path("/spephemeris/download/file_id/99999");
            then.status(200).body(&ephem_data);
        });

        let client =
            SpaceTrackClient::with_base_url("user@example.com", "password", &server.base_url());

        let result = client.spephemeris_download("99999");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ephem_data);
    }

    #[test]
    fn test_spephemeris_list_files() {
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(POST).path("/ajaxauth/login");
            then.status(200).body("");
        });

        server.mock(|when, then| {
            when.method(GET).path("/spephemeris/query/class/file");
            then.status(200)
                .body(r#"[{"FILE_ID":"99999","NORAD_CAT_ID":"25544","FILE_NAME":"iss.e"}]"#);
        });

        let client =
            SpaceTrackClient::with_base_url("user@example.com", "password", &server.base_url());

        let result = client.spephemeris_list_files();
        assert!(result.is_ok());
        let files = result.unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].file_id.as_deref(), Some("99999"));
        assert_eq!(files[0].norad_cat_id, Some(25544));
    }

    #[test]
    fn test_spephemeris_file_history() {
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(POST).path("/ajaxauth/login");
            then.status(200).body("");
        });

        server.mock(|when, then| {
            when.method(GET)
                .path("/spephemeris/query/class/file_history");
            then.status(200)
                .body(r#"[{"FILE_ID":"99999","VERSION":"2"}]"#);
        });

        let client =
            SpaceTrackClient::with_base_url("user@example.com", "password", &server.base_url());

        let result = client.spephemeris_file_history();
        assert!(result.is_ok());
        let history = result.unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0]["FILE_ID"], "99999");
    }

    // -- Public Files tests --

    #[test]
    fn test_publicfiles_download() {
        let server = MockServer::start();

        // No auth mock - publicfiles should NOT require authentication
        let file_data: Vec<u8> = vec![0xCA, 0xFE];
        server.mock(|when, then| {
            when.method(GET)
                .path("/publicfiles/query/class/download")
                .query_param("name", "catalog.txt");
            then.status(200).body(&file_data);
        });

        let client =
            SpaceTrackClient::with_base_url("user@example.com", "password", &server.base_url());

        let result = client.publicfiles_download("catalog.txt");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), file_data);
    }

    #[test]
    fn test_publicfiles_list_dirs() {
        let server = MockServer::start();

        // No auth mock - publicfiles should NOT require authentication
        server.mock(|when, then| {
            when.method(GET).path("/publicfiles/query/class/dirs");
            then.status(200)
                .body(r#"[{"dir":"data"},{"dir":"reports"}]"#);
        });

        let client =
            SpaceTrackClient::with_base_url("user@example.com", "password", &server.base_url());

        let result = client.publicfiles_list_dirs();
        assert!(result.is_ok());
        let dirs = result.unwrap();
        assert_eq!(dirs.len(), 2);
        assert_eq!(dirs[0]["dir"], "data");
    }

    #[test]
    fn test_publicfiles_download_url_encoding() {
        let server = MockServer::start();

        let file_data: Vec<u8> = vec![0x01];
        server.mock(|when, then| {
            when.method(GET)
                .path("/publicfiles/query/class/download")
                .query_param("name", "my file.txt");
            then.status(200).body(&file_data);
        });

        let client =
            SpaceTrackClient::with_base_url("user@example.com", "password", &server.base_url());

        let result = client.publicfiles_download("my file.txt");
        assert!(result.is_ok());
    }

    // -- Binary GET infrastructure tests --

    #[test]
    fn test_execute_get_binary() {
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(POST).path("/ajaxauth/login");
            then.status(200).body("");
        });

        let binary_data: Vec<u8> = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        server.mock(|when, then| {
            when.method(GET).path("/test/binary");
            then.status(200).body(&binary_data);
        });

        let client =
            SpaceTrackClient::with_base_url("user@example.com", "password", &server.base_url());

        let url = format!("{}/test/binary", server.base_url());
        let result = client.authenticated_get_binary(&url);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), binary_data);
    }

    #[test]
    fn test_authenticated_get_binary_reauth_on_401() {
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(POST).path("/ajaxauth/login");
            then.status(200).body("");
        });

        // Always returns 401 to test reauth retry
        let binary_mock = server.mock(|when, then| {
            when.method(GET).path("/test/binary");
            then.status(401).body("Unauthorized");
        });

        let client =
            SpaceTrackClient::with_base_url("user@example.com", "password", &server.base_url());

        let url = format!("{}/test/binary", server.base_url());
        let result = client.authenticated_get_binary(&url);
        assert!(result.is_err());

        // Should be called twice: initial + retry after reauth
        binary_mock.assert_calls(2);
    }

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    fn test_integration_invalid_credentials() {
        let client = SpaceTrackClient::new("invalid@example.com", "wrongpassword");
        let result = client.authenticate();
        assert!(result.is_err());
    }

    // -- Order-by URL encoding tests --

    #[test]
    fn test_query_with_order_by_produces_valid_url() {
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(POST).path("/ajaxauth/login");
            then.status(200).body("");
        });

        let query_mock = server.mock(|when, then| {
            when.method(GET).path_includes("/orderby/EPOCH%20desc/");
            then.status(200)
                .body(r#"[{"OBJECT_NAME":"ISS","NORAD_CAT_ID":"25544"}]"#);
        });

        let client =
            SpaceTrackClient::with_base_url("user@example.com", "password", &server.base_url());

        let query = SpaceTrackQuery::new(RequestClass::GP)
            .filter("NORAD_CAT_ID", "25544")
            .order_by("EPOCH", SortOrder::Desc)
            .limit(1);

        let result = client.query_json(&query);
        assert!(
            result.is_ok(),
            "ureq rejected the order_by URL: {:?}",
            result.err()
        );
        query_mock.assert();
    }

    // -- Rate limiting tests --

    #[test]
    fn test_client_with_rate_limit() {
        let config = crate::spacetrack::RateLimitConfig {
            max_per_minute: 10,
            max_per_hour: 100,
        };
        let client = SpaceTrackClient::with_rate_limit("user@example.com", "password123", config);
        assert_eq!(client.identity, "user@example.com");
        assert_eq!(client.password, "password123");
        assert_eq!(client.base_url, DEFAULT_BASE_URL);
    }

    #[test]
    fn test_client_with_base_url_and_rate_limit() {
        let config = crate::spacetrack::RateLimitConfig::disabled();
        let client = SpaceTrackClient::with_base_url_and_rate_limit(
            "user@example.com",
            "password123",
            "https://test.space-track.org/",
            config,
        );
        assert_eq!(client.base_url, "https://test.space-track.org");
    }

    #[test]
    fn test_client_default_rate_limit_does_not_delay() {
        // With default 25/min limit, a single query should not be delayed
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(POST).path("/ajaxauth/login");
            then.status(200).body("");
        });

        server.mock(|when, then| {
            when.method(GET)
                .path_includes("/basicspacedata/query/class/gp");
            then.status(200)
                .body(r#"[{"OBJECT_NAME":"ISS","NORAD_CAT_ID":"25544"}]"#);
        });

        let client =
            SpaceTrackClient::with_base_url("user@example.com", "password", &server.base_url());

        let query = SpaceTrackQuery::new(RequestClass::GP)
            .filter("NORAD_CAT_ID", "25544")
            .limit(1);

        let start = std::time::Instant::now();
        let result = client.query_json(&query);
        let elapsed = start.elapsed();

        assert!(result.is_ok());
        // Should complete quickly (no rate limit delay)
        assert!(elapsed < std::time::Duration::from_secs(1));
    }

    #[test]
    fn test_client_disabled_rate_limit() {
        let config = crate::spacetrack::RateLimitConfig::disabled();
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(POST).path("/ajaxauth/login");
            then.status(200).body("");
        });

        server.mock(|when, then| {
            when.method(GET)
                .path_includes("/basicspacedata/query/class/gp");
            then.status(200)
                .body(r#"[{"OBJECT_NAME":"ISS","NORAD_CAT_ID":"25544"}]"#);
        });

        let client = SpaceTrackClient::with_base_url_and_rate_limit(
            "user@example.com",
            "password",
            &server.base_url(),
            config,
        );

        let query = SpaceTrackQuery::new(RequestClass::GP)
            .filter("NORAD_CAT_ID", "25544")
            .limit(1);

        let result = client.query_json(&query);
        assert!(result.is_ok());
    }
}
