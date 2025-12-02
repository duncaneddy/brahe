/*!
 * SpaceTrack async client
 *
 * Provides async HTTP client for SpaceTrack API requests with authentication,
 * rate limiting, and cookie-based session management.
 */

use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

use reqwest::Client;
use reqwest::cookie::Jar;
use serde::de::DeserializeOwned;
use serde_json::Value;

use super::error::SpaceTrackError;
use super::operators::{
    QueryValue, equals, greater_than, inclusive_range, less_than, like, not_equal, null_val,
    startswith,
};
use super::query::{
    HasTLEData, SpaceTrackOrder, SpaceTrackPredicate, SpaceTrackPredicateBuilder, SpaceTrackQuery,
};
use super::rate_limiter::RateLimiter;
use super::request_classes::{
    AnnouncementRecord, AnnouncementRequest, BoxscoreRecord, BoxscoreRequest, CDMPublicRecord,
    CDMPublicRequest, DecayRecord, DecayRequest, GPHistoryRecord, GPRecord, GPRequest,
    LaunchSiteRecord, LaunchSiteRequest, OMMRecord, RequestClass, SATCATChangeRecord,
    SATCATChangeRequest, SATCATDebutRecord, SATCATDebutRequest, SATCATRecord, SATCATRequest,
    TIPRecord, TIPRequest, TLERecord, TLERequest,
};
use crate::propagators::SGPPropagator;
use crate::utils::BraheError;

/// Default SpaceTrack API base URL.
pub const DEFAULT_BASE_URL: &str = "https://www.space-track.org/";

/// Test SpaceTrack API base URL.
pub const TEST_BASE_URL: &str = "https://for-testing-only.space-track.org/";

/// Async SpaceTrack API client.
///
/// Provides access to SpaceTrack API with automatic authentication,
/// rate limiting, and cookie-based session management.
///
/// # Example
///
/// ```ignore
/// use brahe::spacetrack::SpaceTrackClient;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let client = SpaceTrackClient::new("username", "password").await?;
///
///     // Query GP data for ISS
///     let gp = client.gp()
///         .norad_cat_id(25544)
///         .limit(1)
///         .fetch()
///         .await?;
///
///     Ok(())
/// }
/// ```
pub struct SpaceTrackClient {
    /// HTTP client with cookie jar for session management
    client: Client,
    /// Base URL for API requests
    base_url: String,
    /// Rate limiter instance
    rate_limiter: Arc<RateLimiter>,
    /// Whether the client has been authenticated
    authenticated: bool,
    /// Username for authentication
    identity: String,
    /// Password for authentication
    password: String,
}

impl SpaceTrackClient {
    /// Create a new SpaceTrack client and authenticate.
    ///
    /// # Arguments
    ///
    /// * `identity` - SpaceTrack username
    /// * `password` - SpaceTrack password
    ///
    /// # Returns
    ///
    /// A new authenticated `SpaceTrackClient` or an error if authentication fails.
    pub async fn new(identity: &str, password: &str) -> Result<Self, SpaceTrackError> {
        Self::with_base_url(identity, password, DEFAULT_BASE_URL).await
    }

    /// Create a new SpaceTrack client with a custom base URL.
    ///
    /// This is useful for testing against the SpaceTrack test server.
    ///
    /// # Arguments
    ///
    /// * `identity` - SpaceTrack username
    /// * `password` - SpaceTrack password
    /// * `base_url` - Base URL for API requests
    ///
    /// # Returns
    ///
    /// A new authenticated `SpaceTrackClient` or an error if authentication fails.
    pub async fn with_base_url(
        identity: &str,
        password: &str,
        base_url: &str,
    ) -> Result<Self, SpaceTrackError> {
        // Create cookie jar for session management
        let cookie_jar = Arc::new(Jar::default());

        // Create HTTP client with cookie support and reasonable timeout
        let client = Client::builder()
            .cookie_provider(cookie_jar)
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| SpaceTrackError::NetworkError(e.to_string()))?;

        let mut client = Self {
            client,
            base_url: base_url.to_string(),
            rate_limiter: Arc::new(RateLimiter::new()),
            authenticated: false,
            identity: identity.to_string(),
            password: password.to_string(),
        };

        // Authenticate immediately
        client.authenticate().await?;

        Ok(client)
    }

    /// Create an unauthenticated client (for testing).
    ///
    /// The client will need to call `authenticate()` before making requests.
    pub fn new_unauthenticated(identity: &str, password: &str, base_url: &str) -> Self {
        let cookie_jar = Arc::new(Jar::default());
        let client = Client::builder()
            .cookie_provider(cookie_jar)
            .timeout(Duration::from_secs(60))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: base_url.to_string(),
            rate_limiter: Arc::new(RateLimiter::new()),
            authenticated: false,
            identity: identity.to_string(),
            password: password.to_string(),
        }
    }

    /// Authenticate with SpaceTrack.
    ///
    /// This is called automatically when creating a new client, but can be
    /// called again if the session expires.
    pub async fn authenticate(&mut self) -> Result<(), SpaceTrackError> {
        if self.authenticated {
            return Ok(());
        }

        let url = format!("{}ajaxauth/login", self.base_url);

        let params = [
            ("identity", self.identity.as_str()),
            ("password", self.password.as_str()),
        ];

        let response = self
            .client
            .post(&url)
            .form(&params)
            .send()
            .await
            .map_err(|e| SpaceTrackError::NetworkError(e.to_string()))?;

        let text = response
            .text()
            .await
            .map_err(|e| SpaceTrackError::NetworkError(e.to_string()))?;

        // Check for authentication failure
        if text.contains("\"Login\":\"Failed\"") || text.contains("\"Login\": \"Failed\"") {
            return Err(SpaceTrackError::AuthenticationError(
                "Invalid credentials".to_string(),
            ));
        }

        self.authenticated = true;
        Ok(())
    }

    /// Log out of SpaceTrack.
    ///
    /// This invalidates the current session.
    pub async fn logout(&mut self) -> Result<(), SpaceTrackError> {
        if !self.authenticated {
            return Ok(());
        }

        let url = format!("{}ajaxauth/logout", self.base_url);

        self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| SpaceTrackError::NetworkError(e.to_string()))?;

        self.authenticated = false;
        Ok(())
    }

    /// Check if the client is authenticated.
    pub fn is_authenticated(&self) -> bool {
        self.authenticated
    }

    /// Get the base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Get the rate limiter.
    pub fn rate_limiter(&self) -> &RateLimiter {
        &self.rate_limiter
    }

    /// Wait for rate limit if necessary.
    async fn wait_for_rate_limit(&self) -> Result<(), SpaceTrackError> {
        if let Some(wait_duration) = self.rate_limiter.check() {
            tokio::time::sleep(wait_duration).await;
        }
        Ok(())
    }

    /// Make a generic API request.
    ///
    /// # Arguments
    ///
    /// * `controller` - The controller name (e.g., "basicspacedata")
    /// * `class` - The request class name (e.g., "gp")
    /// * `predicates` - Query predicates as (name, value) pairs
    /// * `format` - Optional output format (defaults to JSON)
    ///
    /// # Returns
    ///
    /// The response body as a string.
    pub async fn generic_request(
        &self,
        controller: &str,
        class: &str,
        predicates: &[(&str, QueryValue)],
        format: Option<&str>,
    ) -> Result<String, SpaceTrackError> {
        if !self.authenticated {
            return Err(SpaceTrackError::SessionExpired);
        }

        // Wait for rate limit
        self.wait_for_rate_limit().await?;

        // Build URL
        let mut url = format!("{}{}/query/class/{}", self.base_url, controller, class);

        // Add predicates to URL
        for (name, value) in predicates {
            url.push('/');
            url.push_str(name);
            url.push('/');
            url.push_str(&urlencoding::encode(&value.to_url_segment()));
        }

        // Add format if specified
        if let Some(fmt) = format {
            url.push_str("/format/");
            url.push_str(fmt);
        }

        // Make request
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| SpaceTrackError::NetworkError(e.to_string()))?;

        // Record the request for rate limiting
        self.rate_limiter.record_request();

        // Check for rate limit error in response
        let status = response.status();
        if status.as_u16() == 500 {
            let text = response
                .text()
                .await
                .map_err(|e| SpaceTrackError::NetworkError(e.to_string()))?;

            if text.contains("violated your query rate limit") {
                return Err(SpaceTrackError::RateLimitError {
                    retry_after: Duration::from_secs(60),
                });
            }

            return Err(SpaceTrackError::NetworkError(format!("HTTP 500: {}", text)));
        }

        if !status.is_success() {
            return Err(SpaceTrackError::NetworkError(format!(
                "HTTP {}: {}",
                status.as_u16(),
                status.canonical_reason().unwrap_or("Unknown error")
            )));
        }

        let text = response
            .text()
            .await
            .map_err(|e| SpaceTrackError::NetworkError(e.to_string()))?;

        // Normalize CRLF to LF
        Ok(text.replace("\r\n", "\n"))
    }

    /// Make a request and parse JSON response.
    ///
    /// # Arguments
    ///
    /// * `controller` - The controller name
    /// * `class` - The request class name
    /// * `predicates` - Query predicates
    ///
    /// # Returns
    ///
    /// Parsed JSON value.
    pub async fn request_json(
        &self,
        controller: &str,
        class: &str,
        predicates: &[(&str, QueryValue)],
    ) -> Result<Value, SpaceTrackError> {
        let text = self
            .generic_request(controller, class, predicates, None)
            .await?;

        serde_json::from_str(&text)
            .map_err(|e| SpaceTrackError::ParseError(format!("Invalid JSON: {}", e)))
    }

    /// Execute a typed request.
    ///
    /// # Type Parameters
    ///
    /// * `R` - The request class type implementing `RequestClass`
    ///
    /// # Arguments
    ///
    /// * `request` - The request builder
    ///
    /// # Returns
    ///
    /// A vector of the record type defined by the request class.
    pub(crate) async fn query<R: RequestClass>(
        &self,
        request: &R,
    ) -> Result<Vec<R::Record>, SpaceTrackError> {
        let text = self
            .generic_request(
                R::controller(),
                R::class_name(),
                &request.predicates(),
                None,
            )
            .await?;

        serde_json::from_str(&text)
            .map_err(|e| SpaceTrackError::ParseError(format!("Failed to parse response: {}", e)))
    }

    /// Execute a request and return TLEs as SGP propagators.
    ///
    /// This is a convenience method that fetches TLE data and converts it
    /// to SGPPropagator instances.
    ///
    /// # Arguments
    ///
    /// * `request` - The request builder (should be a GP or TLE request)
    /// * `step_size` - The propagator step size in seconds
    ///
    /// # Returns
    ///
    /// A vector of `SGPPropagator` instances.
    pub(crate) async fn fetch_propagators<R: RequestClass>(
        &self,
        request: &R,
        step_size: f64,
    ) -> Result<Vec<SGPPropagator>, BraheError> {
        // Request in 3LE format
        let text = self
            .generic_request(
                R::controller(),
                R::class_name(),
                &request.predicates(),
                Some("3le"),
            )
            .await?;

        parse_3le_to_propagators(&text, step_size)
    }

    // ========================================================================
    // Typed query methods
    // ========================================================================

    /// Query GP (General Perturbations) data.
    ///
    /// Returns the latest GP element sets for cataloged objects.
    ///
    /// # Arguments
    ///
    /// * `norad_cat_id` - Filter by NORAD catalog ID
    /// * `object_name` - Filter by object name (supports operators like "~~STARLINK%")
    /// * `object_id` - Filter by object ID (international designator)
    /// * `epoch` - Filter by epoch (supports operators like ">2024-01-01")
    /// * `object_type` - Filter by type (PAYLOAD, ROCKET BODY, DEBRIS)
    /// * `country_code` - Filter by country code
    /// * `limit` - Maximum number of results
    /// * `orderby` - Field to order by with direction (e.g., "epoch", "epoch desc")
    ///
    /// # Returns
    ///
    /// A vector of GP records.
    #[allow(clippy::too_many_arguments)]
    pub async fn gp(
        &self,
        norad_cat_id: Option<u32>,
        object_name: Option<&str>,
        object_id: Option<&str>,
        epoch: Option<&str>,
        object_type: Option<&str>,
        country_code: Option<&str>,
        limit: Option<u32>,
        orderby: Option<&str>,
    ) -> Result<Vec<GPRecord>, SpaceTrackError> {
        let mut req = GPRequest::new();

        if let Some(v) = norad_cat_id {
            req = req.norad_cat_id(v);
        }
        if let Some(v) = object_name {
            req = req.object_name(parse_query_string(v));
        }
        if let Some(v) = object_id {
            req = req.object_id(parse_query_string(v));
        }
        if let Some(v) = epoch {
            req = req.epoch(parse_query_string(v));
        }
        if let Some(v) = object_type {
            req = req.object_type(parse_query_string(v));
        }
        if let Some(v) = country_code {
            req = req.country_code(parse_query_string(v));
        }
        if let Some(v) = limit {
            req = req.limit(v);
        }
        if let Some(v) = orderby {
            req = apply_orderby(req, v);
        }

        self.query(&req).await
    }

    /// Query GP data and return as SGP propagators.
    ///
    /// # Arguments
    ///
    /// * `step_size` - Propagator step size in seconds
    /// * `norad_cat_id` - Filter by NORAD catalog ID
    /// * `object_name` - Filter by object name
    /// * `limit` - Maximum number of results
    ///
    /// # Returns
    ///
    /// A vector of SGP propagators.
    pub async fn gp_as_propagators(
        &self,
        step_size: f64,
        norad_cat_id: Option<u32>,
        object_name: Option<&str>,
        limit: Option<u32>,
    ) -> Result<Vec<SGPPropagator>, BraheError> {
        let mut req = GPRequest::new();

        if let Some(v) = norad_cat_id {
            req = req.norad_cat_id(v);
        }
        if let Some(v) = object_name {
            req = req.object_name(parse_query_string(v));
        }
        if let Some(v) = limit {
            req = req.limit(v);
        }

        self.fetch_propagators(&req, step_size).await
    }

    /// Query SATCAT (Satellite Catalog) data.
    ///
    /// Returns satellite catalog entries.
    ///
    /// # Arguments
    ///
    /// * `norad_cat_id` - Filter by NORAD catalog ID
    /// * `satname` - Filter by satellite name
    /// * `intldes` - Filter by international designator
    /// * `object_type` - Filter by type
    /// * `country` - Filter by country
    /// * `launch` - Filter by launch date
    /// * `current` - Filter by current status (Y/N)
    /// * `limit` - Maximum results
    /// * `orderby` - Field to order by
    ///
    /// # Returns
    ///
    /// A vector of SATCAT records.
    #[allow(clippy::too_many_arguments)]
    pub async fn satcat(
        &self,
        norad_cat_id: Option<u32>,
        satname: Option<&str>,
        intldes: Option<&str>,
        object_type: Option<&str>,
        country: Option<&str>,
        launch: Option<&str>,
        current: Option<&str>,
        limit: Option<u32>,
        orderby: Option<&str>,
    ) -> Result<Vec<SATCATRecord>, SpaceTrackError> {
        let mut req = SATCATRequest::new();

        if let Some(v) = norad_cat_id {
            req = req.norad_cat_id(v);
        }
        if let Some(v) = satname {
            req = req.satname(parse_query_string(v));
        }
        if let Some(v) = intldes {
            req = req.intldes(parse_query_string(v));
        }
        if let Some(v) = object_type {
            req = req.object_type(parse_query_string(v));
        }
        if let Some(v) = country {
            req = req.country(parse_query_string(v));
        }
        if let Some(v) = launch {
            req = req.launch(parse_query_string(v));
        }
        if let Some(v) = current {
            req = req.current(parse_query_string(v));
        }
        if let Some(v) = limit {
            req = req.limit(v);
        }
        if let Some(v) = orderby {
            req = apply_orderby(req, v);
        }

        self.query(&req).await
    }

    /// Query TLE (Two-Line Element) data.
    ///
    /// Note: This class is deprecated. Use gp() instead.
    ///
    /// # Arguments
    ///
    /// * `norad_cat_id` - Filter by NORAD catalog ID
    /// * `object_name` - Filter by object name
    /// * `epoch` - Filter by epoch
    /// * `limit` - Maximum results
    /// * `orderby` - Field to order by
    ///
    /// # Returns
    ///
    /// A vector of TLE records.
    pub async fn tle(
        &self,
        norad_cat_id: Option<u32>,
        object_name: Option<&str>,
        epoch: Option<&str>,
        limit: Option<u32>,
        orderby: Option<&str>,
    ) -> Result<Vec<TLERecord>, SpaceTrackError> {
        let mut req = TLERequest::new();

        if let Some(v) = norad_cat_id {
            req = req.norad_cat_id(v);
        }
        if let Some(v) = object_name {
            req = req.object_name(parse_query_string(v));
        }
        if let Some(v) = epoch {
            req = req.epoch(parse_query_string(v));
        }
        if let Some(v) = limit {
            req = req.limit(v);
        }
        if let Some(v) = orderby {
            req = apply_orderby(req, v);
        }

        self.query(&req).await
    }

    /// Query decay data.
    ///
    /// Returns predicted and actual re-entry information.
    ///
    /// # Arguments
    ///
    /// * `norad_cat_id` - Filter by NORAD catalog ID
    /// * `decay_epoch` - Filter by decay epoch
    /// * `country` - Filter by country
    /// * `limit` - Maximum results
    ///
    /// # Returns
    ///
    /// A vector of decay records.
    pub async fn decay(
        &self,
        norad_cat_id: Option<u32>,
        decay_epoch: Option<&str>,
        country: Option<&str>,
        limit: Option<u32>,
    ) -> Result<Vec<DecayRecord>, SpaceTrackError> {
        let mut req = DecayRequest::new();

        if let Some(v) = norad_cat_id {
            req = req.norad_cat_id(v);
        }
        if let Some(v) = decay_epoch {
            req = req.decay_epoch(parse_query_string(v));
        }
        if let Some(v) = country {
            req = req.country(parse_query_string(v));
        }
        if let Some(v) = limit {
            req = req.limit(v);
        }

        self.query(&req).await
    }

    /// Query TIP (Tracking and Impact Prediction) data.
    ///
    /// Returns tracking and impact prediction messages.
    ///
    /// # Arguments
    ///
    /// * `norad_cat_id` - Filter by NORAD catalog ID
    /// * `decay_epoch` - Filter by decay epoch
    /// * `high_interest` - Filter by high interest flag
    /// * `limit` - Maximum results
    ///
    /// # Returns
    ///
    /// A vector of TIP records.
    pub async fn tip(
        &self,
        norad_cat_id: Option<u32>,
        decay_epoch: Option<&str>,
        high_interest: Option<&str>,
        limit: Option<u32>,
    ) -> Result<Vec<TIPRecord>, SpaceTrackError> {
        let mut req = TIPRequest::new();

        if let Some(v) = norad_cat_id {
            req = req.norad_cat_id(v);
        }
        if let Some(v) = decay_epoch {
            req = req.decay_epoch(parse_query_string(v));
        }
        if let Some(v) = high_interest {
            req = req.high_interest(parse_query_string(v));
        }
        if let Some(v) = limit {
            req = req.limit(v);
        }

        self.query(&req).await
    }

    /// Query public CDM (Conjunction Data Message) data.
    ///
    /// Returns public conjunction data.
    ///
    /// # Arguments
    ///
    /// * `sat1_norad_cat_id` - Filter by SAT1 NORAD catalog ID
    /// * `sat2_norad_cat_id` - Filter by SAT2 NORAD catalog ID
    /// * `tca` - Filter by time of closest approach
    /// * `limit` - Maximum results
    ///
    /// # Returns
    ///
    /// A vector of CDM records.
    pub async fn cdm_public(
        &self,
        sat1_norad_cat_id: Option<u32>,
        sat2_norad_cat_id: Option<u32>,
        tca: Option<&str>,
        limit: Option<u32>,
    ) -> Result<Vec<CDMPublicRecord>, SpaceTrackError> {
        let mut req = CDMPublicRequest::new();

        if let Some(v) = sat1_norad_cat_id {
            req = req.sat1_norad_cat_id(v);
        }
        if let Some(v) = sat2_norad_cat_id {
            req = req.sat2_norad_cat_id(v);
        }
        if let Some(v) = tca {
            req = req.tca(parse_query_string(v));
        }
        if let Some(v) = limit {
            req = req.limit(v);
        }

        self.query(&req).await
    }

    /// Query boxscore (catalog statistics) data.
    ///
    /// Returns catalog summary statistics by country.
    ///
    /// # Arguments
    ///
    /// * `country` - Filter by country code
    ///
    /// # Returns
    ///
    /// A vector of boxscore records.
    pub async fn boxscore(
        &self,
        country: Option<&str>,
    ) -> Result<Vec<BoxscoreRecord>, SpaceTrackError> {
        let mut req = BoxscoreRequest::new();

        if let Some(v) = country {
            req = req.country(parse_query_string(v));
        }

        self.query(&req).await
    }

    /// Query launch site data.
    ///
    /// Returns launch facility information.
    ///
    /// # Arguments
    ///
    /// * `site_code` - Filter by site code
    ///
    /// # Returns
    ///
    /// A vector of launch site records.
    pub async fn launch_site(
        &self,
        site_code: Option<&str>,
    ) -> Result<Vec<LaunchSiteRecord>, SpaceTrackError> {
        let mut req = LaunchSiteRequest::new();

        if let Some(v) = site_code {
            req = req.site_code(parse_query_string(v));
        }

        self.query(&req).await
    }

    /// Query SATCAT change data.
    ///
    /// Returns information about catalog changes.
    ///
    /// # Arguments
    ///
    /// * `norad_cat_id` - Filter by NORAD catalog ID
    /// * `change_made` - Filter by change timestamp
    /// * `limit` - Maximum results
    ///
    /// # Returns
    ///
    /// A vector of SATCAT change records.
    pub async fn satcat_change(
        &self,
        norad_cat_id: Option<u32>,
        change_made: Option<&str>,
        limit: Option<u32>,
    ) -> Result<Vec<SATCATChangeRecord>, SpaceTrackError> {
        let mut req = SATCATChangeRequest::new();

        if let Some(v) = norad_cat_id {
            req = req.norad_cat_id(v);
        }
        if let Some(v) = change_made {
            req = req.change_made(parse_query_string(v));
        }
        if let Some(v) = limit {
            req = req.limit(v);
        }

        self.query(&req).await
    }

    /// Query SATCAT debut (new objects) data.
    ///
    /// Returns information about new catalog entries.
    ///
    /// # Arguments
    ///
    /// * `debut` - Filter by debut date
    /// * `object_type` - Filter by type
    /// * `country` - Filter by country
    /// * `limit` - Maximum results
    ///
    /// # Returns
    ///
    /// A vector of SATCAT debut records.
    pub async fn satcat_debut(
        &self,
        debut: Option<&str>,
        object_type: Option<&str>,
        country: Option<&str>,
        limit: Option<u32>,
    ) -> Result<Vec<SATCATDebutRecord>, SpaceTrackError> {
        let mut req = SATCATDebutRequest::new();

        if let Some(v) = debut {
            req = req.debut(parse_query_string(v));
        }
        if let Some(v) = object_type {
            req = req.object_type(parse_query_string(v));
        }
        if let Some(v) = country {
            req = req.country(parse_query_string(v));
        }
        if let Some(v) = limit {
            req = req.limit(v);
        }

        self.query(&req).await
    }

    /// Query announcement data.
    ///
    /// Returns Space-Track announcements.
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum results
    ///
    /// # Returns
    ///
    /// A vector of announcement records.
    pub async fn announcement(
        &self,
        limit: Option<u32>,
    ) -> Result<Vec<AnnouncementRecord>, SpaceTrackError> {
        let mut req = AnnouncementRequest::new();

        if let Some(v) = limit {
            req = req.limit(v);
        }

        self.query(&req).await
    }

    // ========================================================================
    // New Query Builder API
    // ========================================================================

    /// Execute a query using the new SpaceTrackQuery builder.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The record type to deserialize the response into
    ///
    /// # Arguments
    ///
    /// * `query` - The SpaceTrackQuery to execute
    ///
    /// # Returns
    ///
    /// A vector of records of type T.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use brahe::spacetrack::{SpaceTrackClient, SpaceTrackQuery, SpaceTrackPredicate, GPRecord};
    ///
    /// let query = SpaceTrackQuery::gp()
    ///     .filter(SpaceTrackPredicate::norad_cat_id().eq(25544))
    ///     .limit(10);
    ///
    /// let records: Vec<GPRecord> = client.execute_query(&query).await?;
    /// ```
    pub async fn execute_query<T: DeserializeOwned>(
        &self,
        query: &SpaceTrackQuery,
    ) -> Result<Vec<T>, SpaceTrackError> {
        let predicates = query.build_predicates();
        let predicates_ref: Vec<(&str, QueryValue)> = predicates
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect();

        let text = self
            .generic_request(query.controller(), query.class(), &predicates_ref, None)
            .await?;

        serde_json::from_str(&text)
            .map_err(|e| SpaceTrackError::ParseError(format!("Failed to parse response: {}", e)))
    }

    /// Create a query builder for GP (General Perturbations) records.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use brahe::spacetrack::SpaceTrackClient;
    ///
    /// let records = client.gp_query()
    ///     .filter(SpaceTrackPredicate::norad_cat_id().eq(25544))
    ///     .limit(10)
    ///     .execute()
    ///     .await?;
    /// ```
    pub fn gp_query(&self) -> SpaceTrackQueryBuilder<'_, GPRecord> {
        SpaceTrackQueryBuilder::new(self, SpaceTrackQuery::gp())
    }

    /// Create a query builder for TLE records.
    pub fn tle_query(&self) -> SpaceTrackQueryBuilder<'_, TLERecord> {
        SpaceTrackQueryBuilder::new(self, SpaceTrackQuery::tle())
    }

    /// Create a query builder for SATCAT (Satellite Catalog) records.
    pub fn satcat_query(&self) -> SpaceTrackQueryBuilder<'_, SATCATRecord> {
        SpaceTrackQueryBuilder::new(self, SpaceTrackQuery::satcat())
    }

    /// Create a query builder for OMM records.
    pub fn omm_query(&self) -> SpaceTrackQueryBuilder<'_, OMMRecord> {
        SpaceTrackQueryBuilder::new(self, SpaceTrackQuery::omm())
    }

    /// Create a query builder for GP History records.
    pub fn gp_history_query(&self) -> SpaceTrackQueryBuilder<'_, GPHistoryRecord> {
        SpaceTrackQueryBuilder::new(self, SpaceTrackQuery::gp_history())
    }

    /// Create a query builder for Decay records.
    pub fn decay_query(&self) -> SpaceTrackQueryBuilder<'_, DecayRecord> {
        SpaceTrackQueryBuilder::new(self, SpaceTrackQuery::decay())
    }

    /// Create a query builder for TIP records.
    pub fn tip_query(&self) -> SpaceTrackQueryBuilder<'_, TIPRecord> {
        SpaceTrackQueryBuilder::new(self, SpaceTrackQuery::tip())
    }

    /// Create a query builder for CDM Public records.
    pub fn cdm_public_query(&self) -> SpaceTrackQueryBuilder<'_, CDMPublicRecord> {
        SpaceTrackQueryBuilder::new(self, SpaceTrackQuery::cdm_public())
    }

    /// Create a query builder for Boxscore records.
    pub fn boxscore_query(&self) -> SpaceTrackQueryBuilder<'_, BoxscoreRecord> {
        SpaceTrackQueryBuilder::new(self, SpaceTrackQuery::boxscore())
    }

    /// Create a query builder for Launch Site records.
    pub fn launch_site_query(&self) -> SpaceTrackQueryBuilder<'_, LaunchSiteRecord> {
        SpaceTrackQueryBuilder::new(self, SpaceTrackQuery::launch_site())
    }

    /// Create a query builder for SATCAT Change records.
    pub fn satcat_change_query(&self) -> SpaceTrackQueryBuilder<'_, SATCATChangeRecord> {
        SpaceTrackQueryBuilder::new(self, SpaceTrackQuery::satcat_change())
    }

    /// Create a query builder for SATCAT Debut records.
    pub fn satcat_debut_query(&self) -> SpaceTrackQueryBuilder<'_, SATCATDebutRecord> {
        SpaceTrackQueryBuilder::new(self, SpaceTrackQuery::satcat_debut())
    }

    /// Create a query builder for Announcement records.
    pub fn announcement_query(&self) -> SpaceTrackQueryBuilder<'_, AnnouncementRecord> {
        SpaceTrackQueryBuilder::new(self, SpaceTrackQuery::announcement())
    }

    /// Create a generic query builder for any endpoint.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The record type to deserialize the response into
    ///
    /// # Arguments
    ///
    /// * `controller` - The controller name (e.g., "basicspacedata")
    /// * `class` - The class name (e.g., "gp")
    ///
    /// # Example
    ///
    /// ```ignore
    /// use brahe::spacetrack::{SpaceTrackClient, SpaceTrackPredicate};
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize)]
    /// struct CustomRecord { /* ... */ }
    ///
    /// let records: Vec<CustomRecord> = client
    ///     .query_builder("basicspacedata", "custom_class")
    ///     .filter(SpaceTrackPredicate::field("FIELD").eq("value"))
    ///     .execute()
    ///     .await?;
    /// ```
    pub fn query_builder<T: DeserializeOwned>(
        &self,
        controller: &str,
        class: &str,
    ) -> SpaceTrackQueryBuilder<'_, T> {
        SpaceTrackQueryBuilder::new(self, SpaceTrackQuery::new(controller, class))
    }
}

// ============================================================================
// SpaceTrackQueryBuilder - Fluent query builder
// ============================================================================

/// Fluent query builder for SpaceTrack API requests.
///
/// This struct provides a type-safe, ergonomic API for building and executing
/// SpaceTrack queries. It is parameterized by the record type `T` that will
/// be deserialized from the response.
///
/// # Type Parameters
///
/// * `'a` - Lifetime of the reference to the SpaceTrackClient
/// * `T` - The record type to deserialize responses into
///
/// # Example
///
/// ```ignore
/// use brahe::spacetrack::{SpaceTrackClient, SpaceTrackPredicate};
///
/// let records = client.gp_query()
///     .filter(SpaceTrackPredicate::norad_cat_id().eq(25544))
///     .filter(SpaceTrackPredicate::epoch().gt("2024-01-01"))
///     .order_by_desc(SpaceTrackPredicate::epoch())
///     .limit(10)
///     .execute()
///     .await?;
/// ```
pub struct SpaceTrackQueryBuilder<'a, T> {
    client: &'a SpaceTrackClient,
    query: SpaceTrackQuery,
    _phantom: PhantomData<T>,
}

impl<'a, T: DeserializeOwned> SpaceTrackQueryBuilder<'a, T> {
    /// Create a new query builder.
    pub(crate) fn new(client: &'a SpaceTrackClient, query: SpaceTrackQuery) -> Self {
        Self {
            client,
            query,
            _phantom: PhantomData,
        }
    }

    /// Add a predicate filter to the query.
    ///
    /// Multiple filters are combined with AND logic.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let query = client.gp_query()
    ///     .filter(SpaceTrackPredicate::norad_cat_id().eq(25544))
    ///     .filter(SpaceTrackPredicate::epoch().gt("2024-01-01"));
    /// ```
    pub fn filter(mut self, predicate: SpaceTrackPredicate) -> Self {
        self.query = self.query.filter(predicate);
        self
    }

    /// Set the maximum number of results to return.
    pub fn limit(mut self, limit: u32) -> Self {
        self.query = self.query.limit(limit);
        self
    }

    /// Set ordering by field with specified direction.
    pub fn order_by(mut self, field: SpaceTrackPredicateBuilder, order: SpaceTrackOrder) -> Self {
        self.query = self.query.order_by(field, order);
        self
    }

    /// Set ordering by field in ascending order.
    pub fn order_by_asc(mut self, field: SpaceTrackPredicateBuilder) -> Self {
        self.query = self.query.order_by_asc(field);
        self
    }

    /// Set ordering by field in descending order.
    pub fn order_by_desc(mut self, field: SpaceTrackPredicateBuilder) -> Self {
        self.query = self.query.order_by_desc(field);
        self
    }

    /// Enable distinct results only.
    pub fn distinct(mut self) -> Self {
        self.query = self.query.distinct();
        self
    }

    /// Execute the query and return records.
    ///
    /// # Returns
    ///
    /// A vector of records of type T.
    pub async fn execute(self) -> Result<Vec<T>, SpaceTrackError> {
        self.client.execute_query(&self.query).await
    }
}

/// Additional methods for query builders that return TLE-containing records.
impl<'a, T: DeserializeOwned + HasTLEData> SpaceTrackQueryBuilder<'a, T> {
    /// Execute the query and convert results to SGP propagators.
    ///
    /// # Arguments
    ///
    /// * `step_size` - The propagator step size in seconds
    ///
    /// # Returns
    ///
    /// A vector of SGPPropagator instances.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let propagators = client.gp_query()
    ///     .filter(SpaceTrackPredicate::norad_cat_id().eq(25544))
    ///     .as_sgp_propagators(60.0)
    ///     .await?;
    /// ```
    pub async fn as_sgp_propagators(
        self,
        step_size: f64,
    ) -> Result<Vec<SGPPropagator>, BraheError> {
        let records: Vec<T> = self.execute().await?;
        records
            .iter()
            .map(|r| r.to_sgp_propagator(step_size))
            .collect()
    }

    /// Execute the query and convert results to SGP propagators, skipping errors.
    ///
    /// # Arguments
    ///
    /// * `step_size` - The propagator step size in seconds
    ///
    /// # Returns
    ///
    /// A vector of SGPPropagator instances. Records that fail to convert
    /// are silently skipped.
    pub async fn as_sgp_propagators_skip_errors(self, step_size: f64) -> Vec<SGPPropagator> {
        match self.execute().await {
            Ok(records) => records
                .iter()
                .filter_map(|r| r.to_sgp_propagator(step_size).ok())
                .collect(),
            Err(_) => Vec::new(),
        }
    }
}

/// Parse a query string value, handling operators.
///
/// Supports operators like:
/// - `>value` - greater than
/// - `<value` - less than
/// - `<>value` or `><value` - not equal
/// - `~~pattern` - like (pattern match)
/// - `^prefix` - starts with
/// - `start--end` - inclusive range
/// - `null-val` - null value
fn parse_query_string(value: &str) -> QueryValue {
    let trimmed = value.trim();

    // Check for operators
    if let Some(rest) = trimmed
        .strip_prefix("><")
        .or_else(|| trimmed.strip_prefix("<>"))
    {
        not_equal(rest)
    } else if let Some(rest) = trimmed.strip_prefix(">=") {
        // Greater than or equal - use range to "now"
        inclusive_range(rest, "now")
    } else if let Some(rest) = trimmed.strip_prefix("<=") {
        // Less than or equal - use range from earliest possible
        inclusive_range("0000-01-01", rest)
    } else if let Some(rest) = trimmed.strip_prefix('>') {
        greater_than(rest)
    } else if let Some(rest) = trimmed.strip_prefix('<') {
        less_than(rest)
    } else if let Some(rest) = trimmed.strip_prefix("~~") {
        like(rest)
    } else if let Some(rest) = trimmed.strip_prefix('^') {
        startswith(rest)
    } else if trimmed.contains("--") {
        let parts: Vec<&str> = trimmed.split("--").collect();
        if parts.len() == 2 {
            inclusive_range(parts[0], parts[1])
        } else {
            equals(trimmed)
        }
    } else if trimmed == "null-val" {
        null_val()
    } else {
        equals(trimmed)
    }
}

/// Trait for request types that support orderby.
trait OrderByMethods: Sized {
    fn set_orderby_asc(&mut self, field: &str);
    fn set_orderby_desc(&mut self, field: &str);
}

/// Apply orderby to a request.
fn apply_orderby<R: OrderByMethods>(mut req: R, orderby: &str) -> R {
    let lower = orderby.to_lowercase();
    if lower.ends_with(" desc") {
        let field = orderby[..orderby.len() - 5].trim();
        req.set_orderby_desc(field);
    } else if lower.ends_with(" asc") {
        let field = orderby[..orderby.len() - 4].trim();
        req.set_orderby_asc(field);
    } else {
        req.set_orderby_asc(orderby);
    }
    req
}

// Implement OrderByMethods for request types that support it
macro_rules! impl_orderby {
    ($($t:ty),*) => {
        $(
            impl OrderByMethods for $t {
                fn set_orderby_asc(&mut self, field: &str) {
                    *self = std::mem::take(self).orderby_asc(field);
                }
                fn set_orderby_desc(&mut self, field: &str) {
                    *self = std::mem::take(self).orderby_desc(field);
                }
            }
        )*
    };
}

impl_orderby!(GPRequest, SATCATRequest, TLERequest);

/// Parse 3LE format text into SGP propagators.
///
/// # Arguments
///
/// * `text` - The 3LE format text (name + 2 TLE lines per satellite)
/// * `step_size` - The propagator step size in seconds
///
/// # Returns
///
/// A vector of `SGPPropagator` instances.
pub fn parse_3le_to_propagators(
    text: &str,
    step_size: f64,
) -> Result<Vec<SGPPropagator>, BraheError> {
    let mut propagators = Vec::new();
    let lines: Vec<&str> = text.lines().collect();

    let mut i = 0;
    while i < lines.len() {
        // Skip empty lines
        if lines[i].trim().is_empty() {
            i += 1;
            continue;
        }

        // Check for 3LE format (name + 2 TLE lines)
        if i + 2 < lines.len()
            && !lines[i].starts_with('1')
            && !lines[i].starts_with('2')
            && lines[i + 1].starts_with('1')
            && lines[i + 2].starts_with('2')
        {
            let name = lines[i].trim();
            let line1 = lines[i + 1].trim();
            let line2 = lines[i + 2].trim();

            let propagator = SGPPropagator::from_3le(Some(name), line1, line2, step_size)?;
            propagators.push(propagator);
            i += 3;
        }
        // Check for 2LE format (just 2 TLE lines)
        else if i + 1 < lines.len() && lines[i].starts_with('1') && lines[i + 1].starts_with('2')
        {
            let line1 = lines[i].trim();
            let line2 = lines[i + 1].trim();

            let propagator = SGPPropagator::from_tle(line1, line2, step_size)?;
            propagators.push(propagator);
            i += 2;
        } else {
            // Skip unrecognized line
            i += 1;
        }
    }

    Ok(propagators)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::utils::testing::setup_global_test_eop;

    // Valid TLE data for testing
    const ISS_LINE1: &str = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    const ISS_LINE2: &str = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

    #[test]
    fn test_default_base_url() {
        assert_eq!(DEFAULT_BASE_URL, "https://www.space-track.org/");
    }

    #[test]
    fn test_test_base_url() {
        assert_eq!(TEST_BASE_URL, "https://for-testing-only.space-track.org/");
    }

    #[test]
    fn test_new_unauthenticated() {
        let client = SpaceTrackClient::new_unauthenticated("user", "pass", DEFAULT_BASE_URL);
        assert!(!client.is_authenticated());
        assert_eq!(client.base_url(), DEFAULT_BASE_URL);
    }

    #[test]
    fn test_parse_3le_to_propagators() {
        setup_global_test_eop();
        let tle_data = format!("ISS (ZARYA)\n{}\n{}", ISS_LINE1, ISS_LINE2);

        let result = parse_3le_to_propagators(&tle_data, 60.0);
        assert!(result.is_ok());
        let propagators = result.unwrap();
        assert_eq!(propagators.len(), 1);
    }

    #[test]
    fn test_parse_2le_to_propagators() {
        setup_global_test_eop();
        let tle_data = format!("{}\n{}", ISS_LINE1, ISS_LINE2);

        let result = parse_3le_to_propagators(&tle_data, 60.0);
        assert!(result.is_ok());
        let propagators = result.unwrap();
        assert_eq!(propagators.len(), 1);
    }

    #[test]
    fn test_parse_empty_3le() {
        let result = parse_3le_to_propagators("", 60.0);
        assert!(result.is_ok());
        let propagators = result.unwrap();
        assert!(propagators.is_empty());
    }

    #[test]
    fn test_parse_3le_with_empty_lines() {
        setup_global_test_eop();
        let tle_data = format!("\nISS (ZARYA)\n{}\n{}\n\n", ISS_LINE1, ISS_LINE2);

        let result = parse_3le_to_propagators(&tle_data, 60.0);
        assert!(result.is_ok());
        let propagators = result.unwrap();
        assert_eq!(propagators.len(), 1);
    }
}
