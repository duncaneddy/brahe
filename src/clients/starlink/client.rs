/*!
 * Blocking client for Starlink's public ephemeris mirror.
 */

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

use crate::clients::spacetrack::EphemerisFileName;
use crate::clients::starlink::STARLINK_RATE_LIMIT;
use crate::clients::starlink::manifest::{
    StarlinkManifest, StarlinkManifestEntry, parse_http_date,
};
use crate::clients::{RateLimitConfig, RateLimiter};
use crate::frames::OrbitRelativeFrameVariant;
use crate::itc::ITC;
use crate::time::Epoch;
use crate::trajectories::DOrbitTrajectory;
use crate::utils::download::{backoff_delay, is_retryable_error};
use crate::utils::network::{CacheDecision, cache_policy, ensure_online};
use crate::utils::{BraheError, atomic_write, get_starlink_cache_dir};

pub(crate) const DEFAULT_BASE_URL: &str = "https://api.starlink.com/public-files/ephemerides";
pub(crate) const DEFAULT_MAX_CACHE_AGE: f64 = 3600.0;
const DEFAULT_MAX_RETRIES: u32 = 3;
pub(crate) const MANIFEST_FILE: &str = "MANIFEST.txt";
pub(crate) const PREVIOUS_MANIFEST_FILE: &str = "MANIFEST.previous.txt";
const MANIFEST_META_FILE: &str = "MANIFEST.meta.json";

/// Validators stored beside the cached manifest so a later refresh can send a
/// conditional GET and so `retrieved` survives a process restart.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct ManifestMeta {
    /// `ETag` reported by the server for the cached body, when known.
    etag: Option<String>,
    /// `Last-Modified` reported by the server for the cached body, when known.
    last_modified: Option<String>,
    /// When the cached body was retrieved, as an [`Epoch`] display string.
    retrieved: Option<String>,
}

/// Result of a conditional GET against the manifest URL.
enum FetchOutcome {
    /// New content with its validators.
    Fresh {
        /// Response body.
        body: String,
        /// `ETag` response header, when present.
        etag: Option<String>,
        /// `Last-Modified` response header, when present.
        last_modified: Option<String>,
    },
    /// Server answered 304; the cached copy is current.
    NotModified,
}

/// Client for Starlink's public Modified ITC ephemerides.
///
/// Requests are blocking, rate limited (1000 per minute and 30000 per hour by
/// default), retried on transient failures, and honour `BRAHE_NETWORK_MODE`.
/// The manifest is cached under `$BRAHE_CACHE/starlink/MANIFEST.txt` and
/// re-fetched with a conditional GET once it is older than `cache_max_age`;
/// when the listing changes, the prior copy is kept as
/// `MANIFEST.previous.txt` so the caller can ask which satellites moved.
///
/// # Examples
///
/// ```no_run
/// use brahe::starlink::StarlinkClient;
///
/// let client = StarlinkClient::new();
/// let manifest = client.get_manifest().unwrap();
/// println!("{} satellites listed", manifest.len());
/// ```
pub struct StarlinkClient {
    base_url: String,
    cache_max_age: f64,
    max_retries: u32,
    agent: ureq::Agent,
    rate_limiter: Mutex<RateLimiter>,
}

impl Default for StarlinkClient {
    fn default() -> Self {
        Self::new()
    }
}

impl StarlinkClient {
    /// Creates a client for the public mirror with the default one-hour manifest cache.
    ///
    /// # Returns
    /// * `StarlinkClient`: Client with default settings
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::starlink::StarlinkClient;
    ///
    /// let client = StarlinkClient::new();
    /// assert_eq!(client.cache_max_age(), 3600.0);
    /// ```
    pub fn new() -> Self {
        Self::with_base_url_and_cache_age(DEFAULT_BASE_URL, DEFAULT_MAX_CACHE_AGE)
    }

    /// Creates a client with a custom manifest cache age.
    ///
    /// # Arguments
    /// * `cache_max_age` - Seconds a cached manifest is served without a refresh
    ///
    /// # Returns
    /// * `StarlinkClient`: Client using the public mirror
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::starlink::StarlinkClient;
    ///
    /// let client = StarlinkClient::with_cache_age(600.0);
    /// assert_eq!(client.cache_max_age(), 600.0);
    /// ```
    pub fn with_cache_age(cache_max_age: f64) -> Self {
        Self::with_base_url_and_cache_age(DEFAULT_BASE_URL, cache_max_age)
    }

    /// Creates a client against a different base URL, for mirrors and tests.
    ///
    /// # Arguments
    /// * `base_url` - Directory URL holding `MANIFEST.txt` and the ephemeris files
    ///
    /// # Returns
    /// * `StarlinkClient`: Client with the default cache age
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::starlink::StarlinkClient;
    ///
    /// let client = StarlinkClient::with_base_url("http://127.0.0.1:8080/ephemerides/");
    /// assert_eq!(client.base_url(), "http://127.0.0.1:8080/ephemerides");
    /// ```
    pub fn with_base_url(base_url: &str) -> Self {
        Self::with_base_url_and_cache_age(base_url, DEFAULT_MAX_CACHE_AGE)
    }

    /// Creates a client with a custom base URL and manifest cache age.
    ///
    /// # Arguments
    /// * `base_url` - Directory URL holding `MANIFEST.txt` and the ephemeris files
    /// * `cache_max_age` - Seconds a cached manifest is served without a refresh
    ///
    /// # Returns
    /// * `StarlinkClient`: Configured client
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::starlink::StarlinkClient;
    ///
    /// let client = StarlinkClient::with_base_url_and_cache_age("http://127.0.0.1:8080", 60.0);
    /// assert_eq!(client.cache_max_age(), 60.0);
    /// ```
    pub fn with_base_url_and_cache_age(base_url: &str, cache_max_age: f64) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            cache_max_age,
            max_retries: DEFAULT_MAX_RETRIES,
            agent: ureq::Agent::new_with_defaults(),
            rate_limiter: Mutex::new(RateLimiter::new(STARLINK_RATE_LIMIT)),
        }
    }

    /// Creates a client for the public mirror with custom request limits.
    ///
    /// # Arguments
    /// * `config` - Per-minute and per-hour request caps
    ///
    /// # Returns
    /// * `StarlinkClient`: Client with the default cache age
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::clients::RateLimitConfig;
    /// use brahe::starlink::StarlinkClient;
    ///
    /// let client = StarlinkClient::with_rate_limit(RateLimitConfig { max_per_minute: 60, max_per_hour: 1000 });
    /// assert_eq!(client.cache_max_age(), 3600.0);
    /// ```
    pub fn with_rate_limit(config: RateLimitConfig) -> Self {
        let mut client = Self::new();
        client.rate_limiter = Mutex::new(RateLimiter::new(config));
        client
    }

    /// Sets the number of retries for transient failures.
    ///
    /// # Arguments
    /// * `max_retries` - Extra attempts after the first (default 3)
    ///
    /// # Returns
    /// * `StarlinkClient`: The client, for chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::starlink::StarlinkClient;
    ///
    /// let client = StarlinkClient::new().max_retries(0);
    /// assert_eq!(client.cache_max_age(), 3600.0);
    /// ```
    pub fn max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Base URL without a trailing slash.
    ///
    /// # Returns
    /// * `&str`: The directory URL
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::starlink::StarlinkClient;
    ///
    /// assert!(StarlinkClient::new().base_url().starts_with("https://api.starlink.com"));
    /// ```
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Manifest cache age in seconds.
    ///
    /// # Returns
    /// * `f64`: Seconds
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::starlink::StarlinkClient;
    ///
    /// assert_eq!(StarlinkClient::new().cache_max_age(), 3600.0);
    /// ```
    pub fn cache_max_age(&self) -> f64 {
        self.cache_max_age
    }

    /// Directory holding the cached manifest and ephemeris files.
    ///
    /// # Returns
    /// * `Ok(PathBuf)`: `$BRAHE_CACHE/starlink`, created if missing
    /// * `Err(BraheError)`: If the cache directory cannot be created
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::starlink::StarlinkClient;
    ///
    /// let dir = StarlinkClient::new().cache_dir().unwrap();
    /// assert!(dir.ends_with("starlink"));
    /// ```
    pub fn cache_dir(&self) -> Result<PathBuf, BraheError> {
        get_starlink_cache_dir().map(PathBuf::from)
    }

    /// Returns the manifest, serving the cached copy while it is younger than
    /// `cache_max_age` and refreshing it with a conditional GET otherwise.
    ///
    /// `BRAHE_NETWORK_MODE` applies as for the other clients: `offline` serves
    /// a cached manifest of any age, `offline-strict` serves only a fresh one
    /// and rejects a stale or missing one, and a refresh that fails is an
    /// error rather than a silent fall back to the stale copy.
    ///
    /// # Returns
    /// * `Ok(StarlinkManifest)`: The current listing
    /// * `Err(BraheError)`: If the cached copy cannot be served under the current mode and no refresh succeeds
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::starlink::StarlinkClient;
    ///
    /// let manifest = StarlinkClient::new().get_manifest().unwrap();
    /// assert!(!manifest.is_empty());
    /// ```
    pub fn get_manifest(&self) -> Result<StarlinkManifest, BraheError> {
        let dir = self.cache_dir()?;
        let path = dir.join(MANIFEST_FILE);
        if path.exists() {
            let stale = self.is_cache_stale(&path)?;
            if cache_policy("Starlink MANIFEST.txt", stale)? == CacheDecision::Serve {
                return self.read_cached_manifest(&dir);
            }
        }
        self.refresh_manifest()
    }

    /// Fetches the manifest from the server regardless of cache age.
    ///
    /// Sends the cached ETag; a 304 answer touches the cached file so it is
    /// fresh again. A changed listing moves the old copy to
    /// `MANIFEST.previous.txt` before the new one is written.
    ///
    /// # Returns
    /// * `Ok(StarlinkManifest)`: The listing now on disk
    /// * `Err(BraheError)`: On network failure, offline mode, or a malformed listing
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::starlink::StarlinkClient;
    ///
    /// let manifest = StarlinkClient::new().refresh_manifest().unwrap();
    /// println!("{}", manifest.len());
    /// ```
    pub fn refresh_manifest(&self) -> Result<StarlinkManifest, BraheError> {
        let dir = self.cache_dir()?;
        let path = dir.join(MANIFEST_FILE);
        let meta = self.read_meta(&dir);
        let url = format!("{}/{}", self.base_url, MANIFEST_FILE);
        let etag = if path.exists() {
            meta.etag.as_deref()
        } else {
            None
        };
        match self.fetch_text(&url, etag)? {
            FetchOutcome::NotModified => {
                touch(&path)?;
                self.write_meta(
                    &dir,
                    &ManifestMeta {
                        retrieved: Some(now_rounded().to_string()),
                        ..meta
                    },
                )?;
                self.read_cached_manifest(&dir)
            }
            FetchOutcome::Fresh {
                body,
                etag,
                last_modified,
            } => {
                let retrieved = now_rounded();
                let manifest = StarlinkManifest::parse(
                    &body,
                    retrieved,
                    last_modified.as_deref().and_then(parse_http_date),
                )?;
                let old = path
                    .exists()
                    .then(|| fs::read_to_string(&path))
                    .transpose()
                    .map_err(|e| {
                        BraheError::IoError(format!("Failed to read cached Starlink manifest: {e}"))
                    })?;
                if let Some(old) = old
                    && old != body
                {
                    atomic_write(&dir.join(PREVIOUS_MANIFEST_FILE), old.as_bytes()).map_err(
                        |e| {
                            BraheError::IoError(format!(
                                "Failed to write previous Starlink manifest: {e}"
                            ))
                        },
                    )?;
                }
                atomic_write(&path, body.as_bytes()).map_err(|e| {
                    BraheError::IoError(format!("Failed to write Starlink manifest: {e}"))
                })?;
                self.write_meta(
                    &dir,
                    &ManifestMeta {
                        etag,
                        last_modified,
                        retrieved: Some(retrieved.to_string()),
                    },
                )?;
                Ok(manifest)
            }
        }
    }

    /// The cached manifest, whatever its age, without touching the network.
    ///
    /// # Returns
    /// * `Ok(Some(StarlinkManifest))`: The cached listing
    /// * `Ok(None)`: If nothing is cached
    /// * `Err(BraheError)`: If the cached file cannot be read or parsed
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::starlink::StarlinkClient;
    ///
    /// let cached = StarlinkClient::new().cached_manifest().unwrap();
    /// println!("{}", cached.map(|m| m.len()).unwrap_or(0));
    /// ```
    pub fn cached_manifest(&self) -> Result<Option<StarlinkManifest>, BraheError> {
        let dir = self.cache_dir()?;
        if dir.join(MANIFEST_FILE).exists() {
            self.read_cached_manifest(&dir).map(Some)
        } else {
            Ok(None)
        }
    }

    /// The listing that was current before the last change, for `changed_since`.
    ///
    /// # Returns
    /// * `Ok(Some(StarlinkManifest))`: The previous listing
    /// * `Ok(None)`: If the manifest has never changed on this machine
    /// * `Err(BraheError)`: If the file cannot be read or parsed
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::starlink::StarlinkClient;
    ///
    /// let previous = StarlinkClient::new().previous_manifest().unwrap();
    /// println!("{}", previous.is_some());
    /// ```
    pub fn previous_manifest(&self) -> Result<Option<StarlinkManifest>, BraheError> {
        let dir = self.cache_dir()?;
        let path = dir.join(PREVIOUS_MANIFEST_FILE);
        if !path.exists() {
            return Ok(None);
        }
        let text = fs::read_to_string(&path).map_err(|e| {
            BraheError::IoError(format!("Failed to read previous Starlink manifest: {e}"))
        })?;
        let retrieved = file_epoch(&path)?;
        StarlinkManifest::parse(&text, retrieved, None).map(Some)
    }

    /// Reads and parses the cached `MANIFEST.txt`, using the sidecar's
    /// `retrieved` timestamp when present and the file's modification time
    /// otherwise.
    ///
    /// # Arguments
    /// * `dir` - The Starlink cache directory
    ///
    /// # Returns
    /// * `Ok(StarlinkManifest)`: The parsed listing
    /// * `Err(BraheError)`: If the file cannot be read or parsed
    fn read_cached_manifest(&self, dir: &Path) -> Result<StarlinkManifest, BraheError> {
        let path = dir.join(MANIFEST_FILE);
        let text = fs::read_to_string(&path).map_err(|e| {
            BraheError::IoError(format!("Failed to read cached Starlink manifest: {e}"))
        })?;
        let meta = self.read_meta(dir);
        let retrieved = match meta.retrieved.as_deref().and_then(Epoch::from_string) {
            Some(retrieved) => retrieved,
            None => file_epoch(&path)?,
        };
        let last_modified = meta.last_modified.as_deref().and_then(parse_http_date);
        StarlinkManifest::parse(&text, retrieved, last_modified)
    }

    /// Reads the manifest sidecar, treating a missing or unparsable file as
    /// carrying no validators.
    ///
    /// # Arguments
    /// * `dir` - The Starlink cache directory
    ///
    /// # Returns
    /// * `ManifestMeta`: The stored validators, or the default (all `None`)
    fn read_meta(&self, dir: &Path) -> ManifestMeta {
        fs::read_to_string(dir.join(MANIFEST_META_FILE))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Writes the manifest sidecar atomically.
    ///
    /// # Arguments
    /// * `dir` - The Starlink cache directory
    /// * `meta` - Validators to persist
    ///
    /// # Returns
    /// * `Ok(())`: The sidecar was written
    /// * `Err(BraheError)`: If serialisation or the write fails
    fn write_meta(&self, dir: &Path, meta: &ManifestMeta) -> Result<(), BraheError> {
        let text = serde_json::to_string(meta).map_err(|e| {
            BraheError::Error(format!(
                "Failed to serialise Starlink manifest metadata: {e}"
            ))
        })?;
        atomic_write(&dir.join(MANIFEST_META_FILE), text.as_bytes()).map_err(|e| {
            BraheError::IoError(format!("Failed to write Starlink manifest metadata: {e}"))
        })
    }

    /// Whether a cached file is older than `cache_max_age`.
    ///
    /// # Arguments
    /// * `path` - The cached file to check
    ///
    /// # Returns
    /// * `Ok(bool)`: `true` when the file's age exceeds `cache_max_age` seconds
    /// * `Err(BraheError)`: If the file's modification time cannot be read
    fn is_cache_stale(&self, path: &Path) -> Result<bool, BraheError> {
        let modified = fs::metadata(path).and_then(|m| m.modified()).map_err(|e| {
            BraheError::IoError(format!("Failed to read file modification time: {e}"))
        })?;
        Ok(SystemTime::now()
            .duration_since(modified)
            .unwrap_or_default()
            .as_secs_f64()
            > self.cache_max_age)
    }

    /// Blocks until the shared rate limiter admits a request to `url`,
    /// rejecting the request outright when `BRAHE_NETWORK_MODE` forbids it.
    ///
    /// # Arguments
    /// * `url` - The URL the caller is about to request
    ///
    /// # Returns
    /// * `Ok(())`: The request may proceed
    /// * `Err(BraheError)`: If the network mode rejects the request
    fn wait_for_rate_limit(&self, url: &str) -> Result<(), BraheError> {
        ensure_online(url, &format!("Starlink request {url}"))?;
        let wait = {
            let mut limiter = self.rate_limiter.lock().map_err(|e| {
                BraheError::Error(format!("Failed to acquire lock on rate limiter: {e}"))
            })?;
            limiter.acquire()
        };
        if wait > Duration::ZERO {
            std::thread::sleep(wait);
        }
        Ok(())
    }

    /// GETs `url`, optionally conditional on `etag`, retrying transient
    /// failures with backoff up to `max_retries` times.
    ///
    /// # Arguments
    /// * `url` - The URL to fetch
    /// * `etag` - Cached `ETag` to send as `If-None-Match`, when known
    ///
    /// # Returns
    /// * `Ok(FetchOutcome::Fresh)`: A new body and its validators
    /// * `Ok(FetchOutcome::NotModified)`: The server answered 304
    /// * `Err(BraheError)`: If the network mode forbids the request or every attempt fails
    fn fetch_text(&self, url: &str, etag: Option<&str>) -> Result<FetchOutcome, BraheError> {
        let mut last_error = None;
        for attempt in 0..=self.max_retries {
            self.wait_for_rate_limit(url)?;
            if attempt > 0 {
                std::thread::sleep(backoff_delay(attempt));
            }
            let mut request = self.agent.get(url);
            if let Some(tag) = etag {
                request = request.header("If-None-Match", tag);
            }
            match request.call() {
                Ok(mut response) => {
                    if response.status() == 304 {
                        return Ok(FetchOutcome::NotModified);
                    }
                    let etag = response
                        .headers()
                        .get("etag")
                        .and_then(|v| v.to_str().ok())
                        .map(str::to_string);
                    let last_modified = response
                        .headers()
                        .get("last-modified")
                        .and_then(|v| v.to_str().ok())
                        .map(str::to_string);
                    let body = response.body_mut().read_to_string().map_err(|e| {
                        BraheError::IoError(format!(
                            "Failed to read Starlink response for {url}: {e}"
                        ))
                    })?;
                    return Ok(FetchOutcome::Fresh {
                        body,
                        etag,
                        last_modified,
                    });
                }
                Err(e) => {
                    if attempt < self.max_retries && is_retryable_error(&e) {
                        last_error = Some(e);
                        continue;
                    }
                    return Err(BraheError::IoError(format!(
                        "Starlink request {url} failed: {e}"
                    )));
                }
            }
        }
        Err(BraheError::IoError(format!(
            "Starlink request {url} failed: {}",
            last_error.unwrap()
        )))
    }

    /// Downloads a satellite's current ephemeris file into the cache.
    ///
    /// The manifest names one file per satellite; if that file is already
    /// cached no request is made. After a download every other cached file
    /// for the same NORAD ID is deleted.
    ///
    /// # Arguments
    /// * `norad_cat_id` - NORAD catalog number
    ///
    /// # Returns
    /// * `Ok(PathBuf)`: Path of the cached file
    /// * `Err(BraheError)`: If the ID is not listed, the download fails, or the body is not a valid ITC file
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::starlink::StarlinkClient;
    ///
    /// let path = StarlinkClient::new().download_ephemeris(100001).unwrap();
    /// println!("{}", path.display());
    /// ```
    pub fn download_ephemeris(&self, norad_cat_id: u32) -> Result<PathBuf, BraheError> {
        let manifest = self.get_manifest()?;
        let entry = manifest.find_by_norad_id(norad_cat_id).ok_or_else(|| {
            BraheError::Error(format!(
                "NORAD ID {} is not in the Starlink manifest",
                norad_cat_id
            ))
        })?;
        self.download_entry(entry)
    }

    /// Downloads one manifest entry's file into the cache if it is not
    /// already present, then evicts any other cached file for the same
    /// NORAD ID.
    ///
    /// # Arguments
    /// * `entry` - Manifest entry naming the file to download
    ///
    /// # Returns
    /// * `Ok(PathBuf)`: Path of the cached file
    /// * `Err(BraheError)`: If the download fails or the body is not a valid ITC file
    fn download_entry(&self, entry: &StarlinkManifestEntry) -> Result<PathBuf, BraheError> {
        let dir = self.cache_dir()?;
        let name = entry.file_name_string();
        let path = dir.join(&name);
        if path.exists() {
            return Ok(path);
        }
        let url = format!("{}/{}", self.base_url, name);
        let body = match self.fetch_text(&url, None)? {
            FetchOutcome::Fresh { body, .. } => body,
            FetchOutcome::NotModified => {
                return Err(BraheError::IoError(format!(
                    "Starlink request {} returned 304 without a validator",
                    url
                )));
            }
        };
        ITC::from_str(&body).map_err(|e| {
            BraheError::ParseError(format!(
                "Starlink file {} is not a valid ITC message: {}",
                name, e
            ))
        })?;
        atomic_write(&path, body.as_bytes()).map_err(|e| {
            BraheError::IoError(format!("Failed to write {}: {}", path.display(), e))
        })?;
        for other in self.cached_files()? {
            if other != path
                && let Some(parsed) = other
                    .file_name()
                    .and_then(|n| n.to_str())
                    .and_then(|n| EphemerisFileName::parse(n).ok())
                && parsed.norad_cat_id == entry.norad_cat_id
            {
                fs::remove_file(&other).map_err(|e| {
                    BraheError::IoError(format!("Failed to delete {}: {}", other.display(), e))
                })?;
            }
        }
        Ok(path)
    }

    /// Downloads (if needed) and parses a satellite's current ephemeris.
    ///
    /// # Arguments
    /// * `norad_cat_id` - NORAD catalog number
    ///
    /// # Returns
    /// * `Ok(ITC)`: The parsed message in SI units
    /// * `Err(BraheError)`: See [`StarlinkClient::download_ephemeris`]
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::starlink::StarlinkClient;
    ///
    /// let itc = StarlinkClient::new().get_ephemeris(100001).unwrap();
    /// println!("{} records", itc.len());
    /// ```
    pub fn get_ephemeris(&self, norad_cat_id: u32) -> Result<ITC, BraheError> {
        ITC::from_file(self.download_ephemeris(norad_cat_id)?)
    }

    /// Downloads (if needed) a satellite's ephemeris as an `OrbitTrajectory`
    /// in the file's state frame with covariance rotated from RTN using the
    /// block-diagonal (inertial) convention.
    ///
    /// # Arguments
    /// * `norad_cat_id` - NORAD catalog number
    ///
    /// # Returns
    /// * `Ok(DOrbitTrajectory)`: Six-dimensional Cartesian trajectory
    /// * `Err(BraheError)`: See [`StarlinkClient::download_ephemeris`]
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::starlink::StarlinkClient;
    /// use brahe::traits::Trajectory;
    ///
    /// let traj = StarlinkClient::new().get_trajectory(100001).unwrap();
    /// println!("{} samples", traj.len());
    /// ```
    pub fn get_trajectory(&self, norad_cat_id: u32) -> Result<DOrbitTrajectory, BraheError> {
        self.get_ephemeris(norad_cat_id)?.to_trajectory()
    }

    /// Like [`StarlinkClient::get_trajectory`] with an explicit RTN covariance convention.
    ///
    /// # Arguments
    /// * `norad_cat_id` - NORAD catalog number
    /// * `variant` - `Inertial` (block-diagonal) or `Rotating` RTN rotation
    ///
    /// # Returns
    /// * `Ok(DOrbitTrajectory)`: Six-dimensional Cartesian trajectory
    /// * `Err(BraheError)`: See [`StarlinkClient::download_ephemeris`]
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::frames::OrbitRelativeFrameVariant;
    /// use brahe::starlink::StarlinkClient;
    /// use brahe::traits::Trajectory;
    ///
    /// let traj = StarlinkClient::new()
    ///     .get_trajectory_with_covariance_variant(100001, OrbitRelativeFrameVariant::Rotating)
    ///     .unwrap();
    /// println!("{} samples", traj.len());
    /// ```
    pub fn get_trajectory_with_covariance_variant(
        &self,
        norad_cat_id: u32,
        variant: OrbitRelativeFrameVariant,
    ) -> Result<DOrbitTrajectory, BraheError> {
        self.get_ephemeris(norad_cat_id)?
            .to_trajectory_with_covariance_variant(variant)
    }

    /// Downloads (if needed) a satellite's ephemeris and copies it to `destination`.
    ///
    /// An existing directory, or a path whose last component has no
    /// extension, is treated as a directory (created if missing) and the
    /// original file name is kept; a path with an extension is the file name.
    /// Saved copies are outside the cache and never evicted.
    ///
    /// # Arguments
    /// * `norad_cat_id` - NORAD catalog number
    /// * `destination` - Directory or file path
    ///
    /// # Returns
    /// * `Ok(PathBuf)`: Path written
    /// * `Err(BraheError)`: On download or filesystem failure
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::starlink::StarlinkClient;
    ///
    /// let path = StarlinkClient::new()
    ///     .save_ephemeris(100001, "./ephemerides")
    ///     .unwrap();
    /// println!("{}", path.display());
    /// ```
    pub fn save_ephemeris<P: AsRef<Path>>(
        &self,
        norad_cat_id: u32,
        destination: P,
    ) -> Result<PathBuf, BraheError> {
        let source = self.download_ephemeris(norad_cat_id)?;
        let name = source
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string();
        let target = resolve_destination(destination.as_ref(), &name)?;
        copy_file(&source, &target)?;
        Ok(target)
    }

    /// Downloads every file the manifest lists that is not already cached,
    /// using up to `concurrency` threads that share the rate limiter.
    ///
    /// Stops at the first failure and returns it. Files already cached are
    /// not re-fetched and no cached files are removed; call
    /// [`StarlinkClient::prune_cache`] for that.
    ///
    /// # Arguments
    /// * `concurrency` - Worker threads, at least 1 (8 is a reasonable default)
    ///
    /// # Returns
    /// * `Ok(Vec<PathBuf>)`: Cache path of every listed file, in manifest order
    /// * `Err(BraheError)`: The first failure, or `concurrency == 0`
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::starlink::StarlinkClient;
    ///
    /// let paths = StarlinkClient::new().download_all(8).unwrap();
    /// println!("{} files cached", paths.len());
    /// ```
    pub fn download_all(&self, concurrency: usize) -> Result<Vec<PathBuf>, BraheError> {
        if concurrency == 0 {
            return Err(BraheError::Error(
                "download_all requires concurrency >= 1".to_string(),
            ));
        }
        let manifest = self.get_manifest()?;
        let dir = self.cache_dir()?;
        let pending: Vec<&StarlinkManifestEntry> = manifest
            .iter()
            .filter(|e| !dir.join(e.file_name_string()).exists())
            .collect();
        let cursor = AtomicUsize::new(0);
        let stop = AtomicBool::new(false);
        let failure: Mutex<Option<BraheError>> = Mutex::new(None);
        std::thread::scope(|scope| {
            for _ in 0..concurrency.min(pending.len()) {
                scope.spawn(|| {
                    loop {
                        if stop.load(Ordering::SeqCst) {
                            break;
                        }
                        let index = cursor.fetch_add(1, Ordering::SeqCst);
                        let Some(entry) = pending.get(index) else {
                            break;
                        };
                        if let Err(e) = self.download_entry(entry) {
                            stop.store(true, Ordering::SeqCst);
                            if let Ok(mut slot) = failure.lock()
                                && slot.is_none()
                            {
                                *slot = Some(e);
                            }
                            break;
                        }
                    }
                });
            }
        });
        if let Some(e) = failure.into_inner().unwrap_or(None) {
            return Err(e);
        }
        Ok(manifest
            .iter()
            .map(|e| dir.join(e.file_name_string()))
            .collect())
    }

    /// Runs [`StarlinkClient::download_all`] and copies every file into `destination`.
    ///
    /// # Arguments
    /// * `destination` - Directory, created if missing
    /// * `concurrency` - Worker threads, at least 1
    ///
    /// # Returns
    /// * `Ok(Vec<PathBuf>)`: Paths written, in manifest order
    /// * `Err(BraheError)`: On download failure or if `destination` is an existing file
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::starlink::StarlinkClient;
    ///
    /// let paths = StarlinkClient::new().save_all("./ephemerides", 8).unwrap();
    /// println!("{} files saved", paths.len());
    /// ```
    pub fn save_all<P: AsRef<Path>>(
        &self,
        destination: P,
        concurrency: usize,
    ) -> Result<Vec<PathBuf>, BraheError> {
        let destination = destination.as_ref();
        if destination.is_file() {
            return Err(BraheError::Error(format!(
                "save_all destination {} is a file, expected a directory",
                destination.display()
            )));
        }
        fs::create_dir_all(destination).map_err(|e| {
            BraheError::IoError(format!("Failed to create {}: {}", destination.display(), e))
        })?;
        let mut written = Vec::new();
        for source in self.download_all(concurrency)? {
            let target = destination.join(source.file_name().unwrap_or_default());
            copy_file(&source, &target)?;
            written.push(target);
        }
        Ok(written)
    }

    /// Deletes cached ephemeris files the current manifest no longer lists.
    ///
    /// # Returns
    /// * `Ok(usize)`: Number of files deleted
    /// * `Err(BraheError)`: If the manifest cannot be obtained or a file cannot be removed
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::starlink::StarlinkClient;
    ///
    /// let removed = StarlinkClient::new().prune_cache().unwrap();
    /// println!("{} stale files removed", removed);
    /// ```
    pub fn prune_cache(&self) -> Result<usize, BraheError> {
        let manifest = self.get_manifest()?;
        let listed: std::collections::HashSet<String> =
            manifest.iter().map(|e| e.file_name_string()).collect();
        let mut removed = 0;
        for path in self.cached_files()? {
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            if !listed.contains(name) {
                fs::remove_file(&path).map_err(|e| {
                    BraheError::IoError(format!("Failed to delete {}: {}", path.display(), e))
                })?;
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// Cached ephemeris files, sorted by name.
    ///
    /// # Returns
    /// * `Ok(Vec<PathBuf>)`: Files whose names are Space-Track ephemeris names
    /// * `Err(BraheError)`: If the cache directory cannot be read
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::starlink::StarlinkClient;
    ///
    /// let files = StarlinkClient::new().cached_files().unwrap();
    /// println!("{} cached", files.len());
    /// ```
    pub fn cached_files(&self) -> Result<Vec<PathBuf>, BraheError> {
        let dir = self.cache_dir()?;
        let mut files: Vec<PathBuf> = fs::read_dir(&dir)
            .map_err(|e| BraheError::IoError(format!("Failed to read {}: {}", dir.display(), e)))?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|p| {
                p.is_file()
                    && p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| EphemerisFileName::parse(n).is_ok())
            })
            .collect();
        files.sort();
        Ok(files)
    }
}

/// Resolves where a downloaded ephemeris file should be copied.
///
/// An existing directory, or a path whose last component has no extension,
/// is treated as a directory (created if missing) and `file_name` is
/// appended; otherwise `destination` itself is the target file (its parent
/// directories are created).
///
/// # Arguments
/// * `destination` - Caller-supplied directory or file path
/// * `file_name` - Name to use when `destination` is a directory
///
/// # Returns
/// * `Ok(PathBuf)`: The file path to write
/// * `Err(BraheError)`: If a directory cannot be created
fn resolve_destination(destination: &Path, file_name: &str) -> Result<PathBuf, BraheError> {
    let is_directory =
        destination.is_dir() || (!destination.exists() && destination.extension().is_none());
    let target = if is_directory {
        fs::create_dir_all(destination).map_err(|e| {
            BraheError::IoError(format!("Failed to create {}: {}", destination.display(), e))
        })?;
        destination.join(file_name)
    } else {
        if let Some(parent) = destination.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).map_err(|e| {
                BraheError::IoError(format!("Failed to create {}: {}", parent.display(), e))
            })?;
        }
        destination.to_path_buf()
    };
    Ok(target)
}

/// Copies a cached file to a destination path, overwriting it if present.
///
/// # Arguments
/// * `source` - File to copy
/// * `target` - Destination path
///
/// # Returns
/// * `Ok(())`: The file was copied
/// * `Err(BraheError)`: If the copy fails
fn copy_file(source: &Path, target: &Path) -> Result<(), BraheError> {
    fs::copy(source, target).map(|_| ()).map_err(|e| {
        BraheError::IoError(format!(
            "Failed to copy {} to {}: {}",
            source.display(),
            target.display(),
            e
        ))
    })
}

/// The current instant, rounded to whatever precision the manifest sidecar's
/// `retrieved` field preserves through a display/parse round trip, so a
/// value written to the sidecar and read back compares equal.
///
/// # Returns
/// * `Epoch`: The current instant, in UTC, rounded to millisecond precision
fn now_rounded() -> Epoch {
    let now = Epoch::now();
    Epoch::from_string(&now.to_string()).unwrap_or(now)
}

/// Sets a file's modification time to now, so a `304 Not Modified` answer
/// restarts the cache's freshness window without rewriting its content.
///
/// # Arguments
/// * `path` - The file to touch
///
/// # Returns
/// * `Ok(())`: The modification time was updated
/// * `Err(BraheError)`: If the file cannot be opened or its time set
fn touch(path: &Path) -> Result<(), BraheError> {
    fs::OpenOptions::new()
        .write(true)
        .open(path)
        .and_then(|f| f.set_modified(SystemTime::now()))
        .map_err(|e| BraheError::IoError(format!("Failed to update {}: {e}", path.display())))
}

/// A file's modification time as an [`Epoch`], for use as a manifest's
/// `retrieved` timestamp when no sidecar value is available.
///
/// # Arguments
/// * `path` - The file whose modification time is read
///
/// # Returns
/// * `Ok(Epoch)`: The modification time, in UTC
/// * `Err(BraheError)`: If the file's modification time cannot be read
fn file_epoch(path: &Path) -> Result<Epoch, BraheError> {
    let modified = fs::metadata(path)
        .and_then(|m| m.modified())
        .map_err(|e| BraheError::IoError(format!("Failed to read file modification time: {e}")))?;
    let secs = modified
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();
    Ok(Epoch::from_unix_timestamp(secs))
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::fs;
    use std::time::{Duration, SystemTime};

    use httpmock::prelude::*;
    use serial_test::serial;

    use super::*;
    use crate::trajectories::traits::Trajectory;
    use crate::utils::testing::{CacheRedirect, NetworkModeGuard};

    const MANIFEST_FIXTURE: &str = "test_assets/starlink/MANIFEST.txt";

    fn fixture_manifest() -> String {
        fs::read_to_string(MANIFEST_FIXTURE).unwrap()
    }

    fn starlink_dir(cache: &CacheRedirect) -> std::path::PathBuf {
        let dir = cache.cache_path().join("starlink");
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn age_file(path: &std::path::Path, seconds: u64) {
        let file = fs::OpenOptions::new().write(true).open(path).unwrap();
        file.set_modified(SystemTime::now() - Duration::from_secs(seconds))
            .unwrap();
    }

    const FULL_FILE: &str =
        "MEME_100001_STARLINK-38128_2540142_Operational_1473385380_UNCLASSIFIED.txt";
    const SHORT_FILE: &str =
        "MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt";

    fn asset(name: &str) -> String {
        fs::read_to_string(format!("test_assets/starlink/{name}")).unwrap()
    }

    /// Manifest listing only the two committed assets.
    fn two_line_manifest() -> String {
        format!("{FULL_FILE}\n{SHORT_FILE}\n")
    }

    fn mock_site(server: &MockServer, manifest: String) {
        server.mock(|when, then| {
            when.method(GET).path("/MANIFEST.txt");
            then.status(200).body(manifest);
        });
        server.mock(|when, then| {
            when.method(GET).path(format!("/{FULL_FILE}"));
            then.status(200).body(asset(FULL_FILE));
        });
        server.mock(|when, then| {
            when.method(GET).path(format!("/{SHORT_FILE}"));
            then.status(200).body(asset(SHORT_FILE));
        });
    }

    #[test]
    #[serial]
    fn test_download_ephemeris_caches_and_evicts_superseded() {
        let cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        mock_site(&server, two_line_manifest());
        let dir = starlink_dir(&cache);
        let old_name = "MEME_100002_STARLINK-37711_2530149_Operational_1472521800_UNCLASSIFIED.txt";
        fs::write(dir.join(old_name), asset(SHORT_FILE)).unwrap();
        let unrelated =
            "MEME_100003_STARLINK-38123_2540140_Operational_1473385260_UNCLASSIFIED.txt";
        fs::write(dir.join(unrelated), asset(SHORT_FILE)).unwrap();
        let client = StarlinkClient::with_base_url(&server.base_url());
        let path = client.download_ephemeris(100002).unwrap();
        assert_eq!(path, dir.join(SHORT_FILE));
        assert_eq!(fs::read_to_string(&path).unwrap(), asset(SHORT_FILE));
        assert!(!dir.join(old_name).exists());
        assert!(dir.join(unrelated).exists());
        assert!(dir.join(MANIFEST_FILE).exists());
        let names: Vec<String> = client
            .cached_files()
            .unwrap()
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec![SHORT_FILE.to_string(), unrelated.to_string()]);
    }

    #[test]
    #[serial]
    fn test_download_ephemeris_serves_cached_file_without_request() {
        let cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/MANIFEST.txt");
            then.status(200).body(two_line_manifest());
        });
        let file_mock = server.mock(|when, then| {
            when.method(GET).path(format!("/{SHORT_FILE}"));
            then.status(200).body(asset(SHORT_FILE));
        });
        let dir = starlink_dir(&cache);
        fs::write(dir.join(SHORT_FILE), asset(SHORT_FILE)).unwrap();
        let client = StarlinkClient::with_base_url(&server.base_url());
        client.download_ephemeris(100002).unwrap();
        file_mock.assert_calls(0);
        let itc = client.get_ephemeris(100002).unwrap();
        assert_eq!(itc.len(), 50);
        let traj = client.get_trajectory(100002).unwrap();
        assert_eq!(traj.len(), 50);
        assert!(traj.covariances.is_some());
        let rotating = client
            .get_trajectory_with_covariance_variant(
                100002,
                crate::frames::OrbitRelativeFrameVariant::Rotating,
            )
            .unwrap();
        assert_eq!(rotating.len(), 50);
    }

    #[test]
    #[serial]
    fn test_download_ephemeris_unknown_id_malformed_body_and_offline() {
        let cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/MANIFEST.txt");
            then.status(200).body(two_line_manifest());
        });
        server.mock(|when, then| {
            when.method(GET).path(format!("/{FULL_FILE}"));
            then.status(200).body("not an ephemeris\n");
        });
        let dir = starlink_dir(&cache);
        let client = StarlinkClient::with_base_url(&server.base_url());
        let err = client.download_ephemeris(424242).unwrap_err();
        assert!(err.to_string().contains("424242"), "{err}");
        assert!(client.download_ephemeris(100001).is_err());
        assert!(!dir.join(FULL_FILE).exists());
        drop(_mode);
        let _strict = NetworkModeGuard::set(Some("offline-strict"));
        let strict = StarlinkClient::with_base_url("https://brahe-network-mode-test.invalid");
        let err = strict.download_ephemeris(100002).unwrap_err();
        assert!(
            err.to_string()
                .contains("BRAHE_NETWORK_MODE is offline-strict"),
            "{err}"
        );
        fs::write(dir.join(SHORT_FILE), asset(SHORT_FILE)).unwrap();
        assert_eq!(
            strict.download_ephemeris(100002).unwrap(),
            dir.join(SHORT_FILE)
        );
    }

    #[test]
    #[serial]
    fn test_save_ephemeris_directory_and_file_destinations() {
        let cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        mock_site(&server, two_line_manifest());
        let out = tempfile::tempdir().unwrap();
        let client = StarlinkClient::with_base_url(&server.base_url());

        let existing_dir = out.path().join("existing");
        fs::create_dir_all(&existing_dir).unwrap();
        let saved = client.save_ephemeris(100002, &existing_dir).unwrap();
        assert_eq!(saved, existing_dir.join(SHORT_FILE));

        let new_dir = out.path().join("nested").join("new_dir");
        let saved = client.save_ephemeris(100002, &new_dir).unwrap();
        assert_eq!(saved, new_dir.join(SHORT_FILE));
        assert!(new_dir.is_dir());

        let renamed = out.path().join("renamed").join("sat.txt");
        let saved = client.save_ephemeris(100002, &renamed).unwrap();
        assert_eq!(saved, renamed);
        assert_eq!(fs::read_to_string(&renamed).unwrap(), asset(SHORT_FILE));
        let saved_again = client.save_ephemeris(100002, &renamed).unwrap();
        assert_eq!(saved_again, renamed);

        let dir = starlink_dir(&cache);
        assert!(dir.join(SHORT_FILE).exists());
        assert!(client.prune_cache().unwrap() == 0);
        assert!(renamed.exists());
    }

    #[test]
    #[serial]
    fn test_download_all_downloads_missing_and_keeps_cached() {
        let cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/MANIFEST.txt");
            then.status(200).body(two_line_manifest());
        });
        let full = server.mock(|when, then| {
            when.method(GET).path(format!("/{FULL_FILE}"));
            then.status(200).body(asset(FULL_FILE));
        });
        let short = server.mock(|when, then| {
            when.method(GET).path(format!("/{SHORT_FILE}"));
            then.status(200).body(asset(SHORT_FILE));
        });
        let dir = starlink_dir(&cache);
        fs::write(dir.join(SHORT_FILE), asset(SHORT_FILE)).unwrap();
        let stale = "MEME_100009_STARLINK-9_2530149_Operational_1472521800_UNCLASSIFIED.txt";
        fs::write(dir.join(stale), asset(SHORT_FILE)).unwrap();
        let client = StarlinkClient::with_base_url(&server.base_url());
        let paths = client.download_all(4).unwrap();
        assert_eq!(paths, vec![dir.join(FULL_FILE), dir.join(SHORT_FILE)]);
        full.assert_calls(1);
        short.assert_calls(0);
        assert!(dir.join(stale).exists());
        assert!(client.download_all(0).is_err());
        assert_eq!(client.prune_cache().unwrap(), 1);
        assert!(!dir.join(stale).exists());
        assert!(dir.join(MANIFEST_FILE).exists());
    }

    #[test]
    #[serial]
    fn test_download_all_stops_on_first_error() {
        let cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/MANIFEST.txt");
            then.status(200).body(two_line_manifest());
        });
        server.mock(|when, then| {
            when.method(GET).path(format!("/{FULL_FILE}"));
            then.status(404);
        });
        server.mock(|when, then| {
            when.method(GET).path(format!("/{SHORT_FILE}"));
            then.status(200).body(asset(SHORT_FILE));
        });
        let dir = starlink_dir(&cache);
        let client = StarlinkClient::with_base_url(&server.base_url()).max_retries(0);
        let err = client.download_all(1).unwrap_err();
        assert!(err.to_string().contains("404"), "{err}");
        assert!(!dir.join(SHORT_FILE).exists());
        assert!(!dir.join(FULL_FILE).exists());
    }

    #[test]
    #[serial]
    fn test_save_all_copies_into_directory() {
        let cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        mock_site(&server, two_line_manifest());
        let out = tempfile::tempdir().unwrap();
        let dest = out.path().join("ephemerides");
        let client = StarlinkClient::with_base_url(&server.base_url());
        let saved = client.save_all(&dest, 2).unwrap();
        assert_eq!(saved, vec![dest.join(FULL_FILE), dest.join(SHORT_FILE)]);
        assert_eq!(
            fs::read_to_string(dest.join(FULL_FILE)).unwrap(),
            asset(FULL_FILE)
        );
        let file_dest = out.path().join("a_file.txt");
        fs::write(&file_dest, "x").unwrap();
        assert!(client.save_all(&file_dest, 2).is_err());
        let dir = starlink_dir(&cache);
        assert!(dir.join(FULL_FILE).exists());
    }

    #[test]
    #[serial]
    fn test_starlink_client_constructors() {
        let client = StarlinkClient::new();
        assert_eq!(client.base_url(), DEFAULT_BASE_URL);
        assert_eq!(client.cache_max_age(), DEFAULT_MAX_CACHE_AGE);
        let client =
            StarlinkClient::with_base_url_and_cache_age("http://127.0.0.1:1/", 10.0).max_retries(0);
        assert_eq!(client.base_url(), "http://127.0.0.1:1");
        assert_eq!(client.cache_max_age(), 10.0);
        assert_eq!(client.max_retries, 0);
        assert_eq!(
            StarlinkClient::default().cache_max_age(),
            StarlinkClient::with_cache_age(DEFAULT_MAX_CACHE_AGE).cache_max_age()
        );
        let limited = StarlinkClient::with_rate_limit(RateLimitConfig {
            max_per_minute: 1,
            max_per_hour: 1,
        });
        assert_eq!(limited.base_url(), DEFAULT_BASE_URL);
        let _cache = CacheRedirect::new();
        assert!(
            StarlinkClient::new()
                .cache_dir()
                .unwrap()
                .ends_with("starlink")
        );
        assert!(StarlinkClient::new().cached_manifest().unwrap().is_none());
    }

    #[test]
    #[serial]
    fn test_get_manifest_downloads_and_caches() {
        let cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/MANIFEST.txt");
            then.status(200)
                .header("ETag", "\"m1\"")
                .header("Last-Modified", "Fri, 11 Sep 2026 05:15:30 GMT")
                .body(fixture_manifest());
        });
        let client = StarlinkClient::with_base_url(&server.base_url());
        let manifest = client.get_manifest().unwrap();
        assert_eq!(manifest.len(), 5);
        assert!(manifest.last_modified.is_some());
        let dir = starlink_dir(&cache);
        assert_eq!(
            fs::read_to_string(dir.join(MANIFEST_FILE)).unwrap(),
            fixture_manifest()
        );
        let meta: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(dir.join(MANIFEST_META_FILE)).unwrap())
                .unwrap();
        assert_eq!(meta["etag"], "\"m1\"");
        assert_eq!(meta["last_modified"], "Fri, 11 Sep 2026 05:15:30 GMT");
        assert!(meta["retrieved"].is_string());
        let again = client.get_manifest().unwrap();
        assert_eq!(again.len(), 5);
        assert_eq!(again.last_modified, manifest.last_modified);
        assert_eq!(again.retrieved, manifest.retrieved);
        mock.assert_calls(1);
        assert_eq!(client.cached_manifest().unwrap().unwrap().len(), 5);
        assert!(client.previous_manifest().unwrap().is_none());
    }

    #[test]
    #[serial]
    fn test_get_manifest_stale_uses_conditional_get_and_304_keeps_cache() {
        let cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let not_modified = server.mock(|when, then| {
            when.method(GET)
                .path("/MANIFEST.txt")
                .header("If-None-Match", "\"m1\"");
            then.status(304).header("ETag", "\"m1\"");
        });
        let dir = starlink_dir(&cache);
        fs::write(dir.join(MANIFEST_FILE), fixture_manifest()).unwrap();
        fs::write(
            dir.join(MANIFEST_META_FILE),
            r#"{"etag":"\"m1\"","last_modified":"Fri, 11 Sep 2026 05:15:30 GMT","retrieved":"2026-09-11T05:20:00Z"}"#,
        )
        .unwrap();
        age_file(&dir.join(MANIFEST_FILE), 7200);
        let client = StarlinkClient::with_base_url_and_cache_age(&server.base_url(), 3600.0);
        let manifest = client.get_manifest().unwrap();
        not_modified.assert_calls(1);
        assert_eq!(manifest.len(), 5);
        assert!(!client.is_cache_stale(&dir.join(MANIFEST_FILE)).unwrap());
        assert!(client.previous_manifest().unwrap().is_none());
        let meta: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(dir.join(MANIFEST_META_FILE)).unwrap())
                .unwrap();
        assert_eq!(meta["etag"], "\"m1\"");
        assert_ne!(meta["retrieved"], serde_json::json!("2026-09-11T05:20:00Z"));
    }

    #[test]
    #[serial]
    fn test_get_manifest_changed_rotates_previous() {
        let cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let mut lines: Vec<String> = fixture_manifest().lines().map(str::to_string).collect();
        lines[0] = "MEME_100001_STARLINK-38128_2540942_Operational_1473414180_UNCLASSIFIED.txt"
            .to_string();
        let updated = lines.join("\n") + "\n";
        let changed = server.mock(|when, then| {
            when.method(GET)
                .path("/MANIFEST.txt")
                .header("If-None-Match", "\"m1\"");
            then.status(200)
                .header("ETag", "\"m2\"")
                .body(updated.clone());
        });
        let dir = starlink_dir(&cache);
        fs::write(dir.join(MANIFEST_FILE), fixture_manifest()).unwrap();
        fs::write(
            dir.join(MANIFEST_META_FILE),
            r#"{"etag":"\"m1\"","last_modified":null,"retrieved":"2026-09-11T05:20:00Z"}"#,
        )
        .unwrap();
        age_file(&dir.join(MANIFEST_FILE), 7200);
        let client = StarlinkClient::with_base_url(&server.base_url());
        let manifest = client.get_manifest().unwrap();
        changed.assert_calls(1);
        assert_eq!(
            manifest
                .find_by_norad_id(100001)
                .unwrap()
                .file_name
                .metadata,
            "1473414180"
        );
        assert_eq!(
            fs::read_to_string(dir.join(PREVIOUS_MANIFEST_FILE)).unwrap(),
            fixture_manifest()
        );
        let previous = client.previous_manifest().unwrap().unwrap();
        let diff: Vec<u32> = manifest
            .changed_since(&previous)
            .iter()
            .map(|e| e.norad_cat_id)
            .collect();
        assert_eq!(diff, vec![100001]);
        let meta: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(dir.join(MANIFEST_META_FILE)).unwrap())
                .unwrap();
        assert_eq!(meta["etag"], "\"m2\"");
        assert!(meta["last_modified"].is_null());
    }

    #[test]
    #[serial]
    fn test_refresh_manifest_ignores_freshness_and_identical_body_keeps_previous() {
        let cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/MANIFEST.txt");
            then.status(200).body(fixture_manifest());
        });
        let dir = starlink_dir(&cache);
        fs::write(
            dir.join(PREVIOUS_MANIFEST_FILE),
            "MEME_100009_STARLINK-1_0010000_Operational_nomnvr_UNCLASSIFIED.txt\n",
        )
        .unwrap();
        let client = StarlinkClient::with_base_url(&server.base_url());
        client.refresh_manifest().unwrap();
        client.refresh_manifest().unwrap();
        mock.assert_calls(2);
        assert_eq!(
            client.previous_manifest().unwrap().unwrap().entries()[0].norad_cat_id,
            100009
        );
        assert!(
            !fs::read_to_string(dir.join(MANIFEST_META_FILE))
                .unwrap()
                .contains("etag\":\"")
        );
    }

    #[test]
    #[serial]
    fn test_get_manifest_offline_strict_serves_fresh_cache_only() {
        let cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("offline-strict"));
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/MANIFEST.txt");
            then.status(200).body(fixture_manifest());
        });
        let client = StarlinkClient::with_base_url("https://brahe-network-mode-test.invalid");
        let err = client.get_manifest().unwrap_err().to_string();
        assert!(
            err.contains("BRAHE_NETWORK_MODE is offline-strict"),
            "{err}"
        );
        let dir = starlink_dir(&cache);
        fs::write(dir.join(MANIFEST_FILE), fixture_manifest()).unwrap();
        assert_eq!(client.get_manifest().unwrap().len(), 5);
        age_file(&dir.join(MANIFEST_FILE), 7200);
        let err = client.get_manifest().unwrap_err();
        assert!(err.to_string().contains("offline-strict"), "{err}");
        let err = client.refresh_manifest().unwrap_err().to_string();
        assert!(
            err.contains("BRAHE_NETWORK_MODE is offline-strict"),
            "{err}"
        );
        mock.assert_calls(0);
    }

    #[test]
    #[serial]
    fn test_get_manifest_offline_serves_stale_cache() {
        let cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("offline"));
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/MANIFEST.txt");
            then.status(200).body(fixture_manifest());
        });
        let dir = starlink_dir(&cache);
        fs::write(dir.join(MANIFEST_FILE), fixture_manifest()).unwrap();
        age_file(&dir.join(MANIFEST_FILE), 7200);
        let client =
            StarlinkClient::with_base_url("https://brahe-network-mode-test.invalid").max_retries(0);
        assert_eq!(client.get_manifest().unwrap().len(), 5);
        let err = client.refresh_manifest().unwrap_err().to_string();
        assert!(err.contains("BRAHE_NETWORK_MODE is offline"), "{err}");
        mock.assert_calls(0);
    }

    #[test]
    #[serial]
    fn test_get_manifest_online_refresh_failure_is_an_error() {
        let cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let dir = starlink_dir(&cache);
        fs::write(dir.join(MANIFEST_FILE), fixture_manifest()).unwrap();
        let client = StarlinkClient::with_base_url("http://127.0.0.1:1").max_retries(0);
        assert_eq!(client.get_manifest().unwrap().len(), 5);
        age_file(&dir.join(MANIFEST_FILE), 7200);
        assert!(client.get_manifest().is_err());
        assert_eq!(
            fs::read_to_string(dir.join(MANIFEST_FILE)).unwrap(),
            fixture_manifest()
        );
        assert_eq!(client.cached_manifest().unwrap().unwrap().len(), 5);
    }

    #[test]
    #[serial]
    fn test_fetch_text_retries_then_fails_on_server_error() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let failing = server.mock(|when, then| {
            when.method(GET).path("/MANIFEST.txt");
            then.status(503);
        });
        let client = StarlinkClient::with_base_url(&server.base_url()).max_retries(2);
        let err = client.get_manifest().unwrap_err();
        assert!(err.to_string().contains("503"), "{err}");
        failing.assert_calls(3);
    }

    #[test]
    #[serial]
    fn test_get_manifest_rejects_malformed_manifest_and_keeps_old_cache() {
        let cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/MANIFEST.txt");
            then.status(200).body("garbage line\n");
        });
        let dir = starlink_dir(&cache);
        fs::write(dir.join(MANIFEST_FILE), fixture_manifest()).unwrap();
        age_file(&dir.join(MANIFEST_FILE), 7200);
        let client = StarlinkClient::with_base_url(&server.base_url());
        assert!(client.get_manifest().is_err());
        assert_eq!(
            fs::read_to_string(dir.join(MANIFEST_FILE)).unwrap(),
            fixture_manifest()
        );
    }

    #[test]
    #[serial]
    fn test_get_manifest_corrupt_sidecar_refreshes_unconditionally() {
        let cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("online"));
        let server = MockServer::start();
        let conditional = server.mock(|when, then| {
            when.method(GET)
                .path("/MANIFEST.txt")
                .header_exists("If-None-Match");
            then.status(304);
        });
        let unconditional = server.mock(|when, then| {
            when.method(GET).path("/MANIFEST.txt");
            then.status(200).body(fixture_manifest());
        });
        let dir = starlink_dir(&cache);
        fs::write(dir.join(MANIFEST_FILE), fixture_manifest()).unwrap();
        fs::write(dir.join(MANIFEST_META_FILE), "{").unwrap();
        age_file(&dir.join(MANIFEST_FILE), 7200);
        let client = StarlinkClient::with_base_url(&server.base_url());
        let manifest = client.get_manifest().unwrap();
        assert_eq!(manifest.len(), 5);
        conditional.assert_calls(0);
        unconditional.assert_calls(1);
    }
}
