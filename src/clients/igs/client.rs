/*!
 * Blocking client for IGS precise-product archives.
 */

use std::env;
use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::clients::igs::source::IGSProductSource;
use crate::utils::{BraheError, get_igs_cache_dir};

/// Default seconds a cached directory listing is served before it is refreshed.
pub const DEFAULT_IGS_LISTING_MAX_AGE: f64 = 3600.0;

const DEFAULT_MAX_RETRIES: u32 = 3;

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::testing::CacheRedirect;
    use serial_test::serial;

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
}
