/*!
 * Network access policy controlled by the `BRAHE_NETWORK_MODE` environment variable.
 *
 * Every function in brahe that opens a network connection calls [`ensure_online`]
 * first, and every cache with a time-to-live consults [`cache_policy`] before
 * deciding whether a cached file is served or refreshed. The variable therefore
 * gives a single switch for running brahe without network access.
 */

use std::env;
use std::fmt;
use std::str::FromStr;

use crate::utils::BraheError;

/// Name of the environment variable that selects the [`NetworkMode`].
pub const NETWORK_MODE_ENV: &str = "BRAHE_NETWORK_MODE";

/// Network access policy for the current process.
///
/// | mode | requests | cached file within TTL | cached file past TTL | no cached file |
/// |---|---|---|---|---|
/// | `Online` | allowed | served | refreshed | downloaded |
/// | `Offline` | never | served | served | error |
/// | `OfflineStrict` | never | served | error | error |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkMode {
    /// Requests are allowed and stale caches are refreshed.
    Online,
    /// No requests are made; cached files are served regardless of age.
    Offline,
    /// No requests are made; cached files past their time-to-live are an error.
    OfflineStrict,
}

impl FromStr for NetworkMode {
    type Err = BraheError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "online" => Ok(NetworkMode::Online),
            "offline" => Ok(NetworkMode::Offline),
            "offline-strict" => Ok(NetworkMode::OfflineStrict),
            _ => Err(BraheError::Error(format!(
                "{NETWORK_MODE_ENV} has unrecognized value {s:?}; expected one of online, offline, offline-strict"
            ))),
        }
    }
}

impl fmt::Display for NetworkMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            NetworkMode::Online => "online",
            NetworkMode::Offline => "offline",
            NetworkMode::OfflineStrict => "offline-strict",
        };
        write!(f, "{name}")
    }
}

/// Read the network mode from the `BRAHE_NETWORK_MODE` environment variable.
///
/// The variable is read on every call, so changing it at runtime takes effect
/// on the next network-relevant operation. An unset or blank variable selects
/// [`NetworkMode::Online`].
///
/// # Returns
///
/// * `Ok(NetworkMode)` - The active mode
/// * `Err(BraheError)` - If the variable holds a value other than `online`,
///   `offline`, or `offline-strict`, including a value that is not valid UTF-8
///
/// # Examples
///
/// ```
/// use brahe::utils::{network_mode, NetworkMode};
///
/// let mode = network_mode().unwrap();
/// if mode == NetworkMode::Online {
///     println!("network requests are allowed");
/// }
/// ```
pub fn network_mode() -> Result<NetworkMode, BraheError> {
    match env::var(NETWORK_MODE_ENV) {
        Ok(value) if value.trim().is_empty() => Ok(NetworkMode::Online),
        Ok(value) => value.parse(),
        Err(env::VarError::NotPresent) => Ok(NetworkMode::Online),
        Err(env::VarError::NotUnicode(raw)) => Err(BraheError::Error(format!(
            "{NETWORK_MODE_ENV} has unrecognized value {raw:?}; expected one of online, offline, offline-strict"
        ))),
    }
}

/// Fail unless the network mode permits a request.
///
/// Called immediately before every HTTP request the library makes.
///
/// # Arguments
///
/// * `resource` - Short description of what would be downloaded, used in the error
///
/// # Returns
///
/// * `Ok(())` - Requests are allowed
/// * `Err(BraheError)` - The mode is `offline` or `offline-strict`, or the
///   variable holds an unrecognized value
pub(crate) fn ensure_online(resource: &str) -> Result<(), BraheError> {
    match network_mode()? {
        NetworkMode::Online => Ok(()),
        mode => Err(BraheError::Error(format!(
            "{NETWORK_MODE_ENV} is {mode}; {resource} is not cached and cannot be downloaded"
        ))),
    }
}

/// Outcome of [`cache_policy`] for a cached file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CacheDecision {
    /// Use the cached file.
    Serve,
    /// Download a fresh copy.
    Refresh,
}

/// Decide whether a cached file is served or refreshed under the current mode.
///
/// Callers compute `stale` from their own time-to-live rule and pass it in; this
/// function only applies the network mode to that result.
///
/// # Arguments
///
/// * `resource` - Short description of the cached file, used in the error
/// * `stale` - Whether the file is older than the caller's time-to-live
///
/// # Returns
///
/// * `Ok(CacheDecision::Serve)` - The file is fresh, or the mode is `offline`
/// * `Ok(CacheDecision::Refresh)` - The file is stale and the mode is `online`
/// * `Err(BraheError)` - The file is stale and the mode is `offline-strict`, or
///   the variable holds an unrecognized value
pub(crate) fn cache_policy(resource: &str, stale: bool) -> Result<CacheDecision, BraheError> {
    let mode = network_mode()?;
    if !stale {
        return Ok(CacheDecision::Serve);
    }
    match mode {
        NetworkMode::Online => Ok(CacheDecision::Refresh),
        NetworkMode::Offline => Ok(CacheDecision::Serve),
        NetworkMode::OfflineStrict => Err(BraheError::Error(format!(
            "{NETWORK_MODE_ENV} is {mode}; {resource} is older than its cache limit and cannot be refreshed"
        ))),
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use serial_test::serial;

    use super::*;
    use crate::utils::testing::NetworkModeGuard;

    #[test]
    #[serial]
    fn test_network_mode_unset_is_online() {
        let _guard = NetworkModeGuard::set(None);
        assert_eq!(network_mode().unwrap(), NetworkMode::Online);
    }

    #[test]
    #[serial]
    fn test_network_mode_empty_is_online() {
        let _guard = NetworkModeGuard::set(Some("   "));
        assert_eq!(network_mode().unwrap(), NetworkMode::Online);
    }

    #[test]
    #[serial]
    fn test_network_mode_parses_values_case_insensitively() {
        for (value, expected) in [
            ("online", NetworkMode::Online),
            ("OFFLINE", NetworkMode::Offline),
            (" offline-strict ", NetworkMode::OfflineStrict),
            ("Offline-Strict", NetworkMode::OfflineStrict),
        ] {
            let _guard = NetworkModeGuard::set(Some(value));
            assert_eq!(network_mode().unwrap(), expected, "value {value:?}");
        }
    }

    #[test]
    #[serial]
    fn test_network_mode_rejects_unknown_value() {
        let _guard = NetworkModeGuard::set(Some("sometimes"));
        let err = network_mode().unwrap_err().to_string();
        assert_eq!(
            err,
            "BRAHE_NETWORK_MODE has unrecognized value \"sometimes\"; expected one of online, offline, offline-strict"
        );
    }

    #[test]
    #[serial_test::parallel]
    fn test_network_mode_display_roundtrip() {
        for mode in [
            NetworkMode::Online,
            NetworkMode::Offline,
            NetworkMode::OfflineStrict,
        ] {
            assert_eq!(mode.to_string().parse::<NetworkMode>().unwrap(), mode);
        }
    }

    #[test]
    #[serial]
    fn test_ensure_online_allows_online() {
        let _guard = NetworkModeGuard::set(Some("online"));
        assert!(ensure_online("test resource").is_ok());
    }

    #[test]
    #[serial]
    fn test_ensure_online_blocks_offline_modes() {
        for mode in ["offline", "offline-strict"] {
            let _guard = NetworkModeGuard::set(Some(mode));
            let err = ensure_online("Celestrak request").unwrap_err().to_string();
            assert_eq!(
                err,
                format!(
                    "BRAHE_NETWORK_MODE is {mode}; Celestrak request is not cached and cannot be downloaded"
                )
            );
        }
    }

    #[test]
    #[serial]
    fn test_cache_policy_fresh_is_served_in_every_mode() {
        for mode in [
            None,
            Some("online"),
            Some("offline"),
            Some("offline-strict"),
        ] {
            let _guard = NetworkModeGuard::set(mode);
            assert_eq!(
                cache_policy("thing", false).unwrap(),
                CacheDecision::Serve,
                "mode {mode:?}"
            );
        }
    }

    #[test]
    #[serial]
    fn test_cache_policy_stale_online_refreshes() {
        let _guard = NetworkModeGuard::set(None);
        assert_eq!(cache_policy("thing", true).unwrap(), CacheDecision::Refresh);
    }

    #[test]
    #[serial]
    fn test_cache_policy_stale_offline_serves() {
        let _guard = NetworkModeGuard::set(Some("offline"));
        assert_eq!(cache_policy("thing", true).unwrap(), CacheDecision::Serve);
    }

    #[test]
    #[serial]
    fn test_cache_policy_stale_strict_errors() {
        let _guard = NetworkModeGuard::set(Some("offline-strict"));
        let err = cache_policy("EOP file finals.all", true)
            .unwrap_err()
            .to_string();
        assert_eq!(
            err,
            "BRAHE_NETWORK_MODE is offline-strict; EOP file finals.all is older than its cache limit and cannot be refreshed"
        );
    }

    #[test]
    #[serial]
    fn test_cache_policy_propagates_bad_value() {
        let _guard = NetworkModeGuard::set(Some("nope"));
        assert!(cache_policy("thing", false).is_err());
    }

    #[test]
    #[serial]
    fn test_guard_restores_previous_value() {
        let _outer = NetworkModeGuard::set(Some("offline"));
        {
            let _inner = NetworkModeGuard::set(Some("online"));
            assert_eq!(network_mode().unwrap(), NetworkMode::Online);
        }
        assert_eq!(network_mode().unwrap(), NetworkMode::Offline);
    }

    #[test]
    #[serial]
    #[cfg(unix)]
    fn test_network_mode_rejects_non_utf8_value() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let prev = env::var(NETWORK_MODE_ENV).ok();
        // SAFETY: single-threaded within a #[serial] test; no other thread
        // reads the environment concurrently.
        unsafe {
            env::set_var(NETWORK_MODE_ENV, OsStr::from_bytes(b"off\xffline"));
        }
        let err = network_mode().unwrap_err().to_string();
        assert!(err.contains("unrecognized value"), "{err}");
        // SAFETY: see above.
        unsafe {
            match &prev {
                Some(p) => env::set_var(NETWORK_MODE_ENV, p),
                None => env::remove_var(NETWORK_MODE_ENV),
            }
        }
    }
}
