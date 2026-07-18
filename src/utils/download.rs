/*!
 * Shared HTTP download helper with retry, exponential backoff, and jitter.
 *
 * Used by the caching data providers (EOP, space weather) to fetch remote data
 * files resiliently and write them to disk atomically.
 */

use std::io::Read;
use std::path::Path;
use std::thread::sleep;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::utils::{BraheError, atomic_write};

/// Maximum number of download attempts (including the first) before giving up.
const MAX_DOWNLOAD_ATTEMPTS: u32 = 4;
/// Base delay used for exponential backoff between download attempts.
const BACKOFF_BASE_DELAY: Duration = Duration::from_millis(500);
/// Upper bound on the backoff delay between download attempts.
const BACKOFF_MAX_DELAY: Duration = Duration::from_secs(30);

/// Check whether an HTTP error is transient and worth retrying.
///
/// Connection refusals, resets, and timeouts surface as [`ureq::Error::Io`],
/// [`ureq::Error::Timeout`], or [`ureq::Error::ConnectionFailed`]; transient
/// server-side failures surface as 429/5xx status codes. Client errors (4xx) and
/// malformed URIs are not retried.
fn is_retryable_error(e: &ureq::Error) -> bool {
    matches!(
        e,
        ureq::Error::StatusCode(429 | 500 | 502 | 503 | 504)
            | ureq::Error::Io(_)
            | ureq::Error::Timeout(_)
            | ureq::Error::HostNotFound
            | ureq::Error::ConnectionFailed
    )
}

/// Compute the backoff delay before the next retry using exponential backoff with
/// full jitter, capped at [`BACKOFF_MAX_DELAY`].
///
/// `attempt` is the 1-based index of the attempt that just failed, so the first
/// retry waits a random duration in `[0, BACKOFF_BASE_DELAY]`, the second in
/// `[0, 2 * BACKOFF_BASE_DELAY]`, and so on.
fn backoff_delay(attempt: u32) -> Duration {
    let base_ms = BACKOFF_BASE_DELAY.as_millis() as u64;
    let max_ms = BACKOFF_MAX_DELAY.as_millis() as u64;

    // Exponential window base * 2^(attempt - 1), saturating and capped at max.
    let window_ms = base_ms
        .saturating_mul(1u64 << (attempt - 1).min(32))
        .min(max_ms);

    // Full jitter within `[0, window_ms]`. Seed the sample from the sub-second wall
    // clock, which decorrelates retries across threads and processes without pulling
    // in a random-number dependency.
    let entropy = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0);
    let jittered_ms = if window_ms == 0 {
        0
    } else {
        entropy % (window_ms + 1)
    };
    Duration::from_millis(jittered_ms)
}

/// Fetch the body of `url` as a string, retrying transient failures with exponential
/// backoff and jitter, then write it atomically to `filepath`.
///
/// `description` is a short human-readable label for the product being downloaded
/// (e.g. `"Standard EOP"`, `"space weather"`) that, together with `url`, is included
/// in any error message to make failures actionable.
pub(crate) fn download_to_file(
    url: &str,
    description: &str,
    filepath: &Path,
) -> Result<(), BraheError> {
    let mut attempt: u32 = 0;

    let body = loop {
        attempt += 1;

        match ureq::get(url)
            .call()
            .and_then(|mut response| response.body_mut().read_to_string())
        {
            Ok(body) => break body,
            Err(e) => {
                if attempt < MAX_DOWNLOAD_ATTEMPTS && is_retryable_error(&e) {
                    sleep(backoff_delay(attempt));
                    continue;
                }
                return Err(BraheError::IoError(format!(
                    "{} download from {} failed after {} attempt(s): {}",
                    description, url, attempt, e
                )));
            }
        }
    };

    atomic_write(filepath, body.as_bytes())?;

    Ok(())
}

/// Fetch the raw bytes of `url`, retrying transient failures with exponential
/// backoff and jitter, and return them in memory.
///
/// Unlike [`download_to_file`], the response body is read through a streaming
/// reader rather than ureq's convenience string method, so large binary payloads
/// (e.g. multi-MB SPICE kernels) are not subject to ureq's default in-memory
/// response size limit. The returned error carries the URL and attempt count;
/// callers are expected to wrap it with product-specific context.
pub(crate) fn download_bytes(url: &str) -> Result<Vec<u8>, BraheError> {
    download_bytes_impl(url, None)
}

/// As [`download_bytes`], but sending a caller-specified `User-Agent` header.
///
/// Some mirrors (e.g. the star catalog data source) return `403 Forbidden` to
/// default HTTP client user agents and require a browser-like one instead.
pub(crate) fn download_bytes_with_user_agent(
    url: &str,
    user_agent: &str,
) -> Result<Vec<u8>, BraheError> {
    download_bytes_impl(url, Some(user_agent))
}

/// Shared retry/backoff core for [`download_bytes`] and
/// [`download_bytes_with_user_agent`]. `user_agent`, if provided, is sent as
/// the request's `User-Agent` header; `None` leaves ureq's default.
fn download_bytes_impl(url: &str, user_agent: Option<&str>) -> Result<Vec<u8>, BraheError> {
    let mut attempt: u32 = 0;

    loop {
        attempt += 1;

        // Connection/status phase. ureq surfaces 4xx/5xx as `Err(StatusCode)`,
        // so this covers both transport failures and retryable server statuses.
        let mut request = ureq::get(url);
        if let Some(ua) = user_agent {
            request = request.header("User-Agent", ua);
        }
        let response = match request.call() {
            Ok(response) => response,
            Err(e) => {
                if attempt < MAX_DOWNLOAD_ATTEMPTS && is_retryable_error(&e) {
                    sleep(backoff_delay(attempt));
                    continue;
                }
                return Err(BraheError::Error(format!(
                    "request to {} failed after {} attempt(s): {}",
                    url, attempt, e
                )));
            }
        };

        // Body phase, read through a streaming reader to bypass ureq's default
        // in-memory size limit. A failure here (connection reset, truncated
        // transfer, unexpected EOF) is a transport error too, so retry the whole
        // request rather than surfacing a partial download.
        let mut buffer = Vec::new();
        match response.into_body().into_reader().read_to_end(&mut buffer) {
            Ok(_) => return Ok(buffer),
            Err(e) => {
                if attempt < MAX_DOWNLOAD_ATTEMPTS {
                    sleep(backoff_delay(attempt));
                    continue;
                }
                return Err(BraheError::Error(format!(
                    "reading response body from {} failed after {} attempt(s): {}",
                    url, attempt, e
                )));
            }
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use httpmock::prelude::*;

    use super::*;

    #[test]
    fn test_download_bytes_returns_body_on_success() {
        let server = MockServer::start();
        let _mock = server.mock(|when, then| {
            when.method(GET).path("/kernel.bsp");
            then.status(200).body("kernel-bytes");
        });

        let bytes = download_bytes(&server.url("/kernel.bsp")).unwrap();
        assert_eq!(bytes, b"kernel-bytes");
    }

    #[test]
    fn test_download_bytes_with_user_agent_sends_header() {
        // Mirrors are known to 403 default HTTP client user agents (e.g. the
        // star catalog data source); download_bytes_with_user_agent must send
        // the caller-provided User-Agent header rather than ureq's default.
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET)
                .path("/catalog.txt")
                .header("User-Agent", "Mozilla/5.0 (compatible; brahe)");
            then.status(200).body("catalog-bytes");
        });

        let bytes = download_bytes_with_user_agent(
            &server.url("/catalog.txt"),
            "Mozilla/5.0 (compatible; brahe)",
        )
        .unwrap();
        assert_eq!(bytes, b"catalog-bytes");
        mock.assert_calls(1);
    }

    #[test]
    fn test_download_bytes_retries_transient_failure() {
        // A persistently transient (503) endpoint is retried up to the attempt
        // cap before failing, so the mock records exactly MAX_DOWNLOAD_ATTEMPTS
        // requests — proving the retry loop is wired into download_bytes.
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/kernel.bsp");
            then.status(503);
        });

        let result = download_bytes(&server.url("/kernel.bsp"));
        assert!(result.is_err());
        assert_eq!(mock.calls(), MAX_DOWNLOAD_ATTEMPTS as usize);
    }

    #[test]
    fn test_download_bytes_does_not_retry_client_error() {
        // A 404 is not transient: download_bytes must fail after a single request.
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/kernel.bsp");
            then.status(404);
        });

        let result = download_bytes(&server.url("/kernel.bsp"));
        assert!(result.is_err());
        assert_eq!(mock.calls(), 1);
    }

    #[test]
    fn test_is_retryable_error() {
        use std::io;

        // Transient transport/server failures are retryable.
        assert!(is_retryable_error(&ureq::Error::ConnectionFailed));
        assert!(is_retryable_error(&ureq::Error::HostNotFound));
        assert!(is_retryable_error(&ureq::Error::Io(io::Error::new(
            io::ErrorKind::ConnectionRefused,
            "Connection refused"
        ))));
        assert!(is_retryable_error(&ureq::Error::StatusCode(503)));
        assert!(is_retryable_error(&ureq::Error::StatusCode(429)));

        // Client errors are not retryable.
        assert!(!is_retryable_error(&ureq::Error::StatusCode(404)));
        assert!(!is_retryable_error(&ureq::Error::StatusCode(400)));
    }

    #[test]
    fn test_backoff_delay_bounds() {
        // Delay for a given attempt must never exceed the exponential window,
        // and the window is capped at BACKOFF_MAX_DELAY.
        for attempt in 1..=8u32 {
            let window_ms = (BACKOFF_BASE_DELAY.as_millis() as u64)
                .saturating_mul(1u64 << (attempt - 1).min(32))
                .min(BACKOFF_MAX_DELAY.as_millis() as u64);

            for _ in 0..100 {
                let delay = backoff_delay(attempt);
                assert!(delay <= Duration::from_millis(window_ms));
                assert!(delay <= BACKOFF_MAX_DELAY);
            }
        }
    }
}
