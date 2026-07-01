/*!
 * Module that provides download functionality for different EOP file sources.
 */

use std::path::Path;
use std::thread::sleep;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::utils::{BraheError, atomic_write};

// Sourced from USNO and Paris Observatory (IERS) mirrors; the primary IERS
// datacenter (datacenter.iers.org) is frequently unavailable.
const STANDARD_FILE_SOURCE: &str = "https://maia.usno.navy.mil/ser7/finals2000A.all";
const C04_FILE_SOURCE: &str = "https://hpiers.obspm.fr/iers/eop/eopc04/eopc04.1962-now";

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
/// `product` is a short human-readable label for the product being downloaded (e.g.
/// `"Standard EOP"`) that, together with `url`, is included in any error message to
/// make failures actionable.
fn download_eop_file(url: &str, product: &str, filepath: &Path) -> Result<(), BraheError> {
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
                    product, url, attempt, e
                )));
            }
        }
    };

    atomic_write(filepath, body.as_bytes())?;

    Ok(())
}

/// Download latest C04 Earth orientation parameter file. Will attempt to download the latest
/// parameter file to the specified location. Creating any missing directories as required.
///
/// The download source is the [Paris Observatory IERS EOP C04 series](https://hpiers.obspm.fr/iers/eop/eopc04/eopc04.1962-now).
///
/// # Arguments
/// - `filepath`: Path of desired output file
pub fn download_c04_eop_file(filepath: &str) -> Result<(), BraheError> {
    download_eop_file(C04_FILE_SOURCE, "C04 EOP", Path::new(filepath))
}

/// Download latest standard Earth orientation parameter file. Will attempt to download the latest
/// parameter file to the specified location. Creating any missing directories as required.
///
/// The download source is the [USNO finals2000A.all Bulletin A product](https://maia.usno.navy.mil/ser7/finals2000A.all).
///
/// # Arguments
/// - `filepath`: Path of desired output file
pub fn download_standard_eop_file(filepath: &str) -> Result<(), BraheError> {
    download_eop_file(STANDARD_FILE_SOURCE, "Standard EOP", Path::new(filepath))
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::env;
    use std::fs;

    use httpmock::prelude::*;
    use tempfile::tempdir;

    use super::*;

    fn get_standard_file_contents() -> String {
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let filepath = Path::new(&manifest_dir)
            .join("test_assets")
            .join("finals.all.iau2000.txt");

        fs::read_to_string(filepath).expect("Failed to read file")
    }

    fn get_c04_file_contents() -> String {
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let filepath = Path::new(&manifest_dir)
            .join("test_assets")
            .join("EOP_C04_one_file_1962-now.txt");

        fs::read_to_string(filepath).expect("Failed to read file")
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

    #[test]
    #[cfg_attr(not(feature = "integration"), ignore)] // Only run in CI to avoid network calls
    fn test_download_standard_eop_file() {
        // Mock return of contents of HTTP call
        let server = MockServer::start();
        let _download_mock = server.mock(|when, then| {
            when.method(GET).path(STANDARD_FILE_SOURCE);
            then.status(200)
                .header("content-type", "text/html; charset=UTF-8")
                .body(get_standard_file_contents());
        });

        // Create Temporary File
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("temp-standard-file.txt");
        let filepath_str = filepath.to_str().unwrap();

        download_standard_eop_file(filepath_str).expect("Failed to download file");
        assert!(Path::new(filepath_str).exists());
    }

    #[test]
    #[cfg_attr(not(feature = "integration"), ignore)] // Only run in CI to avoid network calls
    fn test_download_c04_eop_file() {
        // Mock return of contents of HTTP call
        let server = MockServer::start();
        let _download_mock = server.mock(|when, then| {
            when.method(GET).path(C04_FILE_SOURCE);
            then.status(200)
                .header("content-type", "text/html; charset=UTF-8")
                .body(get_c04_file_contents());
        });

        // Create Temporary File
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("temp-c04-file.txt");
        let filepath_str = filepath.to_str().unwrap();

        download_c04_eop_file(filepath_str).expect("Failed to download file");
        assert!(Path::new(filepath_str).exists());
    }
}
