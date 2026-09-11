/*!
 * Client for Starlink's public ephemeris mirror.
 *
 * Starlink publishes the Modified ITC ephemerides it submits to Space-Track
 * at `https://api.starlink.com/public-files/ephemerides/`, one file per
 * satellite, refreshed about every eight hours, with a plain-text
 * `MANIFEST.txt` listing the current file names. The client caches
 * the manifest and the files locally, re-downloads a satellite's file only
 * when the manifest names a new one, and hands back [`crate::itc::ITC`]
 * messages or trajectories. [`StarlinkManifest`] is the typed listing with
 * the start and stop epochs decoded from the file names, and [`StarlinkClient`]
 * is the client that downloads and caches it.
 */

pub mod client;
pub mod manifest;

pub use client::StarlinkClient;
pub use manifest::{StarlinkManifest, StarlinkManifestEntry};

use crate::clients::RateLimitConfig;

/// Default request limits for the Starlink mirror: 1000 per minute and 30000 per hour.
pub const STARLINK_RATE_LIMIT: RateLimitConfig = RateLimitConfig {
    max_per_minute: 1000,
    max_per_hour: 30000,
};
