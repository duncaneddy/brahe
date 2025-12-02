/*!
 * SpaceTrack API client module
 *
 * This module provides a Rust client for the Space-Track.org API, enabling
 * access to orbital element data, satellite catalog information, and other
 * space surveillance data.
 *
 * # Features
 *
 * - **Async and sync clients**: Both async (`SpaceTrackClient`) and blocking
 *   (`BlockingSpaceTrackClient`) clients are available.
 * - **Typed query methods**: Client methods like `gp()`, `satcat()`, etc.
 *   provide type-safe querying with operator support.
 * - **Rate limiting**: Built-in rate limiting respects SpaceTrack's limits
 *   (30 requests/minute, 300 requests/hour).
 * - **SGP propagator integration**: Convert TLE data directly to `SGPPropagator`
 *   instances for orbit propagation.
 *
 * # Example
 *
 * ```ignore
 * use brahe::spacetrack::BlockingSpaceTrackClient;
 *
 * // Create a blocking client (works in non-async contexts)
 * let client = BlockingSpaceTrackClient::new("username", "password")?;
 *
 * // Query GP data for ISS (NORAD ID 25544)
 * let records = client.gp(
 *     Some(25544),  // norad_cat_id
 *     None, None, None, None, None,  // object_name, object_id, epoch, object_type, country_code
 *     Some(1),      // limit
 *     None,         // orderby
 * )?;
 *
 * // Convert to SGP propagator
 * let propagator = records[0].to_sgp_propagator(60.0)?;
 *
 * // Or use generic_request for arbitrary queries
 * let response = client.generic_request(
 *     "basicspacedata",
 *     "gp",
 *     &[("NORAD_CAT_ID", 25544.into())],
 *     None,
 * )?;
 * ```
 *
 * # Authentication
 *
 * You need a Space-Track.org account to use this API. Register at
 * <https://www.space-track.org/auth/createAccount>.
 *
 * For testing, you can use the test server at
 * `https://for-testing-only.space-track.org/` with test credentials.
 */

pub mod blocking;
pub mod client;
pub mod error;
pub mod operators;
pub mod query;
pub mod rate_limiter;
pub(crate) mod request_classes;
pub(crate) mod serde_helpers;

// Re-export main types for convenience
pub use blocking::BlockingSpaceTrackClient;
pub use client::{
    DEFAULT_BASE_URL, SpaceTrackClient, SpaceTrackQueryBuilder, TEST_BASE_URL,
    parse_3le_to_propagators,
};
pub use error::SpaceTrackError;
pub use operators::{
    QueryValue, equals, greater_than, inclusive_range, less_than, like, not_equal, null_val,
    startswith, stringify_bool, stringify_predicate_value, stringify_sequence,
};
pub use rate_limiter::RateLimiter;

// Re-export query builder types
pub use query::{
    HasTLEData, SpaceTrackOperator, SpaceTrackOrder, SpaceTrackPredicate,
    SpaceTrackPredicateBuilder, SpaceTrackQuery, SpaceTrackValue,
};

// Re-export record types (users receive these as query results)
pub use request_classes::{
    // BasicSpaceData Records
    AnnouncementRecord,
    BoxscoreRecord,
    // ExpandedSpaceData Records
    CARRecord,
    CDMPublicRecord,
    CDMRecord,
    DecayRecord,
    GPHistoryRecord,
    // Extension traits for record conversion
    GPHistoryRecordVecExt,
    GPRecord,
    GPRecordVecExt,
    LaunchSiteRecord,
    ManeuverHistoryRecord,
    ManeuverRecord,
    OMMRecord,
    OMMRecordVecExt,
    OrganizationRecord,
    SATCATChangeRecord,
    SATCATDebutRecord,
    SATCATRecord,
    SatelliteRecord,
    TIPRecord,
    TLERecord,
    TLERecordVecExt,
};
