/*!
 * SpaceTrack API client for querying satellite catalog data from Space-Track.org.
 *
 * This module provides a composable query builder and HTTP client for the
 * Space-Track.org REST API. It supports all major request classes (GP, SATCAT,
 * Decay, etc.) with a fluent builder pattern that works naturally in both
 * Rust and Python.
 *
 * # Architecture
 *
 * - [`types`] - Core enums: [`RequestController`], [`RequestClass`], [`SortOrder`], [`OutputFormat`]
 * - [`operators`] - Filter operator functions: `greater_than`, `less_than`, `inclusive_range`, etc.
 * - [`query`] - [`SpaceTrackQuery`] fluent builder
 * - [`client`] - [`SpaceTrackClient`] with authentication and query execution
 * - [`responses`] - Typed response structs: [`GPRecord`], [`SATCATRecord`]
 *
 * # Examples
 *
 * ```no_run
 * use brahe::spacetrack::*;
 *
 * // Create a client
 * let client = SpaceTrackClient::new("user@example.com", "password");
 *
 * // Build a query for the latest GP data for the ISS
 * let query = SpaceTrackQuery::new(RequestClass::GP)
 *     .filter("NORAD_CAT_ID", "25544")
 *     .order_by("EPOCH", SortOrder::Desc)
 *     .limit(1);
 *
 * // Execute query and get typed response
 * let records = client.query_gp(&query).unwrap();
 * println!("ISS epoch: {:?}", records[0].epoch);
 * ```
 */

pub mod client;
pub mod operators;
pub mod query;
pub mod responses;
pub mod types;

// Re-export commonly used types for convenience
pub use client::SpaceTrackClient;
pub use query::SpaceTrackQuery;
pub use responses::{
    FileShareFileRecord, FolderRecord, GPRecord, SATCATRecord, SpEphemerisFileRecord,
};
pub use types::{OutputFormat, RequestClass, RequestController, SortOrder};
