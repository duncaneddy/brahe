/*!
 * CelestrakClient API client for querying satellite catalog data from CelestrakClient.
 *
 * This module provides a composable query builder and HTTP client for the
 * CelestrakClient REST API. It supports GP queries (`gp.php`), supplemental GP
 * queries (`sup-gp.php`), and SATCAT queries (`satcat/records.php`) with a
 * fluent builder pattern that works naturally in both Rust and Python.
 *
 * GP queries return the same [`GPRecord`] type as the SpaceTrack module,
 * enabling interoperability between both data sources.
 *
 * # Architecture
 *
 * - [`types`] - Core enums: [`CelestrakQueryType`], [`CelestrakOutputFormat`], [`SupGPSource`]
 * - [`query`] - [`CelestrakQuery`] fluent builder
 * - [`client`] - [`CelestrakClient`] with HTTP and caching
 * - [`responses`] - Typed response struct: [`CelestrakSATCATRecord`]
 * - [`filter`] - Client-side filtering engine for SpaceTrack-compatible operators
 *
 * # Examples
 *
 * ```ignore
 * use brahe::celestrak::*;
 *
 * // Create a client
 * let client = CelestrakClient::new();
 *
 * // Build a query for GP data for the ISS station group
 * let query = CelestrakQuery::gp()
 *     .group("stations");
 *
 * // Execute query and get typed response (same GPRecord as SpaceTrack!)
 * let records = client.query_gp(&query).unwrap();
 * println!("First record: {:?}", records[0].object_name);
 * ```
 */

pub mod client;
pub mod filter;
pub mod query;
pub mod responses;
pub mod types;

// Re-export commonly used types for convenience
pub use client::CelestrakClient;
pub use query::CelestrakQuery;
pub use responses::CelestrakSATCATRecord;
pub use types::{CelestrakOutputFormat, CelestrakQueryType, SupGPSource};

// Re-export GPRecord for convenience (users can import from either module)
pub use crate::types::GPRecord;
