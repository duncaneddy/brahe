/*!
 * JPL Small-Body Database (SBDB) Lookup client.
 *
 * Resolves a small-body search string (name or designation) to its NAIF/SPK
 * ID and, when available, SI physical parameters via the SBDB Lookup API
 * (`https://ssd-api.jpl.nasa.gov/sbdb.api`).
 */

pub mod responses;

pub use responses::SBDBObject;
