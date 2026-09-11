/*!
 * Network clients for external satellite data services.
 *
 * Each submodule wraps one service behind a blocking HTTP client with local
 * caching and `BRAHE_NETWORK_MODE` handling:
 *
 * - [`celestrak`] - CelesTrak GP, supplemental GP and SATCAT queries
 * - [`spacetrack`] - Space-Track.org authenticated queries, CDMs and file shares
 * - [`gcat`] - Jonathan McDowell's GCAT SATCAT and PSATCAT catalogs
 *
 * [`rate_limiter`] provides the sliding-window request limiter shared by the
 * clients that expose a configurable request rate.
 *
 * The submodules are re-exported at the crate root, so `brahe::spacetrack::SpaceTrackClient`
 * and `brahe::clients::spacetrack::SpaceTrackClient` name the same type.
 */

pub mod celestrak;
pub mod spacetrack;
