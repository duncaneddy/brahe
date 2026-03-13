/*!
 * Polygon tessellation for satellite collection planning
 *
 * This module divides area-of-interest polygons (and points) into smaller
 * rectangular tiles aligned with satellite ground tracks. Essential for
 * satellite imaging collection planning where sensor footprints are
 * rectangular strips along the orbit track.
 *
 * # Architecture
 *
 * The [`Tessellator`] trait defines the core interface. The primary implementation
 * is [`OrbitGeometryTessellator`], which uses orbital mechanics to compute
 * along-track directions and then tiles the target area into strips.
 *
 * Output tiles are [`PolygonLocation`] instances with metadata properties:
 * - `tile_direction`: along-track unit ECEF vector `[x, y, z]`
 * - `tile_width`: cross-track dimension (m)
 * - `tile_length`: along-track dimension (m)
 * - `tile_area`: width × length (m²)
 * - `tile_group_id`: UUID shared by tiles in the same tiling direction
 * - `spacecraft_ids`: list of spacecraft identifiers
 */

pub mod merging;
pub mod orbit_geometry;
pub mod traits;

pub use merging::*;
pub use orbit_geometry::*;
pub use traits::*;
