/*!
 * Point-in-polygon algorithms for geospatial calculations
 *
 * Provides functions for testing whether a point lies inside a polygon,
 * with support for polygons that cross the anti-meridian (±180° longitude).
 */

use std::f64::consts::PI;

/// Check if a polygon crosses the anti-meridian (±180° longitude boundary).
///
/// A polygon crosses the anti-meridian if any two consecutive vertices have
/// longitudes that differ by more than 180 degrees. This indicates the polygon
/// wraps around the back of the Earth.
///
/// # Arguments
/// * `vertices` - Polygon vertices as (longitude, latitude) pairs in radians
///
/// # Returns
/// `true` if the polygon crosses the anti-meridian, `false` otherwise
///
/// # Example
/// ```
/// use brahe::coordinates::polygon_crosses_antimeridian;
/// use std::f64::consts::PI;
///
/// // Polygon spanning from 170° to -170° (crosses anti-meridian)
/// let vertices = vec![
///     (170.0_f64.to_radians(), 10.0_f64.to_radians()),
///     (-170.0_f64.to_radians(), 10.0_f64.to_radians()),
///     (-170.0_f64.to_radians(), 20.0_f64.to_radians()),
///     (170.0_f64.to_radians(), 20.0_f64.to_radians()),
///     (170.0_f64.to_radians(), 10.0_f64.to_radians()),
/// ];
/// assert!(polygon_crosses_antimeridian(&vertices));
///
/// // Simple polygon not crossing anti-meridian
/// let simple = vec![
///     (10.0_f64.to_radians(), 10.0_f64.to_radians()),
///     (20.0_f64.to_radians(), 10.0_f64.to_radians()),
///     (20.0_f64.to_radians(), 20.0_f64.to_radians()),
///     (10.0_f64.to_radians(), 20.0_f64.to_radians()),
///     (10.0_f64.to_radians(), 10.0_f64.to_radians()),
/// ];
/// assert!(!polygon_crosses_antimeridian(&simple));
/// ```
pub fn polygon_crosses_antimeridian(vertices: &[(f64, f64)]) -> bool {
    if vertices.len() < 2 {
        return false;
    }

    for i in 0..vertices.len() - 1 {
        let lon1 = vertices[i].0;
        let lon2 = vertices[i + 1].0;
        let diff = (lon2 - lon1).abs();

        // If longitude difference > PI, the edge crosses the anti-meridian
        if diff > PI {
            return true;
        }
    }

    false
}

/// Normalize longitude to the range [0, 2π).
///
/// Used internally when handling polygons that cross the anti-meridian.
///
/// # Arguments
/// * `lon` - Longitude in radians
///
/// # Returns
/// Longitude normalized to [0, 2π)
fn normalize_longitude_positive(lon: f64) -> f64 {
    let two_pi = 2.0 * PI;
    let mut normalized = lon % two_pi;
    if normalized < 0.0 {
        normalized += two_pi;
    }
    normalized
}

/// Check if a point is inside a polygon using the ray-casting algorithm.
///
/// Implements the Jordan curve theorem by casting a ray from the test point
/// to positive infinity (along the latitude axis) and counting intersections
/// with polygon edges. An odd count indicates the point is inside.
///
/// # Arguments
/// * `lon` - Test point longitude in radians
/// * `lat` - Test point latitude in radians
/// * `vertices` - Polygon vertices as (longitude, latitude) pairs in radians.
///   The polygon should be closed (first vertex == last vertex).
///
/// # Returns
/// `true` if the point is inside the polygon, `false` otherwise
///
/// # Algorithm
/// Uses ray-casting (Jordan curve theorem):
/// 1. Cast a horizontal ray from the test point toward +∞ latitude
/// 2. Count intersections with polygon edges
/// 3. Odd count = inside, even count = outside
///
/// # Anti-Meridian Handling
/// Automatically detects and handles polygons that cross the ±180° longitude
/// boundary by normalizing all coordinates to the [0, 360°) range.
///
/// # Edge Cases
/// - Points exactly on the boundary may return either true or false
/// - Polygons containing the poles may produce incorrect results
/// - Self-intersecting polygons produce undefined results
///
/// # Example
/// ```
/// use brahe::coordinates::point_in_polygon;
/// use std::f64::consts::PI;
///
/// // Simple square polygon (10-20° lon, 10-20° lat)
/// let vertices = vec![
///     (10.0_f64.to_radians(), 10.0_f64.to_radians()),
///     (20.0_f64.to_radians(), 10.0_f64.to_radians()),
///     (20.0_f64.to_radians(), 20.0_f64.to_radians()),
///     (10.0_f64.to_radians(), 20.0_f64.to_radians()),
///     (10.0_f64.to_radians(), 10.0_f64.to_radians()),  // Closed
/// ];
///
/// // Point inside
/// assert!(point_in_polygon(15.0_f64.to_radians(), 15.0_f64.to_radians(), &vertices));
///
/// // Point outside
/// assert!(!point_in_polygon(5.0_f64.to_radians(), 15.0_f64.to_radians(), &vertices));
/// ```
pub fn point_in_polygon(lon: f64, lat: f64, vertices: &[(f64, f64)]) -> bool {
    if vertices.len() < 4 {
        // Need at least 3 unique vertices + closure
        return false;
    }

    // Check if we need to handle anti-meridian crossing
    let crosses_antimeridian = polygon_crosses_antimeridian(vertices);

    if crosses_antimeridian {
        // Normalize all coordinates to [0, 2π) range
        let normalized_vertices: Vec<(f64, f64)> = vertices
            .iter()
            .map(|(vlon, vlat)| (normalize_longitude_positive(*vlon), *vlat))
            .collect();
        let normalized_lon = normalize_longitude_positive(lon);

        point_in_polygon_internal(normalized_lon, lat, &normalized_vertices)
    } else {
        point_in_polygon_internal(lon, lat, vertices)
    }
}

/// Internal ray-casting implementation (assumes coordinates are already normalized).
fn point_in_polygon_internal(lon: f64, lat: f64, vertices: &[(f64, f64)]) -> bool {
    let n = vertices.len();
    if n < 4 {
        return false;
    }

    let mut inside = false;

    // Ray-casting algorithm
    // Cast a ray from (lon, lat) in the +lat direction
    // Count how many times it crosses polygon edges
    let mut j = n - 2; // Start with the second-to-last vertex (skip closure)

    for i in 0..n - 1 {
        // Skip the closure vertex
        let (xi, yi) = vertices[i];
        let (xj, yj) = vertices[j];

        // Check if the ray crosses this edge
        // The edge goes from (xj, yj) to (xi, yi)
        // We cast a ray in +lat direction from (lon, lat)
        let intersects =
            ((yi > lat) != (yj > lat)) && (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi);

        if intersects {
            inside = !inside;
        }

        j = i;
    }

    inside
}

#[cfg(test)]
#[allow(non_snake_case)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // Helper to convert degrees to radians for cleaner test code
    fn deg_to_rad(deg: f64) -> f64 {
        deg.to_radians()
    }

    // Helper to create a polygon from degree coordinates
    fn polygon_from_degrees(coords: &[(f64, f64)]) -> Vec<(f64, f64)> {
        coords
            .iter()
            .map(|(lon, lat)| (deg_to_rad(*lon), deg_to_rad(*lat)))
            .collect()
    }

    // =========================================================================
    // normalize_longitude_positive tests
    // =========================================================================

    #[test]
    fn test_normalize_longitude_positive_zero() {
        assert_abs_diff_eq!(normalize_longitude_positive(0.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_normalize_longitude_positive_positive() {
        assert_abs_diff_eq!(normalize_longitude_positive(PI), PI, epsilon = 1e-10);
    }

    #[test]
    fn test_normalize_longitude_positive_negative() {
        // -90° should become 270° (in radians: -π/2 → 3π/2)
        let result = normalize_longitude_positive(-PI / 2.0);
        assert_abs_diff_eq!(result, 3.0 * PI / 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_normalize_longitude_positive_wrap() {
        // 370° should become 10° (in radians)
        let result = normalize_longitude_positive(deg_to_rad(370.0));
        assert_abs_diff_eq!(result, deg_to_rad(10.0), epsilon = 1e-10);
    }

    #[test]
    fn test_normalize_longitude_positive_negative_wrap() {
        // -190° should become 170°
        let result = normalize_longitude_positive(deg_to_rad(-190.0));
        assert_abs_diff_eq!(result, deg_to_rad(170.0), epsilon = 1e-10);
    }

    // =========================================================================
    // polygon_crosses_antimeridian tests
    // =========================================================================

    #[test]
    fn test_polygon_crosses_antimeridian_empty() {
        let vertices: Vec<(f64, f64)> = vec![];
        assert!(!polygon_crosses_antimeridian(&vertices));
    }

    #[test]
    fn test_polygon_crosses_antimeridian_single() {
        let vertices = polygon_from_degrees(&[(10.0, 10.0)]);
        assert!(!polygon_crosses_antimeridian(&vertices));
    }

    #[test]
    fn test_polygon_crosses_antimeridian_simple_no_cross() {
        // Simple square in Europe - doesn't cross
        let vertices = polygon_from_degrees(&[
            (10.0, 50.0),
            (20.0, 50.0),
            (20.0, 55.0),
            (10.0, 55.0),
            (10.0, 50.0),
        ]);
        assert!(!polygon_crosses_antimeridian(&vertices));
    }

    #[test]
    fn test_polygon_crosses_antimeridian_crosses() {
        // Polygon spanning 170° to -170° (crosses anti-meridian)
        let vertices = polygon_from_degrees(&[
            (170.0, 10.0),
            (-170.0, 10.0),
            (-170.0, 20.0),
            (170.0, 20.0),
            (170.0, 10.0),
        ]);
        assert!(polygon_crosses_antimeridian(&vertices));
    }

    #[test]
    fn test_polygon_crosses_antimeridian_near_but_no_cross() {
        // Polygon from 170° to 175° - close to anti-meridian but doesn't cross
        let vertices = polygon_from_degrees(&[
            (170.0, 10.0),
            (175.0, 10.0),
            (175.0, 20.0),
            (170.0, 20.0),
            (170.0, 10.0),
        ]);
        assert!(!polygon_crosses_antimeridian(&vertices));
    }

    // =========================================================================
    // point_in_polygon tests - Simple cases
    // =========================================================================

    #[test]
    fn test_point_in_polygon_empty() {
        let vertices: Vec<(f64, f64)> = vec![];
        assert!(!point_in_polygon(0.0, 0.0, &vertices));
    }

    #[test]
    fn test_point_in_polygon_insufficient_vertices() {
        // Only 3 vertices (need 4 including closure)
        let vertices = polygon_from_degrees(&[(10.0, 10.0), (20.0, 10.0), (15.0, 20.0)]);
        assert!(!point_in_polygon(
            deg_to_rad(15.0),
            deg_to_rad(15.0),
            &vertices
        ));
    }

    #[test]
    fn test_point_in_polygon_square_inside() {
        // Simple square: 10-20° lon, 10-20° lat
        let vertices = polygon_from_degrees(&[
            (10.0, 10.0),
            (20.0, 10.0),
            (20.0, 20.0),
            (10.0, 20.0),
            (10.0, 10.0),
        ]);

        // Point at center (15°, 15°) - should be inside
        assert!(point_in_polygon(
            deg_to_rad(15.0),
            deg_to_rad(15.0),
            &vertices
        ));
    }

    #[test]
    fn test_point_in_polygon_square_outside_west() {
        let vertices = polygon_from_degrees(&[
            (10.0, 10.0),
            (20.0, 10.0),
            (20.0, 20.0),
            (10.0, 20.0),
            (10.0, 10.0),
        ]);

        // Point west of polygon (5°, 15°)
        assert!(!point_in_polygon(
            deg_to_rad(5.0),
            deg_to_rad(15.0),
            &vertices
        ));
    }

    #[test]
    fn test_point_in_polygon_square_outside_east() {
        let vertices = polygon_from_degrees(&[
            (10.0, 10.0),
            (20.0, 10.0),
            (20.0, 20.0),
            (10.0, 20.0),
            (10.0, 10.0),
        ]);

        // Point east of polygon (25°, 15°)
        assert!(!point_in_polygon(
            deg_to_rad(25.0),
            deg_to_rad(15.0),
            &vertices
        ));
    }

    #[test]
    fn test_point_in_polygon_square_outside_north() {
        let vertices = polygon_from_degrees(&[
            (10.0, 10.0),
            (20.0, 10.0),
            (20.0, 20.0),
            (10.0, 20.0),
            (10.0, 10.0),
        ]);

        // Point north of polygon (15°, 25°)
        assert!(!point_in_polygon(
            deg_to_rad(15.0),
            deg_to_rad(25.0),
            &vertices
        ));
    }

    #[test]
    fn test_point_in_polygon_square_outside_south() {
        let vertices = polygon_from_degrees(&[
            (10.0, 10.0),
            (20.0, 10.0),
            (20.0, 20.0),
            (10.0, 20.0),
            (10.0, 10.0),
        ]);

        // Point south of polygon (15°, 5°)
        assert!(!point_in_polygon(
            deg_to_rad(15.0),
            deg_to_rad(5.0),
            &vertices
        ));
    }

    // =========================================================================
    // point_in_polygon tests - Concave polygon
    // =========================================================================

    #[test]
    fn test_point_in_polygon_concave_inside() {
        // L-shaped polygon
        let vertices = polygon_from_degrees(&[
            (10.0, 10.0),
            (30.0, 10.0),
            (30.0, 20.0),
            (20.0, 20.0),
            (20.0, 30.0),
            (10.0, 30.0),
            (10.0, 10.0),
        ]);

        // Point in the main body (15°, 15°)
        assert!(point_in_polygon(
            deg_to_rad(15.0),
            deg_to_rad(15.0),
            &vertices
        ));

        // Point in the upper arm (15°, 25°)
        assert!(point_in_polygon(
            deg_to_rad(15.0),
            deg_to_rad(25.0),
            &vertices
        ));
    }

    #[test]
    fn test_point_in_polygon_concave_outside_in_concavity() {
        // L-shaped polygon - point in the concave region
        let vertices = polygon_from_degrees(&[
            (10.0, 10.0),
            (30.0, 10.0),
            (30.0, 20.0),
            (20.0, 20.0),
            (20.0, 30.0),
            (10.0, 30.0),
            (10.0, 10.0),
        ]);

        // Point in the concave corner (25°, 25°) - should be outside
        assert!(!point_in_polygon(
            deg_to_rad(25.0),
            deg_to_rad(25.0),
            &vertices
        ));
    }

    // =========================================================================
    // point_in_polygon tests - Anti-meridian crossing
    // =========================================================================

    #[test]
    fn test_point_in_polygon_antimeridian_inside() {
        // Polygon spanning 170° to -170° (20° wide across anti-meridian)
        let vertices = polygon_from_degrees(&[
            (170.0, 10.0),
            (-170.0, 10.0), // This is 190° when normalized
            (-170.0, 20.0),
            (170.0, 20.0),
            (170.0, 10.0),
        ]);

        // Point at 175°, 15° - inside the polygon
        assert!(point_in_polygon(
            deg_to_rad(175.0),
            deg_to_rad(15.0),
            &vertices
        ));

        // Point at -175° (185° normalized), 15° - inside the polygon
        assert!(point_in_polygon(
            deg_to_rad(-175.0),
            deg_to_rad(15.0),
            &vertices
        ));

        // Point at 180°, 15° - inside the polygon
        assert!(point_in_polygon(
            deg_to_rad(180.0),
            deg_to_rad(15.0),
            &vertices
        ));
    }

    #[test]
    fn test_point_in_polygon_antimeridian_outside() {
        // Polygon spanning 170° to -170°
        let vertices = polygon_from_degrees(&[
            (170.0, 10.0),
            (-170.0, 10.0),
            (-170.0, 20.0),
            (170.0, 20.0),
            (170.0, 10.0),
        ]);

        // Point at 160°, 15° - outside (west of polygon)
        assert!(!point_in_polygon(
            deg_to_rad(160.0),
            deg_to_rad(15.0),
            &vertices
        ));

        // Point at -160°, 15° - outside (east of polygon)
        assert!(!point_in_polygon(
            deg_to_rad(-160.0),
            deg_to_rad(15.0),
            &vertices
        ));

        // Point at 175°, 25° - outside (north of polygon)
        assert!(!point_in_polygon(
            deg_to_rad(175.0),
            deg_to_rad(25.0),
            &vertices
        ));
    }

    // =========================================================================
    // point_in_polygon tests - Southern hemisphere
    // =========================================================================

    #[test]
    fn test_point_in_polygon_southern_hemisphere() {
        // Polygon in southern hemisphere
        let vertices = polygon_from_degrees(&[
            (10.0, -20.0),
            (20.0, -20.0),
            (20.0, -10.0),
            (10.0, -10.0),
            (10.0, -20.0),
        ]);

        // Inside
        assert!(point_in_polygon(
            deg_to_rad(15.0),
            deg_to_rad(-15.0),
            &vertices
        ));

        // Outside
        assert!(!point_in_polygon(
            deg_to_rad(15.0),
            deg_to_rad(-25.0),
            &vertices
        ));
    }

    // =========================================================================
    // point_in_polygon tests - Western hemisphere (negative longitudes)
    // =========================================================================

    #[test]
    fn test_point_in_polygon_western_hemisphere() {
        // Polygon in western hemisphere (e.g., over USA)
        let vertices = polygon_from_degrees(&[
            (-120.0, 35.0),
            (-100.0, 35.0),
            (-100.0, 45.0),
            (-120.0, 45.0),
            (-120.0, 35.0),
        ]);

        // Inside
        assert!(point_in_polygon(
            deg_to_rad(-110.0),
            deg_to_rad(40.0),
            &vertices
        ));

        // Outside (east)
        assert!(!point_in_polygon(
            deg_to_rad(-90.0),
            deg_to_rad(40.0),
            &vertices
        ));
    }

    // =========================================================================
    // point_in_polygon tests - Triangular polygon
    // =========================================================================

    #[test]
    fn test_point_in_polygon_triangle() {
        // Triangle
        let vertices =
            polygon_from_degrees(&[(10.0, 10.0), (30.0, 10.0), (20.0, 30.0), (10.0, 10.0)]);

        // Inside (centroid area)
        assert!(point_in_polygon(
            deg_to_rad(20.0),
            deg_to_rad(15.0),
            &vertices
        ));

        // Outside (below base)
        assert!(!point_in_polygon(
            deg_to_rad(20.0),
            deg_to_rad(5.0),
            &vertices
        ));
    }

    // =========================================================================
    // point_in_polygon tests - Near poles
    // =========================================================================

    #[test]
    fn test_point_in_polygon_high_latitude() {
        // Polygon near (but not at) the north pole
        let vertices = polygon_from_degrees(&[
            (0.0, 80.0),
            (90.0, 80.0),
            (90.0, 85.0),
            (0.0, 85.0),
            (0.0, 80.0),
        ]);

        // Inside
        assert!(point_in_polygon(
            deg_to_rad(45.0),
            deg_to_rad(82.0),
            &vertices
        ));

        // Outside (too far south)
        assert!(!point_in_polygon(
            deg_to_rad(45.0),
            deg_to_rad(75.0),
            &vertices
        ));
    }
}
