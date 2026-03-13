/*!
 * Spherical geometry utilities for great circle computations
 *
 * This module provides general-purpose spherical geometry functions used
 * primarily by the tessellation system. All functions operate on unit sphere
 * vectors (normalized 3D vectors) unless otherwise noted.
 */

use nalgebra::Vector3;
use std::f64::consts::PI;

/// Rotate a vector around an axis by a given angle using Rodrigues' rotation formula.
///
/// # Arguments
/// * `vector` - The vector to rotate
/// * `axis` - The rotation axis (will be normalized internally)
/// * `angle` - Rotation angle in radians
///
/// # Returns
/// The rotated vector
///
/// # Examples
/// ```
/// use brahe::math::spherical_geometry::rodrigues_rotation;
/// use nalgebra::Vector3;
/// use approx::assert_abs_diff_eq;
///
/// // Rotate [1,0,0] around [0,0,1] by π/2 → [0,1,0]
/// let result = rodrigues_rotation(
///     &Vector3::new(1.0, 0.0, 0.0),
///     &Vector3::new(0.0, 0.0, 1.0),
///     std::f64::consts::FRAC_PI_2,
/// );
/// assert_abs_diff_eq!(result.x, 0.0, epsilon = 1e-12);
/// assert_abs_diff_eq!(result.y, 1.0, epsilon = 1e-12);
/// assert_abs_diff_eq!(result.z, 0.0, epsilon = 1e-12);
/// ```
pub fn rodrigues_rotation(vector: &Vector3<f64>, axis: &Vector3<f64>, angle: f64) -> Vector3<f64> {
    let k = axis.normalize();
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    vector * cos_a + k.cross(vector) * sin_a + k * k.dot(vector) * (1.0 - cos_a)
}

/// Test whether two great circle arcs on the unit sphere intersect.
///
/// Uses the plane normal method: compute the great circle plane normals,
/// find the candidate intersection point, and check if it lies within both arcs.
///
/// # Arguments
/// * `a1`, `a2` - Endpoints of the first arc (unit sphere vectors)
/// * `b1`, `b2` - Endpoints of the second arc (unit sphere vectors)
///
/// # Returns
/// `true` if the arcs intersect
pub fn great_circle_arcs_intersect(
    a1: &Vector3<f64>,
    a2: &Vector3<f64>,
    b1: &Vector3<f64>,
    b2: &Vector3<f64>,
) -> bool {
    // Compute great circle plane normals
    let n1 = a1.cross(a2);
    let n2 = b1.cross(b2);

    // If arcs are on the same great circle (parallel normals), treat as non-intersecting
    let cross = n1.cross(&n2);
    if cross.norm() < 1e-15 {
        return false;
    }

    // Candidate intersection point (and its antipode)
    let candidate = cross.normalize();

    // Check if candidate lies within both arcs using triple product signs
    // A point P is within arc A1-A2 if (A1×A2)·P has the same sign as (A1×A2)·(A1×A2),
    // meaning P is on the same side as the arc midpoint.
    // More precisely, P is within arc A1-A2 if:
    // sign((A1×P)·N1_perp) == sign((P×A2)·N1_perp) and both same sign as (A1×A2)·N1_perp
    //
    // Simpler check: P is on arc A1-A2 iff (A1×P)·(A1×A2) >= 0 and (A2×P)·(A2×A1) >= 0
    let within_a = |p: &Vector3<f64>| -> bool {
        let t1 = a1.cross(p).dot(&n1);
        let t2 = a2.cross(p).dot(&n1);
        t1 >= -1e-12 && t2 <= 1e-12
    };

    let within_b = |p: &Vector3<f64>| -> bool {
        let t1 = b1.cross(p).dot(&n2);
        let t2 = b2.cross(p).dot(&n2);
        t1 >= -1e-12 && t2 <= 1e-12
    };

    // Check both the candidate and its antipode
    let antipode = -candidate;
    (within_a(&candidate) && within_b(&candidate)) || (within_a(&antipode) && within_b(&antipode))
}

/// Find the intersection point of two great circle arcs if they intersect.
///
/// When arcs lie on distinct great circles, their planes intersect along a line.
/// The `hemisphere_hint` vector is used to choose which of the two antipodal
/// intersection points to return (the one on the same hemisphere as the hint).
///
/// # Arguments
/// * `a1`, `a2` - Endpoints of the first arc (unit sphere vectors)
/// * `b1`, `b2` - Endpoints of the second arc (unit sphere vectors)
/// * `hemisphere_hint` - Preference vector to disambiguate antipodal points
///
/// # Returns
/// `Some(point)` if the arcs intersect, `None` otherwise
pub fn great_circle_arc_intersection(
    a1: &Vector3<f64>,
    a2: &Vector3<f64>,
    b1: &Vector3<f64>,
    b2: &Vector3<f64>,
    hemisphere_hint: &Vector3<f64>,
) -> Option<Vector3<f64>> {
    let n1 = a1.cross(a2);
    let n2 = b1.cross(b2);

    let cross = n1.cross(&n2);
    if cross.norm() < 1e-15 {
        return None;
    }

    let mut n3 = cross.normalize();

    // Choose hemisphere using hint
    if hemisphere_hint.dot(&n3) < 0.0 {
        n3 = -n3;
    }

    // Verify the point lies within both arcs
    let within_a = {
        let t1 = a1.cross(&n3).dot(&n1);
        let t2 = a2.cross(&n3).dot(&n1);
        t1 >= -1e-10 && t2 <= 1e-10
    };

    let within_b = {
        let t1 = b1.cross(&n3).dot(&n2);
        let t2 = b2.cross(&n3).dot(&n2);
        t1 >= -1e-10 && t2 <= 1e-10
    };

    if within_a && within_b { Some(n3) } else { None }
}

/// Find all intersection points between a great circle arc and a polygon on the unit sphere.
///
/// Iterates polygon edges and collects all intersection points with the given arc.
///
/// # Arguments
/// * `arc_start`, `arc_end` - Endpoints of the arc (unit sphere vectors)
/// * `polygon_vertices` - Polygon vertices on unit sphere (should be closed: first == last)
/// * `hemisphere_hint` - Preference vector for hemisphere disambiguation
///
/// # Returns
/// Vector of intersection points on the unit sphere
pub fn great_circle_arc_polygon_intersections(
    arc_start: &Vector3<f64>,
    arc_end: &Vector3<f64>,
    polygon_vertices: &[Vector3<f64>],
    hemisphere_hint: &Vector3<f64>,
) -> Vec<Vector3<f64>> {
    let mut intersections = Vec::new();
    if polygon_vertices.len() < 2 {
        return intersections;
    }

    let a1 = arc_start.normalize();
    let a2 = arc_end.normalize();

    for i in 0..polygon_vertices.len() - 1 {
        let b1 = polygon_vertices[i].normalize();
        let b2 = polygon_vertices[i + 1].normalize();

        if let Some(pt) = great_circle_arc_intersection(&a1, &a2, &b1, &b2, hemisphere_hint) {
            intersections.push(pt);
        }
    }

    intersections
}

/// Project a point onto a great circle arc and compute the angular distance.
///
/// The projection is the closest point on the great circle containing the arc
/// to the given point. The angular distance is the angle between the original
/// point and its projection on the unit sphere.
///
/// # Arguments
/// * `point` - Point on the unit sphere to project
/// * `arc_start` - Start of the great circle arc (unit sphere)
/// * `arc_end` - End of the great circle arc (unit sphere)
///
/// # Returns
/// `(projected_point, angular_distance)` — projected point is on the unit sphere
pub fn cross_track_projection(
    point: &Vector3<f64>,
    arc_start: &Vector3<f64>,
    arc_end: &Vector3<f64>,
) -> (Vector3<f64>, f64) {
    let p = point.normalize();
    let a = arc_start.normalize();
    let b = arc_end.normalize();

    // Normal to the great circle plane containing the arc
    let n = a.cross(&b);
    if n.norm() < 1e-15 {
        // Degenerate: arc_start == arc_end, project to that point
        return (a, p.dot(&a).clamp(-1.0, 1.0).acos());
    }

    // Project: D = normalize( (A×B) × C × (A×B) )
    // Simplified: project P onto the great circle plane, then normalize
    let d = n.cross(&p).cross(&n);
    if d.norm() < 1e-15 {
        // Point is at a pole of the great circle
        return (a, PI / 2.0);
    }

    let d = d.normalize();
    let ang = p.dot(&d).clamp(-1.0, 1.0).acos();

    (d, ang)
}

/// Compute the circumscription angle of a set of vertices on the unit sphere.
///
/// Uses Jung's theorem: for a set of points on a sphere, the circumscribed
/// radius is at most `arcsin(sqrt(3/2) * sin(max_pairwise_distance/2))`.
/// For small angles this simplifies to `sqrt(3) * 2/3 * max_pairwise_angle`.
///
/// # Arguments
/// * `vertices` - Points on the unit sphere
///
/// # Returns
/// Circumscription angle in radians
pub fn polygon_circumscription_angle(vertices: &[Vector3<f64>]) -> f64 {
    let n = vertices.len();
    if n < 2 {
        return 0.0;
    }

    let mut max_angle = 0.0_f64;
    for (i, vi) in vertices.iter().enumerate() {
        let vi = vi.normalize();
        for vj in &vertices[(i + 1)..] {
            let vj = vj.normalize();
            let angle = vi.dot(&vj).clamp(-1.0, 1.0).acos();
            max_angle = max_angle.max(angle);
        }
    }

    // Jung's theorem factor for spherical geometry
    (3.0_f64).sqrt() * 2.0 / 3.0 * max_angle
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn test_rodrigues_rotation_90_degrees() {
        // Rotate [1,0,0] around [0,0,1] by π/2 → [0,1,0]
        let result = rodrigues_rotation(
            &Vector3::new(1.0, 0.0, 0.0),
            &Vector3::new(0.0, 0.0, 1.0),
            FRAC_PI_2,
        );
        assert_abs_diff_eq!(result.x, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(result.y, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(result.z, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_rodrigues_rotation_identity() {
        // Zero angle → no change
        let v = Vector3::new(0.5, 0.3, 0.8);
        let result = rodrigues_rotation(&v, &Vector3::new(0.0, 0.0, 1.0), 0.0);
        assert_abs_diff_eq!(result.x, v.x, epsilon = 1e-12);
        assert_abs_diff_eq!(result.y, v.y, epsilon = 1e-12);
        assert_abs_diff_eq!(result.z, v.z, epsilon = 1e-12);
    }

    #[test]
    fn test_rodrigues_rotation_180_degrees() {
        // Rotate [1,0,0] around [0,0,1] by π → [-1,0,0]
        let result = rodrigues_rotation(
            &Vector3::new(1.0, 0.0, 0.0),
            &Vector3::new(0.0, 0.0, 1.0),
            PI,
        );
        assert_abs_diff_eq!(result.x, -1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(result.y, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(result.z, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_great_circle_arcs_intersect_orthogonal() {
        // Arc along equator (x to y) intersects arc from south pole to north pole region
        let a1 = Vector3::new(1.0, 0.0, 0.0);
        let a2 = Vector3::new(0.0, 1.0, 0.0);
        let b1 = Vector3::new(0.5, 0.5, -0.707).normalize();
        let b2 = Vector3::new(0.5, 0.5, 0.707).normalize();

        assert!(great_circle_arcs_intersect(&a1, &a2, &b1, &b2));
    }

    #[test]
    fn test_great_circle_arcs_no_intersect() {
        // Two arcs on the same latitude but not overlapping
        let a1 = Vector3::new(1.0, 0.0, 0.0);
        let a2 = Vector3::new(0.0, 1.0, 0.0);
        let b1 = Vector3::new(-1.0, 0.0, 0.0);
        let b2 = Vector3::new(0.0, -1.0, 0.0);

        assert!(!great_circle_arcs_intersect(&a1, &a2, &b1, &b2));
    }

    #[test]
    fn test_great_circle_arc_intersection_at_known_point() {
        // Arc from +x to +y on equator, arc from -z to +z through (1,1,0)/sqrt(2)
        let a1 = Vector3::new(1.0, 0.0, 0.0);
        let a2 = Vector3::new(0.0, 1.0, 0.0);
        let midpoint = Vector3::new(1.0, 1.0, 0.0).normalize();
        let b1 = Vector3::new(midpoint.x, midpoint.y, -0.5).normalize();
        let b2 = Vector3::new(midpoint.x, midpoint.y, 0.5).normalize();

        let hint = Vector3::new(1.0, 1.0, 0.0);
        let result = great_circle_arc_intersection(&a1, &a2, &b1, &b2, &hint);
        assert!(result.is_some());
        let pt = result.unwrap();
        assert_abs_diff_eq!(pt.x, midpoint.x, epsilon = 1e-10);
        assert_abs_diff_eq!(pt.y, midpoint.y, epsilon = 1e-10);
        assert_abs_diff_eq!(pt.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_great_circle_arc_intersection_parallel() {
        // Two arcs on the same great circle → None
        let a1 = Vector3::new(1.0, 0.0, 0.0);
        let a2 = Vector3::new(0.0, 1.0, 0.0);
        let b1 = Vector3::new(-1.0, 0.0, 0.0);
        let b2 = Vector3::new(0.0, -1.0, 0.0);

        let hint = Vector3::new(0.0, 0.0, 1.0);
        let result = great_circle_arc_intersection(&a1, &a2, &b1, &b2, &hint);
        assert!(result.is_none());
    }

    #[test]
    fn test_cross_track_projection_on_arc() {
        // Point on the equator, arc on the equator → distance = 0
        let p = Vector3::new(1.0, 1.0, 0.0).normalize();
        let a = Vector3::new(1.0, 0.0, 0.0);
        let b = Vector3::new(0.0, 1.0, 0.0);

        let (proj, dist) = cross_track_projection(&p, &a, &b);
        assert_abs_diff_eq!(dist, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(proj.x, p.x, epsilon = 1e-10);
        assert_abs_diff_eq!(proj.y, p.y, epsilon = 1e-10);
    }

    #[test]
    fn test_cross_track_projection_off_arc() {
        // Point at north pole, arc on equator → distance = π/2
        let p = Vector3::new(0.0, 0.0, 1.0);
        let a = Vector3::new(1.0, 0.0, 0.0);
        let b = Vector3::new(0.0, 1.0, 0.0);

        let (_proj, dist) = cross_track_projection(&p, &a, &b);
        assert_abs_diff_eq!(dist, FRAC_PI_2, epsilon = 1e-10);
    }

    #[test]
    fn test_polygon_circumscription_angle() {
        // Three points forming an equilateral triangle on the equator
        // at 0°, 120°, 240° longitude
        let v1 = Vector3::new(1.0, 0.0, 0.0);
        let v2 = Vector3::new((2.0 * PI / 3.0).cos(), (2.0 * PI / 3.0).sin(), 0.0);
        let v3 = Vector3::new((4.0 * PI / 3.0).cos(), (4.0 * PI / 3.0).sin(), 0.0);

        let angle = polygon_circumscription_angle(&[v1, v2, v3]);
        // Max pairwise angle is 2π/3, circumscription = sqrt(3) * 2/3 * 2π/3
        let expected = (3.0_f64).sqrt() * 2.0 / 3.0 * (2.0 * PI / 3.0);
        assert_abs_diff_eq!(angle, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_polygon_circumscription_single_point() {
        let v1 = Vector3::new(1.0, 0.0, 0.0);
        assert_abs_diff_eq!(polygon_circumscription_angle(&[v1]), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_great_circle_arc_polygon_intersections() {
        // Square polygon on equator from (1,0,0) CCW
        let v1 = Vector3::new(1.0, 0.0, 0.1).normalize();
        let v2 = Vector3::new(1.0, 0.0, -0.1).normalize();
        let v3 = Vector3::new(0.9, 0.1, -0.1).normalize();
        let v4 = Vector3::new(0.9, 0.1, 0.1).normalize();
        let polygon = vec![v1, v2, v3, v4, v1]; // Closed

        // Arc passing through the polygon
        let arc_start = Vector3::new(1.0, 0.0, 0.5).normalize();
        let arc_end = Vector3::new(1.0, 0.0, -0.5).normalize();
        let hint = Vector3::new(1.0, 0.0, 0.0);

        let intersections =
            great_circle_arc_polygon_intersections(&arc_start, &arc_end, &polygon, &hint);
        // Should intersect 2 edges (top and bottom of the square)
        assert_eq!(intersections.len(), 2);
    }
}
