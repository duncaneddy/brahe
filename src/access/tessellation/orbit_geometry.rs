/*!
 * Orbit geometry tessellator
 *
 * Tessellates locations into rectangular tiles aligned with satellite ground tracks
 * using orbital mechanics to determine along-track and cross-track directions.
 *
 * The algorithm:
 * 1. Compute satellite along-track direction at the target latitude
 * 2. For point targets: create a single centered tile per direction
 * 3. For polygon targets: divide into cross-track strips, then along-track tiles
 */

use std::collections::HashMap;
use std::f64::consts::PI;

use nalgebra::Vector3;
use serde_json::{Value as JsonValue, json};
use uuid::Uuid;

use crate::access::constraints::AscDsc;
use crate::access::location::{AccessibleLocation, PointLocation, PolygonLocation};
use crate::access::tessellation::Tessellator;
use crate::attitude::RotationMatrix;
use crate::constants::{AngleFormat, DEG2RAD, R_EARTH, RAD2DEG};
use crate::coordinates::{position_ecef_to_geodetic, position_geodetic_to_ecef};
use crate::math::spherical_geometry::{
    cross_track_projection, great_circle_arc_polygon_intersections, polygon_circumscription_angle,
    rodrigues_rotation,
};
use crate::orbits::keplerian::{anomaly_eccentric_to_mean, orbital_period};
use crate::utils::errors::BraheError;
use crate::utils::state_providers::SOrbitStateProvider;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the orbit geometry tessellator.
///
/// Controls tile dimensions, overlaps, and ascending/descending pass selection.
/// All dimensions are in meters.
#[derive(Debug, Clone)]
pub struct OrbitGeometryTessellatorConfig {
    /// Desired cross-track tile width (meters)
    pub image_width: f64,
    /// Desired along-track tile length (meters)
    pub image_length: f64,
    /// Cross-track overlap between adjacent strips (meters)
    pub crosstrack_overlap: f64,
    /// Along-track overlap between adjacent tiles (meters)
    pub alongtrack_overlap: f64,
    /// Which orbital passes to use for tiling directions
    pub asc_dsc: AscDsc,
    /// Minimum allowed along-track tile length (meters)
    pub min_image_length: f64,
    /// Maximum allowed along-track tile length (meters)
    pub max_image_length: f64,
}

impl Default for OrbitGeometryTessellatorConfig {
    fn default() -> Self {
        Self {
            image_width: 5000.0,
            image_length: 5000.0,
            crosstrack_overlap: 200.0,
            alongtrack_overlap: 200.0,
            asc_dsc: AscDsc::Either,
            min_image_length: 5000.0,
            max_image_length: 5000.0,
        }
    }
}

impl OrbitGeometryTessellatorConfig {
    /// Create a new configuration with specified tile dimensions.
    ///
    /// # Arguments
    /// * `image_width` - Cross-track tile width (meters)
    /// * `image_length` - Along-track tile length (meters)
    pub fn new(image_width: f64, image_length: f64) -> Self {
        Self {
            image_width,
            image_length,
            ..Default::default()
        }
    }

    /// Set the cross-track overlap (builder pattern)
    pub fn with_crosstrack_overlap(mut self, overlap: f64) -> Self {
        self.crosstrack_overlap = overlap;
        self
    }

    /// Set the along-track overlap (builder pattern)
    pub fn with_alongtrack_overlap(mut self, overlap: f64) -> Self {
        self.alongtrack_overlap = overlap;
        self
    }

    /// Set the ascending/descending pass selection (builder pattern)
    pub fn with_asc_dsc(mut self, asc_dsc: AscDsc) -> Self {
        self.asc_dsc = asc_dsc;
        self
    }

    /// Set the minimum along-track tile length (builder pattern)
    pub fn with_min_image_length(mut self, min_len: f64) -> Self {
        self.min_image_length = min_len;
        self
    }

    /// Set the maximum along-track tile length (builder pattern)
    pub fn with_max_image_length(mut self, max_len: f64) -> Self {
        self.max_image_length = max_len;
        self
    }
}

// ============================================================================
// Tessellator
// ============================================================================

/// Tessellator that uses orbital geometry to create rectangular tiles aligned
/// with satellite ground tracks.
///
/// Uses the satellite's orbital elements to compute along-track directions at
/// the target latitude, then tiles the area perpendicular and parallel to the
/// ground track.
pub struct OrbitGeometryTessellator {
    propagator: Box<dyn SOrbitStateProvider + Send + Sync>,
    config: OrbitGeometryTessellatorConfig,
    spacecraft_id: Option<String>,
    epoch: crate::time::Epoch,
}

impl OrbitGeometryTessellator {
    /// Create a new orbit geometry tessellator.
    ///
    /// # Arguments
    /// * `propagator` - Orbit state provider (e.g., SGPPropagator, SOrbitTrajectory)
    /// * `epoch` - Reference epoch for the propagator (typically TLE epoch)
    /// * `config` - Tessellation configuration
    /// * `spacecraft_id` - Optional spacecraft identifier for tile metadata
    pub fn new(
        propagator: Box<dyn SOrbitStateProvider + Send + Sync>,
        epoch: crate::time::Epoch,
        config: OrbitGeometryTessellatorConfig,
        spacecraft_id: Option<String>,
    ) -> Self {
        Self {
            propagator,
            epoch,
            config,
            spacecraft_id,
        }
    }

    /// Get the tessellator configuration.
    pub fn config(&self) -> &OrbitGeometryTessellatorConfig {
        &self.config
    }

    /// Get the reference epoch.
    pub fn epoch(&self) -> crate::time::Epoch {
        self.epoch
    }
}

impl Tessellator for OrbitGeometryTessellator {
    fn tessellate(
        &self,
        location: &dyn AccessibleLocation,
    ) -> Result<Vec<PolygonLocation>, BraheError> {
        if let Some(point) = location.as_any().downcast_ref::<PointLocation>() {
            self.tessellate_point(point)
        } else if let Some(polygon) = location.as_any().downcast_ref::<PolygonLocation>() {
            self.tessellate_polygon(polygon)
        } else {
            Err(BraheError::Error(
                "Unsupported location type for tessellation".to_string(),
            ))
        }
    }

    fn name(&self) -> &str {
        "OrbitGeometryTessellator"
    }
}

// ============================================================================
// Core Algorithm Implementation
// ============================================================================

impl OrbitGeometryTessellator {
    // ---- Along-Track Direction Computation ----

    /// Compute along-track direction vectors at the target latitude.
    ///
    /// Uses analytical half-orbit computation to find ascending/descending windows,
    /// then finds the latitude crossing within each window and extracts the
    /// along-track direction projected to the surface tangent plane.
    fn compute_along_track_directions(
        &self,
        center_geodetic: &Vector3<f64>,
    ) -> Result<Vec<Vector3<f64>>, BraheError> {
        let lon = center_geodetic.x; // degrees
        let lat = center_geodetic.y; // degrees

        let want_asc =
            self.config.asc_dsc == AscDsc::Ascending || self.config.asc_dsc == AscDsc::Either;
        let want_dsc =
            self.config.asc_dsc == AscDsc::Descending || self.config.asc_dsc == AscDsc::Either;

        // Get orbital elements and epoch from propagator
        let epoch = self.epoch;
        let elements = self.propagator.state_koe_osc(epoch, AngleFormat::Degrees)?;
        // elements = [a, e, i, raan, argp, M] in degrees for angles

        let a = elements[0]; // semi-major axis (m)
        let e = elements[1]; // eccentricity
        let w = elements[4]; // argument of perigee (degrees)
        let m_anom = elements[5]; // mean anomaly (degrees)

        // Compute orbital period and mean motion
        let period = orbital_period(a);
        let n = 2.0 * PI / period; // rad/s

        // Compute half-orbit windows analytically
        // Find mean anomaly at argument of latitude = 90° (descending start)
        let e_target = (((1.0 - e * e).sqrt() * (90.0_f64 * DEG2RAD).sin())
            / ((90.0_f64 * DEG2RAD).cos() + e))
            .atan();
        let m_target = anomaly_eccentric_to_mean(e_target, e, AngleFormat::Radians) * RAD2DEG;

        // Time to descending node
        let dt = (m_target - w - m_anom) * DEG2RAD / n;

        let epc_dsc_start = epoch + dt;
        let epc_dsc_end = epoch + dt + period / 2.0;
        let epc_asc_start = epc_dsc_end;
        let epc_asc_end = epc_asc_start + period / 2.0;

        let mut at_dirs = Vec::new();

        if want_asc
            && let Ok(dir) = self.compute_direction_for_window(lat, lon, epc_asc_start, epc_asc_end)
        {
            at_dirs.push(dir);
        }

        if want_dsc
            && let Ok(dir) = self.compute_direction_for_window(lat, lon, epc_dsc_start, epc_dsc_end)
        {
            at_dirs.push(dir);
        }

        if at_dirs.is_empty() {
            return Err(BraheError::Error(
                "No along-track directions found for the given location and orbit".to_string(),
            ));
        }

        Ok(at_dirs)
    }

    /// Compute along-track direction for a single half-orbit window.
    fn compute_direction_for_window(
        &self,
        target_lat: f64,
        target_lon: f64,
        epc_start: crate::time::Epoch,
        epc_end: crate::time::Epoch,
    ) -> Result<Vector3<f64>, BraheError> {
        // Find latitude crossing within window
        let (crossing_epoch, _) = self.find_latitude_crossing(target_lat, epc_start, epc_end)?;

        // Get ECEF state at crossing
        let state_ecef = self.propagator.state_ecef(crossing_epoch)?;
        let pos = Vector3::new(state_ecef[0], state_ecef[1], state_ecef[2]);
        let vel = Vector3::new(state_ecef[3], state_ecef[4], state_ecef[5]);

        // Extract along-track direction: normalize velocity, remove radial component
        let r_hat = pos.normalize();
        let mut at_dir = vel.normalize();
        at_dir -= r_hat * r_hat.dot(&at_dir);
        at_dir = at_dir.normalize();

        // Get satellite longitude
        let sat_geod = position_ecef_to_geodetic(pos, AngleFormat::Degrees);
        let sat_lon = sat_geod.x; // degrees

        // Rotate direction from satellite longitude to target longitude
        let rz_to = RotationMatrix::Rz(-target_lon, AngleFormat::Degrees);
        let rz_from = RotationMatrix::Rz(sat_lon, AngleFormat::Degrees);
        at_dir = rz_to * (rz_from * at_dir);
        at_dir = at_dir.normalize();

        Ok(at_dir)
    }

    /// Find the epoch when the satellite crosses a target latitude within a time window.
    ///
    /// Uses iterative convergence with adaptive step size.
    fn find_latitude_crossing(
        &self,
        target_lat: f64,
        epc_start: crate::time::Epoch,
        epc_end: crate::time::Epoch,
    ) -> Result<(crate::time::Epoch, Vector3<f64>), BraheError> {
        let tol = 0.001; // degrees
        let max_iterations = 200;
        let mut timestep: f64 = 60.0; // seconds

        let mut epc = epc_start;

        // Determine ascending/descending from midpoint
        let mid_epoch = epc_start + (epc_end - epc_start) / 2.0;
        let ascdir: f64 = if self.is_ascending_at(mid_epoch)? {
            1.0
        } else {
            -1.0
        };

        let mut timedir = 1.0;
        let mut lat_err = self.latitude_error(target_lat, epc)?;
        let mut invalid_its = 0;
        let mut iterations = 0;

        while lat_err.abs() > tol && iterations < max_iterations {
            iterations += 1;

            // Set step direction to reduce error
            timestep = ascdir * timedir * lat_err.signum() * timestep.abs();

            let epc_past = epc;
            epc += timestep;

            lat_err = self.latitude_error(target_lat, epc)?;

            // If we've overshot, halve the step
            if ascdir * lat_err.signum() != timestep.signum() {
                timestep /= 2.0;
            }

            // Check bounds
            if epc < epc_start || epc > epc_end {
                // Check convergence
                let lat_step = self.latitude_step(epc_past, epc)?;
                if lat_step.abs() < 1e-10 {
                    let state = self.propagator.state_ecef(epc)?;
                    let pos = Vector3::new(state[0], state[1], state[2]);
                    return Ok((epc, pos));
                }

                if invalid_its > 50 {
                    return Err(BraheError::Error(
                        "No latitude crossing found in window".to_string(),
                    ));
                }

                // Reverse and halve
                if ascdir * lat_err.signum() == timestep.signum() {
                    timedir = -1.0;
                    timestep /= 2.0;
                }
                invalid_its += 1;
            } else {
                timedir = 1.0;
                invalid_its = 0;
            }
        }

        let state = self.propagator.state_ecef(epc)?;
        let pos = Vector3::new(state[0], state[1], state[2]);
        Ok((epc, pos))
    }

    /// Compute latitude error at a given epoch.
    fn latitude_error(
        &self,
        target_lat: f64,
        epoch: crate::time::Epoch,
    ) -> Result<f64, BraheError> {
        let state = self.propagator.state_ecef(epoch)?;
        let pos = Vector3::new(state[0], state[1], state[2]);
        let geod = position_ecef_to_geodetic(pos, AngleFormat::Degrees);
        Ok(target_lat - geod.y)
    }

    /// Compute latitude change between two epochs.
    fn latitude_step(
        &self,
        epc_past: crate::time::Epoch,
        epc: crate::time::Epoch,
    ) -> Result<f64, BraheError> {
        let s1 = self.propagator.state_ecef(epc_past)?;
        let s2 = self.propagator.state_ecef(epc)?;
        let g1 = position_ecef_to_geodetic(Vector3::new(s1[0], s1[1], s1[2]), AngleFormat::Degrees);
        let g2 = position_ecef_to_geodetic(Vector3::new(s2[0], s2[1], s2[2]), AngleFormat::Degrees);
        Ok(g2.y - g1.y)
    }

    /// Check if the satellite is ascending at a given epoch.
    fn is_ascending_at(&self, epoch: crate::time::Epoch) -> Result<bool, BraheError> {
        let dt = 1.0; // 1 second
        let s1 = self.propagator.state_ecef(epoch)?;
        let s2 = self.propagator.state_ecef(epoch + dt)?;
        let g1 = position_ecef_to_geodetic(Vector3::new(s1[0], s1[1], s1[2]), AngleFormat::Degrees);
        let g2 = position_ecef_to_geodetic(Vector3::new(s2[0], s2[1], s2[2]), AngleFormat::Degrees);
        Ok(g2.y > g1.y)
    }

    // ---- Point Tessellation ----

    /// Tessellate a point location into one tile per direction.
    fn tessellate_point(&self, point: &PointLocation) -> Result<Vec<PolygonLocation>, BraheError> {
        let center_geod = point.center_geodetic();
        let at_dirs = self.compute_along_track_directions(&center_geod)?;

        let mut tiles = Vec::new();
        for dir in &at_dirs {
            let tile = self.tile_point_direction(point, dir)?;
            tiles.push(tile);
        }

        // Latitude merging: if we have 2 tiles and they're at high latitude,
        // the asc/dsc directions converge and one can be dropped
        if tiles.len() == 2 {
            tiles = self.merge_tiles_latitude(tiles)?;
        }

        Ok(tiles)
    }

    /// Create a single tile centered on a point for one direction.
    fn tile_point_direction(
        &self,
        point: &PointLocation,
        direction: &Vector3<f64>,
    ) -> Result<PolygonLocation, BraheError> {
        let center_ecef = point.center_ecef();
        let sgcp = center_ecef.normalize();
        let alt = point.center_geodetic().z;

        let ct_ang = self.config.image_width / R_EARTH;
        let at_ang = self.config.image_length / R_EARTH;

        // Cross-track edges
        let ct_max = rodrigues_rotation(&sgcp, direction, ct_ang / 2.0).normalize();
        let ct_min = rodrigues_rotation(&sgcp, direction, -ct_ang / 2.0).normalize();

        // Along-track axis
        let axis = sgcp.cross(direction);

        // Four corners
        let pnt1 = rodrigues_rotation(&ct_max, &axis, at_ang / 2.0);
        let pnt2 = rodrigues_rotation(&ct_min, &axis, at_ang / 2.0);
        let pnt3 = rodrigues_rotation(&ct_min, &axis, -at_ang / 2.0);
        let pnt4 = rodrigues_rotation(&ct_max, &axis, -at_ang / 2.0);

        create_tile_from_sphere_points(
            &[pnt1, pnt2, pnt3, pnt4],
            alt,
            direction,
            self.config.image_width,
            self.config.image_length,
            &self.spacecraft_id,
            None,
        )
    }

    /// Merge ascending/descending tiles at high latitude where directions converge.
    fn merge_tiles_latitude(
        &self,
        tiles: Vec<PolygonLocation>,
    ) -> Result<Vec<PolygonLocation>, BraheError> {
        if tiles.len() < 2 {
            return Ok(tiles);
        }

        // Get orbital inclination
        let epoch = self.epoch;
        let elements = self.propagator.state_koe_osc(epoch, AngleFormat::Degrees)?;
        let incl = elements[2];
        let ref_incl = if incl > 90.0 {
            incl + (180.0 - 2.0 * incl)
        } else {
            incl
        };

        // Check if first tile center latitude is near the inclination limit
        let center_lat = tiles[0].center_geodetic().y.abs();
        let offset_deg = 5.0;

        if center_lat > ref_incl - offset_deg {
            // Check if second tile's direction is skew-mergable with first
            let dir0 = get_tile_direction(&tiles[0]);
            let dir1 = get_tile_direction(&tiles[1]);

            if let (Some(_d0), Some(d1)) = (dir0, dir1)
                && skew_mergable_check(
                    &tiles[0],
                    &d1,
                    self.config.alongtrack_overlap,
                    self.config.crosstrack_overlap,
                )
            {
                return Ok(vec![tiles[0].clone()]);
            }
        }

        Ok(tiles)
    }

    // ---- Polygon Tessellation ----

    /// Tessellate a polygon into multiple cross-track strips and along-track tiles.
    fn tessellate_polygon(
        &self,
        polygon: &PolygonLocation,
    ) -> Result<Vec<PolygonLocation>, BraheError> {
        let center_geod = polygon.center_geodetic();
        let at_dirs = self.compute_along_track_directions(&center_geod)?;

        let mut all_tiles = Vec::new();

        for dir in &at_dirs {
            let mut dir_tiles = self.tile_direction(polygon, dir)?;

            // Single-tile polygons: force to requested image length
            if dir_tiles.len() == 1 {
                dir_tiles = self.force_requested_tile_len(dir_tiles, dir)?;
            }

            all_tiles.extend(dir_tiles);
        }

        Ok(all_tiles)
    }

    /// Tile a polygon along a single direction with multiple cross-track strips.
    fn tile_direction(
        &self,
        polygon: &PolygonLocation,
        direction: &Vector3<f64>,
    ) -> Result<Vec<PolygonLocation>, BraheError> {
        let tile_group_id = Uuid::new_v4().to_string();

        // Compute cross-track width
        let (ct_width, ct_min_dist, _ct_max_dist) = compute_crosstrack_width(polygon, direction)?;

        let image_width = self.config.image_width;
        let ct_overlap = self.config.crosstrack_overlap;

        // Number of cross-track strips
        let num_ct =
            ((ct_width - image_width).max(0.0) / (image_width - ct_overlap)).ceil() as usize + 1;

        // Center the collection
        let tiling_width = (image_width - ct_overlap) * num_ct as f64 + ct_overlap;
        let excess_width = tiling_width - ct_width;

        let mut ct_offset = ct_min_dist - excess_width / 2.0 + image_width / 2.0;

        let mut tiles = Vec::new();
        let center_ecef = polygon.center_ecef().normalize();
        let alt = polygon.center_geodetic().z;

        for _ in 0..num_ct {
            let ct_angle = ct_offset / R_EARTH;

            // Rotate center to cross-track position
            let ct_pnt = rodrigues_rotation(&center_ecef, direction, ct_angle).normalize();

            // Create along-track tiles for this strip
            let strip_tiles =
                self.create_tiling_rectangle(polygon, direction, &ct_pnt, alt, &tile_group_id)?;

            tiles.extend(strip_tiles);

            ct_offset += image_width - ct_overlap;
        }

        Ok(tiles)
    }

    /// Create along-track tiles for a single cross-track strip.
    fn create_tiling_rectangle(
        &self,
        polygon: &PolygonLocation,
        direction: &Vector3<f64>,
        center_point: &Vector3<f64>,
        alt: f64,
        tile_group_id: &str,
    ) -> Result<Vec<PolygonLocation>, BraheError> {
        let strip_width = self.config.image_width;
        let strip_angle = strip_width / R_EARTH;

        // Circumscription angle for maximum extent
        let polygon_ecef_verts = polygon_vertices_to_unit_sphere(polygon)?;
        let max_angle = polygon_circumscription_angle(&polygon_ecef_verts);

        let n_vec = center_point.cross(direction).normalize();

        // Left/right edge center points
        let ct_pnt_l = rodrigues_rotation(center_point, direction, -strip_angle / 2.0).normalize();
        let ct_pnt_r = rodrigues_rotation(center_point, direction, strip_angle / 2.0).normalize();

        // Extend edges to maximum extent
        let l_fd = rodrigues_rotation(&ct_pnt_l, &n_vec, max_angle / 2.0);
        let l_bk = rodrigues_rotation(&ct_pnt_l, &n_vec, -max_angle / 2.0);
        let r_fd = rodrigues_rotation(&ct_pnt_r, &n_vec, max_angle / 2.0);
        let r_bk = rodrigues_rotation(&ct_pnt_r, &n_vec, -max_angle / 2.0);

        // Find along-track bounds (handles concavity)
        let tile_bounds = self.find_alongtrack_bounds(
            polygon,
            direction,
            strip_angle,
            (&l_bk, &l_fd),
            (&r_bk, &r_fd),
        )?;

        let pref_image_length = self.config.image_length;
        let at_overlap = self.config.alongtrack_overlap;
        let min_image_length = self.config.min_image_length;
        let max_image_length = self.config.max_image_length.max(pref_image_length);

        let min_tile_angle = min_image_length / R_EARTH;
        let max_tile_angle = max_image_length / R_EARTH;
        let pref_tile_angle = pref_image_length / R_EARTH;
        let at_overlap_angle = at_overlap / R_EARTH;

        let mut tiles = Vec::new();
        let mut seg_i = 0;

        while seg_i + 1 < tile_bounds.len() {
            let mut angle_offset = 0.0;

            // Rotate to current polygon segment
            let l_bk_local =
                rodrigues_rotation(&l_bk, &n_vec, tile_bounds[seg_i] - at_overlap_angle);
            let r_bk_local =
                rodrigues_rotation(&r_bk, &n_vec, tile_bounds[seg_i] - at_overlap_angle);

            let mut at_tiling_angle =
                tile_bounds[seg_i + 1] - tile_bounds[seg_i] + 2.0 * at_overlap_angle;

            if at_tiling_angle < min_tile_angle {
                // Recenter for a min length tile
                angle_offset +=
                    (tile_bounds[seg_i + 1] - tile_bounds[seg_i]) / 2.0 - min_tile_angle / 2.0;
                at_tiling_angle = min_tile_angle;
            }

            // Number of tiles with overlap
            let n_tiles_f =
                (at_tiling_angle - at_overlap_angle) / (pref_tile_angle - at_overlap_angle);
            let n_full_tiles = n_tiles_f as usize;

            let partial_tile_angle = (n_tiles_f - n_full_tiles as f64) * pref_tile_angle;
            let mut tile_angles: Vec<f64> = vec![pref_tile_angle; n_full_tiles];

            // Handle partial tile at end
            if partial_tile_angle < min_tile_angle && n_full_tiles > 0 {
                let new_neighbor_angle =
                    tile_angles[n_full_tiles - 1] - (min_tile_angle - partial_tile_angle);

                if new_neighbor_angle < min_tile_angle {
                    let extd_neighbor_angle =
                        tile_angles[n_full_tiles - 1] + partial_tile_angle - at_overlap_angle;

                    if extd_neighbor_angle > max_tile_angle {
                        tile_angles.push(min_tile_angle);
                        angle_offset -=
                            (min_tile_angle - partial_tile_angle - at_overlap_angle) / 2.0;
                    } else {
                        *tile_angles.last_mut().unwrap() = extd_neighbor_angle;
                    }
                } else {
                    *tile_angles.last_mut().unwrap() -= min_tile_angle - partial_tile_angle;
                    tile_angles.push(min_tile_angle);
                }
            } else if partial_tile_angle < min_tile_angle && n_full_tiles == 0 {
                tile_angles.push(min_tile_angle);
            } else {
                tile_angles.push(partial_tile_angle);
            }

            // Create tiles within this segment
            for angle in &tile_angles {
                let point4 = rodrigues_rotation(&r_bk_local, &n_vec, angle_offset);
                let point3 = rodrigues_rotation(&l_bk_local, &n_vec, angle_offset);

                angle_offset += angle - at_overlap_angle;

                let point2 = rodrigues_rotation(&point3, &n_vec, *angle);
                let point1 = rodrigues_rotation(&point4, &n_vec, *angle);

                let tile = create_tile_from_sphere_points(
                    &[point1, point2, point3, point4],
                    alt,
                    direction,
                    strip_width,
                    angle * R_EARTH,
                    &self.spacecraft_id,
                    Some(tile_group_id),
                )?;

                tiles.push(tile);
            }

            seg_i += 2;
        }

        Ok(tiles)
    }

    /// Find along-track boundary angles for a strip, handling concave polygon regions.
    fn find_alongtrack_bounds(
        &self,
        polygon: &PolygonLocation,
        _direction: &Vector3<f64>,
        _strip_angle: f64,
        seg_l: (&Vector3<f64>, &Vector3<f64>),
        seg_r: (&Vector3<f64>, &Vector3<f64>),
    ) -> Result<Vec<f64>, BraheError> {
        let polygon_ecef_verts = polygon_vertices_to_unit_sphere(polygon)?;
        let center = polygon.center_ecef().normalize();

        // Find intersections of left/right strip edges with polygon
        let pts_l =
            great_circle_arc_polygon_intersections(seg_l.0, seg_l.1, &polygon_ecef_verts, &center);
        let pts_r =
            great_circle_arc_polygon_intersections(seg_r.0, seg_r.1, &polygon_ecef_verts, &center);

        // Compute angles from strip start to each intersection
        let seg_l_start = seg_l.0.normalize();
        let seg_r_start = seg_r.0.normalize();

        let mut angles_l: Vec<f64> = pts_l
            .iter()
            .map(|p| seg_l_start.dot(&p.normalize()).clamp(-1.0, 1.0).acos())
            .collect();
        let mut angles_r: Vec<f64> = pts_r
            .iter()
            .map(|p| seg_r_start.dot(&p.normalize()).clamp(-1.0, 1.0).acos())
            .collect();

        angles_l.sort_by(|a, b| a.partial_cmp(b).unwrap());
        angles_r.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // No intersections: polygon fits entirely inside strip
        if angles_l.is_empty() && angles_r.is_empty() {
            let full_angle = seg_l_start
                .dot(&seg_l.1.normalize())
                .clamp(-1.0, 1.0)
                .acos();
            return Ok(vec![0.0, full_angle]);
        }

        // Merge left/right edge intervals using union (OR) logic.
        // Each edge's intersections come in enter/exit pairs defining intervals
        // where that edge is inside the polygon. A tile should exist wherever
        // EITHER edge is inside the polygon (not just where both are).
        //
        // Special case: when one edge has zero intersections (entirely outside
        // the polygon), gaps in the other edge's intervals are caused by polygon
        // concavities that only cut the edge — the strip interior still covers
        // polygon area. In this case, collapse to a single continuous interval.
        let mut intervals: Vec<(f64, f64)> = Vec::new();

        if angles_l.is_empty() && !angles_r.is_empty() {
            // Left edge entirely outside: use full range of right edge
            intervals.push((*angles_r.first().unwrap(), *angles_r.last().unwrap()));
        } else if angles_r.is_empty() && !angles_l.is_empty() {
            // Right edge entirely outside: use full range of left edge
            intervals.push((*angles_l.first().unwrap(), *angles_l.last().unwrap()));
        } else {
            // Both edges intersect the polygon: union their intervals
            for chunk in angles_l.chunks(2) {
                if chunk.len() == 2 {
                    intervals.push((chunk[0], chunk[1]));
                }
            }
            for chunk in angles_r.chunks(2) {
                if chunk.len() == 2 {
                    intervals.push((chunk[0], chunk[1]));
                }
            }
        }

        // Sort by start angle and merge overlapping intervals
        intervals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut merged: Vec<(f64, f64)> = Vec::new();
        for interval in &intervals {
            if let Some(last) = merged.last_mut() {
                if interval.0 <= last.1 {
                    last.1 = last.1.max(interval.1);
                } else {
                    merged.push(*interval);
                }
            } else {
                merged.push(*interval);
            }
        }

        // Flatten to boundary angle pairs
        let mut tile_boundary_angles: Vec<f64> = Vec::new();
        for (start, end) in &merged {
            tile_boundary_angles.push(*start);
            tile_boundary_angles.push(*end);
        }

        // Handle polygon corners poking between strip edges
        let edge_plane_r = seg_r.0.cross(&seg_r.1.normalize());
        let edge_plane_r_norm = if edge_plane_r.norm() > 1e-15 {
            edge_plane_r.normalize()
        } else {
            edge_plane_r
        };
        let swath_angle = seg_l_start.dot(&seg_r_start).clamp(-1.0, 1.0).acos();

        let mut pts_between_angles = Vec::new();
        for i in 0..polygon.vertices().len().saturating_sub(1) {
            let vert_geod = polygon.vertices()[i];
            let vert_ecef = position_geodetic_to_ecef(vert_geod, AngleFormat::Degrees)
                .map_err(BraheError::Error)?
                .normalize();

            let between = vert_ecef.dot(&edge_plane_r_norm);
            let between_angle = PI / 2.0 - between.clamp(-1.0, 1.0).acos();

            if between_angle > 0.0 && between_angle < swath_angle {
                let proj = (vert_ecef - edge_plane_r_norm * between).normalize();
                let angle = seg_r_start.dot(&proj).clamp(-1.0, 1.0).acos();
                pts_between_angles.push(angle);
            }
        }

        // Extend tile boundaries to cover protruding vertices
        for &angle in &pts_between_angles {
            if tile_boundary_angles.is_empty() {
                continue;
            }
            let tile_idx = follows_index(angle, &tile_boundary_angles);

            if tile_idx < 0 {
                tile_boundary_angles[0] = angle;
            } else if !(tile_idx as usize).is_multiple_of(2) {
                let ti = tile_idx as usize;
                if ti == tile_boundary_angles.len() - 1 {
                    *tile_boundary_angles.last_mut().unwrap() = angle;
                } else if angle - tile_boundary_angles[ti] < tile_boundary_angles[ti + 1] - angle {
                    tile_boundary_angles[ti] = angle;
                } else {
                    tile_boundary_angles[ti + 1] = angle;
                }
            }
        }

        // Filter out short gaps
        let min_angle = self.config.min_image_length / R_EARTH;
        let mut reduced = Vec::new();
        if !tile_boundary_angles.is_empty() {
            reduced.push(tile_boundary_angles[0]);
            let mut i = 1;
            while i + 1 < tile_boundary_angles.len() {
                let seg_out = tile_boundary_angles[i + 1] - tile_boundary_angles[i];
                if seg_out < min_angle {
                    i += 2;
                } else {
                    reduced.push(tile_boundary_angles[i]);
                    reduced.push(tile_boundary_angles[i + 1]);
                    i += 2;
                }
            }
            if !tile_boundary_angles.is_empty() {
                reduced.push(*tile_boundary_angles.last().unwrap());
            }
        }

        if reduced.is_empty() {
            let full_angle = seg_l_start
                .dot(&seg_l.1.normalize())
                .clamp(-1.0, 1.0)
                .acos();
            reduced = vec![0.0, full_angle];
        }

        Ok(reduced)
    }

    /// Force single-tile polygons to use the requested tile length.
    fn force_requested_tile_len(
        &self,
        tiles: Vec<PolygonLocation>,
        direction: &Vector3<f64>,
    ) -> Result<Vec<PolygonLocation>, BraheError> {
        let req_len = self.config.image_length;
        let mut out_tiles = Vec::new();

        for tile in &tiles {
            let tile_len =
                super::get_tile_property_f64(tile.properties(), "tile_length").unwrap_or(0.0);
            let tile_width = super::get_tile_property_f64(tile.properties(), "tile_width")
                .unwrap_or(self.config.image_width);

            if (tile_len - req_len).abs() < 1.0 {
                out_tiles.push(tile.clone());
                continue;
            }

            let req_halfangle = req_len / R_EARTH / 2.0;
            let swath_halfangle = tile_width / R_EARTH / 2.0;

            let center = tile.center_ecef().normalize();
            let alt = tile.center_geodetic().z;

            let n_vec = direction.cross(&center).normalize();

            let lhs = rodrigues_rotation(&center, direction, -swath_halfangle).normalize();
            let rhs = rodrigues_rotation(&center, direction, swath_halfangle).normalize();

            let lf = rodrigues_rotation(&lhs, &n_vec, -req_halfangle);
            let lb = rodrigues_rotation(&lhs, &n_vec, req_halfangle);
            let rf = rodrigues_rotation(&rhs, &n_vec, -req_halfangle);
            let rb = rodrigues_rotation(&rhs, &n_vec, req_halfangle);

            let new_tile = create_tile_from_sphere_points(
                &[lf, lb, rb, rf],
                alt,
                direction,
                tile_width,
                req_len,
                &self.spacecraft_id,
                super::get_tile_property_str(tile.properties(), "tile_group_id").as_deref(),
            )?;

            out_tiles.push(new_tile);
        }

        Ok(out_tiles)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert a unit sphere point to geodetic coordinates.
fn sphere_point_to_geodetic(point: &Vector3<f64>, alt: f64) -> Vector3<f64> {
    let ecef = point.normalize() * R_EARTH;
    let mut geo = position_ecef_to_geodetic(ecef, AngleFormat::Degrees);
    geo.z = alt;
    geo
}

/// Convert polygon vertices from geodetic [lon,lat,alt] (degrees) to unit sphere ECEF.
fn polygon_vertices_to_unit_sphere(
    polygon: &PolygonLocation,
) -> Result<Vec<Vector3<f64>>, BraheError> {
    let mut ecef_verts = Vec::new();
    for v in polygon.vertices() {
        let ecef =
            position_geodetic_to_ecef(*v, AngleFormat::Degrees).map_err(BraheError::Error)?;
        ecef_verts.push(ecef.normalize());
    }
    Ok(ecef_verts)
}

/// Create a PolygonLocation tile from unit sphere corner points with metadata.
fn create_tile_from_sphere_points(
    points: &[Vector3<f64>],
    alt: f64,
    direction: &Vector3<f64>,
    tile_width: f64,
    tile_length: f64,
    spacecraft_id: &Option<String>,
    tile_group_id: Option<&str>,
) -> Result<PolygonLocation, BraheError> {
    assert!(points.len() >= 4, "Need at least 4 corner points");

    let vertices: Vec<Vector3<f64>> = points
        .iter()
        .map(|p| sphere_point_to_geodetic(p, alt))
        .collect();

    let mut tile = PolygonLocation::new(vertices)?;

    // Add metadata properties
    tile = tile.add_property(
        "tile_direction",
        json!([direction.x, direction.y, direction.z]),
    );
    tile = tile.add_property("tile_width", json!(tile_width));
    tile = tile.add_property("tile_length", json!(tile_length));
    tile = tile.add_property("tile_area", json!(tile_width * tile_length));

    if let Some(group_id) = tile_group_id {
        tile = tile.add_property("tile_group_id", json!(group_id));
    } else {
        tile = tile.add_property("tile_group_id", json!(Uuid::new_v4().to_string()));
    }

    let sc_ids = if let Some(id) = spacecraft_id {
        json!([id])
    } else {
        json!([])
    };
    tile = tile.add_property("spacecraft_ids", sc_ids);

    Ok(tile)
}

/// Compute the cross-track width of a polygon in a given direction.
///
/// Returns (total_width, min_distance, max_distance) in meters.
fn compute_crosstrack_width(
    polygon: &PolygonLocation,
    direction: &Vector3<f64>,
) -> Result<(f64, f64, f64), BraheError> {
    let center = polygon.center_ecef().normalize();

    // Great circle normal perpendicular to direction
    let cross = center.cross(direction);
    if cross.norm() < 1e-15 {
        return Err(BraheError::Error(
            "Direction is parallel to center position".to_string(),
        ));
    }
    let n_vec = cross.normalize();

    // Create along-track arc endpoints
    let polygon_ecef = polygon_vertices_to_unit_sphere(polygon)?;
    let max_angle = polygon_circumscription_angle(&polygon_ecef);

    let a = rodrigues_rotation(&center, &n_vec, max_angle).normalize();
    let b = rodrigues_rotation(&center, &n_vec, -max_angle).normalize();

    let mut min_dist = f64::MAX;
    let mut max_dist = f64::MIN;

    for vert in polygon
        .vertices()
        .iter()
        .take(polygon.vertices().len().saturating_sub(1))
    {
        let c = position_geodetic_to_ecef(*vert, AngleFormat::Degrees)
            .map_err(BraheError::Error)?
            .normalize();

        // Project onto the along-track great circle
        let (proj, ang) = cross_track_projection(&c, &a, &b);
        let ct_dist = ang * R_EARTH;

        // Signed distance
        let diff = proj - c;
        let dir_dist = n_vec.dot(&diff).signum() * ct_dist;

        min_dist = min_dist.min(dir_dist);
        max_dist = max_dist.max(dir_dist);
    }

    Ok((max_dist - min_dist, min_dist, max_dist))
}

/// Find the index of the array element occurring just before the given value.
fn follows_index(value: f64, array: &[f64]) -> i64 {
    if array.is_empty() {
        return -1;
    }
    if value > *array.last().unwrap() {
        return array.len() as i64 - 1;
    }
    for (i, &arr_val) in array.iter().enumerate() {
        if arr_val > value {
            return i as i64 - 1;
        }
    }
    array.len() as i64 - 1
}

/// Check if a tile can be rotated to a different direction while maintaining coverage.
pub(crate) fn skew_mergable_check(
    tile: &PolygonLocation,
    skew_direction: &Vector3<f64>,
    at_overlap: f64,
    ct_overlap: f64,
) -> bool {
    let base_dir = match get_tile_direction(tile) {
        Some(d) => d.normalize(),
        None => return false,
    };
    let skew_dir = skew_direction.normalize();

    let dot = base_dir.dot(&skew_dir).clamp(-1.0, 1.0);
    let rotation = dot.acos();

    let tile_w = super::get_tile_property_f64(tile.properties(), "tile_width").unwrap_or(0.0);
    let tile_l = super::get_tile_property_f64(tile.properties(), "tile_length").unwrap_or(0.0);

    let cos_r = rotation.cos();
    let sin_r = rotation.sin();

    // Upper-left corner
    let pt_ul = [-tile_w / 2.0, tile_l / 2.0];
    let pt_ul_r = [
        cos_r * pt_ul[0] - sin_r * pt_ul[1],
        sin_r * pt_ul[0] + cos_r * pt_ul[1],
    ];

    // Upper-right corner
    let pt_ur = [tile_w / 2.0, tile_l / 2.0];
    let pt_ur_r = [
        cos_r * pt_ur[0] - sin_r * pt_ur[1],
        sin_r * pt_ur[0] + cos_r * pt_ur[1],
    ];

    let overlaps = [ct_overlap, at_overlap];

    (pt_ul_r[0] - pt_ul[0]).abs() < overlaps[0]
        && (pt_ul_r[1] - pt_ul[1]).abs() < overlaps[1]
        && (pt_ur_r[0] - pt_ur[0]).abs() < overlaps[0]
        && (pt_ur_r[1] - pt_ur[1]).abs() < overlaps[1]
}

/// Extract tile direction from properties (module-level for use by orbit_geometry tests).
fn get_tile_direction(tile: &PolygonLocation) -> Option<Vector3<f64>> {
    super::get_tile_property_vec3(tile.properties(), "tile_direction")
        .map(|arr| Vector3::new(arr[0], arr[1], arr[2]))
}

/// Extract a string property from tile properties.
pub(crate) fn get_tile_property_str(
    props: &HashMap<String, JsonValue>,
    key: &str,
) -> Option<String> {
    props
        .get(key)
        .and_then(|v| v.as_str().map(|s| s.to_string()))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::propagators::SGPPropagator;
    use crate::utils::testing::setup_global_test_eop;

    // ISS TLE for testing (from sgp_propagator.rs tests)
    const ISS_LINE1: &str = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    const ISS_LINE2: &str = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

    fn make_tessellator(asc_dsc: AscDsc) -> OrbitGeometryTessellator {
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.epoch;
        let config = OrbitGeometryTessellatorConfig::default().with_asc_dsc(asc_dsc);
        OrbitGeometryTessellator::new(Box::new(prop), epoch, config, Some("ISS".to_string()))
    }

    #[test]
    fn test_config_builder() {
        let config = OrbitGeometryTessellatorConfig::new(10000.0, 15000.0)
            .with_crosstrack_overlap(300.0)
            .with_alongtrack_overlap(300.0)
            .with_asc_dsc(AscDsc::Ascending)
            .with_min_image_length(5000.0)
            .with_max_image_length(25000.0);

        assert_eq!(config.image_width, 10000.0);
        assert_eq!(config.image_length, 15000.0);
        assert_eq!(config.crosstrack_overlap, 300.0);
        assert_eq!(config.alongtrack_overlap, 300.0);
        assert_eq!(config.asc_dsc, AscDsc::Ascending);
        assert_eq!(config.min_image_length, 5000.0);
        assert_eq!(config.max_image_length, 25000.0);
    }

    #[test]
    fn test_point_tessellation_ascending() {
        setup_global_test_eop();
        let tess = make_tessellator(AscDsc::Ascending);
        let point = PointLocation::new(0.0, 30.0, 0.0);
        let tiles = tess.tessellate(&point).unwrap();

        // Should get 1 tile for ascending only
        assert_eq!(tiles.len(), 1);

        // Verify tile has expected properties
        let props = tiles[0].properties();
        assert!(props.contains_key("tile_direction"));
        assert!(props.contains_key("tile_width"));
        assert!(props.contains_key("tile_length"));
        assert!(props.contains_key("tile_area"));
        assert!(props.contains_key("tile_group_id"));
        assert!(props.contains_key("spacecraft_ids"));

        let width = props["tile_width"].as_f64().unwrap();
        let length = props["tile_length"].as_f64().unwrap();
        assert!((width - 5000.0).abs() < 1.0);
        assert!((length - 5000.0).abs() < 1.0);
    }

    #[test]
    fn test_point_tessellation_either() {
        setup_global_test_eop();
        let tess = make_tessellator(AscDsc::Either);
        let point = PointLocation::new(10.0, 30.0, 0.0);
        let tiles = tess.tessellate(&point).unwrap();

        // Should get 1-2 tiles for ascending and descending
        assert!(!tiles.is_empty() && tiles.len() <= 2);

        // Each tile should have spacecraft_ids containing "ISS"
        for tile in &tiles {
            let sc_ids = tile.properties()["spacecraft_ids"].as_array().unwrap();
            assert!(sc_ids.iter().any(|v| v.as_str() == Some("ISS")));
        }
    }

    #[test]
    fn test_polygon_tessellation_small_rect() {
        setup_global_test_eop();
        let tess = make_tessellator(AscDsc::Ascending);

        // Small ~0.05° rectangle (about 5km x 5km at equator)
        let vertices = vec![
            Vector3::new(10.0, 30.0, 0.0),
            Vector3::new(10.05, 30.0, 0.0),
            Vector3::new(10.05, 30.05, 0.0),
            Vector3::new(10.0, 30.05, 0.0),
        ];
        let polygon = PolygonLocation::new(vertices).unwrap();
        let tiles = tess
            .tessellate(&polygon as &dyn AccessibleLocation)
            .unwrap();

        // Small polygon → should produce a handful of tiles
        assert!(!tiles.is_empty());

        // All tiles should have the required properties
        for tile in &tiles {
            let props = tile.properties();
            assert!(props.contains_key("tile_direction"));
            assert!(props.contains_key("tile_width"));
            assert!(props.contains_key("tile_length"));
            assert!(props.contains_key("tile_group_id"));
        }
    }

    #[test]
    fn test_polygon_tessellation_larger() {
        setup_global_test_eop();
        let tess = make_tessellator(AscDsc::Ascending);

        // Larger polygon (~0.2° = ~22km sides)
        let vertices = vec![
            Vector3::new(10.0, 30.0, 0.0),
            Vector3::new(10.2, 30.0, 0.0),
            Vector3::new(10.2, 30.2, 0.0),
            Vector3::new(10.0, 30.2, 0.0),
        ];
        let polygon = PolygonLocation::new(vertices).unwrap();
        let tiles = tess
            .tessellate(&polygon as &dyn AccessibleLocation)
            .unwrap();

        // Should produce multiple strips with multiple tiles
        assert!(tiles.len() > 1);
    }

    #[test]
    fn test_tessellator_name() {
        setup_global_test_eop();
        let tess = make_tessellator(AscDsc::Ascending);
        assert_eq!(tess.name(), "OrbitGeometryTessellator");
    }

    #[test]
    fn test_follows_index() {
        assert_eq!(follows_index(0.5, &[1.0, 2.0, 3.0]), -1);
        assert_eq!(follows_index(1.5, &[1.0, 2.0, 3.0]), 0);
        assert_eq!(follows_index(4.0, &[1.0, 2.0, 3.0]), 2);
    }

    #[test]
    fn test_skew_mergable_small_angle() {
        // Small angle difference → should be mergable
        let vertices = vec![
            Vector3::new(10.0, 50.0, 0.0),
            Vector3::new(10.05, 50.0, 0.0),
            Vector3::new(10.05, 50.05, 0.0),
            Vector3::new(10.0, 50.05, 0.0),
        ];
        let mut tile = PolygonLocation::new(vertices).unwrap();
        let dir = Vector3::new(0.0, 1.0, 0.0).normalize();
        tile = tile.add_property("tile_direction", json!([dir.x, dir.y, dir.z]));
        tile = tile.add_property("tile_width", json!(5000.0));
        tile = tile.add_property("tile_length", json!(5000.0));

        // Very small rotation
        let skew = Vector3::new(0.01, 1.0, 0.0).normalize();
        assert!(skew_mergable_check(&tile, &skew, 200.0, 200.0));
    }

    #[test]
    fn test_skew_mergable_large_angle() {
        let vertices = vec![
            Vector3::new(10.0, 50.0, 0.0),
            Vector3::new(10.05, 50.0, 0.0),
            Vector3::new(10.05, 50.05, 0.0),
            Vector3::new(10.0, 50.05, 0.0),
        ];
        let mut tile = PolygonLocation::new(vertices).unwrap();
        let dir = Vector3::new(0.0, 1.0, 0.0).normalize();
        tile = tile.add_property("tile_direction", json!([dir.x, dir.y, dir.z]));
        tile = tile.add_property("tile_width", json!(5000.0));
        tile = tile.add_property("tile_length", json!(5000.0));

        // Large rotation
        let skew = Vector3::new(1.0, 1.0, 0.0).normalize();
        assert!(!skew_mergable_check(&tile, &skew, 200.0, 200.0));
    }
}
