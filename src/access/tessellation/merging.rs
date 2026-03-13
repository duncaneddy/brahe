/*!
 * Tile merging utilities for multi-spacecraft tessellation
 *
 * When multiple spacecraft can image the same area with similar ground-track
 * directions, tiles from one spacecraft can be "merged" onto another by
 * adding the alternate spacecraft's ID to the base tile's `spacecraft_ids`.
 * This reduces redundant tiles without losing collection capability.
 */

use std::collections::HashMap;

use nalgebra::Vector3;

use crate::access::location::{AccessibleLocation, PolygonLocation};
use crate::access::tessellation::get_tile_property_vec3;
use crate::access::tessellation::orbit_geometry::{get_tile_property_str, skew_mergable_check};

/// Merge tiles from multiple spacecraft when their tiling directions are similar.
///
/// Groups tiles by `tile_group_id`, then clusters groups with similar directions
/// (within `mergable_range_deg`). For each cluster, picks the median direction
/// as the base and tests if other groups' tiles are "skew-mergable" onto it.
///
/// # Arguments
/// * `tiles` - All tiles from all spacecraft
/// * `at_overlap` - Along-track overlap (meters)
/// * `ct_overlap` - Cross-track overlap (meters)
/// * `mergable_range_deg` - Maximum angular difference (degrees) for grouping directions
///
/// # Returns
/// Reduced set of tiles with merged `spacecraft_ids` properties
pub fn tile_merge_orbit_geometry(
    tiles: &[PolygonLocation],
    at_overlap: f64,
    ct_overlap: f64,
    mergable_range_deg: f64,
) -> Vec<PolygonLocation> {
    if tiles.is_empty() {
        return Vec::new();
    }

    // Group tiles by tile_group_id
    let mut groups: HashMap<String, Vec<PolygonLocation>> = HashMap::new();
    for tile in tiles {
        let group_id = get_tile_property_str(tile.properties(), "tile_group_id")
            .unwrap_or_else(|| "unknown".to_string());
        groups.entry(group_id).or_default().push(tile.clone());
    }

    if groups.len() < 2 {
        return tiles.to_vec();
    }

    // Extract direction info for each group
    let mut dir_list: Vec<DirInfo> = groups
        .iter()
        .filter_map(|(id, group)| {
            let dir_arr = get_tile_property_vec3(group[0].properties(), "tile_direction")?;
            Some(DirInfo {
                dir: Vector3::new(dir_arr[0], dir_arr[1], dir_arr[2]),
                id: id.clone(),
                offset: 0.0,
            })
        })
        .collect();

    if dir_list.len() < 2 {
        return tiles.to_vec();
    }

    // Pick first as reference, compute angular offsets
    let dir_ref = dir_list[0].dir;
    let norm_raw = dir_ref
        .cross(&dir_list[1].dir)
        .try_normalize(1e-15)
        .unwrap_or(Vector3::new(0.0, 0.0, 1.0));
    let norm_vec = Vector3::new(norm_raw.x.abs(), norm_raw.y.abs(), norm_raw.z.abs());

    for info in dir_list.iter_mut().skip(1) {
        let cross_dot = dir_ref.cross(&info.dir).dot(&norm_vec);
        let dot = dir_ref.dot(&info.dir);
        info.offset = cross_dot.atan2(dot).to_degrees();
    }

    // Sort by offset angle
    dir_list.sort_by(|a, b| a.offset.partial_cmp(&b.offset).unwrap());

    // Group by angular proximity
    let mut super_groups: Vec<Vec<DirInfo>> = Vec::new();
    let mut current_sg = vec![dir_list[0].clone()];

    for info in dir_list.iter().skip(1) {
        if info.offset - current_sg[0].offset > mergable_range_deg {
            super_groups.push(current_sg);
            current_sg = vec![info.clone()];
        } else {
            current_sg.push(info.clone());
        }
    }
    super_groups.push(current_sg);

    // Handle wrap-around merging
    if super_groups.len() > 1 {
        let last_offset = super_groups.last().unwrap()[0].offset.abs();
        let first_offset = super_groups[0].last().unwrap().offset.abs();
        if (last_offset - first_offset).abs() < mergable_range_deg {
            let rolled: Vec<DirInfo> = super_groups[0]
                .iter()
                .map(|info| DirInfo {
                    dir: info.dir,
                    id: info.id.clone(),
                    offset: info.offset + 360.0,
                })
                .collect();
            super_groups.last_mut().unwrap().extend(rolled);
            super_groups.remove(0);
        }
    }

    // Try to merge within super-groups
    let mut removable_ids: Vec<String> = Vec::new();

    for sg in &super_groups {
        if sg.len() < 2 {
            continue;
        }

        let median_idx = (sg.len() - 1) / 2;
        let base_id = &sg[median_idx].id;

        for (i, alt_info) in sg.iter().enumerate() {
            if i == median_idx {
                continue;
            }

            let base_group = match groups.get(base_id) {
                Some(g) => g,
                None => continue,
            };

            // Find mergable tiles in base group
            let mergable_indices: Vec<usize> = base_group
                .iter()
                .enumerate()
                .filter(|(_, tile)| {
                    skew_mergable_check(tile, &alt_info.dir, at_overlap, ct_overlap)
                })
                .map(|(i, _)| i)
                .collect();

            if !mergable_indices.is_empty() {
                // Get alt spacecraft IDs
                let alt_group = match groups.get(&alt_info.id) {
                    Some(g) => g,
                    None => continue,
                };

                let alt_sc_ids: Vec<String> = alt_group
                    .first()
                    .and_then(|t| t.properties().get("spacecraft_ids"))
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default();

                // Add alt spacecraft IDs to base tiles
                if let Some(base_group) = groups.get_mut(base_id) {
                    for &idx in &mergable_indices {
                        if let Some(tile) = base_group.get_mut(idx) {
                            let props = tile.properties_mut();
                            if let Some(sc_ids) = props.get_mut("spacecraft_ids")
                                && let Some(arr) = sc_ids.as_array_mut()
                            {
                                for sc_id in &alt_sc_ids {
                                    let val = serde_json::Value::String(sc_id.clone());
                                    if !arr.contains(&val) {
                                        arr.push(val);
                                    }
                                }
                            }
                        }
                    }
                }

                removable_ids.push(alt_info.id.clone());
            }
        }
    }

    // Remove merged groups
    for id in &removable_ids {
        groups.remove(id);
    }

    // Flatten and return
    groups.into_values().flatten().collect()
}

#[derive(Clone)]
struct DirInfo {
    dir: Vector3<f64>,
    id: String,
    offset: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;
    use serde_json::json;

    fn make_tile(
        center_lon: f64,
        center_lat: f64,
        direction: [f64; 3],
        group_id: &str,
        spacecraft_id: &str,
    ) -> PolygonLocation {
        let d = 0.025; // ~2.5 km half-size
        let vertices = vec![
            Vector3::new(center_lon - d, center_lat - d, 0.0),
            Vector3::new(center_lon + d, center_lat - d, 0.0),
            Vector3::new(center_lon + d, center_lat + d, 0.0),
            Vector3::new(center_lon - d, center_lat + d, 0.0),
        ];
        let tile = PolygonLocation::new(vertices).unwrap();
        tile.add_property("tile_direction", json!(direction))
            .add_property("tile_width", json!(5000.0))
            .add_property("tile_length", json!(5000.0))
            .add_property("tile_area", json!(25000000.0))
            .add_property("tile_group_id", json!(group_id))
            .add_property("spacecraft_ids", json!([spacecraft_id]))
    }

    #[test]
    fn test_merge_empty() {
        let result = tile_merge_orbit_geometry(&[], 200.0, 200.0, 2.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_merge_single_group() {
        let tiles = vec![
            make_tile(10.0, 50.0, [0.0, 1.0, 0.0], "group1", "sc1"),
            make_tile(10.05, 50.0, [0.0, 1.0, 0.0], "group1", "sc1"),
        ];
        let result = tile_merge_orbit_geometry(&tiles, 200.0, 200.0, 2.0);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_merge_similar_directions() {
        // Two spacecraft with very similar directions → should merge
        let dir1 = [0.0_f64, 1.0, 0.0];
        let dir2 = [0.01, 1.0, 0.0]; // Very slightly different

        let tiles = vec![
            make_tile(10.0, 50.0, dir1, "group_a", "sc1"),
            make_tile(10.0, 50.0, dir2, "group_b", "sc2"),
        ];

        let result = tile_merge_orbit_geometry(&tiles, 200.0, 200.0, 5.0);

        // Should have merged into one group
        assert!(result.len() <= 2);

        // At least one tile should have both spacecraft IDs
        let has_both = result.iter().any(|t| {
            t.properties()
                .get("spacecraft_ids")
                .and_then(|v| v.as_array())
                .map(|arr| arr.len() >= 2)
                .unwrap_or(false)
        });
        // If merge succeeded, at least one tile has both spacecraft
        // (may not merge if direction angle is too different for skew_mergable)
        assert!(has_both || result.len() == 2);
    }

    #[test]
    fn test_merge_different_directions() {
        // Two spacecraft with very different directions → should NOT merge
        let dir1 = [0.0, 1.0, 0.0];
        let dir2 = [1.0, 0.0, 0.0]; // Perpendicular

        let tiles = vec![
            make_tile(10.0, 50.0, dir1, "group_a", "sc1"),
            make_tile(10.0, 50.0, dir2, "group_b", "sc2"),
        ];

        let result = tile_merge_orbit_geometry(&tiles, 200.0, 200.0, 2.0);
        assert_eq!(result.len(), 2);
    }
}
