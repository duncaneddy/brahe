/*!
 * Core trait and utilities for tessellation algorithms
 *
 * Defines the [`Tessellator`] trait that all tessellation implementations must
 * satisfy, plus helper functions for reading tile metadata properties.
 */

use std::collections::HashMap;

use serde_json::Value as JsonValue;

use crate::access::location::{AccessibleLocation, PolygonLocation};
use crate::utils::errors::BraheError;

/// Core trait for tessellation algorithms.
///
/// Implementations divide geographic locations into rectangular tiles
/// suitable for satellite imaging collection. Each tile is returned as a
/// [`PolygonLocation`] with metadata properties describing the tile geometry.
pub trait Tessellator: Send + Sync {
    /// Tessellate a location into rectangular tiles.
    ///
    /// # Arguments
    /// * `location` - The target location to tessellate
    ///
    /// # Returns
    /// Vector of [`PolygonLocation`] tiles with metadata properties
    fn tessellate(
        &self,
        location: &dyn AccessibleLocation,
    ) -> Result<Vec<PolygonLocation>, BraheError>;

    /// Tessellate multiple locations.
    ///
    /// Default implementation iterates and tessellates each location individually.
    ///
    /// # Arguments
    /// * `locations` - Slice of locations to tessellate
    ///
    /// # Returns
    /// Vector of tile vectors, one per input location
    fn tessellate_batch(
        &self,
        locations: &[&dyn AccessibleLocation],
    ) -> Result<Vec<Vec<PolygonLocation>>, BraheError> {
        locations.iter().map(|l| self.tessellate(*l)).collect()
    }

    /// Human-readable name of the tessellation algorithm.
    fn name(&self) -> &str;
}

/// Extract a tile property as f64 from a PolygonLocation's properties.
pub(crate) fn get_tile_property_f64(props: &HashMap<String, JsonValue>, key: &str) -> Option<f64> {
    props.get(key).and_then(|v| v.as_f64())
}

/// Extract a tile property as a 3-element array from a PolygonLocation's properties.
pub(crate) fn get_tile_property_vec3(
    props: &HashMap<String, JsonValue>,
    key: &str,
) -> Option<[f64; 3]> {
    props.get(key).and_then(|v| {
        v.as_array().and_then(|arr| {
            if arr.len() == 3 {
                Some([arr[0].as_f64()?, arr[1].as_f64()?, arr[2].as_f64()?])
            } else {
                None
            }
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    use crate::access::location::PointLocation;

    /// Mock tessellator for testing the trait interface
    struct MockTessellator;

    impl Tessellator for MockTessellator {
        fn tessellate(
            &self,
            _location: &dyn AccessibleLocation,
        ) -> Result<Vec<PolygonLocation>, BraheError> {
            let vertices = vec![
                Vector3::new(10.0, 50.0, 0.0),
                Vector3::new(11.0, 50.0, 0.0),
                Vector3::new(11.0, 51.0, 0.0),
                Vector3::new(10.0, 51.0, 0.0),
            ];
            let tile = PolygonLocation::new(vertices)?;
            Ok(vec![tile])
        }

        fn name(&self) -> &str {
            "MockTessellator"
        }
    }

    #[test]
    fn test_mock_tessellator() {
        let tess = MockTessellator;
        let point = PointLocation::new(10.5, 50.5, 0.0);
        let tiles = tess.tessellate(&point).unwrap();
        assert_eq!(tiles.len(), 1);
        assert_eq!(tess.name(), "MockTessellator");
    }

    #[test]
    fn test_tessellate_batch_default() {
        let tess = MockTessellator;
        let p1 = PointLocation::new(10.5, 50.5, 0.0);
        let p2 = PointLocation::new(20.0, 40.0, 0.0);
        let locations: Vec<&dyn AccessibleLocation> = vec![&p1, &p2];
        let results = tess.tessellate_batch(&locations).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 1);
        assert_eq!(results[1].len(), 1);
    }

    #[test]
    fn test_tessellate_with_point_and_polygon() {
        let tess = MockTessellator;

        let point = PointLocation::new(10.0, 50.0, 0.0);
        let tiles = tess.tessellate(&point).unwrap();
        assert_eq!(tiles.len(), 1);

        let vertices = vec![
            Vector3::new(10.0, 50.0, 0.0),
            Vector3::new(11.0, 50.0, 0.0),
            Vector3::new(11.0, 51.0, 0.0),
            Vector3::new(10.0, 51.0, 0.0),
        ];
        let polygon = PolygonLocation::new(vertices).unwrap();
        let tiles = tess.tessellate(&polygon).unwrap();
        assert_eq!(tiles.len(), 1);
    }
}
