/*!
 * Location types and AccessibleLocation trait for access computation
 *
 * This module provides point and polygon location types with GeoJSON interoperability.
 * All locations implement the Identifiable trait for traceability.
 */

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use serde_json::{Value as JsonValue, json};
use std::collections::HashMap;
use uuid::Uuid;

use crate::constants::AngleFormat;
use crate::coordinates::position_geodetic_to_ecef;
use crate::utils::errors::BraheError;
use crate::utils::identifiable::Identifiable;

/// Core trait for any location that can be accessed by a satellite
///
/// Provides common interface for point and polygon locations, including:
/// - Geographic coordinates (geodetic and ECEF)
/// - Extensible properties via HashMap
/// - GeoJSON serialization
/// - Identifiable trait support
pub trait AccessibleLocation: Identifiable + Send + Sync {
    /// Geographic center in geodetic coordinates [lon, lat, alt]
    ///
    /// Units: [degrees, degrees, meters]
    fn center_geodetic(&self) -> Vector3<f64>;

    /// Center position in ECEF coordinates
    ///
    /// Units: meters
    fn center_ecef(&self) -> Vector3<f64>;

    /// Extensible properties dictionary
    ///
    /// Allows storing arbitrary metadata with the location
    fn properties(&self) -> &HashMap<String, JsonValue>;

    /// Mutable access to properties
    fn properties_mut(&mut self) -> &mut HashMap<String, JsonValue>;

    /// Export to GeoJSON Feature format
    ///
    /// Returns a GeoJSON Feature object with geometry and properties
    fn to_geojson(&self) -> JsonValue;
}

/// A single point location on Earth's surface
///
/// Represents a discrete point with geodetic coordinates.
/// Commonly used for ground stations, imaging targets, or tessellated polygon tiles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointLocation {
    /// Geodetic coordinates [longitude, latitude, altitude]
    /// Units: [degrees, degrees, meters]
    center: Vector3<f64>,

    /// Cached ECEF position (meters)
    #[serde(skip)]
    center_ecef: Option<Vector3<f64>>,

    /// Extensible metadata dictionary
    #[serde(default)]
    properties: HashMap<String, JsonValue>,

    /// Optional human-readable name
    name: Option<String>,

    /// Optional numeric identifier
    id: Option<u64>,

    /// Optional UUID
    uuid: Option<Uuid>,
}

impl PointLocation {
    /// Create a new point location from geodetic coordinates
    ///
    /// # Arguments
    /// - `lon`: Longitude (degrees)
    /// - `lat`: Latitude (degrees)
    /// - `alt`: Altitude above ellipsoid (meters)
    ///
    /// # Returns
    /// New PointLocation instance
    ///
    /// # Example
    /// ```
    /// use brahe::access::PointLocation;
    ///
    /// let svalbard = PointLocation::new(15.4, 78.2, 0.0);
    /// ```
    pub fn new(lon: f64, lat: f64, alt: f64) -> Self {
        let center = Vector3::new(lon, lat, alt);

        // Compute ECEF position
        let center_ecef = position_geodetic_to_ecef(center, AngleFormat::Degrees)
            .expect("Invalid geodetic coordinates");

        Self {
            center,
            center_ecef: Some(center_ecef),
            properties: HashMap::new(),
            name: None,
            id: None,
            uuid: None,
        }
    }

    /// Create from GeoJSON Point Feature
    ///
    /// # Arguments
    /// - `geojson`: GeoJSON Feature object with Point geometry
    ///
    /// # Returns
    /// New PointLocation or error if invalid
    ///
    /// # Example
    /// ```
    /// use brahe::access::PointLocation;
    /// use serde_json::json;
    ///
    /// let geojson = json!({
    ///     "type": "Feature",
    ///     "geometry": {
    ///         "type": "Point",
    ///         "coordinates": [15.4, 78.2, 0.0]
    ///     },
    ///     "properties": {
    ///         "name": "Svalbard"
    ///     }
    /// });
    ///
    /// let location = PointLocation::from_geojson(&geojson).unwrap();
    /// ```
    pub fn from_geojson(geojson: &JsonValue) -> Result<Self, BraheError> {
        // Validate it's a Feature
        if geojson.get("type").and_then(|t| t.as_str()) != Some("Feature") {
            return Err(BraheError::ParseError(
                "GeoJSON must be a Feature object".to_string(),
            ));
        }

        // Extract geometry
        let geometry = geojson.get("geometry").ok_or_else(|| {
            BraheError::ParseError("GeoJSON Feature missing geometry".to_string())
        })?;

        // Validate it's a Point
        if geometry.get("type").and_then(|t| t.as_str()) != Some("Point") {
            return Err(BraheError::ParseError(
                "GeoJSON geometry must be Point type".to_string(),
            ));
        }

        // Extract coordinates [lon, lat] or [lon, lat, alt]
        let coords = geometry
            .get("coordinates")
            .and_then(|c| c.as_array())
            .ok_or_else(|| BraheError::ParseError("Invalid Point coordinates".to_string()))?;

        if coords.len() < 2 {
            return Err(BraheError::ParseError(
                "Point must have at least [lon, lat]".to_string(),
            ));
        }

        let lon = coords[0]
            .as_f64()
            .ok_or_else(|| BraheError::ParseError("Longitude must be a number".to_string()))?;
        let lat = coords[1]
            .as_f64()
            .ok_or_else(|| BraheError::ParseError("Latitude must be a number".to_string()))?;
        let alt = coords.get(2).and_then(|a| a.as_f64()).unwrap_or(0.0);

        // Create point location
        let mut location = Self::new(lon, lat, alt);

        // Extract properties if present
        if let Some(props) = geojson.get("properties").and_then(|p| p.as_object()) {
            for (key, value) in props.iter() {
                // Special handling for identity fields
                match key.as_str() {
                    "name" => {
                        if let Some(name_str) = value.as_str() {
                            location.name = Some(name_str.to_string());
                        }
                    }
                    "id" => {
                        if let Some(id_num) = value.as_u64() {
                            location.id = Some(id_num);
                        }
                    }
                    "uuid" => {
                        if let Some(uuid_str) = value.as_str()
                            && let Ok(parsed_uuid) = Uuid::parse_str(uuid_str)
                        {
                            location.uuid = Some(parsed_uuid);
                        }
                    }
                    _ => {
                        // Store other properties
                        location.properties.insert(key.clone(), value.clone());
                    }
                }
            }
        }

        Ok(location)
    }

    /// Add a custom property (builder pattern)
    ///
    /// # Arguments
    /// - `key`: Property name
    /// - `value`: Property value (any JSON-serializable type)
    ///
    /// # Returns
    /// Self for chaining
    ///
    /// # Example
    /// ```
    /// use brahe::access::PointLocation;
    /// use serde_json::json;
    ///
    /// let location = PointLocation::new(15.4, 78.2, 0.0)
    ///     .add_property("country", json!("Norway"))
    ///     .add_property("elevation_mask_deg", json!(5.0));
    /// ```
    pub fn add_property(mut self, key: &str, value: JsonValue) -> Self {
        self.properties.insert(key.to_string(), value);
        self
    }

    /// Get longitude in degrees (quick accessor)
    pub fn lon(&self) -> f64 {
        self.center.x
    }

    /// Get latitude in degrees (quick accessor)
    pub fn lat(&self) -> f64 {
        self.center.y
    }

    /// Get altitude in meters (quick accessor)
    pub fn alt(&self) -> f64 {
        self.center.z
    }

    /// Get longitude with angle format conversion
    ///
    /// # Arguments
    /// - `angle_format`: Desired output format (Degrees or Radians)
    ///
    /// # Example
    /// ```
    /// use brahe::access::PointLocation;
    /// use brahe::constants::AngleFormat;
    ///
    /// let loc = PointLocation::new(15.4, 78.2, 0.0);
    /// let lon_deg = loc.longitude(AngleFormat::Degrees);
    /// let lon_rad = loc.longitude(AngleFormat::Radians);
    /// ```
    pub fn longitude(&self, angle_format: AngleFormat) -> f64 {
        match angle_format {
            AngleFormat::Degrees => self.center.x,
            AngleFormat::Radians => self.center.x.to_radians(),
        }
    }

    /// Get latitude with angle format conversion
    ///
    /// # Arguments
    /// - `angle_format`: Desired output format (Degrees or Radians)
    pub fn latitude(&self, angle_format: AngleFormat) -> f64 {
        match angle_format {
            AngleFormat::Degrees => self.center.y,
            AngleFormat::Radians => self.center.y.to_radians(),
        }
    }

    /// Get altitude in meters
    pub fn altitude(&self) -> f64 {
        self.center.z
    }
}

impl AccessibleLocation for PointLocation {
    fn center_geodetic(&self) -> Vector3<f64> {
        self.center
    }

    fn center_ecef(&self) -> Vector3<f64> {
        // Return cached value or recompute if needed
        self.center_ecef.unwrap_or_else(|| {
            position_geodetic_to_ecef(self.center, AngleFormat::Degrees)
                .expect("Invalid geodetic coordinates")
        })
    }

    fn properties(&self) -> &HashMap<String, JsonValue> {
        &self.properties
    }

    fn properties_mut(&mut self) -> &mut HashMap<String, JsonValue> {
        &mut self.properties
    }

    fn to_geojson(&self) -> JsonValue {
        let mut props = self.properties.clone();

        // Add identity fields to properties
        if let Some(name) = &self.name {
            props.insert("name".to_string(), json!(name));
        }
        if let Some(id) = self.id {
            props.insert("id".to_string(), json!(id));
        }
        if let Some(uuid) = self.uuid {
            props.insert("uuid".to_string(), json!(uuid.to_string()));
        }

        json!({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [self.center.x, self.center.y, self.center.z]
            },
            "properties": props
        })
    }
}

impl Identifiable for PointLocation {
    fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    fn with_uuid(mut self, uuid: Uuid) -> Self {
        self.uuid = Some(uuid);
        self
    }

    fn with_new_uuid(mut self) -> Self {
        self.uuid = Some(Uuid::new_v4());
        self
    }

    fn with_id(mut self, id: u64) -> Self {
        self.id = Some(id);
        self
    }

    fn with_identity(mut self, name: Option<&str>, uuid: Option<Uuid>, id: Option<u64>) -> Self {
        self.name = name.map(|s| s.to_string());
        self.uuid = uuid;
        self.id = id;
        self
    }

    fn set_identity(&mut self, name: Option<&str>, uuid: Option<Uuid>, id: Option<u64>) {
        self.name = name.map(|s| s.to_string());
        self.uuid = uuid;
        self.id = id;
    }

    fn set_id(&mut self, id: Option<u64>) {
        self.id = id;
    }

    fn set_name(&mut self, name: Option<&str>) {
        self.name = name.map(|s| s.to_string());
    }

    fn generate_uuid(&mut self) {
        self.uuid = Some(Uuid::new_v4());
    }

    fn get_id(&self) -> Option<u64> {
        self.id
    }

    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn get_uuid(&self) -> Option<Uuid> {
        self.uuid
    }
}

/// A polygonal area on Earth's surface
///
/// Represents a closed polygon with multiple vertices.
/// Commonly used for areas of interest, no-fly zones, or imaging footprints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolygonLocation {
    /// Computed center (centroid) in geodetic coords [lon, lat, alt]
    center: Vector3<f64>,

    /// Cached ECEF center position
    #[serde(skip)]
    center_ecef: Option<Vector3<f64>>,

    /// Polygon vertices in geodetic coords [lon, lat, alt]
    /// First and last vertex must be identical (closed polygon)
    vertices: Vec<Vector3<f64>>,

    /// Extensible metadata dictionary
    #[serde(default)]
    properties: HashMap<String, JsonValue>,

    /// Optional human-readable name
    name: Option<String>,

    /// Optional numeric identifier
    id: Option<u64>,

    /// Optional UUID
    uuid: Option<Uuid>,
}

impl PolygonLocation {
    /// Create a new polygon location from vertices
    ///
    /// Automatically closes the polygon if the first and last vertices don't match.
    /// Validates polygon constraints.
    ///
    /// # Arguments
    /// - `vertices`: List of vertices [[lon, lat, alt], ...]
    ///
    /// # Returns
    /// New PolygonLocation or error if invalid
    ///
    /// # Validation
    /// - At least 3 unique vertices (4 including closure)
    /// - First and last vertex must match (auto-closed if not)
    /// - No duplicate consecutive vertices
    ///
    /// # Example
    /// ```
    /// use brahe::access::PolygonLocation;
    /// use nalgebra::Vector3;
    ///
    /// let vertices = vec![
    ///     Vector3::new(10.0, 50.0, 0.0),
    ///     Vector3::new(11.0, 50.0, 0.0),
    ///     Vector3::new(11.0, 51.0, 0.0),
    ///     Vector3::new(10.0, 51.0, 0.0),
    ///     Vector3::new(10.0, 50.0, 0.0),  // Closed
    /// ];
    ///
    /// let polygon = PolygonLocation::new(vertices).unwrap();
    /// ```
    pub fn new(mut vertices: Vec<Vector3<f64>>) -> Result<Self, BraheError> {
        // Validate minimum vertices
        if vertices.len() < 4 {
            return Err(BraheError::ParseError(
                "Polygon must have at least 4 vertices (3 unique + closure)".to_string(),
            ));
        }

        // Check for duplicate consecutive vertices (except first==last)
        for i in 0..vertices.len() - 2 {
            if vertices[i] == vertices[i + 1] {
                return Err(BraheError::ParseError(format!(
                    "Duplicate consecutive vertices at index {}",
                    i
                )));
            }
        }

        // Auto-close polygon if needed
        let first = vertices[0];
        let last = vertices[vertices.len() - 1];
        if first != last {
            vertices.push(first);
        }

        // Compute centroid (average of unique vertices)
        let num_unique = vertices.len() - 1; // Exclude closure vertex
        let mut sum_lon = 0.0;
        let mut sum_lat = 0.0;
        let mut sum_alt = 0.0;

        for vertex in vertices.iter().take(num_unique) {
            sum_lon += vertex.x;
            sum_lat += vertex.y;
            sum_alt += vertex.z;
        }

        let center = Vector3::new(
            sum_lon / num_unique as f64,
            sum_lat / num_unique as f64,
            sum_alt / num_unique as f64,
        );

        // Compute ECEF center
        let center_ecef = position_geodetic_to_ecef(center, AngleFormat::Degrees)
            .expect("Invalid geodetic coordinates");

        Ok(Self {
            center,
            center_ecef: Some(center_ecef),
            vertices,
            properties: HashMap::new(),
            name: None,
            id: None,
            uuid: None,
        })
    }

    /// Create from GeoJSON Polygon Feature
    ///
    /// # Arguments
    /// - `geojson`: GeoJSON Feature object with Polygon geometry
    ///
    /// # Returns
    /// New PolygonLocation or error if invalid
    ///
    /// # Example
    /// ```
    /// use brahe::access::PolygonLocation;
    /// use serde_json::json;
    ///
    /// let geojson = json!({
    ///     "type": "Feature",
    ///     "geometry": {
    ///         "type": "Polygon",
    ///         "coordinates": [[
    ///             [10.0, 50.0, 0.0],
    ///             [11.0, 50.0, 0.0],
    ///             [11.0, 51.0, 0.0],
    ///             [10.0, 51.0, 0.0],
    ///             [10.0, 50.0, 0.0]
    ///         ]]
    ///     },
    ///     "properties": {
    ///         "name": "AOI-1"
    ///     }
    /// });
    ///
    /// let polygon = PolygonLocation::from_geojson(&geojson).unwrap();
    /// ```
    pub fn from_geojson(geojson: &JsonValue) -> Result<Self, BraheError> {
        // Validate it's a Feature
        if geojson.get("type").and_then(|t| t.as_str()) != Some("Feature") {
            return Err(BraheError::ParseError(
                "GeoJSON must be a Feature object".to_string(),
            ));
        }

        // Extract geometry
        let geometry = geojson.get("geometry").ok_or_else(|| {
            BraheError::ParseError("GeoJSON Feature missing geometry".to_string())
        })?;

        // Validate it's a Polygon
        if geometry.get("type").and_then(|t| t.as_str()) != Some("Polygon") {
            return Err(BraheError::ParseError(
                "GeoJSON geometry must be Polygon type".to_string(),
            ));
        }

        // Extract coordinates (outer ring only, ignore holes)
        let coords = geometry
            .get("coordinates")
            .and_then(|c| c.as_array())
            .ok_or_else(|| BraheError::ParseError("Invalid Polygon coordinates".to_string()))?;

        if coords.is_empty() {
            return Err(BraheError::ParseError(
                "Polygon must have at least one ring".to_string(),
            ));
        }

        // Get outer ring
        let outer_ring = coords[0]
            .as_array()
            .ok_or_else(|| BraheError::ParseError("Invalid outer ring".to_string()))?;

        // Parse vertices
        let mut vertices = Vec::new();
        for vertex_json in outer_ring {
            let vertex_array = vertex_json
                .as_array()
                .ok_or_else(|| BraheError::ParseError("Invalid vertex format".to_string()))?;

            if vertex_array.len() < 2 {
                return Err(BraheError::ParseError(
                    "Vertex must have [lon, lat] or [lon, lat, alt]".to_string(),
                ));
            }

            let lon = vertex_array[0]
                .as_f64()
                .ok_or_else(|| BraheError::ParseError("Longitude must be a number".to_string()))?;
            let lat = vertex_array[1]
                .as_f64()
                .ok_or_else(|| BraheError::ParseError("Latitude must be a number".to_string()))?;
            let alt = vertex_array.get(2).and_then(|a| a.as_f64()).unwrap_or(0.0);

            vertices.push(Vector3::new(lon, lat, alt));
        }

        // Create polygon
        let mut polygon = Self::new(vertices)?;

        // Extract properties if present
        if let Some(props) = geojson.get("properties").and_then(|p| p.as_object()) {
            for (key, value) in props.iter() {
                match key.as_str() {
                    "name" => {
                        if let Some(name_str) = value.as_str() {
                            polygon.name = Some(name_str.to_string());
                        }
                    }
                    "id" => {
                        if let Some(id_num) = value.as_u64() {
                            polygon.id = Some(id_num);
                        }
                    }
                    "uuid" => {
                        if let Some(uuid_str) = value.as_str()
                            && let Ok(parsed_uuid) = Uuid::parse_str(uuid_str)
                        {
                            polygon.uuid = Some(parsed_uuid);
                        }
                    }
                    _ => {
                        polygon.properties.insert(key.clone(), value.clone());
                    }
                }
            }
        }

        Ok(polygon)
    }

    /// Get polygon vertices
    ///
    /// Returns all vertices including the closure vertex (first == last)
    pub fn vertices(&self) -> &[Vector3<f64>] {
        &self.vertices
    }

    /// Get number of unique vertices (excluding closure)
    pub fn num_vertices(&self) -> usize {
        self.vertices.len() - 1
    }

    /// Add a custom property (builder pattern)
    pub fn add_property(mut self, key: &str, value: JsonValue) -> Self {
        self.properties.insert(key.to_string(), value);
        self
    }

    /// Get center longitude in degrees (quick accessor)
    pub fn lon(&self) -> f64 {
        self.center.x
    }

    /// Get center latitude in degrees (quick accessor)
    pub fn lat(&self) -> f64 {
        self.center.y
    }

    /// Get center altitude in meters (quick accessor)
    pub fn alt(&self) -> f64 {
        self.center.z
    }

    /// Get center longitude with angle format conversion
    ///
    /// # Arguments
    /// - `angle_format`: Desired output format (Degrees or Radians)
    ///
    /// # Example
    /// ```
    /// use brahe::access::PolygonLocation;
    /// use brahe::constants::AngleFormat;
    /// use nalgebra::Vector3;
    ///
    /// let vertices = vec![
    ///     Vector3::new(10.0, 50.0, 0.0),
    ///     Vector3::new(11.0, 50.0, 0.0),
    ///     Vector3::new(11.0, 51.0, 0.0),
    ///     Vector3::new(10.0, 51.0, 0.0),
    ///     Vector3::new(10.0, 50.0, 0.0),
    /// ];
    /// let poly = PolygonLocation::new(vertices).unwrap();
    /// let lon_deg = poly.longitude(AngleFormat::Degrees);
    /// let lon_rad = poly.longitude(AngleFormat::Radians);
    /// ```
    pub fn longitude(&self, angle_format: AngleFormat) -> f64 {
        match angle_format {
            AngleFormat::Degrees => self.center.x,
            AngleFormat::Radians => self.center.x.to_radians(),
        }
    }

    /// Get center latitude with angle format conversion
    ///
    /// # Arguments
    /// - `angle_format`: Desired output format (Degrees or Radians)
    pub fn latitude(&self, angle_format: AngleFormat) -> f64 {
        match angle_format {
            AngleFormat::Degrees => self.center.y,
            AngleFormat::Radians => self.center.y.to_radians(),
        }
    }

    /// Get center altitude in meters
    pub fn altitude(&self) -> f64 {
        self.center.z
    }
}

impl AccessibleLocation for PolygonLocation {
    fn center_geodetic(&self) -> Vector3<f64> {
        self.center
    }

    fn center_ecef(&self) -> Vector3<f64> {
        self.center_ecef.unwrap_or_else(|| {
            position_geodetic_to_ecef(self.center, AngleFormat::Degrees)
                .expect("Invalid geodetic coordinates")
        })
    }

    fn properties(&self) -> &HashMap<String, JsonValue> {
        &self.properties
    }

    fn properties_mut(&mut self) -> &mut HashMap<String, JsonValue> {
        &mut self.properties
    }

    fn to_geojson(&self) -> JsonValue {
        let mut props = self.properties.clone();

        // Add identity fields
        if let Some(name) = &self.name {
            props.insert("name".to_string(), json!(name));
        }
        if let Some(id) = self.id {
            props.insert("id".to_string(), json!(id));
        }
        if let Some(uuid) = self.uuid {
            props.insert("uuid".to_string(), json!(uuid.to_string()));
        }

        // Convert vertices to JSON
        let coords: Vec<Vec<f64>> = self.vertices.iter().map(|v| vec![v.x, v.y, v.z]).collect();

        json!({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords]  // Outer ring only
            },
            "properties": props
        })
    }
}

impl Identifiable for PolygonLocation {
    fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    fn with_uuid(mut self, uuid: Uuid) -> Self {
        self.uuid = Some(uuid);
        self
    }

    fn with_new_uuid(mut self) -> Self {
        self.uuid = Some(Uuid::new_v4());
        self
    }

    fn with_id(mut self, id: u64) -> Self {
        self.id = Some(id);
        self
    }

    fn with_identity(mut self, name: Option<&str>, uuid: Option<Uuid>, id: Option<u64>) -> Self {
        self.name = name.map(|s| s.to_string());
        self.uuid = uuid;
        self.id = id;
        self
    }

    fn set_identity(&mut self, name: Option<&str>, uuid: Option<Uuid>, id: Option<u64>) {
        self.name = name.map(|s| s.to_string());
        self.uuid = uuid;
        self.id = id;
    }

    fn set_id(&mut self, id: Option<u64>) {
        self.id = id;
    }

    fn set_name(&mut self, name: Option<&str>) {
        self.name = name.map(|s| s.to_string());
    }

    fn generate_uuid(&mut self) {
        self.uuid = Some(Uuid::new_v4());
    }

    fn get_id(&self) -> Option<u64> {
        self.id
    }

    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn get_uuid(&self) -> Option<Uuid> {
        self.uuid
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_point_location_new() {
        let loc = PointLocation::new(15.4, 78.2, 0.0);

        assert_eq!(loc.lon(), 15.4);
        assert_eq!(loc.lat(), 78.2);
        assert_eq!(loc.alt(), 0.0);

        // ECEF should be computed
        let ecef = loc.center_ecef();
        assert!(ecef.norm() > 0.0);
    }

    #[test]
    fn test_point_location_identifiable() {
        let loc = PointLocation::new(15.4, 78.2, 0.0)
            .with_name("Svalbard")
            .with_id(42)
            .with_new_uuid();

        assert_eq!(loc.get_name(), Some("Svalbard"));
        assert_eq!(loc.get_id(), Some(42));
        assert!(loc.get_uuid().is_some());
    }

    #[test]
    fn test_point_location_properties() {
        let loc = PointLocation::new(15.4, 78.2, 0.0)
            .add_property("country", json!("Norway"))
            .add_property("population", json!(2500));

        assert_eq!(loc.properties().get("country").unwrap(), "Norway");
        assert_eq!(loc.properties().get("population").unwrap(), 2500);
    }

    #[test]
    fn test_point_location_geojson_roundtrip() {
        let original = PointLocation::new(15.4, 78.2, 100.0)
            .with_name("Svalbard")
            .add_property("country", json!("Norway"));

        let geojson = original.to_geojson();
        let reconstructed = PointLocation::from_geojson(&geojson).unwrap();

        assert_eq!(reconstructed.lon(), 15.4);
        assert_eq!(reconstructed.lat(), 78.2);
        assert_eq!(reconstructed.alt(), 100.0);
        assert_eq!(reconstructed.get_name(), Some("Svalbard"));
        assert_eq!(reconstructed.properties().get("country").unwrap(), "Norway");
    }

    #[test]
    fn test_polygon_location_new() {
        let vertices = vec![
            Vector3::new(10.0, 50.0, 0.0),
            Vector3::new(11.0, 50.0, 0.0),
            Vector3::new(11.0, 51.0, 0.0),
            Vector3::new(10.0, 51.0, 0.0),
            Vector3::new(10.0, 50.0, 0.0),
        ];

        let poly = PolygonLocation::new(vertices.clone()).unwrap();

        assert_eq!(poly.num_vertices(), 4);
        assert_eq!(poly.vertices().len(), 5); // Including closure

        // Check centroid
        let center = poly.center_geodetic();
        assert_abs_diff_eq!(center.x, 10.5, epsilon = 0.01); // lon
        assert_abs_diff_eq!(center.y, 50.5, epsilon = 0.01); // lat
    }

    #[test]
    fn test_polygon_location_auto_close() {
        // Missing closure vertex
        let vertices = vec![
            Vector3::new(10.0, 50.0, 0.0),
            Vector3::new(11.0, 50.0, 0.0),
            Vector3::new(11.0, 51.0, 0.0),
            Vector3::new(10.0, 51.0, 0.0),
        ];

        let poly = PolygonLocation::new(vertices).unwrap();

        // Should auto-close
        assert_eq!(poly.vertices().len(), 5);
        assert_eq!(poly.vertices()[0], poly.vertices()[4]);
    }

    #[test]
    fn test_polygon_location_validation() {
        // Too few vertices
        let result = PolygonLocation::new(vec![
            Vector3::new(10.0, 50.0, 0.0),
            Vector3::new(11.0, 50.0, 0.0),
        ]);
        assert!(result.is_err());

        // Duplicate consecutive vertices
        let result = PolygonLocation::new(vec![
            Vector3::new(10.0, 50.0, 0.0),
            Vector3::new(10.0, 50.0, 0.0), // Duplicate
            Vector3::new(11.0, 50.0, 0.0),
            Vector3::new(10.0, 50.0, 0.0),
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn test_polygon_location_geojson_roundtrip() {
        let vertices = vec![
            Vector3::new(10.0, 50.0, 0.0),
            Vector3::new(11.0, 50.0, 0.0),
            Vector3::new(11.0, 51.0, 0.0),
            Vector3::new(10.0, 51.0, 0.0),
            Vector3::new(10.0, 50.0, 0.0),
        ];

        let original = PolygonLocation::new(vertices.clone())
            .unwrap()
            .with_name("AOI-1")
            .add_property("region", json!("Europe"));

        let geojson = original.to_geojson();
        let reconstructed = PolygonLocation::from_geojson(&geojson).unwrap();

        assert_eq!(reconstructed.num_vertices(), 4);
        assert_eq!(reconstructed.get_name(), Some("AOI-1"));
        assert_eq!(reconstructed.properties().get("region").unwrap(), "Europe");
    }

    // ========================================================================
    // Comprehensive PointLocation Tests
    // ========================================================================

    #[test]
    fn test_point_location_coordinate_accessors() {
        let loc = PointLocation::new(15.4, 78.2, 500.0);

        // Quick accessors (always degrees/meters)
        assert_eq!(loc.lon(), 15.4);
        assert_eq!(loc.lat(), 78.2);
        assert_eq!(loc.alt(), 500.0);
        assert_eq!(loc.altitude(), 500.0);

        // Format-aware accessors
        assert_eq!(loc.longitude(AngleFormat::Degrees), 15.4);
        assert_eq!(loc.latitude(AngleFormat::Degrees), 78.2);

        // Radians conversion
        let lon_rad = loc.longitude(AngleFormat::Radians);
        let lat_rad = loc.latitude(AngleFormat::Radians);
        assert_abs_diff_eq!(lon_rad, 15.4_f64.to_radians(), epsilon = 1e-10);
        assert_abs_diff_eq!(lat_rad, 78.2_f64.to_radians(), epsilon = 1e-10);
    }

    #[test]
    fn test_point_location_center_geodetic() {
        let loc = PointLocation::new(15.4, 78.2, 500.0);
        let center = loc.center_geodetic();

        assert_eq!(center.x, 15.4);
        assert_eq!(center.y, 78.2);
        assert_eq!(center.z, 500.0);
    }

    #[test]
    fn test_point_location_center_ecef() {
        let loc = PointLocation::new(0.0, 0.0, 0.0);
        let ecef = loc.center_ecef();

        // At equator, longitude 0, ECEF should be approximately [R_EARTH, 0, 0]
        assert_abs_diff_eq!(ecef.x, 6378137.0, epsilon = 1.0);
        assert_abs_diff_eq!(ecef.y, 0.0, epsilon = 1.0);
        assert_abs_diff_eq!(ecef.z, 0.0, epsilon = 1.0);
    }

    #[test]
    fn test_point_location_properties_accessor() {
        let mut loc = PointLocation::new(15.4, 78.2, 0.0);

        // Test properties() getter
        assert!(loc.properties().is_empty());

        // Test properties_mut() setter
        loc.properties_mut()
            .insert("test_key".to_string(), json!("test_value"));

        assert_eq!(loc.properties().get("test_key").unwrap(), "test_value");
    }

    #[test]
    fn test_point_location_add_property_builder() {
        let loc = PointLocation::new(15.4, 78.2, 0.0)
            .add_property("key1", json!("value1"))
            .add_property("key2", json!(42))
            .add_property("key3", json!(true));

        assert_eq!(loc.properties().get("key1").unwrap(), "value1");
        assert_eq!(loc.properties().get("key2").unwrap(), 42);
        assert_eq!(loc.properties().get("key3").unwrap(), true);
    }

    #[test]
    fn test_point_location_to_geojson_required_fields() {
        let loc = PointLocation::new(15.4, 78.2, 100.0)
            .with_name("TestLocation")
            .with_id(42)
            .with_new_uuid()
            .add_property("custom_prop", json!("custom_value"));

        let geojson = loc.to_geojson();

        // Validate top-level structure
        assert_eq!(geojson.get("type").unwrap(), "Feature");

        // Validate geometry
        let geometry = geojson.get("geometry").unwrap();
        assert_eq!(geometry.get("type").unwrap(), "Point");

        let coords = geometry.get("coordinates").unwrap().as_array().unwrap();
        assert_eq!(coords.len(), 3);
        assert_eq!(coords[0].as_f64().unwrap(), 15.4);
        assert_eq!(coords[1].as_f64().unwrap(), 78.2);
        assert_eq!(coords[2].as_f64().unwrap(), 100.0);

        // Validate properties
        let properties = geojson.get("properties").unwrap();
        assert_eq!(properties.get("name").unwrap(), "TestLocation");
        assert_eq!(properties.get("id").unwrap(), 42);
        assert!(properties.get("uuid").is_some());
        assert_eq!(properties.get("custom_prop").unwrap(), "custom_value");
    }

    #[test]
    fn test_point_location_from_geojson_minimal() {
        let geojson = json!({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [15.4, 78.2]
            },
            "properties": {}
        });

        let loc = PointLocation::from_geojson(&geojson).unwrap();
        assert_eq!(loc.lon(), 15.4);
        assert_eq!(loc.lat(), 78.2);
        assert_eq!(loc.alt(), 0.0); // Default altitude
    }

    #[test]
    fn test_point_location_from_geojson_with_altitude() {
        let geojson = json!({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [15.4, 78.2, 500.0]
            },
            "properties": {}
        });

        let loc = PointLocation::from_geojson(&geojson).unwrap();
        assert_eq!(loc.alt(), 500.0);
    }

    #[test]
    fn test_point_location_from_geojson_invalid_type() {
        let geojson = json!({
            "type": "FeatureCollection",
            "features": []
        });

        let result = PointLocation::from_geojson(&geojson);
        assert!(result.is_err());
    }

    #[test]
    fn test_point_location_from_geojson_wrong_geometry() {
        let geojson = json!({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]]
            },
            "properties": {}
        });

        let result = PointLocation::from_geojson(&geojson);
        assert!(result.is_err());
    }

    #[test]
    fn test_point_location_identifiable_methods() {
        let uuid = Uuid::new_v4();

        // Test with_name
        let loc = PointLocation::new(0.0, 0.0, 0.0).with_name("Test");
        assert_eq!(loc.get_name(), Some("Test"));

        // Test with_id
        let loc = PointLocation::new(0.0, 0.0, 0.0).with_id(123);
        assert_eq!(loc.get_id(), Some(123));

        // Test with_uuid
        let loc = PointLocation::new(0.0, 0.0, 0.0).with_uuid(uuid);
        assert_eq!(loc.get_uuid(), Some(uuid));

        // Test with_new_uuid
        let loc = PointLocation::new(0.0, 0.0, 0.0).with_new_uuid();
        assert!(loc.get_uuid().is_some());

        // Test with_identity
        let loc = PointLocation::new(0.0, 0.0, 0.0).with_identity(
            Some("Combined"),
            Some(uuid),
            Some(456),
        );
        assert_eq!(loc.get_name(), Some("Combined"));
        assert_eq!(loc.get_uuid(), Some(uuid));
        assert_eq!(loc.get_id(), Some(456));

        // Test set_identity
        let mut loc = PointLocation::new(0.0, 0.0, 0.0);
        loc.set_identity(Some("NewName"), Some(uuid), Some(789));
        assert_eq!(loc.get_name(), Some("NewName"));
        assert_eq!(loc.get_uuid(), Some(uuid));
        assert_eq!(loc.get_id(), Some(789));

        // Test set_id
        let mut loc = PointLocation::new(0.0, 0.0, 0.0);
        loc.set_id(Some(999));
        assert_eq!(loc.get_id(), Some(999));

        // Test set_name
        let mut loc = PointLocation::new(0.0, 0.0, 0.0);
        loc.set_name(Some("SetName"));
        assert_eq!(loc.get_name(), Some("SetName"));

        // Test generate_uuid
        let mut loc = PointLocation::new(0.0, 0.0, 0.0);
        loc.generate_uuid();
        assert!(loc.get_uuid().is_some());
    }

    // ========================================================================
    // Comprehensive PolygonLocation Tests
    // ========================================================================

    #[test]
    fn test_polygon_location_coordinate_accessors() {
        let vertices = vec![
            Vector3::new(10.0, 50.0, 100.0),
            Vector3::new(11.0, 50.0, 100.0),
            Vector3::new(11.0, 51.0, 100.0),
            Vector3::new(10.0, 51.0, 100.0),
            Vector3::new(10.0, 50.0, 100.0),
        ];

        let poly = PolygonLocation::new(vertices).unwrap();

        // Center should be centroid
        assert_abs_diff_eq!(poly.lon(), 10.5, epsilon = 0.01);
        assert_abs_diff_eq!(poly.lat(), 50.5, epsilon = 0.01);
        assert_abs_diff_eq!(poly.alt(), 100.0, epsilon = 0.01);
        assert_abs_diff_eq!(poly.altitude(), 100.0, epsilon = 0.01);

        // Format-aware accessors
        assert_abs_diff_eq!(poly.longitude(AngleFormat::Degrees), 10.5, epsilon = 0.01);
        assert_abs_diff_eq!(poly.latitude(AngleFormat::Degrees), 50.5, epsilon = 0.01);

        // Radians conversion
        let lon_rad = poly.longitude(AngleFormat::Radians);
        let lat_rad = poly.latitude(AngleFormat::Radians);
        assert_abs_diff_eq!(lon_rad, 10.5_f64.to_radians(), epsilon = 1e-10);
        assert_abs_diff_eq!(lat_rad, 50.5_f64.to_radians(), epsilon = 1e-10);
    }

    #[test]
    fn test_polygon_location_vertices_accessor() {
        let vertices = vec![
            Vector3::new(10.0, 50.0, 0.0),
            Vector3::new(11.0, 50.0, 0.0),
            Vector3::new(11.0, 51.0, 0.0),
            Vector3::new(10.0, 51.0, 0.0),
            Vector3::new(10.0, 50.0, 0.0),
        ];

        let poly = PolygonLocation::new(vertices.clone()).unwrap();

        // Test vertices() getter
        let verts = poly.vertices();
        assert_eq!(verts.len(), 5);
        assert_eq!(verts[0], vertices[0]);
        assert_eq!(verts[4], vertices[4]);
    }

    #[test]
    fn test_polygon_location_num_vertices() {
        let vertices = vec![
            Vector3::new(10.0, 50.0, 0.0),
            Vector3::new(11.0, 50.0, 0.0),
            Vector3::new(11.0, 51.0, 0.0),
            Vector3::new(10.0, 51.0, 0.0),
            Vector3::new(10.0, 50.0, 0.0),
        ];

        let poly = PolygonLocation::new(vertices).unwrap();

        // Should exclude the closure vertex
        assert_eq!(poly.num_vertices(), 4);
    }

    #[test]
    fn test_polygon_location_center_geodetic() {
        let vertices = vec![
            Vector3::new(10.0, 50.0, 100.0),
            Vector3::new(11.0, 50.0, 100.0),
            Vector3::new(11.0, 51.0, 100.0),
            Vector3::new(10.0, 51.0, 100.0),
            Vector3::new(10.0, 50.0, 100.0),
        ];

        let poly = PolygonLocation::new(vertices).unwrap();
        let center = poly.center_geodetic();

        assert_abs_diff_eq!(center.x, 10.5, epsilon = 0.01);
        assert_abs_diff_eq!(center.y, 50.5, epsilon = 0.01);
        assert_abs_diff_eq!(center.z, 100.0, epsilon = 0.01);
    }

    #[test]
    fn test_polygon_location_center_ecef() {
        let vertices = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 0.0),
        ];

        let poly = PolygonLocation::new(vertices).unwrap();
        let ecef = poly.center_ecef();

        // Center should be near equator, should have valid ECEF coordinates
        assert!(ecef.norm() > 6000000.0); // Should be roughly Earth radius
    }

    #[test]
    fn test_polygon_location_properties_accessor() {
        let vertices = vec![
            Vector3::new(10.0, 50.0, 0.0),
            Vector3::new(11.0, 50.0, 0.0),
            Vector3::new(11.0, 51.0, 0.0),
            Vector3::new(10.0, 51.0, 0.0),
            Vector3::new(10.0, 50.0, 0.0),
        ];

        let mut poly = PolygonLocation::new(vertices).unwrap();

        // Test properties() getter
        assert!(poly.properties().is_empty());

        // Test properties_mut() setter
        poly.properties_mut()
            .insert("test_key".to_string(), json!("test_value"));

        assert_eq!(poly.properties().get("test_key").unwrap(), "test_value");
    }

    #[test]
    fn test_polygon_location_add_property_builder() {
        let vertices = vec![
            Vector3::new(10.0, 50.0, 0.0),
            Vector3::new(11.0, 50.0, 0.0),
            Vector3::new(11.0, 51.0, 0.0),
            Vector3::new(10.0, 51.0, 0.0),
            Vector3::new(10.0, 50.0, 0.0),
        ];

        let poly = PolygonLocation::new(vertices)
            .unwrap()
            .add_property("key1", json!("value1"))
            .add_property("key2", json!(42))
            .add_property("key3", json!(true));

        assert_eq!(poly.properties().get("key1").unwrap(), "value1");
        assert_eq!(poly.properties().get("key2").unwrap(), 42);
        assert_eq!(poly.properties().get("key3").unwrap(), true);
    }

    #[test]
    fn test_polygon_location_to_geojson_required_fields() {
        let vertices = vec![
            Vector3::new(10.0, 50.0, 0.0),
            Vector3::new(11.0, 50.0, 0.0),
            Vector3::new(11.0, 51.0, 0.0),
            Vector3::new(10.0, 51.0, 0.0),
            Vector3::new(10.0, 50.0, 0.0),
        ];

        let poly = PolygonLocation::new(vertices)
            .unwrap()
            .with_name("TestPolygon")
            .with_id(99)
            .with_new_uuid()
            .add_property("custom_prop", json!("custom_value"));

        let geojson = poly.to_geojson();

        // Validate top-level structure
        assert_eq!(geojson.get("type").unwrap(), "Feature");

        // Validate geometry
        let geometry = geojson.get("geometry").unwrap();
        assert_eq!(geometry.get("type").unwrap(), "Polygon");

        let coords = geometry.get("coordinates").unwrap().as_array().unwrap();
        assert_eq!(coords.len(), 1); // One outer ring

        let outer_ring = coords[0].as_array().unwrap();
        assert_eq!(outer_ring.len(), 5); // 4 unique + closure

        // Check first vertex
        let first_vertex = outer_ring[0].as_array().unwrap();
        assert_eq!(first_vertex[0].as_f64().unwrap(), 10.0);
        assert_eq!(first_vertex[1].as_f64().unwrap(), 50.0);
        assert_eq!(first_vertex[2].as_f64().unwrap(), 0.0);

        // Validate properties
        let properties = geojson.get("properties").unwrap();
        assert_eq!(properties.get("name").unwrap(), "TestPolygon");
        assert_eq!(properties.get("id").unwrap(), 99);
        assert!(properties.get("uuid").is_some());
        assert_eq!(properties.get("custom_prop").unwrap(), "custom_value");
    }

    #[test]
    fn test_polygon_location_from_geojson_minimal() {
        let geojson = json!({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [10.0, 50.0],
                    [11.0, 50.0],
                    [11.0, 51.0],
                    [10.0, 51.0],
                    [10.0, 50.0]
                ]]
            },
            "properties": {}
        });

        let poly = PolygonLocation::from_geojson(&geojson).unwrap();
        assert_eq!(poly.num_vertices(), 4);
    }

    #[test]
    fn test_polygon_location_from_geojson_with_altitude() {
        let geojson = json!({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [10.0, 50.0, 100.0],
                    [11.0, 50.0, 100.0],
                    [11.0, 51.0, 100.0],
                    [10.0, 51.0, 100.0],
                    [10.0, 50.0, 100.0]
                ]]
            },
            "properties": {}
        });

        let poly = PolygonLocation::from_geojson(&geojson).unwrap();
        assert_abs_diff_eq!(poly.alt(), 100.0, epsilon = 0.01);
    }

    #[test]
    fn test_polygon_location_from_geojson_invalid_type() {
        let geojson = json!({
            "type": "FeatureCollection",
            "features": []
        });

        let result = PolygonLocation::from_geojson(&geojson);
        assert!(result.is_err());
    }

    #[test]
    fn test_polygon_location_from_geojson_wrong_geometry() {
        let geojson = json!({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [0.0, 0.0]
            },
            "properties": {}
        });

        let result = PolygonLocation::from_geojson(&geojson);
        assert!(result.is_err());
    }

    #[test]
    fn test_polygon_location_identifiable_methods() {
        let vertices = vec![
            Vector3::new(10.0, 50.0, 0.0),
            Vector3::new(11.0, 50.0, 0.0),
            Vector3::new(11.0, 51.0, 0.0),
            Vector3::new(10.0, 51.0, 0.0),
            Vector3::new(10.0, 50.0, 0.0),
        ];

        let uuid = Uuid::new_v4();

        // Test with_name
        let poly = PolygonLocation::new(vertices.clone())
            .unwrap()
            .with_name("Test");
        assert_eq!(poly.get_name(), Some("Test"));

        // Test with_id
        let poly = PolygonLocation::new(vertices.clone()).unwrap().with_id(123);
        assert_eq!(poly.get_id(), Some(123));

        // Test with_uuid
        let poly = PolygonLocation::new(vertices.clone())
            .unwrap()
            .with_uuid(uuid);
        assert_eq!(poly.get_uuid(), Some(uuid));

        // Test with_new_uuid
        let poly = PolygonLocation::new(vertices.clone())
            .unwrap()
            .with_new_uuid();
        assert!(poly.get_uuid().is_some());

        // Test with_identity
        let poly = PolygonLocation::new(vertices.clone())
            .unwrap()
            .with_identity(Some("Combined"), Some(uuid), Some(456));
        assert_eq!(poly.get_name(), Some("Combined"));
        assert_eq!(poly.get_uuid(), Some(uuid));
        assert_eq!(poly.get_id(), Some(456));

        // Test set_identity
        let mut poly = PolygonLocation::new(vertices.clone()).unwrap();
        poly.set_identity(Some("NewName"), Some(uuid), Some(789));
        assert_eq!(poly.get_name(), Some("NewName"));
        assert_eq!(poly.get_uuid(), Some(uuid));
        assert_eq!(poly.get_id(), Some(789));

        // Test set_id
        let mut poly = PolygonLocation::new(vertices.clone()).unwrap();
        poly.set_id(Some(999));
        assert_eq!(poly.get_id(), Some(999));

        // Test set_name
        let mut poly = PolygonLocation::new(vertices.clone()).unwrap();
        poly.set_name(Some("SetName"));
        assert_eq!(poly.get_name(), Some("SetName"));

        // Test generate_uuid
        let mut poly = PolygonLocation::new(vertices.clone()).unwrap();
        poly.generate_uuid();
        assert!(poly.get_uuid().is_some());
    }
}
