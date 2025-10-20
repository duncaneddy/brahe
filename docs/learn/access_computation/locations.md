# Locations

Locations represent ground positions or areas that satellites can access. Brahe provides two fundamental location types—points and polygons—with full GeoJSON interoperability and extensible metadata support.

## Overview

All locations in Brahe share common capabilities:

- **Geographic coordinates**: Geodetic (lat/lon/alt) and ECEF (Earth-fixed Cartesian)
- **Identification**: Name, numeric ID, and UUID support
- **Properties**: Extensible metadata dictionary for custom data
- **GeoJSON**: Import/export compatibility with GIS systems
- **Type safety**: Strong typing with the `AccessibleLocation` trait

## PointLocation

Point locations represent discrete positions on Earth's surface, such as:

- Ground stations and tracking sites
- Imaging targets and waypoints

### Creating Point Locations

```python
import brahe as bh

# Simple creation with lon, lat, alt
svalbard = bh.PointLocation(15.4, 78.2, 0.0)

# With identification
svalbard = bh.PointLocation(15.4, 78.2, 0.0).with_name("Svalbard")

# With multiple identifiers
station = bh.PointLocation(-117.2, 34.1, 500.0) \
    .with_name("Goldstone") \
    .with_id(42) \
    .with_new_uuid()

# With custom properties
loc = bh.PointLocation(15.4, 78.2, 0.0) \
    .with_name("Svalbard") \
    .add_property("country", "Norway") \
    .add_property("operator", "KSAT") \
    .add_property("frequency_band", "X-band")
```

### Coordinate Systems

Point locations maintain coordinates in both geodetic and ECEF systems:

```python
loc = bh.PointLocation(15.4, 78.2, 500.0)

# Geodetic coordinates (degrees, meters)
lon = loc.lon()      # 15.4 degrees
lat = loc.lat()      # 78.2 degrees
alt = loc.alt()      # 500.0 meters

# ECEF coordinates (meters)
ecef = loc.center_ecef()  # Vector3 [x, y, z] in meters

# Access with angle format conversion
lon_deg = loc.longitude(bh.AngleFormat.DEGREES)
lon_rad = loc.longitude(bh.AngleFormat.RADIANS)
```

**Coordinate Conventions**:
- Longitude: -180° to +180° (negative = West, positive = East)
- Latitude: -90° to +90° (negative = South, positive = North)
- Altitude: Height above WGS84 ellipsoid (meters)
- ECEF: Earth-fixed Cartesian coordinates (meters)

### GeoJSON Integration

Point locations seamlessly convert to/from GeoJSON Feature format:

```python
import brahe as bh
import json

# Create from GeoJSON
geojson = {
    "type": "Feature",
    "geometry": {
        "type": "Point",
        "coordinates": [15.4, 78.2, 500.0]  # [lon, lat, alt]
    },
    "properties": {
        "name": "Svalbard",
        "country": "Norway"
    }
}

loc = bh.PointLocation.from_geojson(geojson)

# Export to GeoJSON
geojson = loc.to_geojson()
print(json.dumps(geojson, indent=2))
```

**GeoJSON format details**:
- Coordinates: `[longitude, latitude, altitude]` (note lon/lat order!)
- Altitude is optional (defaults to 0.0)
- Properties include identity fields (name, id, uuid) plus custom metadata
- Fully compatible with QGIS, GeoPandas, and other GIS tools

### Use Cases

**Ground Stations**:
```python
# Ground station network
stations = [
    bh.PointLocation(15.4, 78.2, 0.0).with_name("Svalbard"),
    bh.PointLocation(-64.5, -31.5, 0.0).with_name("Malargue"),
    bh.PointLocation(-117.2, 34.1, 0.0).with_name("Goldstone"),
]

for station in stations:
    station.add_property("elevation_mask_deg", 5.0)
    station.add_property("max_data_rate_mbps", 300.0)
```

**Imaging Targets**:
```python
# Points of interest
targets = [
    bh.PointLocation(2.3, 48.9, 0.0)
        .with_name("Paris")
        .add_property("priority", "high")
        .add_property("min_off_nadir_deg", 5.0)
        .add_property("max_off_nadir_deg", 30.0),

    bh.PointLocation(139.7, 35.7, 0.0)
        .with_name("Tokyo")
        .add_property("priority", "medium")
        .add_property("cloud_tolerance_pct", 20.0),
]
```

**Grid Tessellation**:
```python
# Create global grid
import numpy as np

grid_points = []
for lat in np.arange(-90, 90, 10):
    for lon in np.arange(-180, 180, 10):
        point = bh.PointLocation(lon, lat, 0.0) \
            .with_name(f"Grid_{lat}_{lon}") \
            .add_property("grid_cell_id", f"{lat}_{lon}")
        grid_points.append(point)

print(f"Created {len(grid_points)} grid points")
```

## PolygonLocation

Polygon locations represent areas on Earth's surface, such as:

- Areas of interest (countries, regions, sea zones)
- Imaging swaths and coverage footprints

### Creating Polygon Locations

```python
import brahe as bh
from nalgebra import Vector3  # From brahe Rust bindings
import numpy as np

# Define vertices [lon, lat, alt]
vertices = [
    Vector3(10.0, 50.0, 0.0),
    Vector3(11.0, 50.0, 0.0),
    Vector3(11.0, 51.0, 0.0),
    Vector3(10.0, 51.0, 0.0),
    Vector3(10.0, 50.0, 0.0),  # Closed polygon (first == last)
]

# Create polygon
aoi = bh.PolygonLocation(vertices).with_name("AOI-1")

# Auto-closure: First/last vertex don't need to match
vertices = [
    Vector3(10.0, 50.0, 0.0),
    Vector3(11.0, 50.0, 0.0),
    Vector3(11.0, 51.0, 0.0),
    Vector3(10.0, 51.0, 0.0),
    # Last vertex automatically added to close polygon
]
aoi = bh.PolygonLocation(vertices)  # Auto-closed

# With properties
region = bh.PolygonLocation(vertices) \
    .with_name("Europe-Central") \
    .add_property("population_millions", 82) \
    .add_property("min_cloud_free_pct", 80)
```

### Polygon Properties

Polygons compute their centroid automatically:

```python
poly = bh.PolygonLocation(vertices)

# Centroid coordinates
center_lon = poly.lon  # Average longitude
center_lat = poly.lat  # Average latitude
center_alt = poly.alt  # Average altitude

# Vertex access
verts = poly.vertices  # All vertices (including closure)
num_unique = poly.num_vertices  # Excluding closure vertex

# ECEF center
ecef_center = poly.center_ecef
```

### GeoJSON for Polygons

```python
# Create from GeoJSON
geojson = {
    "type": "Feature",
    "geometry": {
        "type": "Polygon",
        "coordinates": [[  # Note: nested array for outer ring
            [10.0, 50.0, 0.0],
            [11.0, 50.0, 0.0],
            [11.0, 51.0, 0.0],
            [10.0, 51.0, 0.0],
            [10.0, 50.0, 0.0]
        ]]
    },
    "properties": {
        "name": "AOI-1",
        "region": "Europe"
    }
}

poly = bh.PolygonLocation.from_geojson(geojson)

# Export to GeoJSON
geojson = poly.to_geojson()
```

**Polygon GeoJSON notes**:
- Coordinates are nested: `[[[lon, lat, alt], ...]]` (outer ring)
- First and last vertex must match (closed)
- All vertices must be unique (except closure)

??? warning "Polygon Validity"
    Polygons must not contain holes.


### Use Cases

**Coverage Analysis**:
```python
# Define coverage regions
regions = [
    bh.PolygonLocation([
        Vector3(-10.0, 35.0, 0.0),
        Vector3(40.0, 35.0, 0.0),
        Vector3(40.0, 70.0, 0.0),
        Vector3(-10.0, 70.0, 0.0),
    ]).with_name("Europe"),

    bh.PolygonLocation([
        Vector3(-125.0, 25.0, 0.0),
        Vector3(-65.0, 25.0, 0.0),
        Vector3(-65.0, 50.0, 0.0),
        Vector3(-125.0, 50.0, 0.0),
    ]).with_name("North-America"),
]
```

**Sensor Footprint**:
```python
# Imaging sensor footprint (simplified rectangular approximation)
def create_footprint(center_lon, center_lat, width_deg, height_deg):
    """Create rectangular footprint around center point"""
    half_w = width_deg / 2.0
    half_h = height_deg / 2.0

    vertices = [
        Vector3(center_lon - half_w, center_lat - half_h, 0.0),
        Vector3(center_lon + half_w, center_lat - half_h, 0.0),
        Vector3(center_lon + half_w, center_lat + half_h, 0.0),
        Vector3(center_lon - half_w, center_lat + half_h, 0.0),
    ]

    return bh.PolygonLocation(vertices)

# Create footprints
footprint1 = create_footprint(15.0, 50.0, 2.0, 1.5) \
    .with_name("Footprint-1") \
    .add_property("swath_width_km", 100.0)
```

## Extensible Properties

Both location types support arbitrary metadata through a properties dictionary:

### Adding Properties

```python
loc = bh.PointLocation(15.4, 78.2, 0.0)

# Builder pattern
loc = loc.add_property("country", "Norway") \
         .add_property("elevation_mask_deg", 5.0) \
         .add_property("operational_hours", [8, 18])

# Direct access
props = loc.properties()
country = props.get("country")
```

### Common Property Patterns

**Ground Station Metadata**:
```python
station = bh.PointLocation(lon, lat, alt) \
    .with_name("Station-1") \
    .add_property("operator", "ESA") \
    .add_property("dish_diameter_m", 15.0) \
    .add_property("frequency_bands", ["X", "Ka"]) \
    .add_property("max_data_rate_mbps", 800.0) \
    .add_property("elevation_mask_deg", 5.0) \
    .add_property("operational_24_7", True)
```

**Imaging Target Constraints**:
```python
target = bh.PointLocation(lon, lat, 0.0) \
    .with_name("Target-Alpha") \
    .add_property("priority", 10) \
    .add_property("min_off_nadir_deg", 0.0) \
    .add_property("max_off_nadir_deg", 30.0) \
    .add_property("preferred_look_direction", "RIGHT") \
    .add_property("min_sun_elevation_deg", 10.0) \
    .add_property("max_cloud_cover_pct", 20.0)
```

## Identification and Traceability

All locations implement the `Identifiable` trait for tracking and association:

### Identity Methods

```python
# Name-based identification
loc = bh.PointLocation(lon, lat, alt).with_name("Station-1")
assert loc.get_name() == "Station-1"

# Numeric ID
loc = loc.with_id(42)
assert loc.get_id() == 42

# UUID for global uniqueness
import uuid
my_uuid = uuid.uuid4()
loc = loc.with_uuid(my_uuid)
assert loc.get_uuid() == my_uuid

# Or generate new UUID
loc = loc.with_new_uuid()
assert loc.get_uuid() is not None

# Combined identity
loc = loc.with_identity(
    name="Station-1",
    uuid=my_uuid,
    id=42
)
```

### Linking Locations to Access Windows

Access windows preserve location and propagator identifiers:

```python
windows = bh.location_accesses(locations, propagators, start, end, constraint)

for window in windows:
    # Identifiers stored in window
    loc_id = window.location_id
    prop_id = window.propagator_id

    # Find original location/propagator by ID
    matching_loc = next(l for l in locations if l.get_id() == loc_id)
    matching_prop = next(p for p in propagators if p.get_id() == prop_id)

    print(f"Access: {matching_prop.get_name()} -> {matching_loc.get_name()}")
```

## Common Patterns

### Loading Locations from GeoJSON

```python
import json
import brahe as bh

# Load GeoJSON FeatureCollection
with open("ground_stations.geojson") as f:
    data = json.load(f)

locations = []
for feature in data["features"]:
    loc = bh.PointLocation.from_geojson(feature)
    locations.append(loc)

print(f"Loaded {len(locations)} locations")
```

### Exporting Results to GeoJSON

```python
# Collect access results
windows = bh.location_accesses(locations, propagators, start, end, constraint)

# Create GeoJSON with access statistics
features = []
for loc in locations:
    # Find all windows for this location
    loc_windows = [w for w in windows if w.location_id == loc.get_id()]

    # Add access statistics to properties
    loc = loc.add_property("total_passes", len(loc_windows))
    loc = loc.add_property("total_duration_sec", sum(w.duration for w in loc_windows))

    features.append(loc.to_geojson())

# Export FeatureCollection
geojson = {
    "type": "FeatureCollection",
    "features": features
}

with open("access_results.geojson", "w") as f:
    json.dump(geojson, f, indent=2)
```

### Custom Property Computation

```python
# Compute derived properties
for loc in locations:
    # Geographic region classification
    if loc.lat() > 60:
        region = "polar"
    elif abs(loc.lat()) < 23.5:
        region = "tropical"
    else:
        region = "temperate"

    loc = loc.add_property("climate_region", region)

    # Sun-synchronous orbit access potential
    if 96.0 <= abs(loc.lat()) <= 100.0:
        loc = loc.add_property("sso_compatible", True)
```

## See Also

- [Constraints](constraints.md) - Defining access criteria for locations
- [Computation](computation.md) - Access algorithms and property computation
- [API Reference: Locations](../../library_api/access/locations.md)
- [Example: Svalbard Ground Contacts](../../examples/svalbard_ground_contacts.md)
