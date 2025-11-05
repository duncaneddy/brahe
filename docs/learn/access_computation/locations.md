# Locations

Locations represent ground positions or areas that satellites can access. Brahe provides two fundamental location types—points and polygons—with full GeoJSON interoperability and extensible metadata support.

All location types implement the `AccessibleLocation` trait, which provides a common interface for coordinate access, property management, and GeoJSON import/export. This design allows you to work with different location geometries through a unified API.

!!! warning "Coordinate Units"
    All coordinates are specified in geodetic longitude (λ), latitude (φ), and altitude (h) using the WGS84 reference frame. All units are in degrees (for λ and φ) and meters (for h) for consistency with the GeoJSON standard.

## PointLocation

A `PointLocation` represents a single geodetic point on Earth's surface. This is the most common location type, used for ground stations, cities, or specific observation points.

### Initialization from Coordinates

Create a point location from geodetic coordinates (longitude, latitude, altitude):

=== "Python"

    ``` python
    --8<-- "./examples/access/locations/point_from_coordinates.py:9"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/locations/point_from_coordinates.rs:5"
    ```

!!! info "Coordinate Units"
    Python uses degrees for input convenience. Rust uses radians (SI standard). Both use meters for altitude.

### Initialization from GeoJSON

Load locations from GeoJSON strings or files:

=== "Python"

    ``` python
    --8<-- "./examples/access/locations/point_from_geojson.py:9"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/locations/point_from_geojson.rs:5"
    ```

### Accessing Coordinates

Retrieve coordinates in different formats:

=== "Python"

    ``` python
    --8<-- "./examples/access/locations/point_accessing_coordinates.py:9"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/locations/point_accessing_coordinates.rs:5"
    ```

## PolygonLocation

A `PolygonLocation` represents a closed polygon area on Earth's surface. This is useful for imaging regions, coverage zones, or geographic areas of interest.

### Initialization from Vertices

Create a polygon from a list of vertices:

=== "Python"

    ``` python
    --8<-- "./examples/access/locations/polygon_from_vertices.py:9"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/locations/polygon_from_vertices.rs:5"
    ```

### Initialization from GeoJSON

Load polygon areas from GeoJSON:

=== "Python"

    ``` python
    --8<-- "./examples/access/locations/polygon_from_geojson.py:9"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/locations/polygon_from_geojson.rs:5"
    ```

## Working with Properties

Both location types support custom properties for storing metadata:

=== "Python"

    ``` python
    --8<-- "./examples/access/locations/working_with_properties.py:9"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/locations/working_with_properties.rs:5"
    ```

## Exporting to GeoJSON

Convert locations back to GeoJSON format:

=== "Python"

    ``` python
    --8<-- "./examples/access/locations/exporting_to_geojson.py:9"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/locations/exporting_to_geojson.rs:5"
    ```

---

## See Also

- [Constraints](constraints.md) - Defining access criteria for locations
- [Computation](computation.md) - Access algorithms and property computation
- [API Reference: Locations](../../library_api/access/locations.md)
- [Example: Predicting Ground Contacts](../../examples/ground_contacts.md)
