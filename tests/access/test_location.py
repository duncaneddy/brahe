"""
Tests for access location module (PointLocation and PolygonLocation).

These tests mirror the Rust implementation tests to ensure Python bindings
work correctly and maintain parity with the Rust implementation.
"""

import pytest
import numpy as np
import brahe as bh


# ========================================================================
# PointLocation Tests
# ========================================================================


class TestPointLocationBasic:
    """Basic PointLocation functionality tests."""

    def test_point_location_new(self):
        """Test creating a new point location."""
        loc = bh.PointLocation(lon=15.4, lat=78.2, alt=0.0)

        assert loc.lon == 15.4
        assert loc.lat == 78.2
        assert loc.alt == 0.0

        # ECEF should be computed
        ecef = loc.center_ecef()
        assert np.linalg.norm(ecef) > 0.0

    def test_point_location_default_altitude(self):
        """Test that altitude defaults to 0.0."""
        loc = bh.PointLocation(lon=15.4, lat=78.2)
        assert loc.alt == 0.0

    def test_point_location_identifiable(self):
        """Test Identifiable trait methods."""
        loc = (
            bh.PointLocation(lon=15.4, lat=78.2, alt=0.0)
            .with_name("Svalbard")
            .with_id(42)
            .with_new_uuid()
        )

        assert loc.get_name() == "Svalbard"
        assert loc.get_id() == 42
        assert loc.get_uuid() is not None

    def test_point_location_properties(self):
        """Test custom properties."""
        loc = (
            bh.PointLocation(lon=15.4, lat=78.2, alt=0.0)
            .add_property("country", "Norway")
            .add_property("population", 2500)
        )

        props = loc.properties
        assert props["country"] == "Norway"
        assert props["population"] == 2500

    def test_point_location_set_property(self):
        """Test set_property method."""
        loc = bh.PointLocation(lon=15.4, lat=78.2, alt=0.0)

        loc.properties["climate"] = "Arctic"

        assert loc.properties["climate"] == "Arctic"


class TestPointLocationCoordinates:
    """Test coordinate accessor methods."""

    def test_coordinate_accessors_quick(self):
        """Test quick coordinate accessors (always degrees/meters)."""
        loc = bh.PointLocation(lon=15.4, lat=78.2, alt=500.0)

        # Quick accessors (always degrees/meters)
        assert loc.lon == 15.4
        assert loc.lat == 78.2
        assert loc.alt == 500.0
        assert loc.altitude() == 500.0

    def test_coordinate_accessors_format_aware(self):
        """Test format-aware coordinate accessors."""
        loc = bh.PointLocation(lon=15.4, lat=78.2, alt=500.0)

        # Format-aware accessors - degrees
        assert loc.longitude(bh.AngleFormat.DEGREES) == 15.4
        assert loc.latitude(bh.AngleFormat.DEGREES) == 78.2

        # Format-aware accessors - radians
        lon_rad = loc.longitude(bh.AngleFormat.RADIANS)
        lat_rad = loc.latitude(bh.AngleFormat.RADIANS)
        assert lon_rad == pytest.approx(np.deg2rad(15.4))
        assert lat_rad == pytest.approx(np.deg2rad(78.2))

    def test_center_geodetic(self):
        """Test center_geodetic() method."""
        loc = bh.PointLocation(lon=15.4, lat=78.2, alt=500.0)
        center = loc.center_geodetic()

        assert isinstance(center, np.ndarray)
        assert center.shape == (3,)
        assert center[0] == 15.4  # lon
        assert center[1] == 78.2  # lat
        assert center[2] == 500.0  # alt

    def test_center_ecef(self):
        """Test center_ecef() method."""
        # At equator, longitude 0, ECEF should be approximately [R_EARTH, 0, 0]
        loc = bh.PointLocation(lon=0.0, lat=0.0, alt=0.0)
        ecef = loc.center_ecef()

        assert isinstance(ecef, np.ndarray)
        assert ecef.shape == (3,)
        assert ecef[0] == pytest.approx(6378137.0, abs=1.0)
        assert ecef[1] == pytest.approx(0.0, abs=1.0)
        assert ecef[2] == pytest.approx(0.0, abs=1.0)


class TestPointLocationGeoJSON:
    """Test GeoJSON serialization/deserialization."""

    def test_to_geojson_basic(self):
        """Test basic GeoJSON export."""
        loc = bh.PointLocation(lon=15.4, lat=78.2, alt=100.0)
        geojson = loc.to_geojson()

        # Validate top-level structure
        assert geojson["type"] == "Feature"

        # Validate geometry
        geometry = geojson["geometry"]
        assert geometry["type"] == "Point"
        coords = geometry["coordinates"]
        assert len(coords) == 3
        assert coords[0] == 15.4
        assert coords[1] == 78.2
        assert coords[2] == 100.0

    def test_to_geojson_with_metadata(self):
        """Test GeoJSON export with all metadata."""
        loc = (
            bh.PointLocation(lon=15.4, lat=78.2, alt=100.0)
            .with_name("TestLocation")
            .with_id(42)
            .with_new_uuid()
            .add_property("custom_prop", "custom_value")
        )

        geojson = loc.to_geojson()

        # Validate properties
        properties = geojson["properties"]
        assert properties["name"] == "TestLocation"
        assert properties["id"] == 42
        assert "uuid" in properties
        assert properties["custom_prop"] == "custom_value"

    def test_from_geojson_minimal(self):
        """Test creating from minimal GeoJSON."""
        geojson = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [15.4, 78.2]},
            "properties": {},
        }

        loc = bh.PointLocation.from_geojson(geojson)
        assert loc.lon == 15.4
        assert loc.lat == 78.2
        assert loc.alt == 0.0  # Default altitude

    def test_from_geojson_with_altitude(self):
        """Test creating from GeoJSON with altitude."""
        geojson = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [15.4, 78.2, 500.0]},
            "properties": {},
        }

        loc = bh.PointLocation.from_geojson(geojson)
        assert loc.alt == 500.0

    def test_from_geojson_with_properties(self):
        """Test creating from GeoJSON with properties."""
        geojson = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [15.4, 78.2, 0.0]},
            "properties": {"name": "Svalbard", "country": "Norway"},
        }

        loc = bh.PointLocation.from_geojson(geojson)
        assert loc.get_name() == "Svalbard"
        assert loc.properties["country"] == "Norway"

    def test_geojson_roundtrip(self):
        """Test GeoJSON round-trip preserves data."""
        original = (
            bh.PointLocation(lon=15.4, lat=78.2, alt=100.0)
            .with_name("Svalbard")
            .add_property("country", "Norway")
        )

        geojson = original.to_geojson()
        reconstructed = bh.PointLocation.from_geojson(geojson)

        assert reconstructed.lon == 15.4
        assert reconstructed.lat == 78.2
        assert reconstructed.alt == 100.0
        assert reconstructed.get_name() == "Svalbard"
        assert reconstructed.properties["country"] == "Norway"

    def test_from_geojson_invalid_type(self):
        """Test error handling for invalid GeoJSON type."""
        geojson = {"type": "FeatureCollection", "features": []}

        with pytest.raises(ValueError):
            bh.PointLocation.from_geojson(geojson)

    def test_from_geojson_wrong_geometry(self):
        """Test error handling for wrong geometry type."""
        geojson = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]],
            },
            "properties": {},
        }

        with pytest.raises(ValueError):
            bh.PointLocation.from_geojson(geojson)


class TestPointLocationIdentifiable:
    """Test all Identifiable trait methods."""

    def test_with_name(self):
        """Test with_name builder."""
        loc = bh.PointLocation(lon=0.0, lat=0.0, alt=0.0).with_name("Test")
        assert loc.get_name() == "Test"

    def test_with_id(self):
        """Test with_id builder."""
        loc = bh.PointLocation(lon=0.0, lat=0.0, alt=0.0).with_id(123)
        assert loc.get_id() == 123

    def test_with_uuid(self):
        """Test with_uuid builder."""
        uuid_str = "550e8400-e29b-41d4-a716-446655440000"
        loc = bh.PointLocation(lon=0.0, lat=0.0, alt=0.0).with_uuid(uuid_str)
        assert loc.get_uuid() == uuid_str

    def test_with_new_uuid(self):
        """Test with_new_uuid builder."""
        loc = bh.PointLocation(lon=0.0, lat=0.0, alt=0.0).with_new_uuid()
        assert loc.get_uuid() is not None

    def test_set_name(self):
        """Test set_name mutator."""
        loc = bh.PointLocation(lon=0.0, lat=0.0, alt=0.0)
        loc.set_name("NewName")
        assert loc.get_name() == "NewName"

    def test_set_id(self):
        """Test set_id mutator."""
        loc = bh.PointLocation(lon=0.0, lat=0.0, alt=0.0)
        loc.set_id(999)
        assert loc.get_id() == 999

    def test_generate_uuid(self):
        """Test generate_uuid mutator."""
        loc = bh.PointLocation(lon=0.0, lat=0.0, alt=0.0)
        loc.generate_uuid()
        assert loc.get_uuid() is not None


class TestPointLocationString:
    """Test string representations."""

    def test_str_without_name(self):
        """Test __str__ without name."""
        loc = bh.PointLocation(lon=15.4, lat=78.2, alt=100.0)
        s = str(loc)
        assert "PointLocation" in s
        assert "15.4" in s
        assert "78.2" in s
        assert "100.0" in s

    def test_str_with_name(self):
        """Test __str__ with name."""
        loc = bh.PointLocation(lon=15.4, lat=78.2, alt=100.0).with_name("Svalbard")
        s = str(loc)
        assert "Svalbard" in s
        assert "15.4" in s


# ========================================================================
# PolygonLocation Tests
# ========================================================================


class TestPolygonLocationBasic:
    """Basic PolygonLocation functionality tests."""

    def test_polygon_location_new(self):
        """Test creating a new polygon location."""
        vertices = [
            [10.0, 50.0, 0.0],
            [11.0, 50.0, 0.0],
            [11.0, 51.0, 0.0],
            [10.0, 51.0, 0.0],
            [10.0, 50.0, 0.0],
        ]

        poly = bh.PolygonLocation(vertices)

        assert poly.num_vertices == 4
        verts = poly.vertices
        assert verts.shape == (5, 3)  # Including closure

        # Check centroid
        center = poly.center_geodetic()
        assert center[0] == pytest.approx(10.5, abs=0.01)  # lon
        assert center[1] == pytest.approx(50.5, abs=0.01)  # lat

    def test_polygon_location_auto_close(self):
        """Test automatic polygon closure."""
        # Missing closure vertex
        vertices = [
            [10.0, 50.0, 0.0],
            [11.0, 50.0, 0.0],
            [11.0, 51.0, 0.0],
            [10.0, 51.0, 0.0],
        ]

        poly = bh.PolygonLocation(vertices)

        # Should auto-close
        verts = poly.vertices
        assert verts.shape[0] == 5
        assert np.allclose(verts[0], verts[4])

    def test_polygon_location_validation_too_few_vertices(self):
        """Test validation error for too few vertices."""
        vertices = [[10.0, 50.0, 0.0], [11.0, 50.0, 0.0]]

        with pytest.raises(ValueError):
            bh.PolygonLocation(vertices)

    def test_polygon_location_validation_wrong_dimension(self):
        """Test validation error for wrong vertex dimension."""
        vertices = [[10.0, 50.0], [11.0, 50.0], [11.0, 51.0], [10.0, 51.0]]

        with pytest.raises(ValueError):
            bh.PolygonLocation(vertices)


class TestPolygonLocationCoordinates:
    """Test polygon coordinate accessor methods."""

    def test_coordinate_accessors_quick(self):
        """Test quick coordinate accessors for center."""
        vertices = [
            [10.0, 50.0, 100.0],
            [11.0, 50.0, 100.0],
            [11.0, 51.0, 100.0],
            [10.0, 51.0, 100.0],
            [10.0, 50.0, 100.0],
        ]

        poly = bh.PolygonLocation(vertices)

        # Center should be centroid
        assert poly.lon == pytest.approx(10.5, abs=0.01)
        assert poly.lat == pytest.approx(50.5, abs=0.01)
        assert poly.alt == pytest.approx(100.0, abs=0.01)
        assert poly.altitude() == pytest.approx(100.0, abs=0.01)

    def test_coordinate_accessors_format_aware(self):
        """Test format-aware coordinate accessors for center."""
        vertices = [
            [10.0, 50.0, 100.0],
            [11.0, 50.0, 100.0],
            [11.0, 51.0, 100.0],
            [10.0, 51.0, 100.0],
            [10.0, 50.0, 100.0],
        ]

        poly = bh.PolygonLocation(vertices)

        # Format-aware accessors - degrees
        assert poly.longitude(bh.AngleFormat.DEGREES) == pytest.approx(10.5, abs=0.01)
        assert poly.latitude(bh.AngleFormat.DEGREES) == pytest.approx(50.5, abs=0.01)

        # Format-aware accessors - radians
        lon_rad = poly.longitude(bh.AngleFormat.RADIANS)
        lat_rad = poly.latitude(bh.AngleFormat.RADIANS)
        assert lon_rad == pytest.approx(np.deg2rad(10.5), abs=0.01)
        assert lat_rad == pytest.approx(np.deg2rad(50.5), abs=0.01)

    def test_vertices_accessor(self):
        """Test vertices() accessor."""
        vertices = [
            [10.0, 50.0, 0.0],
            [11.0, 50.0, 0.0],
            [11.0, 51.0, 0.0],
            [10.0, 51.0, 0.0],
            [10.0, 50.0, 0.0],
        ]

        poly = bh.PolygonLocation(vertices)

        # Test vertices() getter
        verts = poly.vertices
        assert verts.shape == (5, 3)
        assert np.allclose(verts[0], [10.0, 50.0, 0.0])
        assert np.allclose(verts[4], [10.0, 50.0, 0.0])

    def test_num_vertices(self):
        """Test num_vertices() - should exclude closure."""
        vertices = [
            [10.0, 50.0, 0.0],
            [11.0, 50.0, 0.0],
            [11.0, 51.0, 0.0],
            [10.0, 51.0, 0.0],
            [10.0, 50.0, 0.0],
        ]

        poly = bh.PolygonLocation(vertices)

        # Should exclude the closure vertex
        assert poly.num_vertices == 4

    def test_center_geodetic(self):
        """Test center_geodetic() method."""
        vertices = [
            [10.0, 50.0, 100.0],
            [11.0, 50.0, 100.0],
            [11.0, 51.0, 100.0],
            [10.0, 51.0, 100.0],
            [10.0, 50.0, 100.0],
        ]

        poly = bh.PolygonLocation(vertices)
        center = poly.center_geodetic()

        assert isinstance(center, np.ndarray)
        assert center.shape == (3,)
        assert center[0] == pytest.approx(10.5, abs=0.01)
        assert center[1] == pytest.approx(50.5, abs=0.01)
        assert center[2] == pytest.approx(100.0, abs=0.01)

    def test_center_ecef(self):
        """Test center_ecef() method."""
        vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]

        poly = bh.PolygonLocation(vertices)
        ecef = poly.center_ecef()

        assert isinstance(ecef, np.ndarray)
        assert ecef.shape == (3,)
        # Center should be near equator, should have valid ECEF coordinates
        assert np.linalg.norm(ecef) > 6000000.0  # Should be roughly Earth radius


class TestPolygonLocationGeoJSON:
    """Test polygon GeoJSON serialization/deserialization."""

    def test_to_geojson_basic(self):
        """Test basic GeoJSON export."""
        vertices = [
            [10.0, 50.0, 0.0],
            [11.0, 50.0, 0.0],
            [11.0, 51.0, 0.0],
            [10.0, 51.0, 0.0],
            [10.0, 50.0, 0.0],
        ]

        poly = bh.PolygonLocation(vertices)
        geojson = poly.to_geojson()

        # Validate top-level structure
        assert geojson["type"] == "Feature"

        # Validate geometry
        geometry = geojson["geometry"]
        assert geometry["type"] == "Polygon"

        coords = geometry["coordinates"]
        assert len(coords) == 1  # One outer ring

        outer_ring = coords[0]
        assert len(outer_ring) == 5  # 4 unique + closure

        # Check first vertex
        assert outer_ring[0][0] == 10.0
        assert outer_ring[0][1] == 50.0
        assert outer_ring[0][2] == 0.0

    def test_to_geojson_with_metadata(self):
        """Test GeoJSON export with metadata."""
        vertices = [
            [10.0, 50.0, 0.0],
            [11.0, 50.0, 0.0],
            [11.0, 51.0, 0.0],
            [10.0, 51.0, 0.0],
            [10.0, 50.0, 0.0],
        ]

        poly = (
            bh.PolygonLocation(vertices)
            .with_name("TestPolygon")
            .with_id(99)
            .with_new_uuid()
            .add_property("custom_prop", "custom_value")
        )

        geojson = poly.to_geojson()

        # Validate properties
        properties = geojson["properties"]
        assert properties["name"] == "TestPolygon"
        assert properties["id"] == 99
        assert "uuid" in properties
        assert properties["custom_prop"] == "custom_value"

    def test_from_geojson_minimal(self):
        """Test creating from minimal GeoJSON."""
        geojson = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [10.0, 50.0],
                        [11.0, 50.0],
                        [11.0, 51.0],
                        [10.0, 51.0],
                        [10.0, 50.0],
                    ]
                ],
            },
            "properties": {},
        }

        poly = bh.PolygonLocation.from_geojson(geojson)
        assert poly.num_vertices == 4

    def test_from_geojson_with_altitude(self):
        """Test creating from GeoJSON with altitude."""
        geojson = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [10.0, 50.0, 100.0],
                        [11.0, 50.0, 100.0],
                        [11.0, 51.0, 100.0],
                        [10.0, 51.0, 100.0],
                        [10.0, 50.0, 100.0],
                    ]
                ],
            },
            "properties": {},
        }

        poly = bh.PolygonLocation.from_geojson(geojson)
        assert poly.alt == pytest.approx(100.0, abs=0.01)

    def test_geojson_roundtrip(self):
        """Test GeoJSON round-trip preserves data."""
        vertices = [
            [10.0, 50.0, 0.0],
            [11.0, 50.0, 0.0],
            [11.0, 51.0, 0.0],
            [10.0, 51.0, 0.0],
            [10.0, 50.0, 0.0],
        ]

        original = (
            bh.PolygonLocation(vertices)
            .with_name("AOI-1")
            .add_property("region", "Europe")
        )

        geojson = original.to_geojson()
        reconstructed = bh.PolygonLocation.from_geojson(geojson)

        assert reconstructed.num_vertices == 4
        assert reconstructed.get_name() == "AOI-1"
        assert reconstructed.properties["region"] == "Europe"

    def test_from_geojson_invalid_type(self):
        """Test error handling for invalid GeoJSON type."""
        geojson = {"type": "FeatureCollection", "features": []}

        with pytest.raises(ValueError):
            bh.PolygonLocation.from_geojson(geojson)

    def test_from_geojson_wrong_geometry(self):
        """Test error handling for wrong geometry type."""
        geojson = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
            "properties": {},
        }

        with pytest.raises(ValueError):
            bh.PolygonLocation.from_geojson(geojson)


class TestPolygonLocationIdentifiable:
    """Test all Identifiable trait methods for polygons."""

    def test_with_name(self):
        """Test with_name builder."""
        vertices = [
            [10.0, 50.0, 0.0],
            [11.0, 50.0, 0.0],
            [11.0, 51.0, 0.0],
            [10.0, 51.0, 0.0],
            [10.0, 50.0, 0.0],
        ]

        poly = bh.PolygonLocation(vertices).with_name("Test")
        assert poly.get_name() == "Test"

    def test_with_id(self):
        """Test with_id builder."""
        vertices = [
            [10.0, 50.0, 0.0],
            [11.0, 50.0, 0.0],
            [11.0, 51.0, 0.0],
            [10.0, 51.0, 0.0],
            [10.0, 50.0, 0.0],
        ]

        poly = bh.PolygonLocation(vertices).with_id(123)
        assert poly.get_id() == 123

    def test_with_uuid(self):
        """Test with_uuid builder."""
        vertices = [
            [10.0, 50.0, 0.0],
            [11.0, 50.0, 0.0],
            [11.0, 51.0, 0.0],
            [10.0, 51.0, 0.0],
            [10.0, 50.0, 0.0],
        ]

        uuid_str = "550e8400-e29b-41d4-a716-446655440000"
        poly = bh.PolygonLocation(vertices).with_uuid(uuid_str)
        assert poly.get_uuid() == uuid_str

    def test_with_new_uuid(self):
        """Test with_new_uuid builder."""
        vertices = [
            [10.0, 50.0, 0.0],
            [11.0, 50.0, 0.0],
            [11.0, 51.0, 0.0],
            [10.0, 51.0, 0.0],
            [10.0, 50.0, 0.0],
        ]

        poly = bh.PolygonLocation(vertices).with_new_uuid()
        assert poly.get_uuid() is not None

    def test_set_name(self):
        """Test set_name mutator."""
        vertices = [
            [10.0, 50.0, 0.0],
            [11.0, 50.0, 0.0],
            [11.0, 51.0, 0.0],
            [10.0, 51.0, 0.0],
            [10.0, 50.0, 0.0],
        ]

        poly = bh.PolygonLocation(vertices)
        poly.set_name("NewName")
        assert poly.get_name() == "NewName"

    def test_set_id(self):
        """Test set_id mutator."""
        vertices = [
            [10.0, 50.0, 0.0],
            [11.0, 50.0, 0.0],
            [11.0, 51.0, 0.0],
            [10.0, 51.0, 0.0],
            [10.0, 50.0, 0.0],
        ]

        poly = bh.PolygonLocation(vertices)
        poly.set_id(999)
        assert poly.get_id() == 999

    def test_generate_uuid(self):
        """Test generate_uuid mutator."""
        vertices = [
            [10.0, 50.0, 0.0],
            [11.0, 50.0, 0.0],
            [11.0, 51.0, 0.0],
            [10.0, 51.0, 0.0],
            [10.0, 50.0, 0.0],
        ]

        poly = bh.PolygonLocation(vertices)
        poly.generate_uuid()
        assert poly.get_uuid() is not None


class TestPolygonLocationProperties:
    """Test polygon custom properties."""

    def test_add_property_builder(self):
        """Test add_property builder pattern."""
        vertices = [
            [10.0, 50.0, 0.0],
            [11.0, 50.0, 0.0],
            [11.0, 51.0, 0.0],
            [10.0, 51.0, 0.0],
            [10.0, 50.0, 0.0],
        ]

        poly = (
            bh.PolygonLocation(vertices)
            .add_property("key1", "value1")
            .add_property("key2", 42)
            .add_property("key3", True)
        )

        props = poly.properties
        assert props["key1"] == "value1"
        assert props["key2"] == 42
        assert props["key3"] is True


class TestPolygonLocationString:
    """Test polygon string representations."""

    def test_str_without_name(self):
        """Test __str__ without name."""
        vertices = [
            [10.0, 50.0, 0.0],
            [11.0, 50.0, 0.0],
            [11.0, 51.0, 0.0],
            [10.0, 51.0, 0.0],
            [10.0, 50.0, 0.0],
        ]

        poly = bh.PolygonLocation(vertices)
        s = str(poly)
        assert "PolygonLocation" in s
        assert "4 vertices" in s
        assert "10.5" in s  # center lon
        assert "50.5" in s  # center lat

    def test_str_with_name(self):
        """Test __str__ with name."""
        vertices = [
            [10.0, 50.0, 0.0],
            [11.0, 50.0, 0.0],
            [11.0, 51.0, 0.0],
            [10.0, 51.0, 0.0],
            [10.0, 50.0, 0.0],
        ]

        poly = bh.PolygonLocation(vertices).with_name("AOI-1")
        s = str(poly)
        assert "AOI-1" in s
        assert "4 vertices" in s
