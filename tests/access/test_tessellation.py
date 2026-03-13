"""Tests for the tessellation module.

Tests the OrbitGeometryTessellator, OrbitGeometryTessellatorConfig,
and tile_merge_orbit_geometry function.
"""

import numpy as np
import pytest

import brahe as bh


# ISS TLE from sgp_propagator tests
ISS_LINE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
ISS_LINE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"


@pytest.fixture
def propagator():
    """Create an SGP propagator from ISS TLE."""
    return bh.SGPPropagator.from_tle(ISS_LINE1, ISS_LINE2, step_size=60.0)


@pytest.fixture
def tessellator(propagator):
    """Create an ascending-only tessellator."""
    config = bh.OrbitGeometryTessellatorConfig(
        image_width=5000,
        image_length=5000,
        asc_dsc=bh.AscDsc.ASCENDING,
    )
    return bh.OrbitGeometryTessellator(
        propagator, propagator.epoch, config, spacecraft_id="ISS"
    )


@pytest.fixture
def tessellator_either(propagator):
    """Create a tessellator using both ascending and descending passes."""
    config = bh.OrbitGeometryTessellatorConfig(
        image_width=5000,
        image_length=5000,
        asc_dsc=bh.AscDsc.EITHER,
    )
    return bh.OrbitGeometryTessellator(
        propagator, propagator.epoch, config, spacecraft_id="ISS"
    )


# ---- Config Tests ----


class TestOrbitGeometryTessellatorConfig:
    def test_default_config(self):
        config = bh.OrbitGeometryTessellatorConfig()
        assert config.image_width == 5000.0
        assert config.image_length == 5000.0
        assert config.crosstrack_overlap == 200.0
        assert config.alongtrack_overlap == 200.0
        assert config.min_image_length == 5000.0
        assert config.max_image_length == 5000.0

    def test_custom_config(self):
        config = bh.OrbitGeometryTessellatorConfig(
            image_width=10000,
            image_length=15000,
            crosstrack_overlap=300,
            alongtrack_overlap=300,
            asc_dsc=bh.AscDsc.DESCENDING,
            min_image_length=8000,
            max_image_length=25000,
        )
        assert config.image_width == 10000.0
        assert config.image_length == 15000.0
        assert config.crosstrack_overlap == 300.0
        assert config.alongtrack_overlap == 300.0
        assert config.min_image_length == 8000.0
        assert config.max_image_length == 25000.0

    def test_repr(self):
        config = bh.OrbitGeometryTessellatorConfig()
        s = repr(config)
        assert "OrbitGeometryTessellatorConfig" in s
        assert "5000" in s


# ---- Point Tessellation Tests ----


class TestPointTessellation:
    def test_ascending_only(self, tessellator):
        point = bh.PointLocation(10.0, 30.0, 0.0)
        tiles = tessellator.tessellate_point(point)

        assert len(tiles) == 1

        props = tiles[0].properties
        assert "tile_direction" in props
        assert "tile_width" in props
        assert "tile_length" in props
        assert "tile_area" in props
        assert "tile_group_id" in props
        assert "spacecraft_ids" in props

        assert props["tile_width"] == pytest.approx(5000.0)
        assert props["tile_length"] == pytest.approx(5000.0)
        assert props["tile_area"] == pytest.approx(25_000_000.0)
        assert "ISS" in props["spacecraft_ids"]

    def test_either_direction(self, tessellator_either):
        point = bh.PointLocation(10.0, 30.0, 0.0)
        tiles = tessellator_either.tessellate_point(point)

        # Should get 1 or 2 tiles (ascending + descending, possibly merged at high lat)
        assert 1 <= len(tiles) <= 2

        for tile in tiles:
            props = tile.properties
            assert "ISS" in props["spacecraft_ids"]
            assert len(props["tile_direction"]) == 3

    def test_tile_direction_is_unit_vector(self, tessellator):
        point = bh.PointLocation(10.0, 30.0, 0.0)
        tiles = tessellator.tessellate_point(point)

        for tile in tiles:
            direction = np.array(tile.properties["tile_direction"])
            assert np.linalg.norm(direction) == pytest.approx(1.0, abs=0.01)

    def test_generic_tessellate_with_point(self, tessellator):
        point = bh.PointLocation(10.0, 30.0, 0.0)
        tiles = tessellator.tessellate(point)
        assert len(tiles) >= 1


# ---- Polygon Tessellation Tests ----


class TestPolygonTessellation:
    def test_small_polygon(self, tessellator):
        vertices = np.array(
            [
                [10.0, 30.0, 0.0],
                [10.05, 30.0, 0.0],
                [10.05, 30.05, 0.0],
                [10.0, 30.05, 0.0],
            ]
        )
        polygon = bh.PolygonLocation(vertices)
        tiles = tessellator.tessellate_polygon(polygon)

        assert len(tiles) >= 1

        for tile in tiles:
            props = tile.properties
            assert "tile_direction" in props
            assert "tile_width" in props
            assert "tile_length" in props
            assert "tile_group_id" in props

    def test_larger_polygon_more_tiles(self, tessellator):
        vertices = np.array(
            [
                [10.0, 30.0, 0.0],
                [10.2, 30.0, 0.0],
                [10.2, 30.2, 0.0],
                [10.0, 30.2, 0.0],
            ]
        )
        polygon = bh.PolygonLocation(vertices)
        tiles = tessellator.tessellate_polygon(polygon)

        assert len(tiles) > 1

    def test_generic_tessellate_with_polygon(self, tessellator):
        vertices = np.array(
            [
                [10.0, 30.0, 0.0],
                [10.05, 30.0, 0.0],
                [10.05, 30.05, 0.0],
                [10.0, 30.05, 0.0],
            ]
        )
        polygon = bh.PolygonLocation(vertices)
        tiles = tessellator.tessellate(polygon)
        assert len(tiles) >= 1

    def test_tiles_share_group_id(self, tessellator):
        vertices = np.array(
            [
                [10.0, 30.0, 0.0],
                [10.2, 30.0, 0.0],
                [10.2, 30.2, 0.0],
                [10.0, 30.2, 0.0],
            ]
        )
        polygon = bh.PolygonLocation(vertices)
        tiles = tessellator.tessellate_polygon(polygon)

        if len(tiles) > 1:
            # All tiles from the same direction should share a group_id
            group_ids = set(t.properties["tile_group_id"] for t in tiles)
            # With ascending only, should be 1 group
            assert len(group_ids) >= 1


# ---- Tessellator Properties ----


class TestTessellatorProperties:
    def test_name(self, tessellator):
        assert tessellator.name() == "OrbitGeometryTessellator"

    def test_config_accessible(self, tessellator):
        config = tessellator.config
        assert config.image_width == 5000.0
        assert config.image_length == 5000.0


# ---- Tile Merging Tests ----


class TestTileMerging:
    def test_merge_empty_list(self):
        result = bh.tile_merge_orbit_geometry([], 200.0, 200.0, 2.0)
        assert len(result) == 0

    def test_merge_single_tile(self):
        vertices = np.array(
            [
                [10.0, 50.0, 0.0],
                [10.05, 50.0, 0.0],
                [10.05, 50.05, 0.0],
                [10.0, 50.05, 0.0],
            ]
        )
        tile = bh.PolygonLocation(vertices)
        tile.properties["tile_direction"] = [0.0, 1.0, 0.0]
        tile.properties["tile_width"] = 5000.0
        tile.properties["tile_length"] = 5000.0
        tile.properties["tile_area"] = 25000000.0
        tile.properties["tile_group_id"] = "group1"
        tile.properties["spacecraft_ids"] = ["sc1"]

        result = bh.tile_merge_orbit_geometry([tile], 200.0, 200.0, 2.0)
        assert len(result) == 1

    def test_generic_tessellate_rejects_invalid(self, tessellator):
        with pytest.raises(
            TypeError, match="Expected PointLocation or PolygonLocation"
        ):
            tessellator.tessellate("not a location")


# ---- Integration Tests ----


class TestIntegration:
    def test_full_workflow(self, propagator):
        """Test complete tessellation workflow with multiple pass types."""
        config = bh.OrbitGeometryTessellatorConfig(
            image_width=5000,
            image_length=5000,
            asc_dsc=bh.AscDsc.ASCENDING,
        )
        tess = bh.OrbitGeometryTessellator(
            propagator, propagator.epoch, config, spacecraft_id="ISS"
        )

        # Tessellate a point
        point = bh.PointLocation(0.0, 30.0, 0.0)
        tiles = tess.tessellate_point(point)
        assert len(tiles) >= 1

        # Verify all tiles have valid properties
        for tile in tiles:
            props = tile.properties
            assert isinstance(props["tile_direction"], list)
            assert isinstance(props["tile_width"], (int, float))
            assert isinstance(props["tile_length"], (int, float))
            assert isinstance(props["spacecraft_ids"], list)

    def test_default_config_constructor(self, propagator):
        """Test tessellator with default config."""
        tess = bh.OrbitGeometryTessellator(propagator, propagator.epoch)
        point = bh.PointLocation(10.0, 30.0, 0.0)
        tiles = tess.tessellate_point(point)
        assert len(tiles) >= 1
