# /// script
# dependencies = ["brahe"]
# ///
"""
Load a PolygonLocation from a GeoJSON string.
Demonstrates GeoJSON Polygon format with nested coordinate arrays.
"""

import brahe as bh
import json

bh.initialize_eop()

# GeoJSON Polygon feature
geojson_str = """
{
    "type": "Feature",
    "properties": {"name": "Target Area"},
    "geometry": {
        "type": "Polygon",
        "coordinates": [[
            [-122.5, 37.7, 0],
            [-122.35, 37.7, 0],
            [-122.35, 37.8, 0],
            [-122.5, 37.8, 0],
            [-122.5, 37.7, 0]
        ]]
    }
}
"""

polygon = bh.PolygonLocation.from_geojson(json.loads(geojson_str))

print(f"Name: {polygon.get_name()}")
print(f"Vertices: {polygon.num_vertices}")
print(
    f"Center: ({polygon.longitude(bh.AngleFormat.DEGREES):.4f}, {polygon.latitude(bh.AngleFormat.DEGREES):.4f})"
)

# Expected output:
# Name: Target Area
# Vertices: 5
# Center: (-122.4250, 37.7500)
