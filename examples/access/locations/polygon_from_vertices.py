# /// script
# dependencies = ["brahe"]
# ///
"""
Create a PolygonLocation from a list of vertices.
Demonstrates polygon construction and center/vertex access.
"""

import brahe as bh

bh.initialize_eop()

# Define polygon vertices (longitude, latitude, altitude)
# Simple rectangular region
vertices = [
    [-122.5, 37.7, 0.0],
    [-122.35, 37.7, 0.0],
    [-122.35, 37.8, 0.0],
    [-122.5, 37.8, 0.0],
    [-122.5, 37.7, 0.0],  # Close the polygon
]

polygon = bh.PolygonLocation(vertices).with_name("SF Region")

print(f"Name: {polygon.get_name()}")
print(f"Vertices: {polygon.num_vertices}")
print(
    f"Center: ({polygon.longitude(bh.AngleFormat.DEGREES):.4f}, {polygon.latitude(bh.AngleFormat.DEGREES):.4f})"
)

# Expected output:
# Name: SF Region
# Vertices: 4
# Center: (-122.4250, 37.7500)
