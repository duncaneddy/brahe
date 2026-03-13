# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Access tile metadata properties after tessellation.
"""

import numpy as np
import brahe as bh

bh.initialize_eop()

# ISS TLE
line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

# Create propagator and tessellator
prop = bh.SGPPropagator.from_tle(line1, line2, step_size=60.0)
config = bh.OrbitGeometryTessellatorConfig(
    image_width=5000,
    image_length=5000,
    asc_dsc=bh.AscDsc.ASCENDING,
)
tess = bh.OrbitGeometryTessellator(prop, prop.epoch, config, spacecraft_id="ISS")

# Tessellate a point and inspect properties
point = bh.PointLocation(10.0, 30.0, 0.0)
tiles = tess.tessellate_point(point)
tile = tiles[0]
props = tile.properties

# Along-track direction (unit vector in ECEF)
direction = np.array(props["tile_direction"])
print(f"tile_direction: [{direction[0]:.4f}, {direction[1]:.4f}, {direction[2]:.4f}]")
print(f"  magnitude: {np.linalg.norm(direction):.6f}")

# Tile dimensions
print(f"tile_width: {props['tile_width']:.0f} m")
print(f"tile_length: {props['tile_length']:.0f} m")
print(f"tile_area: {props['tile_area']:.0f} m^2")

# Group and spacecraft identifiers
print(f"tile_group_id: {props['tile_group_id'][:8]}...")
print(f"spacecraft_ids: {props['spacecraft_ids']}")
