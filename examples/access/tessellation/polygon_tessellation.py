# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Tessellate a polygon location into rectangular tiles aligned with satellite ground tracks.
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

# Define a small polygon (approximately 0.1 deg x 0.1 deg)
vertices = np.array(
    [
        [10.0, 30.0, 0.0],
        [10.1, 30.0, 0.0],
        [10.1, 30.1, 0.0],
        [10.0, 30.1, 0.0],
    ]
)
polygon = bh.PolygonLocation(vertices)

# Tessellate the polygon
tiles = tess.tessellate_polygon(polygon)

print(f"Number of tiles: {len(tiles)}")
for i, tile in enumerate(tiles):
    props = tile.properties
    print(
        f"Tile {i}: group_id={props['tile_group_id'][:8]}... "
        f"length={props['tile_length']:.0f} m"
    )
