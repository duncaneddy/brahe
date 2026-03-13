# /// script
# dependencies = ["brahe"]
# ///
"""
Merge tessellation tiles from multiple spacecraft with similar ground-track directions.
"""

import brahe as bh

bh.initialize_eop()

# SC-1 and SC-2 TLEs with slightly different inclinations (~1.4 degree offset)
line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
line2_sc1 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
line2_sc2 = "2 25544  53.0000 247.4627 0006703 130.5360 325.0288 15.72125391563532"

# Create two tessellators with different spacecraft IDs
config = bh.OrbitGeometryTessellatorConfig(
    image_width=5000,
    image_length=5000,
    asc_dsc=bh.AscDsc.ASCENDING,
)

prop1 = bh.SGPPropagator.from_tle(line1, line2_sc1, step_size=60.0)
tess1 = bh.OrbitGeometryTessellator(prop1, prop1.epoch, config, spacecraft_id="SC-1")

prop2 = bh.SGPPropagator.from_tle(line1, line2_sc2, step_size=60.0)
tess2 = bh.OrbitGeometryTessellator(prop2, prop2.epoch, config, spacecraft_id="SC-2")

# Tessellate the same point with both spacecraft
point = bh.PointLocation(10.0, 30.0, 0.0)
tiles_sc1 = tess1.tessellate_point(point)
tiles_sc2 = tess2.tessellate_point(point)
all_tiles = tiles_sc1 + tiles_sc2

print(f"Before merge: {len(all_tiles)} tiles")

# Merge tiles with similar directions
merged = bh.tile_merge_orbit_geometry(all_tiles, 200.0, 200.0, 2.0)

print(f"After merge: {len(merged)} tiles")
for tile in merged:
    print(f"  spacecraft_ids: {tile.properties['spacecraft_ids']}")
