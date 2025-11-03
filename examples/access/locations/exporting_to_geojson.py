# /// script
# dependencies = ["brahe"]
# ///
"""
Convert locations to GeoJSON format.
Demonstrates roundtrip export with names and IDs.
"""

import brahe as bh

bh.initialize_eop()

location = (
    bh.PointLocation(-122.4194, 37.7749, 0.0).with_name("San Francisco").with_id(1)
)

# Export to GeoJSON dict
geojson = location.to_geojson()
print("Exported GeoJSON:")
print(geojson)

# The output includes all properties and identifiers
# Can be loaded back with from_geojson()
reloaded = bh.PointLocation.from_geojson(geojson)
print(f"\nReloaded: {reloaded.get_name()} (ID: {reloaded.get_id()})")

# Expected output:
# Exported GeoJSON:
# {'geometry': {'coordinates': [-122.4194, 37.7749, 0.0], 'type': 'Point'}, 'properties': {'id': 1, 'name': 'San Francisco'}, 'type': 'Feature'}
#
# Reloaded: San Francisco (ID: 1)
