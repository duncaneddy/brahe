# /// script
# dependencies = ["brahe"]
# ///
"""
Load a PointLocation from a GeoJSON string.
Demonstrates GeoJSON Feature format with properties.
"""

import brahe as bh
import json

bh.initialize_eop()

# GeoJSON Point feature
geojson_str = """
{
    "type": "Feature",
    "properties": {"name": "Svalbard Station"},
    "geometry": {
        "type": "Point",
        "coordinates": [15.4038, 78.2232, 458.0]
    }
}
"""

location = bh.PointLocation.from_geojson(json.loads(geojson_str))
print(f"Loaded: {location.get_name()}")
print(f"Longitude: {location.longitude(bh.AngleFormat.DEGREES):.4f} deg")
print(f"Latitude: {location.latitude(bh.AngleFormat.DEGREES):.4f} deg")
print(f"Altitude: {location.altitude():.1f} m")

# Expected output:
# Loaded: Svalbard Station
# Longitude: 15.4038 deg
# Latitude: 78.2232 deg
# Altitude: 458.0 m
