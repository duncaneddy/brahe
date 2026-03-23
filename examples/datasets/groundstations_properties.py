# /// script
# dependencies = ["brahe"]
# ///
"""
Access groundstation location properties.

This example demonstrates how to access geographic coordinates and
metadata properties from groundstation locations.
"""

import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Load KSAT groundstations
stations = bh.datasets.groundstations.load("ksat")

# Access the first station
station = stations[0]

# Geographic coordinates (degrees and meters)
name = station.get_name() if station.get_name() else "Unknown"
print(f"Station: {name}")
print(f"Latitude: {station.lat:.4f}°")
print(f"Longitude: {station.lon:.4f}°")
print(f"Altitude: {station.alt:.1f} m")

# Access metadata properties
props = station.properties
print(f"\nProvider: {props['provider']}")
print(f"Frequency bands: {', '.join(props['frequency_bands'])}")

# Show all stations with their locations
print(f"\n{len(stations)} KSAT Stations:")
for i, gs in enumerate(stations, 1):
    gs_name = gs.get_name() if gs.get_name() else "Unknown"
    print(f"{i:2d}. {gs_name:30s} ({gs.lat:7.3f}°, {gs.lon:8.3f}°)")
