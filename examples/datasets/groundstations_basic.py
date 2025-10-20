# /// script
# dependencies = ["brahe"]
# ///
"""
Basic groundstation data access.

Demonstrates:
- Loading groundstation datasets
- Accessing geographic coordinates
- Reading metadata properties
"""

import brahe as bh

if __name__ == "__main__":
    print("Groundstations Basic Usage")
    print("=" * 60)

    # Load a provider's groundstations
    stations = bh.datasets.groundstations.load("ksat")
    print(f"\nLoaded {len(stations)} KSAT groundstations")

    # Access first station
    station = stations[0]

    # Geographic coordinates (WGS84)
    lon = station.lon  # Longitude in degrees
    lat = station.lat  # Latitude in degrees
    alt = station.alt  # Altitude in meters

    print("\nFirst station:")
    print(f"  Longitude: {lon:.6f}°")
    print(f"  Latitude: {lat:.6f}°")
    print(f"  Altitude: {alt:.1f} m")

    # Metadata properties
    props = station.properties
    name = station.get_name()  # Station name
    provider = props["provider"]  # Provider name (e.g., "KSAT")
    bands = props["frequency_bands"]  # Supported bands (e.g., ["S", "X"])

    print(f"  Name: {name}")
    print(f"  Provider: {provider}")
    print(f"  Frequency bands: {', '.join(bands)}")

    print("\n" + "=" * 60)
    print("Groundstation data access complete!")
