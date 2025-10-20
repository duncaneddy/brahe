# /// script
# dependencies = ["brahe"]
# ///
"""
Creating custom groundstation data.

Demonstrates:
- Creating custom groundstation locations
- Adding properties to locations
- Combining custom with commercial networks
"""

import brahe as bh

if __name__ == "__main__":
    print("Custom Groundstation Data")
    print("=" * 60)

    # Create custom groundstation
    custom_station = (
        bh.PointLocation(
            lon=-122.4,  # degrees
            lat=37.8,  # degrees
            alt=100.0,  # meters
        )
        .add_property("provider", "Custom")
        .add_property("frequency_bands", ["S", "X", "Ka"])
        .add_property("name", "San Francisco Custom")
    )

    print("\nCustom station created:")
    print(f"  Location: {custom_station.lat:.2f}°N, {custom_station.lon:.2f}°E")
    print(f"  Altitude: {custom_station.alt:.1f} m")
    print(f"  Name: {custom_station.get_name()}")
    print(f"  Bands: {', '.join(custom_station.properties['frequency_bands'])}")

    # Combine with commercial network
    ksat_stations = bh.datasets.groundstations.load("ksat")
    all_stations = [custom_station] + ksat_stations

    print(f"\nTotal network size: {len(all_stations)} stations")
    print(f"  Commercial (KSAT): {len(ksat_stations)}")
    print("  Custom: 1")

    print("\n" + "=" * 60)
    print("Custom groundstation integration complete!")
