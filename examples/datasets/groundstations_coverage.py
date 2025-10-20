# /// script
# dependencies = ["brahe"]
# ///
"""
Network coverage analysis for groundstation datasets.

Demonstrates:
- Loading all providers
- Analyzing by latitude band
- Finding stations by capability
"""

import brahe as bh

if __name__ == "__main__":
    print("Groundstation Network Coverage Analysis")
    print("=" * 60)

    # Load all providers
    all_stations = bh.datasets.groundstations.load_all()
    print(f"\nTotal stations across all providers: {len(all_stations)}")

    # Analyze by latitude band
    arctic = [s for s in all_stations if s.lat > 66.5]
    temperate = [s for s in all_stations if -66.5 <= s.lat <= 66.5]
    antarctic = [s for s in all_stations if s.lat < -66.5]

    print("\nLatitude distribution:")
    print(f"  Arctic stations (>66.5°N): {len(arctic)}")
    print(f"  Temperate stations: {len(temperate)}")
    print(f"  Antarctic stations (<66.5°S): {len(antarctic)}")

    # Find stations by capability
    x_band_stations = [
        s for s in all_stations if "X" in s.properties["frequency_bands"]
    ]
    print("\nCapability analysis:")
    print(f"  X-band capable stations: {len(x_band_stations)}")

    print("\n" + "=" * 60)
    print("Coverage analysis complete!")
