# /// script
# dependencies = ["brahe"]
# ///
# FLAGS = [IGNORE]
"""
Get ephemeris data from CelesTrak.

Demonstrates:
- Getting GNSS satellite TLE data
"""

import brahe as bh

if __name__ == "__main__":
    print("Get GNSS Ephemeris from CelesTrak")
    print("=" * 60)

    # Get ephemeris data for GNSS constellation
    ephemeris = bh.datasets.celestrak.get_ephemeris("gnss")
    print(f"\nRetrieved {len(ephemeris)} GNSS satellites")

    # Display first few satellites
    for i, (name, line1, line2) in enumerate(ephemeris[:3]):
        print(f"\nSatellite {i + 1}: {name}")
        print(f"  {line1}")
        print(f"  {line2}")

    print("\n" + "=" * 60)
