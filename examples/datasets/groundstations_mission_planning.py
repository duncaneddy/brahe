# /// script
# dependencies = ["brahe"]
# ///
"""
Mission planning with groundstation networks.

Demonstrates:
- Evaluating multiple providers
- Filtering by capability requirements
- Assessing geographic distribution
"""

import brahe as bh

if __name__ == "__main__":
    print("Groundstation Mission Planning")
    print("=" * 60)

    # Requirements
    required_bands = ["S", "X"]
    print("\nMission requirements:")
    print(f"  Required bands: {', '.join(required_bands)}")
    print("  Preferred: Arctic coverage")

    # Evaluate providers
    providers = bh.datasets.groundstations.list_providers()
    print(f"\n{'-' * 60}")
    print("Provider Evaluation:")
    print(f"{'-' * 60}")

    for provider in providers:
        stations = bh.datasets.groundstations.load(provider)

        # Filter by capability
        capable = [
            s
            for s in stations
            if all(band in s.properties["frequency_bands"] for band in required_bands)
        ]

        # Check geographic distribution
        arctic_count = len([s for s in capable if s.lat > 60])

        print(f"\n{provider.upper()}")
        print(f"  Total stations: {len(stations)}")
        print(f"  Capable stations: {len(capable)}")
        print(f"  Arctic coverage (>60°N): {arctic_count}")

    print("\n" + "=" * 60)
    print("Mission planning analysis complete!")
