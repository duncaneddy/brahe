# /// script
# dependencies = ["brahe"]
# ///
# FLAGS = [IGNORE]
"""
Get CelesTrak ephemeris as propagators.

Demonstrates:
- Getting active satellites as propagators
"""

import brahe as bh

if __name__ == "__main__":
    print("Get Propagators from CelesTrak")
    print("=" * 60)

    # Get ephemeris as propagators for active satellites
    propagators = bh.datasets.celestrak.get_ephemeris_as_propagators(
        "active", step_size=60.0
    )
    print(f"\nCreated {len(propagators)} propagators for active satellites")

    # Propagate first satellite
    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0, tsys="UTC")
    if propagators:
        state = propagators[0].state(epoch)
        print(f"\nFirst satellite state at {epoch}:")
        print(
            f"  Position: [{state[0] / 1000:.3f}, {state[1] / 1000:.3f}, {state[2] / 1000:.3f}] km"
        )
        print(
            f"  Velocity: [{state[3] / 1000:.3f}, {state[4] / 1000:.3f}, {state[5] / 1000:.3f}] km/s"
        )

    print("\n" + "=" * 60)
