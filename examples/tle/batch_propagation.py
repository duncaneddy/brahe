# /// script
# dependencies = ["brahe"]
# ///
# FLAGS = [IGNORE]
"""
Batch propagation of multiple satellites.

Demonstrates:
- Getting ephemeris as propagators directly
- Propagating multiple satellites to same epoch
- Working with propagator collections
"""

import brahe as bh

if __name__ == "__main__":
    # Initialize EOP (required for propagators)
    eop = bh.StaticEOPProvider.from_zero()
    bh.set_global_eop_provider_from_static_provider(eop)

    # Get ephemeris and initialize propagators directly
    propagators = bh.datasets.celestrak.get_ephemeris_as_propagators(
        "gnss", step_size=60.0
    )

    print(f"Loaded {len(propagators)} GNSS propagators")

    # Propagate all satellites to same epoch
    epoch = bh.Epoch.from_datetime(2021, 1, 2, 12, 0, 0, 0, bh.TimeSystem.UTC)
    states = [prop.state(epoch) for prop in propagators]

    print(f"Propagated {len(states)} satellites to {epoch}")
    print(f"First satellite position: {states[0][:3] / 1000} km")
