# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Propagate an orbit using SGP4 from TLE data.

Demonstrates:
- Initializing EOP provider (required for frame transformations)
- Creating an SGP4 propagator from TLE
- Propagating to a specific epoch
- Working with state vectors
"""

import brahe as bh

if __name__ == "__main__":
    # Initialize EOP provider (required for frame transformations)
    eop = bh.StaticEOPProvider.from_zero()
    bh.set_global_eop_provider_from_static_provider(eop)

    # Create an SGP4 propagator from Two-Line Element (TLE) data
    # ISS TLE from January 1, 2021
    line1 = "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997"
    line2 = "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003"
    prop = bh.SGPPropagator.from_tle(line1, line2, step_size=60.0)

    # Propagate to a specific epoch
    epc = bh.Epoch(2021, 1, 2, 0, 0, 0.0, 0.0, time_system=bh.TimeSystem.UTC)
    state = prop.state(epc)  # Returns [x, y, z, vx, vy, vz] in meters and m/s

    print(f"Position: {state[:3] / 1000} km")
    print(f"Velocity: {state[3:] / 1000} km/s")
