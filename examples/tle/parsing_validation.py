# /// script
# dependencies = ["brahe"]
# ///
"""
Parse and validate TLE format.

Demonstrates:
- Validating TLE line format
- Extracting epoch from TLE
- Converting TLE to Keplerian elements
"""

import brahe as bh

if __name__ == "__main__":
    # Initialize EOP (required for TLE operations)
    eop = bh.StaticEOPProvider.from_zero()
    bh.set_global_eop_provider_from_static_provider(eop)

    # Valid ISS TLE
    line1 = "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997"
    line2 = "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003"

    # Validate TLE format
    is_valid = bh.validate_tle_lines(line1, line2)
    print(f"TLE valid: {is_valid}")

    # Extract epoch and convert to Keplerian elements
    epoch, keplerian = bh.keplerian_elements_from_tle(line1, line2)
    print(f"Epoch: {epoch}")
    print("Keplerian elements [a, e, i, raan, argp, M]:")
    print(f"  a (m): {keplerian[0]:.3f}")
    print(f"  e: {keplerian[1]:.6f}")
    print(f"  i (rad): {keplerian[2]:.6f}")
