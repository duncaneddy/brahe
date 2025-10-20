# /// script
# dependencies = ["brahe"]
# ///
"""
Working with time systems and epochs.

Demonstrates:
- Creating epochs from date/time
- Converting between time systems
- Time arithmetic operations
"""

import brahe as bh

if __name__ == "__main__":
    # Create an epoch from a specific date and time
    epc = bh.Epoch(2024, 1, 1, 12, 0, 0.0, 0.0, time_system=bh.TimeSystem.UTC)

    # Convert between time systems
    mjd_utc = epc.mjd_as_time_system(bh.TimeSystem.UTC)
    mjd_tai = epc.mjd_as_time_system(bh.TimeSystem.TAI)

    print(f"MJD (UTC): {mjd_utc}")
    print(f"MJD (TAI): {mjd_tai}")

    # Time arithmetic
    future_epc = epc + 3600  # Add 3600 seconds (1 hour)
    time_diff = future_epc - epc  # Difference in seconds

    print(f"Time difference: {time_diff} seconds")
