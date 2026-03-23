# /// script
# dependencies = ["brahe", "pytest"]
# ///
"""
This example demonstrates how to work with the Epoch class in the Brahe library,
including creating epochs, converting between time systems, and performing
time arithmetic.
"""

import brahe as bh

# Create an epoch from a specific date and time
epc = bh.Epoch(2024, 1, 1, 12, 0, 0.0, time_system=bh.TimeSystem.UTC)

# Print as ISO 8601 string
print(f"Epoch in UTC: {epc.isostring()}")
mjd_tai = epc.mjd_as_time_system(bh.TimeSystem.TAI)
print(f"MJD in TAI: {mjd_tai}")
jd_gps = epc.jd_as_time_system(bh.TimeSystem.GPS)
print(f"JD in GPS: {jd_gps}")
epc2 = bh.Epoch(2024, 1, 2, 13, 30, 0.0, time_system=bh.TimeSystem.GPS)
delta_seconds = epc2 - epc
print(f"Difference between epochs in seconds: {delta_seconds}")
epc_utc = epc2.to_string_as_time_system(bh.TimeSystem.UTC)
print(f"Epoch in GPS: {epc2}")
print(f"Epoch in UTC: {epc_utc}")
