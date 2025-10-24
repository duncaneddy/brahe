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
# Output:
# Epoch in UTC: 2024-01-01T12:00:00Z

# Get the Modified Julian Date (MJD) in different time systems
mjd_tai = epc.mjd_as_time_system(bh.TimeSystem.TAI)
print(f"MJD in TAI: {mjd_tai}")
# Output:
# MJD in TAI: 60310.50042824074

# Get the time as a Julian Date (JD) in GPS time system
jd_gps = epc.jd_as_time_system(bh.TimeSystem.GPS)
print(f"JD in GPS: {jd_gps}")
# Output:
# JD in GPS: 2460311.000208333

# Take the difference between two epochs in different time systems
epc2 = bh.Epoch(2024, 1, 2, 13, 30, 0.0, time_system=bh.TimeSystem.GPS)
delta_seconds = epc2 - epc
print(f"Difference between epochs in seconds: {delta_seconds}")
# Output:
# Difference between epochs in seconds: 91782.0

# Get the epoch as a string in different time systems
epc_utc = epc2.to_string_as_time_system(bh.TimeSystem.UTC)
print(f"Epoch in GPS: {epc2}")
print(f"Epoch in UTC: {epc_utc}")
# Outputs:
# Epoch in GPS: 2024-01-02 13:30:00.000 GPS
# Epoch in UTC: 2024-01-02 13:29:42.000 UTC
