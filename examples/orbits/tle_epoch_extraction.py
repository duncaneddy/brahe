# /// script
# dependencies = ["brahe"]
# ///

"""
Extract the epoch from a Two-Line Element (TLE) set.

This example demonstrates how to extract just the epoch timestamp from a TLE
without parsing the full orbital elements.
"""

import brahe as bh

# ISS TLE (NORAD ID 25544)
line1 = "1 25544U 98067A   25302.48953544  .00013618  00000-0  24977-3 0  9995"
line2 = "2 25544  51.6347   1.5519 0004808 353.3325   6.7599 15.49579513535999"

# Extract epoch from line 1 (epoch is encoded in line 1 only)
epoch = bh.epoch_from_tle(line1)

print(f"TLE Epoch: {epoch}")
print(f"Time System: {epoch.time_system}")
print(f"Julian Date: {epoch.jd():.10f}")
print(f"Modified Julian Date: {epoch.mjd():.10f}")

# Convert to datetime components
dt = epoch.to_datetime()
print("\nDatetime Components:")
print(f"  Year: {dt[0]}")
print(f"  Month: {dt[1]}")
print(f"  Day: {dt[2]}")
print(f"  Hour: {dt[3]}")
print(f"  Minute: {dt[4]}")
print(f"  Second: {dt[5]:.6f}")

# Expected output:
# TLE Epoch: 2025-10-29T11:44:55.766182400 UTC
# Time System: TimeSystem.UTC
# Julian Date: 2460974.9895780
# Modified Julian Date: 60974.4895780
#
# Datetime Components:
#   Year: 2025
#   Month: 10
#   Day: 29
#   Hour: 11
#   Minute: 44
#   Second: 55.766182
