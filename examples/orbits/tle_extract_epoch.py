# /// script
# dependencies = ["brahe"]
# ///

"""Extract just the epoch from a TLE."""

import brahe as bh

# ISS TLE (NORAD ID 25544)
line1 = "1 25544U 98067A   25302.48953544  .00013618  00000-0  24977-3 0  9995"

# Extract epoch from line 1 (epoch is encoded in line 1 only)
epoch = bh.epoch_from_tle(line1)

print(f"TLE Epoch: {epoch}")
print(f"Time System: {epoch.time_system}")

# Expected output:
# TLE Epoch: 2025-10-29 11:44:55.862 UTC
# Time System: UTC
