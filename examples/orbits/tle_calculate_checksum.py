# /// script
# dependencies = ["brahe"]
# ///

"""Calculate checksums for TLE lines."""

import brahe as bh

# ISS TLE (NORAD ID 25544)
line1 = "1 25544U 98067A   25302.48953544  .00013618  00000-0  24977-3 0  9995"
line2 = "2 25544  51.6347   1.5519 0004808 353.3325   6.7599 15.49579513535999"

# Calculate checksums for each line
checksum1 = bh.calculate_tle_line_checksum(line1)
checksum2 = bh.calculate_tle_line_checksum(line2)
print(f"Line 1 checksum: {checksum1}")
print(f"Line 2 checksum: {checksum2}")

# Example with corrupted TLE (wrong checksum)
corrupted_line1 = (
    "1 25544U 98067A   25302.48953544  .00013618  00000-0  24977-3 0  9990"
)
is_corrupted_valid = bh.validate_tle_line(corrupted_line1)
print(f"\nCorrupted line valid: {is_corrupted_valid}")

# Expected output:
# Line 1 checksum: 5
# Line 2 checksum: 9
#
# Corrupted line valid: False
