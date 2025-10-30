# /// script
# dependencies = ["brahe"]
# ///

"""
Create a Two-Line Element (TLE) set from orbital elements.

This example demonstrates how to generate a TLE from an epoch and Keplerian
orbital elements. This is useful for sharing orbital data or using with
SGP4 propagators.
"""

import brahe as bh
import numpy as np

# Define orbital epoch
epoch = bh.Epoch.from_datetime(2025, 10, 29, 11, 44, 55.766182, 0, bh.TimeSystem.UTC)

# Define ISS orbital elements
# Order: [a, e, i, raan, argp, M]
# Note: Angles must be in DEGREES for TLE creation (exception to library convention)
elements = np.array(
    [
        6795445.0,  # Semi-major axis (m)
        0.0004808,  # Eccentricity
        51.6347,  # Inclination (deg)
        1.5519,  # Right Ascension of Ascending Node (deg)
        353.3325,  # Argument of Periapsis (deg)
        6.7599,  # Mean Anomaly (deg)
    ]
)

# Create TLE lines with NORAD ID
norad_id = "25544"
line1, line2 = bh.keplerian_elements_to_tle(epoch, elements, norad_id)

print("Generated TLE:")
print(line1)
print(line2)

# Verify by parsing the generated TLE back
parsed_epoch, parsed_elements = bh.keplerian_elements_from_tle(line1, line2)

print("\nVerification:")
print(f"Epoch matches: {abs(epoch.jd() - parsed_epoch.jd()) < 1e-9}")
print(f"Elements match: {np.allclose(elements, parsed_elements, rtol=1e-5)}")

# Expected output:
# Generated TLE:
# 1 25544U 00000    25302.48953544  .00000000  00000-0  00000-0 0    04
# 2 25544  51.6347   1.5519 0004808 353.3325   6.7599 15.49579513    06
#
# Verification:
# Epoch matches: True
# Elements match: True
