# /// script
# dependencies = ["brahe"]
# ///

"""Create a TLE from orbital elements."""

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

# Expected output:
# Generated TLE:
# 1 25544U          25302.48953433  .00000000  00000+0  00000+0 0 00002
# 2 25544  51.6347   1.5519 0004808 353.3325   6.7599 15.49800901000006
