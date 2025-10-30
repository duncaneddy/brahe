# /// script
# dependencies = ["brahe"]
# ///

"""Verify a generated TLE by parsing it back."""

import brahe as bh
import numpy as np

# Define orbital epoch
epoch = bh.Epoch.from_datetime(2025, 10, 29, 11, 44, 55.766182, 0, bh.TimeSystem.UTC)

# Define ISS orbital elements (angles in degrees)
elements = np.array(
    [
        6795445.0,  # Semi-major axis (m)
        0.0004808,  # Eccentricity
        51.6347,  # Inclination (deg)
        1.5519,  # RAAN (deg)
        353.3325,  # Argument of Periapsis (deg)
        6.7599,  # Mean Anomaly (deg)
    ]
)

# Create TLE
norad_id = "25544"
line1, line2 = bh.keplerian_elements_to_tle(epoch, elements, norad_id)

# Verify by parsing the generated TLE back
parsed_epoch, parsed_elements = bh.keplerian_elements_from_tle(line1, line2)

print("Verification:")
print(f"Epoch matches: {abs(epoch.jd() - parsed_epoch.jd()) < 1e-9}")
print(f"Elements match: {np.allclose(elements, parsed_elements, rtol=1e-5)}")

# Expected output:
# Verification:
# Epoch matches: True
# Elements match: True
