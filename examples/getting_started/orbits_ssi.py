# /// script
# dependencies = ["brahe"]

import brahe as bh

# Define semi-major axis and eccentricity
a = bh.R_EARTH + 500e3  # Semi-major axis (m)
e = 0.01  # Eccentricity

# Compute sun-synchronous inclination
i_ssi = bh.sun_synchronous_inclination(a, e, angle_format=bh.AngleFormat.DEGREES)
print(f"Sun-synchronous inclination: {i_ssi:.3f} degrees")
