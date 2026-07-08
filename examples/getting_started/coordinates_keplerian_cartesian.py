# /// script
# dependencies = ["brahe"]

import brahe as bh
import numpy as np

# Initialize EOP
bh.initialize_eop()

# Initialize a Keplerian state
# Define orbital elements [a, e, i, Ω, ω, M] in meters and degrees
# LEO satellite: 500 km altitude, 97.8° inclination (approx sun-synchronous)
oe_deg = np.array(
    [
        bh.R_EARTH + 500e3,  # Semi-major axis (m)
        0.01,  # Eccentricity
        97.8,  # Inclination (deg)
        15.0,  # Right ascension of ascending node (deg)
        30.0,  # Argument of periapsis (deg)
        45.0,  # Mean anomaly (deg)
    ]
)

# Convert orbital elements to Cartesian state using degrees
x_deg = bh.state_koe_to_eci(oe_deg, bh.AngleFormat.DEGREES)

# Convert back to degrees
oe_deg_2 = bh.state_eci_to_koe(x_deg, bh.AngleFormat.DEGREES)

print("Original Keplerian elements:")
for i, elem in enumerate(oe_deg):
    print(f"  [{i}]: {elem:.3f}")
print("Converted Cartesian state:")
for i, elem in enumerate(x_deg):
    print(f"  [{i}]: {elem:.3f}")
print("Back to Keplerian elements:")
for i, elem in enumerate(oe_deg_2):
    print(f"  [{i}]: {elem:.3f}")
