# /// script
# dependencies = ["brahe"]

import brahe as bh
import numpy as np

# Initialize EOP
bh.initialize_eop()

# Create an epoch
epc = bh.Epoch(2024, 1, 1, 0, 0, 0)

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

# Covert ECI cartesian state to ECEF cartesian state
x_ecef = bh.state_eci_to_ecef(epc, x_deg)

# Convert ECEF cartesian state to geodetic coordinates
lat, lon, alt = bh.position_ecef_to_geodetic(
    x_ecef[:3], angle_format=bh.AngleFormat.DEGREES
)
print(
    f"Geodetic coordinates (lat, lon, alt): {lat:.6f}°, {lon:.6f}°, {alt / 1e3:.3f} km"
)
