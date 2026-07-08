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
x_eci_1 = bh.state_koe_to_eci(oe_deg, bh.AngleFormat.DEGREES)

# Covert ECI cartesian state to ECEF cartesian state
x_ecef = bh.state_eci_to_ecef(epc, x_eci_1)

# Convert ECEF back to ECI to verify consistency
x_eci_2 = bh.state_ecef_to_eci(epc, x_ecef)
print(
    f"ECI -> ECEF -> ECI rountrip difference: {np.linalg.norm(x_eci_2 - x_eci_1):.3e}"
)

# Perform same transformation with GCRF/ITRF naming

x_gcrf_1 = x_eci_1
x_itrf = bh.state_gcrf_to_itrf(epc, x_gcrf_1)
x_gcrf_2 = bh.state_itrf_to_gcrf(epc, x_itrf)
print(
    f"GCRF -> ITRF -> GCRF rountrip difference: {np.linalg.norm(x_gcrf_2 - x_gcrf_1):.3e}"
)

print(f"ECEF <> ITRF difference: {np.linalg.norm(x_ecef - x_itrf):.3e}")
