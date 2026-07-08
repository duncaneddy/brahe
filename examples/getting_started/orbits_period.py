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

# Calculate Orbital period
period = bh.orbital_period(oe_deg[0])
print(f"Orbital period: {period / 60:.3f} minutes")
