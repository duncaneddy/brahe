# /// script
# dependencies = ["brahe"]
# ///
"""
Initialize KeplerianPropagator from Keplerian orbital elements
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Define initial epoch
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Define Keplerian elements [a, e, i, Ω, ω, M]
elements = np.array(
    [
        bh.R_EARTH + 500e3,  # Semi-major axis (m)
        0.001,  # Eccentricity
        97.8,  # Inclination (degrees)
        15.0,  # RAAN (degrees)
        30.0,  # Argument of perigee (degrees)
        45.0,  # Mean anomaly (degrees)
    ]
)

# Create propagator with 60-second step size
prop = bh.KeplerianPropagator.from_keplerian(
    epoch, elements, bh.AngleFormat.DEGREES, 60.0
)

print(f"Orbital period: {bh.orbital_period(elements[0]):.1f} seconds")
# Orbital period: 5677.0 seconds
