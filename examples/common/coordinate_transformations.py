# /// script
# dependencies = ["brahe", "pytest"]
# ///
"""
This example demonstrates how to calculate working with coordinate transformations
using the Brahe library. It shows how you can convert between Keplerian elements and
the Earth-Centered Earth-Fixed (ECEF) coordinate system and vice versa.
"""

import brahe as bh
import numpy as np

# Initialize Earth Orientation Parameter data
bh.initialize_eop()

# Define orbital elements
a = bh.constants.R_EARTH + 700e3  # Semi-major axis in meters (700 km altitude)
e = 0.001  # Eccentricity
i = 98.7  # Inclination in radians
raan = 15.0  # Right Ascension of Ascending Node in radians
arg_periapsis = 30.0  # Argument of Periapsis in radians
mean_anomaly = 45.0  # Mean Anomaly

# Create a state vector from orbital elements
state_kep = np.array([a, e, i, raan, arg_periapsis, mean_anomaly])

# Convert Keplerian state to ECI coordinates
state_eci = bh.state_koe_to_eci(state_kep, bh.AngleFormat.DEGREES)
print(f"ECI Coordinates: {state_eci}")
epoch = bh.Epoch(2024, 6, 1, 12, 0, 0.0, time_system=bh.TimeSystem.UTC)

# Convert ECI coordinates to ECEF coordinates at the given epoch
state_ecef = bh.state_eci_to_ecef(epoch, state_eci)
print(f"ECEF Coordinates: {state_ecef}")
state_eci_2 = bh.state_ecef_to_eci(epoch, state_ecef)
print(f"Recovered ECI Coordinates: {state_eci_2}")
state_kep_2 = bh.state_eci_to_koe(state_eci_2, bh.AngleFormat.DEGREES)
print(f"Recovered Keplerian Elements: {state_kep_2}")
