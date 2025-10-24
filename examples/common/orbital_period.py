# /// script
# dependencies = ["brahe", "pytest"]
# ///
"""
This example demonstrates how to calculate the orbital period of a satellite
given its semi-major axis using the Brahe library.
"""

import brahe as bh

# Define the semi-major axis of a low Earth orbit (in meters)
a = bh.constants.R_EARTH + 400e3  # 400 km altitude

# Calculate the orbital period
T = bh.orbital_period(a)

print(f"Orbital Period: {T / 60:.2f} minutes")
# Outputs:
# Orbital Period: 92.56 minutes
