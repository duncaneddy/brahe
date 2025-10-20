# /// script
# dependencies = ["brahe"]
# ///
"""
Calculate the orbital period of a satellite in low Earth orbit.

Demonstrates:
- Using physical constants (R_EARTH)
- Calculating orbital period from semi-major axis
- Basic orbital mechanics calculations
"""

import brahe as bh

if __name__ == "__main__":
    # Define the semi-major axis of a low Earth orbit (in meters)
    a = bh.R_EARTH + 400e3  # 400 km altitude

    # Calculate the orbital period using Kepler's third law
    T = bh.orbital_period(a)

    print(f"Orbital Period: {T / 60:.2f} minutes")
