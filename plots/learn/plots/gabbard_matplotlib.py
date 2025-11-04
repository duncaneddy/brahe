"""
Gabbard Diagram Example - Matplotlib Backend

This script demonstrates how to create a Gabbard diagram using the matplotlib backend.
A Gabbard diagram plots orbital period vs apogee/perigee altitude, useful for analyzing
debris clouds or satellite constellations.
"""

import brahe as bh
import numpy as np

# Initialize EOP data
bh.initialize_eop()

# Create a reference epoch
epoch = bh.Epoch.from_datetime(2025, 11, 3, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Define multiple objects with varying orbital parameters
# Simulating a debris cloud or constellation
propagators = []

# Base orbital parameters (ISS-like orbit)
base_a = bh.R_EARTH + 420e3  # Semi-major axis (m)
base_e = 0.001  # Eccentricity
base_i = np.radians(51.6)  # Inclination (rad)

# Create variations in semi-major axis and eccentricity
np.random.seed(42)
for i in range(50):
    # Vary semi-major axis and eccentricity
    a = base_a + np.random.normal(0, 50e3)  # ±50 km variation
    e = base_e + np.random.uniform(0, 0.02)  # 0-0.02 variation

    # Random other elements
    raan = np.random.uniform(0, 2 * np.pi)
    argp = np.random.uniform(0, 2 * np.pi)
    anom = np.random.uniform(0, 2 * np.pi)

    # Create Keplerian elements
    oe = np.array([a, e, base_i, raan, argp, anom])

    # Convert to Cartesian state
    state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)

    # Create propagator
    prop = bh.KeplerianPropagator.from_eci(epoch, state, step_size=60.0).with_name(
        f"Object-{i + 1}"
    )
    propagators.append(prop)

# Create Gabbard diagram
fig = bh.plot_gabbard_diagram(propagators, epoch, backend="matplotlib")

# Save figure
fig.savefig("docs/figures/plot_gabbard_matplotlib.png", dpi=150, bbox_inches="tight")
print("Gabbard diagram (matplotlib) saved to: docs/figures/plot_gabbard_matplotlib.png")
