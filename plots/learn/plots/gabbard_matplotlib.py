"""
Gabbard Diagram Example - Matplotlib Backend

This script demonstrates how to create a Gabbard diagram using the matplotlib backend.
A Gabbard diagram plots orbital period vs apogee/perigee altitude, useful for analyzing
debris clouds or satellite constellations.
"""

import brahe as bh
import numpy as np
import matplotlib.pyplot as plt

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

# Create Gabbard diagram in light mode
fig = bh.plot_gabbard_diagram(propagators, epoch, backend="matplotlib")

# Save light mode figure
fig.savefig(
    "docs/figures/plot_gabbard_matplotlib_light.svg", dpi=300, bbox_inches="tight"
)
print(
    "Gabbard diagram (matplotlib, light mode) saved to: docs/figures/plot_gabbard_matplotlib_light.svg"
)
plt.close(fig)

# Create Gabbard diagram in dark mode
with plt.style.context("dark_background"):
    fig = bh.plot_gabbard_diagram(propagators, epoch, backend="matplotlib")

    # Set background color to match Plotly dark theme
    fig.patch.set_facecolor("#1c1e24")
    for ax in fig.get_axes():
        ax.set_facecolor("#1c1e24")

    # Save dark mode figure
    fig.savefig(
        "docs/figures/plot_gabbard_matplotlib_dark.svg", dpi=300, bbox_inches="tight"
    )
    print(
        "Gabbard diagram (matplotlib, dark mode) saved to: docs/figures/plot_gabbard_matplotlib_dark.svg"
    )
    plt.close(fig)
