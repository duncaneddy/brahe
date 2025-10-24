"""
Simple demo of Gabbard diagram plotting.

This example shows how to visualize a satellite debris cloud using a Gabbard
diagram, which plots apogee and perigee altitude vs orbital period.
"""

import numpy as np
import brahe as bh
import matplotlib.pyplot as plt

# Set up EOP
eop = bh.initialize_eop()

# Create epoch
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Parent orbit (LEO satellite before breakup)
oe_parent = np.array([bh.R_EARTH + 500e3, 0.01, np.radians(97.8), 0.0, 0.0, 0.0])

# Simulate debris cloud with various delta-v
debris = []
np.random.seed(42)  # For reproducibility

for i in range(50):
    # Create debris with perturbed orbital elements
    oe = oe_parent.copy()

    # Vary semi-major axis (simulating delta-v)
    dv = np.random.normal(0, 150)  # m/s
    oe[0] += dv * 1000  # rough approximation

    # Vary eccentricity
    oe[1] = max(0.001, min(0.3, oe[1] + np.random.normal(0, 0.03)))

    # Small variations in inclination
    oe[2] += np.random.normal(0, np.radians(0.5))

    # Convert to Cartesian and create propagator
    state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
    prop = bh.KeplerianPropagator.from_eci(epoch, state, 60.0)
    debris.append(prop)

# Plot Gabbard diagram with matplotlib
print("Creating Gabbard diagram...")
fig = bh.plot_gabbard_diagram(
    debris, epoch=epoch, altitude_units="km", period_units="min", backend="matplotlib"
)


plt.savefig("gabbard_demo.png", dpi=150, bbox_inches="tight")
print("✓ Saved gabbard_demo.png")

# Show statistics
periods = [
    bh.orbital_period(
        bh.state_cartesian_to_osculating(prop.state_eci(epoch), bh.AngleFormat.RADIANS)[
            0
        ]
    )
    / 60
    for prop in debris
]
print("\nDebris statistics:")
print(f"  Number of objects: {len(debris)}")
print(f"  Period range: {min(periods):.1f} - {max(periods):.1f} min")

# You can also create an interactive plotly version
# fig_plotly = bh.plot_gabbard_diagram(
#     debris, epoch=epoch, altitude_units="km", period_units="min", backend="plotly"
# )
# fig_plotly.write_html("gabbard_demo.html")
