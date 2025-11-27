#!/usr/bin/env python
# /// script
# dependencies = ["brahe"]
# ///

"""
Demonstration of Earth texture options in 3D trajectory plots.

Shows how to use different Earth textures: simple, blue_marble,
natural_earth_50m, and natural_earth_10m.
"""

import brahe as bh
import numpy as np

# Initialize EOP data
bh.initialize_eop()

# Create a simple LEO orbit
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 500e3, 0.01, 97.8, 0.0, 0.0, 0.0])
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Create propagator and generate trajectory
prop = bh.KeplerianPropagator.from_eci(epoch, state, 60.0)
duration = bh.orbital_period(oe[0])
prop.propagate_to(epoch + duration)
traj = prop.trajectory

print("Demonstrating different Earth texture options...")
print()

# 1. Simple texture (default for matplotlib) - solid lightblue sphere
print("1. Creating plot with 'simple' texture (solid sphere)...")
fig1 = bh.plot_trajectory_3d(
    [{"trajectory": traj, "color": "red", "label": "LEO Orbit"}],
    units="km",
    show_earth=True,
    earth_texture="simple",
    backend="matplotlib",
)
print("   ✓ Simple texture plot created")
print()

# 2. Blue Marble texture (packaged with brahe, no download required)
print("2. Creating plot with 'blue_marble' texture (packaged)...")
fig2 = bh.plot_trajectory_3d(
    [{"trajectory": traj, "color": "red", "label": "LEO Orbit"}],
    units="km",
    show_earth=True,
    earth_texture="blue_marble",
    backend="matplotlib",
)
print("   ✓ Blue Marble texture plot created")
print()

# 3. Plotly with default texture (blue_marble)
print("3. Creating plotly plot (defaults to blue_marble)...")
fig3 = bh.plot_trajectory_3d(
    [{"trajectory": traj, "color": "red", "label": "LEO Orbit"}],
    units="km",
    show_earth=True,
    backend="plotly",
)
print("   ✓ Plotly plot with blue marble created")
print()

# 4. Natural Earth 50m texture (will download on first use - ~20MB)
print("4. Natural Earth 50m texture option available:")
print("   - Use earth_texture='natural_earth_50m' to download and use")
print("   - First use downloads ~20MB to ~/.cache/brahe/textures/")
print("   - Subsequent uses load from cache")
print()

# 5. Natural Earth 10m texture (will download on first use - ~180MB)
print("5. Natural Earth 10m texture option available:")
print("   - Use earth_texture='natural_earth_10m' for highest quality")
print("   - First use downloads ~180MB to ~/.cache/brahe/textures/")
print("   - Best for high-resolution figures")
print()

print("Summary of texture options:")
print("  • 'simple'           - Fast solid sphere (matplotlib default)")
print("  • 'blue_marble'      - NASA Blue Marble (packaged, plotly default)")
print("  • 'natural_earth_50m' - Natural Earth 50m (auto-downloads)")
print("  • 'natural_earth_10m' - Natural Earth 10m HR (auto-downloads)")
print()
print("All plots created successfully!")
print()
print("To display plots, add plt.show() or fig.show() to the script.")
