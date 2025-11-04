"""
Gabbard Diagram Example - Matplotlib Backend

This script demonstrates how to create a Gabbard diagram using the matplotlib backend.
A Gabbard diagram plots orbital period vs apogee/perigee altitude, useful for analyzing
debris clouds or satellite constellations.
"""

import os
import pathlib
import brahe as bh
import matplotlib.pyplot as plt

# Configuration
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))
os.makedirs(OUTDIR, exist_ok=True)

# Initialize EOP data
bh.initialize_eop()

# Get Ephmeris and debris for major events:
cosmos_1408_debris = bh.celestrak.get_tles_as_propagators("cosmos-1408-debris", 60.0)
fengyun_debris = bh.celestrak.get_tles_as_propagators("fengyun-1c-debris", 60.0)
iridium_debris = bh.celestrak.get_tles_as_propagators("iridium-33-debris", 60.0)
cosmos_2251_debris = bh.celestrak.get_tles_as_propagators("cosmos-2251-debris", 60.0)
all_debris = cosmos_1408_debris + fengyun_debris + iridium_debris + cosmos_2251_debris

print(f"Cosmos 1408 debris objects: {len(cosmos_1408_debris)}")
print(f"Fengyun-1C debris objects: {len(fengyun_debris)}")
print(f"Iridium 33 debris objects: {len(iridium_debris)}")
print(f"Cosmos 2251 debris objects: {len(cosmos_2251_debris)}")
print(f"Total debris objects loaded: {len(all_debris)}")

# Get epoch of first debris object
epoch = all_debris[0].epoch

# Get ISS ephemeris for reference altitude line
iss = bh.celestrak.get_tle_by_id_as_propagator(25544, 60.0, "active")
iss_state = iss.state_eci(epoch)
iss_oe = bh.state_cartesian_to_osculating(iss_state, bh.AngleFormat.RADIANS)
iss_altitude_km = (iss_oe[0] - bh.R_EARTH) / 1e3  # Convert to km

print(f"ISS altitude at epoch: {iss_altitude_km:.1f} km")

# Create Gabbard diagram in light mode
fig = bh.plot_gabbard_diagram(all_debris, epoch, backend="matplotlib")

# Add ISS altitude reference line
ax = fig.get_axes()[0]
ax.axhline(
    y=iss_altitude_km,
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"ISS Altitude ({iss_altitude_km:.1f} km)",
)
ax.legend()

# Save light mode figure
light_path = OUTDIR / f"{SCRIPT_NAME}_light.svg"
fig.savefig(light_path, dpi=300, bbox_inches="tight")
print(f"✓ Generated {light_path}")
plt.close(fig)

# Create Gabbard diagram in dark mode
with plt.style.context("dark_background"):
    fig = bh.plot_gabbard_diagram(all_debris, epoch, backend="matplotlib")

    # Set background color to match Plotly dark theme
    fig.patch.set_facecolor("#1c1e24")
    for ax in fig.get_axes():
        ax.set_facecolor("#1c1e24")

    # Add ISS altitude reference line
    ax = fig.get_axes()[0]
    ax.axhline(
        y=iss_altitude_km,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"ISS Altitude ({iss_altitude_km:.1f} km)",
    )
    ax.legend()

    # Save dark mode figure
    dark_path = OUTDIR / f"{SCRIPT_NAME}_dark.svg"
    fig.savefig(dark_path, dpi=300, bbox_inches="tight")
    print(f"✓ Generated {dark_path}")
    plt.close(fig)
