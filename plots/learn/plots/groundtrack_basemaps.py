"""
Ground Track Basemap Styles Example

This script demonstrates different basemap styles available for ground track plots:
- natural_earth: High-quality vector basemap from Natural Earth Data
- stock: Cartopy's built-in coastlines and borders
- None: Plain background without geographic features
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

# ISS TLE for November 3, 2025
tle_line0 = "ISS (ZARYA)"
tle_line1 = "1 25544U 98067A   25306.42331346  .00010070  00000-0  18610-3 0  9999"
tle_line2 = "2 25544  51.6344 342.0717 0004969   8.9436 351.1640 15.49700017536601"

# Create SGP4 propagator
prop = bh.SGPPropagator.from_3le(tle_line0, tle_line1, tle_line2, 60.0)
epoch = prop.epoch

# Propagate for one orbital period
duration = 92.0 * 60.0  # ~92 minutes for ISS
prop.propagate_to(epoch + duration)
traj = prop.trajectory

# Create three versions with different basemaps

# 1. Natural Earth - High-quality vector basemap
fig_ne = bh.plot_groundtrack(
    trajectories=[{"trajectory": traj, "color": "red", "line_width": 2}],
    basemap="natural_earth",
    backend="matplotlib",
)
fig_ne.savefig(
    OUTDIR / f"{SCRIPT_NAME}_natural_earth_light.svg", dpi=300, bbox_inches="tight"
)
print(f"✓ Generated {SCRIPT_NAME}_natural_earth_light.svg")
plt.close(fig_ne)

# 2. Stock - Cartopy built-in features
fig_stock = bh.plot_groundtrack(
    trajectories=[{"trajectory": traj, "color": "red", "line_width": 2}],
    basemap="stock",
    backend="matplotlib",
)
fig_stock.savefig(
    OUTDIR / f"{SCRIPT_NAME}_stock_light.svg", dpi=300, bbox_inches="tight"
)
print(f"✓ Generated {SCRIPT_NAME}_stock_light.svg")
plt.close(fig_stock)

# 3. None - Plain background
fig_plain = bh.plot_groundtrack(
    trajectories=[{"trajectory": traj, "color": "red", "line_width": 2}],
    basemap=None,
    backend="matplotlib",
)
fig_plain.savefig(
    OUTDIR / f"{SCRIPT_NAME}_plain_light.svg", dpi=300, bbox_inches="tight"
)
print(f"✓ Generated {SCRIPT_NAME}_plain_light.svg")
plt.close(fig_plain)

# Generate dark mode versions
with plt.style.context("dark_background"):
    # Natural Earth (dark)
    fig_ne_dark = bh.plot_groundtrack(
        trajectories=[{"trajectory": traj, "color": "red", "line_width": 2}],
        basemap="natural_earth",
        backend="matplotlib",
    )
    fig_ne_dark.patch.set_facecolor("#1c1e24")
    for ax in fig_ne_dark.get_axes():
        ax.set_facecolor("#1c1e24")
    fig_ne_dark.savefig(
        OUTDIR / f"{SCRIPT_NAME}_natural_earth_dark.svg", dpi=300, bbox_inches="tight"
    )
    print(f"✓ Generated {SCRIPT_NAME}_natural_earth_dark.svg")
    plt.close(fig_ne_dark)

    # Stock (dark)
    fig_stock_dark = bh.plot_groundtrack(
        trajectories=[{"trajectory": traj, "color": "red", "line_width": 2}],
        basemap="stock",
        backend="matplotlib",
    )
    fig_stock_dark.patch.set_facecolor("#1c1e24")
    for ax in fig_stock_dark.get_axes():
        ax.set_facecolor("#1c1e24")
    fig_stock_dark.savefig(
        OUTDIR / f"{SCRIPT_NAME}_stock_dark.svg", dpi=300, bbox_inches="tight"
    )
    print(f"✓ Generated {SCRIPT_NAME}_stock_dark.svg")
    plt.close(fig_stock_dark)

    # Plain (dark)
    fig_plain_dark = bh.plot_groundtrack(
        trajectories=[{"trajectory": traj, "color": "red", "line_width": 2}],
        basemap=None,
        backend="matplotlib",
    )
    fig_plain_dark.patch.set_facecolor("#1c1e24")
    for ax in fig_plain_dark.get_axes():
        ax.set_facecolor("#1c1e24")
    fig_plain_dark.savefig(
        OUTDIR / f"{SCRIPT_NAME}_plain_dark.svg", dpi=300, bbox_inches="tight"
    )
    print(f"✓ Generated {SCRIPT_NAME}_plain_dark.svg")
    plt.close(fig_plain_dark)
