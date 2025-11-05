"""
Access Polar Plot Example - Plotly Backend

This script demonstrates how to create an interactive polar access plot using the plotly backend.
Shows satellite azimuth and elevation during ground station passes.
"""

import os
import pathlib
import sys
import brahe as bh
import numpy as np

# Add plots directory to path for importing brahe_theme
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from brahe_theme import save_themed_html

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

# Define ground station (Cape Canaveral)
lat = np.radians(28.3922)  # Latitude in radians
lon = np.radians(-80.6077)  # Longitude in radians
alt = 0.0  # Altitude in meters
station = bh.PointLocation(lat, lon, alt).with_name("Cape Canaveral")

# Define time range (one day to capture multiple passes)
epoch = prop.epoch
duration = 24.0 * 3600.0  # 24 hours in seconds

# Compute access windows
constraint = bh.ElevationConstraint(min_elevation_deg=10.0)
accesses = bh.location_accesses([station], [prop], epoch, epoch + duration, constraint)

# Create polar access plot
if len(accesses) > 0:
    # Use first 3 access windows
    num_windows = min(3, len(accesses))
    windows_to_plot = [
        {"access_window": accesses[i], "label": f"Access {i + 1}"}
        for i in range(num_windows)
    ]

    fig = bh.plot_access_polar(
        windows_to_plot,
        prop,  # Propagator for interpolation
        min_elevation=10.0,
        backend="plotly",
    )

    # Save themed HTML files
    light_path, dark_path = save_themed_html(fig, OUTDIR / SCRIPT_NAME)
    print(f"✓ Generated {light_path}")
    print(f"✓ Generated {dark_path}")
else:
    print("No access windows found in the specified time range")
