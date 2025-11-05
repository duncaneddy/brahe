"""
Access Elevation vs Azimuth Plot Example - Plotly Backend

This script demonstrates how to create an interactive elevation vs azimuth plot using the plotly backend.
Shows the satellite's trajectory across the observed horizon with a sinusoidal elevation mask.
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
lat = 28.4740  # Latitude in degrees
lon = -80.5772  # Longitude in degrees
alt = 0.0  # Altitude in meters
station = bh.PointLocation(lon, lat, alt).with_name("Cape Canaveral")

# Define time range (one day to capture multiple passes)
epoch = prop.epoch
duration = 7.0 * 24.0 * 3600.0  # 24 hours in seconds


# Define sinusoidal elevation mask: 15° + 10° * sin(2*azimuth)
# This varies between 5° and 25° around the horizon
def elevation_mask(az):
    return 15.0 + 10.0 * np.sin(np.radians(2 * az)) + 5.0 * np.sin(np.radians(3 * az))


# Create ElevationMaskConstraint from the sinusoidal mask function
# Sample the mask at 36 points around the horizon (every 10 degrees)
mask_azimuths = np.arange(0, 360, 10)
mask_points = [(az, elevation_mask(az)) for az in mask_azimuths]
constraint = bh.ElevationMaskConstraint(mask_points)

# Compute access windows using the elevation mask constraint
accesses = bh.location_accesses([station], [prop], epoch, epoch + duration, constraint)
print(f"Computed {len(accesses)} access windows")

# Filter for passes longer than 5 minutes (300 seconds) to show complete passes
min_duration = 300.0  # seconds
long_passes = [acc for acc in accesses if acc.duration > min_duration]
print(f"Filtered to {len(long_passes)} long passes (> {min_duration} seconds)")

# Create elevation vs azimuth plot
if len(long_passes) > 0:
    # Use first 3 long passes for better visualization
    passes = long_passes[: min(3, len(long_passes))]
    window_configs = [
        {"access_window": passes[i], "label": f"Pass {i + 1}"}
        for i in range(len(passes))
    ]

    fig = bh.plot_access_elevation_azimuth(
        window_configs,
        prop,  # Propagator for interpolation
        elevation_mask=elevation_mask,
        backend="plotly",
    )

    # Save themed HTML files
    light_path, dark_path = save_themed_html(fig, OUTDIR / SCRIPT_NAME)
    print(f"✓ Generated {light_path}")
    print(f"✓ Generated {dark_path}")
else:
    print("No access windows found in the specified time range")
