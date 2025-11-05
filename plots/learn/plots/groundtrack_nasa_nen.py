"""
Ground Track with NASA NEN Ground Stations Example

This script demonstrates plotting ground tracks with the NASA Near Earth Network (NEN)
ground stations, showing communication coverage at 550km altitude with 10° minimum elevation.
The coverage cones are displayed as geodetic polygons showing actual ground footprints.
"""

import os
import pathlib
import sys
import brahe as bh
import numpy as np
import math

# Add plots directory to path for importing brahe_theme
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from brahe_theme import save_themed_html

# Configuration
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))
os.makedirs(OUTDIR, exist_ok=True)

# Initialize EOP data
bh.initialize_eop()

# Load NASA NEN ground stations
nen_stations = bh.datasets.groundstations.load("nasa nen")
print(f"Loaded {len(nen_stations)} NASA NEN stations")

# Create a LEO satellite at 550km altitude
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 550e3, 0.001, np.radians(51.6), 0.0, 0.0, 0.0])
state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
prop = bh.KeplerianPropagator.from_eci(epoch, state, 60.0).with_name("LEO Sat")

# Propagate for 2 orbits
duration = 2 * bh.orbital_period(oe[0])
prop.propagate_to(epoch + duration)


# Compute geodetic coverage cones for each station
# This calculates the actual ground footprint at the specified altitude and elevation
def compute_cone_radius(elevation_deg, altitude_m):
    """Compute angular radius of communication cone."""
    ele_rad = math.radians(elevation_deg)
    rho = math.asin(bh.R_EARTH / (bh.R_EARTH + altitude_m))
    eta = math.asin(math.cos(ele_rad) * math.sin(rho))
    lam = math.pi / 2.0 - eta - ele_rad
    return lam


cone_radius_rad = compute_cone_radius(10.0, 550e3)

# Create coverage cone polygons for each station
coverage_zones = []
for station in nen_stations:
    lat_deg = math.degrees(station.latitude(bh.AngleFormat.RADIANS))
    lon_deg = math.degrees(station.longitude(bh.AngleFormat.RADIANS))

    # Create a circle of points around the station
    num_points = 64
    circle_points = []
    for i in range(num_points):
        bearing = 2 * math.pi * i / num_points
        # Simple approximation for small circles
        dlat = cone_radius_rad * math.cos(bearing)
        dlon = cone_radius_rad * math.sin(bearing) / math.cos(math.radians(lat_deg))
        circle_points.append(
            (
                math.radians(lat_deg + math.degrees(dlat)),
                math.radians(lon_deg + math.degrees(dlon)),
                0.0,  # altitude
            )
        )

    # Create polygon location from vertices
    zone = bh.PolygonLocation(circle_points)
    coverage_zones.append(zone)

# Create ground track plot with NASA NEN stations and coverage zones
fig = bh.plot_groundtrack(
    trajectories=[{"trajectory": prop.trajectory, "color": "red", "line_width": 2}],
    ground_stations=[{"stations": nen_stations, "color": "blue", "alpha": 0.8}],
    zones=[
        {
            "zone": zone,
            "fill": True,
            "fill_color": "blue",
            "fill_alpha": 0.15,
            "edge": False,
        }
        for zone in coverage_zones
    ],
    basemap="natural_earth",
    backend="plotly",
)

# Save themed HTML files
light_path, dark_path = save_themed_html(fig, OUTDIR / SCRIPT_NAME)
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")
