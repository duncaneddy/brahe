"""
Gabbard Diagram Example - Plotly Backend

This script demonstrates how to create an interactive Gabbard diagram using the plotly backend.
A Gabbard diagram plots orbital period vs apogee/perigee altitude, useful for analyzing
debris clouds or satellite constellations.
"""

import os
import pathlib
import sys
import brahe as bh

# Add plots directory to path for importing brahe_theme
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from brahe_theme import save_themed_html

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

# Create Gabbard diagram
fig = bh.plot_gabbard_diagram(all_debris, epoch, backend="plotly")

# Add ISS altitude reference line
fig.add_hline(
    y=iss_altitude_km,
    line_dash="dash",
    line_color="orange",
    line_width=2,
    annotation_text=f"ISS Altitude ({iss_altitude_km:.1f} km)",
    annotation_position="right",
)

# Save themed HTML files
light_path, dark_path = save_themed_html(fig, OUTDIR / SCRIPT_NAME)
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")
