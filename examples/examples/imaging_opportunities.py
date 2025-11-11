#!/usr/bin/env python
# /// script
# dependencies = ["brahe", "plotly", "numpy"]
# TIMEOUT = 600
# ///

"""
Finding Imaging Opportunities for ICEYE Constellation over San Francisco

This example demonstrates how to:
1. Download TLE data for ICEYE constellation from CelesTrak
2. Visualize the constellation in 3D space
3. Define composite imaging constraints (descending, right-looking, 35-45° off-nadir)
4. Compute imaging opportunities over a 7-day period

The example shows the complete workflow from constellation download to opportunity computation.
"""

# --8<-- [start:all]
# --8<-- [start:preamble]
import time
import csv
import os
import pathlib
import sys
import brahe as bh
import numpy as np

bh.initialize_eop()
# --8<-- [end:preamble]

# Configuration for output files
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))
os.makedirs(OUTDIR, exist_ok=True)

# Download TLE data for ICEYE constellation from CelesTrak
# ICEYE operates a constellation of SAR (Synthetic Aperture Radar) satellites
print("Downloading ICEYE constellation TLEs from CelesTrak...")
start_time = time.time()

# --8<-- [start:ephemeris_download]
# Get all active satellites and filter for ICEYE
all_sats = bh.datasets.celestrak.get_tles_as_propagators("active", 60.0)
iceye_sats = [sat for sat in all_sats if "ICEYE" in sat.get_name().upper()]
# --8<-- [end:ephemeris_download]

elapsed = time.time() - start_time
print(f"Loaded {len(iceye_sats)} ICEYE satellites in {elapsed:.2f} seconds.")

if len(iceye_sats) == 0:
    print("ERROR: No ICEYE satellites found in active constellation data.")
    print(
        "This may indicate the satellites are not currently in the CelesTrak database."
    )
    sys.exit(1)

# Print satellite information
print("\nICEYE Constellation:")
for i, sat in enumerate(iceye_sats[:5]):  # Show first 5
    print(f"  {i + 1}. {sat.get_name()}")
    print(f"     Epoch: {sat.epoch}")
    print(f"     Semi-major axis: {sat.semi_major_axis / 1000:.1f} km")
if len(iceye_sats) > 5:
    print(f"  ... and {len(iceye_sats) - 5} more")

# Propagate all satellites for one orbital period for visualization
print("\nPropagating constellation for visualization...")
start_time = time.time()

# --8<-- [start:constellation_propagation]
for sat in iceye_sats:
    orbital_period = bh.orbital_period(sat.semi_major_axis)
    sat.propagate_to(sat.epoch + orbital_period)

# Create 3D constellation visualization
fig_3d = bh.plot_trajectory_3d(
    [
        {
            "trajectory": sat.trajectory,
            "mode": "lines",
            "line_width": 1.5,
            "label": sat.get_name(),
        }
        for sat in iceye_sats
    ],
    units="km",
    show_earth=True,
    earth_texture="natural_earth_50m",
    backend="plotly",
    view_azimuth=45.0,
    view_elevation=30.0,
    view_distance=2.0,
)
# --8<-- [end:constellation_propagation]
elapsed = time.time() - start_time
print(f"Created 3D visualization in {elapsed:.2f} seconds.")

# Reset propagators for access computation
print("\nResetting propagators for access computation...")
for sat in iceye_sats:
    sat.reset()

# --8<-- [start:target_definition]
# Define San Francisco target location
san_francisco = bh.PointLocation(
    lon=-122.4194,  # longitude in degrees
    lat=37.7749,  # latitude in degrees
    alt=0.0,  # altitude in meters
).with_name("San Francisco")
# --8<-- [end:target_definition]

print(f"\nTarget Location: {san_francisco.get_name()}")
print(f"  Longitude: {san_francisco.lon:.4f}°")
print(f"  Latitude: {san_francisco.lat:.4f}°")

# Define composite imaging constraint
# Requirements:
# - Descending pass only
# - Right-looking geometry
# - Off-nadir angle between 35-45 degrees
print("\nDefining imaging constraints:")
print("  - Descending pass only")
print("  - Right-looking geometry")
print("  - Off-nadir angle: 35-45 degrees")

# --8<-- [start:constraint_definition]
constraint = bh.ConstraintAll(
    constraints=[
        bh.AscDscConstraint(allowed=bh.AscDsc.DESCENDING),
        bh.LookDirectionConstraint(allowed=bh.LookDirection.RIGHT),
        bh.OffNadirConstraint(min_off_nadir_deg=35.0, max_off_nadir_deg=45.0),
    ]
)
# --8<-- [end:constraint_definition]

# Compute imaging opportunities over 7-day period
print("\nComputing 7-day imaging opportunities...")
start_time = time.time()

# --8<-- [start:opportunity_computation]
epoch_start = iceye_sats[0].epoch
epoch_end = epoch_start + 7 * 86400.0  # 7 days in seconds

# Propagate all satellites for full 7-day period
for sat in iceye_sats:
    sat.propagate_to(epoch_end)

# Compute access windows
windows = bh.location_accesses(
    [san_francisco], iceye_sats, epoch_start, epoch_end, constraint
)
# --8<-- [end:opportunity_computation]
elapsed = time.time() - start_time
print(f"Computed {len(windows)} imaging opportunities in {elapsed:.2f} seconds.")

# Print sample of imaging opportunities
print("\n" + "=" * 90)
print("Sample Imaging Opportunities (first 10)")
print("=" * 90)
print(
    f"{'Spacecraft':<20} {'Start Time':<25} {'End Time':<25} {'Duration':>10} {'Off-Nadir':>10}"
)
print("-" * 90)
for i, window in enumerate(windows[:10]):
    duration_sec = window.duration
    off_nadir = (
        window.properties.off_nadir_max - window.properties.off_nadir_min
    ) / 2 + window.properties.off_nadir_min
    start_str = str(window.start).split(".")[0]  # Remove fractional seconds
    end_str = str(window.end).split(".")[0]
    print(
        f"{window.satellite_name:<20} {start_str:<25} {end_str:<25} {duration_sec:>8.1f} s {off_nadir:>8.1f}°"
    )
print("=" * 90)

# Export ~10 imaging opportunities to CSV for documentation
csv_path = OUTDIR / f"{SCRIPT_NAME}_windows.csv"
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(
        [
            "Spacecraft",
            "Start Time (UTC)",
            "End Time (UTC)",
            "Duration (sec)",
            "Off-Nadir Angle (deg)",
        ]
    )
    for window in windows[:10]:  # Only export first 10 for documentation
        duration_sec = window.duration
        off_nadir = (
            window.properties.off_nadir_max - window.properties.off_nadir_min
        ) / 2 + window.properties.off_nadir_min
        start_str = str(window.start).split(".")[0]  # Remove fractional seconds
        end_str = str(window.end).split(".")[0]
        writer.writerow(
            [
                window.satellite_name,
                start_str,
                end_str,
                f"{duration_sec:.1f}",
                f"{off_nadir:.1f}",
            ]
        )
print(f"✓ Exported first 10 imaging opportunities to {csv_path}")

# Print statistics
unique_spacecraft = len(set(w.satellite_name for w in windows))
print("\nImaging Opportunity Statistics:")
print(f"  Total opportunities: {len(windows)}")
print(f"  Spacecraft with opportunities: {unique_spacecraft}")
print(f"  Average duration: {np.mean([w.duration for w in windows]):.1f} seconds")
print(f"  Total imaging time: {sum([w.duration for w in windows]):.1f} seconds")
# --8<-- [end:all]

# ============================================================================
# Plot Output Section (for documentation generation)
# ============================================================================

# Add plots directory to path for importing brahe_theme
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "plots"))
from brahe_theme import save_themed_html  # noqa: E402

# Save the 3D constellation figure as themed HTML
light_path, dark_path = save_themed_html(
    fig_3d, OUTDIR / f"{SCRIPT_NAME}_constellation"
)
print(f"\n✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")
